//! Extraction of images embedded in tool outputs.
//!
//! Responses tool results may carry `input_image` parts. Chat Completions has
//! no place for media inside a `role: tool` message, so an untranslated image
//! would be serialized as a base64 string: the model cannot see it and the
//! prompt balloon well past the context budget. Media is stripped from the tool
//! content and re-attached to a user message emitted right after the tool turn
//! ends.

use super::request::value_to_text;
use super::*;

/// Placeholder replacing an image that was moved out of a tool result.
pub(super) const TOOL_MEDIA_MOVED: &str =
    "Tool output media was moved to the following user message by openai-compat.";

/// Data URLs shorter than this stay inline; they are almost certainly icons or
/// sample values rather than an image the model needs to look at.
const MIN_INLINE_DATA_URL_BYTES: usize = 8 * 1024;

/// Guards against pathological nesting in hostile tool output.
const MAX_MEDIA_DEPTH: usize = 32;

/// Strips images from a tool output, queueing them for the next user message.
///
/// Returns the rewritten tool content when media was found, or `None` so the
/// caller can fall back to its byte-stable serialization.
pub(super) fn extract_tool_output_media(
    output: &Value,
    call_id: &str,
    media: &mut Vec<Value>,
    options: TranslationOptions,
) -> Option<String> {
    let mut owned = output.clone();
    let mut parts = Vec::new();
    if strip_media(&mut owned, &mut parts, 0, options) == 0 {
        return None;
    }
    media.push(json!({
        "type": "text",
        "text": format!("[media output of tool call {call_id}]")
    }));
    media.extend(parts);
    Some(value_to_text(&owned))
}

/// Emits queued tool media as a single user message.
///
/// Must only be called at a tool turn boundary: an interleaved user message
/// would break `repair_tool_pairing`'s contiguous `role: tool` scan.
pub(super) fn flush_tool_media(messages: &mut Vec<Value>, media: &mut Vec<Value>) {
    if media.is_empty() {
        return;
    }
    messages.push(json!({
        "role": "user",
        "content": Value::Array(std::mem::take(media))
    }));
}

fn strip_media(
    value: &mut Value,
    parts: &mut Vec<Value>,
    depth: usize,
    options: TranslationOptions,
) -> usize {
    if depth > MAX_MEDIA_DEPTH {
        return 0;
    }
    let current = std::mem::replace(value, Value::Null);
    let (rewritten, extracted) = match current {
        Value::String(text) => strip_media_string(&text, parts, depth, options),
        Value::Array(mut items) => {
            let mut extracted = 0;
            for item in items.iter_mut() {
                extracted += strip_media(item, parts, depth + 1, options);
            }
            (Value::Array(items), extracted)
        }
        Value::Object(mut object) => {
            if let Some(part) = chat_media_part(&object, options) {
                parts.push(part);
                (json!({ "type": "text", "text": TOOL_MEDIA_MOVED }), 1)
            } else {
                // Only `content` is traversed: an unrelated business payload
                // may legitimately contain an `image_url` field.
                let extracted = match object.get_mut("content") {
                    Some(content) => strip_media(content, parts, depth + 1, options),
                    None => 0,
                };
                (Value::Object(object), extracted)
            }
        }
        other => (other, 0),
    };
    *value = rewritten;
    extracted
}

fn strip_media_string(
    text: &str,
    parts: &mut Vec<Value>,
    depth: usize,
    options: TranslationOptions,
) -> (Value, usize) {
    if let Some(part) = data_url_image(text) {
        parts.push(part);
        return (Value::String(TOOL_MEDIA_MOVED.to_string()), 1);
    }
    let trimmed = text.trim();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        if let Ok(mut parsed) = serde_json::from_str::<Value>(trimmed) {
            let extracted = strip_media(&mut parsed, parts, depth + 1, options);
            if extracted > 0 {
                return (Value::String(parsed.to_string()), extracted);
            }
        }
    }
    (Value::String(text.to_string()), 0)
}

fn data_url_image(text: &str) -> Option<Value> {
    let url = text.trim();
    if url.len() < MIN_INLINE_DATA_URL_BYTES {
        return None;
    }
    if !url.starts_with("data:image/") || !url.contains(";base64,") {
        return None;
    }
    Some(json!({ "type": "image_url", "image_url": { "url": url } }))
}

fn chat_media_part(
    object: &serde_json::Map<String, Value>,
    options: TranslationOptions,
) -> Option<Value> {
    match object.get("type").and_then(Value::as_str) {
        Some("input_image" | "image_url") => object.get("image_url").and_then(image_url_part),
        Some("input_file" | "input_audio") => {
            super::request::chat_content_part(&Value::Object(object.clone()), options)
        }
        _ => None,
    }
}

fn image_url_part(value: &Value) -> Option<Value> {
    let image = match value {
        Value::String(url) => json!({ "url": url }),
        Value::Object(object) => {
            let url = object.get("url").and_then(Value::as_str)?;
            let mut image = json!({ "url": url });
            if let Some(detail) = object
                .get("detail")
                .and_then(Value::as_str)
                .filter(|detail| matches!(*detail, "auto" | "low" | "high"))
            {
                image["detail"] = json!(detail);
            }
            image
        }
        _ => return None,
    };
    if image["url"].as_str().is_none_or(str::is_empty) {
        return None;
    }
    Some(json!({ "type": "image_url", "image_url": image }))
}
