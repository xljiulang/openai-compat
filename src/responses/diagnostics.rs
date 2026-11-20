//! Observability for Responses features the translation layer drops.
//!
//! `responses_to_chat` is a pure function with no logger, so a request can lose
//! items, content parts or whole tool declarations without anything appearing
//! in the log. A caller can inspect [`PreparedResponses::ignored`] or call
//! [`ignored_feature_details`] once per request to record that degradation.

use super::request::{is_compaction_turn, is_handled_input_type, text_format_kind};
use super::*;

/// Content part types `message_content` carries into the Chat message.
const HANDLED_CONTENT_PARTS: [&str; 6] = [
    "input_text",
    "output_text",
    "input_image",
    "input_file",
    "input_audio",
    "refusal",
];

/// Lists, in a stable order, everything the translation layer accepted but
/// could not carry into the Chat request.
pub fn ignored_feature_details(request: &Value) -> Vec<String> {
    ignored_feature_details_with_options(request, TranslationOptions::default())
}

/// Lists dropped features while honouring the caller's media capabilities.
pub fn ignored_feature_details_with_options(
    request: &Value,
    options: TranslationOptions,
) -> Vec<String> {
    let declared = collect_declared_tools(request);
    ignored_feature_details_from_declared(request, options, &declared)
}

pub(super) fn ignored_feature_details_from_declared(
    request: &Value,
    options: TranslationOptions,
    declared: &DeclaredTools,
) -> Vec<String> {
    let mut details = Vec::new();
    collect_input_items(request, &mut details, options);
    collect_content_parts(request, &mut details, options);
    details.extend(declared.notes.iter().cloned());
    details.extend(text_format_kind(request).details());
    if is_compaction_turn(request) && request.get("tools").is_some() {
        details.push("tools.suppressed_by_compaction".to_string());
    }
    details.sort();
    details.dedup();
    details
}

fn collect_input_items(request: &Value, details: &mut Vec<String>, options: TranslationOptions) {
    let Some(items) = request.get("input").and_then(Value::as_array) else {
        return;
    };
    for item in items {
        let Some(object) = item.as_object() else {
            details.push("input.item:non_object".to_string());
            continue;
        };
        match object.get("type").and_then(Value::as_str) {
            Some(name) => {
                if !is_handled_input_type(name) {
                    details.push(format!("input.item:{name}"));
                } else if name == "input_file" && !options.supports_file_input() {
                    details.push("input.item_degraded:input_file_unsupported".to_string());
                } else if name == "input_audio" && !options.supports_audio_input() {
                    details.push("input.item_degraded:input_audio_unsupported".to_string());
                } else if super::request::DEGRADED_INPUT_TYPES.contains(&name) {
                    details.push(format!("input.item_degraded:{name}"));
                }
            }
            // An item without a type is treated as a message only when it
            // carries a role, matching the fallback in `translate_input_items`.
            None if !object.contains_key("role") => details.push("input.item:none".to_string()),
            None => {}
        }
    }
}

fn collect_content_parts(request: &Value, details: &mut Vec<String>, options: TranslationOptions) {
    let Some(items) = request.get("input").and_then(Value::as_array) else {
        return;
    };
    for item in items {
        let Some(parts) = item.get("content").and_then(Value::as_array) else {
            continue;
        };
        for part in parts {
            if part.get("text").and_then(Value::as_str).is_some() {
                continue;
            }
            match part.get("type").and_then(Value::as_str) {
                Some("input_file") if !options.supports_file_input() => {
                    details.push("content.part:input_file_unsupported".to_string())
                }
                Some("input_audio") if !options.supports_audio_input() => {
                    details.push("content.part:input_audio_unsupported".to_string())
                }
                Some(name) if HANDLED_CONTENT_PARTS.contains(&name) => {}
                Some(name) => details.push(format!("content.part:{name}")),
                None => details.push("content.part:none".to_string()),
            }
        }
    }
}
