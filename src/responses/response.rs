use super::*;

/// Optional host-specific cleanup applied to assistant text.
///
/// It is called for non-streaming content and for every streaming text delta.
pub type ResponseCleaner = fn(&str) -> String;

/// Translates a non-streaming Chat completion into a Responses object.
pub fn chat_to_responses(chat: &Value, model: &str, shape: &ToolShape) -> Result<Value, String> {
    chat_to_responses_with_cleaner(chat, model, shape, None)
}

/// Translates a Chat completion with an optional product-specific cleaner.
pub fn chat_to_responses_with_cleaner(
    chat: &Value,
    model: &str,
    shape: &ToolShape,
    cleaner: Option<ResponseCleaner>,
) -> Result<Value, String> {
    let chat_id = chat
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("chatcmpl-local");
    let response_id = format!("resp_{}", chat_id.trim_start_matches("chatcmpl-"));
    let mut output = Vec::new();
    let mut finish_reason = None;
    let upstream_model = chat
        .get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.is_empty())
        .unwrap_or(model);
    if let Some(choice) = chat
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
    {
        let message = choice.get("message").cloned().unwrap_or_else(|| json!({}));
        finish_reason = choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .map(str::to_string);
        let mut content_parts = Vec::new();
        let mut inline_reasoning = None;
        match message.get("content") {
            Some(Value::String(content)) => {
                let (reasoning, answer) = split_leading_think_block(content);
                inline_reasoning = reasoning;
                push_output_text(&mut content_parts, &clean_text(cleaner, &answer));
            }
            Some(Value::Array(parts)) => {
                for part in parts {
                    match part.get("type").and_then(Value::as_str) {
                        Some("text" | "output_text") => {
                            if let Some(text) = part.get("text").and_then(Value::as_str) {
                                let (reasoning, answer) = split_leading_think_block(text);
                                if inline_reasoning.is_none() {
                                    inline_reasoning = reasoning;
                                }
                                push_output_text(&mut content_parts, &clean_text(cleaner, &answer));
                            }
                        }
                        Some("refusal") => {
                            if let Some(refusal) = part.get("refusal").and_then(Value::as_str) {
                                if !refusal.is_empty() {
                                    content_parts.push(json!({
                                        "type": "refusal",
                                        "refusal": refusal
                                    }));
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        if let Some(refusal) = message
            .get("refusal")
            .and_then(Value::as_str)
            .filter(|refusal| !refusal.is_empty())
        {
            content_parts.push(json!({
                "type": "refusal",
                "refusal": refusal
            }));
        }

        if let Some(reasoning) = reasoning_text(&message).or(inline_reasoning) {
            output.push(json!({
                "id": gen_id("rs"),
                "type": "reasoning",
                "summary": [{ "type": "summary_text", "text": reasoning }]
            }));
        }
        if !content_parts.is_empty() {
            output.push(json!({
                "id": gen_id("msg"),
                "type": "message",
                "role": message
                    .get("role")
                    .and_then(Value::as_str)
                    .unwrap_or("assistant"),
                "status": "completed",
                "content": content_parts
            }));
        }
        if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
            for call in calls {
                output.push(translate_tool_call(call, shape)?);
            }
        } else if let Some(function_call) = message.get("function_call") {
            if let Some(item) = translate_legacy_function_call(function_call) {
                output.push(item);
            }
        }
    }
    let incomplete_reason = match finish_reason.as_deref() {
        Some("length") => Some("max_output_tokens"),
        Some("content_filter" | "sensitive") => Some("content_filter"),
        None if !output.is_empty() => Some("max_output_tokens"),
        None => {
            return Err("upstream Chat completion has neither output nor finish_reason".to_string())
        }
        _ => None,
    };
    let status = if incomplete_reason.is_some() {
        "incomplete"
    } else {
        "completed"
    };
    let mut response = json!({
        "id": response_id,
        "object": "response",
        "created_at": chat.get("created").and_then(Value::as_u64).unwrap_or_else(unix_now),
        "status": status,
        "model": upstream_model,
        "output": output,
        "error": null,
        "usage": usage_to_responses(chat.get("usage"))
    });
    if let Some(reason) = incomplete_reason {
        response["incomplete_details"] = json!({ "reason": reason });
    }
    if let Some(fingerprint) = chat
        .get("system_fingerprint")
        .filter(|value| !value.is_null())
    {
        response["system_fingerprint"] = fingerprint.clone();
    }
    Ok(response)
}

fn clean_text(cleaner: Option<ResponseCleaner>, text: &str) -> String {
    cleaner
        .map(|cleaner| cleaner(text))
        .unwrap_or_else(|| text.to_string())
}

fn push_output_text(parts: &mut Vec<Value>, text: &str) {
    if !text.is_empty() {
        parts.push(json!({
            "type": "output_text",
            "text": text,
            "annotations": []
        }));
    }
}

fn split_leading_think_block(text: &str) -> (Option<String>, String) {
    super::inline_think::split_leading_think_block(text)
        .map(|(reasoning, answer)| ((!reasoning.trim().is_empty()).then_some(reasoning), answer))
        .unwrap_or_else(|| (None, text.to_string()))
}

fn translate_legacy_function_call(function_call: &Value) -> Option<Value> {
    let name = function_call
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())?;
    let raw_arguments = function_call
        .get("arguments")
        .and_then(Value::as_str)
        .unwrap_or("");
    let arguments = if raw_arguments.trim().is_empty() {
        "{}".to_string()
    } else {
        raw_arguments.to_string()
    };
    if !serde_json::from_str::<Value>(&arguments).is_ok_and(|value| value.is_object()) {
        return None;
    }
    Some(json!({
        "id": gen_id("fc"),
        "type": "function_call",
        "status": "completed",
        "call_id": function_call
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| gen_id("call")),
        "name": name,
        "arguments": arguments
    }))
}

pub(super) fn translate_tool_call(call: &Value, shape: &ToolShape) -> Result<Value, String> {
    let call_id = call
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let name = call
        .pointer("/function/name")
        .and_then(Value::as_str)
        .unwrap_or("");
    let arguments = call
        .pointer("/function/arguments")
        .and_then(Value::as_str)
        .unwrap_or("");
    let (name, namespace) = shape.resolve(name);
    if shape.is_tool_search(&name) {
        let arguments = serde_json::from_str::<Value>(arguments.trim())
            .ok()
            .filter(Value::is_object)
            .unwrap_or_else(|| json!({ "query": arguments }));
        return Ok(json!({
            "id": gen_id("tsc"),
            "type": "tool_search_call",
            "status": "completed",
            "call_id": call_id,
            "execution": "client",
            "arguments": arguments
        }));
    }
    if shape.is_custom(&name) || shape.custom.contains(&name) {
        return Ok(json!({
            "id": gen_id("ctc"),
            "type": "custom_tool_call",
            "status": "completed",
            "call_id": call_id,
            "name": name,
            "input": custom_tool_input(arguments)
        }));
    }
    let arguments = if arguments.trim().is_empty() {
        "{}".to_string()
    } else {
        arguments.to_string()
    };
    if !serde_json::from_str::<Value>(&arguments).is_ok_and(|value| value.is_object()) {
        return Err(format!(
            "tool call arguments for {name} are not a JSON object"
        ));
    }
    let mut item = json!({
        "id": gen_id("fc"),
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "name": name,
        "arguments": arguments
    });
    if let Some(namespace) = namespace {
        item["namespace"] = json!(namespace);
    }
    if let Some(extra) = call
        .get("extra_content")
        .filter(|value| value.as_object().is_some_and(|extra| !extra.is_empty()))
    {
        item["extra_content"] = extra.clone();
    }
    Ok(item)
}

pub(super) fn custom_tool_input(arguments: &str) -> String {
    let arguments = arguments.trim();
    if arguments.is_empty() {
        return String::new();
    }
    serde_json::from_str::<Value>(arguments)
        .ok()
        .and_then(|value| {
            value
                .get("input")
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| {
                    value
                        .as_object()
                        .filter(|object| object.is_empty())
                        .map(|_| String::new())
                })
        })
        .unwrap_or_else(|| arguments.to_string())
}

/// Maps Chat token usage fields to Responses usage fields.
///
/// Responses clients read every counter unconditionally, so missing upstream
/// usage is reported as zeros instead of an empty object. Chat-native field
/// names win over Responses-style ones, and non-numeric values are treated as
/// absent rather than forwarded.
pub fn usage_to_responses(usage: Option<&Value>) -> Value {
    let Some(usage) = usage.and_then(Value::as_object) else {
        return zero_usage();
    };
    let input_tokens = first_number(usage, &["prompt_tokens", "input_tokens"]).unwrap_or(0);
    let output_tokens = first_number(usage, &["completion_tokens", "output_tokens"]).unwrap_or(0);
    let total_tokens =
        first_number(usage, &["total_tokens"]).unwrap_or(input_tokens + output_tokens);
    let direct_cache_read =
        number(usage.get("cache_read_input_tokens")).filter(|tokens| *tokens > 0);
    let cached_tokens = direct_cache_read
        .or_else(|| {
            nested_number(usage, "prompt_tokens_details", "cached_tokens")
                .filter(|tokens| *tokens > 0)
        })
        .or_else(|| {
            nested_number(usage, "input_tokens_details", "cached_tokens")
                .filter(|tokens| *tokens > 0)
        })
        .or_else(|| number(usage.get("prompt_cache_hit_tokens")).filter(|tokens| *tokens > 0))
        .unwrap_or(0);
    let cache_write_tokens = nested_number(usage, "prompt_tokens_details", "cache_write_tokens")
        .or_else(|| nested_number(usage, "input_tokens_details", "cache_write_tokens"))
        .or_else(|| number(usage.get("cache_creation_input_tokens")))
        .unwrap_or(0);
    let reasoning_tokens = nested_number(usage, "completion_tokens_details", "reasoning_tokens")
        .or_else(|| nested_number(usage, "output_tokens_details", "reasoning_tokens"))
        .unwrap_or(0);
    let mut input_details = json!({ "cached_tokens": cached_tokens });
    if cache_write_tokens > 0 {
        input_details["cache_write_tokens"] = json!(cache_write_tokens);
    }
    let mut result = json!({
        "input_tokens": input_tokens,
        "input_tokens_details": input_details,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "output_tokens_details": { "reasoning_tokens": reasoning_tokens }
    });
    if let Some(cache_read) = direct_cache_read {
        result["cache_read_input_tokens"] = json!(cache_read);
    }
    if cache_write_tokens > 0 {
        result["cache_creation_input_tokens"] = json!(cache_write_tokens);
    }
    result
}

fn zero_usage() -> Value {
    json!({
        "input_tokens": 0,
        "input_tokens_details": { "cached_tokens": 0 },
        "output_tokens": 0,
        "total_tokens": 0,
        "output_tokens_details": { "reasoning_tokens": 0 }
    })
}

fn first_number(usage: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .find_map(|key| usage.get(*key).and_then(Value::as_u64))
}

fn number(value: Option<&Value>) -> Option<u64> {
    value.and_then(Value::as_u64)
}

fn nested_number(usage: &serde_json::Map<String, Value>, outer: &str, inner: &str) -> Option<u64> {
    usage
        .get(outer)
        .and_then(|details| details.get(inner))
        .and_then(Value::as_u64)
}
