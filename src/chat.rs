//! Chat Completions compatibility helpers.
//!
//! This module contains request preparation and stream aggregation. Hosts feed
//! complete SSE data lines into [`ChatCompletionAccumulator`] and decide their
//! own HTTP status, timeout and trace handling.

use crate::ChatError;
use bytes::Bytes;
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// Maximum size of a complete Chat response for non-streaming clients.
pub const DEFAULT_MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

/// Chat request after compatibility preparation.
#[derive(Debug, Clone)]
pub struct PreparedChatRequest {
    /// Body sent to the upstream.
    pub body: Bytes,
    /// Whether the original client requested SSE.
    pub downstream_stream: bool,
    /// Model requested by the client.
    pub requested_model: String,
    /// Whether the original body was a valid JSON object.
    pub valid_json_object: bool,
}

/// Parses a Chat request and forces `stream:true` for the upstream.
///
/// Valid JSON objects are parsed so callers can apply their own mutations to
/// the returned value. Non-object bodies are returned unchanged.
pub fn prepare_chat_request(bytes: &[u8]) -> (PreparedChatRequest, Option<Value>) {
    let Ok(Value::Object(mut object)) = serde_json::from_slice::<Value>(bytes) else {
        return (
            PreparedChatRequest {
                body: Bytes::copy_from_slice(bytes),
                downstream_stream: false,
                requested_model: String::new(),
                valid_json_object: false,
            },
            None,
        );
    };
    let downstream_stream = object
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let requested_model = object
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    object.insert("stream".to_string(), Value::Bool(true));
    let value = Value::Object(object);
    let body = Bytes::from(serde_json::to_vec(&value).expect("JSON values serialize"));
    (
        PreparedChatRequest {
            body,
            downstream_stream,
            requested_model,
            valid_json_object: true,
        },
        Some(value),
    )
}

/// Re-encodes a value after a host-specific mutation.
pub fn encode_chat(value: &Value) -> Bytes {
    Bytes::from(serde_json::to_vec(value).expect("JSON values serialize"))
}

#[derive(Default)]
struct AggregateToolCall {
    id: String,
    kind: String,
    name: String,
    arguments: String,
    saw_arguments: bool,
}

#[derive(Default)]
struct AggregateChoice {
    content: String,
    content_parts: Vec<Value>,
    reasoning_content: String,
    has_content: bool,
    tool_calls: BTreeMap<usize, AggregateToolCall>,
    legacy_function_call: Option<AggregateToolCall>,
    finish_reason: Option<String>,
}

/// Incremental Chat SSE to complete Chat completion aggregator.
#[derive(Default)]
pub struct ChatCompletionAccumulator {
    id: Option<String>,
    created: Option<u64>,
    model: Option<String>,
    system_fingerprint: Option<Value>,
    usage: Option<Value>,
    choices: BTreeMap<usize, AggregateChoice>,
    next_tool_key: usize,
    saw_done: bool,
    saw_finish: bool,
}

impl ChatCompletionAccumulator {
    /// Consumes one SSE data payload, including `[DONE]`.
    pub fn push_data(&mut self, payload: &str) -> Result<(), ChatError> {
        if payload == "[DONE]" {
            self.saw_done = true;
            return Ok(());
        }
        let chunk: Value = serde_json::from_str(payload)
            .map_err(|error| ChatError::new("upstream_server_error", "upstream_error", format!("无效 SSE JSON: {error}")))?;
        if let Some(error) = chunk.get("error").filter(|value| !value.is_null()) {
            return Err(normalized_chat_error(error));
        }
        if chunk.get("choices").is_none() {
            let normalized = crate::responses::normalize_chat_error(Some(&chunk));
            let has_error_shape = chunk.get("detail").is_some()
                || chunk.get("message").is_some()
                || (chunk.get("base_resp").is_some() && chunk.get("usage").is_none());
            if has_error_shape || !matches!(chunk, Value::Object(_)) {
                return Err(ChatError {
                    code: normalized.code.unwrap_or_else(|| "upstream_error".to_string()),
                    error_type: normalized.error_type,
                    message: normalized.message,
                    param: normalized.param,
                });
            }
        }
        if let Some(id) = chunk.get("id").and_then(Value::as_str).filter(|id| !id.is_empty()) {
            self.id = Some(id.to_string());
        }
        if let Some(created) = chunk.get("created").and_then(Value::as_u64) {
            self.created = Some(created);
        }
        if let Some(model) = chunk.get("model").and_then(Value::as_str).filter(|model| !model.is_empty()) {
            self.model = Some(model.to_string());
        }
        if let Some(fingerprint) = chunk.get("system_fingerprint").filter(|value| !value.is_null()) {
            self.system_fingerprint = Some(fingerprint.clone());
        }
        if let Some(usage) = chunk.get("usage").filter(|value| !value.is_null()) {
            self.usage = Some(usage.clone());
        }
        for choice in chunk.get("choices").and_then(Value::as_array).into_iter().flatten() {
            let index = choice.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
            let state = self.choices.entry(index).or_default();
            let delta = choice.get("delta").or_else(|| choice.get("message")).unwrap_or(&Value::Null);
            match delta.get("content") {
                Some(Value::String(content)) => {
                    state.has_content = true;
                    state.content.push_str(content);
                }
                Some(Value::Array(parts)) => {
                    state.has_content = true;
                    state.content_parts.extend(parts.iter().cloned());
                }
                _ => {}
            }
            if let Some(refusal) = delta.get("refusal").and_then(Value::as_str).filter(|refusal| !refusal.is_empty()) {
                state.has_content = true;
                state.content_parts.push(json!({"type": "refusal", "refusal": refusal}));
            }
            if let Some(reasoning) = crate::responses::reasoning_text(delta) {
                state.reasoning_content.push_str(&reasoning);
            }
            if let Some(function_call) = delta.get("function_call") {
                let legacy = state.legacy_function_call.get_or_insert_with(Default::default);
                if let Some(name) = function_call.get("name").and_then(Value::as_str) {
                    legacy.name = name.to_string();
                }
                if let Some(arguments) = function_call.get("arguments").and_then(Value::as_str) {
                    legacy.arguments.push_str(arguments);
                }
            }
            for tool in delta.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
                let tool_index = aggregate_tool_key(&state.tool_calls, &mut self.next_tool_key, tool);
                let tool_state = state.tool_calls.entry(tool_index).or_default();
                if let Some(id) = tool.get("id").and_then(Value::as_str) {
                    tool_state.id = id.to_string();
                }
                if let Some(kind) = tool.get("type").and_then(Value::as_str) {
                    tool_state.kind = kind.to_string();
                }
                if let Some(name) = tool.pointer("/function/name").and_then(Value::as_str) {
                    tool_state.name = name.to_string();
                }
                if let Some(arguments) = tool.pointer("/function/arguments").and_then(Value::as_str) {
                    tool_state.saw_arguments = true;
                    tool_state.arguments.push_str(arguments);
                }
            }
            if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str).filter(|reason| !reason.is_empty()) {
                if state.finish_reason.is_none() {
                    state.finish_reason = Some(reason.to_string());
                    self.saw_finish = true;
                }
            }
        }
        Ok(())
    }

    /// Returns true after a `[DONE]` marker was observed.
    pub fn saw_done(&self) -> bool {
        self.saw_done
    }

    /// Builds the standard non-streaming Chat completion.
    pub fn finish(self, requested_model: &str) -> Result<Value, ChatError> {
        if !self.saw_done && !self.saw_finish {
            return Err(ChatError::new("incomplete_stream", "upstream_error", "上游 SSE 缺少完成分片或 [DONE]"));
        }
        if self.choices.is_empty() {
            return Err(ChatError::new("incomplete_stream", "upstream_error", "上游 SSE 已完成但没有有效输出"));
        }
        let choices = self.choices.into_iter().map(|(index, mut choice)| {
            if choice.finish_reason.is_none() {
                return Err(ChatError::new("incomplete_stream", "upstream_error", format!("choice {index} 有输出但缺少 finish_reason")));
            }
            let mut message = json!({
                "role": "assistant",
                "content": if !choice.content_parts.is_empty() {
                    let mut parts = Vec::new();
                    if !choice.content.is_empty() {
                        parts.push(json!({"type": "text", "text": choice.content}));
                    }
                    parts.extend(choice.content_parts);
                    Value::Array(parts)
                } else if choice.has_content {
                    Value::String(choice.content)
                } else {
                    Value::Null
                }
            });
            if !choice.reasoning_content.is_empty() {
                message["reasoning_content"] = Value::String(choice.reasoning_content);
            }
            if !choice.tool_calls.is_empty() {
                let mut calls = Vec::with_capacity(choice.tool_calls.len());
                for tool in choice.tool_calls.into_values() {
                    if tool.name.is_empty() {
                        return Err(ChatError::new("upstream_tool_call_dropped", "upstream_error", "tool call 缺少函数名"));
                    }
                    let arguments = if !tool.saw_arguments || tool.arguments.trim().is_empty() {
                        "{}".to_string()
                    } else {
                        if !serde_json::from_str::<Value>(&tool.arguments).is_ok_and(|value| value.is_object()) {
                            return Err(ChatError::new("malformed_tool_call_arguments", "upstream_error", "tool call 参数不是合法 JSON 对象"));
                        }
                        tool.arguments
                    };
                    calls.push(json!({
                        "id": tool.id,
                        "type": if tool.kind.is_empty() { "function" } else { tool.kind.as_str() },
                        "function": {"name": tool.name, "arguments": arguments}
                    }));
                }
                message["tool_calls"] = Value::Array(calls);
                if choice.finish_reason.as_deref() != Some("length") && choice.finish_reason.as_deref() != Some("content_filter") {
                    choice.finish_reason = Some("tool_calls".to_string());
                }
            } else if matches!(choice.finish_reason.as_deref(), Some("tool_calls" | "function_call")) {
                return Err(ChatError::new("incomplete_stream", "upstream_error", format!("choice {index} 以 tool_calls 结束但没有工具调用")));
            }
            if let Some(function_call) = choice.legacy_function_call {
                message["function_call"] = json!({"name": function_call.name, "arguments": function_call.arguments});
            }
            Ok(json!({"index": index, "message": message, "finish_reason": choice.finish_reason}))
        }).collect::<Result<Vec<_>, _>>()?;
        let mut completion = json!({
            "id": self.id.unwrap_or_else(|| format!("chatcmpl_{}", now_millis())),
            "object": "chat.completion",
            "created": self.created.unwrap_or_else(|| now_seconds()),
            "model": self.model.unwrap_or_else(|| requested_model.to_string()),
            "choices": choices
        });
        if let Some(fingerprint) = self.system_fingerprint {
            completion["system_fingerprint"] = fingerprint;
        }
        if let Some(usage) = self.usage {
            completion["usage"] = usage;
        }
        Ok(completion)
    }
}

/// Processes one complete SSE line.
pub fn process_sse_line(acc: &mut ChatCompletionAccumulator, line: &[u8]) -> Result<(), ChatError> {
    let line = String::from_utf8_lossy(line);
    let Some(payload) = line.trim_end().strip_prefix("data:") else {
        return Ok(());
    };
    acc.push_data(payload.trim())
}

fn normalized_chat_error(value: &Value) -> ChatError {
    let normalized = crate::responses::normalize_chat_error(Some(value));
    ChatError {
        code: normalized.code.unwrap_or_else(|| "upstream_error".to_string()),
        error_type: normalized.error_type,
        message: normalized.message,
        param: normalized.param,
    }
}

fn aggregate_tool_key(calls: &BTreeMap<usize, AggregateToolCall>, next_key: &mut usize, tool: &Value) -> usize {
    let id = tool.get("id").and_then(Value::as_str).filter(|id| !id.is_empty());
    if let Some(id) = id {
        if let Some((key, _)) = calls.iter().find(|(_, state)| state.id == id) {
            return *key;
        }
    }
    if let Some(index) = tool.get("index").and_then(Value::as_u64) {
        let index = usize::try_from(index).unwrap_or(usize::MAX - 1);
        if let Some(state) = calls.get(&index) {
            if id.is_none() || state.id.is_empty() {
                return index;
            }
        } else {
            return index;
        }
    }
    let last_key = calls.keys().next_back().copied();
    if id.is_none() {
        return last_key.unwrap_or(0);
    }
    if let Some(key) = last_key.and_then(|key| key.checked_add(1)) {
        return key;
    }
    let key = *next_key;
    *next_key = next_key.saturating_add(1);
    while calls.contains_key(&key) {
        *next_key = next_key.saturating_add(1);
    }
    key
}

fn now_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn now_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preparation_forces_upstream_stream_and_keeps_client_intent() {
        let input = br#"{"model":"deepseek-v4","stream":false}"#;
        let (prepared, value) = prepare_chat_request(input);
        let value = value.expect("valid JSON object should be parsed");

        assert_eq!(value["stream"], true);
        assert!(!prepared.downstream_stream);
        assert_eq!(prepared.requested_model, "deepseek-v4");
        let body: Value = serde_json::from_slice(&prepared.body).unwrap();
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn preparation_defaults_missing_stream_to_non_streaming() {
        let input = br#"{"model":"model"}"#;
        let (prepared, value) = prepare_chat_request(input);
        let value = value.unwrap();

        assert_eq!(value["stream"], true);
        assert!(!prepared.downstream_stream);
    }

    #[test]
    fn preparation_preserves_original_streaming_intent_and_non_json_body() {
        let input = br#"{"model":"model","stream":true}"#;
        let (prepared, value) = prepare_chat_request(input);
        assert_eq!(value.unwrap()["stream"], true);
        assert!(prepared.downstream_stream);

        let (prepared, value) = prepare_chat_request(b"not json");
        assert!(!prepared.valid_json_object);
        assert!(!prepared.downstream_stream);
        assert_eq!(prepared.body.as_ref(), b"not json");
        assert!(value.is_none());
    }

    #[test]
    fn aggregation_combines_text_reasoning_and_tools() {
        let mut accumulator = ChatCompletionAccumulator::default();
        process_sse_line(
            &mut accumulator,
            br#"data: {"id":"chatcmpl-1","created":1,"model":"upstream","choices":[{"index":0,"delta":{"reasoning_content":"think"},"finish_reason":null}]}"#,
        )
        .unwrap();
        process_sse_line(
            &mut accumulator,
            br#"data: {"choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}"#,
        )
        .unwrap();
        process_sse_line(
            &mut accumulator,
            br#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"run","arguments":"{\"a\":"}}]},"finish_reason":"tool_calls"}]}"#,
        )
        .unwrap();
        process_sse_line(
            &mut accumulator,
            br#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]},"finish_reason":null}]}"#,
        )
        .unwrap();
        process_sse_line(
            &mut accumulator,
            br#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}"#,
        )
        .unwrap();
        process_sse_line(&mut accumulator, b"data: [DONE]").unwrap();

        let result = accumulator.finish("requested").unwrap();
        assert_eq!(result["model"], "upstream");
        assert_eq!(result["choices"][0]["message"]["content"], "hello");
        assert_eq!(
            result["choices"][0]["message"]["reasoning_content"],
            "think"
        );
        assert_eq!(
            result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{\"a\":1}"
        );
        assert_eq!(
            result["choices"][0]["message"]["tool_calls"][0]["id"],
            "call-1"
        );
        assert_eq!(result["usage"]["total_tokens"], 7);
    }

    #[test]
    fn aggregation_rejects_truncated_and_error_streams() {
        let mut accumulator = ChatCompletionAccumulator::default();
        process_sse_line(
            &mut accumulator,
            br#"data: {"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}"#,
        )
        .unwrap();
        assert_eq!(
            accumulator.finish("model").unwrap_err().code,
            "incomplete_stream"
        );

        let mut accumulator = ChatCompletionAccumulator::default();
        assert!(process_sse_line(
            &mut accumulator,
            br#"data: {"error":{"message":"rate limited"}}"#,
        )
        .is_err());
    }

    #[test]
    fn aggregation_merges_multiple_choices() {
        let mut accumulator = ChatCompletionAccumulator::default();
        process_sse_line(
            &mut accumulator,
            br#"data: {"choices":[{"index":1,"delta":{"content":"second"},"finish_reason":"stop"},{"index":0,"delta":{"content":"first"},"finish_reason":"stop"}],"usage":{"total_tokens":2}}"#,
        )
        .unwrap();

        let result = accumulator.finish("model").unwrap();
        assert_eq!(result["choices"][0]["index"], 0);
        assert_eq!(result["choices"][0]["message"]["content"], "first");
        assert_eq!(result["choices"][1]["index"], 1);
        assert_eq!(result["choices"][1]["message"]["content"], "second");
    }
}
