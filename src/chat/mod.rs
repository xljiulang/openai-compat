//! Chat Completions compatibility helpers.
//!
//! This module contains request preparation and stream aggregation. Callers feed
//! complete SSE data lines into [`ChatCompletionAccumulator`] and keep their own
//! HTTP status, timeout and trace handling.

use crate::{ChatError, CompatError, ErrorKind};
use bytes::Bytes;
use serde_json::{json, Value};
use std::collections::BTreeMap;

/// Rule identifier callers can log when roles are normalized.
pub const ROLE_RULE_ID: &str = "compat/developer-role";

type ChatMutator<'a> = dyn FnMut(&mut Value) -> Result<bool, CompatError> + 'a;

/// Low-level Chat request metadata produced by request preparation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PreparedChatRequest {
    /// Replacement body. `None` means the original request bytes are reused.
    pub(crate) body: Option<Bytes>,
    /// Whether the original client requested SSE.
    downstream_stream: bool,
    /// Model requested by the client.
    requested_model: String,
    /// Whether the original body was a valid JSON object.
    valid_json_object: bool,
}

impl PreparedChatRequest {
    /// Returns true when the original client asked for an SSE response.
    pub fn downstream_stream(&self) -> bool {
        self.downstream_stream
    }

    /// Returns the model the client asked for, or an empty string.
    pub fn requested_model(&self) -> &str {
        &self.requested_model
    }

    /// Returns true when the client body was a valid JSON object.
    pub fn valid_json_object(&self) -> bool {
        self.valid_json_object
    }
}

/// Prepared Chat request. Use the accessors rather than depending on fields.
#[derive(Debug)]
#[non_exhaustive]
pub struct PreparedChat {
    request: PreparedChatRequest,
    /// Parsed request for caller-side prompt policies.
    value: Option<Value>,
    /// Whether the body differs from the original request bytes.
    body_changed: bool,
    role_changes: RoleChanges,
}

impl PreparedChat {
    /// Returns the low-level request metadata.
    pub fn request(&self) -> &PreparedChatRequest {
        &self.request
    }

    /// Returns the parsed request, or `None` when the body was not a JSON object.
    pub fn value(&self) -> Option<&Value> {
        self.value.as_ref()
    }

    /// Returns true when the upstream body differs from the client bytes.
    pub fn body_changed(&self) -> bool {
        self.body_changed
    }

    /// Returns a replacement body when the library changed it.
    pub fn rewritten_body(&self) -> Option<Bytes> {
        self.request.body.clone()
    }

    /// Reports developer role conversions performed during preparation.
    pub fn role_changes(&self) -> RoleChanges {
        self.role_changes
    }
}

/// Parses a valid Chat JSON object (not an array) without rewriting it.
pub fn parse_chat_request(bytes: &[u8]) -> Option<Value> {
    serde_json::from_slice::<Value>(bytes)
        .ok()
        .filter(Value::is_object)
}

/// Parses a Chat request and forces `stream:true` for the upstream.
///
/// Valid JSON objects are returned to the caller for host-specific mutations.
/// Byte changes preserve every field except the top-level `stream` value.
pub fn prepare_chat_request(bytes: &[u8]) -> PreparedChat {
    prepare_chat_request_inner(bytes, None)
        .expect("preparing a request without a mutator cannot fail")
}

/// Prepares a Chat request while letting the caller apply prompt policies.
///
/// The callback runs after parsing and after `stream:true` is inserted. The
/// library then normalizes roles and encodes the final value, so the returned
/// JSON and body cannot diverge.
///
/// # Errors
///
/// Returns the callback's [`CompatError`] unchanged when the caller rejects
/// the request.
pub fn prepare_chat_request_with<F>(
    bytes: &[u8],
    mut mutator: F,
) -> Result<PreparedChat, CompatError>
where
    F: FnMut(&mut Value) -> Result<bool, CompatError>,
{
    prepare_chat_request_inner(bytes, Some(&mut mutator))
}

fn prepare_chat_request_inner(
    bytes: &[u8],
    mutator: Option<&mut ChatMutator<'_>>,
) -> Result<PreparedChat, CompatError> {
    let Ok(Value::Object(mut object)) = serde_json::from_slice::<Value>(bytes) else {
        return Ok(PreparedChat {
            request: PreparedChatRequest {
                body: None,
                downstream_stream: false,
                requested_model: String::new(),
                valid_json_object: false,
            },
            value: None,
            body_changed: false,
            role_changes: RoleChanges::default(),
        });
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
    let stream_was_true = object.get("stream") == Some(&Value::Bool(true));
    object.insert("stream".to_string(), Value::Bool(true));
    let mut value = Value::Object(object);
    let host_changed = mutator.map_or(Ok(false), |mutator| mutator(&mut value))?;
    let role_changes = normalize_roles(&mut value);
    let body_changed = !stream_was_true || role_changes.changed() || host_changed;
    let body = if body_changed {
        if host_changed || role_changes.changed() {
            Some(encode_chat(&value))
        } else {
            Some(force_top_level_stream_true(bytes))
        }
    } else {
        None
    };
    Ok(PreparedChat {
        request: PreparedChatRequest {
            body,
            downstream_stream,
            requested_model,
            valid_json_object: true,
        },
        value: Some(value),
        body_changed,
        role_changes,
    })
}

/// Number of message roles normalized for Chat upstream compatibility.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct RoleChanges {
    /// Messages changed from `developer` to `system`.
    pub developer_to_system: usize,
}

impl RoleChanges {
    /// Returns true when at least one role was changed.
    pub fn changed(self) -> bool {
        self.developer_to_system > 0
    }
}

/// Normalizes message roles accepted by Chat upstreams.
///
/// The upstream only accepts `system` for developer instructions. The match is
/// ASCII case-insensitive; every other role and all message content is left
/// untouched.
pub fn normalize_roles(value: &mut Value) -> RoleChanges {
    let mut changes = RoleChanges::default();
    let Some(messages) = value.get_mut("messages").and_then(Value::as_array_mut) else {
        return changes;
    };
    for message in messages {
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        let is_developer = object
            .get("role")
            .and_then(Value::as_str)
            .is_some_and(|role| role.eq_ignore_ascii_case("developer"));
        if is_developer {
            object.insert("role".to_string(), Value::String("system".to_string()));
            changes.developer_to_system += 1;
        }
    }
    changes
}

/// Replaces only the top-level `stream` value, preserving all other bytes.
///
/// Invalid JSON or nested `stream` fields are returned unchanged. This mirrors
/// prefix-preserving behavior used by compatibility gateways.
///
/// # Examples
///
/// ```
/// use openai_compat::chat::force_top_level_stream_true;
///
/// let body = force_top_level_stream_true(br#"{"model":"m","stream":false}"#);
/// assert_eq!(&body[..], br#"{"model":"m","stream":true}"#);
/// ```
pub fn force_top_level_stream_true(bytes: &[u8]) -> Bytes {
    let Some(object_start) = bytes.iter().position(|byte| !byte.is_ascii_whitespace()) else {
        return Bytes::copy_from_slice(bytes);
    };
    if bytes[object_start] != b'{' {
        return Bytes::copy_from_slice(bytes);
    }
    let mut depth = 1usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut index = object_start + 1;
    let mut closing = None;
    while index < bytes.len() {
        let byte = bytes[index];
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            index += 1;
            continue;
        }
        match byte {
            b'"' => {
                if depth == 1 {
                    let key_start = index + 1;
                    let mut key_end = key_start;
                    let mut key_escaped = false;
                    while key_end < bytes.len() {
                        if key_escaped {
                            key_escaped = false;
                        } else if bytes[key_end] == b'\\' {
                            key_escaped = true;
                        } else if bytes[key_end] == b'"' {
                            break;
                        }
                        key_end += 1;
                    }
                    if key_end < bytes.len() && &bytes[key_start..key_end] == b"stream" {
                        let mut value_start = key_end + 1;
                        while value_start < bytes.len()
                            && (bytes[value_start].is_ascii_whitespace()
                                || bytes[value_start] == b':')
                        {
                            value_start += 1;
                        }
                        let mut value_end = value_start;
                        while value_end < bytes.len() && !matches!(bytes[value_end], b',' | b'}') {
                            value_end += 1;
                        }
                        while value_end > value_start && bytes[value_end - 1].is_ascii_whitespace()
                        {
                            value_end -= 1;
                        }
                        return match &bytes[value_start..value_end] {
                            b"true" => Bytes::copy_from_slice(bytes),
                            b"false" => {
                                let mut replaced =
                                    Vec::with_capacity(bytes.len().saturating_sub(4));
                                replaced.extend_from_slice(&bytes[..value_start]);
                                replaced.extend_from_slice(b"true");
                                replaced.extend_from_slice(&bytes[value_end..]);
                                Bytes::from(replaced)
                            }
                            _ => Bytes::copy_from_slice(bytes),
                        };
                    }
                }
                in_string = true;
            }
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    closing = Some(index);
                    break;
                }
            }
            _ => {}
        }
        index += 1;
    }
    if depth != 0 {
        return Bytes::copy_from_slice(bytes);
    }
    if let Some(closing) = closing {
        let previous = bytes[..closing]
            .iter()
            .rfind(|byte| !byte.is_ascii_whitespace())
            .copied();
        let mut inserted = Vec::with_capacity(bytes.len() + 15);
        inserted.extend_from_slice(&bytes[..closing]);
        if !matches!(previous, Some(b'{') | Some(b',')) {
            inserted.push(b',');
        }
        inserted.extend_from_slice(b"\"stream\":true");
        inserted.extend_from_slice(&bytes[closing..]);
        return Bytes::from(inserted);
    }
    Bytes::copy_from_slice(bytes)
}

/// Re-encodes a value after a caller-side mutation.
///
/// # Panics
///
/// Panics only if `serde_json` cannot serialize a [`Value`], which cannot
/// happen for a value built from the JSON data model.
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
#[non_exhaustive]
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
    max_bytes: usize,
    bytes_received: usize,
}

impl Default for ChatCompletionAccumulator {
    fn default() -> Self {
        Self::new(crate::DEFAULT_MAX_RESPONSE_BYTES)
    }
}

impl ChatCompletionAccumulator {
    /// Creates an accumulator with an explicit payload byte limit.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            id: None,
            created: None,
            model: None,
            system_fingerprint: None,
            usage: None,
            choices: BTreeMap::new(),
            next_tool_key: 0,
            saw_done: false,
            saw_finish: false,
            max_bytes,
            bytes_received: 0,
        }
    }

    /// Consumes one SSE data payload, including `[DONE]`.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::LimitExceeded`] when the accumulated payload passes
    /// the configured limit, [`ErrorKind::Upstream`] when the upstream reports
    /// an error or sends a payload that is not JSON, and
    /// [`ErrorKind::IncompleteStream`] is reported later by
    /// [`ChatCompletionAccumulator::finish`] when the stream ends early.
    pub fn push_data(&mut self, payload: &str) -> Result<(), ChatError> {
        if self.max_bytes == 0 {
            return Err(CompatError::new(
                ErrorKind::LimitExceeded,
                "upstream_server_error",
                "upstream_error",
                "aggregated upstream response exceeds configured limit",
            )
            .with_retryable(true));
        }
        if payload.len() > self.max_bytes {
            return Err(CompatError::new(
                ErrorKind::LimitExceeded,
                "upstream_server_error",
                "upstream_error",
                "aggregated upstream response exceeds configured limit",
            )
            .with_retryable(true));
        }
        self.bytes_received = self.bytes_received.saturating_add(payload.len());
        if self.bytes_received > self.max_bytes {
            return Err(CompatError::new(
                ErrorKind::LimitExceeded,
                "upstream_server_error",
                "upstream_error",
                "aggregated upstream response exceeds configured limit",
            )
            .with_retryable(true));
        }
        if payload == "[DONE]" {
            self.saw_done = true;
            return Ok(());
        }
        let chunk: Value = serde_json::from_str(payload).map_err(|error| {
            CompatError::new(
                ErrorKind::Upstream,
                "upstream_server_error",
                "upstream_error",
                format!("invalid SSE JSON from upstream: {error}"),
            )
        })?;
        if let Some(error) = chunk.get("error").filter(|value| !value.is_null()) {
            return Err(normalized_chat_error(error));
        }
        if chunk.get("choices").is_none() {
            let normalized = crate::responses::normalize_chat_error(Some(&chunk));
            let has_error_shape = chunk.get("detail").is_some()
                || chunk.get("message").is_some()
                || (chunk.get("base_resp").is_some() && chunk.get("usage").is_none());
            if has_error_shape || !matches!(chunk, Value::Object(_)) {
                let mut error = CompatError::new(
                    ErrorKind::Upstream,
                    normalized
                        .code
                        .unwrap_or_else(|| "upstream_error".to_string()),
                    normalized.error_type,
                    normalized.message,
                );
                if let Some(param) = normalized.param {
                    error = error.with_param(param);
                }
                return Err(error);
            }
        }
        if let Some(id) = chunk
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
        {
            self.id = Some(id.to_string());
        }
        if let Some(created) = chunk.get("created").and_then(Value::as_u64) {
            self.created = Some(created);
        }
        if let Some(model) = chunk
            .get("model")
            .and_then(Value::as_str)
            .filter(|model| !model.is_empty())
        {
            self.model = Some(model.to_string());
        }
        if let Some(fingerprint) = chunk
            .get("system_fingerprint")
            .filter(|value| !value.is_null())
        {
            self.system_fingerprint = Some(fingerprint.clone());
        }
        if let Some(usage) = chunk.get("usage").filter(|value| !value.is_null()) {
            self.usage = Some(usage.clone());
        }
        for choice in chunk
            .get("choices")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let index = choice.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
            let state = self.choices.entry(index).or_default();
            let delta = choice
                .get("delta")
                .or_else(|| choice.get("message"))
                .unwrap_or(&Value::Null);
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
            if let Some(refusal) = delta
                .get("refusal")
                .and_then(Value::as_str)
                .filter(|refusal| !refusal.is_empty())
            {
                state.has_content = true;
                state.content_parts.push(json!({
                    "type": "refusal",
                    "refusal": refusal
                }));
            }
            if let Some(reasoning) = crate::responses::reasoning_text(delta) {
                state.reasoning_content.push_str(&reasoning);
            }
            if let Some(function_call) = delta.get("function_call") {
                let legacy = state
                    .legacy_function_call
                    .get_or_insert_with(Default::default);
                if let Some(name) = function_call.get("name").and_then(Value::as_str) {
                    legacy.name = name.to_string();
                }
                if let Some(arguments) = function_call.get("arguments").and_then(Value::as_str) {
                    legacy.arguments.push_str(arguments);
                }
            }
            for tool in delta
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let tool_index =
                    aggregate_tool_key(&state.tool_calls, &mut self.next_tool_key, tool);
                let tool_state = state.tool_calls.entry(tool_index).or_default();
                if let Some(id) = tool
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|id| !id.is_empty())
                {
                    tool_state.id = id.to_string();
                }
                if let Some(kind) = tool.get("type").and_then(Value::as_str) {
                    tool_state.kind = kind.to_string();
                }
                if let Some(name) = tool.pointer("/function/name").and_then(Value::as_str) {
                    tool_state.name = name.to_string();
                }
                if let Some(arguments) = tool.pointer("/function/arguments").and_then(Value::as_str)
                {
                    tool_state.saw_arguments = true;
                    tool_state.arguments.push_str(arguments);
                }
            }
            if let Some(reason) = choice
                .get("finish_reason")
                .and_then(Value::as_str)
                .filter(|reason| !reason.is_empty())
            {
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
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::IncompleteStream`] when the upstream stream ended
    /// without a completion marker or without usable output, and
    /// [`ErrorKind::Upstream`] when a tool call is missing its name or carries
    /// arguments that are not a JSON object.
    pub fn finish(self, requested_model: &str) -> Result<Value, ChatError> {
        if !self.saw_done && !self.saw_finish {
            return Err(CompatError::new(
                ErrorKind::IncompleteStream,
                "incomplete_stream",
                "upstream_error",
                "upstream SSE ended without a finish chunk or [DONE]",
            ));
        }
        if self.choices.is_empty() {
            return Err(CompatError::new(
                ErrorKind::IncompleteStream,
                "incomplete_stream",
                "upstream_error",
                "upstream SSE completed without usable output",
            ));
        }
        let choices = self
            .choices
            .into_iter()
            .map(|(index, mut choice)| {
                if choice.finish_reason.is_none() {
                    return Err(CompatError::new(
                        ErrorKind::IncompleteStream,
                        "incomplete_stream",
                        "upstream_error",
                        format!("choice {index} produced output without a finish_reason"),
                    ));
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
                            return Err(CompatError::new(
                                ErrorKind::Upstream,
                                "upstream_tool_call_dropped",
                                "upstream_error",
                                "upstream tool call is missing its function name",
                            ));
                        }
                        let arguments = if !tool.saw_arguments || tool.arguments.trim().is_empty() {
                            "{}".to_string()
                        } else {
                            if !serde_json::from_str::<Value>(&tool.arguments)
                                .is_ok_and(|value| value.is_object())
                            {
                                return Err(CompatError::new(
                                    ErrorKind::Upstream,
                                    "malformed_tool_call_arguments",
                                    "upstream_error",
                                    "upstream tool call arguments are not a JSON object",
                                ));
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
                    if choice.finish_reason.as_deref() != Some("length")
                        && choice.finish_reason.as_deref() != Some("content_filter")
                    {
                        choice.finish_reason = Some("tool_calls".to_string());
                    }
                } else if matches!(
                    choice.finish_reason.as_deref(),
                    Some("tool_calls" | "function_call")
                ) {
                    return Err(CompatError::new(
                        ErrorKind::IncompleteStream,
                        "incomplete_stream",
                        "upstream_error",
                        format!("choice {index} finished with tool_calls but sent no tool call"),
                    ));
                }
                if let Some(function_call) = choice.legacy_function_call {
                    message["function_call"] = json!({
                        "name": function_call.name,
                        "arguments": function_call.arguments
                    });
                }
                Ok(json!({"index": index, "message": message, "finish_reason": choice.finish_reason}))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut completion = json!({
            "id": self.id.unwrap_or_else(|| format!("chatcmpl_{}", now_millis())),
            "object": "chat.completion",
            "created": self.created.unwrap_or_else(now_seconds),
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
///
/// Lines without a `data:` prefix, such as `event:` and blank lines, are
/// ignored.
///
/// # Errors
///
/// Propagates any error from [`ChatCompletionAccumulator::push_data`].
pub fn process_sse_line(acc: &mut ChatCompletionAccumulator, line: &[u8]) -> Result<(), ChatError> {
    let line = String::from_utf8_lossy(line);
    let Some(payload) = line.trim_end().strip_prefix("data:") else {
        return Ok(());
    };
    acc.push_data(payload.trim())
}

fn normalized_chat_error(value: &Value) -> ChatError {
    let normalized = crate::responses::normalize_chat_error(Some(value));
    let mut error = ChatError::new(
        ErrorKind::Upstream,
        normalized
            .code
            .unwrap_or_else(|| "upstream_error".to_string()),
        normalized.error_type,
        normalized.message,
    );
    if let Some(param) = normalized.param {
        error = error.with_param(param);
    }
    error
}

fn aggregate_tool_key(
    calls: &BTreeMap<usize, AggregateToolCall>,
    next_key: &mut usize,
    tool: &Value,
) -> usize {
    let id = tool
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty());
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
mod tests;
