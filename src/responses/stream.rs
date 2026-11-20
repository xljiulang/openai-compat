use super::*;
use bytes::Bytes;
use std::io::Write as _;

fn clean_text(cleaner: Option<ResponseCleaner>, text: &str) -> String {
    cleaner
        .map(|cleaner| cleaner(text))
        .unwrap_or_else(|| text.to_string())
}

#[derive(Clone)]
pub(super) struct TextTrack {
    id: String,
    index: usize,
    text: String,
}

#[derive(Clone)]
pub(super) struct CallTrack {
    item_id: String,
    call_id: String,
    display_name: String,
    namespace: Option<String>,
    custom: bool,
    tool_search: bool,
    arguments: String,
    emitted_arguments_len: usize,
    extra_content: Option<Value>,
    output_index: usize,
    added: bool,
}

#[derive(Clone)]
struct Failure {
    error_type: String,
    code: String,
    message: String,
    param: Option<String>,
}

/// Stateful Chat SSE chunk to Responses SSE translator.
pub struct ResponsesStream {
    response_id: String,
    model: String,
    created_at: u64,
    service_tier: Option<Value>,
    system_fingerprint: Option<Value>,
    shape: ToolShape,
    created: bool,
    next_index: usize,
    reasoning: Option<TextTrack>,
    reasoning_pending: String,
    text: Option<TextTrack>,
    inline_think: InlineThinkSplitter,
    calls: BTreeMap<usize, CallTrack>,
    completed_call_ids: std::collections::HashSet<String>,
    next_call_key_to_add: usize,
    output: Vec<Value>,
    finish_reason: Option<String>,
    usage: Option<Value>,
    failure: Option<Failure>,
    finished: bool,
    cleaner: Option<ResponseCleaner>,
}

impl ResponsesStream {
    /// Creates a translator for one downstream Responses request.
    pub fn new(response_id: String, model: String) -> Self {
        Self::with_shape(response_id, model, ToolShape::default())
    }

    /// Creates a translator with request-local tool metadata.
    pub fn with_shape(response_id: String, model: String, shape: ToolShape) -> Self {
        Self::with_shape_and_cleaner(response_id, model, shape, None)
    }

    /// Creates a translator with a host-specific assistant text cleaner.
    pub fn with_shape_and_cleaner(
        response_id: String,
        model: String,
        shape: ToolShape,
        cleaner: Option<ResponseCleaner>,
    ) -> Self {
        Self {
            response_id,
            model,
            created_at: unix_now(),
            service_tier: None,
            system_fingerprint: None,
            shape,
            created: false,
            next_index: 0,
            reasoning: None,
            reasoning_pending: String::new(),
            text: None,
            inline_think: InlineThinkSplitter::default(),
            calls: BTreeMap::new(),
            completed_call_ids: std::collections::HashSet::new(),
            next_call_key_to_add: 0,
            output: Vec::new(),
            finish_reason: None,
            usage: None,
            failure: None,
            finished: false,
            cleaner,
        }
    }

    /// Translates one Chat completion chunk into zero or more Responses SSE events.
    pub fn push(&mut self, chunk: &Value) -> Vec<Bytes> {
        if self.finished {
            return Vec::new();
        }
        if let Some(id) = chunk
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
        {
            self.response_id = if id.starts_with("resp_") {
                id.to_string()
            } else {
                format!("resp_{}", id.strip_prefix("chatcmpl-").unwrap_or(id))
            };
        }
        if let Some(model) = chunk
            .get("model")
            .and_then(Value::as_str)
            .filter(|model| !model.is_empty())
        {
            self.model = model.to_string();
        }
        if let Some(created) = chunk.get("created").and_then(Value::as_u64) {
            self.created_at = created;
        }
        if let Some(service_tier) = chunk.get("service_tier").filter(|value| !value.is_null()) {
            self.service_tier = Some(service_tier.clone());
        }
        if let Some(fingerprint) = chunk
            .get("system_fingerprint")
            .filter(|value| !value.is_null())
        {
            self.system_fingerprint = Some(fingerprint.clone());
        }
        let mut events = self.ensure_created();
        if let Some(error) = chunk.get("error").filter(|value| !value.is_null()) {
            let error = normalize_chat_error(Some(error));
            self.failure = Some(Failure {
                error_type: error.error_type,
                code: error.code.unwrap_or_else(|| "upstream_error".to_string()),
                message: error.message,
                param: error.param,
            });
            return events;
        }
        if let Some(usage) = chunk.get("usage").filter(|value| !value.is_null()) {
            self.usage = Some(usage.clone());
        }
        let Some(choices) = chunk.get("choices").and_then(Value::as_array) else {
            return events;
        };
        for choice in choices {
            let delta = choice.get("delta").unwrap_or(&Value::Null);
            if let Some(reasoning) = self.reasoning_fragment(delta) {
                events.extend(self.reasoning_text_fragment(&reasoning));
            }
            if let Some(content) = delta.get("content").and_then(Value::as_str) {
                if !content.is_empty() {
                    let content = clean_text(self.cleaner, content);
                    events.extend(self.inline_content_delta(&content));
                }
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                events.extend(self.flush_inline_think());
                for call in calls {
                    events.extend(self.tool_delta(call));
                }
            }
            if let Some(reason) = choice
                .get("finish_reason")
                .and_then(Value::as_str)
                .filter(|reason| !reason.is_empty() && *reason != "null")
            {
                self.finish_reason = Some(reason.to_string());
            }
            if matches!(
                choice.get("finish_reason").and_then(Value::as_str),
                Some("tool_calls" | "function_call")
            ) {
                events.extend(self.flush_ready_calls());
                events.extend(self.reasoning_done());
                events.extend(self.finish_text());
                events.extend(self.finish_added_calls());
            }
        }
        events
    }

    /// Emits terminal events exactly once.
    pub fn finish(&mut self) -> Vec<Bytes> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        let mut events = self.ensure_created();
        events.extend(self.flush_inline_think());
        events.extend(self.reasoning_done());
        events.extend(self.finish_text());
        events.extend(self.flush_ready_calls());
        for key in self.calls.keys().copied().collect::<Vec<_>>() {
            let (added, missing_name) = self
                .calls
                .get(&key)
                .map(|call| (call.added, call.display_name.is_empty()))
                .unwrap_or((true, true));
            if added || missing_name {
                continue;
            }
            if self.calls[&key].call_id.is_empty() {
                self.calls
                    .get_mut(&key)
                    .expect("call key came from self.calls")
                    .call_id = format!("call_{key}");
            }
            events.extend(self.add_call_events(key));
        }

        let missing_finish = self.finish_reason.is_none();
        if self.failure.is_none()
            && missing_finish
            && self.output.is_empty()
            && self.calls.is_empty()
        {
            self.failure = Some(Failure {
                error_type: "upstream_error".to_string(),
                code: "incomplete_stream".to_string(),
                message: "upstream stream ended without output or finish_reason".to_string(),
                param: None,
            });
        }
        let incomplete_reason = match self.finish_reason.as_deref() {
            Some("length") => Some("max_output_tokens"),
            Some("content_filter" | "sensitive") => Some("content_filter"),
            None if self.failure.is_none() => Some("max_output_tokens"),
            _ => None,
        };
        if self.failure.is_none() && incomplete_reason.is_none() {
            for call in self.calls.values() {
                if call.display_name.is_empty() {
                    self.failure = Some(Failure {
                        error_type: "upstream_error".to_string(),
                        code: "upstream_tool_call_dropped".to_string(),
                        message: "upstream returned a tool call without a function name"
                            .to_string(),
                        param: None,
                    });
                    break;
                }
                if !call.custom && normalize_arguments(&call.arguments).1 {
                    self.failure = Some(Failure {
                        error_type: "upstream_error".to_string(),
                        code: "malformed_tool_call_arguments".to_string(),
                        message: format!(
                            "tool call arguments for {} are invalid",
                            call.display_name
                        ),
                        param: None,
                    });
                    break;
                }
            }
        }
        if self.failure.is_none() {
            let calls = std::mem::take(&mut self.calls);
            events.extend(self.finish_calls(calls.into_values(), incomplete_reason));
        }
        let terminal = if let Some(failure) = self.failure.take() {
            sse_event(
                "response.failed",
                json!({
                    "type": "response.failed",
                    "response": {
                        "id": self.response_id,
                        "object": "response",
                        "created_at": self.created_at,
                        "status": "failed",
                        "model": self.model,
                        "service_tier": self.service_tier,
                        "system_fingerprint": self.system_fingerprint,
                        "output": self.output,
                        "error": {
                            "type": failure.error_type,
                            "code": failure.code,
                            "message": failure.message,
                            "param": failure.param,
                            "retryable": false
                        }
                    }
                }),
            )
        } else if let Some(reason) = incomplete_reason {
            sse_event(
                "response.incomplete",
                json!({
                    "type": "response.incomplete",
                    "response": {
                        "id": self.response_id,
                        "object": "response",
                        "created_at": self.created_at,
                        "status": "incomplete",
                        "model": self.model,
                        "service_tier": self.service_tier,
                        "system_fingerprint": self.system_fingerprint,
                        "output": self.output,
                        "error": null,
                        "incomplete_details": { "reason": reason },
                        "usage": usage_to_responses(self.usage.as_ref())
                    }
                }),
            )
        } else {
            sse_event(
                "response.completed",
                json!({
                    "type": "response.completed",
                    "response": {
                        "id": self.response_id,
                        "object": "response",
                        "created_at": self.created_at,
                        "status": "completed",
                        "model": self.model,
                        "service_tier": self.service_tier,
                        "system_fingerprint": self.system_fingerprint,
                        "output": self.output,
                        "error": null,
                        "usage": usage_to_responses(self.usage.as_ref())
                    }
                }),
            )
        };
        events.push(terminal);
        events
    }

    /// Marks the stream as failed and emits its terminal event.
    pub fn fail(&mut self, code: impl Into<String>, message: impl Into<String>) -> Vec<Bytes> {
        if self.finished {
            return Vec::new();
        }
        self.failure = Some(Failure {
            error_type: "upstream_error".to_string(),
            code: code.into(),
            message: message.into(),
            param: None,
        });
        self.finish()
    }

    fn ensure_created(&mut self) -> Vec<Bytes> {
        if self.created {
            return Vec::new();
        }
        self.created = true;
        let response = json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": "in_progress",
            "model": self.model,
            "service_tier": self.service_tier,
            "system_fingerprint": self.system_fingerprint,
            "output": [],
            "error": null
        });
        vec![
            sse_event(
                "response.created",
                json!({ "type": "response.created", "response": response }),
            ),
            sse_event(
                "response.in_progress",
                json!({ "type": "response.in_progress", "response": response }),
            ),
        ]
    }

    fn inline_content_delta(&mut self, delta: &str) -> Vec<Bytes> {
        let (thinking, text) = self.inline_think.push(delta);
        self.emit_inline_think_parts(thinking, text)
    }

    fn flush_inline_think(&mut self) -> Vec<Bytes> {
        let (thinking, text) = self.inline_think.flush();
        self.emit_inline_think_parts(thinking, text)
    }

    fn emit_inline_think_parts(
        &mut self,
        thinking: Option<String>,
        text: Option<String>,
    ) -> Vec<Bytes> {
        let mut events = Vec::new();
        if let Some(thinking) = thinking {
            events.extend(self.reasoning_delta(&thinking));
        }
        if let Some(text) = text {
            events.extend(self.text_delta(&text));
        }
        events
    }

    fn reasoning_start(&mut self) -> Vec<Bytes> {
        if self.reasoning.is_some() {
            return Vec::new();
        }
        let index = self.next_index;
        self.next_index = self.next_index.saturating_add(1);
        let id = gen_id("rs");
        self.reasoning = Some(TextTrack {
            id: id.clone(),
            index,
            text: String::new(),
        });
        vec![
            sse_event(
                "response.output_item.added",
                json!({
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": { "id": id, "type": "reasoning", "summary": [] }
                }),
            ),
            sse_event(
                "response.reasoning_summary_part.added",
                json!({
                    "type": "response.reasoning_summary_part.added",
                    "item_id": id,
                    "output_index": index,
                    "summary_index": 0,
                    "part": { "type": "summary_text", "text": "" }
                }),
            ),
        ]
    }

    fn reasoning_done(&mut self) -> Vec<Bytes> {
        let Some(track) = self.reasoning.take() else {
            return Vec::new();
        };
        self.reasoning_pending.clear();
        vec![
            sse_event(
                "response.reasoning_summary_text.done",
                json!({
                    "type": "response.reasoning_summary_text.done",
                    "item_id": track.id,
                    "output_index": track.index,
                    "summary_index": 0,
                    "text": track.text
                }),
            ),
            sse_event(
                "response.reasoning_summary_part.done",
                json!({
                    "type": "response.reasoning_summary_part.done",
                    "item_id": track.id,
                    "output_index": track.index,
                    "summary_index": 0,
                    "part": { "type": "summary_text", "text": track.text }
                }),
            ),
            sse_event("response.output_item.done", {
                let item = json!({
                    "id": track.id,
                    "type": "reasoning",
                    "summary": [{ "type": "summary_text", "text": track.text }]
                });
                self.output.push(item.clone());
                json!({
                    "type": "response.output_item.done",
                    "output_index": track.index,
                    "item": item
                })
            }),
        ]
    }

    fn reasoning_delta(&mut self, delta: &str) -> Vec<Bytes> {
        let mut events = self.reasoning_start();
        let track = self
            .reasoning
            .as_mut()
            .expect("reasoning_start must create a track");
        track.text.push_str(delta);
        events.push(sse_event(
            "response.reasoning_summary_text.delta",
            json!({
                "type": "response.reasoning_summary_text.delta",
                "item_id": track.id,
                "output_index": track.index,
                "summary_index": 0,
                "delta": delta
            }),
        ));
        events
    }

    fn reasoning_text_fragment(&mut self, delta: &str) -> Vec<Bytes> {
        let mut events = Vec::new();
        if self.reasoning.is_none() {
            self.reasoning_pending.push_str(delta);
            if self.reasoning_pending.trim().is_empty() {
                return events;
            }
            let pending = std::mem::take(&mut self.reasoning_pending);
            events.extend(self.reasoning_delta(&pending));
        } else {
            events.extend(self.reasoning_delta(delta));
        }
        events
    }

    fn reasoning_fragment(&self, delta: &Value) -> Option<String> {
        if self.reasoning.is_some() {
            return raw_reasoning_text(delta).or_else(|| raw_reasoning_whitespace(delta));
        }
        raw_reasoning_text(delta).or_else(|| raw_reasoning_whitespace(delta))
    }

    fn text_delta(&mut self, delta: &str) -> Vec<Bytes> {
        let mut events = Vec::new();
        if self.text.is_none() {
            let index = self.next_index;
            self.next_index += 1;
            let id = gen_id("msg");
            self.text = Some(TextTrack {
                id: id.clone(),
                index,
                text: String::new(),
            });
            events.push(sse_event(
                "response.output_item.added",
                json!({
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": {
                        "id": id,
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": []
                    }
                }),
            ));
            events.push(sse_event(
                "response.content_part.added",
                json!({
                    "type": "response.content_part.added",
                    "item_id": id,
                    "output_index": index,
                    "content_index": 0,
                    "part": { "type": "output_text", "text": "", "annotations": [] }
                }),
            ));
        }
        let track = self.text.as_mut().unwrap();
        track.text.push_str(delta);
        events.push(sse_event(
            "response.output_text.delta",
            json!({
                "type": "response.output_text.delta",
                "item_id": track.id,
                "output_index": track.index,
                "content_index": 0,
                "delta": delta
            }),
        ));
        events
    }

    fn resolve_tool_key(&self, call: &Value) -> usize {
        let call_id = call
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty());
        if let Some(call_id) = call_id {
            if let Some((key, _)) = self
                .calls
                .iter()
                .find(|(_, state)| state.call_id == call_id)
            {
                return *key;
            }
        }
        if let Some(index) = call.get("index").and_then(Value::as_u64) {
            return usize::try_from(index).unwrap_or(usize::MAX - 1);
        }

        let last_key = self.calls.keys().next_back().copied();
        if call_id.is_none() {
            return last_key.unwrap_or(0);
        }
        match last_key {
            Some(key) => key.checked_add(1).unwrap_or(key),
            None => 0,
        }
    }

    fn finish_calls(
        &mut self,
        calls: impl IntoIterator<Item = CallTrack>,
        incomplete_reason: Option<&str>,
    ) -> Vec<Bytes> {
        let mut events = Vec::new();
        for call in calls {
            if !call.call_id.is_empty() && !self.completed_call_ids.insert(call.call_id.clone()) {
                continue;
            }
            if call.custom {
                let input = custom_tool_input(&call.arguments);
                if !input.is_empty() {
                    events.push(sse_event(
                        "response.custom_tool_call_input.delta",
                        json!({
                            "type": "response.custom_tool_call_input.delta",
                            "item_id": call.item_id,
                            "output_index": call.output_index,
                            "delta": input
                        }),
                    ));
                }
                events.push(sse_event(
                    "response.custom_tool_call_input.done",
                    json!({
                        "type": "response.custom_tool_call_input.done",
                        "item_id": call.item_id,
                        "output_index": call.output_index,
                        "input": input
                    }),
                ));
                events.push(sse_event("response.output_item.done", {
                    let item = json!({
                        "id": call.item_id,
                        "type": "custom_tool_call",
                        "status": "completed",
                        "call_id": call.call_id,
                        "name": call.display_name,
                        "input": input
                    });
                    self.output.push(item.clone());
                    json!({
                        "type": "response.output_item.done",
                        "output_index": call.output_index,
                        "item": item
                    })
                }));
            } else if call.tool_search {
                let (arguments, repaired) = normalize_arguments(&call.arguments);
                let arguments = if incomplete_reason.is_some() && repaired {
                    call.arguments.clone()
                } else {
                    arguments
                };
                let arguments = serde_json::from_str::<Value>(&arguments)
                    .ok()
                    .filter(Value::is_object)
                    .unwrap_or_else(|| json!({ "query": arguments }));
                let item = json!({
                    "id": call.item_id,
                    "type": "tool_search_call",
                    "status": "completed",
                    "call_id": call.call_id,
                    "execution": "client",
                    "arguments": arguments
                });
                events.push(sse_event("response.output_item.done", {
                    self.output.push(item.clone());
                    json!({
                        "type": "response.output_item.done",
                        "output_index": call.output_index,
                        "item": item
                    })
                }));
            } else {
                let arguments = if call.arguments.trim().is_empty() {
                    "{}".to_string()
                } else {
                    call.arguments.clone()
                };
                if call.emitted_arguments_len == 0 && call.arguments.trim().is_empty() {
                    events.push(sse_event(
                        "response.function_call_arguments.delta",
                        json!({
                            "type": "response.function_call_arguments.delta",
                            "item_id": call.item_id,
                            "output_index": call.output_index,
                            "delta": "{}"
                        }),
                    ));
                } else if call.emitted_arguments_len < call.arguments.len() {
                    events.push(sse_event(
                        "response.function_call_arguments.delta",
                        json!({
                            "type": "response.function_call_arguments.delta",
                            "item_id": call.item_id,
                            "output_index": call.output_index,
                            "delta": &call.arguments[call.emitted_arguments_len..]
                        }),
                    ));
                }
                events.push(sse_event(
                    "response.function_call_arguments.done",
                    json!({
                        "type": "response.function_call_arguments.done",
                        "item_id": call.item_id,
                        "output_index": call.output_index,
                        "arguments": arguments
                    }),
                ));
                let mut item = json!({
                    "id": call.item_id,
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call.call_id,
                    "name": call.display_name,
                    "arguments": arguments
                });
                if let Some(namespace) = call.namespace {
                    item["namespace"] = json!(namespace);
                }
                if let Some(extra_content) = call.extra_content {
                    item["extra_content"] = extra_content;
                }
                events.push(sse_event(
                    "response.output_item.done",
                    {
                        self.output.push(item.clone());
                        json!({ "type": "response.output_item.done", "output_index": call.output_index, "item": item })
                    },
                ));
            }
        }
        events
    }

    fn finish_text(&mut self) -> Vec<Bytes> {
        let Some(track) = self.text.take() else {
            return Vec::new();
        };
        vec![
            sse_event(
                "response.output_text.done",
                json!({
                    "type": "response.output_text.done",
                    "item_id": track.id,
                    "output_index": track.index,
                    "content_index": 0,
                    "text": track.text,
                    "annotations": []
                }),
            ),
            sse_event(
                "response.content_part.done",
                json!({
                    "type": "response.content_part.done",
                    "item_id": track.id,
                    "output_index": track.index,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": track.text,
                        "annotations": []
                    }
                }),
            ),
            sse_event("response.output_item.done", {
                let item = json!({
                    "id": track.id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{
                        "type": "output_text",
                        "text": track.text,
                        "annotations": []
                    }]
                });
                self.output.push(item.clone());
                json!({
                    "type": "response.output_item.done",
                    "output_index": track.index,
                    "item": item
                })
            }),
        ]
    }

    fn finish_added_calls(&mut self) -> Vec<Bytes> {
        let keys = self
            .calls
            .iter()
            .filter_map(|(key, call)| {
                (call.added
                    && !call.display_name.is_empty()
                    && (call.custom || !normalize_arguments(&call.arguments).1))
                    .then_some(*key)
            })
            .collect::<Vec<_>>();
        let mut events = Vec::new();
        for key in keys {
            let Some(call) = self.calls.remove(&key) else {
                continue;
            };
            events.extend(self.finish_calls(std::iter::once(call), None));
        }
        events
    }

    fn flush_ready_calls(&mut self) -> Vec<Bytes> {
        let mut events = Vec::new();
        loop {
            let key = self.next_call_key_to_add;
            let Some(state) = self.calls.get(&key) else {
                break;
            };
            if state.added {
                self.next_call_key_to_add = self.next_call_key_to_add.saturating_add(1);
                continue;
            }
            if state.call_id.is_empty() || state.display_name.is_empty() {
                break;
            }
            events.extend(self.add_call_events(key));
            self.next_call_key_to_add = self.next_call_key_to_add.saturating_add(1);
        }
        events
    }

    fn add_call_events(&mut self, key: usize) -> Vec<Bytes> {
        let Some(state) = self.calls.get(&key) else {
            return Vec::new();
        };
        if state.added || state.call_id.is_empty() || state.display_name.is_empty() {
            return Vec::new();
        }

        let raw_name = state.display_name.clone();
        let call_id = state.call_id.clone();
        let arguments = state.arguments.clone();
        let extra_content = state.extra_content.clone();
        let (display_name, namespace) = self.shape.resolve(&raw_name);
        let custom = self.shape.is_custom(&display_name);
        let tool_search = self.shape.is_tool_search(&display_name);
        let output_index = self.next_index;
        self.next_index = self.next_index.saturating_add(1);
        let item_id = gen_id(if custom {
            "ctc"
        } else if tool_search {
            "tsc"
        } else {
            "fc"
        });

        let Some(state) = self.calls.get_mut(&key) else {
            return Vec::new();
        };
        state.added = true;
        state.display_name = display_name.clone();
        state.namespace = namespace.clone();
        state.custom = custom;
        state.tool_search = tool_search;
        state.output_index = output_index;
        state.item_id = item_id.clone();

        let item = if custom {
            json!({
                "id": item_id,
                "type": "custom_tool_call",
                "status": "in_progress",
                "call_id": call_id,
                "name": display_name,
                "input": ""
            })
        } else if tool_search {
            json!({
                "id": item_id,
                "type": "tool_search_call",
                "status": "in_progress",
                "call_id": call_id,
                "execution": "client",
                "arguments": {}
            })
        } else {
            let mut item = json!({
                "id": item_id,
                "type": "function_call",
                "status": "in_progress",
                "call_id": call_id,
                "name": display_name,
                "arguments": ""
            });
            if let Some(namespace) = namespace {
                item["namespace"] = json!(namespace);
            }
            if let Some(extra_content) = extra_content {
                item["extra_content"] = extra_content;
            }
            item
        };

        let mut events = vec![sse_event(
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": item
            }),
        )];
        if !custom && !arguments.trim().is_empty() {
            events.push(sse_event(
                "response.function_call_arguments.delta",
                json!({
                    "type": "response.function_call_arguments.delta",
                    "item_id": item_id,
                    "output_index": output_index,
                    "delta": arguments
                }),
            ));
            state.emitted_arguments_len = state.arguments.len();
        }
        events
    }

    fn tool_delta(&mut self, call: &Value) -> Vec<Bytes> {
        let tool_index = self.resolve_tool_key(call);
        let mut events = Vec::new();
        let was_added = self.calls.get(&tool_index).is_some_and(|track| track.added);
        let state = self.calls.entry(tool_index).or_insert_with(|| CallTrack {
            item_id: String::new(),
            call_id: String::new(),
            display_name: String::new(),
            namespace: None,
            custom: false,
            tool_search: false,
            arguments: String::new(),
            emitted_arguments_len: 0,
            extra_content: None,
            output_index: 0,
            added: false,
        });
        if !state.added {
            if let Some(call_id) = call
                .get("id")
                .and_then(Value::as_str)
                .filter(|call_id| !call_id.is_empty())
            {
                state.call_id = call_id.to_string();
            }
            if let Some(name) = call
                .pointer("/function/name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
            {
                state.display_name = name.to_string();
            }
        }
        if let Some(extra) = call
            .get("extra_content")
            .filter(|value| value.as_object().is_some_and(|extra| !extra.is_empty()))
        {
            state.extra_content = Some(extra.clone());
        }
        let arguments = call
            .pointer("/function/arguments")
            .and_then(Value::as_str)
            .unwrap_or("");
        state.arguments.push_str(arguments);
        let item_id = state.item_id.clone();
        let output_index = state.output_index;
        let custom = state.custom;

        events.extend(self.flush_ready_calls());
        if was_added && !custom {
            let state = self
                .calls
                .get_mut(&tool_index)
                .expect("added tool call state must remain available");
            let pending = &state.arguments[state.emitted_arguments_len..];
            if !pending.is_empty() {
                events.push(sse_event(
                    "response.function_call_arguments.delta",
                    json!({
                        "type": "response.function_call_arguments.delta",
                        "item_id": item_id,
                        "output_index": output_index,
                        "delta": pending
                    }),
                ));
                state.emitted_arguments_len = state.arguments.len();
            }
        }
        events
    }
}

/// Extracts reasoning text from a Chat delta or message.
///
/// Upstreams disagree on the field: `reasoning_content` (DeepSeek, Kimi),
/// `reasoning` (OpenRouter) or `reasoning_details[].text` (aggregated
/// providers). Both the streaming translator and the non-streaming aggregator
/// use this so the two paths cannot drift apart.
pub fn reasoning_text(delta: &Value) -> Option<String> {
    delta
        .get("reasoning_content")
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .map(str::to_string)
        .or_else(|| {
            delta
                .get("reasoning")
                .and_then(Value::as_str)
                .filter(|text| !text.trim().is_empty())
                .map(str::to_string)
        })
        .or_else(|| {
            let reasoning = delta.get("reasoning")?;
            for key in ["content", "text", "summary"] {
                if let Some(text) = reasoning
                    .get(key)
                    .and_then(Value::as_str)
                    .filter(|text| !text.trim().is_empty())
                {
                    return Some(text.to_string());
                }
            }
            None
        })
        .or_else(|| {
            let details = delta.get("reasoning_details")?;
            let parts = match details {
                Value::Array(parts) => parts.iter().collect::<Vec<_>>(),
                other => vec![other],
            };
            let text = parts
                .into_iter()
                .filter_map(reasoning_detail_text)
                .filter(|text| !text.trim().is_empty())
                .collect::<Vec<_>>()
                .join("\n\n");
            (!text.trim().is_empty()).then_some(text)
        })
}

fn raw_reasoning_text(delta: &Value) -> Option<String> {
    reasoning_text(delta)
}

fn raw_reasoning_whitespace(delta: &Value) -> Option<String> {
    delta
        .get("reasoning_content")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            delta
                .get("reasoning")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
}

fn reasoning_detail_text(detail: &Value) -> Option<String> {
    for key in ["text", "content", "summary"] {
        if let Some(text) = detail
            .get(key)
            .and_then(Value::as_str)
            .filter(|text| !text.trim().is_empty())
        {
            return Some(text.to_string());
        }
    }
    if let Some(parts) = detail.get("parts").and_then(Value::as_array) {
        let text = parts
            .iter()
            .filter_map(reasoning_detail_text)
            .collect::<Vec<_>>()
            .join("\n\n");
        return (!text.trim().is_empty()).then_some(text);
    }
    None
}

pub(super) fn sse_event(name: &str, data: Value) -> Bytes {
    let mut event = Vec::with_capacity(96);
    write!(event, "event: {name}\ndata: ").expect("writing to Vec cannot fail");
    serde_json::to_writer(&mut event, &data).expect("SSE event data is serializable");
    event.extend_from_slice(b"\n\n");
    Bytes::from(event)
}

pub(super) fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

pub(super) fn gen_id(prefix: &str) -> String {
    static SEQUENCE: AtomicU64 = AtomicU64::new(1);
    format!(
        "{prefix}_{}_{}",
        unix_now(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    )
}
