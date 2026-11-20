use super::request::dedupe_function_call_arguments;
use super::request::{COMPACTION_PLACEHOLDER, COMPACTION_PROMPT, HANDLED_INPUT_TYPES};
use super::*;
use bytes::Bytes;
use serde_json::json;

fn test_cleaner(text: &str) -> String {
    text.replace("secret", "redacted")
}

#[test]
fn non_stream_cleaner_changes_only_text_outputs() {
    let chat = json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "secret text",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": { "name": "secret_tool", "arguments": "{\"value\":\"secret\"}" }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    let response =
        chat_to_responses_with_cleaner(&chat, "model", &ToolShape::default(), Some(test_cleaner))
            .unwrap();

    assert_eq!(response["output"][0]["content"][0]["text"], "redacted text");
    assert_eq!(response["output"][1]["name"], "secret_tool");
    assert_eq!(response["output"][1]["arguments"], "{\"value\":\"secret\"}");
}

#[test]
fn stream_cleaner_changes_only_text_deltas() {
    let mut stream = ResponsesStream::with_shape_and_cleaner(
        "resp_clean".to_string(),
        "model".to_string(),
        ToolShape::default(),
        Some(test_cleaner),
    );
    let events = stream.push(&json!({
        "choices": [{
            "delta": {
                "reasoning_content": "secret reasoning",
                "content": "secret text"
            },
            "finish_reason": "stop"
        }]
    }));
    let joined = events
        .iter()
        .map(|event| String::from_utf8_lossy(event))
        .collect::<String>();

    assert!(joined.contains("secret reasoning"));
    assert!(joined.contains("redacted text"));
    assert!(!joined.contains("secret text"));
}

#[test]
fn translated_chat_never_contains_developer_roles() {
    let request = json!({
        "model": "model",
        "instructions": "root",
        "input": [{
            "type": "message",
            "role": "developer",
            "content": "rules"
        }, {
            "type": "message",
            "role": "user",
            "content": "hello"
        }]
    });
    let chat = responses_to_chat(&request, true).unwrap();
    let messages = chat["messages"].as_array().unwrap();

    assert!(messages.iter().all(|message| {
        !message["role"]
            .as_str()
            .is_some_and(|role| role.eq_ignore_ascii_case("developer"))
    }));
    assert_eq!(
        messages
            .iter()
            .filter(|message| message["role"] == "system")
            .count(),
        1
    );
    assert_eq!(messages[0]["content"], "root\n\nrules");
}

#[test]
fn prepared_responses_returns_one_consistent_tool_shape_and_diagnostics() {
    let request = json!({
        "model": "model",
        "input": [{
            "type": "message",
            "role": "developer",
            "content": "rules"
        }],
        "tools": [{
            "type": "namespace",
            "name": "ns",
            "tools": [{ "type": "function", "name": "run" }]
        }, {
            "type": "tool_search"
        }]
    });
    let prepared = prepare_responses(&request, true, TranslationOptions::default()).unwrap();

    assert_eq!(prepared.chat["stream"], true);
    assert_eq!(prepared.chat["tools"][0]["function"]["name"], "ns__run");
    assert_eq!(
        prepared.chat["tools"][1]["function"]["name"],
        "openai_compat_tool_search"
    );
    assert!(prepared.shape().resolve("ns__run").0 == "run");
    assert!(prepared.unsupported().is_empty());
    assert!(prepared.ignored().is_empty());
}

fn terminal_response(events: &[Bytes], event_name: &str) -> Value {
    for event in events {
        let event = std::str::from_utf8(event).unwrap();
        if event.starts_with(&format!("event: {event_name}\n")) {
            let data = event
                .lines()
                .find_map(|line| line.strip_prefix("data: "))
                .unwrap();
            return serde_json::from_str(data).unwrap();
        }
    }
    panic!("missing terminal event {event_name}");
}

fn named_events(events: &[Bytes], event_name: &str) -> Vec<Value> {
    events
        .iter()
        .filter_map(|event| {
            let event = std::str::from_utf8(event).ok()?;
            if !event.starts_with(&format!("event: {event_name}\n")) {
                return None;
            }
            let data = event.lines().find_map(|line| line.strip_prefix("data: "))?;
            serde_json::from_str(data).ok()
        })
        .collect()
}

fn event_names(events: &[Bytes]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| {
            std::str::from_utf8(event)
                .ok()?
                .lines()
                .find_map(|line| line.strip_prefix("event: ").map(str::to_string))
        })
        .collect()
}

#[test]
fn translates_instructions_messages_images_and_tools() {
    let request = json!({
        "model": "deepseek-v4.1-flash",
        "instructions": "Be concise",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                { "type": "input_text", "text": "look" },
                { "type": "input_image", "image_url": "https://example.test/a.png", "detail": "high" }
            ]
        }],
        "tools": [{
            "type": "namespace",
            "name": "codexui",
            "tools": [{
                "type": "function",
                "name": "get_usage",
                "description": "usage",
                "parameters": { "type": "object", "properties": {} }
            }]
        }, {
            "type": "custom",
            "name": "apply_patch",
            "description": "FREEFORM patch tool"
        }],
        "max_output_tokens": 123,
        "stream": true
    });

    let chat = responses_to_chat(&request, true).unwrap();
    assert_eq!(chat["messages"][0]["role"], "system");
    assert_eq!(chat["messages"][0]["content"], "Be concise");
    assert_eq!(chat["messages"][1]["content"][1]["type"], "image_url");
    assert_eq!(
        chat["messages"][1]["content"][1]["image_url"]["detail"],
        "high"
    );
    assert_eq!(chat["tools"][0]["function"]["name"], "codexui__get_usage");
    assert_eq!(chat["tools"][1]["function"]["name"], "apply_patch");
    assert_eq!(chat["max_tokens"], 123);
    assert!(chat.get("max_completion_tokens").is_none());
    assert_eq!(chat["stream"], true);
}

#[test]
fn canonicalizes_valid_function_arguments_in_history() {
    let request = json!({
        "model": "model",
        "input": [{
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": "{ \"b\": { \"z\": 2, \"a\": 1 }, \"a\": [3, { \"y\": 2, \"x\": 1 }] }"
        }]
    });

    let chat = responses_to_chat(&request, false).unwrap();

    assert_eq!(
        chat["messages"][0]["tool_calls"][0]["function"]["arguments"],
        "{\"a\":[3,{\"x\":1,\"y\":2}],\"b\":{\"a\":1,\"z\":2}}"
    );
}

#[test]
fn repairs_duplicate_function_argument_keys_in_history() {
    let request = json!({
        "model": "model",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_dup",
                "name": "lookup",
                "arguments": "{\"time\":null,\"meta\":{\"k\":1,\"k\":2},\"items\":[{\"q\":\"a\",\"q\":\"b\"}],\"time\":\"later\"}"
            },
            {
                "type": "function_call_output",
                "call_id": "call_dup",
                "output": "ok"
            }
        ]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let arguments = chat["messages"][0]["tool_calls"][0]["function"]["arguments"]
        .as_str()
        .unwrap();
    let parsed: Value = serde_json::from_str(arguments).unwrap();

    assert_eq!(parsed["time"], "later");
    assert_eq!(parsed["meta"]["k"], 2);
    assert_eq!(parsed["items"][0]["q"], "b");
}

#[test]
fn preserves_custom_input_when_well_formed_function_arguments_are_canonicalized() {
    let raw = "{ \"command\": [\"ls\", \"-la\"], \"workdir\": \"/tmp\" }";
    let patch = "*** Begin Patch\n*** Add File: note.txt\n+hello\n*** End Patch";
    let request = json!({
        "model": "model",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_raw",
                "name": "exec_command",
                "arguments": raw
            },
            {
                "type": "custom_tool_call",
                "call_id": "call_patch",
                "name": "apply_patch",
                "input": patch
            }
        ]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let messages = chat["messages"].as_array().unwrap();
    let calls = messages
        .iter()
        .flat_map(|message| {
            message
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .collect::<Vec<_>>();

    assert_eq!(
        calls[0]["function"]["arguments"],
        "{\"command\":[\"ls\",\"-la\"],\"workdir\":\"/tmp\"}"
    );
    let custom: Value =
        serde_json::from_str(calls[1]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(custom["input"], patch);
}

#[test]
fn duplicate_argument_repair_is_idempotent() {
    let mut messages = vec![json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{
            "id": "call_dup",
            "type": "function",
            "function": {
                "name": "lookup",
                "arguments": "{\"b\":2,\"a\":1,\"b\":3}"
            }
        }]
    })];

    assert_eq!(dedupe_function_call_arguments(&mut messages), 1);
    let once = messages.clone();
    assert_eq!(dedupe_function_call_arguments(&mut messages), 0);
    assert_eq!(messages, once);
}

#[test]
fn preserves_custom_freeform_input_exactly() {
    let request = json!({
        "model": "model",
        "input": [{
            "type": "custom_tool_call",
            "call_id": "call_patch",
            "name": "apply_patch",
            "input": "*** Begin Patch\n*** End Patch"
        }]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let arguments = chat["messages"][0]["tool_calls"][0]["function"]["arguments"]
        .as_str()
        .unwrap();
    let parsed: Value = serde_json::from_str(arguments).unwrap();

    assert_eq!(parsed["input"], "*** Begin Patch\n*** End Patch");
}

#[test]
fn degrades_file_and_audio_parts_when_model_capabilities_are_absent() {
    let request = json!({
        "model": "model",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                { "type": "input_text", "text": "inspect" },
                {
                    "type": "input_file",
                    "filename": "report.pdf",
                    "file_data": "data:application/pdf;base64,AA=="
                },
                {
                    "type": "input_audio",
                    "input_audio": { "data": "AAAA", "format": "wav" }
                }
            ]
        }]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let content = chat["messages"][0]["content"].as_str().unwrap();
    assert!(content.contains("file attachment omitted"));
    assert!(content.contains("report.pdf"));
    assert!(content.contains("audio attachment omitted"));
    let details = ignored_feature_details_with_options(&request, TranslationOptions::default());
    assert!(details.contains(&"content.part:input_file_unsupported".to_string()));
    assert!(details.contains(&"content.part:input_audio_unsupported".to_string()));
}

#[test]
fn maps_supported_file_and_audio_parts() {
    let request = json!({
        "model": "model",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "filename": "report.pdf",
                    "file_data": "data:application/pdf;base64,AA=="
                },
                {
                    "type": "input_audio",
                    "input_audio": { "data": "AAAA", "format": "wav" }
                }
            ]
        }]
    });

    let chat = responses_to_chat_with_options(&request, false, TranslationOptions::new(true, true))
        .unwrap();
    let content = chat["messages"][0]["content"].as_array().unwrap();
    assert_eq!(content[0]["type"], "file");
    assert_eq!(content[0]["file"]["filename"], "report.pdf");
    assert_eq!(
        content[0]["file"]["file_data"],
        "data:application/pdf;base64,AA=="
    );
    assert_eq!(content[1]["type"], "input_audio");
    assert_eq!(content[1]["input_audio"]["format"], "wav");
}

#[test]
fn maps_top_level_input_file_item() {
    let request = json!({
        "model": "model",
        "input": [{
            "type": "input_file",
            "filename": "report.pdf",
            "file_data": "data:application/pdf;base64,AA=="
        }]
    });

    let chat =
        responses_to_chat_with_options(&request, false, TranslationOptions::new(true, false))
            .unwrap();

    assert_eq!(chat["messages"][0]["role"], "user");
    assert_eq!(chat["messages"][0]["content"][0]["type"], "file");
}

#[test]
fn normalizes_base_resp_and_context_length_errors() {
    let base_resp = normalize_chat_error(Some(&json!({
        "base_resp": {
            "status_code": 2013,
            "status_msg": "invalid params"
        }
    })));
    assert_eq!(base_resp.message, "invalid params");
    assert_eq!(base_resp.code.as_deref(), Some("2013"));

    let context = normalize_chat_error(Some(&json!({
        "detail": "input token count exceeds the maximum number of tokens allowed"
    })));
    assert_eq!(context.code.as_deref(), Some("context_length_exceeded"));

    let standard = normalize_chat_error(Some(&json!({
        "error": {
            "message": "bad input",
            "type": "invalid_request_error",
            "code": "E_BAD",
            "param": "input"
        }
    })));
    assert_eq!(standard.error_type, "invalid_request_error");
    assert_eq!(standard.code.as_deref(), Some("E_BAD"));
    assert_eq!(standard.param.as_deref(), Some("input"));

    let plain = normalize_chat_error_text("plain upstream failure");
    assert_eq!(plain.message, "plain upstream failure");

    let missing = normalize_chat_error(None);
    assert!(missing.message.contains("empty error response"));
}

#[test]
fn stream_error_uses_normalized_upstream_code_and_message() {
    let mut stream = ResponsesStream::new("resp_error".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "error": {
            "base_resp": {
                "status_code": 2013,
                "status_msg": "invalid params"
            }
        }
    }));
    events.extend(stream.finish());

    let failed = terminal_response(&events, "response.failed");
    assert_eq!(failed["response"]["error"]["code"], "2013");
    assert_eq!(failed["response"]["error"]["message"], "invalid params");
}

#[test]
fn restores_namespace_and_custom_tools_in_response() {
    let request = json!({
        "tools": [{
            "type": "namespace",
            "name": "codexui",
            "tools": [{ "type": "function", "name": "get_usage" }]
        }, {
            "type": "custom",
            "name": "apply_patch"
        }]
    });
    let shape = ToolShape::from_request(&request);
    let chat = json!({
        "id": "chatcmpl-1",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "done",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "codexui__get_usage",
                        "arguments": "{}"
                    }
                }, {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "apply_patch",
                        "arguments": "{\"input\":\"*** Begin Patch\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    });

    let response = chat_to_responses(&chat, "deepseek-v4.1-flash", &shape).unwrap();
    assert_eq!(response["object"], "response");
    assert_eq!(response["output"][0]["content"][0]["text"], "done");
    assert_eq!(response["output"][1]["type"], "function_call");
    assert_eq!(response["output"][1]["name"], "get_usage");
    assert_eq!(response["output"][1]["namespace"], "codexui");
    assert_eq!(response["output"][2]["type"], "custom_tool_call");
    assert_eq!(response["output"][2]["input"], "*** Begin Patch");
    assert_eq!(response["usage"]["input_tokens"], 10);
    assert_eq!(response["usage"]["output_tokens"], 5);
}

#[test]
fn non_stream_function_arguments_preserve_raw_bytes() {
    let raw_arguments = "{ \"b\": 2,\n  \"a\": \"x\" }";
    let chat = json!({
        "id": "chatcmpl-raw",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_raw",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": raw_arguments
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });

    let response = chat_to_responses(&chat, "model", &ToolShape::default()).unwrap();

    assert_eq!(response["output"][0]["arguments"], raw_arguments);
}

#[test]
fn non_stream_maps_content_parts_refusal_and_upstream_model() {
    let chat = json!({
        "id": "chatcmpl-parts",
        "created": 42,
        "model": "upstream-model",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": [
                    { "type": "text", "text": "hello" },
                    { "type": "refusal", "refusal": "blocked" }
                ]
            },
            "finish_reason": "stop"
        }]
    });

    let response = chat_to_responses(&chat, "requested-model", &ToolShape::default()).unwrap();

    assert_eq!(response["id"], "resp_parts");
    assert_eq!(response["created_at"], 42);
    assert_eq!(response["model"], "upstream-model");
    assert_eq!(
        response["output"][0]["content"][0],
        json!({
            "type": "output_text",
            "text": "hello",
            "annotations": []
        })
    );
    assert_eq!(
        response["output"][0]["content"][1],
        json!({
            "type": "refusal",
            "refusal": "blocked"
        })
    );
}

#[test]
fn non_stream_maps_legacy_function_call() {
    let chat = json!({
        "id": "chatcmpl-legacy",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "function_call": {
                    "name": "read_file",
                    "arguments": "{\"path\":\"README.md\"}"
                }
            },
            "finish_reason": "function_call"
        }]
    });

    let response = chat_to_responses(&chat, "model", &ToolShape::default()).unwrap();

    assert_eq!(response["output"][0]["type"], "function_call");
    assert_eq!(response["output"][0]["name"], "read_file");
    assert_eq!(
        response["output"][0]["arguments"],
        "{\"path\":\"README.md\"}"
    );
}

#[test]
fn non_stream_legacy_function_call_preserves_raw_arguments() {
    let raw_arguments = "{ \"path\": \"README.md\" }";
    let chat = json!({
        "id": "chatcmpl-legacy-raw",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "function_call": {
                    "name": "read_file",
                    "arguments": raw_arguments
                }
            },
            "finish_reason": "function_call"
        }]
    });

    let response = chat_to_responses(&chat, "model", &ToolShape::default()).unwrap();

    assert_eq!(response["output"][0]["arguments"], raw_arguments);
}

#[test]
fn non_stream_splits_inline_think_from_content() {
    let chat = json!({
        "id": "chatcmpl-think",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "<think>check files</think>\nDone."
            },
            "finish_reason": "stop"
        }]
    });

    let response = chat_to_responses(&chat, "model", &ToolShape::default()).unwrap();

    assert_eq!(response["output"][0]["type"], "reasoning");
    assert_eq!(response["output"][0]["summary"][0]["text"], "check files");
    assert_eq!(response["output"][1]["type"], "message");
    assert_eq!(response["output"][1]["content"][0]["text"], "Done.");
}

#[test]
fn preserves_completion_metadata_and_incomplete_status() {
    let chat = json!({
        "id": "chatcmpl-meta",
        "created": 42,
        "system_fingerprint": "fp_1",
        "choices": [{
            "message": { "role": "assistant", "content": "partial" },
            "finish_reason": "length"
        }]
    });

    let response = chat_to_responses(&chat, "model", &ToolShape::default()).unwrap();

    assert_eq!(response["created_at"], 42);
    assert_eq!(response["status"], "incomplete");
    assert_eq!(
        response["incomplete_details"]["reason"],
        "max_output_tokens"
    );
    assert_eq!(response["system_fingerprint"], "fp_1");
}

#[test]
fn non_stream_missing_finish_with_output_is_incomplete() {
    let chat = json!({
        "id": "chatcmpl-no-finish",
        "choices": [{
            "message": { "role": "assistant", "content": "partial" },
            "finish_reason": null
        }]
    });

    let response = chat_to_responses(&chat, "model", &ToolShape::default()).unwrap();

    assert_eq!(response["status"], "incomplete");
    assert_eq!(response["output"][0]["content"][0]["text"], "partial");
}

#[test]
fn non_stream_missing_finish_without_output_fails() {
    let chat = json!({
        "id": "chatcmpl-empty",
        "choices": [{
            "message": { "role": "assistant", "content": null },
            "finish_reason": null
        }]
    });

    let error = chat_to_responses(&chat, "model", &ToolShape::default()).unwrap_err();

    assert!(error.contains("finish_reason"));
}

#[test]
fn reports_unsupported_responses_features_without_rejecting_them() {
    let request = json!({
        "previous_response_id": "resp_old",
        "text": { "format": { "type": "json_schema" } },
        "input": [{ "type": "mcp_call" }]
    });

    // `text.format` is translated now, so only a format this layer cannot
    // carry (no schema) shows up as an ignored detail.
    assert_eq!(
        unsupported_feature_names(&request),
        ["previous_response_id"]
    );
    assert_eq!(
        ignored_feature_details(&request),
        [
            "input.item_degraded:mcp_call",
            "text.format.type:json_schema_without_schema"
        ]
    );
}

#[test]
fn streams_text_and_terminal_events_in_order() {
    let mut stream = ResponsesStream::new("resp_1".to_string(), "deepseek-v4.1-flash".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": { "content": "hello" },
            "finish_reason": null
        }]
    }));
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": {},
            "finish_reason": "stop"
        }],
        "usage": { "prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5 }
    })));
    events.extend(stream.finish());

    let joined = String::from_utf8(events.concat()).unwrap();
    let names: Vec<&str> = joined
        .lines()
        .filter_map(|line| line.strip_prefix("event: "))
        .collect();
    assert_eq!(
        names,
        vec![
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    assert!(joined.contains("\"input_tokens\":3"));
    assert!(joined.contains("\"output_tokens\":2"));
}

#[test]
fn stream_ignores_all_events_after_completed_terminal() {
    let mut stream = ResponsesStream::new("resp_terminal_lock".to_string(), "model".to_string());
    let first = stream.push(&json!({
        "choices": [{
            "delta": { "content": "first" },
            "finish_reason": "stop"
        }]
    }));
    let mut events = first;
    events.extend(stream.finish());
    let expected = events.clone();

    events.extend(stream.push(&json!({
        "choices": [{
            "delta": { "content": "late", "tool_calls": [{
                "index": 0,
                "id": "call_late",
                "function": { "name": "read_file", "arguments": "{}" }
            }] },
            "finish_reason": "tool_calls"
        }],
        "usage": { "prompt_tokens": 99, "completion_tokens": 99 }
    })));
    events.extend(stream.fail("late_error", "late failure"));
    events.extend(stream.finish());

    assert_eq!(events, expected);
    assert_eq!(named_events(&events, "response.completed").len(), 1);
}

#[test]
fn stream_keeps_only_one_terminal_when_error_arrives_after_output() {
    let mut stream = ResponsesStream::new("resp_error_terminal".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": { "content": "visible" },
            "finish_reason": null
        }]
    }));
    events.extend(stream.fail("upstream_network_error", "connection reset"));
    events.extend(stream.push(&json!({
        "error": { "message": "late timeout", "type": "server_error" }
    })));
    events.extend(stream.finish());

    let names = event_names(&events);
    assert_eq!(
        names
            .iter()
            .filter(|name| matches!(
                name.as_str(),
                "response.completed" | "response.incomplete" | "response.failed"
            ))
            .count(),
        1
    );
    assert_eq!(named_events(&events, "response.failed").len(), 1);
}

#[test]
fn stream_uses_upstream_metadata_and_emits_in_progress() {
    let mut stream =
        ResponsesStream::new("resp_requested".to_string(), "requested-model".to_string());
    let mut events = stream.push(&json!({
        "id": "chatcmpl-upstream",
        "created": 42,
        "model": "upstream-model",
        "choices": [{
            "delta": { "content": "hello" },
            "finish_reason": "stop"
        }]
    }));
    events.extend(stream.finish());

    let created = named_events(&events, "response.created");
    let in_progress = named_events(&events, "response.in_progress");
    let completed = terminal_response(&events, "response.completed");

    assert_eq!(created.len(), 1);
    assert_eq!(created[0]["response"]["id"], "resp_upstream");
    assert_eq!(created[0]["response"]["model"], "upstream-model");
    assert_eq!(created[0]["response"]["created_at"], 42);
    assert_eq!(in_progress.len(), 1);
    assert_eq!(completed["response"]["id"], "resp_upstream");
    assert_eq!(completed["response"]["model"], "upstream-model");
    assert_eq!(completed["response"]["created_at"], 42);
}

#[test]
fn stream_preserves_service_tier_and_system_fingerprint() {
    let mut stream = ResponsesStream::new("resp_meta".to_string(), "requested-model".to_string());
    let mut events = stream.push(&json!({
        "id": "chatcmpl-meta",
        "model": "upstream-model",
        "service_tier": "priority",
        "system_fingerprint": "fp-stream",
        "choices": [{
            "delta": { "content": "hello" },
            "finish_reason": "stop"
        }]
    }));
    events.extend(stream.finish());

    let created = named_events(&events, "response.created");
    let completed = terminal_response(&events, "response.completed");
    assert_eq!(created[0]["response"]["service_tier"], "priority");
    assert_eq!(created[0]["response"]["system_fingerprint"], "fp-stream");
    assert_eq!(completed["response"]["service_tier"], "priority");
    assert_eq!(completed["response"]["system_fingerprint"], "fp-stream");
}

#[test]
fn stream_splits_inline_think_without_leaking_tags() {
    let mut stream = ResponsesStream::new("resp_think".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "content": "<think>check files</think>\nDone."
            },
            "finish_reason": "stop"
        }]
    }));
    events.extend(stream.finish());

    let completed = terminal_response(&events, "response.completed");
    let output = completed["response"]["output"].as_array().unwrap();
    assert_eq!(output[0]["type"], "reasoning");
    assert_eq!(output[0]["summary"][0]["text"], "check files");
    assert_eq!(output[1]["type"], "message");
    assert_eq!(output[1]["content"][0]["text"], "Done.");
    let joined = String::from_utf8(events.concat()).unwrap();
    assert!(!joined.contains("<think>"));
    assert!(!joined.contains("</think>"));
}

#[test]
fn completed_stream_includes_accumulated_output() {
    let mut stream = ResponsesStream::new("resp_output".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "reasoning_content": "think",
                "content": "answer",
                "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "function": { "name": "lookup", "arguments": "{}" }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    }));
    events.extend(stream.finish());

    let terminal = terminal_response(&events, "response.completed");
    let output = terminal["response"]["output"].as_array().unwrap();
    assert_eq!(output.len(), 3);
    assert_eq!(output[0]["type"], "reasoning");
    assert_eq!(output[1]["type"], "message");
    assert_eq!(output[1]["content"][0]["text"], "answer");
    assert_eq!(output[2]["type"], "function_call");
    assert_eq!(output[2]["call_id"], "call_1");
}

#[test]
fn streams_custom_tool_input_as_raw_text_after_complete_arguments() {
    let patch = "*** Begin Patch\n*** Add File: note.txt\n+hello\n*** End Patch";
    let request = json!({
        "tools": [{
            "type": "custom",
            "name": "apply_patch"
        }]
    });
    let mut stream = ResponsesStream::with_shape(
        "resp_patch".to_string(),
        "model".to_string(),
        ToolShape::from_request(&request),
    );
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_patch",
                    "function": {
                        "name": "apply_patch",
                        "arguments": "{\"input\":\"*** Begin Patch\\n*** Add File: note.txt\\n"
                    }
                }]
            },
            "finish_reason": null
        }]
    }));
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {
                        "arguments": "+hello\\n*** End Patch\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })));
    events.extend(stream.finish());

    let deltas = named_events(&events, "response.custom_tool_call_input.delta");
    let done = named_events(&events, "response.custom_tool_call_input.done");
    let items = named_events(&events, "response.output_item.done");

    assert_eq!(deltas.len(), 1);
    assert_eq!(deltas[0]["delta"], patch);
    assert_eq!(done.len(), 1);
    assert_eq!(done[0]["input"], patch);
    assert_eq!(items.len(), 1);
    assert_eq!(items[0]["item"]["type"], "custom_tool_call");
    assert_eq!(items[0]["item"]["input"], patch);
    assert!(
        items[0]["item"]["id"].as_str().unwrap().starts_with("ctc_"),
        "{}",
        items[0]
    );
}

#[test]
fn preserves_apply_patch_line_endings_in_stream_and_history() {
    let patches = [
        "*** Begin Patch\n*** Update File: note.txt\n@@\n-old\n+new\n*** End Patch\n",
        "*** Begin Patch\n*** Update File: tail.txt\n@@\n last\n+end\n*** End of File\n*** End Patch\n",
    ];
    let request = json!({
        "tools": [{
            "type": "custom",
            "name": "apply_patch"
        }]
    });

    for (index, patch) in patches.iter().enumerate() {
        let escaped_patch = serde_json::to_string(patch).unwrap();
        let split = escaped_patch.len() / 2;
        let arguments_left = format!("{{\"input\":{}", &escaped_patch[..split]);
        let arguments_tail = format!("{}}}", &escaped_patch[split..]);
        let mut stream = ResponsesStream::with_shape(
            format!("resp_patch_{index}"),
            "model".to_string(),
            ToolShape::from_request(&request),
        );
        let mut events = stream.push(&json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": format!("call_patch_{index}"),
                        "function": {
                            "name": "apply_patch",
                            "arguments": arguments_left
                        }
                    }]
                },
                "finish_reason": null
            }]
        }));
        events.extend(stream.push(&json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "arguments": arguments_tail
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        })));
        events.extend(stream.finish());

        let delta = named_events(&events, "response.custom_tool_call_input.delta");
        let done = named_events(&events, "response.custom_tool_call_input.done");
        let item = named_events(&events, "response.output_item.done");
        assert_eq!(delta[0]["delta"], *patch);
        assert_eq!(done[0]["input"], *patch);
        assert_eq!(item[0]["item"]["input"], *patch);

        let output =
            "Exit code: 0\nWall time: 0 seconds\nOutput:\nSuccess. Updated the following files:\nM note.txt\n"
                .to_string();
        let history = json!({
            "input": [{
                "type": "custom_tool_call",
                "call_id": format!("call_patch_{index}"),
                "name": "apply_patch",
                "input": patch
            }, {
                "type": "custom_tool_call_output",
                "call_id": format!("call_patch_{index}"),
                "output": output
            }]
        });
        let chat = responses_to_chat(&history, false).unwrap();
        let tool_call = &chat["messages"][0]["tool_calls"][0];
        let function_arguments: Value =
            serde_json::from_str(tool_call["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(function_arguments["input"], *patch);
        assert_eq!(chat["messages"][1]["content"], output.as_str());
    }
}

#[test]
fn maps_empty_custom_tool_arguments_to_empty_input() {
    let request = json!({
        "tools": [{
            "type": "custom",
            "name": "apply_patch"
        }]
    });
    let mut stream = ResponsesStream::with_shape(
        "resp_empty_patch".to_string(),
        "model".to_string(),
        ToolShape::from_request(&request),
    );
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_empty",
                    "function": {
                        "name": "apply_patch",
                        "arguments": "{}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    }));
    events.extend(stream.finish());

    assert!(named_events(&events, "response.custom_tool_call_input.delta").is_empty());
    let done = named_events(&events, "response.custom_tool_call_input.done");
    assert_eq!(done.len(), 1);
    assert_eq!(done[0]["input"], "");
    let items = named_events(&events, "response.output_item.done");
    assert_eq!(items[0]["item"]["input"], "");
}

#[test]
fn waits_for_late_tool_name_before_emitting_custom_item() {
    let patch = "*** Begin Patch\n*** End Patch";
    let request = json!({
        "tools": [{
            "type": "custom",
            "name": "apply_patch"
        }]
    });
    let mut stream = ResponsesStream::with_shape(
        "resp_late_name".to_string(),
        "model".to_string(),
        ToolShape::from_request(&request),
    );
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_late",
                    "function": {
                        "arguments": "{\"input\":\"*** Begin Patch\\n"
                    }
                }]
            },
            "finish_reason": null
        }]
    }));
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {
                        "name": "apply_patch",
                        "arguments": "*** End Patch\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })));
    events.extend(stream.finish());

    let added = named_events(&events, "response.output_item.added");
    assert_eq!(added.len(), 1);
    assert_eq!(added[0]["item"]["type"], "custom_tool_call");
    assert_eq!(added[0]["item"]["name"], "apply_patch");
    assert!(named_events(&events, "response.function_call_arguments.delta").is_empty());
    let done = named_events(&events, "response.custom_tool_call_input.done");
    assert_eq!(done.len(), 1);
    assert_eq!(done[0]["input"], patch);
}

#[test]
fn keeps_parallel_calls_separate_when_chat_index_is_missing() {
    let mut stream = ResponsesStream::new("resp_missing_index".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "id": "call_a",
                    "function": {
                        "name": "read_file",
                        "arguments": "{\"path\":\"a.txt\"}"
                    }
                }]
            },
            "finish_reason": null
        }]
    }));
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "id": "call_b",
                    "function": {
                        "name": "write_file",
                        "arguments": "{\"path\":\"b.txt\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })));
    events.extend(stream.finish());

    let completed = terminal_response(&events, "response.completed");
    let output = completed["response"]["output"].as_array().unwrap();
    assert_eq!(output.len(), 2);
    assert_eq!(output[0]["call_id"], "call_a");
    assert_eq!(output[0]["name"], "read_file");
    assert_eq!(output[1]["call_id"], "call_b");
    assert_eq!(output[1]["name"], "write_file");
}

#[test]
fn keeps_repeated_call_id_without_index_in_one_call() {
    let mut stream = ResponsesStream::new("resp_repeated_id".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "id": "call_a",
                    "function": {
                        "name": "read_file",
                        "arguments": "{\"path\":"
                    }
                }]
            },
            "finish_reason": null
        }]
    }));
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "id": "call_a",
                    "function": {
                        "arguments": "\"a.txt\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })));
    events.extend(stream.finish());

    let completed = terminal_response(&events, "response.completed");
    let output = completed["response"]["output"].as_array().unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0]["call_id"], "call_a");
    assert_eq!(output[0]["arguments"], "{\"path\":\"a.txt\"}");
}

#[test]
fn stream_function_arguments_preserve_raw_bytes_across_all_outputs() {
    let raw_arguments = "{ \"b\": 2,\n  \"a\": \"x\" }";
    let mut stream = ResponsesStream::new("resp_raw_arguments".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_raw",
                    "function": {
                        "name": "lookup",
                        "arguments": "{ \"b\": 2,"
                    }
                }]
            },
            "finish_reason": null
        }]
    }));
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {
                        "arguments": "\n  \"a\": \"x\" }"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })));
    events.extend(stream.finish());

    let deltas = named_events(&events, "response.function_call_arguments.delta")
        .into_iter()
        .filter_map(|event| event["delta"].as_str().map(str::to_string))
        .collect::<String>();
    let done = named_events(&events, "response.function_call_arguments.done");
    let item_done = named_events(&events, "response.output_item.done");
    let completed = terminal_response(&events, "response.completed");

    assert_eq!(deltas, raw_arguments);
    assert_eq!(done[0]["arguments"], raw_arguments);
    assert_eq!(item_done[0]["item"]["arguments"], raw_arguments);
    assert_eq!(
        completed["response"]["output"][0]["arguments"],
        raw_arguments
    );
}

#[test]
fn stream_empty_function_arguments_emit_one_empty_object_delta() {
    for raw_arguments in ["", "   "] {
        let mut stream = ResponsesStream::new(
            format!("resp_empty_arguments_{}", raw_arguments.len()),
            "model".to_string(),
        );
        let mut events = stream.push(&json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_empty",
                        "function": {
                            "name": "lookup",
                            "arguments": raw_arguments
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }));
        events.extend(stream.finish());

        let deltas = named_events(&events, "response.function_call_arguments.delta");
        let done = named_events(&events, "response.function_call_arguments.done");
        let item_done = named_events(&events, "response.output_item.done");
        let completed = terminal_response(&events, "response.completed");

        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0]["delta"], "{}");
        assert_eq!(done[0]["arguments"], "{}");
        assert_eq!(item_done[0]["item"]["arguments"], "{}");
        assert_eq!(completed["response"]["output"][0]["arguments"], "{}");
    }
}

#[test]
fn stream_function_arguments_flush_trailing_whitespace_before_done() {
    let raw_arguments = "{ \"a\": 1 }\n";
    let mut stream = ResponsesStream::new("resp_trailing_space".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_space",
                    "function": {
                        "name": "lookup",
                        "arguments": "{ \"a\": 1 }"
                    }
                }]
            },
            "finish_reason": null
        }]
    }));
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": { "arguments": "\n" }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })));
    events.extend(stream.finish());

    let deltas = named_events(&events, "response.function_call_arguments.delta")
        .into_iter()
        .filter_map(|event| event["delta"].as_str().map(str::to_string))
        .collect::<String>();
    let done = named_events(&events, "response.function_call_arguments.done");
    let completed = terminal_response(&events, "response.completed");
    assert_eq!(deltas, raw_arguments);
    assert_eq!(done[0]["arguments"], raw_arguments);
    assert_eq!(
        completed["response"]["output"][0]["arguments"],
        raw_arguments
    );
}

#[test]
fn stream_function_arguments_emit_whitespace_only_tail_fragments() {
    let raw_arguments = "{ \"a\": 1 }\n \t";
    let mut stream = ResponsesStream::new("resp_whitespace_tail".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_space_tail",
                    "function": {
                        "name": "lookup",
                        "arguments": "{ \"a\": 1 }"
                    }
                }]
            },
            "finish_reason": null
        }]
    }));
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": { "arguments": "\n" }
                }]
            },
            "finish_reason": null
        }]
    })));
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": { "arguments": " \t" }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })));
    events.extend(stream.finish());

    let deltas = named_events(&events, "response.function_call_arguments.delta")
        .into_iter()
        .filter_map(|event| event["delta"].as_str().map(str::to_string))
        .collect::<String>();
    let done = named_events(&events, "response.function_call_arguments.done");
    let item = named_events(&events, "response.output_item.done");
    let completed = terminal_response(&events, "response.completed");
    assert_eq!(deltas, raw_arguments);
    assert_eq!(done[0]["arguments"], raw_arguments);
    assert_eq!(item[0]["item"]["arguments"], raw_arguments);
    assert_eq!(
        completed["response"]["output"][0]["arguments"],
        raw_arguments
    );
}

#[test]
fn stream_function_arguments_emit_each_whitespace_fragment_immediately() {
    let mut stream = ResponsesStream::new("resp_live_whitespace".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_live_space",
                    "function": {
                        "name": "lookup",
                        "arguments": "{ \"a\": 1 }"
                    }
                }]
            },
            "finish_reason": null
        }]
    }));
    let whitespace_events = stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": { "arguments": "\n" }
                }]
            },
            "finish_reason": null
        }]
    }));
    assert_eq!(
        named_events(&events, "response.function_call_arguments.delta").len(),
        1
    );
    assert_eq!(
        named_events(&whitespace_events, "response.function_call_arguments.delta").len(),
        1
    );
    events.extend(whitespace_events);
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": { "arguments": " " }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })));
    events.extend(stream.finish());

    let deltas = named_events(&events, "response.function_call_arguments.delta")
        .into_iter()
        .filter_map(|event| event["delta"].as_str().map(str::to_string))
        .collect::<String>();
    assert_eq!(deltas, "{ \"a\": 1 }\n ");
}

#[test]
fn repeated_tool_headers_do_not_erase_restored_namespace() {
    let request = json!({
        "tools": [{
            "type": "namespace",
            "name": "ns",
            "tools": [{
                "type": "function",
                "name": "exec",
                "parameters": { "type": "object", "properties": {} }
            }]
        }]
    });
    let mut stream = ResponsesStream::with_shape(
        "resp_repeated_header".to_string(),
        "model".to_string(),
        ToolShape::from_request(&request),
    );
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_exec",
                    "function": {
                        "name": "ns__exec",
                        "arguments": "{"
                    }
                }]
            },
            "finish_reason": null
        }]
    }));
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_exec",
                    "function": {
                        "name": "ns__exec",
                        "arguments": "\"cmd\":\"ls\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })));
    events.extend(stream.finish());

    let completed = terminal_response(&events, "response.completed");
    let item = &completed["response"]["output"][0];
    assert_eq!(item["name"], "exec");
    assert_eq!(item["namespace"], "ns");
    assert_eq!(item["arguments"], "{\"cmd\":\"ls\"}");
}

#[test]
fn stream_preserves_tool_extra_content() {
    let extra = json!({
        "google": { "thought_signature": "sig-stream" }
    });
    let mut stream = ResponsesStream::new("resp_extra".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_extra",
                    "extra_content": extra,
                    "function": {
                        "name": "read_file",
                        "arguments": "{}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    }));
    events.extend(stream.finish());

    let completed = terminal_response(&events, "response.completed");
    assert_eq!(completed["response"]["output"][0]["extra_content"], extra);
}

#[test]
fn emits_sparse_tool_index_in_stable_output_order() {
    let mut stream = ResponsesStream::new("resp_sparse".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 2,
                    "id": "call_sparse",
                    "function": {
                        "name": "read_file",
                        "arguments": "{}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    }));
    events.extend(stream.finish());

    let added = named_events(&events, "response.output_item.added");
    assert_eq!(added.len(), 1);
    assert_eq!(added[0]["output_index"], 0);
    let completed = terminal_response(&events, "response.completed");
    assert_eq!(completed["response"]["output"][0]["call_id"], "call_sparse");
}

#[test]
fn parallel_function_arguments_preserve_per_call_byte_parity() {
    let mut stream = ResponsesStream::new("resp_parallel_raw".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_a",
                    "function": {
                        "name": "read_file",
                        "arguments": "{ \"path\": \"a\""
                    }
                }, {
                    "index": 2,
                    "id": "call_b",
                    "function": {
                        "name": "write_file",
                        "arguments": "{ \"path\": \"b\""
                    }
                }]
            },
            "finish_reason": null
        }]
    }));
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": { "arguments": " }" }
                }, {
                    "index": 2,
                    "function": { "arguments": " }" }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })));
    events.extend(stream.finish());

    let done = named_events(&events, "response.function_call_arguments.done");
    let items = named_events(&events, "response.output_item.done");
    let completed = terminal_response(&events, "response.completed");
    for output_index in 0usize..2 {
        let expected = if output_index == 0 {
            "{ \"path\": \"a\" }"
        } else {
            "{ \"path\": \"b\" }"
        };
        let deltas = named_events(&events, "response.function_call_arguments.delta")
            .into_iter()
            .filter(|event| event["output_index"].as_u64() == Some(output_index as u64))
            .filter_map(|event| event["delta"].as_str().map(str::to_string))
            .collect::<String>();
        assert_eq!(deltas, expected);
        assert_eq!(done[output_index]["arguments"], expected);
        assert_eq!(items[output_index]["item"]["arguments"], expected);
        assert_eq!(
            completed["response"]["output"][output_index]["arguments"],
            expected
        );
    }
}

#[test]
fn synthesizes_call_id_when_chat_tool_call_omits_it() {
    let mut stream = ResponsesStream::new("resp_missing_id".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {
                        "name": "read_file",
                        "arguments": "{}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    }));
    events.extend(stream.finish());

    let completed = terminal_response(&events, "response.completed");
    assert_eq!(completed["response"]["output"][0]["call_id"], "call_0");
}

#[test]
fn truncated_tool_arguments_are_incomplete_not_failed() {
    let mut stream = ResponsesStream::new("resp_truncated".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_cut",
                    "function": {
                        "name": "read_file",
                        "arguments": "{\"path\":"
                    }
                }]
            },
            "finish_reason": "length"
        }]
    }));
    events.extend(stream.finish());

    let deltas = named_events(&events, "response.function_call_arguments.delta")
        .into_iter()
        .filter_map(|event| event["delta"].as_str().map(str::to_string))
        .collect::<String>();
    let done = named_events(&events, "response.function_call_arguments.done");
    let incomplete = terminal_response(&events, "response.incomplete");
    assert_eq!(incomplete["response"]["status"], "incomplete");
    assert_eq!(
        incomplete["response"]["output"][0]["arguments"],
        "{\"path\":"
    );
    assert_eq!(deltas, "{\"path\":");
    assert_eq!(done[0]["arguments"], "{\"path\":");
    assert!(named_events(&events, "response.failed").is_empty());
}

#[test]
fn stream_done_without_finish_reason_and_output_is_incomplete() {
    let mut stream = ResponsesStream::new("resp_no_finish".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": { "content": "partial" },
            "finish_reason": null
        }]
    }));
    events.extend(stream.finish());

    let incomplete = terminal_response(&events, "response.incomplete");
    assert_eq!(incomplete["response"]["status"], "incomplete");
    assert_eq!(
        incomplete["response"]["output"][0]["content"][0]["text"],
        "partial"
    );
    assert!(named_events(&events, "response.failed").is_empty());
}

#[test]
fn stream_done_without_finish_reason_or_output_fails() {
    let mut stream = ResponsesStream::new("resp_empty".to_string(), "model".to_string());
    let events = stream.finish();

    let failed = terminal_response(&events, "response.failed");
    assert_eq!(failed["response"]["status"], "failed");
    assert_eq!(failed["response"]["error"]["code"], "incomplete_stream");
    assert!(named_events(&events, "response.completed").is_empty());
}

#[test]
fn stream_content_filter_is_incomplete() {
    let mut stream = ResponsesStream::new("resp_filtered".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": { "content": "partial" },
            "finish_reason": "content_filter"
        }]
    }));
    events.extend(stream.finish());

    let incomplete = terminal_response(&events, "response.incomplete");
    assert_eq!(incomplete["response"]["status"], "incomplete");
    assert_eq!(
        incomplete["response"]["incomplete_details"]["reason"],
        "content_filter"
    );
}

#[test]
fn malformed_historical_tool_call_does_not_block_the_next_turn() {
    let request = json!({
        "model": "model",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_bad",
                "name": "read_file",
                "arguments": "{broken"
            },
            {
                "type": "function_call_output",
                "call_id": "call_bad",
                "output": "parse failed"
            },
            {
                "type": "message",
                "role": "user",
                "content": "continue"
            }
        ]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let messages = chat["messages"].as_array().unwrap();

    assert_eq!(roles(messages), ["assistant", "user"]);
    assert!(messages[0].get("tool_calls").is_none());
    let context = messages[0]["content"].as_str().unwrap();
    assert!(context.contains("Historical tool call with invalid arguments"));
    assert!(context.contains("{broken"));
    assert!(context.contains("parse failed"));
    assert_eq!(messages[1]["content"], "continue");
}

#[test]
fn appends_multiple_reasoning_items_to_the_following_tool_call() {
    let request = json!({
        "model": "model",
        "input": [
            {
                "type": "reasoning",
                "summary": [{ "type": "summary_text", "text": "first" }]
            },
            {
                "type": "reasoning",
                "summary": [{ "type": "summary_text", "text": "second" }]
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": "{}"
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok"
            }
        ]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let messages = chat["messages"].as_array().unwrap();

    assert_eq!(messages[0]["reasoning_content"], "first\n\nsecond");
    assert_eq!(messages[0]["tool_calls"][0]["id"], "call_1");
}

#[test]
fn collapses_system_messages_while_preserving_user_boundaries() {
    let request = json!({
        "model": "model",
        "instructions": "root",
        "input": [
            {
                "type": "message",
                "role": "developer",
                "content": "setup"
            },
            {
                "type": "message",
                "role": "user",
                "content": "one"
            },
            {
                "type": "message",
                "role": "developer",
                "content": "new rules"
            },
            {
                "type": "message",
                "role": "user",
                "content": "two"
            }
        ]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let messages = chat["messages"].as_array().unwrap();

    assert_eq!(roles(messages), ["system", "user", "user", "user"]);
    assert_eq!(messages[0]["content"], "root\n\nsetup");
    assert_eq!(messages[1]["content"], "one");
    assert_eq!(messages[2]["content"], "new rules");
    assert_eq!(messages[3]["content"], "two");
}

#[test]
fn repairs_missing_tool_outputs_in_history() {
    let request = json!({
        "model": "deepseek-v4.1-flash",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": "{\"q\":1}"
            },
            {
                "type": "message",
                "role": "user",
                "content": "continue"
            }
        ],
        "stream": false
    });
    let chat = responses_to_chat(&request, false).unwrap();
    assert_eq!(chat["messages"][0]["role"], "assistant");
    assert_eq!(chat["messages"][1]["role"], "tool");
    assert_eq!(chat["messages"][1]["tool_call_id"], "call_1");
    assert_eq!(chat["messages"][2]["content"], "continue");
}

#[test]
fn preserves_orphan_tool_output_as_marked_user_message() {
    let request = json!({
        "model": "model",
        "input": [
            {
                "type": "function_call_output",
                "id": "fco_1",
                "call_id": null,
                "output": "receipt"
            },
            { "type": "message", "role": "user", "content": "continue" }
        ]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let messages = chat["messages"].as_array().unwrap();

    assert_eq!(roles(messages), ["user", "user"]);
    assert!(messages[0]["content"]
        .as_str()
        .unwrap()
        .contains("[tool_result without a matching tool_call]"));
    assert!(messages[0]["content"].as_str().unwrap().contains("receipt"));
}

#[test]
fn preserves_orphan_tool_output_media_as_user_image() {
    let request = json!({
        "model": "model",
        "input": [{
            "type": "function_call_output",
            "id": "fco_image",
            "call_id": null,
            "output": {
                "content": [{
                    "type": "input_image",
                    "image_url": large_image_data_url()
                }]
            }
        }]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let messages = chat["messages"].as_array().unwrap();

    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
    let parts = messages[0]["content"].as_array().unwrap();
    assert!(parts.iter().any(|part| part["type"] == "image_url"));
    let text = parts
        .iter()
        .filter_map(|part| part["text"].as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!text.contains("base64"), "{text}");
}

#[test]
fn falls_back_to_item_id_for_custom_tool_call_history() {
    let request = json!({
        "model": "model",
        "input": [
            {
                "type": "custom_tool_call",
                "id": "ctc_1",
                "name": "apply_patch",
                "input": "*** Begin Patch\n*** End Patch"
            },
            {
                "type": "custom_tool_call_output",
                "call_id": "ctc_1",
                "output": "ok"
            }
        ]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let messages = chat["messages"].as_array().unwrap();

    assert_eq!(roles(messages), ["assistant", "tool"]);
    assert_eq!(messages[0]["tool_calls"][0]["id"], "ctc_1");
    assert_eq!(messages[1]["tool_call_id"], "ctc_1");
}

#[test]
fn coalesces_assistant_text_and_tool_calls_from_the_same_turn() {
    let request = json!({
        "model": "model",
        "input": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "Applying the patch." }]
            },
            {
                "type": "custom_tool_call",
                "call_id": "call_patch",
                "name": "apply_patch",
                "input": "*** Begin Patch\n*** End Patch"
            },
            {
                "type": "custom_tool_call_output",
                "call_id": "call_patch",
                "output": "ok"
            }
        ]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let messages = chat["messages"].as_array().unwrap();

    assert_eq!(roles(messages), ["assistant", "tool"]);
    assert_eq!(messages[0]["content"], "Applying the patch.");
    assert_eq!(messages[0]["tool_calls"][0]["id"], "call_patch");
    assert_eq!(messages[1]["tool_call_id"], "call_patch");
}

#[test]
fn coalesces_assistant_text_that_follows_tool_calls_before_their_outputs() {
    let request = json!({
        "model": "model",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_read",
                "name": "read_file",
                "arguments": "{\"path\":\"README.md\"}"
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "Reading the file." }]
            },
            {
                "type": "function_call_output",
                "call_id": "call_read",
                "output": "contents"
            }
        ]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let messages = chat["messages"].as_array().unwrap();

    assert_eq!(roles(messages), ["assistant", "tool"]);
    assert_eq!(messages[0]["content"], "Reading the file.");
    assert_eq!(messages[0]["tool_calls"][0]["id"], "call_read");
    assert_eq!(messages[1]["tool_call_id"], "call_read");
}

#[test]
fn malformed_stream_tool_arguments_fail_the_response() {
    let mut stream = ResponsesStream::new("resp_bad".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_bad",
                    "function": { "name": "lookup", "arguments": "{" }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    }));
    events.extend(stream.finish());
    let joined = String::from_utf8(events.concat()).unwrap();
    assert!(joined.contains("event: response.failed"));
    assert!(!joined.contains("event: response.completed"));
    assert!(joined.contains("malformed_tool_call_arguments"));
}

#[test]
fn drops_empty_user_messages_after_compaction() {
    let request = json!({
        "model": "model",
        "input": [
            { "type": "message", "role": "user", "content": [] },
            { "type": "message", "role": "user", "content": "" },
            {
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": "keep" }]
            }
        ]
    });

    let chat = responses_to_chat(&request, false).unwrap();

    let messages = chat["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["content"], "keep");
}

#[test]
fn keeps_image_only_user_message() {
    let request = json!({
        "model": "model",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{ "type": "input_image", "image_url": "https://example.test/a.png" }]
        }]
    });

    let chat = responses_to_chat(&request, false).unwrap();

    let messages = chat["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["content"][0]["type"], "image_url");
}

#[test]
fn normalizes_tool_parameters_for_strict_upstreams() {
    let request = json!({
        "model": "model",
        "tools": [
            { "type": "function", "name": "no_parameters" },
            { "type": "function", "name": "null_parameters", "parameters": null },
            {
                "type": "function",
                "name": "array_root",
                "parameters": { "type": "array", "items": { "type": "string" } }
            },
            {
                "type": "function",
                "name": "bad_required",
                "parameters": { "type": "object", "properties": {}, "required": "name" }
            },
            {
                "type": "function",
                "name": "closed_nested",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "inner": { "type": "object", "additionalProperties": false }
                    }
                }
            }
        ]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let tools = chat["tools"].as_array().unwrap();

    let empty_schema = json!({ "type": "object", "properties": {} });
    assert_eq!(tools[0]["function"]["parameters"], empty_schema);
    assert_eq!(tools[1]["function"]["parameters"], empty_schema);
    assert_eq!(tools[2]["function"]["parameters"]["type"], "object");
    assert_eq!(
        tools[2]["function"]["parameters"]["items"]["type"],
        "string"
    );
    assert_eq!(tools[3]["function"]["parameters"]["required"], json!([]));
    assert_eq!(
        tools[4]["function"]["parameters"]["properties"]["inner"]["required"],
        json!([])
    );
}

#[test]
fn leaves_schema_default_values_untouched() {
    let request = json!({
        "model": "model",
        "tools": [{
            "type": "function",
            "name": "with_default",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {
                        "type": "object",
                        "default": { "additionalProperties": false, "required": "nope" }
                    }
                }
            }
        }]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let default = &chat["tools"][0]["function"]["parameters"]["properties"]["y"]["default"];

    assert_eq!(default["additionalProperties"], false);
    assert_eq!(default["required"], "nope");
}

#[test]
fn usage_defaults_to_zero_when_upstream_omits_it() {
    let zero = json!({
        "input_tokens": 0,
        "input_tokens_details": { "cached_tokens": 0 },
        "output_tokens": 0,
        "total_tokens": 0,
        "output_tokens_details": { "reasoning_tokens": 0 }
    });

    assert_eq!(usage_to_responses(None), zero);
    assert_eq!(usage_to_responses(Some(&json!({}))), zero);
    assert_eq!(usage_to_responses(Some(&json!("oops"))), zero);
}

#[test]
fn usage_fills_missing_details_and_accepts_responses_names() {
    let usage = usage_to_responses(Some(&json!({
        "input_tokens": 7,
        "output_tokens": 4,
        "input_tokens_details": { "cached_tokens": 2 }
    })));

    assert_eq!(usage["input_tokens"], 7);
    assert_eq!(usage["output_tokens"], 4);
    assert_eq!(usage["total_tokens"], 11);
    assert_eq!(usage["input_tokens_details"]["cached_tokens"], 2);
    assert_eq!(usage["output_tokens_details"]["reasoning_tokens"], 0);
}

#[test]
fn usage_maps_deepseek_cache_hit_and_cache_write() {
    let usage = usage_to_responses(Some(&json!({
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
        "prompt_cache_hit_tokens": 60,
        "prompt_cache_miss_tokens": 40,
        "prompt_tokens_details": {
            "cache_write_tokens": 5
        }
    })));

    assert_eq!(usage["input_tokens"], 100);
    assert_eq!(usage["output_tokens"], 10);
    assert_eq!(usage["input_tokens_details"]["cached_tokens"], 60);
    assert_eq!(usage["input_tokens_details"]["cache_write_tokens"], 5);
    assert_eq!(usage["cache_creation_input_tokens"], 5);
}

#[test]
fn usage_prefers_direct_cache_read_over_deepseek_alias() {
    let usage = usage_to_responses(Some(&json!({
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "cache_read_input_tokens": 30,
        "prompt_cache_hit_tokens": 60
    })));

    assert_eq!(usage["input_tokens_details"]["cached_tokens"], 30);
    assert_eq!(usage["cache_read_input_tokens"], 30);
}

fn roles(messages: &[Value]) -> Vec<&str> {
    messages
        .iter()
        .filter_map(|message| message["role"].as_str())
        .collect()
}

fn large_image_data_url() -> String {
    format!("data:image/png;base64,{}", "A".repeat(9000))
}

#[test]
fn moves_tool_output_image_into_following_user_message() {
    let url = large_image_data_url();
    let request = json!({
        "model": "model",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "view_image",
                "arguments": "{}"
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": {
                    "content": [
                        { "type": "input_text", "text": "screenshot" },
                        { "type": "input_image", "image_url": url }
                    ]
                }
            }
        ]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let messages = chat["messages"].as_array().unwrap();

    assert_eq!(roles(messages), ["assistant", "tool", "user"]);
    let tool = messages[1]["content"].as_str().unwrap();
    assert!(tool.contains("moved to the following user message"));
    assert!(!tool.contains("base64"));
    assert_eq!(messages[2]["content"][0]["type"], "text");
    assert_eq!(messages[2]["content"][1]["type"], "image_url");
    assert_eq!(messages[2]["content"][1]["image_url"]["url"], url);
}

#[test]
fn keeps_parallel_tool_results_contiguous_around_media() {
    let url = large_image_data_url();
    let request = json!({
        "model": "model",
        "input": [
            { "type": "function_call", "call_id": "c1", "name": "shot", "arguments": "{}" },
            { "type": "function_call", "call_id": "c2", "name": "read", "arguments": "{}" },
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": {
                    "content": [{ "type": "input_image", "image_url": url }]
                }
            },
            { "type": "function_call_output", "call_id": "c2", "output": "ok" }
        ]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let messages = chat["messages"].as_array().unwrap();

    assert_eq!(roles(messages), ["assistant", "tool", "tool", "user"]);
    assert_eq!(messages[1]["tool_call_id"], "c1");
    assert_eq!(messages[2]["tool_call_id"], "c2");
    assert!(messages[3]["content"][0]["text"]
        .as_str()
        .unwrap()
        .contains("c1"));
}

#[test]
fn injects_missing_tool_placeholder_before_media_message() {
    let url = large_image_data_url();
    let request = json!({
        "model": "model",
        "input": [
            { "type": "function_call", "call_id": "c1", "name": "shot", "arguments": "{}" },
            { "type": "function_call", "call_id": "c2", "name": "read", "arguments": "{}" },
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": {
                    "content": [{ "type": "input_image", "image_url": url }]
                }
            }
        ]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let messages = chat["messages"].as_array().unwrap();

    assert_eq!(roles(messages), ["assistant", "tool", "tool", "user"]);
    assert_eq!(messages[1]["tool_call_id"], "c1");
    assert_eq!(messages[2]["tool_call_id"], "c2");
    assert_eq!(messages[2]["content"], MISSING_TOOL_OUTPUT);
    assert_eq!(messages[3]["content"][1]["type"], "image_url");
}

#[test]
fn extracts_nested_media_from_custom_tool_call_output() {
    let url = large_image_data_url();
    let output = json!({
        "content": [{ "type": "input_image", "image_url": url }]
    })
    .to_string();
    let request = json!({
        "model": "model",
        "input": [
            {
                "type": "custom_tool_call",
                "call_id": "call_9",
                "name": "apply_patch",
                "input": "*** Begin Patch"
            },
            { "type": "custom_tool_call_output", "call_id": "call_9", "output": output }
        ]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let messages = chat["messages"].as_array().unwrap();

    assert_eq!(roles(messages), ["assistant", "tool", "user"]);
    let tool = messages[1]["content"].as_str().unwrap();
    assert!(tool.contains("moved to the following user message"));
    assert!(!tool.contains("base64"));
    assert!(serde_json::from_str::<Value>(tool).is_ok());
    assert_eq!(messages[2]["content"][1]["image_url"]["url"], url);
}

#[test]
fn moves_supported_tool_output_file_to_following_user_message() {
    let request = json!({
        "model": "model",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_file",
                "name": "read_file",
                "arguments": "{}"
            },
            {
                "type": "function_call_output",
                "call_id": "call_file",
                "output": {
                    "content": [{
                        "type": "input_file",
                        "filename": "report.pdf",
                        "file_data": "data:application/pdf;base64,AA=="
                    }]
                }
            }
        ]
    });

    let chat =
        responses_to_chat_with_options(&request, false, TranslationOptions::new(true, false))
            .unwrap();
    let messages = chat["messages"].as_array().unwrap();

    assert_eq!(roles(messages), ["assistant", "tool", "user"]);
    assert!(messages[1]["content"].as_str().unwrap().contains("moved"));
    assert!(messages[2]["content"]
        .as_array()
        .unwrap()
        .iter()
        .any(|part| part["type"] == "file"));
}

#[test]
fn leaves_tool_output_without_media_unchanged() {
    let request = json!({
        "model": "model",
        "input": [
            { "type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}" },
            { "type": "function_call_output", "call_id": "c1", "output": "ok" }
        ]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let messages = chat["messages"].as_array().unwrap();

    assert_eq!(roles(messages), ["assistant", "tool"]);
    assert_eq!(messages[1]["content"], "ok");
}

#[test]
fn keeps_first_declaration_when_flat_tool_names_collide() {
    // `ns` + `exec` and the bare function `ns__exec` flatten to the same Chat
    // name. The first declaration is what the upstream is told about, so the
    // losing one must not reach the tool list.
    let request = json!({
        "model": "model",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{ "type": "input_text", "text": "hi" }]
        }],
        "tools": [{
            "type": "namespace",
            "name": "ns",
            "tools": [{
                "type": "function",
                "name": "exec",
                "description": "from namespace",
                "parameters": {
                    "type": "object",
                    "properties": { "cmd": { "type": "string" } }
                }
            }]
        }, {
            "type": "function",
            "name": "ns__exec",
            "description": "from top level",
            "parameters": { "type": "object", "properties": {} }
        }]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let tools = chat["tools"].as_array().unwrap();

    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["function"]["name"], "ns__exec");
    assert_eq!(tools[0]["function"]["description"], "from namespace");
    assert!(tools[0]["function"]["parameters"]["properties"]["cmd"].is_object());
}

#[test]
fn reverse_maps_only_surviving_flat_tool_names() {
    let namespace_first = json!({
        "tools": [{
            "type": "namespace",
            "name": "ns",
            "tools": [{ "type": "function", "name": "exec" }]
        }, {
            "type": "function",
            "name": "ns__exec"
        }]
    });
    let shape = ToolShape::from_request(&namespace_first);
    assert_eq!(
        shape.resolve("ns__exec"),
        ("exec".to_string(), Some("ns".to_string()))
    );

    let function_first = json!({
        "tools": [{
            "type": "function",
            "name": "ns__exec"
        }, {
            "type": "namespace",
            "name": "ns",
            "tools": [{ "type": "function", "name": "exec" }]
        }]
    });
    let shape = ToolShape::from_request(&function_first);
    assert_eq!(shape.resolve("ns__exec"), ("ns__exec".to_string(), None));
}

#[test]
fn reports_duplicate_flat_tool_names() {
    let request = json!({
        "tools": [{
            "type": "function",
            "name": "ns__exec"
        }],
        "input": [{
            "type": "additional_tools",
            "role": "developer",
            "tools": [{
                "type": "namespace",
                "name": "ns",
                "tools": [{ "type": "function", "name": "exec" }]
            }]
        }]
    });

    let details = ignored_feature_details(&request);

    assert!(
        details.contains(&"tools.duplicate:ns__exec kept=top dropped=additional_tools".to_string()),
        "{details:?}"
    );
}

#[test]
fn reports_and_drops_hosted_web_search_from_every_tool_carrier() {
    let request = json!({
        "tools": [{
            "type": "web_search",
            "external_web_access": true
        }],
        "input": [{
            "type": "additional_tools",
            "tools": [{
                "type": "web_search",
                "external_web_access": false
            }]
        }, {
            "type": "tool_search_output",
            "tools": [{
                "type": "web_search",
                "external_web_access": true
            }]
        }]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let details = ignored_feature_details(&request);

    assert!(chat["tools"].as_array().is_none_or(Vec::is_empty));
    assert_eq!(
        details
            .iter()
            .filter(|detail| detail.as_str() == "tools.ignored:web_search")
            .count(),
        1,
        "{details:?}"
    );
}

#[test]
fn reads_reasoning_from_reasoning_details() {
    assert_eq!(
        reasoning_text(&json!({ "reasoning_details": [{ "text": "think" }] })).as_deref(),
        Some("think")
    );
    assert_eq!(
        reasoning_text(&json!({ "reasoning_details": [{ "summary": "short" }] })).as_deref(),
        Some("short")
    );
    // The explicit field still wins when both are present.
    assert_eq!(
        reasoning_text(&json!({
            "reasoning_content": "primary",
            "reasoning_details": [{ "text": "secondary" }]
        }))
        .as_deref(),
        Some("primary")
    );
}

#[test]
fn reasoning_text_ignores_whitespace_only_sources() {
    for value in [
        json!({ "reasoning_content": " " }),
        json!({ "reasoning_content": "\n\t " }),
        json!({ "reasoning": " \n" }),
        json!({ "reasoning": { "content": "\t" } }),
        json!({ "reasoning": { "text": "  " } }),
        json!({ "reasoning": { "summary": "\n" } }),
        json!({ "reasoning_details": [{ "text": " " }] }),
        json!({ "reasoning_details": [{ "parts": [{ "text": "\n" }] }] }),
    ] {
        assert_eq!(reasoning_text(&value), None, "{value}");
    }
    assert_eq!(
        reasoning_text(&json!({
            "reasoning_content": " ",
            "reasoning_details": [{ "text": "fallback" }]
        }))
        .as_deref(),
        Some("fallback")
    );
    assert_eq!(
        reasoning_text(&json!({
            "reasoning": " ",
            "reasoning_details": [{ "text": "fallback detail" }]
        }))
        .as_deref(),
        Some("fallback detail")
    );
}

#[test]
fn stream_reasoning_uses_fallback_when_preferred_field_is_blank() {
    let mut stream = ResponsesStream::new("resp_fallback_think".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "reasoning": " ",
                "reasoning_details": [{ "text": "fallback detail" }]
            },
            "finish_reason": "stop"
        }]
    }));
    events.extend(stream.finish());

    let delta = named_events(&events, "response.reasoning_summary_text.delta")
        .into_iter()
        .filter_map(|event| event["delta"].as_str().map(str::to_string))
        .collect::<String>();
    assert_eq!(delta, "fallback detail");
}

#[test]
fn stream_ignores_whitespace_only_reasoning() {
    let mut stream = ResponsesStream::new("resp_blank_think".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": { "reasoning_content": " " },
            "finish_reason": null
        }]
    }));
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": { "reasoning_content": "\n\t " },
            "finish_reason": "stop"
        }]
    })));
    events.extend(stream.finish());

    assert!(named_events(&events, "response.reasoning_summary_part.added").is_empty());
    assert!(named_events(&events, "response.reasoning_summary_text.delta").is_empty());
    assert!(named_events(&events, "response.reasoning_summary_text.done").is_empty());
    let completed = terminal_response(&events, "response.completed");
    assert!(completed["response"]["output"]
        .as_array()
        .unwrap()
        .iter()
        .all(|item| item["type"] != "reasoning"));
}

#[test]
fn stream_ignores_whitespace_only_inline_think() {
    let mut stream = ResponsesStream::new("resp_blank_inline".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": { "content": "<think> \n\t </think>\nanswer" },
            "finish_reason": "stop"
        }]
    }));
    events.extend(stream.finish());

    assert!(named_events(&events, "response.reasoning_summary_part.added").is_empty());
    assert!(named_events(&events, "response.reasoning_summary_text.delta").is_empty());
    let completed = terminal_response(&events, "response.completed");
    assert_eq!(completed["response"]["output"][0]["type"], "message");
    assert_eq!(
        completed["response"]["output"][0]["content"][0]["text"],
        "answer"
    );
}

#[test]
fn non_stream_ignores_whitespace_only_inline_think() {
    let chat = json!({
        "id": "chatcmpl-blank-inline",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "<thinking>\n \t</thinking>\nanswer"
            },
            "finish_reason": "stop"
        }]
    });
    let response = chat_to_responses(&chat, "model", &ToolShape::default()).unwrap();
    assert_eq!(response["output"].as_array().unwrap().len(), 1);
    assert_eq!(response["output"][0]["type"], "message");
    assert_eq!(response["output"][0]["content"][0]["text"], "answer");
}

#[test]
fn stream_preserves_leading_and_interior_reasoning_whitespace_after_valid_start() {
    let mut stream = ResponsesStream::new("resp_padded_think".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": { "reasoning_content": " \n" },
            "finish_reason": null
        }]
    }));
    assert!(named_events(&events, "response.reasoning_summary_part.added").is_empty());
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": { "reasoning_content": "thinking" },
            "finish_reason": null
        }]
    })));
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": { "reasoning_content": "\n\n" },
            "finish_reason": "stop"
        }]
    })));
    events.extend(stream.finish());

    let delta = named_events(&events, "response.reasoning_summary_text.delta")
        .into_iter()
        .filter_map(|event| event["delta"].as_str().map(str::to_string))
        .collect::<String>();
    assert_eq!(delta, " \nthinking\n\n");
}

#[test]
fn reads_reasoning_object_and_joins_multiple_details() {
    assert_eq!(
        reasoning_text(&json!({
            "reasoning": { "content": "object content" }
        }))
        .as_deref(),
        Some("object content")
    );
    assert_eq!(
        reasoning_text(&json!({
            "reasoning": { "text": "object text" }
        }))
        .as_deref(),
        Some("object text")
    );
    assert_eq!(
        reasoning_text(&json!({
            "reasoning": { "summary": "object summary" }
        }))
        .as_deref(),
        Some("object summary")
    );
    assert_eq!(
        reasoning_text(&json!({
            "reasoning_details": [
                { "text": "first" },
                { "content": "second" },
                { "summary": "third" }
            ]
        }))
        .as_deref(),
        Some("first\n\nsecond\n\nthird")
    );
}

#[test]
fn reasoning_details_ignores_blank_entries_and_keeps_valid_order() {
    assert_eq!(
        reasoning_text(&json!({
            "reasoning_details": [
                { "text": " " },
                { "text": "first" },
                { "text": "\n" },
                { "text": "second" }
            ]
        }))
        .as_deref(),
        Some("first\n\nsecond")
    );
}

#[test]
fn stream_reasoning_blank_source_switch_emits_only_non_blank_values() {
    let mut stream = ResponsesStream::new("resp_blank_switch".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": { "reasoning_content": " " },
            "finish_reason": null
        }]
    }));
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": { "reasoning_details": [{ "text": "valid" }] },
            "finish_reason": "stop"
        }]
    })));
    events.extend(stream.finish());

    let deltas = named_events(&events, "response.reasoning_summary_text.delta")
        .into_iter()
        .filter_map(|event| event["delta"].as_str().map(str::to_string))
        .collect::<String>();
    assert_eq!(deltas, " valid");
}

#[test]
fn history_keeps_reasoning_with_the_assistant_turn_that_follows_it() {
    let request = json!({
        "model": "model",
        "input": [
            { "type": "reasoning", "summary": [{ "type": "summary_text", "text": "before call" }] },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": "{}"
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok"
            },
            { "type": "reasoning", "summary": [{ "type": "summary_text", "text": "after result" }] },
            {
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "done" }]
            }
        ]
    });
    let chat = responses_to_chat(&request, false).unwrap();
    let messages = chat["messages"].as_array().unwrap();
    assert_eq!(roles(messages), ["assistant", "tool", "assistant"]);
    assert_eq!(messages[0]["reasoning_content"], "before call");
    assert_eq!(messages[2]["reasoning_content"], "after result");
    assert_eq!(messages[2]["content"], "done");
}

#[test]
fn non_stream_message_reads_reasoning_details() {
    let chat = json!({
        "id": "chatcmpl-1",
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "done",
                "reasoning_details": [{ "text": "because" }]
            }
        }]
    });

    let response = chat_to_responses(&chat, "model", &ToolShape::default()).unwrap();

    assert_eq!(response["output"][0]["type"], "reasoning");
    assert_eq!(response["output"][0]["summary"][0]["text"], "because");
}

#[test]
fn non_stream_ignores_whitespace_only_reasoning() {
    let chat = json!({
        "id": "chatcmpl-blank",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "done",
                "reasoning_content": " \n"
            },
            "finish_reason": "stop"
        }]
    });
    let response = chat_to_responses(&chat, "model", &ToolShape::default()).unwrap();
    assert!(response["output"]
        .as_array()
        .unwrap()
        .iter()
        .all(|item| item["type"] != "reasoning"));
}

#[test]
fn chat_completion_aggregation_ignores_whitespace_only_reasoning() {
    assert_eq!(
        reasoning_text(&json!({ "reasoning_content": " ", "content": "ok" })),
        None
    );
}

#[test]
fn reasoning_item_text_prefers_non_blank_content_over_blank_summary() {
    let request = json!({
        "model": "model",
        "input": [{
            "type": "reasoning",
            "summary": [{ "type": "summary_text", "text": " " }],
            "content": [{ "type": "reasoning_text", "text": "actual thought" }]
        }, {
            "type": "message",
            "role": "assistant",
            "content": "continue"
        }]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    assert_eq!(chat["messages"][0]["reasoning_content"], "actual thought");

    let blank = json!({
        "model": "model",
        "input": [{
            "type": "reasoning",
            "summary": [{ "type": "summary_text", "text": " " }]
        }, {
            "type": "message",
            "role": "assistant",
            "content": "continue"
        }]
    });
    let chat = responses_to_chat(&blank, false).unwrap();
    assert!(chat["messages"][0].get("reasoning_content").is_none());
}

#[test]
fn stream_preserves_interior_blank_reasoning_after_valid_start() {
    let mut stream = ResponsesStream::new("resp_interior_blank".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": { "reasoning_content": "start" },
            "finish_reason": null
        }]
    }));
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": { "reasoning_content": "\n\n" },
            "finish_reason": null
        }]
    })));
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": { "reasoning_content": "end" },
            "finish_reason": "stop"
        }]
    })));
    events.extend(stream.finish());

    let deltas = named_events(&events, "response.reasoning_summary_text.delta")
        .into_iter()
        .filter_map(|event| event["delta"].as_str().map(str::to_string))
        .collect::<String>();
    let done = named_events(&events, "response.reasoning_summary_text.done");
    assert_eq!(deltas, "start\n\nend");
    assert_eq!(done[0]["text"], "start\n\nend");
}

#[test]
fn stream_merges_reasoning_source_switch_within_one_item() {
    let mut stream = ResponsesStream::new("resp_source_switch".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": { "reasoning_content": "primary " },
            "finish_reason": null
        }]
    }));
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": { "reasoning_details": [{ "text": "fallback" }] },
            "finish_reason": "stop"
        }]
    })));
    events.extend(stream.finish());

    let completed = terminal_response(&events, "response.completed");
    let reasoning: Vec<_> = completed["response"]["output"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "reasoning")
        .collect();
    assert_eq!(reasoning.len(), 1);
    assert_eq!(reasoning[0]["summary"][0]["text"], "primary fallback");
    let deltas = named_events(&events, "response.reasoning_summary_text.delta")
        .into_iter()
        .filter_map(|event| event["delta"].as_str().map(str::to_string))
        .collect::<String>();
    let done = named_events(&events, "response.reasoning_summary_text.done");
    assert_eq!(deltas, "primary fallback");
    assert_eq!(done.len(), 1);
    assert_eq!(done[0]["text"], "primary fallback");
}

#[test]
fn stream_separates_reasoning_after_a_completed_tool_call() {
    let mut stream = ResponsesStream::new("resp_two_turns".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": { "reasoning_content": "first thought" },
            "finish_reason": null
        }]
    }));
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "function": { "name": "read_file", "arguments": "{}" }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })));
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": { "reasoning_content": "second thought" },
            "finish_reason": null
        }]
    })));
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": { "content": "done" },
            "finish_reason": "stop"
        }]
    })));
    events.extend(stream.finish());

    let output = terminal_response(&events, "response.completed")["response"]["output"]
        .as_array()
        .unwrap()
        .clone();
    let kinds = output
        .iter()
        .map(|item| item["type"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        kinds,
        ["reasoning", "function_call", "reasoning", "message"]
    );
    assert_eq!(output[0]["summary"][0]["text"], "first thought");
    assert_eq!(output[2]["summary"][0]["text"], "second thought");

    let names = event_names(&events);
    let first_done = names
        .iter()
        .position(|name| name == "response.output_item.done")
        .unwrap();
    let added_positions = names
        .iter()
        .enumerate()
        .filter(|(_, name)| name.as_str() == "response.output_item.added")
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    assert_eq!(added_positions.len(), 4);
    assert!(added_positions[1] < first_done);
    assert!(first_done < added_positions[2]);
}

#[test]
fn stream_completes_tool_calls_before_later_reasoning() {
    let mut stream =
        ResponsesStream::new("resp_tool_then_reasoning".to_string(), "model".to_string());
    let first = stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "function": { "name": "read_file", "arguments": "{\"path\":\"a\"}" }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    }));
    let call_done = named_events(&first, "response.output_item.done");
    assert_eq!(call_done.len(), 1);
    assert_eq!(call_done[0]["item"]["type"], "function_call");
    assert_eq!(call_done[0]["item"]["call_id"], "call_1");
}

#[test]
fn stream_does_not_complete_the_same_tool_call_twice() {
    let mut stream = ResponsesStream::new("resp_duplicate_done".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "function": { "name": "read_file", "arguments": "{}" }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    }));
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "function": { "name": "read_file", "arguments": "{}" }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })));
    events.extend(stream.finish());

    let items = named_events(&events, "response.output_item.done");
    let calls = items
        .iter()
        .filter(|event| event["item"]["type"] == "function_call")
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        terminal_response(&events, "response.completed")["response"]["output"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|item| item["type"] == "function_call")
            .count(),
        1
    );
}

#[test]
fn stream_reuses_call_identity_when_a_late_index_changes() {
    let mut stream = ResponsesStream::new("resp_moved_index".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "id": "call_1",
                    "function": { "name": "read_file", "arguments": "{\"path\":" }
                }]
            },
            "finish_reason": null
        }]
    }));
    events.extend(stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 4,
                    "id": "call_1",
                    "function": { "arguments": "\"a.txt\"}" }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })));
    events.extend(stream.finish());

    let completed = terminal_response(&events, "response.completed");
    let calls = completed["response"]["output"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["type"] == "function_call")
        .collect::<Vec<_>>();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["arguments"], "{\"path\":\"a.txt\"}");
}

#[test]
fn stream_does_not_publish_empty_shell_tool_calls() {
    let mut stream = ResponsesStream::new("resp_empty_shell".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 3,
                    "function": { "arguments": "" }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    }));
    events.extend(stream.finish());

    let output = terminal_response(&events, "response.failed")["response"]["output"]
        .as_array()
        .unwrap()
        .clone();
    assert!(output.is_empty());
    assert!(named_events(&events, "response.output_item.done").is_empty());
}

#[test]
fn stream_keeps_late_tool_identity_for_finish() {
    let mut stream = ResponsesStream::new("resp_late_identity".to_string(), "model".to_string());
    let first = stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": { "arguments": "{}" }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    }));
    assert!(named_events(&first, "response.output_item.done").is_empty());
    let mut events = first;
    events.extend(stream.finish());
    let failed = terminal_response(&events, "response.failed");
    assert_eq!(
        failed["response"]["error"]["code"],
        "upstream_tool_call_dropped"
    );
}

#[test]
fn stream_closes_same_delta_reasoning_before_tool_item_done() {
    let mut stream = ResponsesStream::new("resp_same_delta".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "reasoning_content": "use a tool",
                "tool_calls": [{
                    "index": 0,
                    "id": "call_1",
                    "function": { "name": "read_file", "arguments": "{}" }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    }));
    events.extend(stream.finish());

    let names = event_names(&events);
    let first_reasoning_done = names
        .iter()
        .position(|name| name == "response.reasoning_summary_text.done")
        .unwrap();
    let first_tool_done = names
        .iter()
        .position(|name| name == "response.output_item.done")
        .unwrap();
    assert!(first_reasoning_done < first_tool_done);
    assert!(
        named_events(&events, "response.reasoning_summary_text.delta")[0]["delta"]
            .as_str()
            .unwrap()
            .contains("use a tool")
    );
}

#[test]
fn stream_finishes_open_reasoning_before_incomplete_terminal() {
    let mut stream = ResponsesStream::new("resp_missing_finish".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": { "reasoning_content": "partial thought" },
            "finish_reason": null
        }]
    }));
    events.extend(stream.finish());

    let incomplete = terminal_response(&events, "response.incomplete");
    assert_eq!(incomplete["response"]["output"][0]["type"], "reasoning");
    assert_eq!(
        incomplete["response"]["output"][0]["summary"][0]["text"],
        "partial thought"
    );
}

#[test]
fn stream_does_not_emit_empty_reasoning_item_when_only_blank_chunks_arrive() {
    let mut stream = ResponsesStream::new("resp_blank_part".to_string(), "model".to_string());
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": { "reasoning_content": " " },
            "finish_reason": null
        }]
    }));
    events.extend(stream.finish());

    let joined = String::from_utf8(events.concat()).unwrap();
    assert!(!joined.contains("response.reasoning_summary_part.added"));
    assert!(!joined.contains("response.reasoning_summary_text.done"));
    assert!(!joined.contains("\"type\":\"reasoning\""));
}

#[test]
fn reports_ignored_input_items_and_content_parts() {
    let request = json!({
        "input": [
            { "type": "web_search_call" },
            "not-an-object",
            { "no_type": true },
            {
                "type": "message",
                "role": "user",
                "content": [
                    { "type": "input_audio", "input_audio": { "data": "x" } },
                    { "type": "input_text", "text": "kept" }
                ]
            },
            { "type": "web_search_call" }
        ]
    });

    let details = ignored_feature_details(&request);

    assert_eq!(
        details,
        [
            "content.part:input_audio_unsupported",
            "input.item:non_object",
            "input.item:none",
            "input.item_degraded:web_search_call"
        ]
    );
}

#[test]
fn degrades_unrepresentable_hosted_items_to_text_context() {
    let request = json!({
        "model": "model",
        "input": [
            {
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed"
            },
            {
                "type": "computer_call",
                "call_id": "computer_1",
                "action": { "type": "click" }
            },
            {
                "type": "mcp_call",
                "name": "lookup"
            },
            {
                "type": "local_shell_call",
                "command": ["dir"]
            }
        ]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let text = chat["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|message| message.get("content").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("web_search_call"));
    assert!(text.contains("computer_call"));
    assert!(text.contains("mcp_call"));
    assert!(text.contains("local_shell_call"));

    let details = ignored_feature_details(&request);
    assert!(details.contains(&"input.item_degraded:web_search_call".to_string()));
    assert!(!details.contains(&"input.item:web_search_call".to_string()));
}

#[test]
fn reports_nothing_for_fully_translated_request() {
    let request = json!({
        "instructions": "be brief",
        "input": [
            { "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "hi" }] },
            { "type": "reasoning", "summary": [{ "type": "summary_text", "text": "thinking" }] },
            { "type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}" },
            { "type": "function_call_output", "call_id": "c1", "output": "ok" },
            {
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "done" }]
            }
        ],
        "tools": [{
            "type": "namespace",
            "name": "ns",
            "tools": [{ "type": "function", "name": "exec" }]
        }]
    });

    assert_eq!(ignored_feature_details(&request), Vec::<String>::new());
}

#[test]
fn handled_input_types_stay_in_sync() {
    for name in HANDLED_INPUT_TYPES {
        let request = json!({ "input": [{ "type": name, "role": "user", "content": [] }] });
        let details = ignored_feature_details(&request);

        assert!(
            !details
                .iter()
                .any(|detail| detail == &format!("input.item:{name}")),
            "handled type {name} is reported as dropped: {details:?}"
        );
    }
}

#[test]
fn passes_extended_chat_fields_and_ignores_nulls() {
    let request = json!({
        "model": "model",
        "input": "hello",
        "frequency_penalty": 0.5,
        "logit_bias": { "42": -5 },
        "logprobs": true,
        "metadata": { "trace": "abc" },
        "n": 1,
        "presence_penalty": 0.25,
        "stop": ["DONE"],
        "top_logprobs": 3,
        "user": "user-1",
        "seed": null
    });

    let chat = responses_to_chat(&request, false).unwrap();

    assert_eq!(chat["frequency_penalty"], 0.5);
    assert_eq!(chat["logit_bias"]["42"], -5);
    assert_eq!(chat["logprobs"], true);
    assert_eq!(chat["metadata"]["trace"], "abc");
    assert_eq!(chat["n"], 1);
    assert_eq!(chat["presence_penalty"], 0.25);
    assert_eq!(chat["stop"], json!(["DONE"]));
    assert_eq!(chat["top_logprobs"], 3);
    assert_eq!(chat["user"], "user-1");
    assert!(chat.get("seed").is_none());
}

#[test]
fn translates_text_format_json_schema() {
    let request = json!({
        "model": "model",
        "input": "hi",
        "text": { "format": {
            "type": "json_schema",
            "name": "openai_compat_output_schema",
            "strict": true,
            "schema": {
                "type": "object",
                "properties": { "answer": { "type": "string" } }
            }
        }}
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let format = &chat["response_format"];

    assert_eq!(format["type"], "json_schema");
    assert_eq!(format["json_schema"]["name"], "openai_compat_output_schema");
    assert_eq!(format["json_schema"]["strict"], true);
    assert_eq!(
        format["json_schema"]["schema"]["properties"]["answer"]["type"],
        "string"
    );
}

#[test]
fn prefers_top_level_format_schema() {
    let request = json!({
        "text": { "format": {
            "type": "json_schema",
            "name": "flat_name",
            "schema": { "type": "object", "properties": { "flat": { "type": "string" } } },
            "json_schema": {
                "name": "nested_name",
                "schema": { "type": "object", "properties": { "nested": { "type": "string" } } }
            }
        }}
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let json_schema = &chat["response_format"]["json_schema"];

    assert_eq!(json_schema["name"], "flat_name");
    assert!(json_schema["schema"]["properties"]["flat"].is_object());
    assert!(json_schema["schema"]["properties"]["nested"].is_null());
}

#[test]
fn translates_text_format_json_object_and_omits_text() {
    let object = responses_to_chat(
        &json!({ "input": "hi", "text": { "format": { "type": "json_object" } } }),
        false,
    )
    .unwrap();
    assert_eq!(object["response_format"], json!({ "type": "json_object" }));

    let text = responses_to_chat(
        &json!({ "input": "hi", "text": { "format": { "type": "text" } } }),
        false,
    )
    .unwrap();
    assert!(text.get("response_format").is_none());
}

#[test]
fn reports_unknown_text_format_type() {
    let request = json!({
        "input": "hi",
        "text": { "format": { "type": "json_schema" } }
    });

    let chat = responses_to_chat(&request, false).unwrap();

    assert!(chat.get("response_format").is_none());
    assert_eq!(
        ignored_feature_details(&request),
        ["text.format.type:json_schema_without_schema"]
    );
}

#[test]
fn merges_additional_tools_without_emitting_a_message() {
    let request = json!({
        "model": "model",
        "input": [
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [{
                    "type": "function",
                    "name": "read_file",
                    "parameters": { "type": "object", "properties": {} }
                }]
            },
            {
                "type": "message",
                "role": "developer",
                "content": [{ "type": "input_text", "text": "rules" }]
            },
            {
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": "hi" }]
            }
        ]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let messages = chat["messages"].as_array().unwrap();

    assert_eq!(roles(messages), ["system", "user"]);
    assert_eq!(messages[0]["content"], "rules");
    assert_eq!(chat["tools"][0]["function"]["name"], "read_file");
}

#[test]
fn lets_top_level_tools_win_over_input_carriers() {
    let request = json!({
        "tools": [{
            "type": "function",
            "name": "read_file",
            "description": "from top level"
        }],
        "input": [{
            "type": "additional_tools",
            "role": "developer",
            "tools": [{
                "type": "function",
                "name": "read_file",
                "description": "from carrier"
            }]
        }]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let tools = chat["tools"].as_array().unwrap();

    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["function"]["description"], "from top level");
}

#[test]
fn defers_tool_search_output_to_a_note_after_the_tool_block() {
    let request = json!({
        "input": [
            { "type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}" },
            { "type": "function_call_output", "call_id": "c1", "output": "ok" },
            {
                "type": "tool_search_output",
                "call_id": "search_1",
                "tools": [{
                    "type": "namespace",
                    "name": "ns",
                    "tools": [{ "type": "function", "name": "exec" }]
                }]
            }
        ]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let messages = chat["messages"].as_array().unwrap();

    assert_eq!(roles(messages), ["assistant", "tool", "user"]);
    let note = messages[2]["content"].as_str().unwrap();
    assert!(note.contains("ns__exec"), "{note}");
    assert_eq!(chat["tools"][0]["function"]["name"], "ns__exec");
}

#[test]
fn shortens_overlong_namespace_tool_names_and_restores_them() {
    let namespace = "namespace".repeat(6);
    let tool_name = "very_long_tool_name".repeat(4);
    let request = json!({
        "model": "model",
        "input": "run",
        "tools": [{
            "type": "namespace",
            "name": namespace,
            "tools": [{
                "type": "function",
                "name": tool_name,
                "parameters": { "type": "object", "properties": {} }
            }]
        }]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let chat_name = chat["tools"][0]["function"]["name"].as_str().unwrap();

    assert!(chat_name.len() <= 64, "{chat_name}");
    let shape = ToolShape::from_request(&request);
    assert_eq!(
        shape.resolve(chat_name),
        (tool_name.clone(), Some(namespace.clone()))
    );

    let response = chat_to_responses(
        &json!({
            "id": "chatcmpl-long",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_long",
                        "function": {
                            "name": chat_name,
                            "arguments": "{}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }),
        "model",
        &shape,
    )
    .unwrap();

    assert_eq!(response["output"][0]["name"], tool_name);
    assert_eq!(response["output"][0]["namespace"], namespace);
}

#[test]
fn compaction_turn_rewrites_trigger_and_drops_tool_fields() {
    let request = json!({
        "model": "model",
        "instructions": "be brief",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": "do the work" }]
            },
            { "type": "compaction_trigger" }
        ],
        "tools": [{ "type": "function", "name": "read" }],
        "tool_choice": "auto",
        "parallel_tool_calls": true,
        "prompt_cache_key": "session-1",
        "text": { "format": { "type": "json_object" } }
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let messages = chat["messages"].as_array().unwrap();

    assert_eq!(messages.last().unwrap()["content"], COMPACTION_PROMPT);
    assert!(chat.get("tools").is_none());
    assert!(chat.get("tool_choice").is_none());
    assert!(chat.get("parallel_tool_calls").is_none());
    assert!(chat.get("response_format").is_none());
    // Cache and tier fields are unrelated to tools and stay for prefix reuse.
    assert_eq!(chat["prompt_cache_key"], "session-1");
}

#[test]
fn replaces_encrypted_compaction_with_placeholder() {
    let request = json!({
        "input": [
            { "type": "compaction", "encrypted_content": "gAAAAABm" },
            {
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": "continue" }]
            }
        ]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let messages = chat["messages"].as_array().unwrap();

    assert_eq!(roles(messages), ["user", "user"]);
    assert_eq!(messages[0]["content"], COMPACTION_PLACEHOLDER);
    assert_eq!(messages[1]["content"], "continue");
}

#[test]
fn uses_plaintext_compaction_summary_when_available() {
    let request = json!({
        "input": [
            { "type": "compaction", "summary": "we shipped the parser" },
            {
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": "continue" }]
            }
        ]
    });

    let chat = responses_to_chat(&request, false).unwrap();

    assert_eq!(chat["messages"][0]["content"], "we shipped the parser");
}

#[test]
fn keeps_tools_on_non_compaction_turns_with_compaction_history() {
    let request = json!({
        "input": [
            { "type": "compaction", "summary": "earlier work" },
            {
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": "next step" }]
            }
        ],
        "tools": [{ "type": "function", "name": "read" }],
        "text": { "format": { "type": "json_object" } }
    });

    let chat = responses_to_chat(&request, false).unwrap();

    assert_eq!(chat["tools"][0]["function"]["name"], "read");
    assert_eq!(chat["response_format"]["type"], "json_object");
}

#[test]
fn normalizes_trace_sized_tool_history_without_leaving_dangling_calls() {
    let mut input = Vec::new();
    for index in 0..9 {
        input.push(json!({
            "type": "message",
            "role": "developer",
            "content": format!("system {index}")
        }));
    }

    let mut emitted_outputs = 0;
    for group in 0..10 {
        let call_count = if group == 9 { 24 } else { 18 };
        for call in 0..call_count {
            input.push(json!({
                "type": "function_call",
                "call_id": format!("call_{group}_{call}"),
                "name": "read_file",
                "arguments": "{}"
            }));
        }
        let output_count = if emitted_outputs < 18 {
            2
        } else if emitted_outputs < 21 {
            3
        } else {
            0
        };
        for call in 0..output_count {
            input.push(json!({
                "type": "function_call_output",
                "call_id": format!("call_{group}_{call}"),
                "output": "ok"
            }));
            emitted_outputs += 1;
        }
    }

    let chat = responses_to_chat(
        &json!({
            "model": "model",
            "input": input
        }),
        false,
    )
    .unwrap();
    let messages = chat["messages"].as_array().unwrap();
    let system_count = messages
        .iter()
        .filter(|message| message["role"] == "system")
        .count();
    let assistant_count = messages
        .iter()
        .filter(|message| message["role"] == "assistant")
        .count();
    let tool_count = messages
        .iter()
        .filter(|message| message["role"] == "tool")
        .count();
    let call_count = messages
        .iter()
        .filter_map(|message| message["tool_calls"].as_array())
        .flatten()
        .count();

    assert_eq!(system_count, 1);
    assert_eq!(assistant_count, 10);
    assert_eq!(tool_count, 186);
    assert_eq!(call_count, 186);
}

#[test]
fn preserves_custom_tool_definition_in_chat_description() {
    let request = json!({
        "model": "model",
        "input": "edit the file",
        "tools": [{
            "type": "custom",
            "name": "apply_patch",
            "description": "Apply a patch to files.",
            "format": {
                "type": "grammar",
                "syntax": "lark",
                "definition": "start: begin_patch hunk+ end_patch"
            }
        }]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let description = chat["tools"][0]["function"]["description"]
        .as_str()
        .unwrap();

    assert!(description.contains("Apply a patch to files."));
    assert!(description.contains("Original tool definition:"));
    assert!(description.contains("\"format\":"));
    assert!(description.contains("\"syntax\":\"lark\""));
    assert!(description.contains("start: begin_patch hunk+ end_patch"));
    assert!(description.contains("Do not wrap"));
}

#[test]
fn preserves_strict_flag_on_function_tools() {
    let request = json!({
        "model": "model",
        "input": "run",
        "tools": [{
            "type": "function",
            "name": "lookup",
            "strict": true,
            "parameters": {
                "type": "object",
                "properties": { "query": { "type": "string" } }
            }
        }]
    });

    let chat = responses_to_chat(&request, false).unwrap();

    assert_eq!(chat["tools"][0]["function"]["strict"], true);
}

#[test]
fn accepts_nested_function_tool_shape() {
    let request = json!({
        "model": "model",
        "input": "run",
        "tools": [{
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "nested",
                "strict": true,
                "parameters": {
                    "type": "object",
                    "properties": { "query": { "type": "string" } }
                }
            }
        }]
    });

    let chat = responses_to_chat(&request, false).unwrap();

    assert_eq!(chat["tools"][0]["function"]["name"], "lookup");
    assert_eq!(chat["tools"][0]["function"]["description"], "nested");
    assert_eq!(chat["tools"][0]["function"]["strict"], true);
    assert_eq!(
        chat["tools"][0]["function"]["parameters"]["properties"]["query"]["type"],
        "string"
    );
}

#[test]
fn preserves_tool_extra_content_in_history_and_response() {
    let extra = json!({
        "google": { "thought_signature": "sig-abc" }
    });
    let request = json!({
        "model": "model",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": "{}",
                "extra_content": extra
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok"
            }
        ]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    assert_eq!(chat["messages"][0]["tool_calls"][0]["extra_content"], extra);

    let response = chat_to_responses(
        &json!({
            "id": "chatcmpl-extra",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "extra_content": extra,
                        "function": {
                            "name": "read_file",
                            "arguments": "{}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }),
        "model",
        &ToolShape::default(),
    )
    .unwrap();
    assert_eq!(response["output"][0]["extra_content"], extra);
}

#[test]
fn translates_custom_tool_choice_to_chat_function() {
    let request = json!({
        "model": "model",
        "input": "edit the file",
        "tools": [{
            "type": "custom",
            "name": "apply_patch"
        }],
        "tool_choice": {
            "type": "custom",
            "name": "apply_patch"
        }
    });

    let chat = responses_to_chat(&request, false).unwrap();

    assert_eq!(
        chat["tool_choice"],
        json!({
            "type": "function",
            "function": { "name": "apply_patch" }
        })
    );
}

#[test]
fn translates_namespaced_tool_choice_to_flattened_chat_name() {
    let request = json!({
        "model": "model",
        "input": "run",
        "tools": [{
            "type": "namespace",
            "name": "codexui",
            "tools": [{
                "type": "function",
                "name": "get_usage",
                "parameters": { "type": "object", "properties": {} }
            }]
        }],
        "tool_choice": {
            "type": "function",
            "name": "get_usage",
            "namespace": "codexui"
        }
    });

    let chat = responses_to_chat(&request, false).unwrap();

    assert_eq!(
        chat["tool_choice"],
        json!({
            "type": "function",
            "function": { "name": "codexui__get_usage" }
        })
    );
}

#[test]
fn maps_tool_search_declaration_and_choice_to_proxy_function() {
    let request = json!({
        "model": "model",
        "input": "find tools",
        "tools": [{ "type": "tool_search" }],
        "tool_choice": { "type": "tool_search" }
    });

    let chat = responses_to_chat(&request, false).unwrap();

    assert_eq!(
        chat["tools"][0]["function"]["name"],
        "openai_compat_tool_search"
    );
    assert_eq!(
        chat["tools"][0]["function"]["parameters"]["required"][0],
        "query"
    );
    assert_eq!(
        chat["tool_choice"]["function"]["name"],
        "openai_compat_tool_search"
    );
}

#[test]
fn restores_tool_search_call_in_streaming_and_non_streaming_responses() {
    let request = json!({
        "tools": [{ "type": "tool_search" }]
    });
    let shape = ToolShape::from_request(&request);
    let chat = json!({
        "id": "chatcmpl-search",
        "choices": [{
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_search",
                    "type": "function",
                    "function": {
                        "name": "openai_compat_tool_search",
                        "arguments": "{\"query\":\"mail\",\"limit\":10}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    let response = chat_to_responses(&chat, "model", &shape).unwrap();
    assert_eq!(response["output"][0]["type"], "tool_search_call");
    assert_eq!(response["output"][0]["arguments"]["query"], "mail");
    assert_eq!(response["output"][0]["arguments"]["limit"], 10);

    let mut stream =
        ResponsesStream::with_shape("resp_search".to_string(), "model".to_string(), shape);
    let mut events = stream.push(&json!({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_search",
                    "function": {
                        "name": "openai_compat_tool_search",
                        "arguments": "{\"query\":\"mail\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    }));
    events.extend(stream.finish());
    let completed = terminal_response(&events, "response.completed");
    assert_eq!(
        completed["response"]["output"][0]["type"],
        "tool_search_call"
    );
    assert_eq!(
        completed["response"]["output"][0]["arguments"]["query"],
        "mail"
    );
}

#[test]
fn converts_paired_tool_search_history_to_function_call_and_tool_result() {
    let request = json!({
        "model": "model",
        "input": [
            {
                "type": "tool_search_call",
                "call_id": "call_search",
                "arguments": { "query": "mail" }
            },
            {
                "type": "tool_search_output",
                "call_id": "call_search",
                "tools": [{
                    "type": "function",
                    "name": "search_mail",
                    "parameters": { "type": "object", "properties": {} }
                }]
            }
        ]
    });

    let chat = responses_to_chat(&request, false).unwrap();
    let messages = chat["messages"].as_array().unwrap();

    assert_eq!(roles(messages), ["assistant", "tool"]);
    assert_eq!(
        messages[0]["tool_calls"][0]["function"]["name"],
        "openai_compat_tool_search"
    );
    assert_eq!(
        messages[0]["tool_calls"][0]["function"]["arguments"],
        "{\"query\":\"mail\"}"
    );
    assert_eq!(messages[1]["tool_call_id"], "call_search");
    assert_eq!(chat["tools"][0]["function"]["name"], "search_mail");
}

#[test]
fn drops_tool_controls_when_no_tool_is_sendable() {
    let request = json!({
        "model": "model",
        "input": "hello",
        "tools": [],
        "tool_choice": "required",
        "parallel_tool_calls": true
    });

    let chat = responses_to_chat(&request, false).unwrap();

    assert!(chat.get("tool_choice").is_none());
    assert!(chat.get("parallel_tool_calls").is_none());
}
