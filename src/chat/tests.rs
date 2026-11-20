use super::*;

#[test]
fn preparation_forces_upstream_stream_and_keeps_client_intent() {
    let input = br#"{"model":"deepseek-v4","stream":false}"#;
    let prepared = prepare_chat_request(input);
    let value = prepared
        .value()
        .expect("valid JSON object should be parsed");

    assert_eq!(value["stream"], true);
    assert!(!prepared.request().downstream_stream());
    assert_eq!(prepared.request().requested_model(), "deepseek-v4");
    assert!(prepared.body_changed());
    let body: Value = serde_json::from_slice(&prepared.rewritten_body().unwrap()).unwrap();
    assert_eq!(body["stream"], true);
}

#[test]
fn preparation_defaults_missing_stream_to_non_streaming() {
    let input = br#"{"model":"model"}"#;
    let prepared = prepare_chat_request(input);
    let value = prepared.value().unwrap();

    assert_eq!(value["stream"], true);
    assert!(!prepared.request().downstream_stream());
    assert!(prepared.body_changed());
}

#[test]
fn preparation_preserves_original_streaming_intent_and_non_json_body() {
    let input = br#"{"model":"model","stream":true}"#;
    let prepared = prepare_chat_request(input);
    assert_eq!(prepared.value().unwrap()["stream"], true);
    assert!(prepared.request().downstream_stream());
    assert!(!prepared.body_changed());

    let prepared = prepare_chat_request(b"not json");
    assert!(!prepared.request().valid_json_object());
    assert!(!prepared.request().downstream_stream());
    assert!(prepared.rewritten_body().is_none());
    assert!(prepared.value().is_none());
    assert!(!prepared.body_changed());
}

#[test]
fn preparation_normalizes_developer_roles_by_default() {
    let input = br#"{"model":"model","stream":true,"messages":[{"role":"developer","content":"rules"},{"role":"DEVELOPER","content":"more"},{"role":"user","content":"hello"},{"role":"assistant","content":"ok"},{"role":"tool","content":"result"}]}"#;
    let prepared = prepare_chat_request(input);
    let value = prepared.value().unwrap();

    assert_eq!(prepared.role_changes().developer_to_system, 2);
    assert!(prepared.body_changed());
    assert_eq!(value["messages"][0]["role"], "system");
    assert_eq!(value["messages"][1]["role"], "system");
    assert_eq!(value["messages"][0]["content"], "rules");
    assert_eq!(value["messages"][1]["content"], "more");
    assert_eq!(value["messages"][2]["role"], "user");
    assert_eq!(value["messages"][3]["role"], "assistant");
    assert_eq!(value["messages"][4]["role"], "tool");
    let upstream = serde_json::from_slice::<Value>(&prepared.rewritten_body().unwrap()).unwrap();
    assert_eq!(upstream["messages"][0]["role"], "system");
    assert_eq!(upstream["messages"][1]["role"], "system");
}

#[test]
fn preparation_reuses_bytes_when_roles_are_already_normalized() {
    let input =
        br#"{"model":"model","stream":true,"messages":[{"role":"system","content":"rules"}]}"#;
    let prepared = prepare_chat_request(input);

    assert_eq!(prepared.role_changes(), RoleChanges::default());
    assert!(!prepared.body_changed());
    assert!(prepared.rewritten_body().is_none());
}

#[test]
fn request_mutator_runs_before_final_encoding() {
    let input =
        br#"{"model":"model","stream":true,"messages":[{"role":"developer","content":"rules"}]}"#;
    let prepared = prepare_chat_request_with(input, |value| {
        value["messages"][0]["content"] = Value::String("rewritten rules".to_string());
        Ok(true)
    })
    .unwrap();

    assert_eq!(prepared.role_changes().developer_to_system, 1);
    assert!(prepared.body_changed());
    let body: Value = serde_json::from_slice(&prepared.rewritten_body().unwrap()).unwrap();
    assert_eq!(body["stream"], true);
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(body["messages"][0]["content"], "rewritten rules");
}

#[test]
fn chat_accumulator_enforces_its_configured_limit() {
    let mut accumulator = ChatCompletionAccumulator::new(16);
    let error = accumulator
        .push_data(r#"{"choices":[{"index":0}]}"#)
        .unwrap_err();

    assert_eq!(error.kind(), crate::ErrorKind::LimitExceeded);
    assert!(error.message().contains("configured limit"));
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

#[test]
fn malformed_upstream_json_is_an_upstream_error() {
    let mut accumulator = ChatCompletionAccumulator::default();
    let error = process_sse_line(&mut accumulator, b"data: {not json").unwrap_err();

    assert_eq!(error.kind(), crate::ErrorKind::Upstream);
    assert!(
        error.message().is_ascii(),
        "error message should stay in English: {}",
        error.message()
    );
}

#[test]
fn incomplete_stream_messages_are_english() {
    let accumulator = ChatCompletionAccumulator::default();
    let error = accumulator.finish("model").unwrap_err();

    assert_eq!(error.kind(), crate::ErrorKind::IncompleteStream);
    assert!(
        error.message().is_ascii(),
        "error message should stay in English: {}",
        error.message()
    );
}
