use super::*;

#[test]
fn preparation_forces_upstream_stream_and_keeps_client_intent() {
    let input = br#"{"model":"deepseek-v4","stream":false}"#;
    let prepared = prepare_chat_request(input);
    let value = prepared
        .value
        .as_ref()
        .expect("valid JSON object should be parsed");

    assert_eq!(value["stream"], true);
    assert!(!prepared.request.downstream_stream);
    assert_eq!(prepared.request.requested_model, "deepseek-v4");
    assert!(prepared.body_changed);
    let body: Value = serde_json::from_slice(&prepared.rewritten_body().unwrap()).unwrap();
    assert_eq!(body["stream"], true);
}

#[test]
fn preparation_defaults_missing_stream_to_non_streaming() {
    let input = br#"{"model":"model"}"#;
    let prepared = prepare_chat_request(input);
    let value = prepared.value.as_ref().unwrap();

    assert_eq!(value["stream"], true);
    assert!(!prepared.request.downstream_stream);
    assert!(prepared.body_changed);
}

#[test]
fn preparation_preserves_original_streaming_intent_and_non_json_body() {
    let input = br#"{"model":"model","stream":true}"#;
    let prepared = prepare_chat_request(input);
    assert_eq!(prepared.value.as_ref().unwrap()["stream"], true);
    assert!(prepared.request.downstream_stream);
    assert!(!prepared.body_changed);

    let prepared = prepare_chat_request(b"not json");
    assert!(!prepared.request.valid_json_object);
    assert!(!prepared.request.downstream_stream);
    assert!(prepared.rewritten_body().is_none());
    assert!(prepared.value.is_none());
    assert!(!prepared.body_changed);
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
