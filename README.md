# openai-compat

OpenAI Responses and Chat Completions protocol compatibility translation for
Rust.

This crate owns protocol conversion only. HTTP servers, authentication,
activation, retries and tracing stay in the host application.

## What it does

- Chat request preparation: force upstream streaming, normalize developer roles
  to `system`, and preserve original request bytes when no change is required.
- Chat streaming aggregation: combine SSE chunks into a standard
  `chat.completion` object for clients that requested a non-streaming response.
- Responses to Chat translation: messages, tools, media, reasoning, compaction
  history and usage fields.
- Chat to Responses translation: non-streaming JSON and streaming Responses SSE
  events, including namespaced and free-form tools.

## Git dependency

```toml
[dependencies]
openai-compat = { git = "https://github.com/xljiulang/openai-compat.git", rev = "<commit>" }
```

## Chat request preparation

```rust
use openai_compat::chat::prepare_chat_request_with;

let input = br#"{"model":"model","stream":false,"messages":[{"role":"developer","content":"rules"}]}"#;
let prepared = prepare_chat_request_with(input, |chat| {
    // Host-specific prompt/brand policy belongs here.
    chat["metadata"] = serde_json::json!({"host": "policy-applied"});
    Ok(true)
})?;
let upstream = prepared
    .rewritten_body()
    .expect("the callback changed the request");
assert!(!prepared.request().downstream_stream());
assert_eq!(prepared.role_changes().developer_to_system, 1);
```

`prepare_chat_request_with` runs the host policy callback, then applies role
normalization and encodes the final value. `PreparedChat` is read-only, so JSON
and upstream bytes cannot diverge. The original bytes are reused when neither
streaming, roles nor the callback require a rewrite.

## Chat stream aggregation

Feed complete SSE lines into `ChatCompletionAccumulator`:

```rust
use openai_compat::chat::{
    process_sse_line, ChatCompletionAccumulator,
};

let mut acc = ChatCompletionAccumulator::default();
// Or: ChatCompletionAccumulator::new(64 * 1024 * 1024)
process_sse_line(&mut acc, br#"data: {"choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":"stop"}]}"#)?;
process_sse_line(&mut acc, b"data: [DONE]")?;
let completion = acc.finish("model")?;
```

The aggregator validates completion markers, in-stream errors, tool arguments
and multiple choices. Hosts remain responsible for HTTP status and timeout
policy.

## Responses translation

```rust
use openai_compat::responses::prepare_responses;
use openai_compat::responses::TranslationOptions;

let prepared = prepare_responses(&request, true, TranslationOptions::default())?;
let chat_request = prepared.chat;
let shape = prepared.shape();
let ignored = prepared.ignored();
let unsupported = prepared.unsupported();
```

Use `prepare_responses` rather than separately calling `responses_to_chat` and
`ToolShape::from_request`; it performs tool collection once and returns the
request, reverse mapping and diagnostics together.

The reserved synthetic tool name is `openai_compat_tool_search`. An input
function with the same name is rejected rather than silently overwritten.

For non-streaming responses, pass the upstream Chat JSON and prepared tool shape
to `chat_to_responses`. For streaming responses, feed Chat chunks to
`ResponsesStream`. The output is SSE bytes ready for the host to forward.

## Extension hooks

Product-specific assistant text cleanup can be supplied through
`ResponseCleaner` when using `chat_to_responses_with_cleaner` or
`ResponsesStream::with_shape_and_cleaner`. Brand replacement, User-Agent policy
and other request gates remain in the host.

## Stability

The crate is pre-1.0. Public structs and errors that may gain fields or variants
are `non_exhaustive`. Prefer accessors over fields. Breaking API changes
increment the minor version.
The host pins a Git revision and upgrades deliberately.

## Not included

- HTTP routing, CORS, authentication and proxy lifecycle.
- CNB workspace activation, SSH tunnels and retries.
- PromptGate, User-Agent profiles and product-specific logging.

## Development

```powershell
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
cargo doc --no-deps
```
