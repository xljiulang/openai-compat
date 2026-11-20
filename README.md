# openai-compat

OpenAI 接口兼容翻译库。

`openai-compat` 是独立维护的 Rust 库，只解决一件事：把 OpenAI Responses API
的请求翻译成 Chat Completions，再把上游 Chat 响应翻译回 Responses。它不隶属
OpenAI，也不是官方项目，不包含任何网关、服务端或产品逻辑。

## 它解决什么问题

客户端说的是 Responses，上游只认 Chat Completions。两边的请求字段、消息角色、
工具声明、流式事件和错误结构都不一样，直接转发 JSON 通常会在某个细节上失败：
`developer` 角色、`stream: false`、命名空间工具、tool call 参数的增量拼接、
`usage` 字段命名，以及 `[DONE]` 之后的收尾。

这个库提供的正是这一层「翻译」：无状态的 JSON/SSE 转换函数，加上需要跨分片
保留状态的小型翻译器。它不监听端口，不做鉴权、路由、重试或日志，也不替调用方
决定提示词和品牌策略。你把字节读进来，把字节写出去，中间交给它。

翻译的原则是**只补不改**：下游没表达的字段不凭空添加，下游已经给出的值原样
保留，只在协议确实要求、而下游无法表达的地方补齐。

## 核心能力

- **Chat 请求准备**：强制上游流式、把 `developer` 角色归一化为 `system`，
  无改动时复用原始请求字节。
- **Chat 流式聚合**：把 SSE 分片合成完整的 `chat.completion`，校验 `[DONE]`、
  流内错误、tool call 参数和多个 choice。
- **Responses → Chat**：messages、tools、媒体、reasoning、compaction 历史与
  usage 映射，含命名空间工具和自由格式工具；流式请求自动声明
  `stream_options.include_usage`。
- **Chat → Responses**：非流式 JSON 与流式 SSE 事件，含错误归一化和文本清洗
  钩子；`openai_compat_tool_search` 为保留工具名。

## 快速开始

本库暂未发布到 crates.io（`publish = false`），通过 Git 依赖并按 commit 固定：

```toml
[dependencies]
openai-compat = { git = "https://github.com/xljiulang/openai-compat.git", rev = "<commit>" }
```

### Chat 请求准备

```rust
use openai_compat::chat::prepare_chat_request;

let input = br#"{"model":"model","stream":false,"messages":[{"role":"developer","content":"rules"}]}"#;
let prepared = prepare_chat_request(input);

let upstream = prepared.rewritten_body().expect("body was rewritten");
assert!(!prepared.request().downstream_stream());
assert_eq!(prepared.role_changes().developer_to_system, 1);
```

需要插入自己的提示词或改写字段时改用 `prepare_chat_request_with`。回调在强制
流式之后、角色归一化之前运行，返回的 `PreparedChat` 是只读的，JSON 视图与
上游字节不会分叉：

```rust
use openai_compat::chat::prepare_chat_request_with;

// input 沿用上一段；回调返回 Ok(true) 表示确实改过请求
let prepared = prepare_chat_request_with(input, |chat| {
    chat["metadata"] = serde_json::json!({"source": "my-app"});
    Ok(true)
})?;
```

### Chat 流式聚合

客户端要求非流式响应时，把上游 SSE 逐行喂进聚合器，得到标准
`chat.completion` 对象：

```rust
use openai_compat::chat::{process_sse_line, ChatCompletionAccumulator};

let mut acc = ChatCompletionAccumulator::default(); // 或 new(64 * 1024 * 1024)
process_sse_line(&mut acc, br#"data: {"choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":"stop"}]}"#)?;
process_sse_line(&mut acc, b"data: [DONE]")?;
let completion = acc.finish("model")?;
```

上面的 `?` 表示在返回 `Result` 的函数中调用。缺少完成分片或 `[DONE]`、流内
报错、tool call 参数不完整时会返回 `IncompleteStream` 或 `Upstream`，HTTP
状态与超时仍由调用方决定。

### Responses 翻译

请求方向用 `prepare_responses` 一次拿到上游 body、工具反查表和诊断信息：

```rust
use openai_compat::responses::{prepare_responses, TranslationOptions};

let request: serde_json::Value = serde_json::from_slice(&body)?; // 客户端原始 Responses 请求
let prepared = prepare_responses(&request, true, TranslationOptions::default())?;
let chat_request = prepared.chat();      // 发给上游 Chat Completions
let shape = prepared.shape().clone();    // 响应翻译时反查工具名
let ignored = prepared.ignored();        // 接受但未能透传的字段
let unsupported = prepared.unsupported();
```

响应方向按上游是否流式二选一：

```rust
use openai_compat::responses::{chat_to_responses, ResponsesStream};

// 非流式：上游 Chat JSON -> Responses 对象
let response = chat_to_responses(&chat_json, "model", &shape)?;

// 流式：上游 Chat chunk -> Responses SSE 字节
let mut stream = ResponsesStream::with_shape("resp_local".into(), "model".into(), shape);
for chunk in chunks {
    for event in stream.push(&chunk) {
        // 逐条写回客户端
        let _ = event;
    }
}
let tail = stream.finish();
```

`shape` 需要贯穿同一次请求的请求翻译与响应翻译，否则命名空间工具和自由格式
工具的回调名无法还原。

`want_stream` 为 `true` 时，上游 body 一定带 `stream_options`，默认
`{"include_usage": true}`。下游自己传了 `stream_options` 就原样保留，只有
`include_usage` 缺失或为 `null` 时才补上 `true`；非流式请求不写这个字段。

## 关键契约

- 所有失败都返回统一的 `CompatError`，用 `ErrorKind` 区分
  `InvalidInput`、`Upstream`、`IncompleteStream`、`LimitExceeded`、
  `NameConflict`；`code` / `error_type` / `param` / `retryable` 可直接映射到
  客户端错误体。
- 聚合器默认上限为 `DEFAULT_MAX_RESPONSE_BYTES`（64 MiB），可用
  `ChatCompletionAccumulator::new(max_bytes)` 覆盖。
- `ResponsesStream` 有同样的默认上限，可用 `.with_max_bytes(..)` 链式调整；
  超限时以 `response_too_large` 失败，而不是静默截断。
- 请求体不是合法 JSON 对象时不重写、不报错，原样透传，交由上游判断。
- 工具名超过 Chat 的 64 字节限制时按命名空间扁平化；与保留名冲突时返回
  `NameConflict`，不会静默覆盖。
- `usage` 只做映射，不做索取之外的推断：上游返回就翻译，上游没返回就输出全
  零（Responses 客户端会无条件读取这些计数）。

## 不做什么

本库只做协议转换。以下内容全部属于调用方：

- HTTP 路由、CORS、鉴权、超时、重试和 tracing。
- 提示词注入、品牌替换、User-Agent 策略等业务规则。
- 日志、指标、运营与部署逻辑。
- 反向路径（Chat 客户端 → Responses 上游）的翻译，目前尚未实现。

它同样不依赖、也不假设某个特定宿主存在：任何 Rust 程序都可以把它当作普通
依赖使用。

## 扩展钩子与版本策略

若需要对助手文本做产品级清理（例如去除特定标记），通过 `ResponseCleaner`
接入 `chat_to_responses_with_cleaner` 或
`ResponsesStream::with_shape_and_cleaner`。它是
`Arc<dyn Fn(&str) -> String + Send + Sync>`，可以捕获请求级配置，会在非流式
内容以及每个流式文本增量上被调用。

当前版本为 pre-1.0。可能新增字段或变体的公共结构与枚举标记为
`#[non_exhaustive]`，请优先使用访问器而不是直接读写字段；破坏性变更会提升
minor 版本。

## 0.2.0 迁移要点

- `ResponseCleaner` 从函数指针改为 `Arc<dyn Fn(&str) -> String + Send + Sync>`，
  调用处包一层 `Arc::new(...)` 即可。
- `PreparedResponses` 的字段改为私有，改用 `chat()` 读取、`into_chat()` 取走。
- 流式 Responses 请求会带 `stream_options.include_usage: true`，下游显式给出
  的值不受影响。
- 上游返回的流不是合法 JSON 时，错误从 `InvalidInput` 归为 `Upstream`。

## 开发

```powershell
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
cargo doc --no-deps
```

## 许可证

MIT，详见 [LICENSE](LICENSE)。
