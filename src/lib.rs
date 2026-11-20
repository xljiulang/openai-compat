//! OpenAI Responses and Chat Completions protocol compatibility translation.
//!
//! This crate does one thing: translate between the OpenAI Responses API and
//! Chat Completions. It listens on no port, performs no authentication, routing,
//! retries or logging, and assumes no particular host application.
//!
//! The supported direction is a Responses client talking to a Chat Completions
//! upstream. Requests are translated forward with [`responses::prepare_responses`],
//! and upstream Chat responses come back through
//! [`responses::chat_to_responses`] or [`responses::ResponsesStream`].
//! The opposite direction (a Chat client behind a Responses upstream) is not
//! implemented.
//!
//! Start with [`chat::prepare_chat_request`] for Chat compatibility,
//! [`responses::prepare_responses`] for Responses translation, and
//! [`chat::ChatCompletionAccumulator`] for non-streaming Chat aggregation.
//!
//! Public structs that may evolve are non-exhaustive; use accessors where
//! available.
//!
//! # Chat request preparation
//!
//! ```
//! use openai_compat::chat::prepare_chat_request;
//!
//! let input = br#"{"model":"model","stream":false,"messages":[{"role":"developer","content":"rules"}]}"#;
//! let prepared = prepare_chat_request(input);
//!
//! let upstream = prepared.rewritten_body().expect("body was rewritten");
//! assert!(!prepared.request().downstream_stream());
//! assert_eq!(prepared.role_changes().developer_to_system, 1);
//! # let _ = upstream;
//! ```
//!
//! When a caller needs to add prompt material, [`chat::prepare_chat_request_with`]
//! mutates the parsed request before roles are normalized, so the JSON view and
//! the upstream bytes cannot diverge.
//!
//! # Chat stream aggregation
//!
//! Clients that asked for a non-streaming response are served by folding the
//! upstream SSE lines into a standard `chat.completion` object:
//!
//! ```
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! use openai_compat::chat::{process_sse_line, ChatCompletionAccumulator};
//!
//! let mut acc = ChatCompletionAccumulator::default();
//! process_sse_line(&mut acc, br#"data: {"choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":"stop"}]}"#)?;
//! process_sse_line(&mut acc, b"data: [DONE]")?;
//! let completion = acc.finish("model")?;
//! # let _ = completion;
//! # Ok(())
//! # }
//! # run().unwrap();
//! ```
//!
//! # Responses translation
//!
//! ```
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! use openai_compat::responses::{prepare_responses, TranslationOptions};
//!
//! let request: serde_json::Value =
//!     serde_json::from_str(r#"{"model":"model","input":"hi"}"#)?;
//! let prepared = prepare_responses(&request, true, TranslationOptions::default())?;
//!
//! let chat_request = prepared.chat();     // sent to the Chat upstream
//! let shape = prepared.shape().clone();   // reverse tool map for the response
//! let ignored = prepared.ignored();       // accepted but not carried upstream
//! let unsupported = prepared.unsupported();
//! # let _ = (chat_request, shape, ignored, unsupported);
//! # Ok(())
//! # }
//! # run().unwrap();
//! ```
//!
//! The same [`responses::ToolShape`] must be reused when translating the
//! upstream response, otherwise namespaced and free-form tool calls cannot be
//! restored.
//!
//! # Usage accounting
//!
//! A streaming Responses request always declares `stream_options.include_usage`
//! for the upstream, because usage is part of the Responses payload the client
//! expects and Chat only reports it on request. A caller-supplied value is kept
//! as-is; only a missing or null `include_usage` is filled in. Upstream usage is
//! mapped when present; when the upstream still omits it, Responses reports
//! zeroes rather than dropping the field.

#![deny(missing_docs)]

pub mod chat;
pub mod responses;

/// Default limit for a complete non-streaming response body.
pub const DEFAULT_MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

/// Machine-readable category for a compatibility failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorKind {
    /// The caller supplied a request this crate cannot translate.
    InvalidInput,
    /// The upstream answered with an error or a malformed payload.
    Upstream,
    /// The upstream stream ended before it was complete.
    IncompleteStream,
    /// A payload or stream exceeded its configured byte limit.
    LimitExceeded,
    /// A caller-supplied name collides with a name this crate reserves.
    NameConflict,
}

/// Unified error produced by the compatibility crate.
///
/// The fields mirror the OpenAI error body so a host can forward them to the
/// client with minimal reshaping.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct CompatError {
    kind: ErrorKind,
    code: String,
    error_type: String,
    message: String,
    param: Option<String>,
    retryable: bool,
}

impl CompatError {
    /// Builds an error from its category and OpenAI-shaped fields.
    pub fn new(
        kind: ErrorKind,
        code: impl Into<String>,
        error_type: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            code: code.into(),
            error_type: error_type.into(),
            message: message.into(),
            param: None,
            retryable: false,
        }
    }

    /// Records the request parameter the failure points at.
    pub fn with_param(mut self, param: impl Into<String>) -> Self {
        self.param = Some(param.into());
        self
    }

    /// Marks whether retrying the same request may succeed.
    pub fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }

    /// Returns the failure category.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// Returns the machine-readable error code.
    pub fn code(&self) -> &str {
        &self.code
    }

    /// Returns the OpenAI-style error type.
    pub fn error_type(&self) -> &str {
        &self.error_type
    }

    /// Returns the human-readable message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns the request parameter this failure points at, when known.
    pub fn param(&self) -> Option<&str> {
        self.param.as_deref()
    }

    /// Reports whether retrying the same request may succeed.
    pub fn retryable(&self) -> bool {
        self.retryable
    }
}

impl std::fmt::Display for CompatError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for CompatError {}

/// Backward-compatible alias for the previous Chat-specific error name.
pub type ChatError = CompatError;
