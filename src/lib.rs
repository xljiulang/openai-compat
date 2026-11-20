//! OpenAI Responses and Chat Completions protocol compatibility translation.
//!
//! The crate owns JSON/SSE protocol conversion only. HTTP routing, retries,
//! authentication, tracing and product-specific prompt policies remain in the
//! host application.
//!
//! Start with [`chat::prepare_chat_request`] for Chat compatibility,
//! [`responses::prepare_responses`] for Responses translation, and
//! [`chat::ChatCompletionAccumulator`] for non-streaming Chat aggregation.
//!
//! Public structs that may evolve are non-exhaustive; use accessors where
//! available.

pub mod chat;
pub mod responses;

/// Default limit for a complete non-streaming response body.
pub const DEFAULT_MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

/// Machine-readable category for a compatibility failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ErrorKind {
    InvalidInput,
    Upstream,
    IncompleteStream,
    LimitExceeded,
    NameConflict,
}

/// Unified error produced by the compatibility crate.
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

    pub fn with_param(mut self, param: impl Into<String>) -> Self {
        self.param = Some(param.into());
        self
    }

    pub fn with_retryable(mut self, retryable: bool) -> Self {
        self.retryable = retryable;
        self
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    pub fn code(&self) -> &str {
        &self.code
    }

    pub fn error_type(&self) -> &str {
        &self.error_type
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn param(&self) -> Option<&str> {
        self.param.as_deref()
    }

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
