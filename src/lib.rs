//! OpenAI Responses and Chat Completions protocol compatibility translation.
//!
//! The crate owns JSON/SSE protocol conversion only. HTTP routing, retries,
//! authentication, tracing and product-specific prompt policies remain in the
//! host application.

pub mod chat;
pub mod responses;

/// Error produced by the Chat Completions compatibility path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatError {
    pub code: String,
    pub error_type: String,
    pub message: String,
    pub param: Option<String>,
}

impl ChatError {
    pub fn new(
        code: impl Into<String>,
        error_type: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code: code.into(),
            error_type: error_type.into(),
            message: message.into(),
            param: None,
        }
    }
}
