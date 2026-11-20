use serde_json::Value;

/// Upstream Chat error fields normalized into the Responses error shape.
///
/// The payload dialects differ between providers: some nest the fields under
/// `error`, some send `detail` or `base_resp`, and some return a bare string.
/// This type is the single decoded shape the crate maps from.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct ChatErrorInfo {
    /// Human-readable error message.
    pub message: String,
    /// OpenAI-style error type, or `upstream_error` when the upstream omits it.
    pub error_type: String,
    /// Machine-readable error code, when the upstream reports one.
    pub code: Option<String>,
    /// Request parameter the error points at, when reported.
    pub param: Option<String>,
}

/// Normalizes an upstream Chat error payload.
///
/// `None` produces a placeholder describing an empty error response, so callers
/// never have to synthesize a message themselves.
#[must_use]
pub fn normalize_chat_error(value: Option<&Value>) -> ChatErrorInfo {
    let Some(value) = value else {
        return ChatErrorInfo {
            message: "Upstream returned an empty error response".to_string(),
            error_type: "upstream_error".to_string(),
            code: None,
            param: None,
        };
    };
    let source = value.get("error").unwrap_or(value);
    let message = source
        .get("message")
        .or_else(|| source.get("detail"))
        .or_else(|| source.get("status_msg"))
        .or_else(|| source.pointer("/base_resp/status_msg"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| source.as_str().map(str::to_string))
        .unwrap_or_else(|| {
            serde_json::to_string(source).unwrap_or_else(|_| "Upstream error".to_string())
        });
    let error_type = source
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("upstream_error")
        .to_string();
    let mut code = source
        .get("code")
        .or_else(|| source.pointer("/base_resp/status_code"))
        .map(value_to_code);
    if message
        .to_ascii_lowercase()
        .contains("input token count exceeds the maximum number of tokens allowed")
        || message.to_ascii_lowercase().contains("context length")
    {
        code = Some("context_length_exceeded".to_string());
    }
    let param = source
        .get("param")
        .and_then(Value::as_str)
        .filter(|param| !param.is_empty())
        .map(str::to_string);
    ChatErrorInfo {
        message,
        error_type,
        code,
        param,
    }
}

/// Normalizes an upstream Chat error from raw response text.
///
/// Text that is not JSON is kept verbatim as the message.
#[must_use]
pub fn normalize_chat_error_text(text: &str) -> ChatErrorInfo {
    serde_json::from_str::<Value>(text).ok().map_or_else(
        || ChatErrorInfo {
            message: if text.is_empty() {
                "Upstream returned an empty error response".to_string()
            } else {
                text.to_string()
            },
            error_type: "upstream_error".to_string(),
            code: None,
            param: None,
        },
        |value| normalize_chat_error(Some(&value)),
    )
}

fn value_to_code(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}
