use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatErrorInfo {
    pub message: String,
    pub error_type: String,
    pub code: Option<String>,
    pub param: Option<String>,
}

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

pub fn normalize_chat_error_text(text: &str) -> ChatErrorInfo {
    serde_json::from_str::<Value>(text)
        .ok()
        .map(|value| normalize_chat_error(Some(&value)))
        .unwrap_or_else(|| ChatErrorInfo {
            message: if text.is_empty() {
                "Upstream returned an empty error response".to_string()
            } else {
                text.to_string()
            },
            error_type: "upstream_error".to_string(),
            code: None,
            param: None,
        })
}

fn value_to_code(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}
