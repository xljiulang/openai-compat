use super::*;

/// Schema name used when the client asks for structured output without naming it.
const DEFAULT_SCHEMA_NAME: &str = "openai_compat_output_schema";

/// Chat Completions rejects `response_format.json_schema.name` over 64 bytes.
const MAX_SCHEMA_NAME_BYTES: usize = 64;

/// Prompt that replaces a `compaction_trigger` item.
pub(super) const COMPACTION_PROMPT: &str = "\
You are compacting this conversation for an AI coding agent that will continue the work.
Write a dense handoff note covering: the user's goals and constraints, decisions made, \
files and paths touched, commands run and their outcomes, what remains to be done, and \
pitfalls to avoid.
Keep exact identifiers, paths and error strings. Do not call tools. Do not ask questions. \
Output only the note.";

/// Stand-in for a compaction checkpoint this proxy cannot read.
pub(super) const COMPACTION_PLACEHOLDER: &str = "\
[The client compacted the earlier conversation into a checkpoint this proxy cannot read, \
so the summary text is unavailable. Rely on the messages that remain.]";

/// Item types that only exist to carry a compaction checkpoint.
const COMPACTION_ITEM_TYPES: [&str; 3] = ["compaction", "compaction_summary", "context_compaction"];
const EXTENDED_CHAT_FIELDS: [&str; 9] = [
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "metadata",
    "n",
    "presence_penalty",
    "stop",
    "top_logprobs",
    "user",
];
pub(super) const DEGRADED_INPUT_TYPES: [&str; 7] = [
    "local_shell_call",
    "local_shell_call_output",
    "computer_call",
    "computer_call_output",
    "mcp_call",
    "mcp_list_tools",
    "web_search_call",
];

#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct TranslationOptions {
    supports_file_input: bool,
    supports_audio_input: bool,
}

impl TranslationOptions {
    /// Declares which media inputs the target model accepts.
    pub fn new(supports_file_input: bool, supports_audio_input: bool) -> Self {
        Self {
            supports_file_input,
            supports_audio_input,
        }
    }

    pub fn supports_file_input(self) -> bool {
        self.supports_file_input
    }

    pub fn supports_audio_input(self) -> bool {
        self.supports_audio_input
    }
}

/// A fully prepared Responses request.
///
/// The upstream Chat body, reverse tool mapping, diagnostics and unsupported
/// feature list all come from the same parse and tool collection.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PreparedResponses {
    pub chat: Value,
    pub shape: ToolShape,
    pub ignored: Vec<String>,
    pub unsupported: Vec<&'static str>,
}

impl PreparedResponses {
    /// Returns the reverse tool mapping required by response translation.
    pub fn shape(&self) -> &ToolShape {
        &self.shape
    }

    /// Returns accepted Responses features that could not be carried upstream.
    pub fn ignored(&self) -> &[String] {
        &self.ignored
    }

    /// Returns accepted but unsupported Responses feature names.
    pub fn unsupported(&self) -> &[&'static str] {
        &self.unsupported
    }
}

/// Translates a Responses request into a Chat Completions request.
pub fn responses_to_chat(request: &Value, want_stream: bool) -> Result<Value, String> {
    responses_to_chat_with_options(request, want_stream, TranslationOptions::default())
}

pub fn responses_to_chat_with_options(
    request: &Value,
    want_stream: bool,
    options: TranslationOptions,
) -> Result<Value, String> {
    let declared = collect_declared_tools(request);
    let shape = ToolShape::from_declared(&declared.tools);
    responses_to_chat_with_parts(request, want_stream, options, &declared, &shape)
}

fn responses_to_chat_with_parts(
    request: &Value,
    want_stream: bool,
    options: TranslationOptions,
    declared: &DeclaredTools,
    shape: &ToolShape,
) -> Result<Value, String> {
    if !request.is_object() {
        return Err("request body must be a JSON object".to_string());
    }
    // A compaction turn is a summarization request: it must not be able to call
    // tools, so this decision gates the tool fields below.
    let compaction = is_compaction_turn(request);
    let mut messages = Vec::new();
    if let Some(instructions) = request
        .get("instructions")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
    {
        messages.push(json!({ "role": "system", "content": instructions }));
    }

    match request.get("input") {
        Some(Value::String(text)) if !text.is_empty() => {
            messages.push(json!({ "role": "user", "content": text }));
        }
        Some(Value::Array(items)) => {
            // Compaction items are rewritten before translation so the message
            // state machine never has to know about them.
            let rewritten = rewrite_compaction_items(items);
            translate_input_items(
                rewritten.as_deref().unwrap_or(items),
                &mut messages,
                options,
            );
        }
        _ => {}
    }
    preserve_invalid_tool_calls_as_text(&mut messages);
    dedupe_function_call_arguments(&mut messages);
    messages = collapse_system_messages_to_head(messages);
    drop_empty_user_messages(&mut messages);
    repair_tool_pairing(&mut messages);

    let mut chat = json!({
        "model": request.get("model").cloned().unwrap_or(Value::Null),
        "messages": messages,
        "stream": want_stream
    });
    if !declared.tools.is_empty() {
        let translated = translate_tools(&declared.tools);
        if !translated.is_empty() {
            chat["tools"] = Value::Array(translated);
        }
    }
    for (destination, source) in [
        ("max_tokens", "max_output_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("seed", "seed"),
    ] {
        if let Some(value) = request.get(source).filter(|value| !value.is_null()) {
            chat[destination] = value.clone();
        }
    }
    if let Some(effort) = request
        .pointer("/reasoning/effort")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        chat["reasoning_effort"] = json!(effort);
    }
    if !compaction {
        if let Some(value) = request
            .get("parallel_tool_calls")
            .filter(|value| !value.is_null())
        {
            chat["parallel_tool_calls"] = value.clone();
        }
    }
    for field in ["prompt_cache_key", "service_tier"] {
        if let Some(value) = request.get(field).filter(|value| !value.is_null()) {
            chat[field] = value.clone();
        }
    }
    for field in EXTENDED_CHAT_FIELDS {
        if let Some(value) = request.get(field).filter(|value| !value.is_null()) {
            chat[field] = value.clone();
        }
    }
    if !compaction {
        if let Some(choice) = translate_tool_choice(request.get("tool_choice"), shape) {
            chat["tool_choice"] = choice;
        }
        if let Some(format) = translate_text_format(request) {
            chat["response_format"] = format;
        }
    }
    if !chat
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty())
    {
        if let Some(object) = chat.as_object_mut() {
            object.remove("tool_choice");
            object.remove("parallel_tool_calls");
        }
    }
    let _ = crate::chat::normalize_roles(&mut chat);
    Ok(chat)
}

/// Builds the upstream Chat request, reverse tool shape and diagnostics once.
///
/// Hosts that translate a Responses request should prefer this entry point over
/// calling [`responses_to_chat_with_options`] and [`ToolShape::from_request`]
/// separately.
pub fn prepare_responses(
    request: &Value,
    want_stream: bool,
    options: TranslationOptions,
) -> Result<PreparedResponses, String> {
    let declared = collect_declared_tools(request);
    let shape = ToolShape::from_declared(&declared.tools);
    let chat = responses_to_chat_from_declared(request, want_stream, options, &declared, &shape)?;
    let ignored =
        crate::responses::diagnostics::ignored_feature_details_with_options(request, options);
    let unsupported = unsupported_feature_names(request);
    Ok(PreparedResponses {
        chat,
        shape,
        ignored,
        unsupported,
    })
}

fn responses_to_chat_from_declared(
    request: &Value,
    want_stream: bool,
    options: TranslationOptions,
    declared: &DeclaredTools,
    shape: &ToolShape,
) -> Result<Value, String> {
    responses_to_chat_with_parts(request, want_stream, options, declared, shape)
}

/// Whether this request asks for a compaction checkpoint.
///
/// `collect_declared_tools` returns nothing for such a turn, so neither the
/// upstream tool list nor the reverse name map can leak tools into a summary.
pub(super) fn is_compaction_turn(request: &Value) -> bool {
    request
        .get("input")
        .and_then(Value::as_array)
        .is_some_and(|items| {
            items
                .iter()
                .any(|item| item.get("type").and_then(Value::as_str) == Some("compaction_trigger"))
        })
}

/// Replaces compaction items with plain messages.
///
/// Returns `None` when the history holds no compaction item, so the common case
/// does not clone the input array.
fn rewrite_compaction_items(items: &[Value]) -> Option<Vec<Value>> {
    let needs_rewrite = items.iter().any(|item| {
        matches!(
            item.get("type").and_then(Value::as_str),
            Some("compaction_trigger")
        ) || COMPACTION_ITEM_TYPES.contains(&item.get("type").and_then(Value::as_str).unwrap_or(""))
    });
    if !needs_rewrite {
        return None;
    }
    Some(
        items
            .iter()
            .map(|item| match item.get("type").and_then(Value::as_str) {
                Some("compaction_trigger") => user_text_message(COMPACTION_PROMPT),
                Some(name) if COMPACTION_ITEM_TYPES.contains(&name) => {
                    let text = compaction_summary_text(item)
                        .unwrap_or_else(|| COMPACTION_PLACEHOLDER.to_string());
                    user_text_message(&text)
                }
                _ => item.clone(),
            })
            .collect(),
    )
}

/// Reads the summary body out of a compaction item when the client kept it in
/// plain text. Encrypted checkpoints are unreadable by design.
fn compaction_summary_text(item: &Value) -> Option<String> {
    for key in ["summary", "text"] {
        if let Some(text) = item
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|text| !text.is_empty())
        {
            return Some(text.to_string());
        }
    }
    let parts = item.get("content")?.as_array()?;
    let text = parts
        .iter()
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n");
    (!text.is_empty()).then_some(text)
}

fn user_text_message(text: &str) -> Value {
    json!({
        "type": "message",
        "role": "user",
        "content": [{ "type": "input_text", "text": text }]
    })
}

/// Classifies the Responses `text.format` field.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum TextFormatKind {
    /// No structured output requested.
    Absent,
    /// Plain text, the default; nothing is sent upstream.
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        schema: Value,
        strict: Option<bool>,
    },
    /// A format this translation cannot carry; `details()` explains why.
    Unknown(String),
}

impl TextFormatKind {
    pub(super) fn details(&self) -> Option<String> {
        match self {
            Self::Unknown(reason) => Some(format!("text.format.type:{reason}")),
            _ => None,
        }
    }
}

/// Reads `text.format`, accepting both the flat Responses shape and the nested
/// Chat shape, because the client's exact serialization is not observable here.
pub(super) fn text_format_kind(request: &Value) -> TextFormatKind {
    let Some(format) = request
        .pointer("/text/format")
        .filter(|value| value.is_object())
    else {
        return TextFormatKind::Absent;
    };
    match format.get("type").and_then(Value::as_str) {
        Some("text") => TextFormatKind::Text,
        Some("json_object") => TextFormatKind::JsonObject,
        Some("json_schema") => {
            let nested = format.get("json_schema");
            let Some(schema) = format
                .get("schema")
                .filter(|value| value.is_object())
                .or_else(|| {
                    nested
                        .and_then(|nested| nested.get("schema"))
                        .filter(|value| value.is_object())
                })
            else {
                return TextFormatKind::Unknown("json_schema_without_schema".to_string());
            };
            let name = format
                .get("name")
                .and_then(Value::as_str)
                .or_else(|| {
                    nested
                        .and_then(|nested| nested.get("name"))
                        .and_then(Value::as_str)
                })
                .map(sanitize_schema_name)
                .filter(|name| !name.is_empty())
                .unwrap_or_else(|| DEFAULT_SCHEMA_NAME.to_string());
            let strict = format.get("strict").and_then(Value::as_bool).or_else(|| {
                nested
                    .and_then(|nested| nested.get("strict"))
                    .and_then(Value::as_bool)
            });
            TextFormatKind::JsonSchema {
                name,
                schema: schema.clone(),
                strict,
            }
        }
        Some(other) => TextFormatKind::Unknown(other.to_string()),
        None => TextFormatKind::Unknown("missing_type".to_string()),
    }
}

fn translate_text_format(request: &Value) -> Option<Value> {
    match text_format_kind(request) {
        TextFormatKind::JsonObject => Some(json!({ "type": "json_object" })),
        TextFormatKind::JsonSchema {
            name,
            schema,
            strict,
        } => {
            // The schema is forwarded verbatim: a response schema may have any
            // root type, so the tool-parameter normalization must not run here.
            let mut json_schema = json!({ "name": name, "schema": schema });
            if let Some(strict) = strict {
                json_schema["strict"] = json!(strict);
            }
            Some(json!({ "type": "json_schema", "json_schema": json_schema }))
        }
        TextFormatKind::Absent | TextFormatKind::Text | TextFormatKind::Unknown(_) => None,
    }
}

fn sanitize_schema_name(name: &str) -> String {
    let mut sanitized = String::new();
    for ch in name.trim().chars() {
        if sanitized.len() + ch.len_utf8() > MAX_SCHEMA_NAME_BYTES {
            break;
        }
        sanitized.push(if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            ch
        } else {
            '_'
        });
    }
    sanitized
}

/// Lists accepted-but-unsupported Responses features for compatibility logging.
pub fn unsupported_feature_names(request: &Value) -> Vec<&'static str> {
    let mut features = Vec::new();
    if request.get("previous_response_id").is_some() {
        features.push("previous_response_id");
    }
    features
}

/// Input item types `translate_input_items` knows how to carry.
///
/// Diagnostics compares against this list, and a table-driven test keeps it in
/// sync with the `match` arms so a newly handled item cannot keep being
/// reported as dropped.
pub(super) const HANDLED_INPUT_TYPES: &[&str] = &[
    "message",
    "reasoning",
    "function_call",
    "custom_tool_call",
    "tool_search_call",
    "function_call_output",
    "custom_tool_call_output",
    "additional_tools",
    "tool_search_output",
    "compaction",
    "compaction_summary",
    "context_compaction",
    "compaction_trigger",
    "local_shell_call",
    "local_shell_call_output",
    "computer_call",
    "computer_call_output",
    "mcp_call",
    "mcp_list_tools",
    "web_search_call",
    "input_text",
    "input_image",
    "input_file",
    "input_audio",
];

pub(super) fn is_handled_input_type(name: &str) -> bool {
    HANDLED_INPUT_TYPES.contains(&name)
}

pub(super) fn translate_input_items(
    items: &[Value],
    messages: &mut Vec<Value>,
    options: TranslationOptions,
) {
    let mut pending: Option<Value> = None;
    let mut reasoning: Option<String> = None;
    // Images lifted out of tool results, emitted once the tool turn ends.
    let mut media: Vec<Value> = Vec::new();
    let mut known_call_ids = HashSet::new();
    let mut orphan_outputs = Vec::new();
    for item in items {
        let Some(object) = item.as_object() else {
            continue;
        };
        let item_type = object
            .get("type")
            .and_then(Value::as_str)
            .or_else(|| object.contains_key("role").then_some("message"));
        if !matches!(
            item_type,
            Some("function_call_output" | "custom_tool_call_output" | "tool_search_output")
        ) {
            flush_orphan_outputs(messages, &mut orphan_outputs, options);
        }
        match item_type {
            Some(name) if DEGRADED_INPUT_TYPES.contains(&name) => {
                flush_pending(messages, &mut pending);
                flush_tool_media(messages, &mut media);
                messages.push(json!({
                    "role": "user",
                    "content": format!(
                        "[Responses item {name} cannot be represented as a Chat tool call; preserved as context]\n{}",
                        item
                    )
                }));
            }
            Some("message") => {
                let role = object.get("role").and_then(Value::as_str).unwrap_or("user");
                let role = if role == "developer" { "system" } else { role };
                if role == "assistant" {
                    if let Some(message) = pending.as_mut() {
                        merge_assistant_content(message, message_content(item, options));
                        continue;
                    }
                }
                flush_pending(messages, &mut pending);
                flush_tool_media(messages, &mut media);
                if role != "assistant" {
                    reasoning = None;
                }
                let mut message = json!({
                    "role": role,
                    "content": message_content(item, options)
                });
                if role == "assistant" {
                    if let Some(text) = reasoning.take() {
                        message["reasoning_content"] = json!(text);
                    }
                }
                messages.push(message);
            }
            Some("reasoning") => {
                let text = reasoning_item_text(item);
                append_reasoning(&mut reasoning, text.clone());
                if text.is_none() {
                    continue;
                }
            }
            Some("function_call" | "custom_tool_call" | "tool_search_call") => {
                // Media from the preceding tool turn belongs before this new
                // assistant turn, not after its results.
                flush_tool_media(messages, &mut media);
                let raw_name = object.get("name").and_then(Value::as_str).unwrap_or("");
                let name = if item_type == Some("tool_search_call") {
                    super::tools::TOOL_SEARCH_PROXY_NAME.to_string()
                } else {
                    object
                        .get("namespace")
                        .and_then(Value::as_str)
                        .filter(|namespace| !namespace.is_empty())
                        .map(|namespace| flat_namespace_name(namespace, raw_name))
                        .unwrap_or_else(|| raw_name.to_string())
                };
                let call_id = object
                    .get("call_id")
                    .or_else(|| object.get("id"))
                    .and_then(Value::as_str)
                    .filter(|call_id| !call_id.is_empty())
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("call_local_{}", known_call_ids.len() + 1));
                known_call_ids.insert(call_id.clone());
                let arguments = if item_type == Some("custom_tool_call") {
                    json!({
                        "input": object
                            .get("input")
                            .map(value_to_text)
                            .unwrap_or_default()
                    })
                    .to_string()
                } else if item_type == Some("tool_search_call") {
                    object
                        .get("arguments")
                        .map(super::canonical::canonical_json_string)
                        .unwrap_or_else(|| "{}".to_string())
                } else {
                    let arguments = object
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim();
                    if arguments.is_empty() {
                        "{}".to_string()
                    } else {
                        super::canonical::canonicalize_object_arguments(arguments)
                            .unwrap_or_else(|| arguments.to_string())
                    }
                };
                let mut call = json!({
                    "id": call_id,
                    "type": "function",
                    "function": { "name": name, "arguments": arguments }
                });
                if let Some(extra) = object
                    .get("extra_content")
                    .filter(|value| value.as_object().is_some_and(|extra| !extra.is_empty()))
                {
                    call["extra_content"] = extra.clone();
                }
                if let Some(message) = pending.as_mut() {
                    if let Some(calls) = message["tool_calls"].as_array_mut() {
                        calls.push(call);
                    }
                } else if let Some(message) = messages.last_mut().filter(|message| {
                    message.get("role").and_then(Value::as_str) == Some("assistant")
                }) {
                    if !message.get("tool_calls").is_some_and(Value::is_array) {
                        message["tool_calls"] = json!([]);
                    }
                    message["tool_calls"]
                        .as_array_mut()
                        .expect("tool_calls was initialized as an array")
                        .push(call);
                    if message.get("content").is_none() {
                        message["content"] = Value::Null;
                    }
                    if let Some(text) = reasoning.take() {
                        if let Some(existing) =
                            message.get("reasoning_content").and_then(Value::as_str)
                        {
                            message["reasoning_content"] = json!(format!("{existing}\n{text}"));
                        } else {
                            message["reasoning_content"] = json!(text);
                        }
                    }
                } else {
                    let mut message = json!({
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [call]
                    });
                    if let Some(text) = reasoning.take() {
                        message["reasoning_content"] = json!(text);
                    }
                    pending = Some(message);
                }
            }
            Some("function_call_output" | "custom_tool_call_output") => {
                let call_id = object.get("call_id").and_then(Value::as_str).unwrap_or("");
                if call_id.is_empty() || !known_call_ids.contains(call_id) {
                    orphan_outputs.push(item.clone());
                    continue;
                }
                flush_pending(messages, &mut pending);
                let output = object.get("output").unwrap_or(&Value::Null);
                let content = extract_tool_output_media(output, call_id, &mut media, options)
                    .unwrap_or_else(|| value_to_text(output));
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content
                }));
            }
            Some("additional_tools") => {
                // The declared tools were lifted by `collect_declared_tools`.
                // The carrier carries a role but no content, so letting it fall
                // through to the message arm would emit an empty system message.
            }
            Some("input_text" | "input_image" | "input_file" | "input_audio") => {
                flush_pending(messages, &mut pending);
                flush_tool_media(messages, &mut media);
                messages.push(json!({
                    "role": "user",
                    "content": message_content(item, options)
                }));
            }
            Some("tool_search_output") => {
                let names = object
                    .get("tools")
                    .and_then(Value::as_array)
                    .map(|tools| declared_tool_names(tools))
                    .unwrap_or_default();
                let text = if names.is_empty() {
                    "[tool search returned no tools]".to_string()
                } else {
                    format!("[tool search loaded: {}]", names.join(", "))
                };
                let call_id = object.get("call_id").and_then(Value::as_str).unwrap_or("");
                if !call_id.is_empty() && known_call_ids.contains(call_id) {
                    flush_pending(messages, &mut pending);
                    let output = object.get("output").map(value_to_text).unwrap_or(text);
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": output
                    }));
                } else {
                    let mut orphan = item.clone();
                    orphan["output"] = json!(text);
                    orphan_outputs.push(orphan);
                }
            }
            _ => {}
        }
    }
    flush_orphan_outputs(messages, &mut orphan_outputs, options);
    flush_pending(messages, &mut pending);
    flush_tool_media(messages, &mut media);
}

fn append_reasoning(target: &mut Option<String>, next: Option<String>) {
    let Some(next) = next.filter(|text| !text.trim().is_empty()) else {
        return;
    };
    match target {
        Some(existing) if !existing.is_empty() => {
            existing.push_str("\n\n");
            existing.push_str(&next);
        }
        _ => *target = Some(next),
    }
}

fn preserve_invalid_tool_calls_as_text(messages: &mut Vec<Value>) {
    let mut output = Vec::with_capacity(messages.len());
    let mut invalid_owners = HashMap::<String, usize>::new();

    for mut message in std::mem::take(messages) {
        if message.get("role").and_then(Value::as_str) == Some("assistant") {
            let mut notes = Vec::new();
            if let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) {
                calls.retain(|call| {
                    let id = call.get("id").and_then(Value::as_str);
                    if let Some(id) = id {
                        invalid_owners.remove(id);
                    }
                    if arguments_are_object(call.pointer("/function/arguments")) {
                        return true;
                    }
                    if let Some(id) = id.filter(|id| !id.is_empty()) {
                        invalid_owners.insert(id.to_string(), output.len());
                    }
                    let name = call
                        .pointer("/function/name")
                        .and_then(Value::as_str)
                        .unwrap_or("<unknown>");
                    let arguments = call
                        .pointer("/function/arguments")
                        .map(value_to_text)
                        .unwrap_or_else(|| "<missing>".to_string());
                    notes.push(format!(
                        "[Historical tool call with invalid arguments; context only, not a new tool invocation]\nname: {name}\ncall_id: {}\narguments: {arguments}",
                        id.unwrap_or("<missing>")
                    ));
                    false
                });
            }
            if message
                .get("tool_calls")
                .and_then(Value::as_array)
                .is_some_and(Vec::is_empty)
            {
                message.as_object_mut().unwrap().remove("tool_calls");
            }
            for note in notes {
                append_message_text(&mut message, &note);
            }
        } else if message.get("role").and_then(Value::as_str) == Some("tool") {
            if let Some((id, owner)) = message
                .get("tool_call_id")
                .and_then(Value::as_str)
                .and_then(|id| invalid_owners.get(id).map(|owner| (id, *owner)))
            {
                let result = message
                    .get("content")
                    .map(value_to_text)
                    .unwrap_or_default();
                append_message_text(
                    &mut output[owner],
                    &format!(
                        "[Historical tool result for invalid call {id}; context only]\n{result}"
                    ),
                );
                continue;
            }
        }
        output.push(message);
    }
    *messages = output;
}

fn arguments_are_object(arguments: Option<&Value>) -> bool {
    match arguments {
        Some(Value::String(arguments)) => {
            serde_json::from_str::<Value>(arguments).is_ok_and(|value| value.is_object())
        }
        Some(Value::Object(_)) => true,
        _ => false,
    }
}

pub(super) fn dedupe_function_call_arguments(messages: &mut [Value]) -> usize {
    let mut repaired = 0;
    for call in messages
        .iter_mut()
        .filter_map(|message| message.get_mut("tool_calls"))
        .filter_map(Value::as_array_mut)
        .flatten()
        .filter(|call| call.get("type").and_then(Value::as_str) == Some("function"))
    {
        let Some(arguments) = call.pointer("/function/arguments").and_then(Value::as_str) else {
            continue;
        };
        let Some(deduped) = dedupe_json_object(arguments) else {
            continue;
        };
        let Some(arguments) = call.pointer_mut("/function/arguments") else {
            continue;
        };
        *arguments = Value::String(deduped);
        repaired += 1;
    }
    repaired
}

fn dedupe_json_object(raw: &str) -> Option<String> {
    let DedupedValue {
        value,
        duplicate_keys,
    } = serde_json::from_str::<DedupedValue>(raw).ok()?;
    if !value.is_object() || duplicate_keys == 0 {
        return None;
    }
    Some(super::canonical::canonical_json_string(&value))
}

struct DedupedValue {
    value: Value,
    duplicate_keys: usize,
}

impl<'de> serde::Deserialize<'de> for DedupedValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(DedupedVisitor)
    }
}

struct DedupedVisitor;

impl<'de> serde::de::Visitor<'de> for DedupedVisitor {
    type Value = DedupedValue;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(DedupedValue {
            value: Value::Bool(value),
            duplicate_keys: 0,
        })
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(DedupedValue {
            value: Value::Number(value.into()),
            duplicate_keys: 0,
        })
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(DedupedValue {
            value: Value::Number(value.into()),
            duplicate_keys: 0,
        })
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(DedupedValue {
            value: serde_json::Number::from_f64(value).map_or(Value::Null, Value::Number),
            duplicate_keys: 0,
        })
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(DedupedValue {
            value: Value::String(value.to_string()),
            duplicate_keys: 0,
        })
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(DedupedValue {
            value: Value::String(value),
            duplicate_keys: 0,
        })
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(DedupedValue {
            value: Value::Null,
            duplicate_keys: 0,
        })
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(DedupedValue {
            value: Value::Null,
            duplicate_keys: 0,
        })
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        serde::Deserialize::deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut values = Vec::new();
        let mut duplicate_keys = 0;
        while let Some(item) = seq.next_element::<DedupedValue>()? {
            duplicate_keys += item.duplicate_keys;
            values.push(item.value);
        }
        Ok(DedupedValue {
            value: Value::Array(values),
            duplicate_keys,
        })
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut values = serde_json::Map::new();
        let mut duplicate_keys = 0;
        while let Some((key, item)) = map.next_entry::<String, DedupedValue>()? {
            if values.insert(key, item.value).is_some() {
                duplicate_keys += 1;
            }
            duplicate_keys += item.duplicate_keys;
        }
        Ok(DedupedValue {
            value: Value::Object(values),
            duplicate_keys,
        })
    }
}

fn append_message_text(message: &mut Value, text: &str) {
    match message.get_mut("content") {
        Some(Value::String(content)) => {
            if !content.is_empty() {
                content.push_str("\n\n");
            }
            content.push_str(text);
        }
        Some(Value::Array(parts)) => {
            parts.push(json!({ "type": "text", "text": text }));
        }
        _ => message["content"] = json!(text),
    }
}

fn collapse_system_messages_to_head(messages: Vec<Value>) -> Vec<Value> {
    let mut head = Vec::new();
    let mut rest = Vec::with_capacity(messages.len());
    let mut seen_non_system = false;

    for (index, message) in messages.iter().enumerate() {
        let is_system = message.get("role").and_then(Value::as_str) == Some("system");
        if is_system {
            let text = message
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim();
            if text.is_empty() {
                continue;
            }
            let followed_by_user = messages[index + 1..]
                .iter()
                .find(|next| next.get("role").and_then(Value::as_str) != Some("system"))
                .and_then(|next| next.get("role"))
                .and_then(Value::as_str)
                == Some("user");
            if seen_non_system && followed_by_user {
                rest.push(json!({ "role": "user", "content": text }));
            } else {
                head.push(text.to_string());
            }
        } else {
            seen_non_system = true;
            rest.push(message.clone());
        }
    }

    let mut output = Vec::with_capacity(rest.len() + usize::from(!head.is_empty()));
    if !head.is_empty() {
        output.push(json!({
            "role": "system",
            "content": head.join("\n\n")
        }));
    }
    output.extend(rest);
    output
}

fn merge_assistant_content(message: &mut Value, content: Value) {
    let is_empty = |value: &Value| {
        value.is_null()
            || value.as_str().is_some_and(str::is_empty)
            || value.as_array().is_some_and(Vec::is_empty)
    };
    if is_empty(&content) {
        return;
    }
    if is_empty(&message["content"]) {
        message["content"] = content;
        return;
    }

    let previous = message["content"].take();
    message["content"] = match (previous, content) {
        (Value::String(mut previous), Value::String(content)) => {
            if !previous.is_empty() {
                previous.push('\n');
            }
            previous.push_str(&content);
            Value::String(previous)
        }
        (Value::Array(mut parts), Value::Array(more)) => {
            parts.extend(more);
            Value::Array(parts)
        }
        (Value::Array(mut parts), Value::String(content)) => {
            parts.push(json!({ "type": "text", "text": content }));
            Value::Array(parts)
        }
        (Value::String(previous), Value::Array(mut more)) => {
            let mut parts = vec![json!({ "type": "text", "text": previous })];
            parts.append(&mut more);
            Value::Array(parts)
        }
        (previous, content) => {
            let mut parts = Vec::new();
            if !is_empty(&previous) {
                parts.push(previous);
            }
            parts.push(content);
            Value::Array(parts)
        }
    };
}

fn flush_orphan_outputs(
    messages: &mut Vec<Value>,
    outputs: &mut Vec<Value>,
    options: TranslationOptions,
) {
    for output in std::mem::take(outputs) {
        let call_id = output.get("call_id").and_then(Value::as_str).unwrap_or("");
        let mut media = Vec::new();
        let body = output
            .get("output")
            .map(|value| {
                extract_tool_output_media(value, call_id, &mut media, options)
                    .unwrap_or_else(|| value_to_text(value))
            })
            .unwrap_or_default();
        let marker = if call_id.is_empty() {
            "[tool_result without a matching tool_call]".to_string()
        } else {
            format!("[tool_result without a matching tool_call] (call_id: {call_id})")
        };
        let text = format!("{marker}\n{body}");
        let content = if media.is_empty() {
            Value::String(text)
        } else {
            let mut parts = vec![json!({ "type": "text", "text": text })];
            parts.extend(media);
            Value::Array(parts)
        };
        messages.push(json!({ "role": "user", "content": content }));
    }
}

pub(super) fn flush_pending(messages: &mut Vec<Value>, pending: &mut Option<Value>) {
    if let Some(message) = pending.take() {
        messages.push(message);
    }
}

/// Drops user messages that translate to empty content.
///
/// Codex rebuilds history after compaction and can emit user turns that only
/// carried media or parts this translation does not recognize. Those become an
/// empty string, which strict Chat upstreams reject with "content cannot be
/// empty". Messages carrying only images keep a non-empty parts array and are
/// therefore preserved.
pub(super) fn drop_empty_user_messages(messages: &mut Vec<Value>) {
    messages.retain(|message| {
        if message.get("role").and_then(Value::as_str) != Some("user") {
            return true;
        }
        match message.get("content") {
            Some(Value::String(text)) => !text.is_empty(),
            Some(Value::Array(parts)) => !parts.is_empty(),
            _ => true,
        }
    });
}

pub(super) fn repair_tool_pairing(messages: &mut Vec<Value>) {
    let mut index = 0;
    while index < messages.len() {
        let ids: Vec<String> = messages[index]
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|call| call.get("id").and_then(Value::as_str).map(str::to_string))
            .collect();
        if ids.is_empty() {
            index += 1;
            continue;
        }
        let mut next = index + 1;
        let mut answered = HashSet::new();
        while next < messages.len()
            && messages[next].get("role").and_then(Value::as_str) == Some("tool")
        {
            if let Some(id) = messages[next].get("tool_call_id").and_then(Value::as_str) {
                answered.insert(id.to_string());
            }
            next += 1;
        }
        let missing: Vec<String> = ids
            .into_iter()
            .filter(|id| !answered.contains(id))
            .collect();
        for (offset, id) in missing.iter().enumerate() {
            messages.insert(
                next + offset,
                json!({
                    "role": "tool",
                    "tool_call_id": id,
                    "content": MISSING_TOOL_OUTPUT
                }),
            );
        }
        index = next + missing.len();
    }
    let mut known_calls = HashSet::new();
    messages.retain(|message| {
        if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
            for id in calls
                .iter()
                .filter_map(|call| call.get("id").and_then(Value::as_str))
            {
                known_calls.insert(id.to_string());
            }
        }
        if message.get("role").and_then(Value::as_str) == Some("tool") {
            message
                .get("tool_call_id")
                .and_then(Value::as_str)
                .is_some_and(|id| known_calls.contains(id))
        } else {
            true
        }
    });
}

pub(super) fn message_content(item: &Value, options: TranslationOptions) -> Value {
    let direct_part = matches!(
        item.get("type").and_then(Value::as_str),
        Some("input_text" | "input_image" | "input_file" | "input_audio")
    );
    let parts = if direct_part {
        vec![item.clone()]
    } else {
        let Some(parts) = item.get("content").and_then(Value::as_array) else {
            return item
                .get("content")
                .cloned()
                .unwrap_or_else(|| Value::String(String::new()));
        };
        parts.clone()
    };
    let has_non_text = parts.iter().any(|part| {
        matches!(
            part.get("type").and_then(Value::as_str),
            Some("input_image" | "input_file" | "input_audio")
        )
    });
    if !has_non_text {
        return Value::String(
            parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join(""),
        );
    }
    let content: Vec<Value> = parts
        .iter()
        .filter_map(|part| chat_content_part(part, options))
        .collect();
    if content
        .iter()
        .all(|part| part.get("type").and_then(Value::as_str) == Some("text"))
    {
        Value::String(
            content
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n"),
        )
    } else {
        Value::Array(content)
    }
}

pub(super) fn chat_content_part(part: &Value, options: TranslationOptions) -> Option<Value> {
    match part.get("type").and_then(Value::as_str) {
        Some("input_text" | "output_text") => part
            .get("text")
            .and_then(Value::as_str)
            .map(|text| json!({ "type": "text", "text": text })),
        Some("refusal") => part
            .get("refusal")
            .and_then(Value::as_str)
            .map(|text| json!({ "type": "text", "text": text })),
        Some("input_image") => part.get("image_url").and_then(Value::as_str).map(|url| {
            let mut image = json!({ "url": url });
            if let Some(detail) = part
                .get("detail")
                .and_then(Value::as_str)
                .filter(|detail| matches!(*detail, "auto" | "low" | "high"))
            {
                image["detail"] = json!(detail);
            }
            json!({ "type": "image_url", "image_url": image })
        }),
        Some("input_file") => {
            let filename = part
                .get("filename")
                .and_then(Value::as_str)
                .unwrap_or("attachment");
            if !options.supports_file_input {
                return Some(json!({
                    "type": "text",
                    "text": format!("[file attachment omitted: {filename}; upstream model does not support file input]")
                }));
            }
            let mut file = json!({ "filename": filename });
            if let Some(file_id) = part.get("file_id").and_then(Value::as_str) {
                file["file_id"] = json!(file_id);
            }
            if let Some(file_data) = part.get("file_data").and_then(Value::as_str) {
                file["file_data"] = json!(file_data);
            }
            if file.get("file_id").is_none() && file.get("file_data").is_none() {
                return Some(json!({
                    "type": "text",
                    "text": format!("[file attachment unavailable: {filename}]")
                }));
            }
            Some(json!({ "type": "file", "file": file }))
        }
        Some("input_audio") => {
            let Some(audio) = part.get("input_audio") else {
                return Some(json!({
                    "type": "text",
                    "text": "[audio attachment unavailable]"
                }));
            };
            if !options.supports_audio_input {
                return Some(json!({
                    "type": "text",
                    "text": "[audio attachment omitted; upstream model does not support audio input]"
                }));
            }
            Some(json!({
                "type": "input_audio",
                "input_audio": audio.clone()
            }))
        }
        _ => None,
    }
}

pub(super) fn reasoning_item_text(item: &Value) -> Option<String> {
    let collect = |parts: Option<&Vec<Value>>| {
        let text = parts?
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        (!text.trim().is_empty()).then_some(text)
    };
    collect(item.get("summary").and_then(Value::as_array))
        .or_else(|| collect(item.get("content").and_then(Value::as_array)))
}

pub(super) fn value_to_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}
