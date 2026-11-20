use super::{json, HashMap, HashSet, Value};

const CHAT_TOOL_NAME_MAX_BYTES: usize = 64;
pub(super) const TOOL_SEARCH_PROXY_NAME: &str = "openai_compat_tool_search";

/// Request-local tool name mapping used to translate responses back.
///
/// It records which Responses tool names the upstream was told about, so tool
/// calls that come back with flattened or namespaced names can be restored.
#[derive(Debug, Default, Clone)]
#[non_exhaustive]
pub struct ToolShape {
    namespaces: HashMap<String, (String, String)>,
    declared: HashSet<String>,
    pub(super) custom: HashSet<String>,
    tool_search: bool,
}

/// Tool declarations after deduplication by their flattened Chat name.
///
/// `translate_tools` (what the upstream is told) and `ToolShape` (how call
/// names are mapped back) both read this so they cannot disagree about which
/// declaration won a name collision.
#[derive(Default)]
pub(super) struct DeclaredTools {
    /// Deduplicated declarations in request order. Never reordered: the prompt
    /// prefix must stay byte-stable for upstream prefix caching.
    pub(super) tools: Vec<Value>,
    /// Name collisions and unrecognized carriers, surfaced by diagnostics.
    pub(super) notes: Vec<String>,
    /// Names that collide with a synthetic tool injected by this crate.
    pub(super) reserved_conflicts: Vec<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ToolSource {
    TopLevel,
    AdditionalTools,
    ToolSearchOutput,
}

impl ToolSource {
    fn label(self) -> &'static str {
        match self {
            Self::TopLevel => "top",
            Self::AdditionalTools => "additional_tools",
            Self::ToolSearchOutput => "tool_search_output",
        }
    }
}

/// Collects every tool a request declares: the top-level `tools` array first,
/// then carriers embedded in `input`.
///
/// Top-level declarations win over carriers, and earlier carriers win over
/// later ones. Dropped declarations are recorded in `notes` instead of being
/// silently merged, because the response side must never resolve a call name
/// back to a declaration the upstream was not given.
pub(super) fn collect_declared_tools(request: &Value) -> DeclaredTools {
    let mut declared = DeclaredTools::default();
    // A compaction turn is a summarization request: exposing tools would let
    // the model answer with a tool call the client never asked for.
    if super::request::is_compaction_turn(request) {
        return declared;
    }
    let mut seen: HashMap<String, ToolSource> = HashMap::new();
    if let Some(tools) = request.get("tools").and_then(Value::as_array) {
        absorb_tools(ToolSource::TopLevel, tools, &mut seen, &mut declared);
    }
    if let Some(items) = request.get("input").and_then(Value::as_array) {
        for item in items {
            let source = match item.get("type").and_then(Value::as_str) {
                Some("additional_tools") => ToolSource::AdditionalTools,
                Some("tool_search_output") => ToolSource::ToolSearchOutput,
                _ => continue,
            };
            if let Some(tools) = item.get("tools").and_then(Value::as_array) {
                absorb_tools(source, tools, &mut seen, &mut declared);
            }
        }
    }
    declared
}

fn absorb_tools(
    source: ToolSource,
    tools: &[Value],
    seen: &mut HashMap<String, ToolSource>,
    declared: &mut DeclaredTools,
) {
    for tool in tools {
        match tool.get("type").and_then(Value::as_str) {
            Some("namespace") => {
                let Some(namespace) = tool.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let Some(children) = tool.get("tools").and_then(Value::as_array) else {
                    continue;
                };
                let mut kept = Vec::new();
                for child in children {
                    let Some(name) = tool_name(child) else {
                        note_unnamed_tool(child, declared);
                        continue;
                    };
                    let flat = flat_namespace_name(namespace, name);
                    if flat == TOOL_SEARCH_PROXY_NAME {
                        declared.reserved_conflicts.push(flat);
                        continue;
                    }
                    if admit(&flat, source, seen, declared) {
                        kept.push(child.clone());
                    }
                }
                if kept.is_empty() {
                    continue;
                }
                let mut clone = tool.clone();
                clone["tools"] = Value::Array(kept);
                declared.tools.push(clone);
            }
            Some("tool_search") => {
                if admit(TOOL_SEARCH_PROXY_NAME, source, seen, declared) {
                    declared.tools.push(tool.clone());
                }
            }
            _ => {
                let Some(name) = tool_name(tool) else {
                    note_unnamed_tool(tool, declared);
                    continue;
                };
                if name == TOOL_SEARCH_PROXY_NAME {
                    declared.reserved_conflicts.push(name.to_string());
                    continue;
                }
                if admit(name, source, seen, declared) {
                    declared.tools.push(tool.clone());
                }
            }
        }
    }
}

fn note_unnamed_tool(tool: &Value, declared: &mut DeclaredTools) {
    if let Some(kind) = tool
        .get("type")
        .and_then(Value::as_str)
        .filter(|kind| !kind.is_empty())
    {
        declared.notes.push(format!("tools.ignored:{kind}"));
    }
}

fn admit(
    flat: &str,
    source: ToolSource,
    seen: &mut HashMap<String, ToolSource>,
    declared: &mut DeclaredTools,
) -> bool {
    if let Some(kept) = seen.get(flat) {
        declared.notes.push(format!(
            "tools.duplicate:{flat} kept={} dropped={}",
            kept.label(),
            source.label()
        ));
        return false;
    }
    seen.insert(flat.to_string(), source);
    true
}

impl ToolShape {
    /// Builds the request-local tool shape from Responses declarations.
    #[must_use]
    pub fn from_request(request: &Value) -> Self {
        Self::from_declared(&collect_declared_tools(request).tools)
    }

    /// Builds the reverse map from declarations the upstream was actually given.
    pub(super) fn from_declared(tools: &[Value]) -> Self {
        let mut shape = Self::default();
        for tool in tools {
            match tool.get("type").and_then(Value::as_str) {
                Some("namespace") => {
                    let Some(namespace) = tool.get("name").and_then(Value::as_str) else {
                        continue;
                    };
                    let Some(children) = tool.get("tools").and_then(Value::as_array) else {
                        continue;
                    };
                    for child in children {
                        let Some(name) = tool_name(child) else {
                            continue;
                        };
                        shape
                            .namespaces
                            .entry(flat_namespace_name(namespace, name))
                            .or_insert_with(|| (namespace.to_string(), name.to_string()));
                    }
                }
                Some("custom") => {
                    if let Some(name) = tool_name(tool) {
                        shape.custom.insert(name.to_string());
                        shape.declared.insert(name.to_string());
                    }
                }
                Some("tool_search") => {
                    shape.tool_search = true;
                    shape.declared.insert(TOOL_SEARCH_PROXY_NAME.to_string());
                }
                _ => {
                    if let Some(name) = tool_name(tool) {
                        shape.declared.insert(name.to_string());
                    }
                }
            }
        }
        shape
    }

    pub(super) fn resolve(&self, name: &str) -> (String, Option<String>) {
        match self.namespaces.get(name) {
            Some((namespace, tool)) => (tool.clone(), Some(namespace.clone())),
            None => (name.to_string(), None),
        }
    }

    pub(super) fn is_custom(&self, name: &str) -> bool {
        self.custom.contains(name)
    }

    pub(super) fn is_tool_search(&self, name: &str) -> bool {
        self.tool_search && name == TOOL_SEARCH_PROXY_NAME
    }

    pub(super) fn chat_name_for_choice(
        &self,
        name: &str,
        namespace: Option<&str>,
    ) -> Option<String> {
        let namespace = namespace.filter(|namespace| !namespace.is_empty());
        if let Some(namespace) = namespace {
            return self
                .namespaces
                .iter()
                .find(|(_, (candidate_namespace, candidate_name))| {
                    candidate_namespace == namespace && candidate_name == name
                })
                .map(|(chat_name, _)| chat_name.clone());
        }
        (self.declared.contains(name) || self.namespaces.contains_key(name))
            .then(|| name.to_string())
    }
}

pub(super) fn translate_tools(tools: &[Value]) -> Vec<Value> {
    let mut translated = Vec::new();
    let mut seen = HashSet::new();
    for tool in tools {
        match tool.get("type").and_then(Value::as_str) {
            Some("namespace") => {
                let Some(namespace) = tool.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let Some(children) = tool.get("tools").and_then(Value::as_array) else {
                    continue;
                };
                for child in children {
                    let Some(name) = tool_name(child) else {
                        continue;
                    };
                    let flat = flat_namespace_name(namespace, name);
                    if seen.insert(flat.clone()) {
                        translated.push(function_tool(&flat, child));
                    }
                }
            }
            Some("custom") => {
                let Some(name) = tool_name(tool) else {
                    continue;
                };
                if seen.insert(name.to_string()) {
                    let description = custom_tool_description(tool, name);
                    translated.push(json!({
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": description,
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "input": {
                                        "type": "string",
                                        "description": "The complete free-form tool input."
                                    }
                                },
                                "required": ["input"]
                            }
                        }
                    }));
                }
            }
            Some("tool_search") => {
                if seen.insert(TOOL_SEARCH_PROXY_NAME.to_string()) {
                    translated.push(json!({
                        "type": "function",
                        "function": {
                            "name": TOOL_SEARCH_PROXY_NAME,
                            "description": "Search and load tools, plugins, connectors, and MCP namespaces for the current task.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "Search query for tools or connectors to load."
                                    },
                                    "limit": {
                                        "type": "integer",
                                        "description": "Maximum number of tool groups to return."
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    }));
                }
            }
            _ => {
                let Some(name) = tool_name(tool) else {
                    continue;
                };
                if seen.insert(name.to_string()) {
                    translated.push(function_tool(name, tool));
                }
            }
        }
    }
    translated
}

pub(super) fn function_tool(name: &str, source: &Value) -> Value {
    let source = tool_function(source);
    let mut function = json!({ "name": name });
    if let Some(description) = source.get("description").filter(|value| !value.is_null()) {
        function["description"] = description.clone();
    }
    function["parameters"] = normalize_parameters(source.get("parameters"));
    if let Some(strict) = source.get("strict").filter(|value| !value.is_null()) {
        function["strict"] = strict.clone();
    }
    json!({ "type": "function", "function": function })
}

fn tool_function(tool: &Value) -> &Value {
    tool.get("function").unwrap_or(tool)
}

fn tool_name(tool: &Value) -> Option<&str> {
    tool_function(tool)
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
}

/// Normalizes a function's `parameters` JSON Schema for strict Chat upstreams.
///
/// Responses tools may declare `parameters: null`, `{"type": null}` or
/// `required: null`, and some gateways reject `additionalProperties: false`
/// without an accompanying `required` array. Only schema-bearing keywords are
/// visited, so literal values under `default`/`examples` stay untouched.
pub(super) fn normalize_parameters(parameters: Option<&Value>) -> Value {
    let mut parameters = match parameters {
        Some(Value::Object(object)) => Value::Object(object.clone()),
        _ => return json!({ "type": "object", "properties": {} }),
    };
    sanitize_parameter_schema(&mut parameters, true);
    parameters
}

fn sanitize_parameter_schema(schema: &mut Value, is_root: bool) {
    let Value::Object(object) = schema else {
        return;
    };
    if is_root && object.get("type").and_then(Value::as_str) != Some("object") {
        object.insert("type".to_string(), json!("object"));
    }
    let invalid_required = object
        .get("required")
        .is_some_and(|required| !required.is_array());
    let strict_object_without_required = object.get("required").is_none()
        && object.get("additionalProperties") == Some(&Value::Bool(false));
    if invalid_required || strict_object_without_required {
        object.insert("required".to_string(), json!([]));
    }

    // Only keywords whose values are themselves schemas are recursed into.
    for key in [
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        "dependentSchemas",
        "dependencies",
    ] {
        if let Some(children) = object.get_mut(key).and_then(Value::as_object_mut) {
            for child in children.values_mut() {
                sanitize_schema_child(child);
            }
        }
    }
    for key in [
        "items",
        "additionalItems",
        "contains",
        "propertyNames",
        "additionalProperties",
        "unevaluatedItems",
        "unevaluatedProperties",
        "contentSchema",
        "not",
        "if",
        "then",
        "else",
    ] {
        if let Some(child) = object.get_mut(key) {
            sanitize_schema_child(child);
        }
    }
    for key in ["prefixItems", "allOf", "anyOf", "oneOf"] {
        if let Some(children) = object.get_mut(key).and_then(Value::as_array_mut) {
            for child in children {
                sanitize_schema_child(child);
            }
        }
    }
}

fn sanitize_schema_child(value: &mut Value) {
    match value {
        Value::Object(_) => sanitize_parameter_schema(value, false),
        Value::Array(children) => {
            for child in children {
                sanitize_schema_child(child);
            }
        }
        _ => {}
    }
}

pub(super) fn custom_tool_description(tool: &Value, name: &str) -> String {
    let mut sections = Vec::new();
    if let Some(description) = tool
        .get("description")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|description| !description.is_empty())
    {
        sections.push(description.to_string());
    } else {
        sections.push(format!("Run the {name} tool."));
    }
    sections.push(
        "This Chat function wraps a free-form Responses tool. Pass the complete tool input as \
         the JSON string in the \"input\" field. Preserve the input text exactly. Do not wrap the \
         free-form payload in another object or add JSON escaping inside the string."
            .to_string(),
    );
    sections.push(format!("Original tool definition:\n```json\n{tool}\n```"));
    sections.join("\n\n")
}

pub(super) fn translate_tool_choice(choice: Option<&Value>, shape: &ToolShape) -> Option<Value> {
    let choice = choice.filter(|value| !value.is_null())?;
    match choice {
        Value::String(value) => Some(json!(value)),
        Value::Object(object)
            if matches!(
                object.get("type").and_then(Value::as_str),
                Some("function" | "custom")
            ) =>
        {
            let name = object.get("name").and_then(Value::as_str)?;
            let namespace = object.get("namespace").and_then(Value::as_str);
            let name = shape.chat_name_for_choice(name, namespace)?;
            Some(json!({ "type": "function", "function": { "name": name } }))
        }
        Value::Object(object)
            if object.get("type").and_then(Value::as_str) == Some("tool_search") =>
        {
            Some(json!({
                "type": "function",
                "function": { "name": TOOL_SEARCH_PROXY_NAME }
            }))
        }
        _ => None,
    }
}

/// Flattened Chat names contributed by a list of Responses tool declarations.
pub(super) fn declared_tool_names(tools: &[Value]) -> Vec<String> {
    let mut names = Vec::new();
    for tool in tools {
        match tool.get("type").and_then(Value::as_str) {
            Some("namespace") => {
                let Some(namespace) = tool.get("name").and_then(Value::as_str) else {
                    continue;
                };
                for child in tool
                    .get("tools")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    if let Some(name) = tool_name(child) {
                        names.push(flat_namespace_name(namespace, name));
                    }
                }
            }
            _ => {
                if let Some(name) = tool_name(tool) {
                    names.push(name.to_string());
                }
            }
        }
    }
    names
}

pub(super) fn flat_namespace_name(namespace: &str, name: &str) -> String {
    let full_name = format!("{namespace}__{name}");
    if full_name.len() <= CHAT_TOOL_NAME_MAX_BYTES {
        return full_name;
    }

    let suffix = format!(
        "__{:016x}",
        xxhash_rust::xxh3::xxh3_64(full_name.as_bytes())
    );
    let prefix_limit = CHAT_TOOL_NAME_MAX_BYTES.saturating_sub(suffix.len());
    let mut prefix = String::with_capacity(prefix_limit);
    for ch in full_name.chars() {
        if prefix.len() + ch.len_utf8() > prefix_limit {
            break;
        }
        prefix.push(ch);
    }
    format!("{prefix}{suffix}")
}

pub(super) fn normalize_arguments(raw: &str) -> (String, bool) {
    let raw = raw.trim();
    if raw.is_empty() {
        return ("{}".to_string(), false);
    }
    super::canonical::canonicalize_object_arguments(raw)
        .map_or_else(|| ("{}".to_string(), true), |arguments| (arguments, false))
}
