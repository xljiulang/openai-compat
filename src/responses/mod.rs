//! Responses API to Chat Completions compatibility translation.

use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

const MISSING_TOOL_OUTPUT: &str =
    "Tool output was missing from conversation history and has been replaced by openai-compat.";

mod canonical;
mod diagnostics;
mod error;
mod inline_think;
mod media;
mod request;
mod response;
mod stream;
mod tools;

pub use diagnostics::{ignored_feature_details, ignored_feature_details_with_options};
pub use request::{
    prepare_responses, responses_to_chat, responses_to_chat_with_options,
    unsupported_feature_names, PreparedResponses, TranslationOptions,
};
pub use response::{
    chat_to_responses, chat_to_responses_with_cleaner, usage_to_responses, ResponseCleaner,
};
pub use stream::ResponsesStream;
pub use tools::ToolShape;

pub use error::{normalize_chat_error, normalize_chat_error_text, ChatErrorInfo};
pub use stream::reasoning_text;

use inline_think::InlineThinkSplitter;
use media::{extract_tool_output_media, flush_tool_media};
use response::custom_tool_input;
use stream::{gen_id, unix_now};
use tools::{
    collect_declared_tools, declared_tool_names, flat_namespace_name, normalize_arguments,
    translate_tool_choice, translate_tools, DeclaredTools,
};

#[cfg(test)]
mod tests;
