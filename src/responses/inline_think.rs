const INLINE_THINK_TAG_PAIRS: [(&str, &str); 2] =
    [("<think>", "</think>"), ("<thinking>", "</thinking>")];

const THINK_SEPARATORS: [char; 2] = ['\r', '\n'];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum InlineThinkMode {
    #[default]
    Detecting,
    Reasoning,
    Text,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThinkPrefixDecision {
    NeedMore,
    Reasoning,
    Text,
}

fn leading_think_prefix_decision(buffer: &str) -> ThinkPrefixDecision {
    let trimmed = buffer.trim_start();
    if trimmed.is_empty() {
        return ThinkPrefixDecision::NeedMore;
    }
    if INLINE_THINK_TAG_PAIRS
        .iter()
        .any(|(open_tag, _)| trimmed.starts_with(open_tag))
    {
        return ThinkPrefixDecision::Reasoning;
    }
    if INLINE_THINK_TAG_PAIRS
        .iter()
        .any(|(open_tag, _)| open_tag.starts_with(trimmed))
    {
        return ThinkPrefixDecision::NeedMore;
    }
    ThinkPrefixDecision::Text
}

fn strip_leading_think_open_tag(text: &str) -> Option<&str> {
    let after_whitespace = text.trim_start();
    INLINE_THINK_TAG_PAIRS
        .iter()
        .find_map(|(open_tag, _)| after_whitespace.strip_prefix(open_tag))
}

fn find_think_close_tag(text: &str) -> Option<(usize, &'static str)> {
    INLINE_THINK_TAG_PAIRS
        .iter()
        .filter_map(|(_, close_tag)| text.find(close_tag).map(|index| (index, *close_tag)))
        .min_by_key(|(index, _)| *index)
}

fn reasoning_holdback_len(buffer: &str) -> usize {
    let partial_close_tag = INLINE_THINK_TAG_PAIRS
        .iter()
        .filter_map(|(_, close_tag)| {
            (1..close_tag.len())
                .rev()
                .find(|length| buffer.ends_with(&close_tag[..*length]))
        })
        .max()
        .unwrap_or(0);
    let body = &buffer[..buffer.len() - partial_close_tag];
    buffer.len() - body.trim_end_matches(THINK_SEPARATORS).len()
}

fn non_empty(text: &str) -> Option<String> {
    (!text.trim().is_empty()).then(|| text.to_string())
}

pub(super) fn split_leading_think_block(text: &str) -> Option<(String, String)> {
    let body = strip_leading_think_open_tag(text)?;
    let (close_start, close_tag) = find_think_close_tag(body)?;
    let answer = body[close_start + close_tag.len()..].trim_start_matches(THINK_SEPARATORS);
    let thinking = body[..close_start].trim_matches(THINK_SEPARATORS);
    Some((thinking.to_string(), answer.to_string()))
}

#[derive(Debug, Default)]
pub(super) struct InlineThinkSplitter {
    mode: InlineThinkMode,
    buffer: String,
    strip_separator: bool,
}

impl InlineThinkSplitter {
    pub(super) fn push(&mut self, delta: &str) -> (Option<String>, Option<String>) {
        match self.mode {
            InlineThinkMode::Text => (None, self.push_text(delta)),
            InlineThinkMode::Detecting => {
                self.buffer.push_str(delta);
                match leading_think_prefix_decision(&self.buffer) {
                    ThinkPrefixDecision::NeedMore => (None, None),
                    ThinkPrefixDecision::Reasoning => {
                        self.mode = InlineThinkMode::Reasoning;
                        self.strip_separator = true;
                        self.buffer = strip_leading_think_open_tag(&self.buffer)
                            .unwrap_or_default()
                            .to_string();
                        self.drain_reasoning()
                    }
                    ThinkPrefixDecision::Text => {
                        self.mode = InlineThinkMode::Text;
                        (None, non_empty(&std::mem::take(&mut self.buffer)))
                    }
                }
            }
            InlineThinkMode::Reasoning => {
                self.buffer.push_str(delta);
                self.drain_reasoning()
            }
        }
    }

    fn push_text(&mut self, delta: &str) -> Option<String> {
        if !self.strip_separator {
            return non_empty(delta);
        }
        let rest = delta.trim_start_matches(THINK_SEPARATORS);
        if rest.is_empty() {
            return None;
        }
        self.strip_separator = false;
        Some(rest.to_string())
    }

    fn drain_reasoning(&mut self) -> (Option<String>, Option<String>) {
        if self.strip_separator {
            let separators =
                self.buffer.len() - self.buffer.trim_start_matches(THINK_SEPARATORS).len();
            self.buffer.drain(..separators);
            if self.buffer.is_empty() {
                return (None, None);
            }
            self.strip_separator = false;
        }

        if let Some((close_start, close_tag)) = find_think_close_tag(&self.buffer) {
            let buffered = std::mem::take(&mut self.buffer);
            self.mode = InlineThinkMode::Text;
            self.strip_separator = true;
            let thinking = non_empty(buffered[..close_start].trim_end_matches(THINK_SEPARATORS));
            let text = self.push_text(&buffered[close_start + close_tag.len()..]);
            (thinking, text)
        } else {
            let split_at = self.buffer.len() - reasoning_holdback_len(&self.buffer);
            let thinking: String = self.buffer.drain(..split_at).collect();
            (non_empty(&thinking), None)
        }
    }

    pub(super) fn flush(&mut self) -> (Option<String>, Option<String>) {
        let buffered = std::mem::take(&mut self.buffer);
        let was_reasoning = self.mode == InlineThinkMode::Reasoning;
        self.mode = InlineThinkMode::Text;
        self.strip_separator = false;
        if was_reasoning {
            (non_empty(&buffered), None)
        } else {
            (None, non_empty(&buffered))
        }
    }
}
