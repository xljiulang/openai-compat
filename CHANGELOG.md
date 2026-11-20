# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-11-20

### Added

- `stream_options.include_usage` is now declared for every streaming Responses
  request, so upstream Chat usage can be mapped back. A caller-supplied
  `stream_options` object is preserved as-is; only a missing or `null`
  `include_usage` is filled in. Non-streaming requests do not carry the field.
- `ResponsesStream::with_max_bytes` caps how much translated output the stream
  retains, defaulting to `DEFAULT_MAX_RESPONSE_BYTES` (64 MiB). A breached limit
  fails the response with `response_too_large` instead of truncating silently.
- `PreparedResponses::chat` and `PreparedResponses::into_chat` expose the
  translated Chat body.
- `CHANGELOG.md` and a CI workflow covering formatting, lints, tests and docs.

### Changed

- **Breaking:** `ResponseCleaner` is now
  `Arc<dyn Fn(&str) -> String + Send + Sync>` instead of a bare function
  pointer, so a cleaner can capture per-request configuration.
- **Breaking:** `PreparedResponses` fields are private. Use `chat()`,
  `shape()`, `ignored()` and `unsupported()` instead of direct field access.
- **Breaking:** a malformed upstream SSE payload is reported as
  `ErrorKind::Upstream` rather than `ErrorKind::InvalidInput`; the caller's own
  invalid request body still reports `InvalidInput`.
- Chat-side error messages are English only, matching the Responses module and
  OpenAI error bodies.
- Placeholder text injected into translated conversations now names
  `openai-compat` instead of "the proxy".
- `rust-version` is declared as 1.82, and the unused `serde` derive feature was
  dropped.

### Fixed

- Streaming Responses translation no longer accumulates output without bound.
- Several internal `#[must_use]`, documentation and clone-efficiency lints were
  cleared; `clippy::pedantic` now passes with warnings denied.
- Empty-result assertions in the test suite now use `assert_eq!`, which reports
  the actual value on failure and satisfies the newer `assert_is_empty` lint.

### CI

- The lint toolchain is pinned via `CI_TOOLCHAIN` instead of tracking floating
  `stable`, so a newly released clippy lint cannot fail the build on its own.

## [0.1.0] - 2026-11-20

### Added

- Initial extraction of the protocol compatibility kernel: Chat request
  preparation, Chat SSE aggregation, Responses to Chat translation and Chat to
  Responses translation (JSON and SSE), including namespaced and free-form
  tools, media relocation, reasoning extraction, compaction handling and usage
  mapping.

[Unreleased]: https://github.com/xljiulang/openai-compat/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/xljiulang/openai-compat/releases/tag/v0.2.0
[0.1.0]: https://github.com/xljiulang/openai-compat/commits/main
