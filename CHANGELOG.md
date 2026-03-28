# Changelog

This file records notable repository changes.

## [Unreleased]

### Added

- Added structured web export classifications under `results.classifications` with entries shaped like `{ "system": "DK|RVK", "code": "...", "display": "..." }`.
- Added a deprecated compatibility alias note for `results.dk_classifications`, which remains available as the legacy string list.
- Added export-time RVK validation metadata in structured `results.classifications` entries by checking RVK codes against the official RVK API and marking non-standard notations explicitly.
- Added RVK validation surfacing in the web UI results panel and stream log, including per-code badges and a compact summary of standard versus non-standard RVK notations.

### Fixed

- Fixed the web GUI so completed analyses render the final results panel reliably after pipeline completion.
- Fixed the polling/results rendering path so string-valued keyword fields no longer trigger `forEach is not a function` errors in the web UI.
- Fixed stale recovery UI behavior in the web GUI so `Verbindung unterbrochen` and `Wiederherstellen` are only shown for an actually interrupted live analysis, not after a successful completion.
- Fixed MarcXML SRU DK/RVK extraction so keyword-based catalog results use the same keyword-centric shape as the rest of the pipeline and are no longer discarded before DK/RVK classification.
- Fixed HTML-escaped values in parsed LLM output and SRU search terms by decoding entities such as `&lt;` before downstream processing.
- Fixed image OCR with OpenAI-compatible vision models so ALIMA uses the selected model-specific OCR prompt, disables JSON mode for raw OCR text, omits unsupported sampling parameters for multimodal OpenAI chat requests, and no longer reports provider error strings as successful OCR output.

### Changed

- Clarified the semantics of legacy `dk_classifications`: the field now remains as a backward-compatible alias for final DK/RVK classification strings rather than implying DK-only output.
- Updated PyQt classification labels and selectors to use neutral DK/RVK wording, added a `classifications` alias on `KeywordAnalysisState`, and fixed PyQt title lookup/K10+ export helpers to preserve RVK systems correctly.
