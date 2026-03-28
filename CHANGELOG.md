# Changelog

This file records notable repository changes.

## [Unreleased]

### Added

- Added structured web export classifications under `results.classifications` with entries shaped like `{ "system": "DK|RVK", "code": "...", "display": "..." }`.
- Added a deprecated compatibility alias note for `results.dk_classifications`, which remains available as the legacy string list.
- Added export-time RVK validation metadata in structured `results.classifications` entries by checking RVK codes against the official RVK API and marking non-standard notations explicitly.
- Added RVK validation surfacing in the web UI results panel and stream log, including per-code badges and a compact summary of standard versus non-standard RVK notations.
- Added a branch-aware RVK API fallback for classification: when catalog search returns no RVK candidates, ALIMA now looks up authority-backed RVK notations via the official RVK API, includes hierarchy paths for ranking, and rejects free-form RVK output that is not among the supplied authority-backed candidates.
- Added a local RVK MarcXML GND indexer and changed RVK fallback order to use direct `GND-ID -> RVK` candidates from the official RVK dump before falling back to label-based RVK API search.
- Added periodic RVK dump update checks so the local MarcXML GND index is revalidated against the official RVK release page and rebuilt when a newer dump is published.

### Fixed

- Added the missing `pdf2image` Python dependency to `requirements.txt` and documented the Poppler runtime prerequisite for PDF OCR.
- Fixed the web GUI so completed analyses render the final results panel reliably after pipeline completion.
- Fixed the polling/results rendering path so string-valued keyword fields no longer trigger `forEach is not a function` errors in the web UI.
- Fixed stale recovery UI behavior in the web GUI so `Verbindung unterbrochen` and `Wiederherstellen` are only shown for an actually interrupted live analysis, not after a successful completion.
- Fixed MarcXML SRU DK/RVK extraction so keyword-based catalog results use the same keyword-centric shape as the rest of the pipeline and are no longer discarded before DK/RVK classification.
- Fixed HTML-escaped values in parsed LLM output and SRU search terms by decoding entities such as `&lt;` before downstream processing.
- Fixed image OCR with OpenAI-compatible vision models so ALIMA uses the selected model-specific OCR prompt, disables JSON mode for raw OCR text, omits unsupported sampling parameters for multimodal OpenAI chat requests, and no longer reports provider error strings as successful OCR output.
- Fixed RVK candidate handling so catalog-derived RVK is validated before final selection, obvious artifacts are dropped, plausible non-standard/local RVK is preserved, and the official RVK API fallback now triggers whenever no standard RVK candidate survives catalog validation.
- Fixed excessive RVK API validation fan-out by ranking catalog RVK candidates on catalog evidence first and validating only the strongest unique candidates instead of every plausible RVK-like code.
- Fixed final RVK instability by replacing the LLM's last-step RVK choice with deterministic selection from validated RVK candidates while keeping DK selection on the LLM path.

### Changed

- Clarified the semantics of legacy `dk_classifications`: the field now remains as a backward-compatible alias for final DK/RVK classification strings rather than implying DK-only output.
- Updated PyQt classification labels and selectors to use neutral DK/RVK wording, added a `classifications` alias on `KeywordAnalysisState`, and fixed PyQt title lookup/K10+ export helpers to preserve RVK systems correctly.
