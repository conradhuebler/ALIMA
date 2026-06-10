# ALIMA AI Changelog

> **Developer log.** Detailed dated entries per feature: file lists,
> phase plans, internal refactors. For user-facing release notes
> (topic-grouped), see [`CHANGELOG.md`](CHANGELOG.md).

## 2026

### WP B+C — Debugbarkeit + E2E-Sicherheitsnetz (June 10, 2026)

Fortsetzung des Maßnahmenplans (nach WP A). Suite: 579 passed / 5 skipped.

**WP B — Debugbarkeit:**
- **Zentrale Logging-Konfiguration komplett**: GUI und CLI nutzten
  `logging_utils.setup_logging` bereits; die Webapp (vorher nur
  `basicConfig`, Konsole) nutzt es jetzt auch → Konsole + `alima_webapp.log`,
  `LOG_LEVEL=DEBUG` env-Var wird auf Stufe 2 gemappt (`src/webapp/app.py`).
- **print() → Logger**: `swb_suggester.py` (18×) und `lobid_suggester.py` (4×)
  auf `self.logger.debug` umgestellt; die zwei „could not extract"-Fälle in
  SWB-Einzeltreffer-Seiten sind jetzt unbedingte `logger.warning` (Datenverlust).
  Nicht angefasst: `src/core/lobid_subjects.py` (13×) und
  `src/core/katalog_subject.py` (10×) — werden von nichts importiert,
  **Dead-Code-Kandidaten für WP E**; `registry.py`-Treffer sind Docstring-Beispiele.
- **except:pass-Audit** (~55 Stellen): nackte `except:` auf konkrete Typen
  eingegrenzt (`OSError` bei unlink-Cleanups ×6, `ValueError/TypeError` bei
  Datums-/JSON-Parsing ×3); Silent-Swallows mit Logging versehen
  (`unified_provider_tab` Modell-Lookup/-Persist → warning,
  `pipeline_config_dialog` Prompt-Fallback → debug, Bus-Emits in
  `pipeline_manager` → warning bzw. `llm_agent_step`/`pipeline_utils` → debug);
  übrige Best-Effort-Stellen mit Begründungskommentar. Übersprungen:
  `pipeline_chat_panel.py` (3 Stellen, WP12-Datei).

**WP C — E2E-Smoke-Tests** (`tests/test_e2e_smoke.py`, LLM an der
LlmService-Grenze gemockt, Netzwerk an SearchCLI-/Tool-Grenze gefakt):
- Klassische Pipeline: `execute_complete_pipeline` initialisation → search →
  keywords → `KeywordAnalysisState` mit Keywords, Suchergebnissen, Streaming.
- Agentisch: `alima_classic.yaml` (7 Steps) durch `WorkflowExecutor` mit
  `LLMAgentStep` + deterministischen Funktionen; Kontext trägt Ergebnisse
  durch die ganze Kette; plus Negativ-Test (LLM down → `report.success=False`).
- `AgentLoop` Multi-Turn: 2 Tool-Calls + finale Antwort über 3 LLM-Turns,
  Tool-Results landen in der Konversation, Hooks feuern.

**Dabei gefundener+behobener Silent-Fail** (vom Negativ-Test aufgedeckt):
`AgentLoop` wandelte LLM-Exceptions in `content="Error: …"` um und
`LLMAgentStep` wertete das als Erfolg → Workflow lief mit Müll weiter und
meldete `success=True`. Jetzt: `AgentResult.error`-Feld (rückwärtskompatibel),
`AgentLoop` setzt es, `LLMAgentStep` lässt den Step fehlschlagen
(`src/core/data_models.py`, `src/core/agent_loop.py`,
`src/core/agents/steps/llm_agent_step.py`).

### WP A — Fehler sichtbar machen / Silent-Fail-Härtung (June 10, 2026)

Erste Stufe des Maßnahmenplans aus der Basis-Bewertung (Plan-Datei
`ich-h-tte-gerne-eine-snappy-backus.md`): Fehler, die bisher geschluckt
wurden und leere Ergebnisse als Erfolg erscheinen ließen, werden jetzt
gemeldet. Tests: `tests/test_error_visibility.py` (10 Negativ-/Positiv-Tests);
Suite 575 passed / 5 skipped.

- **Worker**: `PipelineWorker` hat neues Signal `pipeline_error(str)` und
  emittiert es im bisher stummen `except`-Block (`src/ui/workers.py`);
  `PipelineTab.on_pipeline_error` zeigt Dialog, setzt Status, reaktiviert
  den Start-Button (`src/ui/pipeline_tab.py`).
- **Klassische Pipeline stoppt bei Schritt-Fehlschlag**: `_execute_next_step`
  hatte keinen `else`-Zweig für `success=False` — die Pipeline lief nach
  einem fehlgeschlagenen Schritt weiter (auto_advance), und Schritte, die
  „sauber" `False` zurückgaben (z. B. DK-Klassifikation), lösten gar keinen
  `step_error_callback` aus. Jetzt: Status `error`, Callback genau einmal,
  Bus-Event `state.pipeline_step` mit `status="error"` + `error`-Payload,
  kein Auto-Advance (`src/core/pipeline_manager.py`).
- **WorkflowExecutor (agentisch)**: try/except um Step-Konstruktor,
  `step.execute()` und `ConditionalEngine.evaluate` → `StepResult(success=False)`
  statt Thread-Crash; kaputte `when:`-Bedingung ist Step-Fehler, kein
  stilles Überspringen (`src/core/agents/workflow_executor.py`).
  Hinweis: `BaseStep.execute` fing `run()`-Exceptions schon ab — ungeschützt
  waren Konstruktor, Condition und execute-Overrides.
- **Parse-Fehler ≠ leeres Ergebnis**: unparsebare LLM-Antwort bei der
  Initialisierung wirft jetzt `ValueError` mit Response-Preview statt mit
  0 Schlagwörtern „erfolgreich" weiterzulaufen (`src/utils/pipeline_utils.py`);
  `extract_keywords_from_response` loggt WARNING bei leerem Resultat aus
  nicht-leerer Antwort (`src/core/processing_utils.py`); generischer Pfad in
  `alima_manager._create_analysis_result` warnt (kein Raise, da
  `match_keywords_against_text`-Fallback legitime Teilergebnisse liefert).
- **Suggester: Quelle-down ≠ kein Treffer**: `BaseSuggester` bekommt
  `last_errors` + `_record_search_error` (immer `logger.warning`, nicht mehr
  `if self.debug: print`). SWB cached fehlerbehaftete Suchen **nicht** mehr
  (vorher wurde ein API-Ausfall dauerhaft als „kein Treffer" persistiert).
  Propagation: Suggester → `MetaSuggester` → `SearchCLI.last_errors` →
  `execute_gnd_search` streamt `⚠️ Quelle(n) fehlgeschlagen für '<term>'`
  und eine Abschluss-Warnung an GUI/CLI/Webapp.
- **Zurückgestellt** (WP12-Dateien, Vermischung vermeiden): Rendering des
  `status="error"`-Bus-Events im Chat-Panel/Webapp.

### WP12 — Unified Render Layer (GUI ↔ Webapp) (June 9, 2026)

GUI and webapp rendered the same pipeline data with separately-maintained
chrome (the WP2 DK/GND divergence). Now both render from **one** CSS + JS
render layer driven by a versioned JSON render-event protocol over two
transports. Spec: [`docs/wp12_unified_render_layer.md`](docs/wp12_unified_render_layer.md).

- **WP12.1 — Asset extraction**: the theme CSS + DOM-dispatcher JS were lifted
  out of the inline `_HTML_TEMPLATE` in `src/ui/web_log_view.py` into
  `src/webapp/static/alima_render.{css,js}` (single source). `WebLogView`
  inlines them at construction (lowest-risk QWebEngine load path); the webapp
  serves them as static assets. All content CSS is **scoped under `#log`** so it
  can load into the multi-element webapp page without clobbering its theme or
  page-level `<details>`/`<a>`/`<table>`. Font size moved to the `--alima-fs`
  custom property. GUI document chrome (page bg, scrollbars) stays in the
  scaffold.
- **WP12.2 — Event protocol + producer abstraction**: new Qt-free
  `src/core/render_events.py` (event builders 1:1 with the JS funcs +
  `RenderTransport` protocol + `MockTransport`); new `src/ui/render_transport.py`
  (`WebLogViewTransport`). `UnifiedMessageRenderer` now emits JSON render events
  to an injected transport instead of calling `WebLogView` directly; historical
  callers passing a `WebLogView` are auto-wrapped (back-compat, no call-site
  change). Events are append-only + idempotent per id; `block` events carry a
  semantic `kind` so Tier-3 frontends can drop GUI-only chrome (`proposal`).
- **WP12.3 — Webapp consumes shared chrome**: the webapp drives the *same*
  `UnifiedMessageRenderer` producer headless via a per-session
  `WebSocketRenderTransport`; events are buffered on the `Session` (monotonic
  `seq`) and broadcast over the WS (`render_events` field on `status`/`complete`,
  full replay on reconnect via a per-connection cursor; polling cursor for the
  fallback). `app.js` dispatches them into a `#log` region in the results panel
  via the shared funcs, deduping by `seq`. The webapp **keeps its 5-step widget**
  (WP9 Tier-3) and only adopts the DK/GND result-card chrome.
- **WP12.4 — Consolidation**: DK/GND card HTML is now produced by shared
  `PipelineResultFormatter.format_dk_search_card_html` /
  `format_dk_classifications_card_html`, called by **both** the GUI panel
  (`pipeline_chat_panel.py`) and the webapp — one maintenance location. No
  duplicate chrome CSS to remove (the `#log` scoping is non-overlapping with the
  webapp's `.classification-*` summary cards, which are kept).
- **Follow-up (reverse port)**: the webapp's nicer **structured DK/RVK badge
  cards** were lifted into the shared layer — new
  `PipelineResultFormatter.normalize_classifications` +
  `format_classification_badge_card_html`, with the `.classification-*` CSS
  ported into `alima_render.css` (scoped `#log`, recoloured for the dark
  surface). `format_dk_classifications_card_html` (GUI agentic-chat log + webapp
  `#log`) now renders the badge card with system badges (DK/RVK), RVK
  validation badges (standard / nicht standard / API-Fehler), a hit-count
  confidence badge, and per-code catalog titles. The Pipeline-Tab keeps its own
  `format_dk_classifications_html` confidence card (untouched; `test_pipeline_utils`
  green). This is the symmetry payoff of WP12: the GUI being a QWebEngineView
  means webapp render components flow back into it through the same shared layer.

Tests: `tests/test_unified_message_renderer.py` gains `MockTransport`
event-emission + `WebLogViewTransport`-mapping classes; new
`tests/test_webapp_render_events.py` covers the session buffer, cursors,
headless producer, and an end-to-end WS broadcast + reconnect-replay
(`fastapi.testclient`). Full suite: 556 passed, 5 skipped.

**Caveats (conservative self-assessment).** Verified via headless tests
(`QT_QPA_PLATFORM=offscreen`, `TestClient`) and JS `node --check` — **not**
visually confirmed in a running GUI or browser. The webapp now shows DK/GND
classifications in both its compact summary panel **and** the new shared `#log`
cards (complementary, like the GUI, but not yet de-duplicated by an operator UX
review). Streaming/assistant/collapsible events are wired on the webapp client
but only exercised in the classic pipeline's DK/GND path server-side; the
agentic tool-bus chrome is not emitted to the webapp.

### Chat/log rendering moved to QWebEngineView — reliable collapse + live streaming (June 9, 2026)

The chat/pipeline log rendered everything into a single `QTextBrowser` via
`QTextCursor` surgery (`UnifiedMessageRenderer`). Two regressions followed the
June 8 "declutter" change: (1) collapsible blocks were unreliable — "once
expanded, won't close" — because `_rerender_tool_call_block` re-rendered a block
in place by `setUserState` marker, which broke when the expanded body spanned
more than one `QTextBlock` or when concurrent streaming shifted block positions;
(2) intermediate LLM reasoning no longer streamed live.

**Redesign** (operator chose QWebEngineView; collapse-first):
- New `src/ui/web_log_view.py` — `WebLogView(QWidget)` wrapping a `QWebEngineView`.
  Collapsible blocks are native `<details>/<summary>` (toggle is 100% browser-side
  → no Python re-render, reliable even mid-stream). Streaming appends text nodes to
  an isolated `<div>`; markdown is rendered once on finalize. JS calls are queued
  until `loadFinished`; link clicks (`mutation://`, `http(s)://`) route back via
  `acceptNavigationRequest` → `link_clicked` (replaces `QTextBrowser.anchorClicked`).
- `UnifiedMessageRenderer` keeps its public API + `history` contract; internals now
  emit HTML strings into the `WebLogView` instead of cursor surgery. Deleted the
  cursor machinery (`_rerender_tool_call_block`, `_tool_call_blocks`, `setUserState`);
  `toggle_tool_call` is now a server-side mirror only.
- Panel + both mini-logs (`pipeline_chat_panel.py`, `analysis_review_tab.py`,
  `image_analysis_tab.py`) construct `WebLogView` instead of `QTextBrowser`.
- **Import-order constraint:** `QtWebEngineWidgets` must be imported before the
  `QApplication` — explicit early import added to `alima_gui.py`.

**Backend streaming-with-tools** (`llm_service.py`, partial P-δ.5/#7): Anthropic
`_generate_anthropic_with_tools` now uses `messages.stream()` + `get_final_message()`
to stream text deltas when a `stream_callback` is set (Ollama/OpenAI already did);
Gemini still completes-then-delivers (noted in-code).

**Live LLM stream → collapsible block** (follow-up): the flat inline streaming
line is replaced by an expanded `<details>` block. `start_streaming_line` opens it
open, `render_streaming_token` appends to its body live, `end_streaming_line`
collapses it and writes a one-line text preview into the summary. Both classic
(`step_id=""`) and agentic (`step_id="agentic"`) LLM output already route through
these three methods (`workers.py` → `on_llm_stream_token` → panel), so streamed
content — including the agent's initial keywords — is now visible live and then
folded away with a preview, consistent with the deterministic step summaries.
Caveat: agentic prose still passes `_AgenticStreamFilter` (raw-JSON suppression,
off when `ChatConfig.agentic_verbose`); content emitted as tool-call JSON rather
than prose is still filtered.

**Dependency:** `PyQt6-WebEngine==6.10.0` (+ `PyQt6-WebEngine-Qt6==6.10.2`) added to
`requirements.txt` — pulls in a Chromium runtime.

**Tests:** `test_unified_message_renderer.py` rewritten against a mock `WebLogView`
(captured HTML strings) — native collapse means the body is always in the DOM and
toggling is a mirror. Suite bootstrap (`tests/__init__.py` + `tests/conftest.py`)
imports WebEngine before any `QApplication`, creates the app with a non-empty argv,
and swaps a lightweight `WebLogView` stub so headless Chromium isn't constructed in
unit tests. **531 passed, 5 skipped.**

**Caveat (per self-assessment rules):** verified that native `<details>` toggling is
reliable while streaming (expand → re-close → re-expand, stream intact) and that the
suite is green — this does not prove correctness across all providers/inputs. Markdown
is still rendered post-stream (unchanged). The QWebEngine route adds a heavyweight
Chromium dependency and three render processes in the running app.

### Unified DK/GND result rendering + agentic-log declutter (June 8, 2026)

Commit `0cfba1a`. Pipeline-Tab and the agentic chat panel rendered the same
pipeline data differently (catalog research, final DK/RVK notations, GND hits).
Root cause: divergent ad-hoc formatters per surface. Consolidated into shared
formatters and fixed several agentic-mode display bugs.

**Shared formatters** (`src/utils/pipeline_utils.py` → `PipelineResultFormatter`,
single source of truth, pure-Python, unit-tested):
- `format_dk_classifications_html` (HTML fragment, confidence colours + title list),
  `format_dk_search_results_text`, `split_classification_code`,
  `get_titles_for_dk_code`.
- `select_dk_title_source` — picks the title-carrying source regardless of mode
  (classic stores the rich list in `dk_search_results_flattened`, agentic in
  `dk_search_results`; the other field is keyword-centric / thin). **This field
  inversion between modes is the recurring trap behind the agentic display bugs.**
- `flatten_gnd_hits` (dict / List[SearchResult] / flat `gnd_entries` → dedup rows),
  `extract_selected_gnd_keys` (final keywords → gnd-id + label sets).

**Fixes**:
- Agentic completion (`pipeline_tab._sync_classical_tabs_from_state`) cleared the
  Katalog-Recherche view and dropped titles on final notations — now uses
  `select_dk_title_source`.
- GND-Recherche tab: flat text → sortable `QTableWidget` (Begriff / GND-ID /
  Häufigkeit / Auswahl) + "nur ausgewählte" filter; completion no longer collapses
  to bare search terms (`_populate_gnd_hits` / `_render_gnd_hits_table` / `_filter_gnd_hits`).

**Agentic GUI polish**:
- Input prompt → collapsible, timestamped 📥 block via
  `UnifiedMessageRenderer.render_collapsible` + `state.pipeline_prompt` /
  `state.pipeline_prompt_done` bus events (emitted in `llm_agent_step._emit_prompts`
  / `_emit_prompt_done`, reflection tagged `kind="reflection"` → 🔍). Prompt no
  longer streamed inline (killed the duplicate dump). Added `render_html_block`.
- Decluttered the agentic log: compact MetaAgent/LLMAgent banners, hidden empty
  `[]` stream tag, dropped duplicate "Pipeline gestartet".

**Open follow-ups / findings** (not yet done):
1. **Agentic GND `Häufigkeit` column = 0** — the agentic `search_results` structure
   (`SharedContext.to_analysis_state`) carries no per-entry count; thread it through
   `gnd_entries` to populate the column.
2. **GND "only free keywords" — cache-vs-live hypothesis unverified**: the display
   fix is done, but whether the mapping-first cache narrows results to the exact
   GND mapping (vs the broad live Lobid aggregation) needs a runtime check.
3. **"LLM Antwort:" prefix on agentic orchestration**: orchestration text and the
   real LLM response share one streaming line / step_id `agentic`, so orchestration
   inherits the misleading prefix. Clean separation (orchestration as discrete log
   lines) needs a small stream-routing refactor.
4. **Duplicate selection logic** in `analysis_review_tab.py:~595-639`
   (`_split_classification_code` + title lookup) — consolidate onto the shared
   `PipelineResultFormatter` helpers.
5. **GUI runtime verification** — all changes are unit-tested (499 green) but not
   GUI-verified end-to-end; confirm in the running app.

### Chat-Agent P-η + P-θ: Input-Beschaffung + Export & Reporting (May 26, 2026)

Closes both open chat-agent roadmap phases (`docs/chat_agent_roadmap.md`).
The agent can now drive the full DOI/URL/PDF/Image → Pipeline → Export/Report
workflow without operator GUI interaction.

**New helper modules** (`src/utils/`, pure-Python, no Qt):
- `pdf_extractor.py` — PyPDF2 text extraction + quality heuristic
  (`_assess_text_quality`) + optional Vision-LLM OCR fallback via pdf2image.
  Extracted from `unified_input_widget.py:93-158`.
- `image_analyzer.py` — sync wrapper over `LlmService.generate_response(image=...)`
  with generator coalescing. Default `DEFAULT_PROMPT` = OCR. Extracted from
  `ImageAnalysisWorker`.
- `exporters.py` — `export_json/csv/tex/marc` + `load_state('latest'|file|abspath)`
  + `default_output_path`. Reuses `webapp.result_serialization.build_export_payload`
  as JSON schema source. K10+/WinIBW tags (5550/6700) via `generate_k10plus_lines`.
- `report_renderer.py` + `report_templates/{ub_freiberg,short}.tex.j2` — Jinja2 LaTeX
  with custom delimiters `(((  )))` / `((* *))` to avoid LaTeX brace collision.
  Optional pdflatex two-pass build; missing binary is non-fatal.

**New MCP tools** (`src/mcp/`):
- `read_pdf(path, max_chars, ocr_fallback, provider, model)`
- `analyze_image(path, prompt, provider, model, temperature)`
- `export_results(source, format, output_path, validate_rvk)`
- `generate_report(source, template, output_path, build_pdf)`
- `scrape_url` extended with Content-Type / .pdf-suffix auto-detect →
  temp download → `pdf_extractor.extract_text`.

**ToolRegistry**: gains optional `llm_service` constructor arg; `_get_llm_service()`
lazy-inits from config if not injected. New `export` tool-set + `EXPORT_TOOLS` list
in `tool_schemas.py`.

**Tests**: `tests/test_input_export_tools.py` (30 tests, all pass) covers
extractor, analyzer, all 4 exporter formats, both templates, MCP dispatch +
scrape PDF branch.

**Doku**: `docs/chat_agent_roadmap.md` (P-η/P-θ marked done, tool matrix updated),
`src/mcp/CLAUDE.md` + `src/utils/CLAUDE.md` mention new modules. Plan file:
`~/.claude/plans/p-input-beschaffung-immutable-spring.md`.

**Operator decisions** baked in: kept `resolve_doi` name (no rename to
`fetch_doi_metadata`); Jinja2 + `paper/`-style templates for report; e-mail
delivery deliberately deferred.

### P-η: Provider-Variants + Seed-Retrofit (May 18, 2026)

WP10 Foundation Phase 3/3. Closes the agentic reproducibility blocker
(WP2 Sek 3) and seeds the family-aware prompt-routing.
Pre-tag: `wp10-pη-pre`.

**Seed Retrofit** (WP11 Sek 8 — 7+1 sites):
- `LlmService.generate_with_tools()` gains `seed: Optional[int] = None`.
  Dispatch forwards seed to all sub-handlers except Anthropic.
- `_generate_ollama_native_with_tools`, `_generate_openai_with_tools`,
  `_generate_gemini_with_tools`, `_generate_text_fallback_with_tools`
  accept seed and propagate to provider API.
- `_generate_anthropic_with_tools` **deliberately skipped** —
  Anthropic SDK has no `seed` parameter and operator config is empty.
  Dispatch omits seed entirely when routing to Anthropic; text-path
  Anthropic seed setting at `llm_service.py:1851` is unchanged
  (silently ignored by SDK). See operator decision in P-η plan.
- `AgentLoop.run()` gains seed param; forwards to both main and
  force-final `generate_with_tools()` calls.
- `BaseSharedContext` + `SharedContext` add `seed` field with
  serde symmetry in `to_dict`/`from_dict`.
- `LLMAgentStep._llm_params()` resolves
  `step.llm.seed > context.seed > None` and forwards via
  `_invoke_loop()` to `AgentLoop.run(seed=...)`.
- `shared_context.py` 3 hardcoded `seed=None` in `LlmKeywordAnalysis`
  factories replaced with `seed=self.seed`.

**Workflow YAML seed schema** (Track C):
- All 6 workflows (`alima_classic`, `alima`, `catalog_search`,
  `synonym_expansion`, `title_list_search`, `batch_metadata`) gain
  optional `settings.seed: null` field.
- `WorkflowExecutor.run()` propagates `settings.seed` to
  `context.seed` when the latter is unset (caller wins otherwise).
- Per-step override remains via `steps[].llm.seed`.

**Capability YAML** (WP11 Sek 3, Track A):
- New file: `config/model_capabilities.yaml` covering 3 providers
  (openai_compatible, ollama, gemini) × 12 model patterns × 10 flags
  (json_mode, tool_use, vision, max_context_tokens, seed_support,
  streaming, thinking_tokens, parallel_tool_calls, system_prompt,
  family). Anthropic excluded by operator decision.
- New helpers in `src/utils/model_capabilities.py`:
  `load_capabilities_yaml(path)`, `get_capability(provider, model, flag,
  default)`, `reset_capability_cache()`. 3-tier lookup: exact →
  fnmatch wildcard → caller default. Cached per-path.
- Existing `KNOWN_CAPABILITIES` regex registry untouched (chunking
  threshold lookup unaffected).

**Prompt Variants** (WP11 Sek 5, Track D):
- 9 new family-specific variants added to `prompts.json`:
  - `keywords` × {thinking, instruct-open, openai-chat} (+3)
  - `dk_classification` × {thinking, instruct-open, openai-chat} (+3)
  - `initialisation` × {thinking, instruct-open} (+2)
  - `dk_list` × instruct-open (+1, on top of existing 2)
- All existing 5-tuple variants canonicalized to 6-tuple with
  `seed="0"`. PromptService 3-tier selector unchanged.
- Backup at `prompts.json.pre-pη.bak`.

**Tests** (Track E, +22 tests):
- New `tests/test_llm_service_seed.py` (12 tests): handler dispatch,
  ollama options pass-through, AgentLoop forward, SharedContext
  roundtrip.
- New `tests/test_model_capabilities_yaml.py` (10 tests): YAML load,
  3-tier resolution, default fallback, shipped-YAML smoke.
- `tests/test_agents_v2.py` +4: settings/context/step seed resolution.
- Full suite: 188 passed / 6 pre-existing failures in
  `test_pipeline_utils.py` (unrelated to P-η, verified via stash).

**Verification**:
- 22 new tests green; 0 regressions.
- PromptService picks correct family variant for `llama3.1:8b`
  (instruct-open), `qwen2.5:32b` (thinking), `gpt-4o-mini` (openai-chat),
  `exotic-model:1b` (default fallback).
- End-to-end seed reproducibility smoke test deferred to manual run
  (requires Ollama runtime).

**Out of scope**: Anthropic family + claude variants, test matrix
(WP11 Sek 9), per-step provider-mix UI (WP11 Sek 10),
`KNOWN_CAPABILITIES` → YAML migration of existing consumers.

**Next phase**: P-γ — SingleStepDialog (4 PT, first user-visible win).

### v4 Agent Workflow System (April 22, 2026)
- **Replaces MetaAgent + SubAgents**: The hardcoded 4-SubAgent pipeline (`KeywordExtractionAgent`, `SearchAgent`, `KeywordSelectionAgent`, `ClassificationAgent`) was deleted. Agent dispatch now runs through the generic v4 `WorkflowExecutor`.
- **Plan**: Option B from Agent-System-Restructuring plan — Generic LLMAgentStep + DeterministicStep + plugin registry.
- **Phase 1-2 (Foundation + Migration)**:
  - New files: `registry.py`, `workflow_loader.py`, `workflow_executor.py`, `context_path.py`, `steps/{base_step,llm_agent_step,deterministic_step}.py`, `deterministic_functions.py`
  - `SharedContext.extra: Dict` added for non-ALIMA fields + `${steps.X.Y}` / `${extra.Y}` context-path resolver
  - `workflows/alima_classic.yaml` reproduces the classic 4-step pipeline in v4 schema
- **Phase 3 (PoC workflows)**:
  - `workflows/catalog_search.yaml` — multi-source catalog lookup (SWB + Lobid + catalog) with optional LLM ranking
  - `workflows/synonym_expansion.yaml` — single keyword → GND entry → LLM expansion → validated GND candidates
  - `workflows/batch_metadata.yaml` — bulk GND-ID metadata fetch with optional Lobid fallback
- **Phase 4 (CLI/GUI integration)**:
  - New CLI: `alima workflow <name> [--input|--input-file|--output|--only-step]` + `alima workflows list`
  - `PipelineConfigDialog` gained a workflow-selection `QComboBox` populated from discovered v4 YAMLs
  - Fixed pre-existing argparse conflict: CLI `--step` (provider override, `append`) vs single-step agentic `--step`; renamed the second to `--only-step`
- **Phase 5 (cleanup)**:
  - Deleted: `meta_agent.py`, `base_sub_agent.py`, `keyword_extraction_agent.py`, `search_agent.py`, `keyword_selection_agent.py`, `classification_agent.py`
  - Archived: `workflows/{meta_agent_default,default_alima,extended,minimal}.yaml` → `workflows/legacy/` (no longer discovered)
  - Removed MetaAgent fallback branch from `PipelineManager._start_agentic_pipeline()`
  - Default `workflow_name` changed from `meta_agent_default` → `alima_classic`
  - `tests/test_agents.py` reduced to `SharedContext` + `ToolResultCache` + `CachingToolRegistry` coverage; MetaAgent/SubAgent tests removed (replacement coverage in `tests/test_agents_v2.py`, 59 tests total)
- **Kept unchanged**: `CachingToolRegistry`, MCP tool layer, `agent_loop.py`, `LlmService`, rigid `pipeline_utils.py` path

### WebApp Auto-Save & Recovery System (January 6, 2026)
- **Complete reliability upgrade** for long-running pipeline analyses in web interface
- **Auto-Save Infrastructure**: Incremental JSON saving after each pipeline step
  - Auto-save directory: `/tmp/alima_webapp_autosave/` with session-specific files
  - Metadata tracking: session_id, timestamp, last_step, status
  - Uses existing `PipelineJsonManager` for consistent serialization
- **Extended WebSocket Timeout**: Increased from 5 minutes to 30 minutes
  - Heartbeat mechanism: Sends heartbeat every 5 seconds to maintain connection
  - Prevents timeout during long DK searches (100+ keywords)
  - Frontend filters heartbeat messages (no console spam)
- **Recovery Mechanism**: Complete result restoration after connection loss
  - New API endpoint: `GET /api/session/{id}/recover`
  - Auto-detection of WebSocket errors (code 1006, 1011)
  - Recovery UI: Orange "🔄 Ergebnisse wiederherstellen" button with status messages
  - Full result reconstruction using shared `_extract_results_from_analysis_state()` helper
- **Auto-Cleanup**: Automatic deletion of old auto-save files (>24h) on webapp startup
- **Progress Enhancement**: DK search now shows percentage progress `[idx/total] (pct%)`
- **Code Quality**: DRY principle - shared result extraction logic between callback and recovery
- **Backward Compatibility**: Old sessions without auto-save continue to work
- **Files Modified**:
  - `src/webapp/app.py`: +4 functions, +1 endpoint, auto-save infrastructure
  - `src/webapp/static/index.html`: Recovery button + message span
  - `src/webapp/static/app.js`: +2 recovery functions, WebSocket handler enhancements
  - `src/utils/pipeline_utils.py`: Percentage display in DK search
  - `src/webapp/CLAUDE.md`: Documentation update

### DK Deduplication Statistics Display (January 2026)
- **Phase 2 Complete**: Comprehensive statistics visualization for DK classification deduplication
- **CLI Statistics Display**: New `format_dk_statistics()` in `show-protocol` detailed mode
  - Shows deduplication metrics: original→deduplicated count, duplicates removed, rate, token savings
  - Top 10 most frequent classifications with keyword provenance and title counts
  - Keyword coverage summary showing keywords→DK codes mapping
- **GUI Statistics Tab**: New "📊 DK-Statistik" tab in AnalysisReviewTab (index 9)
  - Deduplication Summary box with 5 key metrics
  - Top 10 table with rank, DK code, type, count, keywords, and color-coded confidence
  - Keyword Coverage table showing keyword→DK codes relationships
  - Color-coded confidence indicators: Green (>50 titles), Teal (>20), Yellow (>5), Red (<5)
- **Critical Bug Fixes**:
  - Fixed `dk_statistics` not being loaded from JSON in CLI display functions (3 locations)
  - Fixed incorrect tab navigation indices in GUI `on_step_selected()` method
  - Added missing navigation cases: chunk_details, k10plus, dk_statistics
- **Backward Compatibility**: Old JSON files without statistics handled gracefully with fallback messages
- **Files Modified**: `src/alima_cli.py`, `src/ui/analysis_review_tab.py`, `CLAUDE.md`

## 2025

### Unified Database Configuration (November 2025)
- Eliminated duplicate `SystemConfig.database_path` + `DatabaseConfig.sqlite_path` → single source of truth
- Implemented OS-specific default paths (Windows, macOS, Linux) via `get_default_db_path()`
- Singleton pattern for UnifiedKnowledgeManager with thread-safe `__new__()` override
- Automatic backward compatibility migration for old configs
- All 12 UnifiedKnowledgeManager instantiations now use singleton automatically

### K10+/WinIBW Catalog Export (October 2025)
- Direct export in K10+/WinIBW format for seamless catalog integration
- GUI: New "K10+ Export" Tab with Copy-Button
- CLI: `--format k10plus` for direct Copy-Paste
- Configuration: K10PLUS_KEYWORD_TAG, K10PLUS_CLASSIFICATION_TAG

### DK Classification Transparency (October 2025)
- Automatic display of which catalog titles led to each DK classification
- GUI: PipelineStreamWidget shows sample titles during DK search, AnalysisReviewTab with color coding
- CLI: show-protocol with DK titles in detailed/compact/k10plus format

### Protocol Display CLI Command (October 2025)
- `show-protocol` command for displaying pipeline results from JSON files
- Three modes: `--format detailed` (readable), `--format compact` (CSV), `--format k10plus` (catalog export)

### Batch Processing System (August 2025)
- Complete batch processing engine using PipelineManager for full pipeline execution
- ALL Source Types Supported: DOI (via doi_resolver), PDF (PyPDF2 + LLM-OCR fallback), TXT, IMG (vision model), URL (BeautifulSoup4)
- Batch Review UI: Toggle mode for batch overview vs. detail view, table with Status/Source/Keywords/Date/Actions
- Continue-on-error vs. stop-on-error modes with detailed error reporting
- Resume functionality for interrupted batches via JSON persistence
- Pipeline configuration inheritance from global settings

### Unified Logging System (August 2025)
- Central logging infrastructure with 4-level verbosity system (0=Quiet, 1=Normal, 2=Debug, 3=Verbose)
- CLI: `--log-level` argument (0-3, default=1)
- GUI: Uses level 1 (Normal) by default
- Setup function: `setup_logging(level)` with automatic third-party suppression
- Result output respecting quiet mode: `print_result()` function

### Three-Mode CLI System (July 2025)
- Smart Mode: Uses task preferences from config.json automatically
- Advanced Mode: Manual provider|model override with `|` separator
- Expert Mode: Full parameter control (temperature, top-p, seed)

### Vertical Pipeline UI (June 2025)
- Chat-like vertical workflow with 5 pipeline steps
- Visual status indicators: ▷ (Pending), ▶ (Running), ✓ (Completed), ✗ (Error)
- Auto-Pipeline button for one-click complete analysis
- Integrated input tabs (DOI, Image, PDF, Text) in first step
- Real-time result display in each step
- Direct integration with PipelineManager for workflow orchestration

### Global Status Bar (June 2025)
- Unified provider information display across all tabs
- Real-time cache statistics (entries count, database size)
- Pipeline progress tracking with color-coded status
- Auto-updating every 5 seconds for live monitoring
- Integration with LlmService and CacheManager

### Pipeline Manager (May 2025)
- Orchestrates complete ALIMA workflow using existing AlimaManager logic
- 5-step pipeline: Input → Keywords → Search → Verification → Classification
- Uses proven `KeywordAnalysisState` for data management
- UI callback system for real-time progress updates
- Auto-advance functionality for seamless workflow
- Refactored to use shared `PipelineStepExecutor` from utils

### Automated Data Flow (May 2025)
- AbstractTab automatically sends results to AnalysisReviewTab
- New `analysis_completed` signal in AbstractTab
- `receive_analysis_data()` method in AnalysisReviewTab
- Seamless workflow progression without manual data transfer
