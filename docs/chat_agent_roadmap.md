# Chat-Agent Roadmap

Status: 2026-05-21. Vorstufe der hier skizzierten Vision ist mit
**P-δ.3** ([commit](../) `c5fe0ae`) gelandet: `ChatAgentWorker` +
`AgentLoop`-Hooks + WhatsApp-Style-Bubbles + Tool-aware Anti-Halluc-
Prompt. Read-only Tools für alle Pipeline-Daten sind angeschlossen.

Die Roadmap unten beschreibt den Weg vom heutigen **Read-Only Chat-
Co-Pilot** zum **Chat-First / Headless-Agent** als gleichwertiges
Frontend neben GUI und CLI.

## Vision

**Dual-Mode-Endziel.**

1. **Chat-First Workflow**: Power-User arbeitet primär im Chat. Von
   Zero (nur DOI eintippen) bis zum exportierten TeX-Bericht alles
   im Chat erledigt. GUI sekundär für visuelle Inspektion.
2. **Headless-Agent**: Selbe Tool-Spec via CLI/API exponiert
   (`alima agent --doi 10.xxx`, `alima agent --input batch.txt`,
   HTTP-Endpoint). Chat-Widget ist nur eines mehrerer Frontends auf
   demselben Agent-Loop.

GUI bleibt First-Class für visuelle Pipeline-Inspektion und
Operator-Mutationen mit hohem Risiko.

## Berechtigungsmodell (Operator-Entscheidung)

`ChatConfig.autonomous_pipeline: bool = False` (Default explicit).

- **Default (explicit)**: Agent fragt vor *jedem* Pipeline-Start /
  jeder Mutation nach. Bestätigungsdialog inline im Chat (Buttons
  `▶ Starten` / `✗ Abbrechen`).
- **autonomous=True**: Agent darf bei klarer Intention selbständig
  starten. Destruktive Schritte (Cache-Writes, Mutations,
  Katalog-Calls mit Kosten) bleiben auch hier bestätigungspflichtig.

Schalter pro Session umschaltbar (Header-Toggle im Chat-Dock).

## Phasen-Plan

### P-δ.4 — Mini-Polish (~0.3 PT, ✅ Reduziert)
Nur isolierte, kleine Items. Größere Polish-Items in P-δ.5 verschoben
(Begründung: gemeinsame Renderer-Refactor lohnt sich mit Pipeline-Logger
zusammen, statt zweimal).
- **C** Auto-Show des Chat-Docks nach Pipeline-Abschluss.
- **E** Combo-Persist-Toggle "💾 Default" im Chat-Header (Default off,
  schreibt bei Wechsel `ChatConfig.default_provider/model` + save).

### P-δ.5a — PipelineChatPanel (✅ done, 2026-05-22)
`chat_dock` entfernt. `PipelineStreamWidget` + `ChatWidget` zu einem
einzigen `PipelineChatPanel` vereint (inline im PipelineTab-Right-Splitter).
- `src/ui/pipeline_chat_panel.py` (~1000 LOC) — portiert Bubble-Rendering,
  Chat-Input, Typing-Indicator, Combo-Persist, ChatSession + ChatAgentWorker.
- `src/ui/chat_widget.py` gelöscht (SystemPromptDialog migriert).
- AlimaStateBus-Plumbing: `LLMAgentStep._invoke_loop` schreibt
  `tool.called`/`tool.result`; Panel abonniert → erstmals sichtbare
  Tool-Marker für agentische Pipelines.
- 12 neue Tests grün (235 passed, 1 skipped gesamt).

### P-δ.5b — Streaming + Cancel (✅ done, 2026-05-22)
- **OpenAI-kompatibel**: Token-Streaming auch wenn Tools im Request —
  Text-Deltas live via `stream_callback`, Tool-Call-Deltas akkumuliert
  (`delta.tool_calls[i].function.arguments`), nach Stream zu `ToolCall`
  zusammengesetzt.
- **`should_stop` durch die gesamte Aufrufkette**: `generate_with_tools` +
  alle 5 Sub-Handler + `AgentLoop.run` leiten `should_stop` weiter.
  OpenAI: per-Chunk-Check → Cancel-Latenz < 1 Chunk.
  Ollama/Anthropic/Gemini/Fallback: post-blocking-call-Check → Cancel-
  Latenz = 1 LLM-Generation (verbessert von 1 Iteration).
- **Ollama**: API-Limit dokumentiert — kein Streaming mit Tools möglich.
  TODO P-δ.5c: re-evaluate.
- 17 neue Tests grün (252 passed, 1 skipped gesamt).

### P-δ.5c — Ollama Streaming + Cancel (✅ done, 2026-05-22)
- **Ollama**: Restriktion aufgehoben — Ollama SDK 0.6.1 unterstützt
  `Client.chat(stream=True, tools=[...])`. Text-Tokens streamen live via
  `message.content` pro Chunk; `tool_calls` werden atomar auf dem finalen
  `done=True`-Chunk eingesammelt (keine Delta-Akkumulation nötig).
- **Per-Chunk Cancel**: `should_stop` wird in der Stream-Schleife geprüft
  → Cancel-Latenz Ollama jetzt < 1 Chunk (vorher: 1 LLM-Generation).
- **Helper `_extract_tool_calls`**: dict/attr-safe ToolCall-Konstruktion
  für Pydantic-Models *und* dict-Returns (Test-Doubles).
- 4 neue Tests in `tests/test_streaming_with_tools.py`
  (`TestOllamaStreamingWithTools`); regression-Test invertiert.

### P-ε — Mutation Tools (✅ done, 2026-05-22)
Schreibende Operationen + Inline-Proposal-Bubble:
- `propose_keyword_replacement(old, new, reason)` — ruft
  `KeywordAnalysisState.apply_keyword_replacement` nach Bestätigung.
- `propose_dk_change(code, action='add'|'remove', reason)` — ruft
  `apply_classification_update`.
- **Inline Chat-Bubble**: `stream_text` von `QTextEdit` auf `QTextBrowser`
  umgestellt; `mutation://{audit_id}/{accept|reject}` Hyperlinks im Bubble,
  Anchor-Click routet zur `ProposalGateway`.
- **ProposalGateway** (`src/ui/chat_tools/proposal_gateway.py`) —
  cross-thread `QSemaphore`-Bridge: Tool blockt im Worker-Thread, UI-
  Thread emittiert/rendert/resolved, Tool unblockt mit User-Entscheidung.
- **Audit-Log**: neue `chat_mutations` Tabelle in `alima_knowledge.db`
  (tri-state `accepted`: pending/accepted/rejected) + Indizes.
- **ChatConfig.autonomous_pipeline**: Header-Toggle "🤖 Autonom" schaltet
  Bestätigung pro Session aus; persistiert via ConfigManager.
- 17 neue Tests (`tests/test_mutation_tools.py`).

### P-ζ — Pipeline-Orchestration (✅ done, 2026-05-26)
Chat-Agent fährt die Pipeline:
- Tool `run_pipeline(input_source, mode='classic'|'agentic', workflow=...)`.
- Tool `rerun_step(step_id, params)` — z.B. nur DK-Klassifikation mit
  anderem Modell neu fahren. Setzt vorhandenen SharedContext voraus.
- Stream-Updates aus Pipeline → Chat (Status-Marker `🔄 Step 3/5: Search`).
- Cancel via existing `should_stop`-Hook.
- Berechtigungs-Gate (siehe Berechtigungsmodell).

### P-η — Input-Beschaffung (✅ done, 2026-05-26)
Agent holt Daten selbst. Helper-Module pure-Python in `src/utils/`,
MCP-Wrapper in `src/mcp/tool_registry.py`:
- `resolve_doi(doi)` — Crossref/OpenAlex/DataCite via `UnifiedResolver`
  (vorhandener `src/utils/doi_resolver.py`; Roadmap-Begriff `fetch_doi_metadata`
  belassen als `resolve_doi`).
- `scrape_url(url)` — HTML-Reader + Content-Type-PDF-Auto-Detect: bei
  `application/pdf` → temp-Download → `pdf_extractor.extract_text`.
- `search_catalog(query)` / `search_catalog_titles(...)` — K10plus/SWB-Suche.
- `read_pdf(path, max_chars, ocr_fallback)` — `src/utils/pdf_extractor.py`,
  PyPDF2 + Quality-Heuristik + optionaler Vision-LLM-OCR-Fallback.
- `analyze_image(path, prompt, provider, model)` — `src/utils/image_analyzer.py`,
  synchroner Wrapper über `LlmService.generate_response(image=...)`.

### P-θ — Export & Reporting (✅ done, 2026-05-26)
- `export_results(source, format, output_path, validate_rvk)` —
  `src/utils/exporters.py` mit Formaten `json` | `csv` | `tex` | `marc`
  (K10+/WinIBW-Tags). Reuses `webapp.result_serialization.build_export_payload`
  als JSON-Schema-Quelle.
- `generate_report(source, template, output_path, build_pdf)` —
  `src/utils/report_renderer.py` mit Jinja2-Templates in
  `src/utils/report_templates/` (`ub_freiberg.tex.j2`, `short.tex.j2`).
  Optional `pdflatex` (kein Hard-Fail bei fehlendem PATH).
- E-Mail-Versand bewusst ausgelassen (Operator-Entscheidung).

### P-ι — Headless / CLI / API (✅ done, 2026-05-27)
Selber `AgentLoop` + Toolset headless, ohne PyQt6.
- **Shared core** (Qt-frei): `src/core/chat_prompts.py` (Prompt aus
  `pipeline_chat_panel.py` extrahiert), `src/core/headless_agent.py`
  (`HeadlessAgentRunner` + `StoppableAgentThread`), `src/core/headless_gateway.py`
  (`StdinProposalGateway` + `AutoRejectGateway`).
- **CLI**: `alima agent --doi … | --input … | --input-file … | --input-image …`
  `[--prompt] [--provider] [--model] [--temperature] [--max-iterations]
  [--autonomous] [--output] [--quiet]`. Tokens→stdout, Status/Tool-Marker→stderr,
  Ergebnis-JSON (`build_export_payload` + `agent`-Block) →`--output`/stdout.
  `src/cli/commands/agent_cmd.py`, verkabelt in `src/cli/main.py`.
- **HTTP**: `POST /agent/run` (`src/webapp/app.py`). Body: `input` (abstract/text/doi),
  `prompt`, `provider`, `model`, `temperature`, `max_iterations`, `autonomous`,
  `stream`. SSE (`text/event-stream`) mit Events `token|status|tool_call|
  tool_result|done|error`; `stream:false` ⇒ ein JSON. Per-Request isolierter
  PipelineManager.
- **Permissions**: CLI Default = stdin y/N (`StdinProposalGateway`), `--autonomous`
  ⇒ `autonomous_pipeline=True`. HTTP kann nicht prompten ⇒ ohne `autonomous`
  `AutoRejectGateway` (fail-safe reject). Cancel: `_stop_event` am Thread →
  greift in P-ζ `_resolve_should_stop` + `AgentLoop(should_stop=…)`.

## Tool-Matrix

| Tool | δ.3 | P-ε | P-ζ | P-η | P-θ | Berechtigung |
|---|---|---|---|---|---|---|
| `list_available_data` | ✅ | | | | | read |
| `get_keywords` / `_chains` / `_dk_*` | ✅ | | | | | read |
| `search_in_gnd_pool` / `validate_gnd_term` | ✅ | | | | | read |
| MCP Read-Tools | ✅ | | | | | read |
| `propose_keyword_replacement` | | ✅ | | | | mutation (Inline-Bubble) |
| `propose_dk_change` | | ✅ | | | | mutation (Inline-Bubble) |
| `run_pipeline` | | | ✅ | | | confirm/auto |
| `rerun_step` | | | ✅ | | | confirm/auto |
| `resolve_doi` | | | | ✅ | | safe |
| `scrape_url` (HTML + PDF auto-detect) | | | | ✅ | | safe |
| `read_pdf` | | | | ✅ | | safe (FS read) |
| `search_catalog` / `search_catalog_titles` | | | | ✅ | | safe |
| `analyze_image` | | | | ✅ | | safe |
| `export_results` (json/csv/tex/marc) | | | | | ✅ | safe (FS write) |
| `generate_report` (ub_freiberg/short, opt PDF) | | | | | ✅ | safe |

## Known Limitations

### Streaming-with-Tools Backend-Limit
**Behoben für OpenAI-kompatibel in P-δ.5b**: Text-Tokens streamen live,
Tool-Call-Deltas werden akkumuliert. `_generate_openai_with_tools` verwendet
jetzt immer `stream=True` wenn `stream_callback` gesetzt ist.

**Behoben für Ollama in P-δ.5c**: Ollama SDK 0.6.1 unterstützt
`stream=True` mit `tools=[...]`. Text streamt per Chunk, `tool_calls`
werden atomar vom finalen `done=True`-Chunk übernommen.

**Anthropic / Gemini**: Deferred. Weiterhin `stream=False` mit Tools.

### Cancel-Latenz
**Verbessert in P-δ.5b + P-δ.5c**: `should_stop` wird an
`generate_with_tools` übergeben. OpenAI + Ollama: per-Chunk-Check →
sub-Sekunde. Anthropic/Gemini/Fallback: post-blocking-call-Check →
1 LLM-Generation (vorher: 1 volle Iteration). Mid-token-Abbruch bei
Anthropic/Gemini nicht möglich (blocking API).

### Kein Permissions-Audit
Heute hat Chat-Agent denselben DB-Access wie Pipeline-Worker. Für
P-ε / P-ζ braucht es ein Permission-Layer das z.B. Cache-Writes
blockiert (`ChatConfig.no_cache_writes` greift heute nur für MCP-
Adapter). Audit-Log fehlt komplett.

### Provider-Abhängigkeit
Aktueller User-Test mit `gemma4:e2b` (2B Params) zeigt: Tool-Use-
Adherence und Anti-Halluc-Adherence skalieren stark mit Modellgröße.
≥7B empfohlen für produktiven Einsatz. ChatConfig sollte Default auf
ein größeres Modell setzen (z.B. `cogito:14b`).

## Offene Fragen

1. **Identitäts-Layer**: Wenn Chat-Agent eine Pipeline startet, wer
   ist "Owner" des resultierenden SharedContext? Single-User-Modus
   trivial. Multi-User später separat planen.
2. **State-Reset**: Soll `run_pipeline` den aktuellen SharedContext
   immer überschreiben? Oder Verlauf von SharedContexts behalten?
   Vorschlag: Append-only Log in DB + "current" Pointer.
3. **Tool-Discovery für CLI**: Wenn `alima agent --doi X` ohne
   Prompt aufgerufen wird — soll Agent autonom alle "vernünftigen"
   Schritte fahren? Oder Workflow-Name pflichtparameter?
4. **HTTP-Endpoint Auth**: Wenn P-ι Webapp realisiert wird — wie
   wird Auth gelöst? Token / SSO / IP-Allowlist? Separates WP.

## Referenzen

- δ.3 Plan: `~/.claude/plans/4-tasks-0-vivid-zebra.md`
- δ.1 + δ.2 Commits: `5e45019`, `adde1fa`, `c4b667f`
- δ.3 Commit: `c5fe0ae`
- AgentLoop: `src/core/agent_loop.py`
- ChatWidget: `src/ui/chat_widget.py`
- Chat-Tools: `src/ui/chat_tools/`
- WP-Übersicht: `docs/agentic_ui_workpackages.md`
