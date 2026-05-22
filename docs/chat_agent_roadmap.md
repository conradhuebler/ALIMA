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

### P-ε — Mutation Tools (1.5 PT)
Schreibende Operationen + Proposal-Dialog:
- `propose_keyword_replacement(old, new, reason)`
- `propose_dk_change(code, action='add'|'remove', reason)`
- Mutation-Proposal-Dialog (Diff-View, Accept/Reject).
- Audit-Log in `alima_knowledge.db.chat_mutations` Tabelle.
- KAS-Mutations API (vorbereitet in δ.1 follow-up `adde1fa`).

### P-ζ — Pipeline-Orchestration (2 PT)
Chat-Agent fährt die Pipeline:
- Tool `run_pipeline(input_source, mode='classic'|'agentic', workflow=...)`.
- Tool `rerun_step(step_id, params)` — z.B. nur DK-Klassifikation mit
  anderem Modell neu fahren. Setzt vorhandenen SharedContext voraus.
- Stream-Updates aus Pipeline → Chat (Status-Marker `🔄 Step 3/5: Search`).
- Cancel via existing `should_stop`-Hook.
- Berechtigungs-Gate (siehe Berechtigungsmodell).

### P-η — Input-Beschaffung (1.5 PT)
Agent holt Daten selbst:
- `fetch_doi_metadata(doi)` — Crossref via vorhandenem `CrossrefTab`-Code.
- `fetch_url(url)` — HTML/PDF-Scraping. Reuse `pdf_processor` + neuer
  HTML-Reader. PDF-URLs auto-detect.
- `search_catalog(query)` — K10plus/SWB-Suche, Kandidaten-Liste, User wählt.
- `read_pdf(path)` — Lokaler PDF-Pfad → Abstract-Extraktion.
- `analyze_image(path)` — Buchcover/Inhaltsverzeichnis via existing
  `image_analysis_tab.py` Codepfad. Liefert OCR + LLM-Klassifizierung.

### P-θ — Export & Reporting (1 PT)
- `export_results(format='json'|'csv'|'tex'|'marc')` — schreibt SharedContext.
- `generate_report(template='ub_freiberg'|'short')` — TeX/PDF-Report via
  `paper/`-Templates.
- E-Mail-Versand (optional, hinter Berechtigung).

### P-ι — Headless / CLI / API (2 PT)
- `alima agent --doi 10.xxx --workflow=alima_classic --output=results.json`
  — CLI-Frontend benutzt denselben `AgentLoop` + Toolset.
- HTTP-Endpoint: `POST /agent/run` mit JSON-Spec.
- SSE-Streaming für Token/Tool-Events.
- Identische Permissions-Logik wie GUI-Chat.

## Tool-Matrix

| Tool | δ.3 | P-ε | P-ζ | P-η | P-θ | Berechtigung |
|---|---|---|---|---|---|---|
| `list_available_data` | ✅ | | | | | read |
| `get_keywords` / `_chains` / `_dk_*` | ✅ | | | | | read |
| `search_in_gnd_pool` / `validate_gnd_term` | ✅ | | | | | read |
| MCP Read-Tools | ✅ | | | | | read |
| `propose_keyword_replacement` | | ✅ | | | | mutation (Dialog) |
| `propose_dk_change` | | ✅ | | | | mutation (Dialog) |
| `run_pipeline` | | | ✅ | | | confirm/auto |
| `rerun_step` | | | ✅ | | | confirm/auto |
| `fetch_doi_metadata` | | | | ✅ | | safe |
| `fetch_url` / `read_pdf` | | | | ✅ | | safe (sandboxed) |
| `search_catalog` | | | | ✅ | | safe |
| `analyze_image` | | | | ✅ | | safe |
| `export_results` | | | | | ✅ | safe (FS write) |
| `generate_report` | | | | | ✅ | safe |

## Known Limitations

### Streaming-with-Tools Backend-Limit
**Behoben für OpenAI-kompatibel in P-δ.5b**: Text-Tokens streamen live,
Tool-Call-Deltas werden akkumuliert. `_generate_openai_with_tools` verwendet
jetzt immer `stream=True` wenn `stream_callback` gesetzt ist.

**Ollama**: API-Limit — kein Streaming mit Tools möglich. `stream=False`
bleibt. TODO P-δ.5c: re-evaluate.

**Anthropic / Gemini**: Deferred. Weiterhin `stream=False` mit Tools.

### Cancel-Latenz
**Verbessert in P-δ.5b**: `should_stop` wird jetzt an `generate_with_tools`
übergeben. OpenAI: per-Chunk-Check → sub-Sekunde. Ollama/Anthropic/Gemini:
post-blocking-call-Check → 1 LLM-Generation (vorher: 1 volle Iteration).
Mid-token-Abbruch bei Ollama/Anthropic/Gemini nicht möglich (blocking API).

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
