# Chat-UI-Konzept (WP8)

**Status**: Faktenbasis-Dokument für T3-Decision **„Chat-UI-Position +
Tool-Call-Rendering"**. Output von WP8 aus
[`wp_detailed_plans.md`](wp_detailed_plans.md). **Pseudo-Code +
Wireframes only — keine Implementation.**

**Verhältnis zu [`agentic_chat_plan.md`](agentic_chat_plan.md)**:
chat-plan = MVP-Phasen-Roadmap (6.5 PT). WP8 = Architektur-Skizze
für UI-Aspekte: Dock-Positionierung, Tool-Call-Block-Rendering,
Mutations-Modal, Re-Run-Stream-Routing, Webapp-API-Vorbereitung.

**Methode**: Code-Inspektion (ChatWidget, ChatWorker,
PipelineStreamWidget, MainWindow-Dock-Integration, SystemPromptDialog,
Webapp-WS-Endpoint), WP7 (Tools), WP6 (EventBus), WP11 (Provider).

**Querverweise**:
- [`chat_tools_design.md`](chat_tools_design.md) (WP7) — Tool-
  Architektur, Schreib-Tools, ChatAgentWorker.
- [`state_sync_design.md`](../state_sync_design.md) (WP6) — Mutations-API
  + EventBus für `proposal_*`-Events.
- [`provider_portability_design.md`](provider_portability_design.md)
  (WP11) — Chat-Provider-Default + Capability-Check.
- [`frontend_tier_model.md`](frontend_tier_model.md) (WP9) — Chat-
  Tier-Mapping (F-M6.1-5).
- [`agentic_chat_plan.md`](agentic_chat_plan.md) — MVP-Phasen 1-7.
- [`wp_detailed_plans.md`](wp_detailed_plans.md) WP8 — Soll-Definition.

## 1. Executive Summary

- **UI-Position**: Haupt-Dock (RightDockWidgetArea) bleibt
  (heutige `chat_dock` in
  [`main_window.py:452-470`](../src/ui/main_window.py)). Mini-Chat
  pro Tab als Tier-3-Erweiterung **out-of-MVP**.
- **Tool-Call-Rendering**: Custom-QWidget-pro-Turn (kein QTextEdit-
  Only), kollabierbare Tool-Call-Blöcke mit Status-Icons.
- **Mutations-Modal**: QDialog mit 3-Button-Pattern (Ja/Nein/
  Bearbeiten), subscribed auf `AlimaStateBus` `proposal_*`-Events.
- **Re-Run-Stream**: `propose_step_rerun` → Pipeline-Worker → Stream
  zurück in `PipelineStreamWidget`, **nicht** in Chat. Chat zeigt
  nur Status-Zeile.
- **Persistenz**: JSON-Export pro Session, Auto-Reset bei neuem
  Pipeline-Run.
- **Multi-Turn**: `ChatAgentWorker` ersetzt heutigen single-shot
  `ChatWorker` (WP7 Sek 9).
- **Webapp-API**: Endpoints definiert (`POST /api/chat/{sid}`,
  `WS /ws/chat/{sid}`), Implementation = WP10/spätere WP (Tier-2 nach
  WP9).
- **Provider-Combo**: bleibt im Header, Default aus `chat.default_*`,
  Capability-Check warnt bei `tool_use=none`.

## 2. UI-Position

### Heutige Lage
- `ChatWidget` = `QDockWidget` auf `RightDockWidgetArea`
  ([`main_window.py:452-470`](../src/ui/main_window.py)).
- Movable + Floatable + Closable.
- Parallel mit `AgenticContextWidget` im selben Area (stacked).
- Header: Title `💬 ALIMA Chat`, Model-Combo, System-Prompt-Button,
  Reset-Toggle.

### Vorschlag aus WP8-Outline
- Dock bleibt für Haupt-Chat.
- **Plus** kontextueller Mini-Chat in jedem Tab — User stellt Frage
  zum sichtbaren Daten-Slice direkt im Tab.

### Pro / Contra Mini-Chat-pro-Tab
| Pro | Contra |
|---|---|
| Frage-im-Kontext (User muss nicht Tab wechseln) | Mini-Chat-Inflation: 11 Tabs × Mini-Chat = 11 Sub-Widgets |
| Daten-Slice automatisch als implizites Tool-Argument | Inkonsistente UX (welcher Chat ist „der" Chat?) |
| Tier-3-Feature (WP9 F-M6.1-5 sind Tier-2, „Mini" wäre Tier-3) | Maintenance: jede Tab-Klasse braucht Hook |

### Empfehlung
**Haupt-Dock primär, Mini-Chat OOS für MVP**. Begründung:
- Tier-2 ist schon ambitioniert (`agentic_chat_plan.md` 6.5 PT MVP).
- Mini-Chat-UX nicht gelöst (welcher Speicher? gleicher AgentLoop?).
- Operator kann später eigene WP starten.

### Discoverability-Pfade
- Menü `Tools → Chat aufklappen / einklappen`.
- Keybinding `Ctrl+Shift+C`.
- Auto-Slide-in beim ersten Tool-Call (out-of-MVP).

## 3. Tool-Call-Rendering

### Problem
Heute = `QTextEdit` mit `setHtml()`-formatierten Strings. Tool-Calls
inline als Text dargestellt → User sieht raw-JSON-Dumps,
nicht-kollabierbar, kein Status-Tracking.

### Empfehlung
**Custom-QWidget pro Chat-Turn** (kein QTextEdit-Only). History wird
zu `QScrollArea` mit `QVBoxLayout` von Turn-Widgets.

### Wireframe: kollabierter Tool-Call
```
┌────────────────────────────────────────────────────────────────┐
│ 🤖  Ich prüfe gerade die verfügbaren Daten…                    │
│                                                                │
│  ▶ 🔧 list_available_data()                              ✓     │
│  ▶ 🔧 get_dk_classifications()                           ✓     │
│  ▶ 🔧 search_in_gnd_pool(query="Cadmium")                ✓     │
│                                                                │
│ Es gibt 3 DK-Klassifikationen plus 47 GND-Pool-Einträge.       │
│ Soll ich die Klassifikation überprüfen?                        │
└────────────────────────────────────────────────────────────────┘
```

### Wireframe: aufgeklappter Tool-Call
```
│  ▼ 🔧 search_in_gnd_pool(query="Cadmium")                ✓     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ {                                                        │  │
│  │   "matches": [                                           │  │
│  │     {"gnd_id": "4007249-3", "title": "Cadmium", ...},    │  │
│  │     ...                                                  │  │
│  │   ]                                                      │  │
│  │ }                                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
```

### Status-Icons
| Icon | Bedeutung |
|---|---|
| `⏳` | läuft |
| `✓` | erfolgreich |
| `✗` | Fehler (Error-Tooltip zeigt `tool_def.name` + Exception-Msg) |
| `⏸` | abgebrochen (User-Cancel) |

### Implementation
- `class ChatTurnWidget(QWidget)` — pro Turn ein Widget.
- Tool-Call-Block via `QToolButton` (kollabierbar) + nested
  `QTextEdit` (read-only) für JSON-Result.
- Result-JSON optional via WP4-Renderer rendern (wenn Tool-Output
  einen bekannten Slot zurückgibt — z.B. `slot:gnd_pool`).

### Begründung gegen QTextEdit-only
- Kollabieren ist nicht via HTML+`<details>` möglich (QTextEdit
  rendert das nicht).
- Status-Tracking braucht referenzierbare Widgets (Icon-Update
  während Run).
- Mid-Term: Hyperlinks auf GND-IDs etc. (klick → SearchTab öffnen).

## 4. Mutations-Modal

### Subscribe-Pfad
ChatWidget abonniert beim Init:
```python
# Pseudo-Code in ChatWidget.__init__
bus = AlimaStateBus()
for op in ("replace_keyword", "add_keyword", "remove_keyword",
           "step_rerun"):
    bus.subscribe(f"proposal_{op}", self._on_proposal)
```

### Wireframe: Vorschlag „Keyword-Ersetzung"
```
┌──────────────────────────────────────────────────────────────┐
│ Vorschlag der KI                                       [×]   │
├──────────────────────────────────────────────────────────────┤
│ Ersetze "Freilandökologie" durch "Pflanzenökologie"          │
│ (GND-ID: 4174277-3)                                          │
│                                                              │
│ Begründung:                                                  │
│   "Pflanzenökologie" ist im GND als bevorzugte Form          │
│   verzeichnet. "Freilandökologie" ist ein veralteter         │
│   Synonym-Eintrag.                                           │
│                                                              │
│ Validation: ✓ via validate_gnd_term                          │
│                                                              │
│         [ Ja ]    [ Nein ]    [ Bearbeiten… ]                │
└──────────────────────────────────────────────────────────────┘
```

### Wireframe: Bearbeitungs-Modus
```
┌──────────────────────────────────────────────────────────────┐
│ Vorschlag bearbeiten                                   [×]   │
├──────────────────────────────────────────────────────────────┤
│ Alt:    [ Freilandökologie                              ]    │
│ Neu:    [ Pflanzenökologie                              ]    │
│ GND-ID: [ 4174277-3                                     ]    │
│                                                              │
│ [ GND-ID neu prüfen ]                                        │
│                                                              │
│         [ Übernehmen ]                  [ Abbrechen ]        │
└──────────────────────────────────────────────────────────────┘
```

### Flow
1. WP7-Tool emittiert `AlimaStateBus.emit_event("proposal_replace_keyword", payload)`.
2. ChatWidget subscriber öffnet `MutationProposalDialog(payload)`.
3. User:
   - **Ja** → Dialog ruft `state.apply_keyword_replacement(old, new, gnd_id)`
     (WP6 Mutations-API).
   - **Nein** → Dialog schließt; nichts passiert.
   - **Bearbeiten** → Form-Modus, validiert GND-ID neu via
     `validate_gnd_term`, dann `apply_*` mit neuen Werten.
4. Bei `Ja`/`Übernehmen`: `state_changed`-Event löst Multi-Tab-Update
   aus (WP6).

### Multi-Modal-Stacking
Wenn 2+ `propose_*`-Events kommen während ein Modal offen ist: Queue,
nicht stack. Zweiter Modal wartet bis erster geschlossen ist.

### Pattern-Vorbild
- `SystemPromptDialog` (chat_widget.py:34-71) für Layout.
- `QMessageBox.Yes | No | Edit`-3-Button-Pattern aus
  [`first_start_wizard.py:627-629`](../src/ui/first_start_wizard.py).
- `pipeline_config_dialog.py` für 3-Button-Precedent.

## 5. Re-Run-Stream

### Problem
`propose_step_rerun(step_id, modified_inputs?)` startet einen
Pipeline-Step neu. Stream-Tokens müssen **nicht** in Chat-History,
sondern in `PipelineStreamWidget` (wo Pipeline-User sie erwartet).

### Sequence-Diagramm (ASCII)
```
LLM                Chat-Tool            EventBus         ChatWidget       PipelineWorker    PipelineStreamWidget
 │                    │                    │                  │                  │                   │
 │ tool_call          │                    │                  │                  │                   │
 │ propose_step_      │                    │                  │                  │                   │
 │ rerun(...)         │                    │                  │                  │                   │
 ├───────────────────▶│                    │                  │                  │                   │
 │                    │ emit               │                  │                  │                   │
 │                    │ proposal_step_     │                  │                  │                   │
 │                    │ rerun              │                  │                  │                   │
 │                    ├───────────────────▶│                  │                  │                   │
 │                    │                    │ subscribe-cb     │                  │                   │
 │                    │                    ├─────────────────▶│                  │                   │
 │                    │                    │                  │ Modal: Ja/Nein   │                   │
 │                    │                    │                  │ User → Ja        │                   │
 │                    │                    │                  │                  │                   │
 │                    │                    │                  │ start_step_rerun │                   │
 │                    │                    │                  ├─────────────────▶│                   │
 │                    │                    │                  │                  │ stream-callback   │
 │                    │                    │                  │                  ├──────────────────▶│
 │                    │                    │                  │                  │ token … token …   │
 │                    │                    │                  │ status-Zeile     │                   │
 │                    │                    │                  │ "⏳ Re-run läuft" │                   │
 │                    │                    │                  │                  │                   │
 │                    │                    │                  │                  │ done              │
 │                    │                    │                  │ status-Zeile     │                   │
 │                    │                    │                  │ "✓ Re-run fertig"│                   │
```

### Implementation
- ChatWidget zeigt `QLabel` mit Status-Text in Chat-History (nicht
  in Pipeline-Stream-Widget).
- Stream-Callback wird an `PipelineStreamWidget.stream_callback`
  geroutet (heutiger Pfad bleibt unverändert).
- AlimaStateBus emittiert `pipeline_run_started` / `pipeline_run_done`
  → ChatWidget aktualisiert Status-Zeile.

### Race-Condition mit WP6-Lock
- Re-Run hält `bus.write_lock()` während Lauf.
- Chat-Mutation währenddessen: `acquire_write(timeout=5)` → false →
  Modal zeigt Toast „Pipeline läuft, bitte später".
- Konsistent mit WP6 Sek 6.

### Edge Case: User in anderem Tab
- Re-Run-Stream geht in PipelineTab (eventuell unsichtbar wenn User
  in AnalysisReviewTab).
- ChatWidget zeigt Status-Zeile prominent → User weiß was läuft.
- Optional: Toast-Notification beim `pipeline_run_done` (via
  GlobalStatusBar).

## 6. History-Persistenz

### Default-Verhalten
- Heute: Reset-Toggle in Header
  ([`chat_widget.py:152-158`](../src/ui/chat_widget.py)). Default an.
- Reset bei `pipeline_run_started`-Event (siehe WP6).

### JSON-Export-Schema
```json
{
  "version": "1",
  "session_id": "2026-05-13T08:30:00Z-uuid",
  "workflow": "alima_classic.yaml",
  "provider": "ollama",
  "model": "llama3.1:8b",
  "messages": [
    {"role": "user", "content": "Was sind die Top-3 DK-Codes?"},
    {"role": "assistant", "content": "...", "tool_calls": [...]}
  ],
  "tool_calls": [
    {"turn_idx": 1, "name": "get_dk_classifications",
     "args": {}, "result": {...}, "ts": "..."}
  ],
  "proposals": [
    {"proposal_id": "abc-123",
     "op": "replace_keyword",
     "payload": {"old": "...", "new": "...", "gnd_id": "..."},
     "decision": "accepted" | "rejected" | "edited",
     "edited_payload": {...} | null,
     "ts": "..."}
  ]
}
```

### Speicherort
- Datei: `chat_session_<session_id>.json` neben
  `analysis_export_<session_id>.json` (gleicher Pattern).
- Auto-Save: jeder Turn flush'd zu Disk (analog Webapp-Auto-Save).
- Lade-UI: Menü `File → Chat-Session laden…` (QFileDialog).

### Schema-Versionierung
Top-Level `version: "1"`. WP10 macht Migration falls Schema-Drift.

## 7. Multi-Turn-Integration

### `ChatAgentWorker` (Ersetzt heutiger ChatWorker)
Siehe WP7 Sek 9 — Pseudo-Code:

```python
class ChatAgentWorker(QThread):
    token_received     = pyqtSignal(str)
    tool_called        = pyqtSignal(dict)      # {turn_idx, name, args}
    tool_result        = pyqtSignal(dict)      # {turn_idx, name, result, success}
    iteration_done     = pyqtSignal(int)
    finished_response  = pyqtSignal(str)
    error_occurred     = pyqtSignal(str)

    def run(self):
        loop = AgentLoop(
            llm_service=self.session.llm_service,
            tool_registry=self.session.tool_registry,
            max_iterations=20,
            stream_callback=self.token_received.emit,
            tool_call_callback=self.tool_called.emit,
            tool_result_callback=self.tool_result.emit,
        )
        result = loop.run(
            messages=self.session.messages,
            tools_schema=[t.to_schema() for t in self.tools],
        )
        self.finished_response.emit(result.final_response)
```

### Signal-zu-UI-Wiring
| Signal | ChatWidget-Action |
|---|---|
| `token_received(token)` | Append-to-current-assistant-message in `ChatTurnWidget` |
| `tool_called(meta)` | Insert Tool-Call-Block (kollabiert, Status `⏳`) |
| `tool_result(meta)` | Update Tool-Call-Block (Status `✓`/`✗`, Result-Body) |
| `iteration_done(idx)` | Optional: Iteration-Marker in UI |
| `finished_response(text)` | Finalize Turn, persist to JSON |
| `error_occurred(msg)` | Toast + status-Zeile |

### Cancel-Path
- Heutiger Cancel-Button (chat_widget.py:202+) bleibt sichtbar
  während Run.
- Click → `worker.stop()` → setzt `loop.stopped = True` (neue Flag
  in AgentLoop, OOS für WP8-Skizze aber genannt).
- AgentLoop prüft Flag pro Iteration → exits clean.

## 8. Webapp-API-Vorbereitung

**Definieren, nicht bauen** — Tier-2 (WP9). Bauen = WP10 oder spätere
WP.

### Endpoints
| Endpoint | Methode | Body / Params | Output |
|---|---|---|---|
| `POST /api/chat/{session_id}` | POST | `{message: str, workflow: str, no_cache_writes: bool}` | `{turn_id: str, queued: bool}` |
| `WS /ws/chat/{session_id}` | WebSocket | (handshake mit `turn_id`) | Event-Stream (siehe unten) |
| `GET /api/chat/{session_id}/history` | GET | — | full JSON-Export (Sek 6 Schema) |
| `POST /api/chat/{session_id}/cancel` | POST | `{turn_id}` | `{cancelled: bool}` |

### Stream-Event-Format
```json
{"type": "token", "data": "..."}
{"type": "tool_call", "turn_idx": 1, "name": "...", "args": {...}}
{"type": "tool_result", "turn_idx": 1, "result": {...}, "success": true}
{"type": "proposal", "proposal_id": "...", "op": "...", "payload": {...}}
{"type": "iteration_done", "idx": 2}
{"type": "done", "final_response": "...", "total_turns": 3}
{"type": "error", "msg": "..."}
```

### Authn / Authz
- Session-basiert wie heute (`/ws/{session_id}` Pattern,
  [`app.py:679`](../src/webapp/app.py)).
- Read-only-Mode `chat.no_cache_writes` per Body-Flag override.
- Schreib-Tools: Webapp braucht eigenen Confirmation-Flow
  (HTML-Modal statt QDialog). WP9 Tier-2 detail.

### Tier-2-Notiz (für WP9)
F-M6.1-5 sind Tier-2 (GUI + Webapp). WP8-Endpoints-Skizze ist Input
für WP9-Catch-Up-Roadmap Schritt 5 (Webapp-Chat). Implementation
blockiert durch WP10.

## 9. Provider-Combo

### Heutige Lage
Header hat Model-Combo
([`chat_widget.py:124-158`](../src/ui/chat_widget.py)), populated aus
`LlmService.list_models()`. Provider implizit aus Model-Auswahl.

### Vorschlag
- Provider-Combo + Model-Combo (zwei Combos, cascading).
- Default aus `config.chat.default_provider` + `config.chat.default_model`
  (WP7 Sek 8, WP11 Sek 11).
- Operator-Empfehlung: Ollama `llama3.1:8b` lokal als Default.

### Capability-Check
Beim Connect:
```python
# Pseudo-Code
caps = load_model_capabilities(provider, model)  # WP11
if caps.get("tool_use", "none") == "none":
    self._show_toast(
        f"Provider {provider} / {model} unterstützt kein Tool-Calling. "
        "Chat funktioniert nur als reine Konversation.",
        level="warning",
    )
```

Wenn `tool_use=text_fallback`: Warnung mit Hinweis auf möglicherweise
schlechteren Tool-Compliance.

### Wiring zu ChatAgentWorker
Provider-Combo-Change → ChatAgentWorker neu instanziieren (alte
Worker beendet), AgentLoop mit neuer `llm_service`-Config.

## 10. Decision-Point T3

**T3 = Tool-Set-Design (WP7) + Chat-UI-Position (WP8)**.

### WP8-Beitrag
- **UI-Position**: Haupt-Dock primär, Mini-Chat OOS für MVP (Sek 2).
- **Tool-Call-Rendering**: Custom-QWidget statt QTextEdit (Sek 3).
- **Modal-Pattern**: 3-Button-Pattern Ja/Nein/Bearbeiten (Sek 4).
- **Re-Run-Stream-Routing**: in PipelineStreamWidget, nicht in Chat
  (Sek 5).
- **Persistenz**: JSON pro Session, Auto-Reset bei neuem Lauf (Sek 6).
- **Worker**: `ChatAgentWorker` mit 6 Signals (Sek 7).
- **Webapp**: 4 Endpoints + Stream-Event-Format definiert,
  Implementation OOS (Sek 8).

## 11. Risiken + offene Validierungen

| Risiko | Bewertung | Mitigation |
|---|---|---|
| Modaler Dialog während Token-Streaming irritiert | mittel | Modal queued nicht (Stack), zeigt sich erst nach LLM-Turn-`done`. |
| Re-Run aus Chat während User in anderem Tab | niedrig | Status-Zeile in ChatWidget + GlobalStatusBar-Toast bei `done`. |
| Floating-Dock + Mutations-Modal hinter Hauptfenster | mittel | Modal-Parent = MainWindow (nicht Dock), Always-On-Top-Flag. |
| Custom-QWidget-pro-Turn = mehr Memory bei langen Sessions | niedrig | Limit z.B. max 200 Turns sichtbar, ältere kollabiert. |
| Webapp-Chat-Implementation deferred → Drift mit GUI | mittel | Stream-Event-Format **jetzt** definieren (Sek 8), spätere Implementation refactor-frei. |
| AgentLoop `loop.stop()`-Flag fehlt heute | niedrig | Helper-Flag in AgentLoop hinzufügen (kleine WP10-Task). |
| Provider-Capability-Check bei `tool_use=none` zu strikt | niedrig | Warnung, kein Block — User darf trotzdem chatten ohne Tools. |
| 2 parallele `propose_*`-Events: Queue oder erst-kommt-zuerst | niedrig | Queue, FIFO (Sek 4). |

### Out-of-MVP
- Mini-Chat-pro-Tab (Sek 2).
- Hyperlinks auf GND-IDs in Tool-Call-Results.
- Path-B (User-Approval per Tool-Call, WP7 Sek 9).
- Webapp-Chat-UI-Implementation (Sek 8).

## 12. Cross-References / Folge-WPs

| Folge-WP | Konsumiert aus WP8 |
|---|---|
| **WP10** (Migration) | Implementations-Reihenfolge: ChatAgentWorker (chat-plan Phase 3), Tool-Call-Block (Phase 4), Mutations-Modal (Phase 7), Webapp-Endpoints (späterer Sprint). |
| **WP9** (Frontend-Tier-Modell) | Webapp-Endpoint-Skizze (Sek 8) als Tier-2-Roadmap-Input. |
| **WP7** (Tools) | `propose_*`-Events sind Schreib-Tool-Output → Mutations-Modal. |
| **WP6** (State-Sync) | EventBus-Subscribe für `proposal_*`, Lock-Coordination für Re-Run. |
| **WP11** (Provider) | `chat.default_*`-Config + Capability-Check. |
| **`agentic_chat_plan.md`** | MVP-Phasen sind die Implementations-Roadmap. |

## 13. Operator-Fragen (max 6)

### F1: Mini-Chat-pro-Tab als spätere WP oder ganz verworfen?
**Empfehlung**: später eigene WP, falls Bedarf entsteht.

### F2: Tool-Call-Block via WP4-Renderer rendern (wenn Result einen Slot hat) oder immer raw-JSON?
**Empfehlung**: WP4-Renderer wenn Slot bekannt. Fallback raw_json.

### F3: Schreib-Tools per Default an oder hinter Feature-Flag `chat.allow_writes`?
**Empfehlung**: Feature-Flag, default off (analog WP7 Empfehlung F3).

### F4: Re-Run-aus-Chat öffnet `SingleStepDialog` (WP5) oder läuft direkt im Background?
**Empfehlung**: SingleStepDialog → User sieht Pre-Inputs, kann editieren
oder bestätigen. Konsistenz mit WP5 Sek 4.

### F5: Chat-Session-JSON automatisch oder nur auf User-Klick speichern?
**Empfehlung**: automatisch (Auto-Save analog Webapp). Lokal in
`~/.alima/chat_sessions/`.

### F6: Webapp-Chat-Endpoints jetzt definieren reicht, oder Implementation in WP10 mit reinpacken?
**Empfehlung**: jetzt definieren reicht. Implementation = eigene WP
nach WP10 (Operator-Priorität abhängig von WP9 F1 Webapp-Nutzeranzahl).

## 14. Status

✅ WP8 abgeschlossen.
- UI-Position-Begründung (Sek 2)
- Tool-Call-Rendering + Wireframes (Sek 3)
- Mutations-Modal + Wireframes (Sek 4)
- Re-Run-Stream + Sequence-Diagramm (Sek 5)
- History-Persistenz + JSON-Schema (Sek 6)
- Multi-Turn-Integration (Sek 7)
- Webapp-API-Skizze (Sek 8)
- Provider-Combo (Sek 9)
- T3-Decision (Sek 10)
- Risiken (Sek 11)
- Folge-WP-Mapping (Sek 12)

**Damit alle 8 T2/T3-Architektur-WPs abgeschlossen**: WP4, WP5, WP6,
WP7, WP8, WP11 (+ T1-Audit-WPs WP1, WP2, WP3, WP9 zuvor).
**Nur WP10 (Migration) verbleibt** — synthetisiert alle Architektur-
WPs zu konkretem Implementierungsplan.

Pendend: Operator-Antworten F1-F6.
