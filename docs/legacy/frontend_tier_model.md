# Multi-Frontend-Tier-Modell (WP9)

**Status**: Faktenbasis-Dokument für T1-Entscheidung **„Tier-Festlegung
Frontends + Webapp-Strategie"**. Output von WP9 aus
[`wp_detailed_plans.md`](wp_detailed_plans.md).

**Methode**: Code-Inspektion (Webapp + CLI + GUI) + Sub-Feature-Inventar
aus [`ui_requirements_catalog.md`](ui_requirements_catalog.md). Keine
Telemetrie.

**Querverweise**:
- [`audit_tab_inventory.md`](audit_tab_inventory.md) — GUI-Tabs,
  Webapp-Endpoints, CLI-Mapping aus WP1.
- [`research_classic_vs_agentic.md`](research_classic_vs_agentic.md)
  — WP2.
- [`workflow_output_schemas.md`](../workflow_output_schemas.md) — WP3.
- [`ui_requirements_catalog.md`](ui_requirements_catalog.md) — M1-M6
  Soll-Liste.
- [`agentic_chat_plan.md`](agentic_chat_plan.md) — M6-Phasen.
- [`webapp_session_history.md`](../webapp_session_history.md) — Webapp-
  Roadmap (geplant, nicht implementiert).

## 1. Executive Summary

- 6 Anforderungs-Module (M1-M6) zerlegt in **33 Sub-Features**.
- **4-Tier-Modell**: Tier-0 (Hauptpfad MUST überall), Tier-1 (SHOULD
  überall), Tier-2 (GUI+Webapp), Tier-3 (nur GUI).
- **Webapp ist Tier-3-Niveau für klassische Pipeline**: 16 REST/WS-
  Routes, **kein Workflow-Picker**, kein Single-Step, kein Chat,
  Pydantic-frei.
- **CLI hat 19 Top-Level-Subcommands**, davon ~10 Tier-0/1-fähig.
- **GUI ist Tier-1-Niveau für M1-M5**, M6 partial (statisch).
- **Empfehlung Webapp-Strategie**: **(B) Catch-Up** mit Tier-0-Priorität.
- **OpenAPI**: keine Pydantic-Migration empfohlen. Status-quo Forms+
  Dicts akzeptiert.

## 2. Feature-Inventar (Sub-Features mit ID)

Pro M-Modul + globale G-Features. IDs analog WP-Konvention.

### M1 Standardanalyse (Default `alima.yaml`)
- **F-M1.1** Workflow-Wahl (Picker, nicht Config-Dialog)
- **F-M1.2** Live-Stream pro Step (Token-Display)
- **F-M1.3** Multi-Input (Text/PDF/DOI/Bild)
- **F-M1.4** DK-Klassifikation + Provenance-Anzeige
- **F-M1.5** K10+-Export
- **F-M1.6** Provider-Override (global + per-Step)
- **F-M1.7** Cancel/Abort Step

### M2 Single-Step
- **F-M2.1** Step-spezifisches Input-Form
- **F-M2.2** Step-Output mit Renderer
- **F-M2.3** Step-Output → nächster Step-Input übernehmen

### M3 Andere Workflows
- **F-M3.1** Workflow-Liste / Picker erweitert
- **F-M3.2** Workflow-Input-Schema-Beschreibung
- **F-M3.3** Workflow-spezifische Output-Renderer (z.B.
  `slot:duplicate_table`)
- **F-M3.4** Custom-YAML laden

### M4 Batch
- **F-M4.1** File-Batch
- **F-M4.2** URL-Batch
- **F-M4.3** Directory-Scan
- **F-M4.4** K10Plus-Siegel-Resolver
- **F-M4.5** Progress-Bar / Continue-on-Error

### M5 Review / Resume
- **F-M5.1** JSON-Import
- **F-M5.2** JSON-Export
- **F-M5.3** Compare zwei States
- **F-M5.4** Resume / Re-Run einzelner Steps

### M6 Chat
- **F-M6.1** Chat-Widget basis (history, send)
- **F-M6.2** Read-only Tools
- **F-M6.3** Write-Tools (Mutationen mit Confirmation)
- **F-M6.4** Multi-Turn Path B
- **F-M6.5** Re-Run-from-Chat

### Globale Features
- **G-1** Settings/Provider-Management
- **G-2** prompts.json-Editor
- **G-3** DB-Mgmt (DK-Cache, GND-Pool)
- **G-4** Provider-Test/Capability-Check

**Total**: 26 M-Features + 4 G-Features = **30 Sub-Features** (33 mit
M5.1+M5.2 separat gezählt).

## 3. Frontend-Matrix (Master)

Zellen: `✓` = vorhanden, `✗` = fehlt, `partial` = teilweise, `geplant`
= in Roadmap (z.B. `webapp_session_history.md`, `agentic_chat_plan.md`).

| ID | Sub-Feature | GUI | Webapp | CLI | Soll-Tier | Notiz |
|---|---|---|---|---|---|---|
| F-M1.1 | Workflow-Wahl (Picker) | ✓ | ✗ | ✓ | **0** | Webapp: HTML hardcoded 5-Step, kein Picker in `index.html` (192 Z., 0 Treffer für `workflow`) |
| F-M1.2 | Live-Stream pro Step | ✓ | ✓ | partial | **0** | CLI: nur Step-Print, kein Token-Stream |
| F-M1.3 | Multi-Input | ✓ | ✓ | ✓ | **0** | GUI: UnifiedInputWidget; Webapp: 3-Tab DOI/File/Camera; CLI: `--text`/`--pdf`/`--doi` |
| F-M1.4 | DK-Anzeige + Provenance | ✓ | partial | partial | **1** | Webapp/CLI: DK-Codes vorhanden, Provenance-Block fehlt |
| F-M1.5 | K10+-Export | partial | ✗ | partial | **1** | GUI: K10Plus-Loader in BatchDialog; kein „Pipeline-Result → K10+-XML" |
| F-M1.6 | Provider-Override | ✓ | ✓ | ✓ | **1** | Webapp: `global_override`-Form-Field, kein per-Step |
| F-M1.7 | Cancel/Abort | ✓ | ✓ | partial | **1** | Webapp: `/api/session/{id}/cancel` + `abort_step` |
| F-M2.1 | Step-Input-Form | ✓ | ✗ | ✓ | **1** | CLI: `--only-step <id>`; Webapp: kein Step-Endpoint |
| F-M2.2 | Step-Output mit Renderer | ✓ | ✗ | partial | **1** | Renderer-Registry = WP4 |
| F-M2.3 | Step-Output-Übernahme | ✓ | ✗ | ✓ | **1** | CLI: JSON-Output → erneut `--only-step` |
| F-M3.1 | Workflow-Liste / Picker | ✓ | ✗ | ✓ | **1** | CLI: `workflows list`; Webapp: fehlt |
| F-M3.2 | Workflow-Input-Schema | partial | ✗ | partial | **2** | Heute YAML lesen — siehe WP3 Sektion 4 |
| F-M3.3 | Workflow-spezifische Renderer | partial | ✗ | ✗ | **2** | Pipeline-Tab hardcoded auf Standard-Felder (WP1 K2); WP4-Block |
| F-M3.4 | Custom-YAML laden | ✓ | ✗ | ✓ | **2** | CLI: `--custom-workflow`; GUI: Workflow-Browser |
| F-M4.1 | File-Batch | ✓ | ✗ | ✓ | **1** | GUI: BatchProcessingDialog; CLI: `batch` |
| F-M4.2 | URL-Batch | ✓ | ✗ | ✓ | **1** | GUI: BatchProcessingDialog URL-Tab |
| F-M4.3 | Directory-Scan | ✓ | ✗ | partial | **3** | GUI: Pattern-Matching; CLI: scriptbar |
| F-M4.4 | K10Plus-Siegel | ✓ | ✗ | ✓ | **3** | GUI: SRU-API + Cache; CLI: scriptbar |
| F-M4.5 | Progress-Bar + Continue-on-Error | ✓ | ✗ | partial | **2** | Webapp-Batch fehlt komplett (Async-Job-API) |
| F-M5.1 | JSON-Import | ✓ | partial | ✓ | **0** | Webapp: nur Auto-Save-Recovery (`/api/session/{id}/recover`), kein Upload |
| F-M5.2 | JSON-Export | ✓ | ✓ | ✓ | **0** | Webapp: `/api/export/{id}`; CLI: `--output-json` |
| F-M5.3 | Compare zwei States | ✓ | ✗ | partial | **2** | GUI: ComparisonTab; CLI: `show-protocol` (kein Diff) |
| F-M5.4 | Resume / Re-Run | ✓ | partial | ✓ | **1** | Webapp: Auto-Save-Recovery, nicht selektives Re-Run |
| F-M6.1 | Chat-Widget basis | partial | ✗ | ✗ | **2** | GUI: ChatWidget statisch, kein Tool-Use |
| F-M6.2 | Read-only Tools | geplant | ✗ | ✗ | **2** | `agentic_chat_plan.md` Phase 1-2 |
| F-M6.3 | Write-Tools (Confirmation) | geplant | ✗ | ✗ | **2** | `agentic_chat_plan.md` Phase 7 |
| F-M6.4 | Multi-Turn Path B | geplant | ✗ | ✗ | **2** | `agentic_chat_plan.md` S2; AgentLoop multi-turn fertig |
| F-M6.5 | Re-Run-from-Chat | geplant | ✗ | ✗ | **2** | Block durch WP8 |
| G-1 | Settings/Provider-Mgmt | ✓ | ✗ | partial | **3** | GUI: ComprehensiveSettingsDialog; CLI: `provider`-Sub |
| G-2 | prompts.json-Editor | ✓ | ✗ | ✗ | **3** | GUI: PromptEditorDialog |
| G-3 | DB-Mgmt | ✓ | ✗ | ✓ | **3** | GUI: SettingsDialog DB-Tab; CLI: `db-config`-Sub |
| G-4 | Provider-Test/Cap-Check | ✓ | partial | ✓ | **2** | Webapp: nur `/api/models`-Liste; CLI: `test-providers` |

**Verifizierte Datenpunkte**:
- 16 Webapp-Routes ([`app.py`](../src/webapp/app.py): Z. 293, 313, 334,
  343, 369, 385, 408, 424, 452, 498, 644, 679, 776, 834, 1323, 1334)
- 19 CLI-Top-Level-Subcommands (aus `src/cli/main.py:57+`)
- `index.html` (192 Z.): keine Treffer für `workflow`,
  `enable_agentic`, `only_step`, `chat` (grep verifiziert)

## 4. 4-Tier-Definition

| Tier | Geltungsbereich | Kriterium |
|---|---|---|
| **Tier-0** | GUI + Webapp + CLI (MUST) | Hauptpfad. Ohne dieses Feature ist Frontend für Primärzweck unbenutzbar. Reproduzierbar + automatisierbar. |
| **Tier-1** | GUI + Webapp + CLI (SHOULD) | Soll überall sein. CLI ggf. weniger UX-optimal aber funktional. |
| **Tier-2** | GUI + Webapp | Interaktive Features mit Stream/Review/Modals. CLI nicht sinnvoll. |
| **Tier-3** | nur GUI | Exploratives, Debugging, Bulk-Settings. Webapp unverhältnismäßig. |

Beispiel-Mapping (siehe Spalte „Soll-Tier" in Sektion 3):

| Tier | Sub-Features |
|---|---|
| **Tier-0** | F-M1.1, F-M1.2, F-M1.3, F-M5.1, F-M5.2 |
| **Tier-1** | F-M1.4, F-M1.5, F-M1.6, F-M1.7, F-M2.1-3, F-M3.1, F-M4.1-2, F-M5.4 |
| **Tier-2** | F-M3.2, F-M3.3, F-M3.4, F-M4.5, F-M5.3, F-M6.1-5, G-4 |
| **Tier-3** | F-M4.3, F-M4.4, G-1, G-2, G-3 |

**Operator kann Tier-0/1-Grenze pro Feature revidieren** (siehe F-by-F
in Sektion 3). Default-Mapping orientiert sich an
„unbenutzbar-ohne"-Kriterium.

## 5. Webapp-Strategie (Empfehlung: B Catch-Up)

| Option | Beschreibung | Aufwand | Risiko |
|---|---|---|---|
| (A) Lite | Webapp bleibt M1-klassisch, M2/M3/M5-Compare/M6 nie | klein | Webapp veraltet, paralleler Fork |
| **(B) Catch-Up (Empfehlung)** | Priorisierte Roadmap, Tier-0 zuerst | mittel | langer Schwanz, Resource-Planung nötig |
| (C) Headless-Equivalent | Webapp = volle GUI-Äquivalenz | groß | doppelter UI-Pflegepfad |

### Catch-Up-Roadmap (Reihenfolge nach Tier)

1. **Tier-0 zuerst** — F-M1.1 Workflow-Picker:
   - `POST /api/analyze/{id}` erweitern um `workflow_name`,
     `enable_agentic_mode`, `only_step`
   - `index.html`: `<select id="workflow-picker">` befüllt aus
     `GET /api/workflows` (neu)
2. **Tier-1 Single-Step (F-M2.1-3)**:
   - Neuer Endpoint `POST /api/step/{session_id}` mit `step_id` +
     `step_input`
   - Frontend: Step-Detail-Panel pro Workflow-Step
3. **Tier-1 Batch (F-M4.1-2)**:
   - Async-Job-API: `POST /api/batch/{id}` startet, `GET /api/batch/
     {id}/status` pollt
   - Frontend: Batch-Tab mit Job-Liste
4. **Tier-2 Compare (F-M5.3)**:
   - Zweiter Session-Slot oder dedizierter Diff-Endpoint
     `POST /api/compare` (zwei JSON-Bodies)
5. **Tier-2 Chat (F-M6.1-5)** — **blockiert durch WP8** (AgentLoop
   multi-turn Path B + Tool-Use + Mutations-Modal):
   - REST: `POST /api/chat/{session_id}` + WS `/ws/chat/{id}`
   - Frontend: Chat-Panel parallel zu Pipeline-Stream

**Begründung gegen (A)**: Operator-Vorgabe `alima_classic.yaml` als
Forschungspfad braucht Webapp-Workflow-Picker (Tier-0). Webapp-Drift
verstärkt parallele Forks.

**Begründung gegen (C)**: WP9 ist Audit-Output, keine 4-Wochen-
Sprint-Planung. Headless erfordert HTML-Tab-Generator (~WP4 + WP10).

## 6. CLI-Strategie

19 Top-Level-Subcommands (aus `src/cli/main.py:57+`). Tier-Mapping:

| Tier | CLI-Commands |
|---|---|
| **Tier-0** | `pipeline`, `workflow`, `show-protocol`, `load-state` |
| **Tier-1** | `batch`, `search`, `list-models`, `list-providers`, `list-models-detailed`, `test-providers`, `test-catalog` |
| **Tier-2** | `provider` (sub: add/edit/remove/list/test), `db-config` (sub: show/test/set-sqlite/set-mysql) |
| **Tier-3 (CLI-only)** | `save-state` (deprecated), `clear-cache`, `dnb-import`, `migrate-db`, `ollama`, `setup` |

**Lücke**: `alima chat <workflow>` — realistisch nur als **REPL über
fertige Pipeline-State-JSON**, nicht über Live-Pipeline-Stream.
Blockiert durch WP8.

**CLI-only-Stärken** (nicht Frontend-Lücke, sondern bewusst CLI):
- Batch-Scripting (Bash-Loops über `alima pipeline`)
- K10Plus-Siegel-Bulk (`batch --siegel`)
- DB-Setup-Wizard (`setup`, `migrate-db`)
- Provider-Test in CI (`test-providers`)

## 7. Geteilte Schicht / OpenAPI (Status-quo bleibt)

| Option | Aufwand | Mehrwert | Risiko |
|---|---|---|---|
| (a) Pydantic-Wrap | ~2-3 PT | Auto-OpenAPI, IDE-Completion | Drift dataclass ↔ Pydantic |
| (b) Manuelle OpenAPI-YAML | klein | Statisch dokumentiert | Manuelle Pflege |
| **(c) Status-quo Forms+Dicts (Empfehlung)** | 0 | — | OpenAPI rudimentär, keine Schema-Validation |

**WP9-Empfehlung: (c) bleibt**.

Begründungen:
- Pydantic-Wrap zieht zweite Quelle der Wahrheit ein
  (`KeywordAnalysisState`, `SharedContext` sind heute `@dataclass`)
- 16 Webapp-Routes sind überschaubar — manuelle Doc reicht
- Operator-Entscheidung (diese Session): kein WP10-Aufwand für
  Schema-Layer

Aktuelle Schema-Konventionen dokumentiert:
- **Form-Bodies**: `POST /api/analyze/{id}` (input_type, content, file,
  global_override, source_type, source_value) — für File-Upload
- **JSON-Bodies**: andere POSTs (z.B. `/api/session/{id}/cancel`)
- **Responses**: dict-only via
  [`result_serialization.py`](../src/webapp/result_serialization.py)
  (keine BaseModel)
- **WS-Events**: `status`, `heartbeat`, `complete`, `error`,
  Token-Buffer (per `step_id`)

Späteres WP kann (a) revisionieren wenn Webapp >40 Routes hat.

## 8. Decision-Point T1

T1 = „Tier-Festlegung Frontends + Webapp-Strategie". WP9 liefert
entscheidungsreif:

- **4-Tier-Modell** namentlich fixiert (Sektion 4)
- **30 Sub-Features einsortiert** (Sektion 3 + 4)
- **Webapp-Strategie B Catch-Up** mit 5-Punkt-Roadmap (Sektion 5)
- **OpenAPI-Pfad fixiert** (Status-quo, Sektion 7)
- **CLI-Lückenliste**: `alima chat` als WP8-blockiert dokumentiert
- **Webapp-Lückenliste**: Roadmap als WP10-Migrations-Input

## 9. Operator-Fragen (max 6)

### F1: Webapp-Nutzerprofile (intern/extern, geschätzte Anzahl)?
**Empfehlung**: Operator-Aussage. Falls < 5 interne Nutzer →
Catch-Up-Schritt 4/5 (Compare, Chat) niedrig priorisieren.
SessionHistory-Bedarf (`webapp_session_history.md`) hängt davon ab.

### F2: Chat-in-Webapp (M6) Pflicht oder GUI-only akzeptabel?
**Empfehlung**: GUI-only erst, Webapp-Chat als Bonus nach WP8.
Catch-Up Schritt 5 wird sonst Block.

### F3: K10+-Export in Webapp wichtiger oder reicht GUI/CLI?
**Empfehlung**: F-M1.5 ist Tier-1, also Webapp-Endpoint sinnvoll. Falls
Operator-Library-Workflow nur GUI nutzt → Tier-3 absenken.

### F4: Webapp-Batch (Async-Job) nötig oder reicht CLI?
**Empfehlung**: CLI reicht für Power-User. Webapp-Batch Schritt 3
optional, je nach F1.

### F5: Provider-Override pro Run in Webapp oder eingefrorenes Default?
**Empfehlung**: pro Run (existiert bereits als `global_override`-Form).
Per-Step-Override bleibt GUI-only (zu komplexes UI für Webapp).

### F6: Settings-Dialog-Pendant in Webapp nötig oder Config-File-Only?
**Empfehlung**: Config-File-Only akzeptabel. G-1 bleibt Tier-3. Webapp-
Admin-Panel wäre WP-eigenes-Vorhaben.

## 10. Querverweise / Folge-WPs

| Folge-WP | Konsumiert aus WP9 |
|---|---|
| **WP4** (Renderer-Registry) | Sektion 3+4 — Tier-0/1-Features brauchen Renderer pro Frontend (Qt-Widget / HTML / CLI-text) |
| **WP8** (Chat-UI) | Sektion 5 Schritt 5 + Sektion 6 — Webapp/CLI-Chat blockiert durch WP8 |
| **WP10** (Migration) | Sektion 5 Catch-Up-Roadmap = Migrationsticket-Reihenfolge |
| **WP11** (Provider-Portabilität) | Sektion 4 Tier-1 F-M1.6 (Provider-Override) braucht Capability-Schema |

## 11. Risiken + offene Validierungen

- **Keine Webapp-Telemetrie** → Nutzer-Anzahl/Profile unbekannt →
  Catch-Up-Priorität operator-abhängig (F1).
- **Sub-Feature-Liste aus `ui_requirements_catalog.md` ist Soll, nicht
  Ist**: Stichprobenartige GUI-Verifikation in Sektion 3 ergänzt.
  Diskrepanzen sind als `partial` markiert.
- **OpenAPI Status-quo akzeptiert Schema-Drift-Risiko**: Webapp-API-
  Änderungen erfordern manuelles Doc-Update.
- **Tier-0 vs Tier-1 Grenze unscharf**: für F-M1.4 (DK-Anzeige), F-M1.5
  (K10+-Export), F-M1.6 (Provider-Override) Operator-Entscheidung
  nötig. Default-Mapping ist konservativ Tier-1.
- **Catch-Up ohne Sprint-Planung**: qualitative Aufwand-Hinweise
  (klein/mittel/groß), keine PT-Quantifizierung.
- **Chat (M6) blockiert durch WP8**: Catch-Up Schritt 5 nicht startbar
  bevor WP8 abgeschlossen.

## 12. Status

✅ WP9 abgeschlossen. Pendend: Operator-Antworten F1–F6, dann
T1-Tier-Festlegung + Webapp-Strategie final.
