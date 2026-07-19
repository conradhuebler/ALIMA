# ALIMA UI Anforderungskatalog (Multi-Frontend)

**Status**: Anforderungs- und Diskussions-Dokument. KEINE Implementierung.
**Scope**: PyQt6-GUI + Webapp + CLI. Ziel: konsistente Abbildung von
klassischer + agentischer Pipeline + Einzelschritt-Tools + offenen
Workflows in allen drei Frontends.

## 0. Status quo (kondensiert)

### PyQt6 — `src/ui/` (28 K Zeilen, 30 Module)
**MainWindow Tabs** (Reihenfolge wie heute):
1. 🚀 Pipeline (`pipeline_tab.py`, 2690 Z.) — Workflow-Orchestrator. Kann classic ODER agentic. Eigenes Input-Widget, Step-Stream, Config-Dialog.
2. 🌐 DOI (`crossref_tab.py`) — DOI-Lookup als eigenständiges Tool.
3. 📷 Bild (`image_analysis_tab.py`) — Vision-Modell Bildanalyse.
4. 📝 Abstract (`abstract_tab.py`, 1266 Z.) — Manueller LLM-Call. `set_task()` wechselt prompts.json-Task.
5. 🔍 GND-Suche (`find_keywords.py`, 1537 Z.) — SearchTab mit manueller Nachsuche.
6. ✅ Verifikation (`analyse_keywords` / `comparison_tab.py`?) — Vermutlich keywords-step viewer.
7. 📚 UB-Katalog (`ub_catalog_tab.py`) — DK-Suche im Katalog.
8. 📊 Klassifikationen (`dk_analysis_unified_tab.py`) — DK-Zuordnung + Statistik + UB-Suche unified.
9. 📊 Review (`analysis_review_tab.py`, 1247 Z.) — JSON import/export, multi-step result viewer.
10. 🔍 Vergleich (`comparison_tab.py`) — Erschließungsvergleich.

**Docks**:
- AgenticContextWidget (rechts, live SharedContext)
- ChatWidget (rechts, tabified mit AgenticContext)
- GlobalStatusBar (unten, Provider/Cache/Pipeline-Progress)

**Dialoge**: PipelineConfigDialog (2738 Z.), ComprehensiveSettingsDialog (1581 Z.), BatchProcessingDialog (1511 Z.), PromptEditorDialog, FirstStartWizard, LiberoLoginDialog.

### Webapp — `src/webapp/` (FastAPI + WS)
Endpoints (eigener Code-Pfad, kein Tab-Mirror):
- `POST /api/session` → session erstellen
- `POST /api/input/{sid}` → Eingabe setzen
- `POST /api/analyze/{sid}` → Pipeline triggern
- `WS /ws/{sid}` → Live-Streaming
- `GET /api/export/{sid}` → Export
- `GET /api/session/{sid}/recover` → Recovery

Funktional ≈ Pipeline-Tab + Streaming. **Kein** Single-Step-Modus, **kein** Workflow-Browser, **kein** Chat, **keine** Tab-Tools (Crossref, Bild, GND-Suche separat).

### CLI — `src/cli/commands/`
Module: `pipeline_cmd`, `workflow_cmd`, `search_cmd`, `protocol_cmd`, `state_cmd`, `provider_cmd`, `database_cmd`, `setup_cmd`.

Einziges Frontend mit dediziertem Workflow-Subcommand (`alima workflow <name>`). Single-Step via `--only-step`. Kein Chat.

## 1. User-Modi (was die UI abbilden muss)

### M1 — Standardanalyse (häufigster Fall)
„Eine ALIMA-Erschließung machen": Text/PDF/DOI/Bild rein, vollständige Pipeline läuft, Ergebnisse + DK-Klassifikationen + K10+-Export raus.

- **Default-Workflow**: `alima.yaml` (agentic).
- Klassische Pipeline ist Fallback-Option, nicht Default.
- Nutzer soll **nicht wissen müssen**, dass es zwei Modi gibt — aber Power-User soll umschalten können.

### M2 — Einzelschritt-Analyse / Tool-Modus
„Ich will nur die GND-Suche", „nur die DK-Klassifikation", „nur Keywords aus Abstract extrahieren".

- Heute: separate Tabs (find_keywords, abstract, dk_analysis_unified, ub_catalog, crossref).
- Konflikt: Tabs sind feature-zentriert, nicht workflow-step-zentriert. Welcher Tab entspricht welchem Step?

### M3 — Andere Workflows (heute schon: catalog_search, title_list_search, synonym_expansion, batch_metadata)
„Ich will eine Wunschliste auf Duplikate prüfen", „ich will einen Begriff per Synonyme erweitern".

- Heute: Workflow-Dropdown im PipelineConfigDialog. Pipeline-Tab rendert egal welcher Workflow läuft. Workflow-spezifische Outputs (z.B. duplicate analysis) werden nicht spezialisiert dargestellt.

### M4 — Batch (M1 oder M3 für viele Inputs)
- Heute: BatchProcessingDialog (modal). Eigenes UI, separater Workflow-Pfad.

### M5 — Review / Resume
„Ich will ein altes JSON laden, anschauen, weitermachen, vergleichen."

- Heute: AnalysisReviewTab + ComparisonTab.

### M6 — Chat über Ergebnis
Siehe `agentic_chat_plan.md`. Heute: ChatWidget-Dock, statisch.

## 2. Funktionale Anforderungen

### 2.1 Workflow als zentrales Konzept
**REQ-W1** UI muss erkennen: jeder analytische Vorgang ist ein Workflow (auch „nur GND-Suche" → ein-step-Workflow).

**REQ-W2** Workflow-Auswahl muss präsent sein (nicht hinter Config-Dialog versteckt) — aber default-pre-selected, damit naive User nicht stolpern.

**REQ-W3** UI muss workflow-spezifische Outputs darstellen können. Nicht alle Workflows haben „dk_classifications" o.ä. — Pipeline-Tab darf nicht hardcoded auf Standard-Pipeline-Felder bauen.

**REQ-W4** Custom-Workflow-Upload (eigene YAML laden) muss möglich sein, ohne Code-Änderung.

### 2.2 Standardanalyse (M1)
**REQ-M1.1** Default-Pfad: Input-Eingabe → "Analysieren"-Button → Stream → Ergebnis. Maximal 2 Klicks zum Start.

**REQ-M1.2** Live-Stream pro Step (existiert: `pipeline_stream_widget`).

**REQ-M1.3** Ergebnis-Sichten direkt nach Lauf navigierbar: Final-Keywords, Schlagwortketten, DK-Klassifikationen + Provenance, K10+-Export.

**REQ-M1.4** Abbruch jederzeit möglich (existiert).

**REQ-M1.5** Resume nach Crash (existiert via JSON-Persistenz).

### 2.3 Einzelschritt (M2) — kritisch und ungelöst
**REQ-M2.1** Jeder Step eines Workflows muss einzeln ausführbar sein gegen ein vorhandenes oder neu eingegebenes Teil-Eingabe.

**REQ-M2.2** Eingabe für Einzelstep muss klar typisiert sein: extraction → Abstract+Keywords; selection_chunks → Abstract+GND-Pool; classification → Abstract+DK-Material.

**REQ-M2.3** Ergebnis eines Einzelsteps muss in nachfolgende Schritte übernehmbar sein („mit diesem Ergebnis jetzt classification machen").

**REQ-M2.4** Single-Step-Tabs sollten **aus Workflow-Definition generiert** werden (nicht hardcoded). Ergibt automatisch Konsistenz.

### 2.4 Andere Workflows (M3)
**REQ-M3.1** Workflow-Browser/Picker (nicht nur Dropdown im Config-Dialog).

**REQ-M3.2** Pro Workflow: Input-Schema-Beschreibung (was muss rein?). Heute fehlt — User muss YAML lesen.

**REQ-M3.3** Output-Renderer pro Step-Typ (nicht pro Workflow). z.B. "duplicate_analysis"-Output → Tabelle mit Status-Badges. Konfigurierbar in Workflow-YAML?

### 2.5 Chat (M6)
**REQ-CH1** Chat sieht aktiven Workflow-State (siehe `agentic_chat_plan.md`).

**REQ-CH2** Chat-Tab muss ggf. mit dem im UI gerade angezeigten Step-Result interagieren („zum aktuellen Tab" Modus).

### 2.6 Multi-Frontend-Parität
**REQ-P1** Webapp muss M1, M3, M5, M6 abbilden können. M2 (Einzelschritt) und M4 (Batch) wünschenswert, niedrigere Prio.

**REQ-P2** CLI muss alle nicht-interaktiven Modi (M1, M3, M4, M5) anbieten. Chat-CLI ist möglich (REPL), niedrige Prio.

**REQ-P3** Logik darf NICHT pro Frontend dupliziert werden — alles über `PipelineManager`/`WorkflowExecutor`/`AgentLoop`.

## 3. Berührungspunkte mit aktuellem UI

### 3.1 Was passt
- `pipeline_tab.py` Zentralisierung der Pipeline-Orchestrierung.
- `pipeline_stream_widget.py` Live-Streaming pro Step.
- `agentic_context_widget.py` SharedContext-Visualisierung — könnte universell für jeden Workflow genutzt werden, nicht nur ALIMA.
- Workflow-Discovery + Dropdown ist da.
- `analysis_review_tab.py` JSON-Import/Export-Pfad — gute Basis für Workflow-Output-Browser.
- `pipeline_config_dialog.py` granulare Step-Config (auch wenn 2738 Zeilen).

### 3.2 Was bricht bei Workflow-Pluralität
- **`dk_analysis_unified_tab.py`** — hardcoded auf DK-Step der Standard-Pipeline. Bei `title_list_search`-Workflow leer.
- **`comparison_tab.py`** — vergleicht Erschließungen einer bestimmten Form. Bei custom-workflow-Output unklar.
- **`abstract_tab.py`** — generischer LLM-Call mit Task-Wechsel. Verbindet sich aber NICHT zu einem Workflow-Step. Doppelpfad.
- **`ub_catalog_tab.py`** — eigenständig + tut auch was `dk_analysis_unified_tab` tut. Redundanz.
- **`find_keywords.py`** + manuelle-Suche-Subwidget — eigenständig + überlappt mit `search`-Step.

### 3.3 Was komplett neu nötig wäre (oder massive Erweiterung)
- **Workflow-Browser**: Liste aller discovered workflows mit Description, Input-Schema, "Run"-Button.
- **Output-Renderer-System**: pluggable Renderer pro Step-Output-Typ. Workflow-YAML könnte rendererhint geben (`render: dk_table | duplicate_table | keyword_chains | ...`).
- **Single-Step-Generator**: Tab pro Step eines Workflows automatisch erzeugt. Ersetzt manuell gepflegte Tabs.

### 3.4 Webapp-Lücken
- Workflow-Auswahl im Frontend? Heute hardcoded?
- Chat fehlt komplett.
- Single-Step fehlt.
- Worflow-spezifische Output-Render fehlen.

### 3.5 CLI-Lücken
- `alima workflow` existiert. Aber kein CLI-Command für „Tab-Funktionen" (DOI-Lookup, Bild-OCR) als eigenständige Operationen außerhalb Workflows.

## 4. Lösungs-Optionen (NUR Vorschläge, keine Entscheidung)

### Option A — Workflow-zentrierte Reorganisation
Tabs werden ersetzt durch:
- **Workflow-Tab** (statt aktuell 10 Feature-Tabs): Workflow-Picker oben, ausgewählter Workflow rendert seine Steps. Standard-Workflow bei App-Start: `alima` (agentic).
- **Tools-Tab**: nicht-workflow-bezogene Einzeltools (Konfiguration, Crossref-Lookup, Image-OCR). Quasi „Sidebar".
- **Review-Tab**: JSON-Import/Export, Vergleich.
- **Chat-Dock**: bleibt.

Vorteil: einheitliches Modell, jeder Workflow first-class.
Nachteil: massiver Umbau, alte Tabs gehen verloren oder müssen migriert.

### Option B — Workflow-Picker im Pipeline-Tab, Rest bleibt
Pipeline-Tab kriegt prominenten Workflow-Switcher. Single-Step-Tabs bleiben aber werden zu "Workflow-Step-Runners" für genau einen Step des aktuellen Workflows.

Vorteil: kleinerer Umbau, weniger Reibung.
Nachteil: Tab-Wildwuchs bleibt, neue Workflows kriegen weiterhin keine spezialisierten Outputs.

### Option C — Hybrid mit Output-Renderer-Plugin
Tabs bleiben strukturell. Aber Pipeline-Tab + Single-Step-Tabs nutzen ein neues **OutputRendererRegistry**. Workflow-YAML kann pro Step `render: <name>` setzen. Wenn nicht gesetzt → Fallback-Renderer (Raw-JSON).

Vorteil: rückwärtskompatibel, schrittweise einführbar.
Nachteil: Renderer-Plugin-System dazu, mehr Komplexität.

### Empfehlung (Diskussionsgrundlage)
**C**, weil organisches Wachstum nicht über Nacht weg-refaktoriert wird. Renderer-Registry erlaubt iterative Migration: ein Tab nach dem anderen umstellen, neue Workflows direkt damit versorgen.

## 5. Techniken (für später)

### Frontend-Patterns
- **Step-Card-Pattern** (statt vertikale Pipeline-UI): jeder Step als kollabierbare Card mit Input-Schema, Run-Button, Stream-Output, Result-Renderer. Einheitlich für agentic + classic + custom.
- **Renderer-Plugin-Registry** (analog zu `STEP_REGISTRY` für YAML): `@register_renderer("dk_table")` decorator. UI ruft `get_renderer(step_output_type)`.
- **Output-Schema in YAML** ergänzen (z.B. `output_schema: dk_classifications` pro Step), damit Renderer wissen was zu erwarten.

### State-Sync
- **EventBus** statt direkte Tab-zu-Tab-Signale. Reduziert N×N-Kopplung.
- **Single Source of Truth** = aktiver `KeywordAnalysisState` + (für agentic) `SharedContext`. Tabs subscriben Read-Only.

### Frontend-Vereinheitlichung
- **JSON-Schema für Workflow-Input/Output** generieren aus YAML-Definition + register-fns. Webapp + CLI können dann auto-generieren (Forms, Help-Text).
- **OpenAPI-Spec für Webapp** automatisch aus Workflow-Definitionen.

### Single-Step-Auto-Tabs
- Bei App-Start: aktiver Workflow geladen → Tabs automatisch generiert pro Step. Reduziert hardcoded Tabs.

## 6. Kritische Punkte in deiner Planung (Hauptkritik)

### K1 — „Standardaufgabe" + Single-Step-Tabs sind in Konflikt
Wenn `alima` (agentic) Standard ist, sollten Tabs Steps DIESES Workflows entsprechen. Aktuelle Tabs entsprechen aber Steps der **alten klassischen** Pipeline (Verifikation, DK-Suche, GND-Suche). Bei agentic mit anderen Steps (z.B. `selection_chunks` mit Chunking, `dk_collect` deterministic) passen Tabs nicht 1:1.

→ Entscheidung nötig: Tabs an `alima.yaml`-Steps anpassen ODER Tab-Konzept aufgeben (workflow-generated Tabs, Option A).

### K2 — „Andere Use-Cases ermöglichen" ist nicht trivial
Heute hat Pipeline-Tab Annahmen über die Felder (`extracted_keywords`, `dk_classifications`, etc.). Ein `title_list_search`-Workflow hat aber `duplicate_analysis` als Output — kein Renderer existiert. „Workflow-Switch" funktioniert technisch, aber Output ist Raw-JSON.

→ Renderer-System ist Voraussetzung, nicht nice-to-have.

### K3 — Drei Frontends, ein Mensch
GUI 28K Zeilen + Webapp + CLI parallel zu pflegen ist heute schon Belastung. Jede neue Anforderung × 3.

→ Vorschlag: definiere klare **Feature-Tiers**:
- Tier-1 (alle Frontends): Standardanalyse, Workflow-Auswahl, Stream, Export.
- Tier-2 (GUI + Webapp): Chat, Review, Recovery.
- Tier-3 (nur GUI): Vergleich, Batch, Provider-Settings, Single-Step-Tools.

### K4 — Klassische + Agentische beide gut abbilden = doppelte Last
Klassische Pipeline = `pipeline_utils.py`-Pfad mit eigener `KeywordAnalysisState`-Mutation. Agentic = `WorkflowExecutor`+`SharedContext`-Pfad. Beide existieren seit ALIMA-Pipeline-Architektur v2.0.0. Wenn beide UIs first-class gleich gut sein sollen → doppelter Code, doppelte Bugs.

→ Frage: kann klassische Pipeline als spezial-Workflow `alima_classic.yaml` betrachtet werden, mit gleicher UI wie agentic? Wenn ja: KEIN paralleler Code-Pfad, nur ein YAML-Knopf-Druck zwischen agentic/classic. Wenn nein: warum?

### K5 — Webapp ist stark zurück
Wenn Standard-User die Webapp nutzt (Mehrnutzerszenario, kein Setup), aber Webapp die agentic Pipeline + Chat + Single-Step + Workflow-Auswahl nicht hat → Webapp wird de-facto „Lite-Variante". Strategieentscheidung: Webapp soll mithalten oder bewusst eingeschränkt bleiben?

### K6 — „Wildwuchs" diagnostiziert, aber ohne Migrationsplan kollabiert es nicht
10 Tabs, mehrere überlappen funktional (UB-Katalog ↔ Klassifikationen, Verifikation ↔ Vergleich). Wegnehmen ist gefährlich (User-Gewohnheit) — aber „addieren statt ersetzen" ist genau, was zum Wildwuchs geführt hat.

→ Brauche pro Tab Entscheidung: bleiben, mergen, ersetzen, löschen. Mit User-Feedback (welche Tabs werden überhaupt benutzt?). Telemetrie?

### K7 — Chat-Plan reicht in UI rein, ist aber nicht im UI-Plan
Mein eigener `agentic_chat_plan.md` Phase 7 (proposal-dialogs, rerun-step) ist ein **UI-Eingriff**, nicht nur Backend. Müsste hier integriert sein.

## 7. Kritische Punkte in MEINER eigenen Planung (`agentic_chat_plan.md`)

Selbstkritik:

### S1 — Chat-UI-Komplexität unterschätzt
Phase 4 „ChatWidget integration: render tool calls + proposal dialogs" ist mit 1 Tag veranschlagt. Realistisch sind tool-call-collapsibles + modal-confirmation-dialogs + state-sync-back-to-chat eher 2-3 Tage. Plus Webapp-Mirror.

### S2 — Multi-turn path A ist Schulden, nicht MVP
History als String reinrendern wird bei langen Chats teuer + AgentLoop kennt keine echten conversational messages. Path B sollte direkt MVP sein.

### S3 — Chat-Provider-Auswahl ist nicht durchdacht
Pipeline läuft ggf. mit Opus-class. Chat soll dann auch Opus benutzen (teuer)? Oder kleineres Modell (mismatch zur Pipeline-Logik)? Heute ChatWidget hat eigenes Combo, aber Default-Logik ist „Auto" und unklar.

### S4 — Mutationen + Tabs-State
Wenn Chat keyword ersetzt → `KeywordAnalysisState.apply_keyword_replacement()` → welche Tabs müssen sich neu rendern? Heute kein State-Sync-Bus. Wir bauen Phase 2 (Schreib-Tools) BEVOR State-Sync gebaut ist → Risiko inkonsistente Tabs.

### S5 — Read-tools mit live-search verändern den Pool
Phase 1 read-only ist nur teilweise read-only: `search_gnd` schreibt Cache. Pool, der Pipeline benutzt hat, wächst während Chat. Bei Re-Run ggf. andere Treffer als ursprünglich.

### S6 — Tools für Chat skalieren nicht zu „andere workflows"
Tool-Set ist auf `KeywordAnalysisState` + ALIMA-Pipeline-Felder zugeschnitten. Bei `title_list_search`-Result hat Chat NICHTS zum reinschauen, weil keine Tools dafür registriert.

→ Tool-Registry muss auch workflow-aware sein. Pro Workflow ein Tool-Set, oder generische Tools die `SharedContext.extra` introspecten.

## 8. Open Questions

1. **Tab-Inventar reduzieren**: welche Tabs darf man entfernen, welche müssen bleiben? Telemetrie verfügbar?
2. **Klassische Pipeline = `alima_classic.yaml` Workflow**: wegfaktorisieren? Oder bewusst dual halten?
3. **Webapp-Strategie**: Feature-Parität oder Lite-Variante?
4. **Renderer-Registry**: bauen oder weiter Step-spezifische Tabs hardcoden?
5. **Workflow-YAML-Editor in UI**: nice-to-have oder out-of-scope (User editiert YAML in IDE)?
6. **Chat als universelles Frontend?** Statt Tab-Klicks → "ALIMA, mach DK-Klassifikation für …". Long-term Vision.
7. **State-Sync-Bus** vor oder nach Chat-Schreib-Tools?
8. **Custom-Workflow-Upload**: über UI oder nur Filesystem (`~/.config/alima/workflows/`)?
9. **Welche Frontends kriegen Renderer-Registry zuerst** — GUI ist 28K Zeilen, Migration einzelnen Tabs aufwendig.

## 9. Vorgeschlagenes Vorgehen (Diskussion)

1. **Konsens über Konfliktpunkte K1, K4, K5** — bevor irgendwo Code angefasst wird.
2. **Tab-Inventar-Audit**: was wird genutzt, was redundant, was sollte mergen.
3. **Renderer-Registry-Skizze** (Code-Sketch, kein Build) — beweist ob Option C trägt.
4. **Webapp-Strategie festlegen**.
5. **Chat-Plan mit UI-Anforderungen verschmelzen** (S1, S4, S6).
6. **Migrationsplan** (welche Phase ersetzt was) bevor irgendein Tab angefasst wird.

## Related

- [`agentic_workflow.md`](../agentic_workflow.md) — Backend, das die UI bedient.
- [`agentic_chat_plan.md`](agentic_chat_plan.md) — Chat-Plan, hat UI-Berührung in Phase 4 + 7.
- [`legacy/ui_restructuring_2025.md`](legacy/ui_restructuring_2025.md) — vorherige Restrukturierung (Service-Layer-Unification 2025). War backend-fokussiert, hat UI-Wildwuchs nicht reduziert.
