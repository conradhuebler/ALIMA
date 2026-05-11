# UI-Inventar-Audit (WP1)

**Status**: Faktenbasis-Dokument. Input für WP4 (Renderer-Registry),
WP5 (Single-Step-Modell), WP10 (Migrationsplan).

**Methode**: Code-Inspektion (Phase-1-Audit, drei Explore-Agents +
Verifikations-Reads). Telemetrie nicht vorhanden — User-Aussagen
müssen Sektion 7 schließen.

**Querverweise**:
- [`audit_findings.md`](audit_findings.md) — 17 punktuelle Vorbefunde.
- [`consolidation_inventory.md`](consolidation_inventory.md) —
  wiederverwendbare Bausteine (Sektion B = UI).
- [`wp_detailed_plans.md`](wp_detailed_plans.md) — WP-Pläne, die
  diese Empfehlungen konsumieren.

## 1. GUI-Tab-Tabelle

9 Top-Level-Tabs, 8977 LOC total. Reihenfolge wie in
`main_window.py:629-642`.

| # | Tab | Klasse / Datei | LOC | Zweck (kurz) | State READS | State WRITES | Sub-Tabs | Eigene Worker | Hat eigenen LLM-Pfad |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 🚀 Pipeline | `PipelineTab` / `pipeline_tab.py` | 2350 | Pipeline-Orchestrator (classic + agentic) | working_title, original_abstract, initial_keywords, search_results, final_llm_analysis, dk_* | — | 6 | `PipelineWorker` (workers.py) | Nein (delegiert PipelineManager) |
| 2 | 🌐 DOI | `CrossrefTab` / `crossref_tab.py` | 295 | Standalone DOI-Lookup | — | — | 4 | `CrossrefWorker` (eigene Klasse) | Nein (HTTP only) |
| 3 | 📷 Bild | `ImageAnalysisTab` / `image_analysis_tab.py` | 678 | Vision-LLM Bildanalyse | — | — | 0 | `ImageAnalysisWorker` (eigene Klasse) | **Ja** (eigener LLM-Call) |
| 4 | 📝 Abstract | `AbstractTab` / `abstract_tab.py` | 1267 | Generischer LLM-Runner mit prompts.json-Task-Selector | original_abstract, initial_keywords, final_llm_analysis, initial_llm_call_details | — | 2 | `SingleStepWorker` (workers.py) | **Ja** (Single-Step) |
| 5 | 🔍 GND-Suche | `SearchTab` / `find_keywords.py` | 1538 | Manuelle GND/SWB/Lobid-Suche + Result-Curation | initial_keywords, search_results, final_llm_analysis.extracted_gnd_keywords | manual_additions, modified_selections, has_unsaved_changes | 0 | direct `MetaSuggester` (kein QThread) | Nein |
| 6 | ✅ Verifikation | `AbstractTab` / `abstract_tab.py` | (s. Tab 4) | **Identische Klasse wie Tab 4**, default `task="keywords"` | s. Tab 4 | s. Tab 4 | 2 | s. Tab 4 | s. Tab 4 |
| 7 | 📚 UB-Katalog | `UBCatalogTab` / `ub_catalog_tab.py` | 471 | DK-Catalog-Suche | final_llm_analysis.extracted_gnd_keywords | — | 0 | `DkSearchWorker` (eigene Klasse) | Nein (delegiert PipelineStepExecutor) |
| 8 | 📊 Klassifikationen | `DkAnalysisUnifiedTab` / `dk_analysis_unified_tab.py` | 140 | DK-Klassifikations-LLM-Sicht (erbt AbstractTab) | original_abstract, dk_search_results, dk_search_results_flattened, dk_llm_analysis | — | 2 (geerbt) | `SingleStepWorker` (geerbt) | **Ja** (geerbt von AbstractTab) |
| 9 | 📊 Review | `AnalysisReviewTab` / `analysis_review_tab.py` | 1100 | Multi-Sub-Tab-Result-Viewer + JSON-Import/Export | timestamp, final_llm_analysis, initial_keywords, working_title, classifications | — | **11** | — (file I/O) | Nein |
| 10 | 🔍 Vergleich | `ComparisonTab` / `comparison_tab.py` | 658 | Side-by-side Diff zweier KeywordAnalysisStates | alle Felder (read-only) | — | 6 | — (statisch) | Nein |

### Wichtige Beobachtungen aus Sektion 1

- **Tab 4 + Tab 6 sind identische Klasse** mit unterschiedlichem
  Default-`task`. AbstractTab ist dadurch faktisch ein generischer
  Single-Step-Tool-Tab. Pattern wiederverwendbar (DkAnalysisUnifiedTab
  zeigt es bereits — Tab 8 erbt AbstractTab).
- **3 Tabs haben eigenen LLM-Pfad** (3, 4, 8). Andere delegieren oder
  haben keinen LLM.
- **Nur SearchTab (Tab 5) mutiert State** — alle anderen sind
  read-only-Konsumenten.
- **5 Worker-Klassen verteilt über UI**:
  `workers.py` hat 4 (StoppableWorker base, SingleStepWorker,
  PipelineWorker, DNBSyncWorker), 4 weitere lokal in Tabs
  (`CrossrefWorker`, `ImageAnalysisWorker`, `DkSearchWorker`, plus
  3 in `batch_processing_dialog.py`: `SiegelFetchWorker`,
  `SiegelCacheLoadWorker`, `BatchProcessingWorker`). Audit-Finding 17.

## 2. Sub-Tab-Karte

5 Tabs haben innere `QTabWidget`-Strukturen.

### 🚀 PipelineTab (6 Sub-Tabs)
Source: `pipeline_tab.py:527-596`. Step-für-Step-Visualisierung:

1. 📥 Input & Datenquellen
2. 🔤 Schlagwort-Extraktion
3. 🔍 GND-Recherche
4. ✅ Schlagwort-Verifikation
5. 📊 Katalog-Recherche
6. 📚 DK/RVK-Klassifikation

### 🌐 CrossrefTab (4 Sub-Tabs)
Source: `crossref_tab.py:123-144`. DOI-Lookup-Resultate:

1. Hauptergebnisse
2. Über das Buch
3. Inhaltsverzeichnis
4. Schlüsselwörter

### 📝 AbstractTab + ✅ Verifikation (2 Sub-Tabs)
Source: `abstract_tab.py:308-378`. LLM-Konfiguration:

1. Prompt
2. Parameter

### 📊 Klassifikationen / DkAnalysisUnifiedTab (2 Sub-Tabs)
Erbt AbstractTab → identische 2 Sub-Tabs (Prompt, Parameter).

### 📊 AnalysisReviewTab (11 Sub-Tabs) ⚠️ Komplexität
Source: `analysis_review_tab.py:243-392`. Detail-Sicht aller Pipeline-
Aspekte:

1. Original Abstract
2. Initial Keywords
3. Such-Ergebnisse
4. GND-Keywords
5. Finale Analyse
6. Chunks
7. Iterationsverlauf
8. DK/RVK
9. K10+ Export
10. Statistiken
11. Klassifikations-Statistik

**→ Hauptkandidat für Renderer-Registry (WP4)**: 11 Sub-Tabs sind
faktisch 11 Renderer-Slots, die heute hardcoded angelegt werden.

### 🔍 ComparisonTab (6 Sub-Tabs)
Source: `comparison_tab.py:158-163`. Vergleichssicht:

1. Keywords
2. Initialisierung
3. Suche
4. Input
5. Klassifikation
6. Meta

**Hinweis**: Labels überlappen NICHT mit AnalysisReviewTab (anders
als initial vermutet). Inhalte komplett verschieden — ComparisonTab
ist Diff-Tool, ReviewTab ist Single-State-Browser.

## 3. Webapp-Endpoint-Mapping

Source: `src/webapp/app.py`. 15 FastAPI-Routes.

| Endpoint | Line | Funktion | GUI-Tab-Äquivalent | Agentic? |
|---|---|---|---|---|
| `GET /` | 293 | Redirect zu /webapp | — | — |
| `GET /webapp` | 313 | HTML-Frontend ausliefern | alle | — |
| `POST /api/session` | 334 | Session erstellen | (intern) | — |
| `GET /api/session/{id}` | 343 | Session-Status pollen | (intern) | — |
| `POST /api/session/{id}/clear` | 369 | Session zurücksetzen | (intern) | — |
| `POST /api/session/{id}/cancel` | 385 | Pipeline abbrechen | PipelineTab Cancel | — |
| `POST /api/session/{id}/abort_step` | 408 | LLM-Generation mid-step abbrechen | (kein direktes GUI-Pendant) | — |
| `GET /api/models` | 424 | Modell-Dropdown füllen | Provider-Combo | — |
| `POST /api/analyze/{id}` | 452 | Pipeline starten | PipelineTab Run | **Nein** |
| `POST /api/input/{id}` | 498 | Input-Resolve (DOI/PDF/Image) ohne Pipeline | CrossrefTab + ImageAnalysisTab + Pipeline-Input | — |
| `GET /api/queue/status` | 644 | LLM-Queue-Status | GlobalStatusBar | — |
| `WS /ws/{id}` | 679 | Live-Stream (status/heartbeat/complete/error) | PipelineStreamWidget | — |
| `GET /api/export/{id}` | 776 | JSON-Result-Download | AnalysisReviewTab Export | — |
| `GET /api/session/{id}/recover` | 834 | Auto-saved State laden | (kein GUI-Pendant) | — |
| `DELETE /api/session/{id}` | 1323 | Session löschen | (intern) | — |
| `GET /health` | 1334 | Liveness-Probe | — | — |

### Webapp-Lücken (kritisch für WP9)

- **0 Treffer für `enable_agentic_mode` oder `workflow_name`** in
  `app.py` → Webapp führt **immer** klassische Pipeline aus.
- **Kein Workflow-Picker im Frontend** (`src/webapp/static/index.html`
  hat nur Input-Tabs DOI/File/Camera + 5-Step-Pipeline-Visualisierung,
  aber keine Workflow-Auswahl).
- **Kein Chat-Endpoint** — ChatWidget hat keine Webapp-Entsprechung.
- **Kein Single-Step-Endpoint** — `--only-step` aus CLI fehlt.

→ Webapp ist heute strikt Tier-3-äquivalent zur klassischen Pipeline.
WP9 Tier-Modell muss explizit Catch-Up-Plan oder Lite-Verzicht setzen.

## 4. CLI-Subcommand-Mapping

Source: `src/cli/commands/*.py`. 8 Hauptkommando-Module.

| CLI-Subcommand | Datei | Funktion | GUI-Tab-Äquivalent | Agentic? |
|---|---|---|---|---|
| `pipeline` | `pipeline_cmd.py:47` | Vollständige Pipeline-Analyse | PipelineTab | **Ja** (via `--agentic` Legacy-Flag, plus alle agentic-* Optionen) |
| `batch` | `pipeline_cmd.py:388` | Batch-Verarbeitung | BatchProcessingDialog | Nein |
| `workflow <name>` | `workflow_cmd.py:104` | v4-Workflow ausführen | PipelineTab (mit workflow_name) | **Ja** (nativ v4) |
| `workflows list` | `workflow_cmd.py:84` | Workflows auflisten | PipelineConfigDialog Workflow-Combo | — |
| `show-protocol` | `protocol_cmd.py:15` | JSON-Result anzeigen | AnalysisReviewTab | — |
| `search` | `search_cmd.py:17` | Keyword-Suche standalone | SearchTab | Nein |
| `load-state` | `state_cmd.py:15` | KeywordAnalysisState laden | AnalysisReviewTab Load | — |
| `save-state` | `state_cmd.py:68` | State speichern (deprecated) | (kein direktes GUI) | — |
| `provider list/add/edit/test` | `provider_cmd.py` | Provider-Verwaltung | SettingsDialog | — |
| `database setup/migrate/clear-cache` | `database_cmd.py` | DB-Verwaltung | (kein GUI-Pendant) | — |
| `setup` | `setup_cmd.py:13` | First-Run-Wizard | FirstStartWizard | — |

### CLI-Besonderheiten (relevant für WP10)

- **Zwei agentic-Einstiegspunkte**: `pipeline --agentic …` (Legacy mit
  agentic-Flags) und `workflow <name>` (nativ v4). Konsolidierung
  nötig?
- **`--only-step` funktioniert** in beiden (`pipeline_cmd.py:237`,
  `workflow_cmd.py:152`). UI hat das nicht (WP5).
- **Provider/Database/Setup haben kein GUI-Pendant** für Power-User-
  Operationen außer der Settings-Dialog.

## 5. Überlapp-Matrix

6 identifizierte Funktional-Überlappungen (Code-Inspektion).

| # | Paar | Code-Overlap | UX-Differenz | Empfehlung |
|---|---|---|---|---|
| O1 | UB-Katalog ↔ DK-Analysis-Unified | beide rufen `PipelineStepExecutor.execute_dk_search`, gleiches Token-Config, gleicher Result-Formatter | UB-Katalog: Stats + Top-10-Display; DK-Analysis-Unified: LLM-Input-Transformer | **Merge zu unified DK-Flow** (DK-Analysis konsumiert UB-Katalog-Results) |
| O2 | SearchTab ↔ Pipeline-`search`-Step | identischer `GndSearcher`/`MetaSuggester`, gleiches `SearchResult`-Modell | SearchTab: interaktiv mit manueller Curation; Pipeline-Step: programmatisch automatisch | **Beide behalten**, Config-Loading zentralisieren |
| O3 | AbstractTab ↔ Pipeline-LLM-Steps | identischer `PipelineStepExecutor`, identische Prompts via prompts.json-Tasks | AbstractTab: manuelle Task-Wahl + Parameter-Editor; Pipeline-Steps: vorkonfiguriert | **Beide behalten**, AbstractTab nutzt Pipeline-Pfad bereits intern (siehe `start_analysis()`) |
| O4 | ComparisonTab ↔ AnalysisReviewTab | beide laden `KeywordAnalysisState` via `PipelineJsonManager.load_analysis_state()`, lesen identische Felder | ComparisonTab: Zwei-State-Diff mit Color-Chips; ReviewTab: Single-State-Detail über 11 Sub-Tabs | **Beide behalten**, Sub-Tab-Strukturen sind verschieden trotz Label-Ähnlichkeit |
| O5 | CrossrefTab ↔ Pipeline-DOI-Input | beide nutzen `UnifiedResolver.resolve(doi)`, identisches DOI-Config | CrossrefTab: interaktive DOI-Eingabe + 4-Tab-Result; Pipeline: programmatisch als Input-Resolution | **CrossrefTab refactor** zu Pipeline-Result-Renderer (`display_metadata()` existiert schon line 275-294) |
| O6 | ImageAnalysisTab ↔ Pipeline-Image-Input | beide rufen `llm_service.generate_response(image=...)` mit gleichen Image-Prompts | ImageAnalysisTab: Multi-Image-Queue + Stop-Button + Temp/Seed-Controls; Pipeline: auto, einzelnes Bild | **Beide behalten** — Image-Utils extrahieren, ImageAnalysisTab UI-Polish behalten |

## 6. Empfehlung pro Tab

| # | Tab | Aktion | Begründung |
|---|---|---|---|
| 1 | 🚀 Pipeline | **Keep** + Workflow-Picker prominenter (WP9) | Zentral, alle Frontends bauen drauf auf |
| 2 | 🌐 DOI | **Refactor** zu Pipeline-DOI-Renderer | O5 — Code-Pfad identisch, UX-Mehrwert nur im Standalone-Anteil |
| 3 | 📷 Bild | **Keep** | O6 — UI-Polish (Multi-Image-Queue, Stop, Controls) fehlt im Pipeline-Pfad |
| 4 | 📝 Abstract | **Keep** + zu generischem Single-Step-Tab ausbauen (WP5) | O3 — schon Single-Step-Pattern, ideal für WP5-Auto-Tab-Generator |
| 5 | 🔍 GND-Suche | **Keep** als Komfort-Tool + Pipeline-Result-Renderer-Funktion ergänzen | O2 — manuelle Curation hat eigenständigen Wert; State-Mutation hier ist heute einziger Schreib-Pfad in UI |
| 6 | ✅ Verifikation | **Refactor / mergen mit Tab 4** | Identische Klasse — Tab-Inflation ohne funktionalen Grund. Lösung: Tab 4 mit Task-Switcher prominent statt zwei Tabs |
| 7 | 📚 UB-Katalog | **Merge mit Tab 8** (DkAnalysisUnifiedTab) | O1 — beide rufen `execute_dk_search`, redundant |
| 8 | 📊 Klassifikationen | **Keep** als unified DK-Sicht + UB-Katalog-Funktion absorbieren | O1 + ist bereits AbstractTab-Subclass-Pattern (B2) |
| 9 | 📊 Review | **Keep** + WP4 Renderer-Registry-Quelle | 11 Sub-Tabs sind faktisch Renderer-Slots, ideale Migration für WP4 |
| 10 | 🔍 Vergleich | **Keep** als eigenständig | Forschungspfad-Vergleichswerkzeug (WP2) |

### Zusammenfassung Tab-Bewegungen

- **Keep**: Pipeline, Bild, Abstract, GND-Suche, Klassifikationen,
  Review, Vergleich (7 Tabs)
- **Refactor**: DOI (zu Pipeline-Renderer), Verifikation (mergen mit
  Abstract via Task-Switcher) — 2 Tabs reduziert
- **Merge**: UB-Katalog → Klassifikationen — 1 Tab reduziert

→ **10 Tabs → 7 Tabs** mit klarer Funktion. Konsolidierung um 30%
ohne Funktionsverlust.

## 7. Operator-Befragungs-Liste

Pro Frage Default-Empfehlung. Operator bestätigt oder revidiert.

### F1: Welche Tabs werden täglich/wöchentlich/nie genutzt?
**Empfehlung**: Reine User-Frage — Code-Audit liefert kein Kriterium.
Vorschlag für Antwort: persönliche Einschätzung Operator + Library-
Team-Befragung wenn möglich. Unbenutzte Tabs könnten in WP10-Phase
deletiert werden.

### F2: Ist UB-Katalog separat brauchbar (jenseits DK-Klassifikation)?
**Empfehlung**: **Nein** — beide Tabs (UB-Katalog + DK-Analysis-Unified)
rufen `execute_dk_search`, gleicher Backend-Pfad (Überlapp O1). UB-
Katalog historisch eigenständig, heute redundant. **Merge** mit
DK-Analysis-Unified.

### F3: Wer nutzt ComparisonTab und wofür?
**Empfehlung**: **Behalten als eigenständig** für Forschungspfad
(WP2 quantifiziert classic ↔ agentic). ComparisonTab ist genau das
Werkzeug dafür ("Aktuell → A/B" + Diff). Nicht in Review einbetten.

### F4: Crossref-Standalone-Bedarf vs nur Pipeline-Input?
**Empfehlung**: **Refactor zu Pipeline-DOI-Renderer**. Standalone
liefert keinen UX-Mehrwert über Pipeline-Input + Result-Anzeige
hinaus. UnifiedResolver und 4-Tab-Result-Display können im Pipeline-
Step-Tab bleiben.

### F5: Image-Analyse-Standalone-Bedarf?
**Empfehlung**: **Behalten** — UI-Polish (Multi-Image-Queue, Stop-
Button, Temp/Seed-Controls) fehlt im Pipeline-Pfad. Sinnvoll für
schnelle Vorab-OCR ohne ganze Pipeline.

### F6: SearchTab manuelle Nachsuche — kritisch oder Komfort?
**Empfehlung**: **Behalten als Komfort-Tool** plus Pipeline-Result-
Renderer-Funktion ergänzen (B2-Pattern aus
`consolidation_inventory.md`). Manuelle Curation hat eigenständigen
Wert. SearchTab ist heute einziger State-Mutator → bei WP6-State-Sync
wichtiger Testfall.

### F7: DkAnalysisUnifiedTab — Standard-Sicht oder Tool-Modus?
**Empfehlung**: **Standard-Sicht** für post-Pipeline-DK-Review +
zusätzlich Tool-Modus für manuelle DK-Klassifikation einzelner
Texte. AbstractTab-Subclass-Pattern bereits dafür ausgelegt.

### F8: Tab-Inflation Verifikation ↔ Abstract — Operator-Schmerz oder akzeptiert?
**Empfehlung**: **Mergen via Task-Switcher** in einem Tab "📝
Manuelle Analyse". Heute zwei Tabs für dieselbe Klasse mit
verschiedenem Default-Task ist verwirrend. Switcher macht Wechsel
transparent.

## Footnotes / Querverweise zu Vorbefunden

- Sub-Tab-Korrekturen verifiziert gegen Audit-Finding 12 (`audit_findings.md`):
  AnalysisReviewTab hat tatsächlich 11 Sub-Tabs (verifiziert
  `analysis_review_tab.py:243-392`), nicht 10 wie in Finding 12 vermerkt.
- DkAnalysisUnifiedTab erbt AbstractTab → Audit-Finding 13 bestätigt.
- AgentLoop ist multi-turn-fähig → Audit-Finding 14 — relevant für
  WP8-Chat-UI-Anbindung an Tabs.
- AbstractTab Multi-Variant-Selector → Audit-Finding 15 — relevant
  für WP11-Provider-Portabilität (Modell-Familie wählt Variante).
- LlmService 4 Provider-Typen, openai_compatible deckt OpenAI + GWDG →
  Audit-Finding 16 — Operator-Setup (OpenAI + GWDG + Ollama) trifft
  2 API-Familien.
- Worker-Inflation BatchProcessingDialog → Audit-Finding 17 — 3 lokale
  Worker zusätzlich zu workers.py. Konsolidierungspotenzial.

## Status

✅ WP1 abgeschlossen. Output dieses Dokument.

**Folge-WPs**:
- WP4 (Renderer-Registry) konsumiert: Sektion 2 + 5 + 6
- WP5 (Single-Step-Modell) konsumiert: Sektion 1 (LLM-Pfad-Spalte) +
  Sektion 6 (Tab-4-Empfehlung)
- WP6 (State-Sync) konsumiert: Sektion 1 (State-WRITES-Spalte —
  SearchTab als einziger Mutator)
- WP9 (Frontend-Tier) konsumiert: Sektion 3 + 4 (Webapp/CLI-Mapping)
- WP10 (Migration) konsumiert: alle Sektionen, insbesondere Sektion 6
  + 7 nach Operator-Bestätigung
