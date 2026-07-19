# Migrations- und Decision-Timeline (WP10)

**Status**: Synthese-Dokument für T4-Decision **„Phasen-Reihenfolge"**.
Output von WP10 aus [`wp_detailed_plans.md`](wp_detailed_plans.md).
**Doc-only, kein Code-Bau.** Implementation startet pro Phase als
separates Ticket nach Operator-Approval.

**Methode**: Synthese aller 10 abgeschlossenen Architektur/Audit-WPs
plus Operator-Memory zur Tab-Nutzung.

**Inputs**:
- [WP1](audit_tab_inventory.md) — Tab-Audit (10→7 Empfehlung)
- [WP2](research_classic_vs_agentic.md) — Forschungspfad-Definition
- [WP3](../workflow_output_schemas.md) — 11 Render-Slots
- [WP4](renderer_registry_design.md) — Renderer-Architektur
- [WP5](../single_step_model.md) — Single-Step-Modell
- [WP6](../state_sync_design.md) — Mutations-API + EventBus
- [WP7](chat_tools_design.md) — Chat-Tool-Architektur
- [WP8](chat_ui_design.md) — Chat-UI-Konzept
- [WP9](frontend_tier_model.md) — Frontend-Tier-Modell
- [WP11](provider_portability_design.md) — Provider-Portabilität
- Operator-Memory: Pipeline-Haupteinsatz, Crossref-entfernen,
  Abstract+Verifikation-mergen, Review-needs-love, GND+UB-Katalog-
  mergen, Klassifikationen-separat.

## 1. Executive Summary

- **8 Phasen P-α bis P-θ** + **1 Webapp-Pull-Group**.
- **10 → 7 GUI-Tabs** mit Operator-Memory-Overrides
  (Crossref-DELETE, UB-Katalog-zu-GND-Merge).
- **Empfohlene Reihenfolge**: Foundation (P-α, P-β, P-η) →
  User-Wins (P-γ, P-θ) → Chat (P-δ, P-ε) → Webapp (P-ζ parallel).
- **Total-Aufwand**: ~29 PT (~6 Wochen 1 PT/Tag inkl. Tests).
- **Forschungspfad-Schutz**: Smoke-Test-Set aus
  `Cadmium_Cogito*.json` + `Cadmium_Umwelttoxikologie_text_*.json`
  pro Phase.
- **Roll-Back**: git-Tag `wp10-p<letter>-pre` vor jeder Phase.

## 2. Phasen-Definition

### Übersicht

| Phase | Ziel | Quell-WPs | Aufwand (PT) |
|---|---|---|---|
| **P-α** | Renderer-Skelett: `BaseRenderer` + `RENDERER_REGISTRY` + `raw_json`-Fallback | WP3, WP4 | 2 |
| **P-β** | AnalysisReviewTab-Sub-Tabs → Renderer extrahieren (dk_table, keyword_chains, +3 weitere) | WP4, WP1 | 3 |
| **P-γ** | `SingleStepDialog` + Form-Builder + Warm-Start + Auto-Tabs für aktiven Workflow | WP5 | 4 |
| **P-δ** | Chat-Tools read-only + `AlimaStateBus` + `ChatAgentWorker` ersetzt single-shot ChatWorker | WP6, WP7, WP11 | 5 |
| **P-ε** | Chat-Mutations-Tools + Mutations-Modal + Schreib-Tool-Events über EventBus | WP6, WP7, WP8 | 4 |
| **P-ζ** | Webapp-Catch-Up Tier-1: Workflow-Picker + Single-Step + Async-Batch | WP9 | 5 |
| **P-η** | Provider-Variants in prompts.json (9 neue Varianten) + Seed-Retrofit (7+1 Stellen) | WP11 | 3 |
| **P-θ** | Tab-Konsolidierung 10→7: Crossref-delete + Abstract+Verifikation-merge + GND+UB-merge | WP1, operator-memory | 3 |

**Total**: 29 PT.

### P-α — Renderer-Skelett
- **Inputs**: WP3 Slot-Vokabular (11 Slots), WP4 Sek 2-3 API + Registry.
- **Deliverables**:
  - `src/ui/renderers/base.py` (`BaseRenderer` ABC).
  - `src/ui/renderers/registry.py` (`RENDERER_REGISTRY`,
    `@register_renderer`, `get_renderer`).
  - `src/ui/renderers/raw_json.py` (`RawJsonRenderer`-Fallback).
  - Auto-Discovery in `src/ui/renderers/__init__.py`.
  - 3 Unit-Tests: Registration-Collision-Check, Lookup-Fallback,
    raw_json-Rendering.
- **Smoke-Test**: `get_renderer("slot:unknown")` liefert
  `RawJsonRenderer`; `render_html({any: "data"})` crash-frei.
- **Roll-Back-Tag**: `wp10-pα-pre`.

### P-β — AnalysisReviewTab Renderer-Migration
- **Inputs**: WP4 Sek 5 (3+1 Beispiel-Renderer), WP1
  AnalysisReviewTab-Sub-Tab-Karte.
- **Deliverables**:
  - `DkTableRenderer` (extrahiert aus
    [`analysis_review_tab.py:670-732`](../src/ui/analysis_review_tab.py)).
  - `KeywordChainsRenderer` (aus AnalysisReviewTab Sub-Tab Keywords).
  - `DuplicateTableRenderer` (für `title_list_search`-Workflow).
  - `TitleListRenderer` (für `title_list_search`).
  - `GndPoolRenderer` (Sub-Tab Such-Ergebnisse).
  - AnalysisReviewTab.populate_* ruft intern Renderer (HTML/Widget
    identisch zu HEAD).
- **Smoke-Test**: Snapshot-Test pro Renderer-Output gegen HEAD-
  Rendering. Differenz = 0.
- **Roll-Back-Tag**: `wp10-pβ-pre`.

### P-γ — Single-Step-Execution
- **Inputs**: WP5 vollständig.
- **Deliverables**:
  - `SingleStepDialog` (Modal in `src/ui/dialogs/single_step_dialog.py`).
  - `StepFormBuilder` (Pseudo-Code aus WP5 Sek 3).
  - Pre-flight-Check (Sek 4 WP5).
  - Warm-Start-Helper `KeywordAnalysisState → SharedContext`-Bridge.
  - PipelineTab: Auto-Tabs für aktiven Workflow beim Workflow-Wechsel.
  - Menü-Eintrag `Tools → Run Single Step…` + Keybinding `Ctrl+Shift+S`.
- **Smoke-Test**: Cadmium-Workflow JSON laden, Step
  `classification` selektiv re-runnen, identische
  `dk_classifications` wie HEAD.
- **Roll-Back-Tag**: `wp10-pγ-pre`.

### P-δ — Chat-Tools read-only + AgentLoop-Wiring
- **Inputs**: WP6 (EventBus + Mutations-API skeleton), WP7 (Tool-
  Hierarchie + read-only-Tools), WP11 (Chat-Provider-Default).
- **Deliverables**:
  - `src/core/state_bus.py` (`AlimaStateBus` Singleton).
  - `KeywordAnalysisState`-Mutations-Methoden (Sek 4 WP6),
    Aufrufer in `pipeline_manager.py` migriert (no behavior change).
  - `src/ui/chat_tools/base.py` (`BaseChatTool`).
  - `src/ui/chat_tools/registry.py` (`CHAT_TOOL_CLASSES` +
    `@register_chat_tool`).
  - 4 Generic-Tools (`list_available_data`, `get_extra`,
    `get_step_result`, `get_messages_history`).
  - 7 ALIMA-Tools inkl. `validate_gnd_term`.
  - MCP-Tool-Adapter.
  - `chat.default_provider`, `chat.default_model`,
    `chat.no_cache_writes`, `chat.max_iterations` in `config.json`.
  - `ChatAgentWorker` ersetzt `ChatWorker`.
  - Custom-QWidget-pro-Turn-History (Sek 3 WP8).
  - `PipelineManager.last_shared_context`-Retention (WP6 Sek 2).
- **Smoke-Test**: Pipeline-Lauf abgeschlossen → Chat öffnen →
  `list_available_data`-Aufruf returnt JSON mit
  `slots_populated` ≥ 1.
- **Roll-Back-Tag**: `wp10-pδ-pre`.

### P-ε — Chat-Mutations
- **Inputs**: WP6 (Mutations-API + Locking), WP7 (Schreib-Tools),
  WP8 (Mutations-Modal).
- **Deliverables**:
  - `propose_keyword_replacement`, `propose_keyword_addition`,
    `propose_keyword_removal`, `propose_step_rerun` Chat-Tools.
  - `MutationProposalDialog` (3-Button: Ja/Nein/Bearbeiten).
  - EventBus-Subscribe in ChatWidget.
  - WP6-Lock-Disziplin in `PipelineWorker` (acquire vor Step-
    Mutation, release nach `done`).
  - Re-Run-Stream-Routing in `PipelineStreamWidget`.
  - `chat.allow_writes`-Feature-Flag (default off).
- **Smoke-Test**: Chat → `propose_keyword_replacement("Cd", "Cadmium",
  "4007249-3")` → Modal öffnet, `Ja` → `state.apply_keyword_replacement`
  ruft → `state_changed`-Event broadcastet → AnalysisReviewTab
  reflektiert neuen Keyword.
- **Roll-Back-Tag**: `wp10-pε-pre`.

### P-ζ — Webapp-Catch-Up Tier-1
- **Inputs**: WP9 Sek 5 (5-Punkt-Roadmap).
- **Deliverables (Schritt 1-3 der WP9-Roadmap)**:
  - `GET /api/workflows` — Liste aller Workflows.
  - `POST /api/analyze/{id}` erweitert um `workflow_name`,
    `enable_agentic_mode`, `only_step`.
  - `<select id="workflow-picker">` in `index.html`.
  - `POST /api/step/{session_id}` mit `step_id` + `step_input`.
  - Step-Detail-Panel im Frontend.
  - `POST /api/batch/{id}` + `GET /api/batch/{id}/status` (Async-Job).
  - Batch-Tab im Frontend.
  - Banner: „Workflow-Picker neu — Standard ist `alima_classic`".
- **Smoke-Test**: `curl POST /api/analyze/test
  -d 'workflow_name=catalog_search&...'` → 200 mit valid
  session_id. `GET /api/workflows` listet ≥ 6 Workflows.
- **Roll-Back-Tag**: `wp10-pζ-pre`.
- **Pull-Group**: Eigene Group, kann parallel ab P-α starten wenn
  Resourcen.

### P-η — Provider-Variants + Seed-Retrofit
- **Inputs**: WP11 Sek 5 (Prompt-Varianten-Strategie), Sek 8 (Seed-
  Retrofit-Spec 7+1 Stellen).
- **Deliverables**:
  - `config/model_capabilities.yaml` initial-populated für 3
    Provider × ~5 Modelle (WP11 Sek 3).
  - `model_capabilities.py` YAML-Loader.
  - 9 neue Prompt-Varianten in `prompts.json` (WP11 Sek 5).
  - Seed-Retrofit in 7+1 Stellen
    ([WP11 Sek 8](provider_portability_design.md)).
  - `seed`-Parameter in `LLMAgentStep._llm_params` + `AgentLoop.run()`.
  - YAML-Schema: `seed:`-Feld in `llm:`-Block.
- **Smoke-Test**: 2× Run `alima pipeline --seed 42 --workflow
  alima_classic` mit identischem Input → identische
  `extracted_keywords` (byte-exact). Falls Provider
  `seed_support: false` (Anthropic) → fallback `temperature=0` +
  Warnung.
- **Roll-Back-Tag**: `wp10-pη-pre`.

### P-θ — Tab-Konsolidierung
- **Inputs**: WP1 Sek 6 + Operator-Memory-Overrides (siehe Sek 3).
- **Deliverables (gestaffelt, 1 Tab pro Sub-Phase)**:
  - **P-θ.1**: Crossref-Tab DELETE. Pipeline-DOI-Input bleibt.
  - **P-θ.2**: Abstract + Verifikation → „📝 Manuelle Analyse"
    mit Task-Switcher.
  - **P-θ.3**: GND-Suche + UB-Katalog → „🔍 Suche" mit Quellen-
    Auswahl (Radio: GND / SWB / Lobid / UB-Katalog / DK).
  - **P-θ.4**: Pipeline-Tab Workflow-Picker prominenter (Header-
    Combo).
  - One-Time-Banner beim ersten App-Start nach P-θ.
- **Smoke-Test**: 7 Tabs sichtbar (Pipeline, Bild, Manuelle-Analyse,
  Suche, Klassifikationen, Review, Vergleich); Crossref-Tab nicht
  mehr da. Pipeline-Run liefert identische Outputs wie HEAD.
- **Roll-Back-Tag**: `wp10-pθ-pre`. Wegen Gestaffelung pro Sub-Phase
  eigener Tag (`wp10-pθ1-pre`, etc.).

## 3. Tab-Entscheidungs-Tabelle

Operator-Memory ([`tab_usage_reality.md`](../tab_usage_reality.md))
überschreibt WP1-Audit wo abweichend.

| # | Tab | Aktion | Phase | Begründung |
|---|---|---|---|---|
| 1 | 🚀 Pipeline | **Keep** + Workflow-Picker prominent | P-θ.4 | Operator: Haupteinsatz. WP9: Tier-0. |
| 2 | 🌐 Crossref (DOI) | **DELETE** | P-θ.1 | Operator-Override vs WP1 (Audit wollte refactor zu Renderer): „nicht refactor, weg". Pipeline-DOI-Input deckt Bedarf. |
| 3 | 📷 Bild | **Keep** | — | Operator: Advanced. UI-Polish (Multi-Image-Queue, Stop) lohnt. |
| 4 | 📝 Abstract | **Keep** + Task-Switcher mit Verifikation | P-θ.2 | Operator: identische Klasse mit Verifikation. Switcher transparent. |
| 5 | 🔍 GND-Suche | **MERGE** mit UB-Katalog → „🔍 Suche" | P-θ.3 | Operator-Override vs WP1 (Audit wollte UB-Klassifikations-Merge): „generischer determ. Such-Tab mit Quellen-Auswahl". |
| 6 | ✅ Verifikation | **MERGE** mit Abstract | P-θ.2 | siehe #4. |
| 7 | 📚 UB-Katalog | **MERGE** mit GND-Suche | P-θ.3 | siehe #5. |
| 8 | 📊 Klassifikationen | **Keep** separat | — | Operator-Override: „LLM-Sicht, nicht deterministisch". Trennung von Suche bewusst. |
| 9 | 📊 Review | **Keep** + Renderer-Migration | P-β | Operator: „braucht Liebe". WP4 Renderer-Quelle. |
| 10 | 🔍 Vergleich | **Keep** | — | Operator: Forschungspfad-Tool. WP2-Konsument. |

**Resultat**: 10 → 7 Tabs (Pipeline, Bild, Manuelle Analyse, Suche,
Klassifikationen, Review, Vergleich). Konsolidierung 30%.

## 4. Forschungspfad-Schutz (Smoke-Tests pro Phase)

### Test-Inputs (existierende Fixtures)
- `Cadmium_Cogito.json` — agentic-Pfad.
- `Cadmium_Cogito_CLI.json` — CLI-Pfad.
- `Cadmium_Umwelttoxikologie_text_*.json` — classic-Pfad.

Plus pro Phase ein zusätzlicher synthetischer Edge-Case (z.B. leere
Initial-Keywords, max_iterations exhaust).

### Pro-Phase-Kriterien

| Phase | Smoke-Test (Pass = Phase mergebar) |
|---|---|
| P-α | `RawJsonRenderer.render_html({"x": 1})` returnt valid HTML. `get_renderer("slot:nonexistent")` returnt RawJsonRenderer. |
| P-β | `AnalysisReviewTab.populate_detail_tabs` mit Cadmium-State liefert HTML byte-identisch zu HEAD-Snapshot. |
| P-γ | `WorkflowExecutor.run(workflow, ctx, only_step="classification")` mit Cadmium-`SharedContext`-JSON → `dk_classifications` byte-identisch zu HEAD-Pipeline-Run. |
| P-δ | Chat-Tool `list_available_data` returnt `{slots, step_results, extra_keys}`. `validate_gnd_term("Cadmium")` returnt `{verified: True, gnd_id: "4007249-3"}`. |
| P-ε | `propose_keyword_replacement("Cd", "Cadmium", "4007249-3")` → Modal öffnet → `Ja` → `state.apply_keyword_replacement` ruft → State enthält neuen Keyword. EventBus `state_changed` empfangen. |
| P-ζ | `GET /api/workflows` returnt JSON-Liste mit `alima_classic`, `catalog_search`, `synonym_expansion`, `batch_metadata`, `title_list_search`. `POST /api/analyze` mit `workflow_name` startet Session. |
| P-η | 2× `alima pipeline --seed 42 -t "Cadmium..."` → byte-identische `extracted_keywords`. |
| P-θ | App startet → 7 Tabs sichtbar (Crossref weg). Pipeline-Run mit `Cadmium_Cogito.json`-Input liefert identische Outputs. |

### Blockier-Regel
Drift in Smoke-Test → Phase **nicht mergebar**. Roll-Back via Tag.
Operator entscheidet ob fix in Branch oder revert.

## 5. Roll-Back-Punkte

### Schema
- **Tag**: `wp10-p<letter>-pre` vor Phasen-Start.
- **Branch**: `wp10-p<letter>-impl` für Entwicklung.
- **Merge**: nach `main` erst nach Smoke-Test ✓ + Operator-Approval.

### Beispiel-Workflow
```
# Pre-Phase
git checkout main
git pull
git tag wp10-pα-pre
git checkout -b wp10-pα-impl

# Implementation + Tests
# ... Code-Änderungen ...
pytest tests/ -v -k renderer
# Smoke-Test pro Phase aus Sek 4

# Merge
git checkout main
git merge --no-ff wp10-pα-impl
git push

# Falls Smoke-Test fehlschlägt nach merge:
git reset --hard wp10-pα-pre
git push --force-with-lease  # nur Operator
```

### Roll-Back-Test
Vor echtem Implementation-Start jeder Phase: einmal manuell
`git reset --hard wp10-p<letter>-pre` ausführen + verifizieren dass
Codebasis state-pre-Phase ist. Bestätigt dass Tag funktioniert.

## 6. User-Kommunikation (Release-Notes-Template)

Pro Phase ein Release-Notes-Eintrag im
[`CHANGELOG.md`](../CHANGELOG.md). Template:

```markdown
## v0.X.Y — <Datum> (P-<Phase>)

### Sichtbar
- <Was sich für User ändert>

### Unsichtbar / Backend
- <Was im Hintergrund passiert>

### Migration-Hinweis (optional)
- <Was muss User wissen>
```

### Pro-Phase-Skizzen

**P-α/P-β**: keine User-sichtbare Änderung (Renderer-Migration
identisch zu HEAD). Nur Backend-Eintrag.

**P-γ**: „Neuer Eintrag `Tools → Run Single Step…` — einzelne
Pipeline-Schritte selektiv ausführen mit gespeichertem Kontext.
Keybinding `Ctrl+Shift+S`."

**P-δ**: „Chat-Widget nutzt jetzt Tool-Use — Fragen wie ‚Welche
DK-Codes wurden gefunden?' werden via Tool-Aufrufe beantwortet,
nicht Halluzination."

**P-ε**: „Chat kann Keyword-Vorschläge machen — Confirmation-
Dialog erlaubt Übernahme oder Bearbeitung."

**P-ζ**: „Webapp: Workflow-Picker neu — Standard `alima_classic`.
Batch-Tab für Async-Bulk-Analyse."
**+ In-App-Banner** beim ersten Start.

**P-η**: „Pipeline-Reproduzierbarkeit: `--seed`-Parameter retrofit-
giert in alle Steps. Identische Inputs + Seed → identische Outputs
(außer Anthropic)."

**P-θ**: „GUI-Tabs konsolidiert: 10 → 7. Crossref-Tab entfernt
(Pipeline-DOI-Input deckt Bedarf). Abstract + Verifikation gemergt.
GND-Suche + UB-Katalog gemergt zu generischer Suche."
**+ One-Time-Banner**.

## 7. Risiko-Matrix

### Pro-Phase

| Phase | Risiko | Mitigation |
|---|---|---|
| P-α | Renderer-API inkonsistent zu späteren Renderer-Klassen | Pseudo-Code-API aus WP4 Sek 2 unverändert übernehmen |
| P-β | Snapshot-Tests false-negative (HTML-Whitespace) | Normalize-HTML im Snapshot-Comparator |
| P-γ | Bridge `KeywordAnalysisState → SharedContext` unvollständig | typed-field-Liste in WP5 Sek 6 abarbeiten; failing-Test pro Feld |
| P-δ | Race-Condition `last_shared_context` vs neuer Pipeline-Run | Operator-Tests: Pipeline läuft → Chat öffnen → neuer Run → `last_shared_context` zeigt neuesten State |
| P-ε | User-Confirmation übersehen (Modal hinter floating dock) | Modal-Parent = MainWindow, Always-On-Top |
| P-ζ | Webapp-Frontend (HTML) Drift zu Backend-Changes | API-Schema-Test in CI; Frontend-Smoke-Test |
| P-η | Seed-Retrofit Anthropic-Fallback ungetestet | manuelle Verifikation mit aktivierter Anthropic-Config in Test-Env |
| P-θ | User-Verwirrung durch Tab-Wegfall | One-Time-Banner mit Migration-Guide-Link + CHANGELOG-Eintrag |

### Top-Risiken übergreifend

- **R1: Forschungspfad-Drift** — Smoke-Test fehlschlägt nach Merge.
  → Phase blockiert. Strikte Smoke-Test-Gate vor Merge.
- **R2: User-Verwirrung** — Tab-Konsolidierung visible-Change.
  → Banner + Release-Notes + Migration-Guide-Link.
- **R3: AgentLoop-Race-Conditions** in P-δ/P-ε.
  → WP6-Lock-Disziplin (siehe state_sync_design.md Sek 6).
- **R4: Webapp-Catch-Up unterschätzt** — 5 PT für 3 Roadmap-Schritte
  knapp.
  → Pull-Group-Modus, Multi-Sprint-fähig.
- **R5: Phasen-Abhängigkeiten ändern sich** während Implementation.
  → Re-Plan-Check nach jeder Phase, P-Outputs informieren nächste
  Phase.

## 8. Reihenfolge-Empfehlung (T4-Decision)

### Empfehlung: Foundation → User-Wins → Chat → Webapp (parallel)

```
        Foundation                User-Wins              Chat                Webapp-PG
        ──────────────          ─────────────       ──────────────        ─────────────
   1.   P-α  (2 PT)
   2.   P-β  (3 PT)
   3.   P-η  (3 PT)
   4.                         P-γ  (4 PT)
   5.                         P-θ  (3 PT)
   6.                                              P-δ  (5 PT)
   7.                                              P-ε  (4 PT)
   parallel ab P-α:                                                    P-ζ (5 PT)
```

### Begründung
- **Foundation-first**: Renderer (P-α/β) + Seed (P-η) sind versteckter
  Hebel. Alle späteren Phasen profitieren (P-γ Output-Renderer, P-δ
  Tool-Output-Render, P-θ Tab-Migration trivial mit Renderer).
- **User-Wins-mitte**: P-γ + P-θ liefern sichtbaren Wert
  (Single-Step + 10→7 Tabs). Operator + Library-Team sehen Fortschritt,
  Vertrauen wächst.
- **Chat-spät**: P-δ/P-ε sind komplexest (multi-turn, locking,
  mutations). Letzte GUI-Phasen, mit voller Foundation.
- **Webapp-parallel**: P-ζ unabhängig von GUI-Pfad. Eigene Pull-Group
  kann parallel laufen ab P-α (wenn Resourcen).

### Alternative-Reihenfolgen (kurz)

**Alt-1: User-Wins-zuerst (P-θ vor P-α)**
- **Pro**: Operator + Team sieht sofort sichtbare Konsolidierung.
- **Contra**: Tab-Merge ohne Renderer-Foundation = mehr manuelle Code-
  Duplication. Spätere Renderer-Migration muss merged Tabs anpassen.
- **Verworfen**.

**Alt-2: Chat-zuerst (P-δ/P-ε vor P-γ/P-θ)**
- **Pro**: Chat ist „sexy"-Feature.
- **Contra**: Chat ohne Renderer = Tool-Call-Block kann nicht
  workflow-aware rendern. P-β Renderer als Pre-Req sinnvoller.
- **Verworfen**.

**Alt-3: Webapp-zuerst (P-ζ vor allem)**
- **Pro**: WP9 Tier-0-Gap (Workflow-Picker) am dringendsten für
  externe Nutzer.
- **Contra**: WP9 F1 unbeantwortet — Nutzer-Profil unklar. Operator
  kann Pull-Group jederzeit starten.
- **Bedingt valide** wenn Operator F1 mit „viele Webapp-Nutzer"
  antwortet.

## 9. Out-of-Scope für WP10-MVP

Bewusst zurückgestellt (eigene WPs falls Bedarf):

- **Mini-Chat-pro-Tab** (WP8 Sek 2) — eigene WP.
- **Path-B AgentLoop** (WP7 Sek 9) — User-Approval pro Tool-Call,
  zweiter Sprint.
- **Webapp-Chat-UI** (WP8 Sek 8) — Endpoints definiert, Implementation
  eigene WP nach WP9 F1 Operator-Antwort.
- **Per-Step-UI-Variante B** (WP11 Sek 10) — Workflow-Variante parallel
  zur UI-Combo. A reicht für MVP.
- **Audit-Log-Persistenz** (WP6 Sek 8) — Diff-Stream als JSON. Optional,
  eigener Sprint wenn Forschungspfad-Validierung explizit gewünscht.
- **Undo/Redo-UI-Wiring** (WP6 Sek 7) — Backend in P-δ, UI später.
- **Per-Chunk-Step** (WP5 Sek 7) — Chunking bleibt in-Step-atomar.

## 10. Operator-Fragen

### F1: Reihenfolge OK oder User-Wins-zuerst (Alt-1)?
**Empfehlung**: Foundation-first wie skizziert. Alt-1 nur wenn
Operator-Team Druck für sichtbare Tab-Konsolidierung hat.

### F2: P-θ Tab-Konsolidierung Big-Bang oder gestaffelt (P-θ.1-4)?
**Empfehlung**: gestaffelt. Risiko-reduziert + Smoke-Test pro Sub-
Phase einfacher.

### F3: P-ζ Webapp-Catch-Up parallele Pull-Group startbar?
**Empfehlung**: ja, wenn Operator/Team zwei Pfade managen kann.
Sonst seriell nach P-θ.

### F4: 29 PT total — Sprint-Längen? Wöchentlich, zweiwöchentlich?
**Empfehlung**: frei. WP10 liefert PT-Schätzung pro Phase, nicht
Sprint-Plan. Operator-Team-spezifisch.

### F5: Smoke-Test-Fixtures (Cadmium-Sets) ausreichend oder weitere
Workflows als Fixtures?
**Empfehlung**: ausreichend für MVP. Bei neuen Workflows (z.B.
`title_list_search`) eigene Fixture ergänzen.

### F6: User-Banner für P-θ Tab-Konsolidierung Pflicht oder reicht
CHANGELOG?
**Empfehlung**: einmaliges In-App-Banner beim ersten Start nach
P-θ. CHANGELOG separat.

## 11. Cross-References zurück zu Quell-WPs

| Phase | Quell-WP-Sektionen |
|---|---|
| P-α | [WP3](../workflow_output_schemas.md) Sek 2, [WP4](renderer_registry_design.md) Sek 2-3 |
| P-β | [WP4](renderer_registry_design.md) Sek 5, [WP1](audit_tab_inventory.md) Sub-Tab-Karte (AnalysisReviewTab) |
| P-γ | [WP5](../single_step_model.md) Sek 2-10 |
| P-δ | [WP6](../state_sync_design.md) Sek 2-4, [WP7](chat_tools_design.md) Sek 2-5, [WP11](provider_portability_design.md) Sek 11 |
| P-ε | [WP6](../state_sync_design.md) Sek 4-7, [WP7](chat_tools_design.md) Sek 6-7, [WP8](chat_ui_design.md) Sek 4-5 |
| P-ζ | [WP9](frontend_tier_model.md) Sek 5 (5-Punkt-Roadmap) |
| P-η | [WP11](provider_portability_design.md) Sek 3-8 |
| P-θ | [WP1](audit_tab_inventory.md) Sek 6, [tab_usage_reality.md](../../.claude/projects/-home-conrad-src-ALIMA/memory/tab_usage_reality.md) |

## 12. Status

✅ WP10 abgeschlossen.
**Damit alle 11 Architektur/Audit-WPs abgeschlossen**:
- T0/T1-Audit: WP1, WP2, WP3, WP9, WP11
- T2/T3-Architektur: WP4, WP5, WP6, WP7, WP8
- T4-Synthese: WP10 (dieses Doc)

**Implementation startet pro Phase** nach Operator-Approval. P-α
empfohlen als erster Schritt (Foundation, geringes Risiko, 2 PT).

Pendend: Operator-Antworten F1-F6 für T4-Reihenfolge final.
