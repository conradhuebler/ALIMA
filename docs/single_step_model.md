# Single-Step-Execution-Modell (WP5)

**Status**: Faktenbasis-Dokument für T3-Decision **„Auto-generierte
Tabs vs hardcoded"**. Output von WP5 aus
[`wp_detailed_plans.md`](wp_detailed_plans.md). **Pseudo-Code only —
keine Implementation.** Implementierung erfolgt in WP10.

**Methode**: Code-Inspektion (CLI `--only-step`, `WorkflowExecutor`,
`LLMAgentStep`-Chunking, AbstractTab, SingleStepWorker), YAML-Audit
aller 6 Workflows + Status-quo-Audit der heutigen UI.

**Querverweise**:
- [`workflow_output_schemas.md`](workflow_output_schemas.md) (WP3) —
  Input-Schema-Quelle.
- [`renderer_registry_design.md`](renderer_registry_design.md) (WP4) —
  Output-Renderer.
- [`state_sync_design.md`](state_sync_design.md) (WP6) — Mutations-API
  für Output-Übernahme.
- [`frontend_tier_model.md`](frontend_tier_model.md) (WP9) — Single-
  Step ist Tier-1 (F-M2.1, F-M2.2, F-M2.3).
- [`audit_tab_inventory.md`](audit_tab_inventory.md) (WP1) — Tab-
  Inflation-Risiko.
- [`wp_detailed_plans.md`](wp_detailed_plans.md) WP5 — Soll-Definition.

## 1. Executive Summary

- **Backend ist da**: `WorkflowExecutor.run(workflow, ctx, only_step=)`
  filtert Steps auf `[s for s in steps if s.id == only_step]`. CLI
  `--only-step` ruft das.
- **Lücke 1**: kein Auto-Resolve fehlender Inputs — User muss
  vollständigen SharedContext-JSON liefern.
- **Lücke 2**: kein generisches UI — heute `AbstractTab` +
  `SingleStepWorker` mit hardcoded Step-IDs (`keywords`,
  `dk_classification`), kein Step-Combo, kein Form-Builder pro
  Step.
- **Empfehlung**: Modal-Dialog (B7-Pattern) für Step-Auswahl + Form-
  Builder aus `inputs:`-YAML-Block. **Kein** Tab-pro-Step (Tab-
  Inflation 6×7=42).
- **T3-Decision**: hybrid — auto-generierte Top-Level-Tabs nur für
  **aktiven** Workflow (max 7 Tabs), inaktive nicht.
- **Chunking-Verhalten**: per-Chunk-Step bleibt out-of-scope; Single-
  Step läuft alle Chunks in einer Step-Invocation (heutiges Verhalten
  in `LLMAgentStep`).

## 2. Status-Quo CLI

### Path
```
src/cli/main.py:127-129
  argparse: --only-step STEP_ID
                              │
src/cli/commands/workflow_cmd.py:152-153
  WorkflowExecutor.run(wf, ctx, only_step=step_id, stop_on_error=True)
                              │
src/core/agents/workflow_executor.py:59-102
  steps_to_run = [s for s in wf.steps if s.id == only_step]   # Z.94-95
```

### Heutiges Aufruf-Muster
```bash
alima workflow alima_classic \
    --input-file my_state.json \
    --only-step classification
```

`my_state.json` muss enthalten:
- alle `inputs:`-Felder, die `classification`-Step erwartet
  (siehe YAML Sek 3)
- alle typed-fields die der Step-Code direkt aus SharedContext liest

**Kein automatic-dependency-resolution**: wenn `classification`-Step
`dk_search_results` braucht aber JSON nur `dk_classifications` liefert,
schlägt der Step in-flight fehl (KeyError oder leerer Loop).

### Konsequenz für UI
UI muss vor `run(only_step=)`:
1. Step auswählbar machen (Combo)
2. **Pre-flight-Check**: liest YAML `inputs:`-Block, vergleicht mit
   vorhandenen Kontext-Feldern, listet fehlende Felder dem User auf.
3. Form-Builder zeigt User-fillable-Felder (siehe Sek 3).

## 3. Input-Schema-Auflösung (Algorithmus)

### YAML-Quelle
`workflows/alima_classic.yaml:105-110` für Step `selection_chunks`:

```yaml
inputs:
  abstract: "${abstract}"
  keywords: "${gnd_entries}"

outputs:
  selected_keywords: "response.keywords"
```

### Klassifizierung pro Feld
| Pattern | Kind | Quelle | UI-Verhalten |
|---|---|---|---|
| `${<root_field>}` mit `<root_field>` in SharedContext-typed-fields ohne Schreiber-Step | `user_fill` | User-Input (oder JSON) | Form-Field rendern |
| `${steps.<id>.<key>}` | `derived` | Output von Step `<id>` | Aus SharedContext lesen, anzeigen (read-only) |
| `${<root_field>}` mit `<root_field>` = Schreib-Ziel eines früheren Steps | `derived` | typed-field, geschrieben in Workflow | aus SharedContext lesen |
| literal-String (kein `${...}`) | `static` | YAML-Konstante | unsichtbar im UI |

### Pseudo-Code Generator
```python
# Pseudo-Code für src/ui/forms/step_form_builder.py
from typing import List, Dict

def build_step_form(workflow: WorkflowDef, step_id: str,
                    context: SharedContext) -> List["FormField"]:
    step = workflow.get_step(step_id)
    upstream_writes = _collect_upstream_outputs(workflow, step_id)
    fields = []
    for name, expr in (step.inputs or {}).items():
        kind = _classify(expr, upstream_writes)
        if kind == "user_fill":
            present = _has_value(context, expr)
            fields.append(FormField(
                name=name, expr=expr,
                kind="user_fill",
                value=_resolve(context, expr) if present else None,
                widget=_pick_widget(name, expr),  # text/list/json
            ))
        elif kind == "derived":
            fields.append(FormField(
                name=name, expr=expr,
                kind="derived",
                value=_resolve(context, expr),
                widget="readonly",
            ))
    return fields


def _classify(expr: str, upstream_writes: Set[str]) -> str:
    if not expr.startswith("${"):
        return "static"
    root = expr.strip("${}").split(".")[0]
    if root == "steps" or root in upstream_writes:
        return "derived"
    return "user_fill"
```

### Beispiel: `classification`-Step in `alima_classic.yaml`
```yaml
inputs:
  abstract: "${abstract}"               # user_fill (root)
  selected_keywords: "${selected_keywords}"  # derived (selection-Step)
  dk_entries: "${extra.dk_entries}"     # derived (dk_collect)
```

UI rendert:
- 1 `user_fill`-Form-Field für `abstract` (QTextEdit, wenn leer).
- 2 `derived`-Anzeigen für `selected_keywords` + `dk_entries`
  (read-only, mit Renderer aus WP4 wenn Slot bekannt).

## 4. UI-Pattern

### Empfehlung: Modal-Dialog (B7) statt Tab-pro-Step

```
┌──────────────────────────────────────────────────────────────────┐
│  Run Single Step                                            [×]  │
├──────────────────────────────────────────────────────────────────┤
│  Workflow:  [alima_classic ▾]                                    │
│  Step:      [classification ▾]                                   │
│  Provider:  [ollama / qwen2.5:14b ▾]   (override)                │
│                                                                  │
│  ─ Inputs ─────────────────────────────────────────────────────  │
│                                                                  │
│  abstract            (user_fill)                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Untersuchung zur Cadmium-Toxikologie in aquatischen ...    │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  selected_keywords   (derived from selection)                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ [Cadmium] [Ökotoxikologie] [Aquatische Systeme] + 17 more  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  dk_entries          (derived from dk_collect)  ⚠️ leer          │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ — fehlt — Step kann nicht laufen.                          │ │
│  │ [ Run prerequisite: dk_collect ]                            │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  [ Load state ... ]              [ Cancel ]    [ Run step ]      │
└──────────────────────────────────────────────────────────────────┘
```

### Begründung
- **Tab-Inflation vermeiden** (siehe WP9): 6 Workflows × 7 Steps =
  potenziell 42 Tabs. Modal hat keinen Top-Level-Footprint.
- **B7-Pattern** ist im Bestand etabliert (z.B. ImageAnalysisDialog,
  BatchProcessingDialog).
- **Workflow-Kontext bleibt sichtbar** — User wechselt nicht Tab.

### Alternative: Sub-Tab im PipelineTab
Pro aktivem Workflow ein zusätzlicher „Single-Step"-Sub-Tab im
PipelineTab. **Verwerfung**: PipelineTab heute schon dicht; Sub-Tab-
Dispatcher wäre noch ein Code-Layer.

### Entry-Points
- Menü `Tools → Run Single Step ...`
- Keybinding `Ctrl+Shift+S`
- Kontext-Menü auf einem Step-Output im PipelineTab: „Re-run this step…"

## 5. Output-Übernahme

### Heutige Lage
Nach `WorkflowExecutor.run(only_step=)` ist `step_results[step_id]`
gesetzt + typed-fields per `apply_outputs` geschrieben. UI hat aber
keinen Knopf „Output in nächsten Step übernehmen".

### Flow
1. Step läuft, Output landet in `SharedContext.step_results[step_id]`.
2. Modal zeigt Output via WP4-Renderer (per Slot, default `raw_json`).
3. Button „Apply & continue" → context wird persistiert
   (`save_to_file()`), Modal lädt nächsten Step.
4. **Mutations-Event**: über `AlimaStateBus.emit_event("state_changed",
   diff)` (WP6) — andere UI-Komponenten (PipelineTab,
   AnalysisReviewTab) updaten sich.

### Pseudo-Code
```python
# In SingleStepDialog (neu)
def on_run_clicked(self):
    self._stash_user_fill_inputs()        # write user-fill → context
    report = self.executor.run(
        self.workflow, self.context, only_step=self.step_id,
    )
    if report.success:
        # Persistieren + Event-Emit
        self.context.save_to_file(self._state_path)
        bus = AlimaStateBus()
        bus.emit_event(
            "state_changed",
            {"op": "step_executed", "step_id": self.step_id,
             "ts": _now()},
        )
        self._render_output(report.step_results[self.step_id])
        self._load_next_step()
    else:
        self._show_error(report.error)
```

## 6. Warm-Start aus JSON

### Sequence
```
1. User: "Load state..." klicken (Modal-Toolbar oder Menü)
2. QFileDialog → wählt SharedContext-JSON oder
   KeywordAnalysisState-JSON
3. SingleStepDialog._load_state(path):
     ctx = SharedContext.load_from_file(path)
     # oder: state = KeywordAnalysisState.from_json(path);
     #        ctx = _build_context_from_state(state)
4. Step-Combo aus Workflow-Steps (alle, mit
   "✓"-Tag pro Step, dessen Output bereits in ctx steht).
5. User wählt Step → Form-Builder läuft (Sek 3).
6. „Run step" → WorkflowExecutor.run(only_step=).
```

### Persistenz (heute schon da)
- `SharedContext.save_to_file(path)` / `load_from_file(path)` in
  [`shared_context.py`](../src/core/agents/shared_context.py).
- `SharedContext.to_keyword_analysis_state()` (Z. 168-375) Bridge bei
  Bedarf.

### State-Bridge-Logik (für `KeywordAnalysisState`-Input)
`KeywordAnalysisState` ist post-Lauf-State. Bridge in umgekehrter
Richtung (`from_keyword_analysis_state`) müsste rekonstruieren:
- `abstract` ← `state.original_abstract`
- `initial_keywords` ← `state.initial_keywords`
- `gnd_entries` ← `state.search_results` (flatten)
- … (Auflistung in WP10-Implementation)

**Ungelöst heute**: Bridge KAS→SharedContext existiert nicht;
für Warm-Start aus `analysis_export_*.json` ist neue Helper-Funktion
nötig. WP10-Aufgabe.

## 7. Chunking-Single-Step-Verhalten

### Heutiges Verhalten
`LLMAgentStep.run()` (siehe
[`src/core/agents/steps/llm_agent_step.py:121-200`](../src/core/agents/steps/llm_agent_step.py))
führt bei `chunking: enabled: true` die LLM-Aufrufe intern aus, **eine
Step-Invocation = N Chunks**. Beispiel-YAML
(`workflows/alima_classic.yaml:120-129`):

```yaml
chunking:
  enabled: true
  chunk_field: keywords
  chunk_fields: [title, gnd_id]
  chunk_size: 350
  sort_by: count
  sort_desc: true
  merge_key: keywords
  dedup_field: keyword
  max_merged: 80
```

### Single-Step-Aufruf
`WorkflowExecutor.run(only_step="selection_chunks")` ruft genau einen
`LLMAgentStep.run()` → der Step macht intern N LLM-Calls + Merge.
Aus User-Sicht eine atomare Operation.

### Empfehlung
- **Beibehalten** — per-Chunk-Step wäre Workflow-Schema-Bruch.
- **UI-Feedback**: Modal zeigt Chunk-Progress („Chunk 3/12 …") via
  `stream_callback`. Heute schon möglich, aber UI muss subscriben.
- **Selektives Re-Run einzelner Chunks**: out-of-scope für WP5,
  potenziell eigene WP.

### Risiken
- Single-Step für chunking-Step ist teuer (N LLM-Calls) — Warnung im
  UI vor `Run` wäre nice-to-have.

## 8. Generator vs Hardcode (T3-Decision)

### Optionen
| Option | Beschreibung | Aufwand | Risiko |
|---|---|---|---|
| (A) Hardcoded Tabs pro Step | wie heute, pro Workflow neu codieren | groß (6 Workflows × 7 Steps = 42 Tabs) | Tab-Inflation, Code-Duplikation |
| (B) Auto-Generator ALLE Steps | `BaseSingleStepTab` aus YAML | mittel | UX rigide (kein per-Step-Customizing), 42 Tabs |
| **(C) Hybrid (Empfehlung)** | nur **aktiver** Workflow als Auto-Tabs, max 7. Modal zusätzlich für ad-hoc. | klein-mittel | Workflow-Wechsel braucht Re-Build-Schritt |
| (D) Modal-only | gar keine Tabs, nur Modal-Dialog (Sek 4) | klein | weniger discoverable |

### Empfehlung **C (Hybrid)** + zusätzlich Modal
- Top-Level: max 7 Tabs für **aktiven** Workflow (auto-generiert beim
  Workflow-Wechsel).
- Modal: Cross-Workflow-Single-Step (z.B. „chunk eines anderen
  Workflows analysieren"). Modal ist auch der Entry-Point für Warm-
  Start aus JSON anderen Workflows.
- Begründung: 7-Tabs-Cap entspricht Operator-Mentalmodell (1
  Workflow = 1 Pipeline-View); Modal ist Power-User-Pfad.

### Implementations-Skizze für (C)
```python
# Pseudo-Code in PipelineTab
def on_workflow_changed(self, name: str):
    self._dispose_single_step_tabs()
    wf = self.executor.load_workflow(name)
    for step in wf.steps:
        if step.type in ("llm_agent", "deterministic"):
            tab = SingleStepTab(workflow=wf, step_id=step.id,
                                context=self.shared_context)
            self.tabs.addTab(tab, step.id)
```

`SingleStepTab` ist eine Subklasse von `QWidget` mit Form-Builder aus
Sek 3 + Renderer aus WP4. Kein Sub-Tab-Sprawl: alle 7 sind direkt
unter PipelineTab.

## 9. Decision-Point T3

**T3 = „Auto-generierte Tabs vs hardcoded"**.

### Empfehlung
**Hybrid (C)** — Auto-generierte Tabs nur für aktiven Workflow + Modal
für ad-hoc/cross-workflow. Begründung in Sek 8.

### Abhängigkeit
WP4 (Renderer-Registry) muss für Output-Render verfügbar sein. Das
Renderer-Doc empfiehlt Per-Frontend-Split — passt direkt.

## 10. Migration-Plan (für WP10)

### Phasen
| Phase | Schritt | Sichtbarkeit |
|---|---|---|
| 1 | `SingleStepDialog` als Modal anlegen, eingebunden in Menü `Tools` | sichtbar: neuer Eintrag |
| 2 | Form-Builder + Pre-flight-Check (Sek 3) | sichtbar: Inputs werden klassifiziert |
| 3 | Warm-Start aus JSON (Sek 6) | sichtbar: „Load state…"-Button |
| 4 | `KeywordAnalysisState` → `SharedContext`-Bridge (Helper) | unsichtbar: Bridge intern |
| 5 | (C) Auto-Tabs für aktiven Workflow im PipelineTab | sichtbar: Tabs werden bei Workflow-Wechsel neu gebaut |
| 6 | `AbstractTab`-Migration: vereinfachen oder ersetzen | sichtbar: AbstractTab fällt weg oder wird zu „Generic LLM Tool" |

### `AbstractTab`-Schicksal
Heute = generischer LLM-Call mit Task-Selector. Drei Optionen:
- **Ersetzen** durch `SingleStepDialog` mit ad-hoc-Step.
- **Behalten** als „Generic LLM Tool" (kein Workflow-Kontext nötig).
- **Transformieren** in Form-Builder-Frontend für beliebigen
  Task aus `prompts.json`.

WP5 empfiehlt **Behalten** — Operator nutzt AbstractTab für Quick-
LLM-Calls außerhalb der Pipeline; Migration wäre Verlust.

## 11. Risiken + offene Validierungen

| Risiko | Bewertung | Mitigation |
|---|---|---|
| Form-Builder zu rigide für komplexe `inputs:` (z.B. nested dicts) | mittel | Fallback auf JSON-Editor pro Feld, wenn Schema unbekannt |
| Pre-flight-Check false-negative (Feld als „derived" eingestuft obwohl User es überschreiben will) | niedrig | Edit-Override per „Edit"-Button pro derived-Feld |
| Chunking-Single-Step teuer (viele LLM-Calls in Modal) | mittel | Warnung mit Chunk-Count + geschätzter Dauer vor Run |
| Auto-Tabs bei Workflow-Wechsel (C) führen zu Tab-Flash | niedrig | Tab-Rebuild defer bis Workflow-Picker `applyClicked` |
| Bridge `KeywordAnalysisState → SharedContext` unvollständig | mittel | WP10 spezifiziert Bridge per typed-field-Liste |
| `AbstractTab` bleibt als legacy | niedrig | Akzeptiert (Sek 10), Doc-Markierung als „pre-WP5" |

## 12. Cross-References / Folge-WPs

| Folge-WP | Konsumiert aus WP5 |
|---|---|
| **WP10** (Migration) | Phasen 1-6 (Sek 10) als Migrationsticket-Reihenfolge. |
| **WP8** (Chat-UI) | `propose_step_rerun`-Tool ruft denselben Flow wie SingleStepDialog (Modal mit pre-populated Inputs). |
| **WP4** (Renderer-Registry) | Renderer pro Step-Output-Slot. |
| **WP6** (State-Sync) | Mutations-API für Output-Übernahme (Sek 5). |

## 13. Operator-Fragen (max 6)

### F1: Modal-only oder Hybrid (C) bevorzugt?
**Empfehlung**: Hybrid. Modal als Default-Entry, Auto-Tabs als
Discoverability-Pfad.

### F2: AbstractTab behalten oder ersetzen?
**Empfehlung**: behalten als „Generic LLM Tool".

### F3: Warm-Start aus `KeywordAnalysisState`-JSON Pflicht (Bridge bauen) oder reicht SharedContext-JSON?
**Empfehlung**: Pflicht. `analysis_export_*.json`-Files sind heute der
gebräuchlichste Persistenz-Pfad.

### F4: Chunking-Single-Step Warnung anzeigen (Modal) oder still ausführen?
**Empfehlung**: Warnung mit Chunk-Count + Zeit-Schätzung.

### F5: Pre-flight-Check „fehlende Inputs" — `Run prerequisite`-Button anbieten oder nur Read-Only-Hinweis?
**Empfehlung**: Anbieten. UX-Mehrwert. Implementation: cascading
`only_step=<dependency>`-Run.

### F6: Selektives Re-Run einzelner Chunks (per-Chunk-Step) Pflicht oder OOS?
**Empfehlung**: OOS für WP5. Eigene WP wenn Bedarf.

## 14. Status

✅ WP5 abgeschlossen.
- Status-Quo CLI (Sek 2)
- Input-Schema-Auflösung (Sek 3)
- UI-Pattern Modal (Sek 4)
- Output-Übernahme (Sek 5)
- Warm-Start aus JSON (Sek 6)
- Chunking-Verhalten (Sek 7)
- Generator-vs-Hardcode (Sek 8, T3-Decision)
- Migrations-Plan (Sek 10)
- Risiken (Sek 11)
- Folge-WP-Mapping (Sek 12)

Pendend: Operator-Antworten F1-F6, dann T3 (Auto-Tabs vs hardcoded)
final.
