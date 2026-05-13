# State-Sync + Mutations-Fundament (WP6)

**Status**: Faktenbasis-Dokument für T2-Decision **„EventBus-Pattern
fix"**. Output von WP6 aus
[`wp_detailed_plans.md`](wp_detailed_plans.md). **Pseudo-Code only —
keine Implementation.** Implementierung erfolgt in WP10.

**Methode**: Code-Inspektion (`KeywordAnalysisState`, `SharedContext`,
`PipelineManager` + alle Mutations-Sites, bestehende `pyqtSignal`-
Pattern) plus Audit-Finding 5
([`audit_findings.md`](audit_findings.md)) und
[`agentic_chat_plan.md`](agentic_chat_plan.md) Sek 7.

**Querverweise**:
- [`audit_tab_inventory.md`](audit_tab_inventory.md) (WP1)
- [`research_classic_vs_agentic.md`](research_classic_vs_agentic.md) (WP2)
- [`workflow_output_schemas.md`](workflow_output_schemas.md) (WP3)
- [`renderer_registry_design.md`](renderer_registry_design.md) (WP4)
- [`provider_portability_design.md`](provider_portability_design.md) (WP11)
- [`agentic_chat_plan.md`](agentic_chat_plan.md) — Sek 7 für
  `last_shared_context`-Retention.
- [`wp_detailed_plans.md`](wp_detailed_plans.md) WP6 — Soll-Definition.

## 1. Executive Summary

- **Heute keine kanonische Sync-Mechanik**: Mutations geschehen
  direkt in `PipelineManager` (10+ Stellen), UI liest Felder ad-hoc
  per `getattr`. Kein Event-Bus → Chat-Mutationen können keine
  konsistenten Multi-Tab-Updates auslösen.
- **State-Master**: `KeywordAnalysisState` für post-Lauf,
  `SharedContext` während agentic-Lauf. Heute wird `SharedContext`
  nach Lauf-Ende verworfen.
- **Empfehlung**: `AlimaStateBus`-Singleton (Qt-basiert für GUI,
  Session-scoped für Webapp), Mutations-API auf
  `KeywordAnalysisState`, kompaktes JSON-Diff-Format,
  `PipelineManager.last_shared_context` als Read-Only-Retention.
- **6-8 Diff-Ops** decken die geplante Mutations-API ab.
- **Locking**: Pipeline-Worker hat Vorrang über `threading.Lock`,
  Chat-Mutation queued.
- **Undo/Redo**: Diff-Stack pro Session, Reset bei neuem Pipeline-Run.
- **Persistenz**: Diff-Log neben Pipeline-Result-JSON optional.
- **T2-Decision**: pyqtSignal-Wrapper für GUI, separater Webapp-Backend
  (Session-scoped Event-Channel).

## 2. State-Master-Modell

### Heutige Lage
| State | Definition | Lebensdauer | Mutationen heute |
|---|---|---|---|
| `KeywordAnalysisState` | [`src/core/data_models.py:83-117`](../src/core/data_models.py) — dataclass, 17+ Felder | persistiert bis App-Restart oder JSON-Save | 10+ Stellen in [`pipeline_manager.py`](../src/core/pipeline_manager.py) (Z. 681, 1158, 1329-1331, 1394, 1570-1572, 1606, 1734-1735, 1824) |
| `SharedContext` | [`src/core/agents/shared_context.py:102-507`](../src/core/agents/shared_context.py) — extends `BaseSharedContext`, 30+ ALIMA-Felder + 7 Base-Felder | lebt nur während agentic-Workflow-Run | direkt im Worker via `setattr` / typed-field |

`SharedContext` hat heute schon:
- `save_to_file()` / `load_from_file()` (JSON-Persistenz).
- `to_keyword_analysis_state()` (Z. 168-375) — Bridge auf
  `KeywordAnalysisState` für UI-Konsum nach Lauf-Ende.

### Empfohlenes Modell

```
                       ┌──────────────────────────────────┐
                       │   KeywordAnalysisState (SOT)     │
                       │   – classic AND post-agentic     │
                       │   – exposes mutation methods     │
                       └──────────────┬───────────────────┘
                                      │ subscribe / emit
                                      ▼
                       ┌──────────────────────────────────┐
                       │   AlimaStateBus (Singleton)      │
                       │   – pyqtSignal-Wrapper (GUI)     │
                       │   – Session-Channel (Webapp)     │
                       └──────────────┬───────────────────┘
                                      │
                       ┌──────────────┴──────────────┬─────────────┐
                       ▼                             ▼             ▼
                   AnalysisReviewTab          ChatWidget       Webapp-Client
                   AgenticContextWidget       Tool-Output      (SSE)
                   PipelineTab

  WÄHREND agentic Run:                       NACH Run-Ende:
  PipelineManager._shared_context            PipelineManager.last_shared_context  ◀── NEU
  (mutates SharedContext)                    (read-only retention; Chat reads it)
                                              + to_keyword_analysis_state()
                                                → updates KeywordAnalysisState
```

### `last_shared_context`-Retention
Heute: `SharedContext` wird nach Lauf verworfen → Chat hat keinen
Zugriff auf Raw-`gnd_entries` (1000 Items), Chunk-Responses,
Tool-Result-Cache.

[`docs/agentic_chat_plan.md`](agentic_chat_plan.md) Sek 7 schlägt vor,
`PipelineManager.last_shared_context` als
Read-Only-Attribut zu exponieren. **WP6 übernimmt diese Empfehlung**:
- Property `PipelineManager.last_shared_context: Optional[SharedContext]`
- Set nach erfolgreichem `WorkflowExecutor.run()`
- Reset auf `None` bei `start_pipeline()` (neuer Run = alter Retention
  veraltet).
- Chat-Tools (WP7) lesen ausschließlich read-only.

### Klare Schreibverantwortung
| Schreiber | Schreibt was | Wann |
|---|---|---|
| Pipeline-Worker | `SharedContext.<typed_field>`, `SharedContext.extra.<key>` | während Lauf, Step-für-Step |
| Pipeline-Worker (Ende) | `KeywordAnalysisState` via `to_keyword_analysis_state()` | nach letztem Step |
| Chat-Tools (WP7) | `KeywordAnalysisState` via **Mutations-API** (Sek 4) | jederzeit nach Pipeline-Ende |
| UI-Komponenten | NIEMALS direkt | — |

## 3. EventBus-API (`AlimaStateBus`)

### Vertrag
Pseudo-Code für `src/core/state_bus.py` (neue Datei in WP10):

```python
# Pseudo-Code, nicht zur Implementation
from __future__ import annotations
import threading
from typing import Any, Callable, Dict, List
from PyQt6.QtCore import QObject, pyqtSignal


class AlimaStateBus(QObject):
    """Singleton EventBus für State-Mutations.

    GUI: pyqtSignal-Bridge — Subscribers binden Qt-Slots an die
    `*_signal`-Attribute (thread-safe Qt-Cross-Thread-Dispatch).

    Webapp: Session-scoped Instanz (kein App-globales Singleton im
    Webapp-Backend). Subscribers nutzen `subscribe()` mit Plain-Python-
    Callbacks; intern emittiert über `asyncio.Queue` für SSE-Channel.
    """

    # Qt-Signals (deklarativ, ein Signal pro Event-Typ)
    state_changed = pyqtSignal(dict)      # diff-payload
    keyword_added = pyqtSignal(dict)
    keyword_removed = pyqtSignal(dict)
    keyword_replaced = pyqtSignal(dict)
    classification_replaced = pyqtSignal(dict)
    dk_search_results_set = pyqtSignal(dict)
    pipeline_run_started = pyqtSignal(dict)
    pipeline_run_done = pyqtSignal(dict)

    _instance: "AlimaStateBus | None" = None
    _lock = threading.Lock()

    def __new__(cls):
        # Singleton-Pattern analog UnifiedKnowledgeManager (MEMORY.md)
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def emit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Emit an event by name. Looks up the matching pyqtSignal."""
        signal = getattr(self, event_type, None)
        if signal is not None:
            signal.emit(payload)
        # Plus generic broadcast
        self.state_changed.emit(payload)

    def subscribe(self, event_type: str, handler: Callable[[Dict], None]) -> None:
        """Connect a handler to an event-type signal.

        GUI: equivalent to `bus.<event_type>.connect(handler)`. Plain
        callback variant exists for non-Qt contexts (Webapp).
        """
        signal = getattr(self, event_type, None)
        if signal is not None:
            signal.connect(handler)
```

### Pattern-Vorbild im Bestand
- [`src/core/search_engine.py`](../src/core/search_engine.py) — Signal-
  basierte Search-Komponenten, Qt-Thread-Cross-Dispatch.
- [`src/llm/provider_status_service.py`](../src/llm/provider_status_service.py)
  — globale Status-Signale.
- [`src/core/agents/shared_context.py:18-99`](../src/core/agents/shared_context.py)
  `ToolResultCache` — kein Pub-Sub, aber Thread-safe-Singleton-Pattern.

### Webapp-Adaption (Out-of-Scope für WP6-Skizze)
Webapp-Backend: pro HTTP-Session eigene `AlimaStateBus`-Instanz (kein
App-Singleton, da Sessions isoliert). Event-Channel via `asyncio.Queue`
→ SSE / WebSocket. Konkrete Architektur in WP9 / WP10.

## 4. Mutations-API auf `KeywordAnalysisState`

### Heutige Lage
10+ direkte `state.<feld> = ...`-Mutationen in
[`pipeline_manager.py`](../src/core/pipeline_manager.py):

| Stelle | Feld | Lauf-Phase |
|---|---|---|
| Z. 681 | `input_type`, `source_value` | Start |
| Z. 1158 | `original_abstract` | nach Extraction |
| Z. 1329-1331 | `initial_keywords`, `initial_gnd_classes`, `initial_llm_call_details` | nach Keyword-Extraction |
| Z. 1394 | `search_results` | nach GND-Suche |
| Z. 1570-1572 | `refinement_iterations`, `convergence_achieved`, `max_iterations_reached` | nach Refinement |
| Z. 1606 | `final_llm_analysis` | nach Verifikation |
| Z. 1734-1735 | `dk_search_results`, `dk_search_results_flattened` | nach DK-Aggregation |
| Z. 1824 | `dk_classifications` (Property-Setter) | nach Klassifikation |

### Neue API
Neue Methoden auf `KeywordAnalysisState` — jede emittiert
`state_changed(diff)`:

```python
# Pseudo-Code, Erweiterung von src/core/data_models.py
@dataclass
class KeywordAnalysisState:
    ...  # bestehende Felder unverändert

    def apply_keyword_replacement(self, old: str, new: str, gnd_id: str) -> None:
        """Replace a keyword in final_llm_analysis.extracted_gnd_keywords."""
        diff = {"op": "replace_keyword", "old": old, "new": new, "gnd_id": gnd_id}
        self._apply_and_emit(diff)

    def add_keyword(self, kw: str, gnd_id: str) -> None:
        diff = {"op": "add_keyword", "keyword": kw, "gnd_id": gnd_id}
        self._apply_and_emit(diff)

    def remove_keyword(self, kw: str) -> None:
        diff = {"op": "remove_keyword", "keyword": kw}
        self._apply_and_emit(diff)

    def replace_classification(self, old: str, new: str) -> None:
        diff = {"op": "replace_classification", "old": old, "new": new}
        self._apply_and_emit(diff)

    def set_dk_search_results(self, results: list) -> None:
        diff = {"op": "set_dk_search_results", "count": len(results)}
        self._dk_search_results = results
        self._emit(diff)

    def _apply_and_emit(self, diff: Dict[str, Any]) -> None:
        """Apply diff to internal fields then emit via bus."""
        self._apply_diff(diff)
        self._emit(diff)

    def _emit(self, diff: Dict[str, Any]) -> None:
        from src.core.state_bus import AlimaStateBus
        bus = AlimaStateBus()
        bus.emit_event("state_changed", diff)
        # Spezifischer Event-Typ zusätzlich:
        bus.emit_event(diff["op"].replace("_", "_") + "d", diff)
        # (Konvention: "replace_keyword" -> "keyword_replaced"-Signal)
```

### Migration
- Phase 1: Mutations-API anlegen, `PipelineManager`-Mutations weiterhin
  direkt (no behavior change).
- Phase 2 (WP10): `PipelineManager` ruft Mutations-API statt direkt-
  Schreibens. Streaming-Pfade (z.B. `final_llm_analysis` mid-stream)
  behalten direct-write, Diff-Emit erst nach `done`.
- Phase 3 (WP7): Chat-Tools nutzen ausschließlich Mutations-API.

## 5. Diff-Format (JSON)

### Op-Vokabular
Kompakt, deckt Mutations-API ab. 6-8 Ops reichen heute:

| Op | Payload-Felder | Reverse-Op (für Undo) |
|---|---|---|
| `replace_keyword` | `old`, `new`, `gnd_id`, `ts` | `replace_keyword` mit getauschten `old`/`new` |
| `add_keyword` | `keyword`, `gnd_id`, `ts` | `remove_keyword` |
| `remove_keyword` | `keyword`, `ts`, `_prev_gnd_id` (für Undo) | `add_keyword` |
| `replace_classification` | `old`, `new`, `ts` | symmetrisch |
| `set_dk_search_results` | `count`, `_prev` (für Undo, ggf. groß) | `set_dk_search_results` mit `_prev` |
| `pipeline_run_started` | `run_id`, `workflow`, `ts` | (kein Undo) |
| `pipeline_run_done` | `run_id`, `success`, `ts` | (kein Undo) |

**Pflichtfelder**: `op`, `ts` (ISO-8601 UTC). Op-spezifische Felder
darüber hinaus.

### Beispiel
```json
{
  "op": "replace_keyword",
  "old": "Cd",
  "new": "Cadmium",
  "gnd_id": "4007249-3",
  "ts": "2026-05-13T08:42:11Z"
}
```

### Versionierung
Top-Level optionales Feld `v: 1`. Bei Bedarf in WP10 Schema-Migration
(Op-Renames, Feld-Defaults).

## 6. Locking / Race-Conditions

### Problem
Während Pipeline-Worker den State schreibt, könnte Chat-Tool eine
Mutation versuchen → inkonsistenter State (z.B. Chat ersetzt `Cd` durch
`Cadmium`, Worker überschreibt mit neuer Keyword-Liste).

### Lösung — Worker-Priorität
```python
# Pseudo-Code in AlimaStateBus
class AlimaStateBus(QObject):
    ...
    _mutation_lock = threading.Lock()

    def acquire_write(self, holder: str, timeout: float | None = None) -> bool:
        """Acquire mutation lock. Pipeline-Worker uses timeout=None,
        Chat-Tools use timeout=5.0 (short, retry-friendly)."""
        return self._mutation_lock.acquire(timeout=timeout if timeout is not None else -1)

    def release_write(self) -> None:
        self._mutation_lock.release()
```

### Verhalten
- **Pipeline-Worker**: hält Lock pro Step-Mutation (`with bus.write_lock():`).
  Lock wird auch während kurzem Streaming nicht gehalten — nur beim
  finalen Set des Step-Outputs.
- **Chat-Mutation**: `acquire_write(timeout=5.0)`. Bei `False` →
  Toast „Pipeline läuft, bitte warten" in UI; Mutation **wird nicht
  gequeued** automatisch (User entscheidet erneut).
- **Convenience-Manager**: `with bus.write_lock(): state.add_keyword(...)`
  als Context-Manager.

### Edge Cases
- Pipeline-Worker stürzt mid-Step → Lock wird durch `try/finally`-Pattern
  freigegeben.
- Chat-Tool im selben Thread wie GUI → kein echter Race, aber Lock-
  Akquise dient Synchronisations-Disziplin.

## 7. Undo / Redo

### Stack-Modell
```python
# Pseudo-Code in AlimaStateBus
class AlimaStateBus(QObject):
    ...
    _undo_stack: List[Dict[str, Any]] = []
    _redo_stack: List[Dict[str, Any]] = []
    UNDO_LIMIT = 100

    def push_diff(self, diff: Dict[str, Any]) -> None:
        self._undo_stack.append(diff)
        if len(self._undo_stack) > self.UNDO_LIMIT:
            self._undo_stack.pop(0)
        self._redo_stack.clear()   # neue Mutation invalidiert redo

    def undo(self, state: "KeywordAnalysisState") -> bool:
        if not self._undo_stack:
            return False
        diff = self._undo_stack.pop()
        reverse = self._reverse_diff(diff)
        state._apply_diff(reverse)
        self._redo_stack.append(diff)
        return True

    def redo(self, state: "KeywordAnalysisState") -> bool:
        if not self._redo_stack:
            return False
        diff = self._redo_stack.pop()
        state._apply_diff(diff)
        self._undo_stack.append(diff)
        return True
```

### Reset bei neuem Pipeline-Run
- `pipeline_run_started` → `_undo_stack.clear()`, `_redo_stack.clear()`.
- Begründung: Diffs auf altem State sind nach neuem Run nicht
  reversibel.

### Limit
`UNDO_LIMIT = 100`. Bei Überlauf älteste Diffs droppen (FIFO).

### UI-Wiring (WP10)
- Keybindings `Ctrl+Z` / `Ctrl+Shift+Z` in `MainWindow` (heute keine).
- Toast bei `undo()`/`redo()` → false: „Nichts zu undo".

## 8. Persistenz (Audit-Log)

### Optional, nicht Pflicht für T2
Diff-Stream als JSON-Anhang neben Pipeline-Result. Vorbild:
`analysis_export_*.json`-Pattern im Repo-Root.

### Format
```json
{
  "run_id": "2026-05-13T08:30:00Z",
  "workflow": "alima_classic.yaml",
  "diffs": [
    {"op": "pipeline_run_started", "ts": "...", "run_id": "..."},
    {"op": "replace_keyword", "old": "Cd", "new": "Cadmium", "gnd_id": "...", "ts": "..."},
    ...
    {"op": "pipeline_run_done", "ts": "...", "success": true}
  ]
}
```

### Speicherort
Pro Pipeline-Run eine Datei `audit_<run_id>.json` neben dem
`analysis_export_*.json`. Bei Bedarf via Config-Flag
`audit.enabled: false` ausschalten.

### Replay (WP10-Optional)
`AuditReplayer.replay(file, fresh_state)` wendet Diff-Stream auf
leeren State an → reproduziert den Mutations-Endzustand. Nützlich für
Forschungspfad-Validierung (WP2-Reproduzierbarkeit auf
Mutations-Ebene).

## 9. Decision-Point T2

**T2 = „EventBus-Pattern fix"**.

### Empfehlung
`AlimaStateBus`-Singleton (QObject + pyqtSignal) für GUI, plus
Session-scoped Plain-Python-Instanz für Webapp (`asyncio.Queue`-basiert).

### Begründung
- pyqtSignal ist bereits Standard im Bestand
  ([`search_engine.py`](../src/core/search_engine.py),
   [`provider_status_service.py`](../src/llm/provider_status_service.py)).
- Thread-Cross-Dispatch ist automatisch (Qt-Slot-Queue).
- Webapp-Sessions sind isoliert → kein App-globales Singleton im
  Webapp-Backend. Eigene Instanz pro Session-Init.
- Keine zusätzliche Dependency.

### Alternative
- **Pure-Python-Event-Bus** (z.B. `blinker`-Lib oder eigene Class).
  Kontra: extra Dependency, kein automatic-Qt-Thread-Dispatch, GUI muss
  Marshalling selbst bauen.

→ pyqtSignal-Wrapper gewinnt.

## 10. Risiken + offene Validierungen

| Risiko | Bewertung | Mitigation |
|---|---|---|
| Webapp-Session-Isolation | mittel | Session-scoped Instanz, kein App-Singleton im Webapp-Backend. Klar in Architektur dokumentiert. |
| Race-Condition Pipeline ↔ Chat-Mutation | mittel | `threading.Lock` (Sek 6), Worker hat Vorrang. |
| Diff-Format-Versionierung | niedrig | `v:`-Feld optional, Migration bei Bedarf. |
| Undo nach Pipeline-Run gefährlich | mittel | Stack-Clear auf `pipeline_run_started`. |
| Mutations-API mid-stream (LLM streamt `final_llm_analysis`) | niedrig | Mutations-API wird **nur** nach Step-`done` aufgerufen; Streaming-Update bleibt direct-write. |
| Singleton-Anti-Pattern für Tests | niedrig | `_reset_for_tests()` analog `registry.py:131`. |
| Diff-Audit-Log groß bei vielen DK-Results | niedrig | `_prev`-Snapshot in `set_dk_search_results` optional weglassen wenn Persistenz off. |

### Out-of-Scope für T2
- Webapp-Backend-Konkret-Implementation (SSE vs WebSocket): WP9 + WP10.
- Diff-Replay-Tool: WP10.
- Audit-Log-Viewer-UI: spätere WP.

## 11. Cross-References / Folge-WPs

| Folge-WP | Konsumiert aus WP6 |
|---|---|
| **WP7** (Chat-Tools workflow-aware) | Mutations-API für Schreib-Tools (`propose_keyword_replacement` etc.). `last_shared_context`-Retention für Read-Only-Zugriff auf agentic-Run-Interna. |
| **WP8** (Chat-UI / Modals) | EventBus-Subscribe für Modal-Updates; Diff-Toast-UI bei Mutations. |
| **WP10** (Migration) | Phasen-Migration der `PipelineManager`-Mutations auf API; Audit-Log-Implementation; Undo-UI-Wiring. |
| **WP4** (Renderer-Registry) | Indirekt: Renderer können auf `state_changed`-Signal abonnieren um sich live zu updaten (statt manueller `populate_*`-Aufrufe). |

## 12. Status

✅ WP6 abgeschlossen.
- State-Master-Modell (Sek 2) inkl. `last_shared_context`-Retention.
- EventBus-API `AlimaStateBus` (Sek 3) — pyqtSignal-Wrapper.
- Mutations-API auf `KeywordAnalysisState` (Sek 4) — 5 Methoden.
- Diff-Format JSON (Sek 5) — 7 Ops.
- Locking-Pattern (Sek 6) — Worker-Priorität.
- Undo/Redo (Sek 7) — Diff-Stack mit Reset.
- Persistenz (Sek 8) — Audit-Log optional.
- T2-Decision (Sek 9).
- Risiken (Sek 10).
- Folge-WP-Mapping (Sek 11).
