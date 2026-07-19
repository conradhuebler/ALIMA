# Chat-Tools workflow-aware Design (WP7)

**Status**: Faktenbasis-Dokument für T3-Decision **„Tool-Set-Design"**.
Output von WP7 aus [`wp_detailed_plans.md`](wp_detailed_plans.md).
**Pseudo-Code only — keine Implementation.**

**Verhältnis zu [`agentic_chat_plan.md`](agentic_chat_plan.md)**:
- `agentic_chat_plan.md` = MVP-Phasen-Plan (6.5 PT, Phasen 1-7),
  Implementations-Roadmap.
- **WP7 (dieses Doc) = Architektur-Skizze**: Tool-Klassen-Vertrag,
  Discovery-Mechanik, Provider-Strategie, Anti-Halluzination-Pattern.
- WP7 verweist auf chat-plan-Phasen, dupliziert sie **nicht**.

**Methode**: Code-Inspektion (AgentLoop, ToolRegistry, ChatWidget,
ChatWorker), Cross-Read agentic_chat_plan.md MVP-Phasen, WP3 Slot-
Vokabular für Discovery-Mechanik.

**Querverweise**:
- [`workflow_output_schemas.md`](../workflow_output_schemas.md) (WP3) —
  Slot-Vokabular für Tool-Discovery.
- [`state_sync_design.md`](../state_sync_design.md) (WP6) — Mutations-API
  + EventBus für Schreib-Tools.
- [`provider_portability_design.md`](provider_portability_design.md)
  (WP11) — Chat-Provider-Default.
- [`agentic_chat_plan.md`](agentic_chat_plan.md) — MVP-Phasen 1-7.
- [`audit_findings.md`](../audit_findings.md) — Audit-Finding S5
  (Read-Only-Modus), Finding 14 (AgentLoop multi-turn).
- [`wp_detailed_plans.md`](wp_detailed_plans.md) WP7 — Soll-Definition.

## 1. Executive Summary

- **Foundation da**: `AgentLoop`
  ([`src/core/agent_loop.py:18-244`](../src/core/agent_loop.py)) ist
  provider-agnostisch und multi-turn-fähig (max 20 iter, diminishing-
  returns detect). `ToolRegistry`
  ([`src/mcp/tool_registry.py:19-500`](../src/mcp/tool_registry.py))
  hat 16-17 MCP-Tools mit class-based-Registration.
- **Lücke**: Heutiges `ChatWorker`
  ([`src/ui/chat_worker.py:13-92`](../src/ui/chat_worker.py)) =
  single-shot, **kein Tool-Use**. Muss durch `ChatAgentWorker`
  (AgentLoop-Wrapper) ersetzt werden (chat-plan Phase 3).
- **WP7-Beitrag**: 4-Layer Tool-Architektur — `BaseChatTool` (neu),
  Generic-Tools (4), ALIMA-Tools (7+), MCP-Tools (16). Discovery-
  Filter via `available_for(workflow_name)` + WP3-Slot-Match.
- **Schreib-Tools** emittieren über `AlimaStateBus` (WP6), keine
  direct-mutation.
- **Anti-Halluzination**: `validate_gnd_term`-Mandatory-Tool —
  existiert heute nicht, wird in WP10/chat-plan-Phase-1 gebaut.
- **Chat-Provider** unabhängig vom Pipeline-Provider, eigener Default
  in `config.json` (heute keine `chat.*`-Sektion).
- **T3-Decision**: Generic + spezialisiert (4-Layer), Code-Registriert
  (kein YAML-deklariertes Tool-Set für MVP).

## 2. Tool-Klassen-Hierarchie

### Vertrag `BaseChatTool`
Pseudo-Code für `src/ui/chat_tools/base.py`:

```python
# Pseudo-Code, nicht zur Implementation
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseChatTool(ABC):
    """Session-scoped chat tool. Mirror to ToolDefinition but
    session-aware (no app-global registry)."""

    #: JSON-Schema-tool-name (matcht LLM-Tool-Calling-Convention).
    name: str = ""

    #: 1-Zeilen-Beschreibung für LLM (zeigt sich in Tool-List).
    description: str = ""

    #: JSON-Schema für Parameter (matcht function-calling).
    parameters_schema: Dict[str, Any] = {"type": "object", "properties": {}}

    #: Klasse-Tag, dient Discovery (siehe Sek 4).
    category: str = "generic"  # "generic" | "alima" | "mcp"

    def available_for(self, workflow_name: Optional[str],
                      context: "ChatSession") -> bool:
        """Return True if tool is applicable for the given workflow.

        Default: always available. Subclasses override for workflow-
        specific filtering (z.B. dk_*-Tools nur bei Workflows mit
        slot:classification_list).
        """
        return True

    @abstractmethod
    def execute(self, session: "ChatSession", **args) -> Dict[str, Any]:
        """Run the tool. Returns a JSON-serializable dict that gets
        passed back to the LLM as the tool result.

        Raises:
            ToolError: caller converts to error-result for LLM.
        """
        raise NotImplementedError
```

### Spiegelung `ToolDefinition` aus MCP-Layer
`ToolDefinition` ([`tool_registry.py:35`](../src/mcp/tool_registry.py))
ist app-global, registriert via `ToolRegistry.register(tool_def, handler)`.
`BaseChatTool` ist **session-scoped** — pro Chat-Session eine
Instanz mit gebundener `ChatSession`-Referenz (für Zugriff auf
`last_shared_context`, `current_messages`, …).

### Adapter MCP → Chat
Existing MCP-Tools werden via Adapter eingebunden:

```python
# Pseudo-Code
class MCPToolAdapter(BaseChatTool):
    """Wraps a registered ToolDefinition+handler as BaseChatTool."""
    category = "mcp"

    def __init__(self, tool_def, handler):
        self.name = tool_def.name
        self.description = tool_def.description
        self.parameters_schema = tool_def.parameters_schema
        self._handler = handler

    def execute(self, session, **args):
        return self._handler(**args)  # MCP-Pfad ist session-frei
```

### Registry für Chat-Session
```python
# Pseudo-Code in src/ui/chat_tools/registry.py
CHAT_TOOL_CLASSES: List[Type[BaseChatTool]] = []

def register_chat_tool(cls):
    CHAT_TOOL_CLASSES.append(cls)
    return cls


def build_chat_toolset(workflow_name: Optional[str],
                       session: "ChatSession") -> List[BaseChatTool]:
    """Instantiate tools applicable for the workflow."""
    tools = [cls() for cls in CHAT_TOOL_CLASSES]
    tools.extend(_mcp_adapters_for(session))
    return [t for t in tools if t.available_for(workflow_name, session)]
```

## 3. Generische Tools

Workflow-agnostisch, immer verfügbar (`available_for` returns True).

| Tool | Beschreibung | Output-Form | Konsumiert |
|---|---|---|---|
| `list_available_data` | Schema-Dump des aktuellen `ChatSession`-Context: welche Slots, welche Step-Outputs, welche `extra`-Keys. | `{slots: [...], step_results: {...}, extra: {...}}` | `last_shared_context` |
| `get_extra(path)` | Liest `extra.<path>` aus dem Context. | `{value: any}` | `last_shared_context.extra` |
| `get_step_result(step_id, path=None)` | Liest `step_results[step_id]` (optional sub-path). | `{value: any}` | `last_shared_context.step_results` |
| `get_messages_history()` | Liest die bisherige Chat-Konversation (für Reflexion / „was hatte ich gefragt"). | `[{role, content}, ...]` | `ChatSession.messages` |

### `list_available_data` als Discovery-Tool
**Wichtigster Tool** — LLM nutzt es als erstes, um zu erfahren welche
Daten existieren. Output-Format-Skizze:

```json
{
  "workflow": "alima_classic",
  "slots_populated": ["slot:gnd_pool", "slot:dk_table",
                      "slot:classification_list", "slot:text_blob"],
  "step_results": {
    "extraction": ["extracted_keywords", "working_title"],
    "search": ["entries"],
    "dk_collect": ["dk_entries", "formatted_prompt"]
  },
  "extra_keys": ["final_keywords", "dk_entries", "dk_prompt_text"]
}
```

Implementation = einfacher Dump via `SharedContext.to_dict()`-Subset.

## 4. ALIMA-spezifische Tools

Workflow-aware: `available_for` filtert nach WP3-Slot-Präsenz.

| Tool | `available_for` (Slot-Gate) | Output | Konsumiert |
|---|---|---|---|
| `get_keywords(kind: "initial"|"selected"|"final")` | `slot:keyword_list` vorhanden | `{keywords: [...]}` | typed-fields |
| `get_keyword_chains()` | `slot:keyword_chains` vorhanden | `{chains: [{chain, reason}]}` | `keyword_chains` |
| `get_dk_classifications()` | `slot:classification_list` vorhanden | `{classifications: [{code, type}]}` | `dk_classifications` |
| `get_dk_titles_for_code(dk_code)` | `slot:dk_table` vorhanden | `{titles: [...], count: int}` | `dk_search_results` |
| `get_chunk_response(chunk_idx)` | Workflow nutzt `chunking:` (z.B. alima_classic) | `{response: str, keywords: [...]}` | `step_results.selection_chunks` |
| `find_chunk_for_keyword(keyword)` | wie oben | `{chunk_idx: int|null, keyword_in_chunk: bool}` | dito |
| `search_in_gnd_pool(query)` | `slot:gnd_pool` vorhanden | `{matches: [{gnd_id, title, ...}]}` | `gnd_entries` |
| `validate_gnd_term(term)` | immer (MCP-search nötig) | `{verified: bool, gnd_id: str|null, match_type: "exact"|"fuzzy"|"none"}` | calls `search_gnd` exact-match |

### Slot-Discovery-Logik
`available_for` ruft `_workflow_has_slot(workflow_name, slot)`:

```python
# Pseudo-Code
def _workflow_has_slot(workflow_name: str, slot: str) -> bool:
    """Check WP3 slot-mapping: does the workflow's output schema
    declare this slot?"""
    wf = WORKFLOW_LOADER.load(workflow_name)
    declared_slots = _resolve_workflow_slots(wf)  # WP4 slot-resolver
    return slot in declared_slots
```

Slot-Auflösung kommt aus WP4 (`docs/renderer_registry_design.md`
Sek 4) — 4-stufiger Algorithmus mit YAML-Hint, TOOL_FN_SLOT_HINT,
Feldname-Heuristik, raw_json-Fallback.

### Mapping Tool → WP3-Slot (Konsistenz-Check)
| Tool | Slot |
|---|---|
| `get_keywords` | `slot:keyword_list` |
| `get_keyword_chains` | `slot:keyword_chains` |
| `get_dk_classifications` | `slot:classification_list` |
| `get_dk_titles_for_code` | `slot:dk_table` |
| `search_in_gnd_pool` | `slot:gnd_pool` |
| `validate_gnd_term` | — (Read-Only-MCP, immer aktiv) |

## 5. MCP-Tool-Integration

### Bestehende 16-17 Tools
Per WP3 Sek 5.1 + [`tool_registry.py:439-459`](../src/mcp/tool_registry.py):

| Familie | Tools | Read/Write |
|---|---|---|
| `knowledge-read` | `search_gnd`, `get_gnd_entry`, `get_gnd_batch`, `get_search_cache`, `get_dk_cache`, `get_classification`, `get_db_stats` | R |
| `knowledge-write` | `store_search_result` | W |
| `library-search` | `search_lobid`, `search_swb`, `search_catalog`, `search_catalog_titles`, `resolve_doi` | R |
| `pipeline-result-read` | `list_pipeline_results`, `get_pipeline_result`, `get_pipeline_keywords`, `get_pipeline_abstract` | R |

### Read-Only-Modus (Audit-Finding S5)
Neues Config-Flag `chat.no_cache_writes: true` schaltet:
- `search_gnd` ruft `KnowledgeManager.search(..., persist=False)` →
  Result NICHT in DB persistiert.
- `store_search_result` aus Tool-Liste entfernt.
- `knowledge-write`-Familie ganz ausgeschlossen.

Pseudo-Code Filter:
```python
def _mcp_adapters_for(session) -> List[BaseChatTool]:
    cfg = config.get("chat", {})
    no_write = cfg.get("no_cache_writes", False)
    adapters = []
    for tool_def, handler in TOOL_REGISTRY.iter_all():
        if no_write and tool_def.category == "knowledge-write":
            continue
        adapters.append(MCPToolAdapter(tool_def, handler))
    return adapters
```

### Tool-Filter pro Workflow
`pipeline-result-read`-Familie ist Chat-only (WP3 Sek 5.1 verifiziert:
nicht in den 6 Workflows). Tools werden für **Cross-Run-Queries**
exponiert („Was war in der letzten Pipeline-Analyse?"). Filter via
`available_for` zurückgewiesen wenn Workflow `pipeline_result_read`-
Pfad nicht nutzt.

## 6. Anti-Halluzination

### Problem
LLM könnte Keywords / GND-IDs erfinden ("Cadmium oxidativ", GND-ID
"5555-X" — beide ggf. nicht echte GND-Einträge). Schreib-Tools
würden dann ungültige Daten produzieren.

### Lösung: `validate_gnd_term` als Mandatory-Pre-Tool
Tool-Beschreibung enthält LLM-Anweisung:
```
You MUST call validate_gnd_term(term) before suggesting a keyword
replacement via propose_keyword_replacement. Use the verified=true
result; if verified=false, do not suggest the term.
```

Implementation (heute **nicht** im Code, geplant für WP10/chat-plan
Phase 1):

```python
@register_chat_tool
class ValidateGndTerm(BaseChatTool):
    name = "validate_gnd_term"
    description = "Check if a term has an exact GND match. Returns the canonical GND-ID if so."
    parameters_schema = {
        "type": "object",
        "properties": {"term": {"type": "string"}},
        "required": ["term"],
    }
    category = "alima"

    def execute(self, session, *, term):
        # Reuse MCP search_gnd
        results = session.tool_registry.invoke("search_gnd", query=term, limit=5)
        for r in results.get("hits", []):
            if r.get("title", "").lower() == term.lower():
                return {"verified": True, "gnd_id": r["gnd_id"],
                        "match_type": "exact"}
        if results.get("hits"):
            return {"verified": False, "gnd_id": None,
                    "match_type": "fuzzy",
                    "near_matches": results["hits"][:3]}
        return {"verified": False, "gnd_id": None,
                "match_type": "none"}
```

### Restrisiko
- LLM ignoriert Tool-Description-Anweisung (Compliance-abhängig).
  Mitigation: in `propose_keyword_replacement`-Tool-Signatur `gnd_id`-
  Parameter als required → Schreib-Tool wirft Fehler wenn ohne ID
  aufgerufen.
- Tool-Description ist nicht zwingend (LLM nicht durchsetzbar).
- Mittelfristig: Hard-Validation im Schreib-Tool selbst (siehe Sek 7).

## 7. Schreib-Tools

### Tool-Set
| Tool | Effekt | Confirmation |
|---|---|---|
| `propose_keyword_replacement(old, new, gnd_id)` | Emit `AlimaStateBus.emit_event("state_changed", {op: "replace_keyword", ...})`. UI zeigt Mutations-Modal (WP8). | User-Confirmation **pflicht** |
| `propose_step_rerun(step_id, modified_inputs?)` | Emit Event → UI öffnet `SingleStepDialog` (WP5) pre-populated. | User-Confirmation pflicht |
| `propose_keyword_addition(keyword, gnd_id)` | wie replace | pflicht |
| `propose_keyword_removal(keyword)` | wie replace | pflicht |

### Architektur — Emit statt Mutation
Schreib-Tool **mutiert NICHT direkt**. Es emittiert über `AlimaStateBus`
(WP6 Sek 3-4) ein `proposal_<op>`-Event mit Payload. UI-Subscriber
(Mutations-Modal in WP8) zeigt User-Confirmation; User → confirm →
Mutations-API aus WP6 wird gerufen (`state.apply_keyword_replacement(...)`).

```python
# Pseudo-Code für propose_keyword_replacement
@register_chat_tool
class ProposeKeywordReplacement(BaseChatTool):
    name = "propose_keyword_replacement"
    description = "Propose replacing keyword 'old' with 'new'. Requires verified GND-ID. User must confirm."
    parameters_schema = {
        "type": "object",
        "properties": {
            "old": {"type": "string"},
            "new": {"type": "string"},
            "gnd_id": {"type": "string"}
        },
        "required": ["old", "new", "gnd_id"],
    }
    category = "alima"

    def execute(self, session, *, old, new, gnd_id):
        # Hard-validation: gnd_id muss existieren (auch wenn LLM
        # validate_gnd_term übersprungen hat)
        if not session.tool_registry.invoke("get_gnd_entry", gnd_id=gnd_id):
            raise ToolError(f"GND-ID {gnd_id} not found. Use validate_gnd_term first.")

        bus = AlimaStateBus()
        proposal_id = _new_proposal_id()
        bus.emit_event("proposal_replace_keyword", {
            "proposal_id": proposal_id,
            "op": "replace_keyword",
            "old": old, "new": new, "gnd_id": gnd_id,
            "ts": _now(),
        })
        return {"proposal_id": proposal_id, "status": "pending_user_confirmation"}
```

### Confirmation-Flow
1. LLM ruft `propose_keyword_replacement(...)`
2. Tool emittiert Event mit `proposal_id`, returnt `pending_user_confirmation`
3. WP8-Mutations-Modal subscribed `proposal_*`-Events → zeigt Dialog
4. User confirm → Modal ruft `state.apply_keyword_replacement(...)` (WP6)
5. State emittiert `state_changed`-Event → andere UI-Komponenten updaten

Detail-UI in WP8.

## 8. Provider-Strategie

### Chat-Provider unabhängig von Pipeline
- Pipeline-Provider: per-Workflow-YAML oder GUI-Override.
- Chat-Provider: eigenes Config-Feld, **separat einstellbar**.

### Neue `config.json`-Sektion
```json
{
  "chat": {
    "default_provider": "ollama",
    "default_model": "llama3.1:8b",
    "max_iterations": 20,
    "no_cache_writes": true
  }
}
```

### Default-Empfehlung (aus WP11 Sek 11)
- **Lokal**: Ollama `llama3.1:8b` — 0-Cost, privacy-friendly, latenz-
  abhängig vom Host.
- **Cloud-mini**: `gpt-4o-mini` — schneller, kostenpflichtig.

Operator-Frage F4 in WP11 entscheidet — WP7 nimmt Ollama-Default
(reversibel).

### Capability-Konsumtion
- Chat-Provider muss `tool_use ∈ {native, parallel_native}` haben
  (sonst kein Tool-Calling). Check via Capability-YAML (WP11 Sek 3).
- Wenn nicht: Fallback auf Text-Tool-Use (LLM streamt JSON-Tool-Call,
  Parser extrahiert) — heute in
  [`llm_service.py:2868`](../src/llm/llm_service.py) implementiert.

### UI-Combo
`ChatWidget`-Header behält Provider-Combo (B8-Pattern). Default aus
`chat.default_*`, User kann overriden pro Session.

## 9. Multi-Turn-Integration mit AgentLoop

### Heutige Lage
- `AgentLoop` ist multi-turn (`src/core/agent_loop.py:18-244`,
  max 20 iter), provider-agnostisch.
- `ChatWorker` (`src/ui/chat_worker.py:13-92`) ruft
  `llm_service.generate_response()` single-shot, **nicht** AgentLoop.

### Plan: `ChatAgentWorker` (chat-plan Phase 3)
Rewrite `ChatWorker`:

```python
# Pseudo-Code
class ChatAgentWorker(QThread):
    token_received = pyqtSignal(str)
    tool_called = pyqtSignal(dict)
    tool_result = pyqtSignal(dict)
    iteration_done = pyqtSignal(int)
    finished_with_response = pyqtSignal(str)

    def __init__(self, session, user_message, tools):
        super().__init__()
        self.session = session
        self.user_message = user_message
        self.tools = tools

    def run(self):
        loop = AgentLoop(
            llm_service=self.session.llm_service,
            tool_registry=self.session.tool_registry,
            max_iterations=20,
            stream_callback=self.token_received.emit,
            tool_call_callback=self.tool_called.emit,
            tool_result_callback=self.tool_result.emit,
        )
        self.session.messages.append({"role": "user",
                                      "content": self.user_message})
        result = loop.run(
            messages=self.session.messages,
            tools_schema=[t.to_schema() for t in self.tools],
        )
        self.session.messages.append({"role": "assistant",
                                      "content": result.final_response})
        self.finished_with_response.emit(result.final_response)
```

### Aggregations-Pfad
- **Path A** (chat-plan default): Tool-Calls werden im UI als
  collapsible-Block dargestellt, AgentLoop läuft autonom durch.
- **Path B** (chat-plan Phase 8, optional): User sieht jeden Tool-Call
  + kann zwischen Turns Korrekturen einfügen.

WP7 empfiehlt **Path A** für MVP (einfacher, schneller in der UX).

### Diminishing-Returns-Detect
AgentLoop hat schon (`repeat_threshold=3` in
[`agent_loop.py:33`](../src/core/agent_loop.py)). Wenn LLM 3× hintereinander
denselben Tool-Call mit gleichen Args macht → Loop bricht ab. Schutz
vor Infinite-Loops.

## 10. Decision-Point T3

**T3 = „Generisch vs spezialisiert + Tool-Set-Erweiterungs-Mechanik"**.

### Empfehlung
- **Layered**: Generic (4 Tools) + ALIMA-Specific (7+) + MCP-Adapter
  (16) — 27+ Tools insgesamt, davon `available_for`-filtered.
- **Code-registriert** für MVP (`@register_chat_tool`-Decorator),
  YAML-deklariert wäre OOS.
- **Eigene Mechanik** (CHAT_TOOL_CLASSES-Liste) statt Spiegelung in
  TOOL_REGISTRY, weil session-scoped vs app-global. MCP-Tools werden
  via Adapter eingebunden.

### Begründung
- Generic-only zu unspezifisch (LLM hat zu viel Freiheit, raten Slot-
  Pfade).
- Specialised-only inkompatibel mit nicht-ALIMA-Workflows (heute keine,
  aber Architektur soll wachsen).
- Layered erlaubt graduellen Ausbau pro Workflow.

## 11. Risiken + offene Validierungen

| Risiko | Bewertung | Mitigation |
|---|---|---|
| Tool-Inflation (LLM verwirrt) | mittel | `available_for`-Filter pro Workflow; default-Filter MCP read-only-Mode. |
| Anti-Halluzination Compliance | mittel | Hard-Validation in Schreib-Tools (Sek 6/7), nicht nur Tool-Description. |
| AgentLoop max_iterations=20 zu klein für komplexe Tasks | niedrig | Config `chat.max_iterations` macht es einstellbar. |
| Schreib-Tools confirmen ohne User-Aktion (UI nicht subscribed) | hoch | Timeout im Tool-Aufruf: returnt nach 60s mit `status=timed_out`. WP8-UI muss garantieren-subscribed. |
| `pipeline-result-read`-Family exposed in falscher Session | niedrig | Filter via `available_for` strikt. |
| MCP-Adapter überträgt `session` nicht (handler stateless) | niedrig | MCP-Handler sind schon stateless (KnowledgeManager + SuggesterFactory App-singletons). |
| Path-A AgentLoop blockiert UI thread | niedrig | `ChatAgentWorker` ist QThread; Cross-Thread-Signal-Dispatch via Qt. |

## 12. Cross-References / Folge-WPs

| Folge-WP | Konsumiert aus WP7 |
|---|---|
| **WP8** (Chat-UI) | Tool-Call-Rendering (Sek 9 Path A), Mutations-Modal (Sek 7), `ChatAgentWorker` (Sek 9). |
| **WP10** (Migration) | Implementierungs-Phasen (verweist auf `agentic_chat_plan.md` 1-7). |
| **WP6** (State-Sync) | `AlimaStateBus.emit_event` für Schreib-Tools (Sek 7). |
| **WP11** (Provider-Portabilität) | `chat.default_*`-Config + Capability-Check (Sek 8). |

## 13. Operator-Fragen (max 6)

### F1: Path A oder Path B als MVP?
**Empfehlung**: Path A. Schneller in der UX, weniger State-Maschine in
der UI. Path B als spätere Erweiterung.

### F2: `pipeline-result-read`-Tools immer aktiv oder nur in „Cross-Run-Modus"?
**Empfehlung**: immer aktiv. Operator-Use-Case („was war in letzter
Analyse") rechtfertigt das. Filter via `available_for` zurückweisen
wenn nicht gewünscht.

### F3: Schreib-Tools per Default an oder hinter Feature-Flag `chat.allow_writes`?
**Empfehlung**: hinter Feature-Flag. Default off (Read-Only-Chat).
User aktiviert bewusst. Reduziert Forschungspfad-Risiko.

### F4: Validate-GND auch für `propose_keyword_addition` Pflicht (nicht nur für replacement)?
**Empfehlung**: ja. Add ohne validierte GND-ID nutzlos.

### F5: Tool-Discovery via WP3-Slot oder via Workflow-Name?
**Empfehlung**: Slot. Workflow-Name als zusätzlicher Filter wenn nötig.
Slot-Match ist die generischere Lösung — neue Workflows funktionieren
ohne Tool-Whitelist-Update.

### F6: `chat.default_provider` — Ollama-lokal oder Cloud-mini?
**Empfehlung**: Ollama `llama3.1:8b`. Per Config-Override änderbar.
Identisch zu WP11 Empfehlung.

## 14. Status

✅ WP7 abgeschlossen.
- Tool-Klassen-Hierarchie (Sek 2)
- Generische Tools (Sek 3)
- ALIMA-spezifische Tools (Sek 4)
- MCP-Tool-Integration + Read-Only-Modus (Sek 5)
- Anti-Halluzination (Sek 6)
- Schreib-Tools (Sek 7)
- Provider-Strategie (Sek 8)
- Multi-Turn-Integration (Sek 9)
- T3-Decision (Sek 10)
- Risiken (Sek 11)
- Folge-WP-Mapping (Sek 12)

Pendend: Operator-Antworten F1-F6 + Implementierungs-Pfad
([`agentic_chat_plan.md`](agentic_chat_plan.md) MVP-Phasen 1-7,
6.5 PT) als WP10-Eingang.
