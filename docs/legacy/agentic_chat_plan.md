# Agentic Chat over Pipeline Results — Architecture Plan

**Status**: Design (not yet implemented).
**Scope**: Replace static-context `ChatWidget` with a tool-using agent that
introspects the active session state and fetches data on demand.

## Goal

After the ALIMA pipeline runs, the user chats about the result. The chat
agent **knows what data exists**, **decides which parts to load**, and
**fetches them via tools** — instead of the entire result being stuffed
into every prompt.

## Current state (what to replace)

- `src/ui/chat_widget.py` — UI dock, single-shot LLM via `ChatWorker`.
- `src/ui/chat_worker.py` — wraps `LlmService.generate_response()`, no
  tool-use.
- Context loading: `ChatWidget.load_context(analysis_state)` flattens
  `KeywordAnalysisState` into one big string (`self.current_context`),
  truncates abstract to 500 chars, drops chunk-level data, drops DK
  catalog titles.

**Limits of the static approach**:
- Token cost grows with every turn.
- Big fields (DK catalog titles, chunk responses, full GND entries) get
  dropped → user can't ask about them.
- LLM can't follow up: "show me the chunk where keyword X was filtered out".

## Target architecture

```
User message
    │
    ▼
ChatWidget
    │
    ▼
AgentLoop  ◄────── tool_registry (session-scoped)
    │                  │
    │                  ├── list_available_data
    │                  ├── get_abstract
    │                  ├── get_keywords            (initial / extracted / final)
    │                  ├── get_keyword_chains
    │                  ├── get_missing_concepts
    │                  ├── get_dk_classifications
    │                  ├── get_dk_titles_for_code
    │                  ├── get_chunk_response
    │                  ├── get_step_result
    │                  ├── search_gnd              (existing MCP)
    │                  ├── get_gnd_entry           (existing MCP)
    │                  └── search_lobid / swb / catalog (existing MCP)
    ▼
Streaming response back to UI
```

## Components

### 1. Session state holder

```python
# src/ui/chat_session.py  (new)
@dataclass
class ChatSession:
    analysis_state: Optional[KeywordAnalysisState] = None
    shared_context: Optional[SharedContext] = None  # raw v4 ctx if available
    messages: List[Dict[str, str]] = field(default_factory=list)
    system_prompt: str = ""

    def update_from_pipeline(self, state, ctx=None):
        self.analysis_state = state
        self.shared_context = ctx
        # Don't clear messages — chat survives state updates if user wants
```

The session owns both the user-facing `KeywordAnalysisState` and (when
the agentic v4 pipeline ran) the underlying `SharedContext` — chunked
intermediate results live there, not in `KeywordAnalysisState`.

### 2. Session-scoped tool functions

New file `src/ui/chat_session_tools.py`. Each function closes over a
`ChatSession` and returns JSON-serialisable dicts. Registered with
`register_tool_fn` so they are reachable from any agent loop.

Sketch:

```python
def make_chat_tools(session: ChatSession) -> Dict[str, Callable]:
    state = session.analysis_state
    ctx   = session.shared_context

    def list_available_data():
        """Return the data inventory as a schema."""
        return {
            "abstract": {"present": bool(state.original_abstract),
                         "length": len(state.original_abstract or "")},
            "working_title": state.working_title,
            "initial_keywords":  {"count": len(state.initial_keywords)},
            "extracted_keywords":{"count": len(_extracted(state))},
            "final_keywords":    {"count": len(_final(state))},
            "keyword_chains":    {"count": len(_chains(state))},
            "missing_concepts":  {"count": len(_missing(state))},
            "dk_classifications":{"count": len(state.dk_classifications)},
            "dk_search_results": {
                "keywords_searched": len(state.dk_search_results),
                "unique_codes":      len(state.dk_search_results_flattened),
            },
            "chunks": {"present": _has_chunks(ctx),
                       "count":   _chunk_count(ctx)},
            "step_results": list((ctx and ctx.step_results or {}).keys()),
        }

    def get_abstract(): return {"abstract": state.original_abstract}
    def get_keywords(kind="final"):
        # kind ∈ {"initial", "extracted", "final"}
        ...
    def get_keyword_chains(): ...
    def get_missing_concepts(): ...
    def get_dk_classifications(include_analyse=False): ...
    def get_dk_titles_for_code(code: str, max_titles: int = 20):
        # Walks state.dk_search_results to find which catalog hits had `code`
        ...
    def get_chunk_response(chunk_index: int):
        # Reads ctx.step_results["selection_chunks"]["chunks"][chunk_index]
        ...
    def get_step_result(step_id: str, path: Optional[str] = None):
        # Generic accessor into ctx.step_results — uses context_path resolver
        ...
    def get_search_hits_for_keyword(keyword: str):
        # Looks up which GND entries were returned for a search term
        ...

    return {
        "list_available_data":      list_available_data,
        "get_abstract":             get_abstract,
        "get_keywords":             get_keywords,
        "get_keyword_chains":       get_keyword_chains,
        "get_missing_concepts":     get_missing_concepts,
        "get_dk_classifications":   get_dk_classifications,
        "get_dk_titles_for_code":   get_dk_titles_for_code,
        "get_chunk_response":       get_chunk_response,
        "get_step_result":          get_step_result,
        "get_search_hits_for_keyword": get_search_hits_for_keyword,
    }
```

### 3. ToolRegistry adapter

`AgentLoop` expects a `ToolRegistry` (MCP-style). Build a per-session
adapter that:

- Wraps `make_chat_tools(session)` into `ToolDefinition` + handler.
- Inherits all existing MCP tools (`search_gnd`, `get_gnd_entry`,
  `search_lobid`, …) for live lookups.
- Schema descriptions in the `ToolDefinition` matter — those are what
  the LLM reads to decide what to call.

```python
def build_chat_registry(
    session: ChatSession,
    base_registry: ToolRegistry,
) -> ToolRegistry:
    reg = base_registry.clone_or_view()
    for name, fn in make_chat_tools(session).items():
        reg.register(_chat_tool_def(name), _wrap(fn))
    return reg
```

### 4. Replace `ChatWorker` with `ChatAgentWorker`

```python
# src/ui/chat_worker.py  (rewrite)
class ChatAgentWorker(QThread):
    token_received = pyqtSignal(str)
    tool_invoked   = pyqtSignal(str, dict)   # tool_name, args
    finished_ok    = pyqtSignal(str)         # final assistant text
    failed         = pyqtSignal(str)

    def __init__(self, llm_service, tool_registry, system_prompt,
                 messages, provider, model, **llm_params):
        ...

    def run(self):
        loop = AgentLoop(
            llm_service=self.llm_service,
            tool_registry=self.tool_registry,
            stream_callback=self.token_received.emit,
        )
        # Convert chat history into a single user_prompt or use
        # messages directly if AgentLoop supports it.
        result = loop.run(
            system_prompt=self.system_prompt,
            user_prompt=self._render_history(),
            tools=[],   # empty list = all registered → both chat tools + MCP
            provider=self.provider,
            model=self.model,
            ...
        )
        self.finished_ok.emit(result.content)
```

`AgentLoop.run` already streams tokens via `stream_callback` and logs
tool calls in `result.tool_log` — surface tool calls in the chat UI as
`🔧 list_available_data(...)` lines so the user sees what the agent is
doing.

### 5. System prompt

Replace `ChatWidget.DEFAULT_SYSTEM_PROMPT` with a tool-aware version:

```text
Du bist Chat-Agent für die ALIMA-Pipeline-Ergebnisse.

Du hast Tools, um auf den aktuellen Pipeline-Lauf zuzugreifen. Lade
Daten NUR bei Bedarf — wirf nicht alles auf einmal.

Empfohlener Ablauf:
  1. Bei der ersten Frage: rufe `list_available_data` auf, um zu sehen,
     was vorhanden ist.
  2. Wähle gezielt die Felder, die für die Frage relevant sind.
  3. Bei Detail-Fragen (z. B. "warum DK 543.42?"):
     `get_dk_titles_for_code("DK 543.42")` zeigt die Katalog-Titel.
  4. Bei Chunk-Fragen ("warum wurde X gefiltert?"):
     `get_chunk_response(<idx>)` zeigt LLM-Output pro Chunk.
  5. Wenn Daten fehlen, nutze die Suche-Tools (search_gnd, search_lobid)
     für Live-Lookups.

Sprache: Deutsch. Präzise, fachlich, keine Floskeln.
```

### 6. Multi-turn handling

`AgentLoop.run` is currently single-shot (one user_prompt). Two paths:

- **A** (minimal): Render the chat history into one big `user_prompt`
  with `User: ... / Assistant: ... / User: <new>` and run AgentLoop per
  turn. Simple, works today.
- **B** (cleaner): Extend `AgentLoop` to accept a `messages` list
  directly so it appends turns instead of rebuilding from prompt.
  `LlmService.generate_with_tools()` already takes message lists, so
  this is mostly plumbing.

Recommendation: start with A, migrate to B once stable.

### 7. Wiring

```python
# src/ui/main_window.py — replace existing wiring
self.chat_session = ChatSession()
self.chat_widget = ChatWidget(
    llm_service=self.llm_service,
    chat_session=self.chat_session,
    tool_registry=build_chat_registry(self.chat_session, self.mcp_registry),
    pipeline_manager=self.pipeline_manager,
)
self.pipeline_tab.pipeline_results_ready.connect(
    lambda state: self.chat_session.update_from_pipeline(
        state, ctx=self.pipeline_manager.last_shared_context
    )
)
```

`PipelineManager` needs to expose `last_shared_context` (currently it
discards the v4 SharedContext after building `KeywordAnalysisState`).

## Phasing

| Phase | Scope | Effort |
|-------|-------|--------|
| 1 | `ChatSession` + `make_chat_tools` (5–7 minimal tools) | 1 day |
| 2 | Tool-registry adapter + `ToolDefinition` schemas | 0.5 day |
| 3 | `ChatAgentWorker` with AgentLoop, multi-turn path A | 1 day |
| 4 | `ChatWidget` integration: render tool calls in UI | 0.5 day |
| 5 | New tool-aware system prompt + provider/model defaults | 0.25 day |
| 6 | Expose `last_shared_context` from `PipelineManager` | 0.25 day |
| 7 | Testing + multi-turn path B if needed | 1 day |

**MVP** = phases 1–6 (~3.5 days).

## Use-Case stress-test

User says: "Statt *Freilandökologie* hätte ich gerne *Pflanzenökologie*."
Pipeline ran with 1000 GND hits → chunked to 80 → 20 selected.

Walk-through and what tools the agent must invoke:

```
1. get_keywords("final")              confirm: is "Freilandökologie" final?
2. search_in_gnd_pool("Pflanzen*")    was "Pflanzenökologie" in the 1000?
   ├─ YES, in chunk 7 (not the selected one)
   │   → get_chunk_response(7)        why was it dropped? show LLM reasoning
   └─ NO
       → search_gnd("Pflanzenökologie")    live lookup
       → validate_gnd_term(...)            confirm GND existence (anti-hallucination)
3. propose to user: "Pflanzenökologie (GND-ID 4174277-3) found.
                    Replace Freilandökologie?"
4. user confirms → write-tool path (Phase 2)
5. user: "and re-run DK search"        → rerun_step("dk_collect")
```

### What the original plan misses

Surfaced by the stress-test:

| Gap | Impact | Fix |
|-----|--------|-----|
| No tool to query the *full* GND pool (1000 items) | Agent sees only the 20 final keywords; can't tell user "was available but rejected" | Add `search_in_gnd_pool(pattern)` reading from `ctx.gnd_entries` |
| Per-chunk **input** not stored | Can answer "what was the result for chunk 7" but not "which chunk did keyword X live in" | Extend `_run_chunked` in `llm_agent_step.py` to record `chunk_input` per `per_chunk` entry |
| `last_shared_context` not retained | `gnd_entries` (the 1000) live only in `SharedContext`, not in `KeywordAnalysisState` — without this, chat is blind to 980 of 1000 entries | **Phase 6 is mandatory, not optional** |
| No anti-hallucination check | Agent could invent `GND-ID 9999999` for a fake term | `validate_gnd_term(term)` mandatory before any keyword recommendation |
| Iteration budget too tight | 5+ tool calls per turn × multi-turn → 20-iteration default exhausts | Per-turn reset, raise default to 30, expose as setting |
| Live `search_gnd` slow + cache-mutating | 3-10 s per call; pollutes pool used by next pipeline run | Tag tools as fast/slow; agent prefers fast first; chat searches read-only mode? |
| No write tools | "replace X with Y" / "re-run DK" not possible | Phase 2 (below) |

### Tools to add to MVP (read-side)

Beyond the original list:

- `search_in_gnd_pool(pattern: str, limit: int = 20)` —
  glob/substring search over `ctx.gnd_entries`.
- `find_chunk_for_keyword(keyword: str)` — returns chunk index(es)
  containing the term (requires `chunk_input` storage fix above).
- `get_chunk_input(chunk_index: int)` — the slice fed to the LLM for
  that chunk (provenance).
- `get_filter_reasoning(keyword: str)` — combines `find_chunk_for_keyword`
  + `get_chunk_response` to show why a candidate was kept/dropped.
- `validate_gnd_term(term: str)` — `search_gnd(term)` exact match,
  returns `{verified: bool, gnd_id: str|null}`. Required before the
  agent recommends any keyword to the user.
- `find_related_gnd(keyword_or_gnd_id: str)` — wraps existing
  `extract_gnd_related` for "show me similar terms".

### Code change required outside the chat module

`src/core/agents/steps/llm_agent_step.py:_run_chunked`:

```python
# current
per_chunk.append({"index": idx, "response": parsed})

# needed
per_chunk.append({
    "index": idx,
    "input": chunk,                # <-- NEW: the items fed to LLM
    "input_count": len(chunk),
    "response": parsed,
    "response_count": len(parsed.get(merge_key, [])) if isinstance(parsed, dict) else 0,
})
```

This is a small change but **essential** for chat provenance. Without
it, the agent cannot answer "warum wurde X verworfen?".

## Phase 2 — Write operations (with confirmation)

User confirmed: write tools allowed in MVP, **with mandatory UI
confirmation** before mutation. Adds:

### Mutation tools

- `propose_keyword_replacement(old: str, new: str, gnd_id: str, reason: str)` —
  does NOT write directly. Emits a Qt signal; UI shows a confirmation
  dialog ("Replace *Freilandökologie* with *Pflanzenökologie* (GND-ID
  4174277-3)? [Yes / No / Edit]"). User decision flows back to the agent
  via the next turn.
- `propose_step_rerun(step_id: str, reason: str)` — same pattern; UI
  prompts "Re-run *dk_collect* with current keyword list? [Yes / No]".
  On confirm, `PipelineManager.execute_single_step(step_id)` runs and
  streams output into the chat.

### Why proposals, not direct writes

- Halluzinations-Schutz — confirmation step forces user review of any
  hallucinated GND-ID.
- Undo by default — without confirmation, no undo path; with
  confirmation, "no" is the undo.
- Race-condition avoidance — Pipeline-rerun and chat-mutation can't
  collide because rerun blocks UI.

### Plumbing required

| Component | Change |
|-----------|--------|
| `ChatSession` | Add `pending_proposals: List[Proposal]` queue. |
| `ChatWidget` | New `proposal_received` signal → modal dialog. |
| `PipelineManager` | Expose `execute_single_step(step_id)` reusing existing single-step path; emit progress to chat callback. |
| Agent system prompt | Make "always propose, never auto-execute mutations" a hard rule. |
| `KeywordAnalysisState` | `apply_keyword_replacement(old, new, gnd_id)` — single point of mutation, emits `state_changed` for UI sync. |

### Phasing update

| Phase | Scope | Effort |
|-------|-------|--------|
| 1 | `ChatSession` + read-tools (incl. pool/provenance/validation) | 1.5 d |
| 1.5 | `_run_chunked` provenance fix in `llm_agent_step.py` | 0.5 d |
| 2 | Tool-registry adapter + `ToolDefinition` schemas | 0.5 d |
| 3 | `ChatAgentWorker` with AgentLoop, multi-turn path A | 1 d |
| 4 | `ChatWidget` integration: render tool calls, proposal dialogs | 1 d |
| 5 | New tool-aware system prompt | 0.25 d |
| 6 | Expose `last_shared_context` from `PipelineManager` | 0.25 d |
| 7 | Mutation tools (`propose_keyword_replacement`, `propose_step_rerun`) + confirmation flow | 1.5 d |
| 8 | Testing + multi-turn path B if needed | 1 d |

**MVP** = phases 1–7 (~6.5 days). Phase 8 optional.

## Open questions

1. **Tool-call visibility in UI** — render every tool call as a
   collapsible block? A status line? Hidden by default?
2. **Token budget per chat turn** — cap `max_tokens` in `AgentLoop` or
   let user set per-session?
3. **Provider selection** — chat agent uses a different model than the
   pipeline (smaller, faster)? Currently `ChatWidget` already has its
   own model combo.
4. **Persistence** — save chat sessions to JSON next to pipeline
   results? Off-scope for MVP.
5. **Resetting on new pipeline run** — `reset_toggle` exists; keep
   behaviour, but warn user "the data the chat refers to was just
   replaced".
6. **MCP-tool side effects** — `search_gnd` may write to local cache.
   Acceptable side effect during chat? (Yes — same as during pipeline.)

## Files to add / change

**Add**
- `src/ui/chat_session.py`
- `src/ui/chat_session_tools.py`
- `tests/test_chat_session_tools.py`

**Change**
- `src/ui/chat_widget.py` — drop `current_context` flattening, use
  session + tool registry.
- `src/ui/chat_worker.py` — replace with `ChatAgentWorker`.
- `src/ui/main_window.py` — wire session.
- `src/core/pipeline_manager.py` — keep `last_shared_context`.
- `docs/agentic_workflow.md` — link to this plan.

## Related docs

- [`agentic_workflow.md`](../agentic_workflow.md) — v4 architecture (same
  AgentLoop / ToolRegistry stack the chat will reuse).
- [`workflow_yaml_spec.md`](../workflow_yaml_spec.md) — registered tool
  functions overview.
