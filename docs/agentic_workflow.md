# Agentic Workflow System (v4)

**Status**: Active. Default for `enable_agentic_mode=True`.
**History**: Replaces v3 hardcoded MetaAgent + 4 SubAgents (April 2026, commit `23729cc`).
v3 doc archived at [`legacy/agentic_workflow_v3.md`](legacy/agentic_workflow_v3.md).

## What it is

A YAML-driven pipeline runner. Each workflow is a list of `steps`, each step
is either an LLM call or a deterministic Python function. Steps share state
through a `SharedContext` and can reference each other's outputs via
`${steps.<id>.<field>}` placeholders.

## Layers

```
┌─ workflows/*.yaml          User-editable pipelines
├─ WorkflowExecutor          Linear step runner (always)
├─ MetaAgent                 Optional planning/reflection loop
│                            (PLAN → EXECUTE → OBSERVE → REFLECT)
├─ Step types (registry)     llm_agent, deterministic, reflection, composite
├─ Tool functions (registry) gnd_batch_search, dk_search_agentic, …
└─ SharedContext             Typed fields + `extra: Dict` for free fields
```

## Execution paths

* **Sequential** (default): `WorkflowExecutor.run(workflow, context)` walks
  steps in YAML order respecting `depends_on:` and `condition:`.
* **MetaAgent loop** (opt-in): set `meta_agent.enabled: true` in the workflow
  YAML — `MetaAgent` decides which step runs next based on
  `ReflectionStep` quality reports, until quality goals are met or
  `max_cycles` reached.

## Step types

| Type | Class | Purpose |
|------|-------|---------|
| `llm_agent` | `LLMAgentStep` | One LLM call. Supports tool-use via `AgentLoop`, chunking, prompt resolution from `prompts.json`. |
| `deterministic` | `DeterministicStep` | Calls a registered Python function. Receives `inputs` as kwargs, optional `config`/`tool_registry`/`context` kwargs. |
| `reflection` | `ReflectionStep` | LLM quality report — used inside MetaAgent loop. |
| `composite` | `CompositeStep` | Groups child steps (rarely used). |

Add a new step type by subclassing `BaseStep` and decorating with
`@register_step("name")`.

## Tool functions

Register a deterministic function with `@register_tool_fn("name")` and
reference it from YAML as `function: name`. Currently registered (see
`src/core/agents/deterministic_functions.py`):

| Name | Purpose |
|------|---------|
| `gnd_batch_search` | Batch GND/SWB/Lobid search, enriches local DB |
| `dk_data_collect` | Aggregate DK data from SharedContext |
| `dk_search_agentic` | Per-keyword catalog DK search (mirrors classic `execute_dk_search`) |
| `build_dk_search_results` | Merge DK entries with LLM classifications for GUI |
| `catalog_multi_search` | Fan-out search SWB+Lobid+catalog, merge by title |
| `catalog_title_search` | Title-mode catalog search (Libero `k`) |
| `gnd_entry_lookup` | Single GND entry lookup with optional Lobid fallback |
| `extract_gnd_related` | Extract related-GND graph from a GND entry |
| `gnd_batch_metadata` | Bulk metadata fetch from local GND DB |

## MCP tools (LLM tool-use)

LLM agents can call MCP tools via `tools:` in YAML. Available tools (see
`src/mcp/tool_registry.py`): `search_gnd`, `get_gnd_entry`, `get_gnd_batch`,
`get_search_cache`, `get_dk_cache`, `get_classification`, `search_lobid`,
`search_swb`, `search_catalog`, `search_catalog_titles`, `resolve_doi`,
`list_pipeline_results`, `get_pipeline_result`, `get_pipeline_keywords`,
`get_pipeline_abstract`, …

Presets bundle tools — defined in `src/mcp/default_presets.yaml`:
`library`, `gnd`, `classification`, `none`. Custom presets via
`~/.config/alima/tool_presets.yaml`.

## SharedContext

State container passed through every step. Key fields:

* `abstract: str` — input text
* `initial_keywords: List[str]` — user-provided seeds
* `extracted_keywords: List[str]` — from initialisation
* `step_results: Dict[str_id, dict]` — every step's raw output
* `extra: Dict[str, Any]` — free-form fields written by steps via
  `outputs: { extra.foo: ... }`

Read with `${path}` resolver:
* `${abstract}` → typed field
* `${steps.search.gnd_entries}` → output of step `search`
* `${extra.dk_entries}` → entry written via `extra.dk_entries`

## Built-in workflows

Located in `workflows/`:

| File | Purpose |
|------|---------|
| `alima_classic.yaml` | Classic 5-step ALIMA (extraction → search → selection → DK) |
| `catalog_search.yaml` | Multi-source catalog lookup with optional LLM ranking |
| `synonym_expansion.yaml` | Single keyword → GND entry → LLM expansion → validated GND |
| `batch_metadata.yaml` | Bulk GND-ID metadata fetch |
| `title_list_search.yaml` | Acquisition wishlist duplicate analysis |

Archived v3 workflows: `workflows/legacy/`.

## Invocation

### CLI
```bash
alima workflows list                        # discover workflows
alima workflow alima_classic --input '{"abstract": "..."}'
alima workflow title_list_search --input-file titles.json --output result.json
alima workflow alima_classic --only-step extraction
```

### GUI
Pipeline-Tab → workflow dropdown → enable agentic mode → run. Live state
is shown by `AgenticContextWidget` (left dock).

### Programmatically
```python
from src.core.agents.workflow_loader import load_workflow
from src.core.agents.workflow_executor import WorkflowExecutor
from src.core.agents.shared_context import SharedContext

wf = load_workflow("workflows/alima_classic.yaml")
ctx = SharedContext(abstract="...", initial_keywords=[])
report = WorkflowExecutor(llm_service=..., tool_registry=...).run(wf, ctx)
```

## Configuration knobs

In `PipelineManager.config` (set via GUI dialog or CLI flags):

* `enable_agentic_mode: bool` — switch sequential pipeline ↔ workflow runner
* `workflow_name: str` — default `"alima_classic"`, must match a discovered YAML
* `custom_workflow_path: str` — override discovery with explicit path
* `agentic_step_id: str` — run only one step (warm-start)
* `agentic_input_context_path: str` — JSON file as warm-start context
* `agentic_verbose: bool` — log full prompts to stream
* `meta_agent_enabled: bool` — enable planning loop
* `agentic_max_iterations: int` — cap MetaAgent cycles

## Prompt sourcing

Per-step prompts are resolved (in order):

1. `prompts:` block at workflow root (per-step override map)
2. `system_prompt:` / `user_prompt:` directly in step YAML
3. `prompts.json` task lookup via `PromptService` (matches step `id` to task name)

So `alima_classic.yaml` can keep prompts inline OR delegate to
`prompts.json` for per-provider/per-model variants. See
`src/core/agents/prompt_resolver.py`.

## Schema reference

See [`workflow_yaml_spec.md`](workflow_yaml_spec.md) for full YAML schema.

## Adding capabilities

1. **New deterministic step** — add function to
   `src/core/agents/deterministic_functions.py`, decorate with
   `@register_tool_fn("name")`, reference from YAML.
2. **New step type** — subclass `BaseStep`, decorate with
   `@register_step("name")`, place under `src/core/agents/steps/`,
   import once at module load.
3. **New workflow** — drop YAML into `workflows/` (or
   `~/.config/alima/workflows/`). Discovered automatically.
4. **New MCP tool** — add schema to `src/mcp/tool_schemas.py`, handler to
   `src/mcp/tool_registry.py`, register in `register_all_tools()`.

## Related docs

* [`workflow_yaml_spec.md`](workflow_yaml_spec.md) — YAML schema reference
* [`iterative_gnd_search.md`](iterative_gnd_search.md) — missing-concept loop
* [`dk_classification_splitting.md`](dk_classification_splitting.md) — DK chunking
* [`legacy/agentic_workflow_v3.md`](legacy/agentic_workflow_v3.md) — old MetaAgent+SubAgents design
