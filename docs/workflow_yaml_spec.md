# Workflow YAML Schema (v4)

Reference for `workflows/*.yaml` files consumed by `WorkflowExecutor`.
Companion to [`agentic_workflow.md`](agentic_workflow.md).

## Top-level

```yaml
name: "Human readable workflow name"          # str, required
version: "4.0"                                 # str, required ("4.x" expected)
description: "What this workflow does"         # str, optional

settings:                                      # global LLM defaults
  temperature: 0.5                             # float, fallback for all llm_agent steps
  top_p: 0.9                                   # float
  max_tokens: 32768                            # int

context_init:                                  # initial SharedContext fields
  abstract: ""                                 # any field of SharedContext
  initial_keywords: []

prompts:                                       # OPTIONAL per-step prompt overrides
  <step_id>:
    system: "..."
    user: "..."
    # OR reference a prompts.json task:
    task: "initialisation"

meta_agent:                                    # OPTIONAL planning loop config
  enabled: false                               # default false
  max_cycles: 10
  reflection_model: ""                         # provider model id; empty = default
  reflection_provider: ""
  quality_rules:
    - "dk_depth: at_least_4_digits"
    - "keywords: min_10_verified"

steps:                                         # required, non-empty list
  - id: ...
    type: ...
    ...
```

## Step block (common fields)

```yaml
- id: extraction                               # str, unique within workflow
  type: llm_agent | deterministic |            # str, must be in STEP_REGISTRY
        reflection | composite
  description: "..."                           # str, optional
  enabled: true                                # bool, default true
  depends_on: [other_step_id]                  # list[str], topo-sort hint

  condition: "${extra.intent} != ''"           # OR `when:` — see Conditional
  when: "len(${gnd_entries}) > 0"

  inputs:                                      # mapping resolved against context
    abstract: "${abstract}"
    keywords: "${steps.extraction.keywords}"
    explicit_value: "literal string"           # non-${} values pass through

  outputs:                                     # where step writes results
    extracted_keywords: "response.keywords"    # SharedContext.extracted_keywords ← step_result["response"]["keywords"]
    extra.dk_entries: "result.dk_entries"      # SharedContext.extra["dk_entries"] ← step_result["result"]["dk_entries"]
```

## Path resolver

`${path}` is replaced from `SharedContext`:

| Path | Source |
|------|--------|
| `${abstract}` | `context.abstract` |
| `${initial_keywords}` | `context.initial_keywords` |
| `${working_title}` | `context.working_title` |
| `${extracted_keywords}` | `context.extracted_keywords` |
| `${gnd_entries}` | `context.gnd_entries` |
| `${selected_keywords}` | `context.selected_keywords` |
| `${keyword_chains}` | `context.keyword_chains` |
| `${missing_concepts}` | `context.missing_concepts` |
| `${dk_classifications}` | `context.dk_classifications` |
| `${rvk_classifications}` | `context.rvk_classifications` |
| `${dk_search_results}` | `context.dk_search_results` |
| `${steps.<id>.<field>...}` | `context.step_results[id][field]...` |
| `${extra.<key>...}` | `context.extra[key]...` |
| `${input.<key>}` | alias for `${extra.input.<key>}` |

* Dotted access traverses dicts.
* Integer segments index into lists: `${steps.search.gnd_entries.0.title}`.
* Unknown paths raise `KeyError` (LLMAgentStep falls back gracefully where
  reasonable; see source).

## Conditional (`when:` / `condition:`)

Minimal expression engine (`conditional_engine.py`):

```yaml
when: "${extra.flag} == true"
when: "len(${gnd_entries}) > 0"
when: "${extra.score} >= 0.5 and ${extra.intent} != ''"
```

Supported: `==`, `!=`, `>`, `<`, `>=`, `<=`, `and`, `or`,
`len/bool/str/int(...)`. Falsy results → step skipped.

## Step type: `llm_agent`

```yaml
- id: extraction
  type: llm_agent

  inputs: {abstract: "${abstract}"}
  outputs: {extracted_keywords: "response.keywords"}

  llm:
    temperature: 0.5
    top_p: 0.9
    max_tokens: 32768
    max_iterations: 1                          # AgentLoop iteration cap (tool-use rounds)
    provider: ""                               # optional override (default: pipeline config)
    model: ""

  tools:                                       # OPTIONAL tool exposure to LLM
    explicit: [search_swb, get_gnd_batch]      # exact tool names from MCP registry
    preset: library                            # preset name (explicit wins)

  chunking:                                    # OPTIONAL — slice an input list
    enabled: true
    chunk_field: keywords                      # key in resolved inputs (must be list)
    chunk_fields: [title, gnd_id]              # OPTIONAL — project items to subset of fields
    chunk_size: 350
    sort_by: count                             # sort items before chunking
    sort_desc: true
    merge_key: keywords                        # response[merge_key] expected to be list
    dedup_field: keyword                       # dedupe merged items by lowercased field
    max_merged: 80                             # cap merged result length

  system_prompt: |
    ...
  user_prompt: |
    ...
    {abstract}                                 # f-string-style substitution from inputs
```

**Output shape** (what `outputs:` paths see):

```python
{
  "response": <parsed JSON from LLM, dict>,
  "response_text": "<raw text>",
  "iterations": 1,
  "tool_log": [...],
}
```

Reference `response.keywords`, `response.classifications`, etc.

**Prompt resolution order**:
1. workflow root `prompts:` block for this step
2. inline `system_prompt:` / `user_prompt:` in step
3. `prompts.json` task (matched by step `id`) via `PromptService`

## Step type: `deterministic`

```yaml
- id: search
  type: deterministic
  function: gnd_batch_search                   # must be in TOOL_FN_REGISTRY

  inputs:                                      # passed as kwargs
    keywords: "${extracted_keywords}"

  config:                                      # passed as `config=` kwarg
    sources: [swb, lobid]
    enrich_from_local_db: true

  outputs:                                     # paths into function return value
    gnd_entries: "result.entries"
```

The function receives:
* every `inputs` key as a kwarg
* `config=<config dict>` (if param exists)
* `tool_registry=<ToolRegistry>` (if param exists)
* `context=<SharedContext>` (if param exists)

**Return shape** (what `outputs:` paths see):
```python
{
  "result": <function return value>,
  ...
}
```

Reference `result.entries`, `result.tool_calls`, etc.

## Step type: `reflection`

LLM quality-check step used by MetaAgent loop. Not normally placed
directly in `steps:`. See `src/core/agents/steps/reflection_step.py`.

## Step type: `composite`

Groups child steps. Rarely used. See `composite_step.py`.

## Registered tool functions

| Function | Inputs | Returns (key fields) |
|----------|--------|----------------------|
| `gnd_batch_search` | `keywords: List[str\|dict]`, `config: {sources, enrich_from_local_db}` | `entries`, `tool_calls` |
| `dk_data_collect` | (reads from context) | `dk_entries` |
| `dk_search_agentic` | (reads context for keywords), `config: {max_keywords}` | `dk_entries`, `formatted_prompt`, `dk_search_results` |
| `build_dk_search_results` | `dk_entries`, `dk_classifications` | `results` |
| `catalog_multi_search` | `queries: List[str]`, `config: {sources, enrich_from_local_db}` | `hits`, `tool_calls` |
| `catalog_title_search` | `queries: List[dict]`, `config: {search_type, max_results}` | `hits`, `tool_calls` |
| `gnd_entry_lookup` | `gnd_id: str`, `config: {fallback_to_lobid}` | `entry`, `source` |
| `extract_gnd_related` | `gnd_id: str` | `related` |
| `gnd_batch_metadata` | `gnd_ids: List[str]`, `config: {fallback_to_lobid}` | `entries`, `missing`, `tool_calls` |

Source: `src/core/agents/deterministic_functions.py`.

## MCP tools (for `llm_agent` `tools:` block)

Exact tool names available to LLM agents (see `src/mcp/tool_schemas.py`):

* **GND/knowledge**: `search_gnd`, `get_gnd_entry`, `get_gnd_batch`,
  `get_search_cache`, `store_search_result`
* **Classification**: `get_dk_cache`, `get_classification`, `get_db_stats`
* **External catalogs**: `search_lobid`, `search_swb`, `search_catalog`,
  `search_catalog_titles`, `resolve_doi`
* **Pipeline result access**: `list_pipeline_results`, `get_pipeline_result`,
  `get_pipeline_keywords`, `get_pipeline_abstract`

**Presets** (`src/mcp/default_presets.yaml`):
* `library`: `search_gnd`, `search_lobid`, `search_swb`, `get_search_cache`
* `gnd`: `search_gnd`, `get_gnd_entry`, `get_gnd_batch`
* `classification`: `get_dk_cache`, `get_classification`, `search_catalog`
* `none`: `[]`

Custom presets: `~/.config/alima/tool_presets.yaml`.

## Discovery

Workflows are discovered (in order) from:
1. `./workflows/`
2. `~/.config/alima/workflows/`
3. `<package>/workflows/` (relative to source root)

`alima workflows list` enumerates all discovered files.

## Output mapping syntax

`outputs:` keys can target three destinations:

| Pattern | Destination |
|---------|-------------|
| `<typed_field>` | `SharedContext.<typed_field>` (must exist as attribute) |
| `extra.<key>` | `SharedContext.extra[<key>]` (free-form dict) |
| _(no path here — `step_results[id]` is set automatically)_ | `SharedContext.step_results[step_id]` |

Values (`"response.keywords"`, `"result.entries"`) are dotted paths into
the step's return dict.

## Minimal examples

**Pure deterministic**:
```yaml
name: "GND Batch Metadata"
version: "4.0"
steps:
  - id: fetch
    type: deterministic
    function: gnd_batch_metadata
    inputs:
      gnd_ids: "${extra.gnd_ids}"
    outputs:
      extra.metadata: "result.entries"
```

**Single LLM call**:
```yaml
name: "Title Extractor"
version: "4.0"
steps:
  - id: extract
    type: llm_agent
    inputs: {raw_text: "${abstract}"}
    outputs: {extra.titles: "response.titles"}
    llm: {temperature: 0.1, max_tokens: 2048}
    system_prompt: "Extract bibliographic titles. Return JSON."
    user_prompt: "Input:\n{raw_text}"
```

**Conditional branching**:
```yaml
steps:
  - id: search
    type: deterministic
    function: gnd_batch_search
    inputs: {keywords: "${initial_keywords}"}
    outputs: {gnd_entries: "result.entries"}

  - id: rerun
    type: deterministic
    function: gnd_batch_search
    when: "len(${gnd_entries}) < 5"
    depends_on: [search]
    inputs: {keywords: "${extra.fallback_terms}"}
    outputs: {extra.fallback_entries: "result.entries"}
```

See `workflows/alima_classic.yaml` for a full reference implementation.
