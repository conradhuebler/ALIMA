# ALIMA Workflow Definitions

YAML-driven workflows consumed by `WorkflowExecutor`.

## Core Workflows

- **`alima_classic.yaml`** — Classic linear pipeline (5 rigid steps).
- **`alima_classic_v51.yaml`** — Classic pipeline with v5.1 prompt tuning.
- **`alima_v51.yaml`** — Default agentic v5.1 pipeline.

## Institution-Specific Variants

- **`alima_v51_105.yaml`** — UB Freiberg variant of `alima_v51.yaml`.
  - **Only difference:** classification prompt restricts RVK to economics/business (WiWi); non-WiWi subjects use DK only. DDC is not used in Freiberg.
  - Proof-of-concept for local, prompt-level institution customization.

## RVK via `rvk_lookup` tool (both v51 + v51_105)

- `dk_collect.config.rvk_inline: false` → the deterministic DK step collects
  DK/DDC only (no RVK anchor derivation, no RVK-API calls inline).
- `classification` is a tool-calling step: `tools: [rvk_lookup]`,
  `llm.max_iterations: 4`, and an extra `final_keywords` input passed to the
  tool. RVK is fetched on demand by the classification LLM.
  - `alima_v51`: LLM calls `rvk_lookup` whenever RVK fits thematically.
  - `alima_v51_105`: prompt instructs the LLM to call `rvk_lookup` **only for
    WiWi**; non-WiWi → no tool call, DK only.
- **Sync rule:** the non-prompt wiring (`rvk_inline`, `tools`,
  `max_iterations`, `final_keywords` input) is identical in both files; only
  the RVK prompt block differs (general vs. WiWi-only).

## MetaAgent: generic core + YAML domain rules

- The MetaAgent (PLAN→EXECUTE→REFLECT) is always active in agentic mode. It
  serializes the `steps:` workers (planner) and checks results (reflection).
- **The planner + reflection cores are GENERIC (ALIMA-branded) in code** — they
  carry no GND/DK/phase assumptions. Each prompt has a `{workflow_rules}` slot.
  Domain rules live in the **workflow YAML** and are injected there by
  `MetaAgent._inject_rules` (fills the slot, else appends).
  - `meta_agent.planning.rules` — phase/routing rules (which step needs GND/DK…).
  - `meta_agent.reflection.rules` — per-phase quality checks (DK depth, missing…).
  - `rules:` **augments** the generic base (preferred); `system_prompt:` would
    fully replace it. Keep planner + reflection rules in sync (same phase model).
- **Reflection is opt-in**: the gate runs **only** if a `meta_agent.reflection:`
  block exists. No block → pure PLAN→EXECUTE chain (planner alone finishes).
  `alima_classic`, `catalog_search`, … have no block → no reflection.
- The `reflection:` block is self-contained: `model`/`provider` (empty = main
  model), `temperature`/`top_p`/`max_tokens`, `rules`, optional
  `system_prompt`/`user_prompt`. The reflection **user prompt** (state dump,
  code default) feeds the **real DK codes** + a deterministic `has_deep_dk`
  flag — so the gate stops demanding "deeper" DK on already-deep codes.
- Planner and reflection stay **two separate prompts** on purpose (cross-check).
- Missing-concept reruns (`search_missing`) are reflection-driven: clean terms
  are searched targeted-and-merged; concepts flagged "kein GND" are skipped.

## Default Workflow

- Configurable via `SystemConfig.default_workflow` (`alima_v51` by default).
- GUI: Settings → System → Standard-Workflow.
- Used by `PipelineConfig.create_from_provider_preferences()` and respected by the Qt6 Pipeline-Tab and the webapp workflow dropdown.

## Token Budget

- Each step's `llm.max_tokens` in the YAML is the default (4096 in both v5.1 workflows, 2048 for reflection).
- `PipelineConfig.global_max_tokens_override` outranks it for every step of a run: GUI toolbar "Budget", CLI `--max-tokens`. Unset = the YAML decides.
- A reasoning model spends this budget on its thinking channel before the answer starts; measurements in [`src/llm/CLAUDE.md`](../src/llm/CLAUDE.md), probe: `scripts/probe_thinking.py`.

## Maintenance Rule

- **`alima_v51_105.yaml` must stay in sync with `alima_v51.yaml`** except for the institution-specific classification prompt block.
- When updating `alima_v51.yaml`, apply the same non-prompt changes to `alima_v51_105.yaml`.
- The `meta_agent:` block (incl. `reflection:`) is identical in both files — keep it in sync.
- If the two files diverge beyond the classification prompt, that divergence is a bug.
