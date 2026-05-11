# ALIMA Documentation

Technical documentation for the ALIMA project. User-facing changes:
[`../CHANGELOG.md`](../CHANGELOG.md). Developer log:
[`../AIChangelog.md`](../AIChangelog.md).

## Agentic Workflow System (v4)

The active pipeline architecture — YAML-driven, plugin-extensible.

- [`agentic_workflow.md`](agentic_workflow.md) — Architecture, step types,
  invocation, configuration knobs.
- [`workflow_yaml_spec.md`](workflow_yaml_spec.md) — Full YAML schema
  reference. All step fields, registered tool functions, MCP tools, presets,
  path resolver, conditional engine.

## Classic Pipeline (rigid)

Used when `enable_agentic_mode=False`. Both modes share `PipelineManager`
state shape and `PipelineStepExecutor` logic.

- [`pipeline_classic.md`](pipeline_classic.md) — 5-step pipeline overview
  (input → initialisation → search → keywords → classification).
- [`pipeline_classic_flow.md`](pipeline_classic_flow.md) — Mermaid flow
  diagram.
- [`initialization_pattern.md`](initialization_pattern.md) — Service
  initialization order across GUI/CLI/Webapp (ConfigManager →
  PromptService → LlmService → AlimaManager → PipelineManager).

## Subsystems

- [`iterative_gnd_search.md`](iterative_gnd_search.md) — Missing-concept
  feedback loop (extracts `<missing_list>`, re-searches, re-runs).
- [`dk_classification_splitting.md`](dk_classification_splitting.md) —
  Splitting large DK classification lists for parallel LLM processing.
- [`dk_retrieval_flow.md`](dk_retrieval_flow.md) /
  [`dk_retrieval_flow_technical.md`](dk_retrieval_flow_technical.md) —
  DK retrieval Mermaid diagrams.
- [`webapp_session_history.md`](webapp_session_history.md) — Webapp
  session/recovery model.
- [`configuration.md`](configuration.md) — Config file locations + format.
- [`llm_reproducibility_analysis.md`](llm_reproducibility_analysis.md) —
  LLM determinism analysis (Nov 2025).

## Provider Strategy

Analysis-only (not yet implemented).

- [`provider_strategy_analysis.md`](provider_strategy_analysis.md)
- [`provider_strategy_technical_spec.md`](provider_strategy_technical_spec.md)
- [`provider_strategy_migration_guide.md`](provider_strategy_migration_guide.md)
- [`provider_strategy_summary.md`](provider_strategy_summary.md)

## Legacy / Archived

[`legacy/`](legacy/) — Historical design docs and superseded plans. Kept
for context, not authoritative.

- `agentic_workflow_v3.md` — Old MetaAgent + 4 SubAgents design (replaced
  by v4 in April 2026).
- `agentic_v3_plan.md` — Original planning doc (iterative search + DK
  splitting + agentic v3 vision). Iterative-search and DK-splitting
  parts have since been implemented.
- `ui_restructuring_2025.md` — UI service-layer refactor (completed).
- `vision_implementation_plan.md` — Old high-level vision.
- `logging_refactor_completed.md` — Completed logging refactor plan.
