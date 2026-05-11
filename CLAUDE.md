# CLAUDE: Your AI Assistant for the ALIMA Project

## Overview

**ALIMA** (Automatic Library Indexing and Metadata Analysis) — pipeline for library science workflows combining LLM-powered text analysis with GND/SWB keyword search and DK/RVK classification.

## Core
1. Don't assume. Don't hide confusion. Surface tradeoffs.
2. Minimum code that solves the problem. Nothing speculative.
3. Touch only what you must. Clean up only your own mess.
4. Define success criteria. Loop until verified.

### Conservative Self-Assessment Rules for AI
When documenting implemented features, the AI must apply these rules:

1. **Automated tests pass ≠ correct** — tests only cover what was anticipated. Unknown failure modes exist.
2. **Agreement with reference ≠ general correctness** — the reference comparison is only as broad as the test set.
3. **No gaps visible ≠ no gaps exist** — absence of a known bug is not the same as correctness. Especially for AI-generated scientific code: the most dangerous bugs are those that produce plausible but wrong results.
4. **"Implemented" means the code compiles and runs** — it does not imply correctness, stability across all inputs, or completeness relative to the reference method.
5. **When in doubt, add a caveat** — a caveat that turns out to be unnecessary is harmless. A missing caveat on wrong code causes user errors.

## Very General Instructions for AI Coding
- Avoid flattery, compliments, or positive language. Be clear and concise. Do not use agreeable language to deceive.
- Do comprehensive verification before claiming completion.
- Show me proof of completion, don't just assert it.
- Prioritize thoroughness over speed.
- If I correct you, adapt your method for the rest of the task.
- No completion claims until you can demonstrate zero remaining instances.

## CLAUDE.md Hygiene
- Each source code dir has a CLAUDE.md with basic information and logic.
- **Keep CLAUDE.md files FOCUSED and CONCISE** — ONE clear idea per bullet, max 1-2 lines.
  - ❌ DON'T: Multi-paragraph explanations, code examples, historical details, completed features.
  - ✅ DO: Brief statements, links to detailed docs.
  - ✅ DO: `✅ **Feature name** — Brief description` for completed items.
- Remove completed/resolved items after 2-3 updates (move to git history / `AIChangelog.md`).
- Tasks corresponding to code go in the right CLAUDE.md.
- Each CLAUDE.md has a variable part (short-term info, bugs) and preserved part (permanent knowledge).
- **Instructions blocks** contain operator-defined future tasks and visions.
- Only include information important for ALL subdirectories in main CLAUDE.md.
- **Rule of thumb**: section >20 lines → place elsewhere.

## Implementation Standards
- Mark new functions as `Claude Generated` for traceability.
- Remove TODO hashtags after approved.
- Implement comprehensive error handling and logging.
- Maintain backward compatibility where possible.
- **Always check instructions blocks** in relevant CLAUDE.md files before implementing.
- Reformulate task/vision entries if not yet CLAUDE-formatted.
- Avoid hardcoded provider lists — read from `llmanager`.

## Workflow States
- **ADD**: to be added • **WIP**: in progress • **ADDED**: implemented • **TESTED**: works (operator confirmed) • **APPROVED**: → changelog, remove from CLAUDE.md.

## Documentation Update Rules
- Replace debugging details with architecture decisions when issues are resolved.
- Document the *why* behind decisions, not the *what*.
- Eliminate redundant info that doesn't add architectural value.
- Significant improvements → [`AIChangelog.md`](AIChangelog.md).

## Git Best Practices
- **Only commit source files**: `git add <file>`, never `git add -A` without review.
- **Review before committing**: `git diff` + `git status`.
- **Commit message**: action verb (Fix/Add/Improve/Refactor) + brief description.
- Include Claude Co-Author line.
- Test artifacts stay local (`.gitignore`).

## Quality Assurance — Test Maintenance Rules

**The test suite MUST stay green. Broken tests silently ignored are worse than no tests.**

### Refactoring or renaming APIs
- Update ALL affected tests in the same commit.
- If a class/method is renamed, grep for all test references and update them.
- If a table schema changes, update `test_search.py` to match.

### Test isolation
- Tests using `UnifiedKnowledgeManager` MUST call `UnifiedKnowledgeManager.reset()` in both `setUp` and `tearDown`.
- Tests MUST use `DatabaseConfig(db_type='sqlite', sqlite_path=<tempfile>)` — never the production config (may point to MariaDB).
- Tests using Qt classes (SearchEngine, LlmService) require a `QApplication` — use `Mock(spec=...)` to avoid it.

### Submitting/reviewing PRs
- Run `python -m pytest tests/ -v` locally before opening a PR.
- A PR introducing new test failures is not ready to merge.
- Pre-existing failures: fix in a separate commit and note explicitly.

### Incident: 9-month silent test debt
- `SearchEngine` rewritten async → Qt-signal-based in July 2025.
- `test_cache.py` / `test_search.py` not updated → silently broken for 9 months.
- Discovered during PR #9 review (March 2026). Apply the rules above to prevent recurrence.

## Critical Requirements

**All pipeline changes MUST be usable by both CLI and GUI.**
- Shared logic: `src/utils/pipeline_utils.py`.
- Configuration parity: identical params across interfaces.
- Use `PipelineConfigParser` + `PipelineConfigBuilder` (single source of truth).

## [Preserved Section — Permanent Documentation]
*Change only if explicitly wanted by operator.*

### Pipeline Modes
- **Classic (rigid)**: 5-step linear pipeline (input → initialisation → search → keywords → classification). Details: [`docs/pipeline_classic.md`](docs/pipeline_classic.md).
- **Agentic (v4)**: YAML-driven workflows via `WorkflowExecutor`. Default workflow: `alima_classic.yaml`. Details: [`docs/agentic_workflow.md`](docs/agentic_workflow.md), schema: [`docs/workflow_yaml_spec.md`](docs/workflow_yaml_spec.md).
- Both modes share `PipelineManager` state and `PipelineStepExecutor`.

### Database
- `alima_knowledge.db` — facts (`gnd_entries`, `classifications`) + mappings (`search_mappings`).
- `UnifiedKnowledgeManager` — singleton, mapping-first search. Thread-safety details in `MEMORY.md`.

## [Variable Section — Current Tasks]
*(Empty. Move active work here as bullets; move completed → `AIChangelog.md`.)*

## [Instructions Block — Operator-Defined Tasks]

### Vision
- Restructure code: consolidate distributed logic from `utils`, `core`, `suggestors`.
- Maintain unified pipeline architecture (CLI/GUI/Webapp parity).
- Extend agentic v4 to cover more workflow types beyond classical pipeline.

### Future Tasks
1. **Code Restructuring**: Consolidate distributed logic.
2. **Pipeline Enhancement**: Templates, advanced configuration UI.
3. **Batch Enhancement**: Extended image analysis, URL scraping.
4. **Performance**: Connection pooling, result pagination, memory optimization.
5. **Agentic Hauptagent**: `main_agent:` block in YAML — meta-orchestrator that calls sub-workflows as tools.

## Module Documentation
- [`src/core/CLAUDE.md`](src/core/CLAUDE.md) — Core business logic, pipeline orchestration, data management.
- `src/core/agents/` — Agentic v4: `WorkflowExecutor`, `LLMAgentStep`, `DeterministicStep`, optional `MetaAgent` loop. (v3 SubAgents removed April 2026 — see [`docs/legacy/agentic_workflow_v3.md`](docs/legacy/agentic_workflow_v3.md).)
- [`src/mcp/CLAUDE.md`](src/mcp/CLAUDE.md) — MCP tool layer: schemas, registry, handlers.
- [`src/ui/CLAUDE.md`](src/ui/CLAUDE.md) — PyQt6 GUI components.
- [`src/utils/CLAUDE.md`](src/utils/CLAUDE.md) — Configuration, batch processing, logging.
- [`docs/`](docs/) — Architecture docs (agentic, classic pipeline, subsystems, legacy).
- [`AIChangelog.md`](AIChangelog.md) — Detailed dated developer log.
- [`CHANGELOG.md`](CHANGELOG.md) — User-facing release notes.
