# CLI - Command Line Interface

## [Preserved Section - Permanent Documentation]

### Architecture
- **Entry**: `src/alima_cli.py` is a backward-compat shim → `src/cli/main.py` (the original 3062-line monolith was split into this package).
- **`main.py`**: argparse setup + dispatch (one if/elif → handler per command).
- **`commands/`**: one handler module per command group (pipeline, provider, database, search, state, protocol, setup, workflow, agent). Handlers are thin frontends that delegate to the shared services.
- **`formatters/protocol_formatters.py`**: CLI-specific result rendering (detailed / compact CSV / K10+ WinIBW catalog export); loads state via `PipelineJsonManager`.
- **Parity**: the CLI reuses the shared pipeline logic — `PipelineConfigBuilder` (config), `PipelineStepExecutor`/`PipelineManager` (execution), `LlmService`/`ConfigManager` (providers). No parallel reimplementation.
- **Shared in `main.py`**: `_SETUP_EXEMPT_COMMANDS` (commands skipping first-run/prompts checks), `_add_llm_args` (shared `--provider`/`--model`/`--temperature`).

## [Variable Section - Short-term Information]

### ✅ FIXED (Oct 2026): `--agentic` ohne `--workflow` konnte den `__classic__`-Sentinel als Workflow-Namen leaken
`PipelineConfig.create_from_provider_preferences()` übernimmt `SystemConfig.default_workflow`
unconditionally als `workflow_name` — das kann legitim `"__classic__"` sein (der Sentinel, den
`src/ui/settings/workflow_list.py::CLASSIC_STEM` schreibt, wenn die klassische Pipeline als
Standard markiert ist). `handle_pipeline` setzte `enable_agentic_mode = True` für `--agentic`,
ohne das zu prüfen — ohne `--workflow` lief ein solcher Aufruf auf einen Versuch,
`__classic__.yaml` zu laden (gibt es nicht), mit einer Fehlermeldung, die den internen Sentinel
nannte, den der Operator nie eingegeben hat. `agentic_mode_needs_an_explicit_workflow()`
(`pipeline_cmd.py`) fängt den Widerspruch jetzt vorher ab und meldet klar: `--workflow <name>`
angeben. Tests: `tests/test_cli_agentic_classic_sentinel.py`.

### Known Issues / Cleanup
- **stdout convention (resolved — non-issue)**: `formatters/protocol_formatters.py` uses bare `print()` for display/export output (correct), `print_result` is for command result lines. `print_result` always writes stdout and only adds logging, so there's no quiet-handling gap to fix — the split is intentional.
- `save-state` is deprecated but intentionally retained (operator decision, June 2026).

## [Instructions Block - Operator-Defined Tasks]

_(no open operator tasks beyond the cleanup above)_
