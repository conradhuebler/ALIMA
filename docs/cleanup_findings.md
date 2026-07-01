# Cleanup Findings Register (project-wide)

> Living register of technical-debt findings from the June 2026 core→subsystem
> debt sweep (pipeline, agentic, CLI, webapp, UI). Each open item has a clear
> recommendation so it can be picked up and decided deliberately. Done items are
> listed for context; details live in `AIChangelog.md` + the linked specs.

## Prioritization axes
**Value** (user-facing / maintainability) · **Risk** (regression, esp. GUI which is
not runtime-testable in the dev sandbox) · **Effort** (size) · **Deps**.

Recurring lesson this sweep: *surface scans overstate duplication.* Every area was
verified at the code; the safe core was taken, the risky/unclear parts documented
(not blindly refactored). When splitting/moving modules, run an AST undefined-name
scan ([[module-split-verification]]).

---

## ✅ Done this session (context)
Branch `agent`, commits `601614a … caec4a3`:
- Shared GND-search core (`gnd_search_core.py`) + chunking dedup (`chunking.py`).
- `pipeline_utils.py` god-module split (7615→5098) into focused modules (facade).
- `llm_service.py` superseded dead per-provider generators removed (−436 LoC).
- WP13: dead modules/tests/backups/legacy workflows removed; sub-CLAUDE.md hygiene.
- CLI: `_SETUP_EXEMPT_COMMANDS` + `_add_llm_args`; new `cli/CLAUDE.md`.
- Workflow discovery consolidated into `workflow_loader.discover_workflow_files()`
  (was 6× duplicated across CLI/webapp/4 GUI); JSON-serializer dedup.
- UI: dead `widgets.py` removed; `single_step_dialog` adopts `ProviderModelSelector`.
- Classic↔agentic convergence operator-confirmed (TESTED).

---

## Open findings — prioritized

### P1 — Quick, safe wins
| ID | Area | Finding | Action | Risk |
|----|------|---------|--------|------|
| F-1 | data | Stale `swb_gnd_cache.json` may hold cached network-failures as "no hit" (pre-WPA) | ✅ DONE (June 29): purged the 3 local gitignored caches (`src/{,cli/,webapp/}data/swbsuggester/`); they rebuild on next search | none |
| F-2 | CLI | Mixed stdout: `protocol_formatters` uses bare `print()`, handlers use `print_result` | ✅ `print()`-vs-`print_result` split confirmed correct, no change there. ✅ **Detail fixed (July 1):** `display_protocol()`'s 3 error prints (lines 36/72/74) now use `file=sys.stderr`, matching siblings `display_protocol_compact()`/`display_protocol_k10plus()`. Suite green (858 passed). | none |

### P2 — Major workstream: Search-Provider-Plugins (+ coupled bug)
| ID | Area | Finding | Action | Risk |
|----|------|---------|--------|------|
| F-3 | core/suggesters | DB/API search sources were ad-hoc; finc bypassed the suggester abstraction; `SuggesterType` enum + per-frontend wiring | ✅ DONE (June 29, commits `5061801`/`35bc487`/`85900e2` + GUI): capability-based `SearchProvider` standard + single `@register_provider` registry (`src/core/search/`); `SuggesterType` retired; caching as `CachingProvider` wrapper; finc folded into the standard; MCP search tools generated from `ProviderToolSpec`; `SearchProviderConfig` + GUI selector. Spec: [`search_provider_plugins.md`](search_provider_plugins.md) | high (touches search core) |
| F-4 | agentic | "Häufigkeit zeigt 1": mapping-cache hits got `count=1`; agentic GND counts looked wrong | ✅ DONE (June 29, with F-3 P2): cache now stores per-GND-ID `gnd_counts`; cache hits keep pool `count=1` (ranking unchanged) and restore a display-only `display_count` read by `flatten_gnd_hits`/GUI/agentic. `rank_pool` never reads it. Open: final agentic-run confirmation (needs LLM). | medium |

F-3/F-4 delivered as one workstream (P1+P2+P3). The MCP auto-generation was proven
byte-identical to the former hand-written handlers; the GUI provider-selector needs
an operator click-test (sandbox-untestable, per the GUI gate below).

### P3 — Structural refactors (high value, sign-off-gated)
**Lesson (June 30): the *refactor* is sandbox-doable and was done — only the final
operator click-test sign-off is gated, not the implementation.** Verbatim moves
verified statically (`py_compile` + AST undefined-name scan + MRO smoke + green suite)
catch the real risk class (a stdlib/const import not travelling with a moved body).
What import + headless suite do NOT catch: a moved body calling a cross-mixin `self.x`
on a runtime path no test exercises (no widget is constructed in tests) — that needs a
click-test. Do them one unit at a time with a hand-off, not in big blind batches.
| ID | Area | Finding | Action | Risk |
|----|------|---------|--------|------|
| F-5 | UI | God-files. **Code-complete (June 30) — all 5 split; operator click-test sign-off pending:** ✅ `unified_provider_tab` 2115→681 (→ `provider_dialogs.py` 540 + `task_preferences_widget.py` 932); ✅ `pipeline_config_dialog` 2226→765 (→ `step_config_widgets.py` 1487); ✅ `pipeline_chat_panel` 1923→698 (→ `chat_input_widgets.py` 122 + `repetition_warning_bar.py` 256 + `_chat_panel_pipeline_log.py` 346 `PipelineLogMixin` + `_chat_panel_chat_agent.py` 453 `ChatAgentMixin` + `_chat_panel_bus.py` 219 `BusEventMixin`; commit `1305b73`); ✅ `pipeline_tab` 2882→560 (→ `_pipeline_tab_ui/control/events` mixins + `pipeline_step_widget.py` 191; commit `e75b80d`); ✅ `main_window` 2650→495 (→ `_main_window_data/menu/results/settings` mixins + `commit_selector_dialog.py` 278; commit `2c0a46c`). All splits re-verified: `py_compile` clean, MRO smoke OK, suite **858 passed / 10 skipped / 0 failed**. **Full-sweep recheck (July 1) found one real bug**, missed by the original AST scan: `unified_provider_tab` split left `TaskPreferencesWidget._add_model_to_task_priority`/`_bulk_add_model_to_tasks` calling `TaskModelSelectionDialog` without importing it from `provider_dialogs.py` — `NameError` on every "Add Model" click, undetected by import + suite for ~16h. Caught by a bytecode `LOAD_GLOBAL` scan (more precise than AST text-scan — see [[module-split-verification]]); fixed (import added), suite stayed green. Re-scanned all 13 split classes afterward with the bytecode method — no other hits. | Extract cohesive widgets/sections; one file at a time + operator click-test (verbatim moves verified by bytecode `LOAD_GLOBAL` scan + construct-smoke + green suite — prefer this over the noisier AST text-scan). For a single god-*class* (no embeddable sub-widgets), decompose into behavioral mixins — verbatim method moves, reachable via MRO, zero call-site changes. | high (Qt, untestable here) |
| F-6 | webapp | ✅ **DONE (June 30)** — `app.py` 2537→240 LoC (−90%). | Split into thin app factory + `session_state.py`/`render_bridge.py`/`session_io.py` + `routers/{workflows,models,sessions,export,websocket,analysis,agent}.py`; re-export shims keep `src.webapp.app` test contract; route table byte-identical; suite green (841). Sandbox-verified (no GUI gate). | done |
| F-7 | UI | 17 `QThread`/`StoppableWorker` subclasses; base only partly adopted; 2 model-loaders overlap (`_ModelLoadWorker` vs `ModelFetchWorker`) | **Partly done (June 30):** model-loaders merged into shared `ModelLoadWorker(StoppableWorker)` in `workers.py` (signal contracts preserved; unit-tested `tests/test_model_load_worker.py`) — GUI wiring needs a click-test. **Blanket base-adoption descoped:** most remaining plain-`QThread` workers are short probes that don't need cancellation; convert only genuinely-cancellable long workers case-by-case (GUI-gated). | medium (threading) |

### P4 — Decide-on-touch (behavior changes, not pure dedup)
| ID | Area | Finding | Action | Risk |
|----|------|---------|--------|------|
| F-8 | UI | `ProviderModelSelector` non-adopters have *intentional* differences (see `ui/CLAUDE.md`): `comprehensive_settings` (common+enabled provider list), `abstract_tab` (pushed `available_models` cache + async-incompatible restore), `pipeline_config_dialog` (per-step grid) | Only when already editing those tabs, with operator OK on the UX change + click-test | medium-high |
| F-9 | UI | Pre-existing (not a sweep regression). `PipelineConfigDialog.save_as_provider_preferences()` had a corrupted comment from commit `b373236` (Sept 2025): a literal `\n` text merged a `# TODO: ...` comment with `if False:  # Disabled: ...` onto one physical line, so the dead "show validation issues" block (referencing the never-assigned `validation_issues`) was actually live — nested inside the `for step_config in step_configs.values():` loop. Reachable from a real button; silently swallowed a `NameError` on every save via a broad `except Exception`. | ✅ **DONE (July 1):** dead block deleted cleanly (the commented-out `validation_issues = unified_config.validate_preferences(...)` TODO above it stays, genuinely inert). `py_compile` + bytecode `LOAD_GLOBAL` scan clean, suite green (858 passed). | none |

---

## Cross-cutting / meta
- **GUI testing gate (refined June 30)**: the gate is on *final sign-off*, not on
  doing the work. F-5's refactor was completed in-sandbox (all 5 god-files split,
  statically verified); only the operator click-test confirmation is outstanding.
  F-7's unit-testable core is done+tested; F-8 is a decide-on-touch guard, not a task.
  Offscreen import + headless suite verify import-time correctness but not runtime
  method paths (no widget is constructed in tests) — hence the click-test hand-off.
- ✅ **Pre-existing test debt** (baseline, not from this sweep): RESOLVED (June 30).
  `test_core_convergence` DK-title fail was stale test data (fixed); `test_analysis_review_tab`
  Qt abort now subprocess-probe-skipped in headless (runs on a real display); dead
  `src/core/tests/test_suggesters.py` removed. `pytest tests/` → 836 passed, 10 skipped,
  0 failed (no `--ignore`).
- Out of scope here (own efforts): `pipeline_utils` was split but not further
  decomposed; `llm_service` live generators are genuinely provider-specific (no
  further safe dedup).

## Suggested order
1. ✅ F-1 done (caches purged); F-2 done (core split confirmed correct + detail fix
   applied July 1 — `display_protocol` now uses stderr for errors like its siblings).
2. ✅ F-3 + F-4 done (plugin system + count bug, June 29) — the headline.
3. ✅ Pre-existing test-debt triage done (June 30); dead `test_suggesters.py` removed.
4. ✅ F-6 done (June 30) — webapp `app.py` 2537→240, sandbox-verified; re-confirmed
   clean July 1 (bytecode `LOAD_GLOBAL` scan, 32 webapp tests green).
5. ✅ F-5 code-complete (June 30) — all 5 god-files split, statically verified;
   **July 1 full-sweep recheck found + fixed one real bug** (missing
   `TaskModelSelectionDialog` import, see F-5 row) that the original AST scan missed.
   Operator click-test sign-off is the only outstanding F-5 work.
6. F-7 unit-core done+tested; remainder (loader wiring click-test + opportunistic
   worker conversions) GUI-gated. F-8 decide-on-touch only. **← next: operator click-test**
7. ✅ F-9 done (July 1) — pre-existing dead-but-live validation block in
   `pipeline_config_dialog` deleted; "Save as Provider Preferences" no longer
   silently fails on the normal case.

## Recheck log
- **July 1, 2026** — full reconciliation of every closed/in-progress finding (F-1
  through F-8) against current code, not just trusting commit messages: re-ran
  `py_compile` + bytecode `LOAD_GLOBAL` scans + targeted test subsets + full suite
  for each. F-1/F-3/F-4/F-6/F-7 confirmed exactly as documented. F-5 confirmed
  code-complete (register had gone stale, still listing `pipeline_tab`/`main_window`
  as "remaining" after they were already split) **and** the deeper bytecode scan
  caught one real regression (missing import, fixed same session — see F-5 row).
  F-2 closure verified mostly right but incomplete (one function's error-printing
  inconsistent with its siblings — see F-2 row). One unrelated pre-existing bug
  (F-9) surfaced incidentally while reading `pipeline_config_dialog.py`.
