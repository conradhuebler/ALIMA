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
| F-2 | CLI | Mixed stdout: `protocol_formatters` uses bare `print()`, handlers use `print_result` | ✅ CLOSED — non-issue (verified): `print_result` always writes stdout and only *adds* logging; the formatters' bare `print()` is display/export output (correct), errors already go to stderr. Routing them through `print_result` would only add log noise. No change. | — |

### P2 — Major workstream: Search-Provider-Plugins (+ coupled bug)
| ID | Area | Finding | Action | Risk |
|----|------|---------|--------|------|
| F-3 | core/suggesters | DB/API search sources were ad-hoc; finc bypassed the suggester abstraction; `SuggesterType` enum + per-frontend wiring | ✅ DONE (June 29, commits `5061801`/`35bc487`/`85900e2` + GUI): capability-based `SearchProvider` standard + single `@register_provider` registry (`src/core/search/`); `SuggesterType` retired; caching as `CachingProvider` wrapper; finc folded into the standard; MCP search tools generated from `ProviderToolSpec`; `SearchProviderConfig` + GUI selector. Spec: [`search_provider_plugins.md`](search_provider_plugins.md) | high (touches search core) |
| F-4 | agentic | "Häufigkeit zeigt 1": mapping-cache hits got `count=1`; agentic GND counts looked wrong | ✅ DONE (June 29, with F-3 P2): cache now stores per-GND-ID `gnd_counts`; cache hits keep pool `count=1` (ranking unchanged) and restore a display-only `display_count` read by `flatten_gnd_hits`/GUI/agentic. `rank_pool` never reads it. Open: final agentic-run confirmation (needs LLM). | medium |

F-3/F-4 delivered as one workstream (P1+P2+P3). The MCP auto-generation was proven
byte-identical to the former hand-written handlers; the GUI provider-selector needs
an operator click-test (sandbox-untestable, per the GUI gate below).

### P3 — Structural refactors (high value, GUI-test-gated)
**All blocked on an operator GUI click-test loop — not safely doable blind.**
| ID | Area | Finding | Action | Risk |
|----|------|---------|--------|------|
| F-5 | UI | God-files. **In progress (June 30):** ✅ `unified_provider_tab` 2115→681 (→ `provider_dialogs.py` 540 + `task_preferences_widget.py` 932); ✅ `pipeline_config_dialog` 2226→765 (→ `step_config_widgets.py` 1487). **Remaining:** `pipeline_tab` 2882, `main_window` 2650, `pipeline_chat_panel` 1922 (+ the 3 new modules are still large but cohesive). | Extract cohesive widgets/sections; one file at a time + operator click-test (verbatim moves verified by AST scan + construct-smoke + green suite). | high (Qt, untestable here) |
| F-6 | webapp | ✅ **DONE (June 30)** — `app.py` 2537→240 LoC (−90%). | Split into thin app factory + `session_state.py`/`render_bridge.py`/`session_io.py` + `routers/{workflows,models,sessions,export,websocket,analysis,agent}.py`; re-export shims keep `src.webapp.app` test contract; route table byte-identical; suite green (841). Sandbox-verified (no GUI gate). | done |
| F-7 | UI | 17 `QThread`/`StoppableWorker` subclasses; base only partly adopted; 2 model-loaders overlap (`_ModelLoadWorker` vs `ModelFetchWorker`) | **Partly done (June 30):** model-loaders merged into shared `ModelLoadWorker(StoppableWorker)` in `workers.py` (signal contracts preserved; unit-tested `tests/test_model_load_worker.py`) — GUI wiring needs a click-test. **Blanket base-adoption descoped:** most remaining plain-`QThread` workers are short probes that don't need cancellation; convert only genuinely-cancellable long workers case-by-case (GUI-gated). | medium (threading) |

### P4 — Decide-on-touch (behavior changes, not pure dedup)
| ID | Area | Finding | Action | Risk |
|----|------|---------|--------|------|
| F-8 | UI | `ProviderModelSelector` non-adopters have *intentional* differences (see `ui/CLAUDE.md`): `comprehensive_settings` (common+enabled provider list), `abstract_tab` (pushed `available_models` cache + async-incompatible restore), `pipeline_config_dialog` (per-step grid) | Only when already editing those tabs, with operator OK on the UX change + click-test | medium-high |

---

## Cross-cutting / meta
- **GUI testing gate**: F-5/F-7/F-8 are GUI-heavy and cannot be runtime-verified (F-6 was sandbox-verifiable — done)
  in the dev sandbox (offscreen import only). They need an operator click-test loop;
  do them one unit at a time with a hand-off, not in big blind batches.
- ✅ **Pre-existing test debt** (baseline, not from this sweep): RESOLVED (June 30).
  `test_core_convergence` DK-title fail was stale test data (fixed); `test_analysis_review_tab`
  Qt abort now subprocess-probe-skipped in headless (runs on a real display); dead
  `src/core/tests/test_suggesters.py` removed. `pytest tests/` → 836 passed, 10 skipped,
  0 failed (no `--ignore`).
- Out of scope here (own efforts): `pipeline_utils` was split but not further
  decomposed; `llm_service` live generators are genuinely provider-specific (no
  further safe dedup).

## Suggested order
1. ✅ F-1 done (caches purged); F-2 closed (verified non-issue).
2. ✅ F-3 + F-4 done (plugin system + count bug, June 29) — the headline.
3. ✅ Pre-existing test-debt triage done (June 30); dead `test_suggesters.py` removed.
4. ✅ F-6 done (June 30) — webapp `app.py` 2537→240, sandbox-verified.
5. F-5/F-7 once a GUI test loop exists; F-8 opportunistically. **← next**
