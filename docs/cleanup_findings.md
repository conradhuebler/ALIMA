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
| F-5 | UI | God-files: `pipeline_tab` 2882, `main_window` 2650, `pipeline_config_dialog` 2177, `unified_provider_tab` 2111, `pipeline_chat_panel` 1922 | Extract cohesive widgets/sections; one file at a time + click-test | high (Qt, untestable here) |
| F-6 | webapp | `app.py` 2539 LoC god-file (~25 routes + session/streaming/agent) | `APIRouter` split (sessions/streaming/agent/workflows/export) | med-high (shared `app`+state) |
| F-7 | UI | 17 `QThread`/`StoppableWorker` subclasses; base only partly adopted; 2 model-loaders overlap (`_ModelLoadWorker` vs `ModelFetchWorker`) | Adopt `StoppableWorker` base consistently; merge model-loaders | medium (threading) |

### P4 — Decide-on-touch (behavior changes, not pure dedup)
| ID | Area | Finding | Action | Risk |
|----|------|---------|--------|------|
| F-8 | UI | `ProviderModelSelector` non-adopters have *intentional* differences (see `ui/CLAUDE.md`): `comprehensive_settings` (common+enabled provider list), `abstract_tab` (pushed `available_models` cache + async-incompatible restore), `pipeline_config_dialog` (per-step grid) | Only when already editing those tabs, with operator OK on the UX change + click-test | medium-high |

---

## Cross-cutting / meta
- **GUI testing gate**: F-5/F-6/F-7/F-8 are GUI-heavy and cannot be runtime-verified
  in the dev sandbox (offscreen import only). They need an operator click-test loop;
  do them one unit at a time with a hand-off, not in big blind batches.
- **Pre-existing test debt** (baseline, not from this sweep): 2 known fails —
  `test_analysis_review_tab` (Qt abort) + `test_core_convergence::test_fallback_when_rich_has_no_titles`
  (DK-title). Triage: fix or quarantine so the suite is meaningfully green.
- Out of scope here (own efforts): `pipeline_utils` was split but not further
  decomposed; `llm_service` live generators are genuinely provider-specific (no
  further safe dedup).

## Suggested order
1. ✅ F-1 done (caches purged); F-2 closed (verified non-issue).
2. ✅ F-3 + F-4 done (plugin system + count bug, June 29) — the headline.
3. Pre-existing test-debt triage (small, unblocks confidence). NOTE: dead
   `src/core/tests/test_suggesters.py` (stale `core/suggesters` layout, never
   collected) surfaced during F-3 — candidate for deletion.
4. F-5/F-6/F-7 once a GUI test loop exists; F-8 opportunistically.
