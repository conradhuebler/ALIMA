# Cleanup Findings Register (project-wide)

> Living register of technical-debt findings from the June 2026 core→subsystem
> debt sweep (pipeline, agentic, CLI, webapp, UI). Each open item has a clear
> recommendation so it can be picked up and decided deliberately. Done items are
> listed for context; details live in `AIChangelog.md` + the linked specs.

## Plugin-System-Refactor Debt-Register (D-1 … D-13, July 1 2026)

Debts found while building the generic plugin system (framework + Search + Input
categories). Most are *fixed* by that work; a few remain open. Spec:
[`docs/plugin_system.md`](plugin_system.md), changelog: `AIChangelog.md` (July 1).

- **D-1 ✅** `MetaSuggester.__init__` hardcoded a static catalog-config dict for all
  providers (finc keys absent) → now built via `src/core/search/factory.py`.
- **D-2 ✅ (partial)** finc special-cased in `tool_registry._init_suggesters`
  (`self._finc` + `_handle_search_finc`) → per-instance tool generation added;
  primary-instance path still uses the hand-written finc handler (kept for its
  availability/web_url shaping — additional instances use the factory handler).
- **D-3 ✅** stale `SearchProviderConfig` docstring updated (endpoints now in instances).
- **D-4 ✅** provider config wired by hand at ≥3 sites → single `build_provider` factory.
- **D-5 ✅ (done in WP Plugin-Konvergenz P4, July 16)** `CatalogConfig.get_catalog_type()` was
  wrongly recorded as "superseded" in June — it was still the *live* DK-backend selector.
  **P4 fixed it for real:** `resolve_dk_extractor` now reads every backend from its instance
  settings; `sru` got its own `dk_enabled` ConfigField (symmetric to finc). `catalog_type` is
  vestigial — the resolver reads it only as a *transition fallback* (via `_sru_selected_for_dk`)
  so configs migrated before `sru.dk_enabled` existed keep their SRU backend. `get_catalog_type()`
  (config_models.py:800) is now **dead** (its only caller, the old `pipeline_utils` `'auto'`
  resolution, is gone) → delete with the mirror in **P7**. The `catalog_type` ConfigField +
  fallback read also go in P7 once operators have migrated to `sru.dk_enabled`.
- **D-6 ✅** `save_config` `preserve_unified` merge extended to carry `plugins` +
  `approved_plugins` (derive-on-save keeps the mirrors exact; round-trip test gates it).
- **D-7 (open, security)** API keys/tokens still plaintext in `config.json`; plugin
  settings add more secrets. `ConfigField(secret=True)` is the anchor for a future
  keyring backend — not done here.
- **D-8 (open, transitional)** `CatalogConfig`/`SystemConfig` are now *derived mirrors*;
  the legacy readers should migrate to the factory/instances over time, then the
  mirrors can be dropped. ⚠️ **The "~298" figure was wrong (recounted July 16): 53 actual
  attribute reads across 10 files.** 302 counts field-name *string occurrences* — incl.
  `resolve_dk_extractor`'s 15 kwargs (0 reads), `marcxml_client`'s constructor params
  (0 reads), `ConfigField` key declarations (the *replacement*), and ~84 wizard **writes**.
  The dominant cluster is `execute_dk_search` (18) — which **P4 eats**, leaving ~35.
  `SearchProviderConfig`: 5 production sites. The inflated number is what made this read
  as un-attemptable; scheduled as **WP Plugin-Konvergenz P7** (gated on P2 + P4).
- **D-9 ✅** URL scraper was inline in `batch_processor` → extracted to
  `input_sources/url_fetch.py` (`scrape_url`); batch now delegates.
- **D-10 ✅** monolithic `UnifiedResolver` + `SystemConfig` DOI flags → three separately
  configurable input-source plugins (`doi_crossref/openalex/datacite`).
- **D-11 ✅** `execute_input_extraction` if/elif → `INPUT_SOURCE_REGISTRY` dispatch.
- **D-12 (open)** parallel DOI/metadata paths (`doi_resolver`, `crossref_worker`,
  `k10plus_resolver`) — consolidation not attempted; noted only.
- **D-13 (open)** `UnifiedResolver` also does Springer/generic URL crawling (overlaps
  `url_fetch`); left in the DOI facade for now, consolidate onto `url_fetch` later.

---

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
| F-10 | search/data | `LobidSuggester` had **two** GND-authority data paths next to `gnd_local.db`: it downloaded+parsed the 25 MB DNB JSON-LD dump **in `__init__`** (`prepare(False)`) on every first provider build, historically into script-relative dirs (triplicated `src/{,cli/,webapp/}data/` copies), later into tempdir (re-download after every temp clean). | ✅ **Partly DONE (July 19, `3e4d0cd`):** construction is I/O-free (lazy memoised load on first `transform`), default dir now persistent `~/.config/alima/suggesters/<name>/`, stale copies deleted. **Mid-term (open, decide-on-touch):** serve the GND-ID→label lookup from the `gnd_local` store and drop the JSON-LD dump entirely — needs a check which label fields the dump has that `gnd_local.db` lacks. | low |
| F-11 | project-wide | Compat re-export shims accumulate after each module split (`alima_cli.py` wrapper, `pipeline_utils` facade, webapp `app` re-exports; the `clients/finc_*` pair is deleted). Without a policy they live forever. | **Policy (July 19):** internal call sites always import the canonical path; shims exist only for external/test compatibility, carry a dated note, and get deleted once `grep` shows no importers (as done for `clients/finc_*` in `04d9eb0`). Check on touch, not as a sweep. | none |

---

### P1 — Neue Befunde (July 20–21, 2026)

Diese drei standen in **keinem** Registereintrag und sind nicht durch Codelesen
aufgefallen, sondern dadurch, dass der Operator die Pipeline laufen ließ und
danach in die Datenbank geschaut wurde.

| # | Bereich | Befund | Status |
|---|---|---|---|
| F-12 | core | **Aufrufe nicht existierender Methoden.** `update_gnd_entry` (2 GUI-Stellen), `gnd_keyword_exists` (`search_cmd`), `get_all_gnd_ids_for_keyword` (`_validate_catalog_subjects`). Wurzel: eine halbe „CacheManager compatibility"-Fassade auf `UnifiedKnowledgeManager` — weil die Hälfte existierte, wirkte der Rest plausibel. Zwei der drei waren ungeschützt, einer wurde von einem breiten `except` geschluckt. | ✅ **DONE** (`a3bb317`): alle drei behoben, tote Fassade (−216 Z., 4 Methoden ohne Aufrufer) gelöscht. AST-Scan über die zwei zentralen Manager findet danach **0** Fälle. ⚠️ Der Scan deckt nur statisch auflösbare Ziele ab — Provider/Suggester/LLM-Backends nicht. |
| F-13 | tests | **Kernlogik ohne jede Testabdeckung.** 84 Module ohne Test-Erwähnung; drei davon sind reine Logik, kein GUI: `_pipeline_rvk_scoring` (1992 Z.), `_pipeline_dk_steps` (1186), `batch_processor` (895). **Der Batch-Save-Crash lag genau in dem Modul, das die Suite nie berührt.** Der einzige Test, der RVK-Scoring nennt, mockt es weg. | ✅ **Teilweise** (`11f77c8`, `06c4417`): 84 Charakterisierungstests für LLM-Antwort-Extraktion, Batch-Parsing/Naming/OCR-Heuristik und die RVK-Scoring-Primitive. **Offen:** die großen Entscheidungsmethoden (`_select_final_rvk_candidates` 209 Z., `_validate_catalog_rvk_candidates` 394 Z.). |
| F-14 | utils | **Untestbar konstruiert.** In `_pipeline_rvk_scoring` waren 19 von 40 Funktionen in Methodenkörper verschachtelt, u.a. die Hierarchie-Helfer; `_source_rank`/`_status_rank` existierten byte-identisch **doppelt**. | ✅ **Teilweise** (`06c4417`): 5 reine Helfer verbatim auf Modulebene (Opcode-Vergleich + `LOAD_GLOBAL`-Scan), Duplikate zusammengeführt, Test gegen Wieder-Einnistung. **Offen:** 14 verschachtelte Funktionen mit Closure-Bindung (`_validate_code` 160 Z., `_prefilter_candidates`, `_pick_diverse`) — keine reine Verschiebung mehr. |

**Offen als Politikfrage (B):** 336 breite `except`-Blöcke schlucken still (59 nur
`pass`, 277 nur Debug/Warning). Vier der sieben Defekte dieser Runde lebten davon,
dass ein `except Exception` *Programmierfehler* (`AttributeError`, `ImportError`)
wie erwartbare Laufzeitfehler behandelte. Die Frage ist nicht „alle anfassen",
sondern ob ein breiter `except` Programmierfehler mitfangen darf.
✅ **Politik entschieden (July 21, `8e4a80a`):** `error_visibility.log_caught`
loggt Defekt-Formen auf ERROR, ohne den Kontrollfluss zu ändern; in den
Implementation Standards (CLAUDE.md). Übernommen 9/189 (reiner-Code-Blöcke),
Rest bei Berührung.

### F-15 · Core-God-Files (D) — July 21, 2026

F-5 war die UI-Seite (5 Dateien, alle gesplittet). **F-15 ist die Core-Seite**,
mit derselben Mixin-Technik (verbatim, via MRO, null Aufrufstellen) und
derselben Verifikation (Opcode-Vergleich gegen HEADs echte Datei +
`LOAD_GLOBAL`-Scan — [[module-split-verification]]).

| Datei | vorher → jetzt | Mixins | Commit |
|---|---|---|---|
| `unified_knowledge_manager.py` | 2077 → 1460 | `_ukm_schema.SchemaMigrationMixin` (350 Z.), `_ukm_catalog_dk.CatalogDkCacheMixin` (249 Z.) | `8d75d6c` |
| `tool_registry.py` | 1980 → 1381 | `_tool_generation.ToolGenerationMixin` (17 Fabrik-Methoden) | `0dcb04c` |
| `biblio_client.py` | 2106 → 1642 | `_biblio_parsing` (Parser, 347 Z.) + `_biblio_transport` (Reliability, 182 Z.) | `54a5fd6`, `5aa7bda` |
| `pipeline_manager.py` | 2437 → 1690 (Klasse 1955 → 1230) | `_pipeline_classic_steps.ClassicStepExecutorMixin` (10 klassische Step-Executoren, 728 Z.) | `4895a47` |

**Der `LOAD_GLOBAL`-Scan hat sich erneut bezahlt gemacht** (wie bei F-5): beim
`tool_registry`-Split fehlte `logger` im neuen Modul — benutzt in 5 Fehlerpfaden,
also ein `NameError`, den Import + grüne Suite NICHT gefangen hätten (der Name
fällt erst im `except` an). Beim `biblio_client` hatte der Logger zusätzlich
einen nicht-`__name__`-Namen (`"biblio_extractor"`) — ein stummer Fehl-Logger
statt Absturz, wenn man ihn nicht exakt übernimmt. Beim `pipeline_manager` war
der Scan sauber, weil der Block nur über `self.logger` loggt — **geprüft, nicht
angenommen** (die Annahme „hier gibt's keinen Modul-Logger" ist genau die
teure). Deshalb neu in der Methodik: State-teilende Mixins (Circuit-Breaker) und
reparierte/verlagerte Pfade werden **real getrieben**, nicht nur per Opcode
verglichen. Import-Zyklus-Falle (`pipeline_manager`): eine im Zielmodul nur als
**Typannotation** benutzte Herkunftsklasse (`PipelineStep`) braucht keinen
Laufzeit-Import, wenn das Mixin `from __future__ import annotations` trägt.

**Vorbelegt / nicht angefasst:** `llm_service.py` (3069) an WP-T1 gekoppelt
(Gemini-Streaming — „nicht vorher, nicht getrennt"); `pipeline_utils.py` (2049)
bewusst gestoppt (Cross-cutting-Notiz); `_pipeline_rvk_scoring.py` (1992) hat 14
Closure-gebundene Funktionen (kein reiner Move — F-14). **Die frei zerlegbare
God-File-Spitze ist damit abgetragen** (kein zerlegbarer Core-God-File mehr über
~1690). Reste an bereits gesplitteten Dateien (opportunistisch, nicht dringend):
die SOAP/Web-Achse in `biblio_client` (~600 Z.), Smart-Search/Raw-Cache im UKM
(~475 Z.), die agentische Workflow-Achse in `pipeline_manager`.

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
