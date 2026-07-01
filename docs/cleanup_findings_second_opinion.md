# Cleanup Findings — Second Opinion (independent debt audit)

> Independent second-opinion technical-debt audit (July 1, 2026), requested by the
> operator alongside the existing register [`cleanup_findings.md`](cleanup_findings.md)
> (F-1…F-9). Every finding below is cited with `file:line` and was spot-verified
> against the current code, not taken from the prior sweep.

## Framing — what the first sweep did well, and its blind spots
The prior register is a competent **code-structure** sweep (GUI god-file splits,
search-provider plugin system, worker consolidation, two concrete bug fixes). Its
claims largely hold up: the frontend dedup of `workflow_loader.discover_workflow_files`
and `result_serialization` is real, and the F-5 splits are done.

What it **systematically under-weighted or omitted** falls on four axes, addressed here:
1. **Operations & hygiene** (logging, git-tracked binaries, packaging) — barely touched.
2. **Correctness/robustness as a *systemic* pattern** — F-9 was one swallowed error; there are ~38.
3. **The large backend files it explicitly *descoped*** — `pipeline_utils`, `llm_service`, etc.
4. **Drift in its own docs + test gaps** — one live audit doc is now factually wrong.

IDs are prefixed **S-** to avoid collision with the F-series.

---

## Axis 1 — Operations & Hygiene

| ID | Sev | Finding (evidence) | Recommendation |
|----|-----|--------------------|----------------|
| **S-1** | CRITICAL | **Unbounded logging.** `utils/logging_utils.py:69` used `logging.FileHandler` (no rotation) → `alima.log` **3.4 GB**, `gnd_fetcher.log` **244 MB**, written to repo CWD. 4 stray `logging.basicConfig` at import (`core/dnb_utils.py`, `utils/print_abstracts.py`, `utils/qt_plugin_setup.py:205`, `utils/clients/marcxml_client.py:25`); 2 extra unbounded `FileHandler`s in `biblio_client.py:48,54`. | ✅ **DONE this session** (see below). |
| **S-2** | CRITICAL | **34 MB binary DB tracked in git.** `search_cache.db` (SQLite) + `blob/prompts.json` were committed; `.git` is 90 MB. `.gitignore` rules (`*.db`:253, `blob/`:262) were ineffective (added after the commit). | ✅ **`git rm --cached` DONE** (forward-untrack). ⚠️ **History rewrite** (`git filter-repo` to shrink `.git`) is a **separate operator decision** — not done. |
| **S-3** | MEDIUM | **~3.7 GB untracked root clutter.** JSON exports (`*_text_*.json`, `analysis_export_*`, `test_*.json`), DB backups (`*.db.backup*`), prompt variants (`prompts~.json`, `prompts.tmp.json`, `prompt2.jsonn`, `prompts.json.pre-*`). | Operator cleanup (no code). Add export-name patterns to `.gitignore`. |
| **S-4** | MEDIUM | **Config-file zoo.** `llmachine.json` (386 K, legacy), `migrated_llm_config.json`, `config.json.{backup,ui,bak}`, `alima_presets*.json` ×5. `ConfigManager` is the single loader (good) — the files obscure what's active. | Delete legacy/backup configs; keep `config.json` + one `.example`. |
| **S-5** | HIGH | **No packaging / dependency strategy.** No `pyproject.toml`/`setup.py`, no `console_scripts` → CLI/webapp launch undocumented (README shows only GUI). `requirements.txt` is a full `pip freeze` (direct+transitive, hard-pinned); `google.genai` (line 30) was unpinned **and** mis-named. | ✅ **Pin DONE** (`google-genai==2.8.0`). Add `pyproject.toml` with entry-points + split direct/transitive deps → roadmap. |

## Axis 2 — Correctness & Robustness

| ID | Sev | Finding (evidence) | Recommendation |
|----|-----|--------------------|----------------|
| **S-6** | HIGH | **Silent exception swallowing is systemic, not a one-off.** 768 `except Exception` + 3 bare `except:` across `src/`+`webapp/`; ~38 swallow with `pass`, only ~7 % use `logger.exception`. F-9 fixed one instance; the pattern remains. Worst (data/config path): `webapp/routers/analysis.py:354,374,387,521` (drop DK/classification), `webapp/routers/agent.py:152,328,463,536`, `webapp/render_bridge.py:77,145,182`, bare `except:` in `core/alima_manager.py:818,833` (→ `localhost` fallback masks provider misconfig), `ui/batch_processing_dialog.py:1209` (→ empty `PipelineConfig()`). | **Not** a blanket rewrite. Target the ~38 silent swallows on data/config paths: add `logger.exception` + surface/re-raise where a failure should be visible. Behavior-adjacent → own effort, click-test webapp paths. |

## Axis 3 — Backend god-files (descoped by the first sweep)

| ID | Sev | Finding (evidence) | Recommendation |
|----|-----|--------------------|----------------|
| **S-7** | HIGH | `pipeline_utils.py` (5098) — 8 domains in one class; `execute_dk_search` **558 lines** (:4398-4956); RVK scoring (~:3031-3530, ~500 lines) + 15+ magic numbers (:3487-3516). | Extract `RvkScoringService` + `RvkScoringConfig` (magic numbers); then `CatalogSearchManager`. Verbatim-move discipline ([[module-split-verification]]). |
| **S-8** | HIGH | `llm_service.py` (3057) — 5 near-identical `_generate_*_with_tools` (~80 % similar, :2076-2981); **hardcoded model names/reasoning-prefixes** (:1344, :1461, :1533, :2823) violate the project rule "read from llmanager"; `# TODO Altlast` :1348. | Extract shared tool-dispatch scaffold; move model/prefix lists into provider config. |
| **S-9** | MEDIUM | `pipeline_manager.py` (2456) — two `stream_callback` implementations (:1367-1485 = 119 L, :1564-1749 = 186 L). | Unify into one callback factory. |
| **S-10** | MEDIUM | `biblio_client.py` (2369) — dead `extract_dk_classifications_for_keywords_OLD` (268 L, 0 refs). | ✅ **DELETED this session.** (`get_title_details` 313 L → later.) |
| **S-11** | MEDIUM | `unified_knowledge_manager.py` (1974) — DB schema + cache + classification repo intermixed. | Split into `DatabaseLayer`/`CacheManager`/`ClassificationRepository` (not urgent). |
| **S-12** | MEDIUM | **Hardcoded provider lists** violate the project rule. `["ollama","gemini","openai","anthropic"]` in `comprehensive_settings_dialog.py:1689,1700`, `unified_input_widget.py:396-401`, `step_config_widgets.py:815-819`, `provider_dialogs.py:123`; magic-string branches in `smart_provider_selector.py:217,219,231,233,290,293`. New provider = 5+ edit sites. | Read provider set from `llmanager`/config in one place. |

## Axis 4 — Tests & Docs-drift

| ID | Sev | Finding (evidence) | Recommendation |
|----|-----|--------------------|----------------|
| **S-13** | HIGH | **4 load-bearing modules have no direct test:** `alima_manager`, `llm_service`, `unified_knowledge_manager`, `pipeline_manager` (tested only indirectly). No coverage config (`pytest-cov` absent). | Add direct unit tests + `pytest-cov` with a floor threshold. |
| **S-14** | MEDIUM | **5 orphan root `test_*.py`** (`test_grace_period/keyword_fix/model_capabilities/soap_direct/pipeline_keywords.py`) not collected (`testpaths=tests`). *(Verified NOT git-tracked — gitignored; they're local scratch.)* | Move real cases into `tests/` or delete. |
| **S-15** | MEDIUM | **Live doc factually wrong.** `docs/audit_findings.md §1` claimed `YamlPromptService` is "never imported/wired" — false: `prompt_service.py:19-24` loads `prompts.json` then merges `prompts.yaml` over it (`merge_into`). | ✅ **CORRECTED this session** (added a dated correction block). |
| **S-16** | LOW-MED | **Doc sprawl.** 4 `provider_strategy_*.md` (~1130 L) describe shipped work; 41 docs total with overlap. | Archive superseded provider docs to `docs/legacy/`. |
| **S-17** | DESIGN | **Dual prompt source of truth.** `prompts.json` + `prompts.yaml` both live (YAML overrides) → silent-drift risk; "edit both" is a workaround, not a resolution. See [[prompts_yaml_overrides_json]]. | Decide: consolidate to one source, or log/annotate overrides explicitly. |

---

## Executed this session (safe quick-wins, operator-approved)
All verified: `py_compile` clean, logging-rotation smoke OK, `marcxml`/`dnb` no longer
hijack the root logger on import, full suite **858 passed / 10 skipped / 0 failed**
([[test_runner_venv]]).

1. **S-1** — `logging_utils.py`: `FileHandler` → `RotatingFileHandler` (`max_bytes` 10 MiB,
   `backup_count` 5, both params). `biblio_client.py:48/54` handlers → rotating.
   Module-level `basicConfig` removed from `dnb_utils.py`; moved into `__main__` in
   `marcxml_client.py` + `print_abstracts.py`. (`qt_plugin_setup.py:205` was already
   `__main__`-guarded — left as-is.)
2. **S-2** — `git rm --cached search_cache.db blob/prompts.json` (files kept on disk,
   `.gitignore` already covers them). History rewrite **not** done (operator decision).
3. **S-10** — deleted dead `extract_dk_classifications_for_keywords_OLD` (268 L).
4. **S-5 (partial)** — pinned `google-genai==2.8.0`.
5. **S-15** — corrected `docs/audit_findings.md §1`.

> ⚠️ **Not yet committed** — per operator convention, commit needs explicit sign-off
> ([[feedback_commit_confirmation]]). The 3.4 GB `alima.log` on disk is not deleted
> by these changes; rotation only bounds *future* growth — the operator should
> truncate/rotate the existing file once.

## Suggested remediation order (remaining)
1. **S-6** targeted except-hardening on the webapp data/config paths (highest correctness value).
2. **S-12** provider lists from `llmanager` (unblocks clean provider addition; low risk).
3. **S-5** `pyproject.toml` + `console_scripts` + doc the CLI/webapp launch.
4. **S-7 / S-8 / S-9** god-file extractions (one unit at a time, verbatim-move verified).
5. **S-13** core-module tests + `pytest-cov`.
6. **S-2 (history rewrite)**, **S-3/S-4** hygiene, **S-16** doc archive, **S-17** prompt decision.
