# Session Handoff — WP2 Raw-First Response Cache (done) + follow-ups

> For a fresh session with no prior context. Refreshed July 2, 2026. Branch `agent`.
> Read this + `docs/wp_raw_response_cache.md` to resume.

## Where we are

- **Branch `agent`, HEAD `0225912`.** Suite: **962 passed, 10 skipped, 0 failed**
  (`.venv/bin/python -m pytest tests/`).
- **Working tree clean** except untracked local `.claude/` (never commit) and this doc.
- WP2 (Raw-First Response Cache) is **implemented P1–P5** and committed (13 commits,
  `330a8c1` → `0225912`), plus a config/GUI toggle and two follow-up fixes.

## What shipped this session

**WP2 — cache the source response verbatim, derive the pool on read.** Full detail:
`docs/wp_raw_response_cache.md` + `AIChangelog.md` (July 2 entry).

- **P1** `search_response_cache` table + UKM `params_hash`/`store_raw_response`/
  `get_raw_response` (1 MB size cap, 5000 soft row cap, 24 h TTL); dual-write at the one
  shared fetch seam `SuggesterBackedProvider._gnd_search` + `factory.build_provider`
  injection; `SystemConfig.enable_response_cache` master switch (+ per-instance
  `settings['cache_responses']`).
- **P2** lobid `fetch()`/`transform()`/`transform_agent_view()`; `search_lobid` gains an
  additive `agent_view` (member/totalItems) via transform-on-read.
- **P3** swb (page HTML) + catalog (parsed records) capture `last_raw`.
- **P4** `aggregate_gnd_results` engine (`src/core/search/aggregate.py`) + MCP tool:
  counter (`display_count`) + provenance (`sources`/`source_count`) from raw,
  **raw-first with mapping fallback**. Both pipelines converged onto it —
  agentic `gnd_batch_search` and classic `execute_gnd_search` → `SearchCLI.search_from_raw`.
  Count-landmine kept (pool count = 1, real count in `display_count`).
- **P5** finc + catalog-title record raw capture; `InputToolSpec.cacheable` + DOI
  read-through cache; docs.
- **Toggle:** `SystemConfig.aggregate_from_raw` (default True) + resolver
  `aggregate.default_aggregate_from_raw()`; GUI checkboxes in Settings → System
  ("Aggregate" + "Response-Cache"). Rollback the convergence with **no code change**.

**Follow-up fixes (this session, unrelated to each other):**
- `clear_search_cache` now clears **both** `search_mappings` **and**
  `search_response_cache` (else the raw-derived pool survives a "clear"), and dropped a
  bogus `commit_transaction()` (QtSql is autocommit → it raised "no transaction is
  active"). `clear_database` also wipes raw + `catalog_dk_cache`.
- **seed=0 → None** at both `LlmService` entry points: the classic path defaulted
  `seed=0` and sent it; Mistral rejects `seed` (HTTP 422) → classic pipeline aborted at
  the first LLM call while the agent (seed=None) worked. Now aligned.

## Verified vs. not

- **Verified:** full suite green; `clear_search_cache` empties both caches; seed=0
  normalized on both paths; the aggregate engine's counter/provenance + count-landmine
  + mapping fallback (unit tests). Operator ran classic + agentic end-to-end without
  errors.
- **NOT verified (operator to do):** an **A/B comparison lauf** — same abstract with the
  "Aggregate" toggle off (mapping-first) vs on (raw-first), comparing the GND pool +
  selected keywords. "Ran through" ≠ "results equivalent/better". This is the check that
  would validate the P4 convergence.

## Known edge-case risks (follow-up WP only if leaning on the raw path hard)

- **Params brittleness:** raw is keyed by the default `search_type="kw"`/`max_pages=5`;
  a non-default search stores raw under different params than the reader looks up → raw
  miss → mapping fallback (no data loss, but the raw path is effectively default-only).
- **swb size-cap fallback:** swb pages > 1 MB skip the raw write; the mapping fallback
  then needs `gnd_entries` facts for the gndids — swb doesn't store facts, so such a
  term can still yield nothing if no other source stored its facts.
- **Raw-vs-fallback titles:** raw titles come from the suggester transform, fallback
  titles from `gnd_entries.title`; if they ever differ the pool won't merge them
  (possible duplicate entry). Not observed, not verified.

## Gotchas / conventions

- **Rollback:** untick "Aggregate" (or `SystemConfig.aggregate_from_raw=False`,
  or per-workflow step `config['aggregate_from_raw']=False`).
- **Count semantics changed on the converged path:** pool `count` is always 1 (real
  count in `display_count`) — consistent with the cached path, different from old
  first-fetch. Feeds agentic selection ordering.
- Test runner: `.venv/bin/python -m pytest tests/` (system python has no pytest).
- Commit discipline (operator): explicit permission before every `git commit`;
  `git add <files>` not `-A`; never stage `.claude/`; end messages with the
  `Co-Authored-By: Claude Opus 4.8` line.
