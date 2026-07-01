# WP: Raw-First Response Cache (Fetch ≠ Transform)

> **Status:** 🚧 Design (July 1, 2026). Operator decision: **raw-first
> re-architecture** — cache the source response verbatim, derive everything else on
> read. Phased + facade-preserving (suite green per phase), like the plugin work.

## Problem

The search cache is **mapping-first**: it stores the already-*transformed* result.
`search_mappings(term, source) → {found_gnd_ids, gnd_counts}` — the raw API response
(lobid `member` records, finc `raw`, full JSON-LD, `totalItems`, …) is computed away
**before** caching. So on a cache hit only the reduced `{count,gndid,ddc,dk}` view
exists; agent tools can never see the full data (WP: tool data passthrough).

**Root cause is deeper than the cache:** the *suggesters transform on fetch*.
`LobidSuggester._get_results` parses the raw JSON straight into the reduced dict and
drops `member`. So "cache raw" requires **separating fetch from transform**.

## Target architecture

```
fetch(source, query, params)  → RAW response (verbatim JSON)   ── cached here ──┐
                                                                                │
read/transform(raw, view)  →  { pool view:  {count,gndid,ddc,dk}   (unchanged)  │
                                agent view:  full response (member, raw, …) }  ◄─┘
```

- **`search_response_cache`** (new table): `(source, normalized_query, params_hash)` →
  `raw_json`, `last_updated`, `http_status`. One cache for all GND-keyword sources
  (lobid/swb/catalog) — and extensible to finc/title searches.
- **Fetcher layer**: each source exposes a `fetch(query, **params) -> raw` that does
  *only* I/O (no transform). Today's suggester parsing becomes `transform(raw) -> view`.
- **Transform-on-read**: the reduced pool view (`from_gnd_keywords`, F-4 `display_count`)
  and the agent's full view are both derived from the cached raw.
- **`gnd_entries` + `search_mappings` become derived views** over the raw cache (a fast
  index for the pool), not the source of truth.

## Per-tool optional caching (operator requirement)

Caching is **opt-in per tool/source**, not global. A tool declares whether its
responses are cached:
- Search: `ProviderToolSpec.cached` already exists (lobid/swb use it; catalog is
  `cached=False`) — extend it to gate the raw cache too.
- Input: add a `cacheable` flag to `InputToolSpec` (e.g. DOI metadata cacheable; a
  live web scrape maybe not).
- Optionally per-*instance* override in `PluginInstanceConfig.settings` (an operator can
  disable caching for one endpoint) + a global `SystemConfig.enable_response_cache`
  master switch.

Non-cacheable tools always fetch live and never write the raw cache. This keeps
volatile sources fresh and lets the operator decide per source.

## Migration phases (green after each)

- **P1 — Raw cache infra (additive).** Add `search_response_cache` + UKM
  `store_raw_response` / `get_raw_response`. `CachingProvider` (or the suggester) writes
  the raw response on every live call **in addition to** the existing mapping (dual
  write). Nothing reads it yet. Pool path untouched. *Immediately preserves full data.*
- **P2 — Fetch/transform split for lobid.** Give `LobidSuggester` a `fetch()` (raw) +
  `transform(raw)` (current parsing). Cache raw in P1's table; transform-on-read.
  Surface `member`/`totalItems` in `search_lobid` from the raw cache. Verify the reduced
  view is byte-identical to today (regression gate).
- **P3 — swb + catalog** same split; then **finc/title** searches join the raw cache.
- **P4 — Derive mapping-first from raw.** `search_mappings`/`gnd_entries` become a
  derived index rebuilt from the raw cache (single source of truth). Retire the
  pre-transform write. `get_search_cache`/`store_search_result`/DK-cache re-pointed.
- **P5 — Cleanup.** Remove dead pre-transform paths; docs.

## Risks / caveats (why phased, not big-bang)

- **Load-bearing:** mapping-first backs the pool, `gnd_local`, F-3/F-4 convergence, the
  DK cache, and the `get_search_cache`/`store_search_result` tools. Each must keep
  working through every phase (facade discipline).
- **Byte-identical reduced view:** the pool's `count` drives `selection_chunks` (the
  count-landmine); the transform-on-read must reproduce today's reduced view exactly —
  a regression test capturing current output *before* P2 is mandatory.
- **Storage:** raw responses are larger than mappings; add TTL + size caps.
- **Cache key:** normalized query + params (search_type, max_pages, facets) must be in
  the key, or different queries collide.
- **The `CachingProvider` operates on `ProviderResult` (already transformed)** — the raw
  must be captured *below* it (suggester/fetcher), not at the provider wrapper.

## Verification

- Per phase: `pytest tests/` green (esp. `test_cache.py`, `test_search.py`,
  `test_providers.py`).
- P2 gate: reduced `search_lobid` output diff-free vs a snapshot taken before the split;
  `member`/`totalItems` now present in the agent view.
- Pipeline parity (classic + agentic) unchanged; local comparison harness.
