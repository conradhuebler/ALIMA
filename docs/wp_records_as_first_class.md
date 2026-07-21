# WP: Records as first-class (bibliographic records flow fully through the pipeline)

> **Status:** Design / vision (July 9, 2026). Operator direction: k10plus/SRU/DOI
> sources return **rich bibliographic records**, but we exploit a thin slice (DOIs
> from k10plus, abstracts from DOIs) — "eigentlich geht mehr, wir müssen umfangreicher
> denken". Decisions: pursue **all four record-consumption paths**; **extend the
> existing plugin categories** (no new `record_source` capability). Grounded in the
> July-9 single-endpoint inventory (below) and continues Phase C.
>
> **Update (July 10, 2026):** a field-name audit (new section below, verified against
> the code) confirms the debt is real — ALIMA already ships round-trip rename shims.
> The prereq (Phase C) is committed. A **decision point for a new session** is at the
> end of this doc. The sharpest single instance of the count fragmentation is split off
> as a ready-to-execute bugfix: `docs/wp_gnd_counter_divergence.md`.
>
> **Update (July 19, 2026): P0 ✅ EXECUTED** (5 commits from `6991d57`; details
> in `AIChangelog.md`). Decisions pinned by the operator — see "Pinned
> decisions" section: canonical vocabulary `{count, gnd_ids, classifications:
> {system: codes}, display_count?}` end-to-end (suggester contract v2 →
> persisted KAS), the **generalized notation data shape pulled forward from
> WP-D2** (dk/ddc/rvk equal-rank system keys), **hard cut** — no legacy
> readers, the old key set was never used in the wild.
>
> **Update (July 20, 2026): P0 ✅ VERIFIED + `to_bibrecord()` and F-2 landed**
> (6 commits `2e2b647`…`87b6eb4`; suite 1324 → 1357). The classic↔agentic
> comparison is a **deterministic headless test**, not a live run: both paths
> ingest one fixture through real production code and must persist identical
> `{count, display_count, gnd_ids, classifications}`. Also in this round: the
> system-name vocabulary unified on **UPPERCASE** (`classification_systems.py`
> is the single owner — the data layer had drifted to lowercase), two
> pre-existing persistence bugs fixed (batch save crashed on sets; `rvk` was
> not an equal-rank system on reload), and the swb pre-v2 fallback finally
> removed. **Next: consumer paths P1–P4** — note P1's parity landmine, the
> `input_type` vocabulary drifts across GUI/CLI/webapp/batch and DOI bypasses
> `execute_input_extraction` entirely today.

## Problem

Every bibliographic source emits the *same shape* of thing — a record with
identifiers, title, authors, year, abstract, subjects, and its own classifications —
yet we built a **narrow extractor per source**:

| Source | Category today | What we keep | What we drop |
|---|---|---|---|
| k10plus (Siegel) | `lookup` (`k10plus_package`) | DOIs (CLI batch path) | title/authors/subjects/ddc/url |
| DOI APIs (crossref/openalex/datacite) | `input_source` | abstract | full metadata record |
| SRU endpoints (DNB/LoC/GBV/SWB/K10+) | `search_provider` (`sru`, no agent tool) | DK/title backend | records as data |
| finc / lobid / catalog | `search_provider` | GND-keyword pool entries | subjects / record detail |

`K10PlusRecord` already carries `ppn, doi, title, authors, year, ddc, subjects, url` —
we throw ~90% of it away in the Siegel→DOI path.

## Findings — field-name audit (July 10, 2026, verified)

A read-only audit (findings spot-checked against the code) confirms the unification
value is **not speculative**: ALIMA already ships hand-written translation shims to
bridge field-name divergence — `_entry_from_kw_data` (`src/core/gnd_search_core.py:116-128`)
renames `gndid→gnd_ids`, `ddc→ddc_codes`, `dk→dk_codes`, and `nested_from_aggregate`
(`src/core/search/aggregate.py:188-193`) renames them **back**. Round-trip renaming with
zero semantic gain = direct proof of debt.

Confirmed inconsistencies, ranked by downstream impact:

| # | Concept | Divergence | Evidence |
|---|---|---|---|
| F-1 | GND-id / DDC / DK | three names each, bridged by a live round-trip rename layer | `gnd_search_core.py:116-128` ↔ `aggregate.py:188-193` |
| F-2 | DOI record | two code paths for the SAME APIs — `Title/DOI/Abstract/Source` (capitalized) vs `source/doi/metadata` (lowercase, already flagged "D-12") | `doi_resolver.py:258-262` vs `input_sources/doi.py:142,160` |
| F-3 | authors | type-polymorphic under one name: dict (finc) / list (k10plus, catalog) / joined str (DOI) | `finc_client.py:546`, `doi_resolver.py:738` |
| F-4 | ddc | one key, three value TYPES: set-of-str / str / list-of-`{code,determinancy}`; DNB dict key is also misspelled `determinancy` | `dnb_utils.py:140` |
| F-5 | count / frequency | 9+ names (`count`/`display_count`/`doc_count`/`result_count`/`source_count`/`matched_count`/`total`/`totalItems`…); the count-landmine is the pipeline-critical subset | crosswalk |
| F-6 | classification | 4 incompatible encodings: `ddc/dk` sets · prefixed strings `"DK 530.145"` · `{code,type}`/`{dk,classification_type}` · finc facet `{value,count,translated}` | `biblio_client.py:1732`, `sru/provider.py:154`, `uk_manager.py:38` |
| F-7 | url | `url` alone means ≥3 targets (DOI-link / page / GND-authority); + `web_url`/`resource_url`/`swb_url` | `k10plus_resolver.py:62`, `webindex/provider.py:237`, `tool_registry.py:285` |
| F-8 | record-id | `rsn` (catalog) / `id` (finc) / `ppn` (k10plus) / `gnd_id` — no shared `identifiers{}` envelope | `biblio_client.py:1573`, `finc_client.py:544` |

**Three concepts the `BibRecord` draft below leaves UNDER-SPECIFIED — pin them down**
(this is exactly where the worst fragmentation lives):
1. **`authors` container type** — canonical `List[str]`; each producer normalizer converts (dict/str → list).
2. **URL roles** — a single `url` is insufficient (F-7). Add a typed sub-map, e.g.
   `urls{landing?, catalog?, fulltext?, authority?}` plus one canonical `url`.
3. **frequency convention** — the draft is silent on counts. Add `count` (ranking
   placeholder, `=1` on cache hits — the landmine, must stay) **and** `display_count`
   (real Häufigkeit). This is also the fix boundary for the counter-divergence bug —
   see `docs/wp_gnd_counter_divergence.md`.

Deviation ranking (worst → cheapest to fix): DOI producers (capitalized keys) · DNB
lookup (bespoke shape + misspelling) · catalog-vs-finc parallel record shapes · the
GND-pool rename layer (worst symptom, **cheapest fix** — isolated to 2 functions).

Not-a-bug (conservative flags): `gndid`(set) vs `gnd_ids`(list) is *partly intentional*
(set dedups on merge, list preserves display order) — unify names, not container types.
finc keeping the full payload under `raw` (`finc_client.py:566`) is the passthrough
*target* per `docs/wp_tool_data_passthrough.md`, not a defect.

## Principle — a shared `BibRecord`, no new category

Normalize at the source boundary to one shape and let pipeline consumers read it:

```
BibRecord = {
  identifiers: {doi?, ppn?, isbn?, gnd?},
  title, authors[], year, language?,
  abstract?,
  subjects[]              # often GND-linked
  classifications[(system, notation)]   # DDC/RVK/DK — General-Notation shape
  url?, source
}
```

**No new plugin category** (operator constraint). Producers stay where they are; we
(1) make them pass the *full* record through (Tool-Data-Passthrough), and (2) add
**consumers** on the pipeline/tool side that read `BibRecord`.

## Producers (extend existing categories)

- **k10plus** (`lookup`): `k10plus_package` already returns full record dicts — audit
  for complete passthrough; normalize to `BibRecord`.
- **DOI** (`input_source`): `resolve_doi_*` already return the full raw record; keep
  the abstract for input but expose the normalized `BibRecord` too. (Watch D-12: the
  merged `doi_resolver.py` duplicates the per-source endpoints.)
- **SRU endpoints** (`search_provider`, `sru`): seed **one activatable instance per
  preset** (DNB/LoC/GBV/SWB/K10+) using `MarcXmlClient.KNOWN_ENDPOINTS`
  (`clients/marcxml_client.py:127-153`) as the seed table and
  `synthesize_lookup_instances` (`plugin_migration.py`) as the template — the same
  pattern the DOI trio already uses. Optionally expose SRU records as a tool
  (currently SRU declares none).
- **finc/lobid/catalog** (`search_provider`): normalize their records to `BibRecord`.

## Consumers — the four paths (extend the pipeline, not a new category)

1. **Record → analysis input.** Format a `BibRecord` (title + abstract + subjects)
   into analysis text and feed the keyword-extraction step. Reuse the input dispatcher
   `execute_input_extraction` (`src/utils/input_sources/`) — add a record-shaped input
   so a Siegel/catalog record becomes analysis input just like a DOI abstract does.
2. **Record → classification priors.** Feed `record.classifications[(system,notation)]`
   into DK/RVK classification as priors/seeds instead of re-deriving via catalog
   DK-search. Ties into **General-Notation-Direction** (equal `(system, notation)`
   pairs) and the `resolve_dk_extractor` path (`search/factory.py:258-290`).
3. **Record → GND signals.** Use `record.subjects` (frequently GND-linked) as direct
   keyword candidates / verification, bypassing a live GND search where a record
   already names its subjects.
4. **Identifier crosswalk.** Resolve PPN ↔ DOI ↔ ISBN: a DOI → its K10+ record (richer
   than crossref alone) and back. Extend the `lookup`/DOI-resolver path with a
   crosswalk rather than a new category.

## Enablers from the July-9 inventory

- **SRU per-endpoint seeding** — the reference-case conversion. `_PRESETS`
  (`providers/sru/provider.py:25`) + `KNOWN_ENDPOINTS` are the seed data; the DOI trio
  (`input_sources/doi.py` + `synthesize_input_instances`) is the working template.
- **`catalog_type` DK-backend** (`providers/catalog/provider.py:55-58`) +
  `resolve_dk_extractor` if-elif (`factory.py:258-290`): today an exclusive single
  pick (Debt D-5). Path 2 wants "run classification against/with several sources" —
  candidate for a **multi-select** (checkbox) of active DK/record backends. The DOI
  `use_crossref/openalex/datacite` booleans (`doi_resolver.py`) are the reference
  multi-select implementation.
- **Full-record passthrough** — continues the **Tool-Data-Passthrough WP**
  (`docs/wp_tool_data_passthrough.md`), already done for DOI/finc/lobid.

## Proposed sequencing

- **P0 — `BibRecord` + normalizers.** Define the shape (incl. the three under-specified
  concepts above); normalize k10plus/DOI/SRU records to it. Small; unlocks every consumer.
  **Cheapest first step (from the audit):** collapse F-1 by adopting the existing typed
  `ResultItem` (`src/core/search/provider.py:68-92`) end-to-end so the suggester
  `gndid/ddc/dk` shape is emitted once and the round-trip rename
  (`_entry_from_kw_data`/`nested_from_aggregate`) is deleted — contained to 2 modules,
  highest ratio. Then add `to_bibrecord()` per producer (finc's `raw` passthrough is the
  template) and lowercase the DOI keys (F-2/D-12). Defer the classification-encoding
  merge (F-6) to General-Notation-Direction, not P0.
- **P1 — Record → analysis input** (highest immediate value; reuses input dispatcher;
  makes the Siegel/catalog path do more than DOIs today).
- **P2 — Record → classification priors** (connects General-Notation; feeds the DK/RVK
  step).
- **P3 — Record → GND signals** (subjects → keyword candidates).
- **P4 — Identifier crosswalk** (PPN↔DOI↔ISBN).
- **Enabler, sequence-independent — SRU per-endpoint seeding** so SRU endpoints are
  activatable record sources out of the box.

## Risks / open questions

- **No new category** (operator constraint) — the tension is that a record source is
  *one→many* (a Siegel/query → N records), which is why k10plus sits in `lookup`, not
  `input_source` (one→one). Extending existing categories means the "record" input is a
  special input-dispatcher shape, and the crosswalk is a lookup — acceptable, but keep
  an eye on whether the one→many mismatch forces awkwardness (revisit if it does).
- **Abstract availability**: not all catalog records carry an abstract — P1 must degrade
  to title+subjects.
- **Classification trust**: a record's own DDC/RVK is near-ground-truth but not
  infallible — priors should inform, not hard-override, the LLM step (conservative
  per CLAUDE.md).
- **Prereq**: land + commit WP Phase C first (this WP builds on the decoupled stores
  and the lookup/record plumbing).

## Not in scope
- A new `record_source` plugin category (explicitly rejected — extend existing).
- Changing agent-facing tool names.

## Pinned decisions (operator, July 19, 2026)

All P0 conventions are decided; this section is the contract.

### Canonical GND-pool vocabulary

| Concept | Canonical key | Container |
|---|---|---|
| GND ids | `gnd_ids` | `set` in nested per-term views (merge dedup), `list` in pool entries (display order; `gnd_id` = first element stays as pool convenience) |
| classifications | `classifications` | dict `{system: [entry]}` — systems are equal-rank UPPERCASE keys (`"DK"`, `"DDC"`, `"RVK"`, `"BK"`); an entry is `{code, count?, origin}`. **Replaces the separate `ddc`/`dk` fields**. See the P0 revision below — the original `{system: [codes]}` did not survive contact with the harvest. |
| ranking count | `count` | int — max-merged, never summed; stays `1` on cache hits (count landmine) |
| display count | `display_count` | int, optional — the real Häufigkeit; display-only, NEVER read by ranking (`rank_pool`) |

Container types are intentionally not unified (see "Not-a-bug" above).

**System keys are UPPERCASE** (`"DK"`, `"DDC"`, `"RVK"`) — pinned July 20 after
a consistency check found two vocabularies: the data layer wrote lowercase while
`src/utils/classification_systems.py` (which owns the display prefix and the
prefixed-string splitter) knew only uppercase. Display prefix and data key are
now the same string, so there is no upper/lower translation layer — exactly the
kind of round-trip rename P0 removed. `classification_systems` is the single
owner (`KNOWN_SYSTEMS`/`SYSTEM_KEYS`, `normalize_system`).

`BibRecord.classifications` is **the same dict**, not `(system, notation)` pairs
as the draft above sketched — identical to `ResultItem.classifications`, so a
record's own classifications feed the classification step (P2) with no adapter.

### P0 revision (July 20, 2026): classifications carry weight and origin

P0 pinned `{system: [codes]}`, implicitly treating a classification as a **fact**
("this concept has DDC X"). Measuring the lobid harvest showed that most
classifications are not facts but **weighted evidence** ("RVK WI 4700 appeared in
13 catalogue records about this term") — and that the same field would otherwise
carry both, indistinguishably: an authority DDC from the GND record beside a
statistical co-occurrence.

An entry is therefore `{code, count?, origin}` with
`origin ∈ {authority, cooccurrence}`:

| | authority | cooccurrence |
|---|---|---|
| Means | the record/authority states this classification | it co-occurred with the term in N records |
| `count` | absent — there is no frequency to report | the observation count |
| Source | `gnd_local`, every `BibRecord` producer | the lobid harvest |

Rules, both mutation-tested:
* Entries are kept **sorted** — authority first, then descending evidence — so
  "the first" is "the best" and `[:3]` is "the three best" without the consumer
  knowing the rules (`codes_for_system`/`primary_code`).
* On merge, `count` is combined with **max, never sum** (the same landmine as
  the pool `count`: summing inflates evidence when two sources saw the same
  records), and `origin` keeps the stronger claim.

Consequences accepted deliberately:
* The pinned set/list duality **does not apply** to this field — entries are
  dicts, hence unhashable; every container is an ordered list.
* `SET_FIELDS` loses the system keys (a set conversion would raise and destroy
  the ranking); only `missing_concepts` remains.
* Merge and reader helpers **normalise their inputs** rather than assuming the
  entry shape: plugins emit bare codes, and `dict("333")` raises. Tolerant at
  the single choke point so every caller stays simple.

### The three formerly under-specified conventions

1. **`authors: List[str]`** — canonical container; every producer normalizer
   converts (dict/str → list).
2. **URL roles** — typed sub-map `urls{landing?, catalog?, fulltext?, authority?}`
   plus one canonical `url`.
3. **Frequency** — `count` + `display_count` exactly as in the table above
   (the `038738e` counter-fix convention, now the written contract).

### Scope + compatibility decisions

- **P0 scope = the F-1 collapse** (this doc's "cheapest first step"), not the full
  `to_bibrecord()` set. DOI casing (F-2), classification encodings (F-6 → WP-D2)
  stay out.
- **Suggester plugin contract v2**: the `transform()` output uses the canonical
  keys (`gnd_ids` + `classifications{system: codes}`) — the legacy
  `{gndid, ddc, dk}` shape is retired everywhere, including the blueprint
  provider dirs and the docs.
- **Hard cut, no backward compatibility**: no tolerant legacy readers, no
  `gndid` fallbacks, no old-save migration — the legacy key set was never used
  in production. Final gate: `grep -rn "gndid" src/` → only the documented
  purge helper.

  *Correction (July 20): P0 claimed this gate was met; it was not. Five hits
  remained — a live read fallback in `swb/suggester.py` plus three blueprint
  README lines. Removing the fallback needed more than a delete: 63 of 194 swb
  raw-cache rows in the production DB still carried the pre-v2 shape, and that
  cache expires by row count, not by age. Read with v2 keys they would have
  yielded an empty `gnd_ids` — a silently keyword-less hit rather than a cache
  miss. Fallback removed AND the rows dropped once
  (`UnifiedKnowledgeManager._purge_pre_v2_swb_raw_rows`).*

Execution plan (5 phases, per-phase green suite): see the WP-D1 entry in
[`open_workpackages.md`](open_workpackages.md). The counter bug
(`wp_gnd_counter_divergence.md`) was already fixed (`038738e`) and its
convention is pinned above.
