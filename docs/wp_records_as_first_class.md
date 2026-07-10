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

## Decision point (for a new session)

The design is grounded, the audit confirms real debt, and the prereq (Phase C) is
committed — **B is startable.** Open operator decisions before P0:
1. **P0 scope** — just the F-1 collapse (2 modules, high ratio) as a first landing, or
   the full `to_bibrecord()` normalizer set + DOI casing fix in one go?
2. **Canonical shapes** — confirm the three under-specified conventions above
   (`authors: List[str]`, typed `urls{}` sub-map, `count` + `display_count`).
3. **Sequencing vs. the counter bug** (`docs/wp_gnd_counter_divergence.md`) — that is a
   self-contained ~2-edit fix; **recommended as a quick-win before B**, because it also
   exercises the exact `count`/`display_count` convention P0 will formalize.

Status of inputs: field-name audit + counter-divergence diagnosis both done + verified
(July 10, 2026); no code written for either yet.
