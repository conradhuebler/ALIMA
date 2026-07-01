# WP: Tool Data Passthrough — Audit & Cut Old Braids

> **Status:** 🚧 In progress (started July 1, 2026). DOI + web/pdf tools done; the
> GND/catalog search tools need a consumer-aware audit before change.

## Principle

Agent-facing tools must **forward the complete data their source provides**, not the
narrow subset the *old classic pipeline* happened to need. Several MCP tools were
shaped by pipeline needs (e.g. the GND-pool only wants `{count, gndid, ddc, dk}`), so
they silently drop fields the source actually returns (variant names, definitions,
relations, holdings, full records, …). The agent then can't use data that exists.

**Caveat (why this is a WP, not a blind edit):** some of these tools feed *both* the
agent *and* the deterministic pipeline (via `deterministic_functions` →
`tool_registry.execute`). The pipeline's pool/ranking depends on the current shape.
So each tool must be audited against **its live source response** *and* **its
consumers** before widening. Where a consumer needs the reduced shape, add the full
data alongside (don't remove the reduced fields).

## Method (per tool)

1. Call the tool against a real query; capture the raw source response (API JSON /
   HTML / DB row) and diff it against what the tool forwards.
2. List every dropped field + whether it's useful.
3. Find consumers (`grep tool_registry.execute("<name>"` + deterministic funcs +
   pipeline). Confirm which rely on the current shape.
4. Widen the tool to forward everything; keep reduced fields the pipeline needs.
5. Test + note in `AIChangelog.md`.

## Inventory (initial pass — verify each against live source)

| Tool | Source | Forwards now | Suspected dropped data | Consumers | Status |
|---|---|---|---|---|---|
| `resolve_doi_crossref/openalex/datacite` | source API | **full raw record** | — | agent | ✅ done |
| `resolve_doi` (merged) | 3 DOI APIs | **all 3 full records** + abstract | — | agent | ✅ done |
| `scrape_url` | web/PDF | **full page text** (script/style only removed), no default trunc | — | agent | ✅ done |
| `read_pdf` | PDF | full text (default `max_chars=0`) | — | agent | ✅ done |
| `search_lobid` | lobid **resources** search w/ subject aggregation | `{count, gndid, ddc, dk}` per subject + gnd_urls | **`member`** resource list (15/page) + `totalItems`; subject *labels* fall back to raw id | agent **+ pipeline pool** | ✅ **audited** (see below) |
| `search_swb` | SWB | same reduced GND-keyword shape | full MARC/record fields | agent + pipeline | ⬜ audit |
| `search_catalog` (GND) | Libero SOAP | same reduced shape | full catalog record | agent + pipeline | ⬜ audit |
| `search_catalog_titles` | Libero | record list (rsn, title, authors, year, dk, rvk, subjects) | verify vs raw SOAP record | agent | ⬜ audit |
| `search_finc` | VuFind JSON | records (id, title, authors, subjects, formats, languages, series, urls, web_url, resource_url) **+ full `raw`** + facets | none — server whitelists 8 fields (verified) | agent + pipeline | ✅ **audited — clean** |
| `get_gnd_entry` / `get_gnd_batch` | local GND DB | verify columns vs row | possible dropped columns | agent + pipeline | ⬜ audit |
| `rvk_lookup` | RVK data | verify | — | agent | ⬜ audit |
| `get_classification` / `get_dk_cache` / `get_search_cache` | local DB | verify | — | agent | ⬜ audit |
| `analyze_image` | Vision LLM | OCR text | inherent (text only) | agent | ✅ n/a |
| pipeline/workflow/export tools | local | — | — | agent | ✅ n/a |

The reduction for the GND-keyword tools lives in the **suggester layer**
(`src/utils/suggesters/{lobid,swb,biblio,finc}_suggester.py`) + the serializer
`_serialize_suggester_results` — the audit must check what each suggester keeps vs
what its upstream API returns, not just the MCP handler.

## Design for the GND tools (proposal, pending audit)

Because the pipeline pool needs `{count, gndid, ddc, dk}`, don't remove it. Instead
add a `record`/`raw` field carrying the full source record per hit (like `finc`
already keeps `record`). The pool code keeps reading the reduced fields; the agent
gains the full data. Decide per tool after seeing the real dropped fields.

## Audit result — `search_finc` (verified live, July 1 2026)

**finc drops nothing in ALIMA — it is the reference model.** The chain forwards
everything: `FincClient._normalize_record` keeps the **complete VuFind record under
`raw`** (+ normalized fields + `web_url`/`resource_url` enrichment); `FincSuggester`
and `_handle_search_finc` pass records through **verbatim**.

The limit is **server-side**, verified against `dobby.ub.tu-freiberg.de`
(`fincsolrproxy`):
- Both the search and record endpoints expose a **fixed 8-field projection**:
  `authors, formats, id, languages, series, subjects, title, urls`.
- `field[]=*` returns **empty**; naming individual `field[]` returns *fewer* fields
  (proxy whitelist). ⇒ Adding a `field[]` request would **reduce** output — the
  current no-`field[]` default is correct. **No ALIMA change.**
- DK/RVK classifications come via **facets** (`udk_raw_de105`, `rvk_facet`), already
  supported by the tool's `facets` param.
- More per-record data (year/publisher/ISBN/holdings) would require the *operator* to
  widen the proxy's field whitelist — outside ALIMA.

**Method note for the remaining audits:** verify against the *live* source before
changing. Here the assumption "finc drops data" was wrong; the real losses are in the
GND-keyword reducers (lobid/swb/catalog), which is where the WP effort should go next.

## Audit result — `search_lobid` (verified live, July 1 2026)

lobid is queried as a **subject aggregator**: `…/resources/search?q=…&aggregations=
subject.componentList.id`. Live response (`q=Quantenchemie`, `totalItems=875`) has:
- `aggregation.subject.componentList.id`: 100 buckets, each **only** `{key: gnd-id,
  doc_count}`. The suggester extracts **both** (count + gndid) → its *primary* data is
  fully forwarded; the buckets are inherently minimal (no hidden `raw`, unlike finc).
- `member`: 15 **resource records** (title, contribution, publication, isbn, subject,
  extent, medium, …) — currently **dropped** (`_get_results` reads only `aggregation`).
- `totalItems`: ignored.

**Two honest constraints on "just forward the member records":**
1. The subject *labels* come from a local file (`gnd_subjects`) with a **raw-GND-id
   fallback** — that's a label-quality gap, not dropped source data.
2. **Mapping-first cache**: the agent's `search_lobid` normally serves from
   `search_mappings` and **does not call lobid live**, so `member` records exist only on
   a cache-miss. Reliably surfacing them means caching them too (invasive) — otherwise
   the tool would sometimes have resources and sometimes not.

**Recommendation (pending operator decision):** don't plumb the fragile, cache-bypassed
`member` list into the gnd-keyword shape. Per-subject enrichment (variant names, DDC,
relations) is better served by the existing **`get_gnd_entry`** tool / local `gnd_entries`
DB, which the agent can call for any returned gnd-id. If a "matching resources" view is
wanted, add it as a *separate* tool (like `search_finc`/`search_catalog_titles`), not by
overloading `search_lobid`. → **conclusion: `search_lobid` forwards its primary data;
no un-drop change made pending the decision above.**

## Done (July 1, 2026)

- DOI tools query each source API directly and return the complete record
  (`mcp_execute`); `resolve_doi` aggregates all enabled sources; `scrape_url` /
  `read_pdf` return full content (default no truncation). See `AIChangelog.md`.
- `search_finc` audited (live) — already full passthrough; server whitelists 8 fields.
