# Search Provider Plugin System (Design / Implemented)

> **Update (July 8, 2026): construction unified on the factory; MetaSuggester retired.**
> A single GND-keyword entry point `src/core/search/service.py`
> (`search_gnd_keywords` / `resolve_gnd_instances` / `underlying_suggester`) now
> builds every provider through `factory.build_provider` from the authoritative
> `PluginInstanceConfig`. All three former construction sites converge on it —
> classic `SearchCLI`, the MCP `ToolRegistry` primaries (`_provider_for`), and the
> GUI `find_keywords` standalone/manual search (which no longer bypasses the WP2 raw
> cache). `src/utils/suggesters/meta_suggester.py` is deleted. **Defaults vs
> blueprints:** lobid + swb are the zero-config web defaults (oGND = lobid); catalog
> (Libero) + finc are enabled-but-`is_available()`-gated blueprints (token /
> base_url); gnd_local is the offline option. One residual: finc's MCP handler
> (`_handle_search_finc`) still reads `CatalogConfig` (institution-specific). See
> `AIChangelog.md` (July 8, 2026).

> **Status:** ✅ Implemented (June 29, 2026) in `src/core/search/` — P1+P2+P3 per the
> migration plan below. Commits `5061801` (P1), `35bc487` (P2 + F-4), `85900e2`
> (P3 MCP generation), + `SearchProviderConfig`/GUI selector. `SuggesterType` is
> retired; finc is folded into the standard; MCP search tools are generated from
> `ProviderToolSpec`s. The design notes below are kept for context; deviations from
> the original sketch: caching wraps `GND_KEYWORDS` only; the `currentTerm` Qt signal
> survives via an optional `progress` callback (providers themselves are Qt-free);
> `finc`'s rich availability/web_url handler is kept (wired via its spec) rather than
> fully generated. Open: operator GUI click-test + a final agentic run.

## Motivation

Adding a search source today is uneven and touches several places inconsistently.
finc is the cautionary example: because it returns **bibliographic records**
(id/title/authors/subjects/facets) rather than the GND-keyword shape the suggester
contract assumes, it was bolted onto the MCP tool layer directly and **skipped the
suggester abstraction entirely**. The result is an "uneasy" structure with no
single standard for what a search source is or how it is exposed.

## Current state (verified June 2026)

Three overlapping layers, two parallel registries, one rigid result contract:

1. **Suggester layer** (`src/utils/suggesters/`)
   - `BaseSuggester` (ABC, **QObject** — coupled to Qt signals) with one contract:
     `search(terms) -> Dict[term][keyword]{count, gndid, ddc, dk}`.
   - Concrete: `LobidSuggester`, `SWBSuggester`, `BiblioSuggester` (catalog),
     `FincSuggester`.
   - `MetaSuggester`: orchestrator + mapping-first caching (baked in), wired by a
     **hardcoded if/elif** over the `SuggesterType` enum (`LOBID|SWB|CATALOG|ALL`).
     `SuggesterType` does **not** include finc.

2. **MCP tool layer** (`src/mcp/`)
   - Per-source `_handle_search_*` + a hand-written `ToolDefinition`
     (`SEARCH_GND/LOBID/SWB/CATALOG/CATALOG_TITLES/FINC`).
   - `tool_registry._init_suggesters` lazy-inits lobid/swb via `MetaSuggester`,
     biblio directly, and **finc separately** (`_handle_search_finc` + own `_finc`).
   - `default_presets.yaml`: named tool sets (`library/gnd/classification/none`) —
     the current "selectable tools" mechanism for agentic workflows.

3. **Result-format fragmentation**
   - GND-keyword sources (lobid/swb/catalog): `{count, gndid, ddc, dk}`.
   - finc: **records** + facet distributions (`udk_raw_de105`, `rvk_facet`) — a
     different shape that the single `search()` contract cannot express.

**Root cause:** one `search()` contract that only fits GND-keyword sources; two
registries to keep in sync (`SuggesterType` enum **and** the MCP tool registry);
caching fused into `MetaSuggester`; ad-hoc config gating (`finc_base_url`,
catalog token); Qt coupling in the provider base. Adding a source means editing
enum + if/elif + handler + schema + preset + config — and finc skipped half of it.

## Goals / non-goals

**Goals**
- One **standard** for "a search source": a capability-declaring provider with a
  typed result.
- One **registry**; new providers self-register (no enum, no if/elif, no
  hand-written tool plumbing).
- Caching and failure-tracking as **reusable cross-cutting wrappers**, not baked
  into the orchestrator.
- Providers **auto-exposed as selectable tools** (config + GUI + workflow presets).

**Non-goals**
- No behaviour change to the live pipeline during P1 (facade-preserving, as with
  the `pipeline_utils` split).
- Not a debt fix; tackled deliberately, not under cleanup.
- `SuggesterType` is **not** a constraint — it may be retired (operator note,
  June 29, 2026).

## Proposed architecture

### 1. Capability-based provider contract (the standard)

Qt-free, lives in `core/` (not `utils/suggesters`):

```python
class SearchCapability(Enum):
    GND_KEYWORDS    # term -> GND keyword candidates      (lobid, swb, catalog)
    TITLE_RECORDS   # query -> bibliographic records       (finc, catalog titles)
    SUBJECT_FACETS  # term -> classification distribution   (finc udk/rvk facets)
    CLASSIFICATION  # title/keyword -> DK/RVK codes         (catalog DK lookup)

@dataclass
class ProviderResult:
    capability: SearchCapability
    per_term: dict[str, list[ResultItem]]   # typed items, not a nested dict
    errors: dict[str, str]                   # term -> message (source-failure aware)

class SearchProvider(Protocol):
    id: str                          # "lobid", "swb", "finc", ...
    label: str
    capabilities: set[SearchCapability]
    def is_available(self, cfg) -> bool        # uniform config gating
    def search(self, capability, query, **opts) -> ProviderResult
```

A typed `ProviderResult` makes the GND-keyword shape and the record/facet shape
**explicit variants** of one standard — finc no longer has to break the contract.

### 2. One registry

`@register_provider` — the same idiom already used in this codebase for
`@register_tool_fn` / `@register_step` (`src/core/agents/registry.py`). Providers
self-register on import. Both the orchestrator and the tool layer **enumerate the
registry**; the `SuggesterType` enum and the `MetaSuggester` if/elif are removed.

### 3. Cross-cutting concerns as wrappers

- **mapping-first caching** becomes a decorator around any `GND_KEYWORDS` provider
  (today it is fused into `MetaSuggester.search`). Reusable, opt-in per capability,
  no duplication.
- **source-failure tracking** (`last_errors`) is standardized in `ProviderResult`.

### 4. Auto-exposed as selectable tools

A provider's `ToolDefinition` + handler are **generated** from its capability
declaration — no more hand-written `_handle_search_finc` + schema + preset entry.
Presets and the GUI provider-selector read the registry; per-provider config
(enable flag + endpoints/token) drives availability → real "selectable tools".

### 5. Uniform provider config

A `ProviderConfig` (enabled, endpoints, token, capability opt-outs) in
`ConfigManager` replaces the ad-hoc `finc_base_url` / catalog-token gating. The GUI
lists registered providers with checkboxes.

### How current sources map

| Source | Provider id | Capabilities |
|---|---|---|
| Lobid | `lobid` | GND_KEYWORDS |
| SWB | `swb` | GND_KEYWORDS |
| Catalog (Libero/Biblio) | `catalog` | GND_KEYWORDS, TITLE_RECORDS, CLASSIFICATION |
| finc (VuFind) | `finc` | TITLE_RECORDS, SUBJECT_FACETS |
| Local GND DB | `gnd_local` | GND_KEYWORDS (no network; not cache-wrapped) |

## Migration (incremental, facade-preserving)

- **P1 — Standard + registry, adapt existing.** Define `SearchCapability`,
  `ProviderResult`, `SearchProvider`, `@register_provider`. Wrap the existing
  suggesters as providers (behaviour unchanged). Bring **finc into the standard**
  (`TITLE_RECORDS` + `SUBJECT_FACETS`) so it is no longer special-cased. Keep the
  current `MetaSuggester`/tool entry points as thin facades.
- **P2 — Caching as wrapper; retire `SuggesterType`.** Move mapping-first into a
  decorator. `MetaSuggester` (or its successor) enumerates the registry; drop the
  enum + if/elif. Update the `SuggesterType` importers: `src/cli/commands/search_cmd.py`,
  `src/ui/find_keywords.py`, `src/core/search_cli.py`, `src/core/pipeline_manager.py`
  (replace enum members with provider ids; a short-lived compat shim is optional
  but not required).
- **P3 — Auto tool exposure + GUI selection.** Generate MCP schemas/handlers and
  presets from the registry; add a GUI provider-selector backed by `ProviderConfig`.

Each phase keeps existing imports working until its importers are migrated — the
same facade discipline used for the `pipeline_utils` split.

## Open design questions

1. **Result granularity:** one `ProviderResult` with a capability tag, or distinct
   typed results per capability? (Tag is simpler; per-capability is stricter.)
2. **Qt decoupling:** keep the `currentTerm` progress signal as an optional
   callback in the Protocol, or drop Qt from providers entirely (GUI adapts)?
3. **Caching scope:** wrap only `GND_KEYWORDS`, or also cache `TITLE_RECORDS`
   (records are heavier and change more often)?
4. **Provider discovery:** decorator-on-import (simple) vs Python entry-points
   (true third-party plugins, installable separately)?
