# MCP - Model Context Protocol Tool Layer

## Architecture
- **tool_schemas.py**: JSON Schema definitions for all tools (knowledge, library, pipeline)
- **tool_registry.py**: Tool name → handler mapping, dispatches to existing ALIMA services. God-file split July 21 (F-15): die 17 Tool-*Fabrik*-Methoden (`_make_*_handler`/`_generated_*_tools` + Agent-View/Raw-Cache-Klempnerei) → `_tool_generation.ToolGenerationMixin` (verbatim, via MRO). Die Tool-*Handler* (`_handle_*`) bleiben in `tool_registry.py`.
- **mcp_types.py**: Shared types (ToolDefinition)

## Tool Sets
- **Knowledge tools**: Wrap `UnifiedKnowledgeManager` (search_gnd, get_gnd_entry, etc.)
  - `aggregate_gnd_results` (WP2) → ranked GND pool with counter (`display_count`) +
    provenance (`sources`/`source_count`) derived from the **raw response cache**
    (`src/core/search/aggregate.py`, raw-first + mapping fallback). `search_lobid`
    also returns an additive `agent_view` (member/totalItems) via transform-on-read.
    Input tools read-through the raw cache when `InputToolSpec.cacheable`. Spec:
    [`docs/wp_raw_response_cache.md`](../../docs/wp_raw_response_cache.md).
  - `list_plugins` → introspects the active **plugins** (search providers + input sources) from `AlimaConfig.plugins` with each plugin's self-doc (description + input/output). Distinct from `list_workflows` (workflows ≠ plugins). Handler: `ToolRegistry._handle_list_plugins`.
- **Library tools**: search_lobid/swb/catalog/catalog_titles/finc + resolve_doi, scrape_url, read_pdf, analyze_image
  - The search_* tools are **generated per enabled *instance*** (`AlimaConfig.plugins`, search category) from each provider's `ProviderToolSpec` via `ToolRegistry._generated_search_tools()` — no hand-written schema/handler. **All canonical handlers now build through the search factory** (`_provider_for` → `factory.build_provider`, memoised per instance) — lobid/swb use the mapping-first `CachingProvider` on default opts (non-default opts bypass via the raw suggester + WP2 dual-write), catalog/catalog_titles go straight to the built suggester. `_source_transform` (agentic `aggregate_gnd_results`) reads its transforms off the same factory-built providers. **MetaSuggester + the BiblioSuggester-from-CatalogConfig mirror are retired (July 8); finc folded into the factory too (WP P2.2, July 16).** The primary finc tool is `_make_finc_handler` → `_provider_for` (so a copied finc plugin uses its own backend); availability→facet_avail + dk/rvk auto-facets moved into `FincProvider.search`, the web_url fallback base is the finc instance's own `catalog_web_record_url` (config fallback only for pre-P2.2 instances). Additional instances get `search_<type>_<id>` + `_make_instance_handler` (factory), with the `usage_hint` appended. Instances are the only source: an unreadable config yields no search tools (the `SearchProviderConfig` fallback was dropped in WP P7 — it was unreachable in production).
  - `search_finc` → finc/VuFind-JSON catalog: search by subject / one-or-many titles / author; optional `facets` (e.g. `udk_raw_de105`, `rvk_facet`) for DK/RVK distribution. Config-gated (`finc_base_url`).
  - **Input-source tools** generated per enabled instance (`ToolRegistry._generated_input_tools`) for sources declaring an `InputToolSpec`: `resolve_doi_crossref/openalex/datacite` — each hits that source's API directly and returns the **complete raw metadata record** (for source comparison), distinct from the merged abstract-oriented `resolve_doi`.
  - **Runtime toggle**: `ToolRegistry.refresh()` rebuilds all tools from current config (clears + reloads); wired to the settings-save so plugin enable/disable applies without restart.
- **Pipeline result tools**: Access saved JSON results (list, load, extract keywords/abstract)
- **Export tools** (P-θ): `export_results` (json/csv/tex/marc), `generate_report` (Jinja2 TeX, optional pdflatex)

## Input-Beschaffung Tools (P-η)
- `read_pdf` → `src/utils/pdf_extractor.py` (PyPDF2 + quality heuristic + optional Vision-LLM OCR fallback)
- `analyze_image` → `src/utils/image_analyzer.py` (sync wrapper over `LlmService.generate_response(image=...)`)
- `scrape_url` Content-Type auto-detects `application/pdf` → temp download → `read_pdf`

## Export Tools (P-θ)
- `export_results` → `src/utils/exporters.py` (loads via `load_state('latest'|filename|abspath)`)
- `generate_report` → `src/utils/report_renderer.py` + `src/utils/report_templates/*.tex.j2`
- Jinja2 uses custom delimiters `(((  )))` / `((* *))` to avoid LaTeX brace collision

## Integration
- `ToolRegistry.register_all_tools()` sets up all handlers with lazy service init
- Agents access tools via `ToolRegistry.execute(name, args)` → JSON string result
- No network transport needed - in-process tool execution
