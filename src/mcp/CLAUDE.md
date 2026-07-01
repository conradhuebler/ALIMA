# MCP - Model Context Protocol Tool Layer

## Architecture
- **tool_schemas.py**: JSON Schema definitions for all tools (knowledge, library, pipeline)
- **tool_registry.py**: Tool name → handler mapping, dispatches to existing ALIMA services
- **mcp_types.py**: Shared types (ToolDefinition)

## Tool Sets
- **Knowledge tools**: Wrap `UnifiedKnowledgeManager` (search_gnd, get_gnd_entry, etc.)
  - `list_plugins` → introspects the active **plugins** (search providers + input sources) from `AlimaConfig.plugins` with each plugin's self-doc (description + input/output). Distinct from `list_workflows` (workflows ≠ plugins). Handler: `ToolRegistry._handle_list_plugins`.
- **Library tools**: search_lobid/swb/catalog/catalog_titles/finc + resolve_doi, scrape_url, read_pdf, analyze_image
  - The search_* tools are **generated per enabled *instance*** (`AlimaConfig.plugins`, search category) from each provider's `ProviderToolSpec` via `ToolRegistry._generated_search_tools()` — no hand-written schema/handler. The *primary* instance of a type keeps the canonical name (`search_lobid`) + existing handler; additional instances (e.g. a 2nd finc endpoint) get `search_finc_<id>` + a factory-built handler, with the instance `usage_hint` appended to the description. No-config fallback = one primary per registered type gated by `SearchProviderConfig`. `_handle_search_finc` kept for the primary finc's availability/web_url logic.
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
