# MCP - Model Context Protocol Tool Layer

## Architecture
- **tool_schemas.py**: JSON Schema definitions for all tools (knowledge, library, pipeline)
- **tool_registry.py**: Tool name → handler mapping, dispatches to existing ALIMA services
- **mcp_types.py**: Shared types (ToolDefinition)

## Tool Sets
- **Knowledge tools**: Wrap `UnifiedKnowledgeManager` (search_gnd, get_gnd_entry, etc.)
- **Library tools**: Wrap suggesters/resolvers (search_lobid, search_swb, search_catalog, search_finc, resolve_doi, scrape_url, read_pdf, analyze_image)
  - `search_finc` → finc/VuFind-JSON catalog (`FincSuggester`): search by subject / one-or-many titles / author; optional `facets` (e.g. `udk_raw_de105`, `rvk_facet`) for DK/RVK distribution. Config-gated (`finc_base_url`).
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
