# MCP - Model Context Protocol Tool Layer

## Architecture
- **tool_schemas.py**: JSON Schema definitions for all tools (knowledge, library, pipeline)
- **tool_registry.py**: Tool name → handler mapping, dispatches to existing ALIMA services
- **mcp_types.py**: Shared types (ToolDefinition)

## Tool Sets
- **Knowledge tools**: Wrap `UnifiedKnowledgeManager` (search_gnd, get_gnd_entry, etc.)
- **Library tools**: Wrap suggesters/resolvers (search_lobid, search_swb, resolve_doi)
- **Pipeline result tools**: Access saved JSON results (list, load, extract keywords/abstract)

## Integration
- `ToolRegistry.register_all_tools()` sets up all handlers with lazy service init
- Agents access tools via `ToolRegistry.execute(name, args)` → JSON string result
- No network transport needed - in-process tool execution
