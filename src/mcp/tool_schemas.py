"""JSON Schema definitions for all ALIMA MCP tools - Claude Generated

Each tool wraps an existing ALIMA service method. Tools are grouped into:
- Knowledge tools: Database access via UnifiedKnowledgeManager
- Library tools: Web service access via Suggesters/Resolvers
- Pipeline tools: Access to saved pipeline results
"""
from src.mcp.mcp_types import ToolDefinition


# ============================================================
# Knowledge Server Tools (Database)
# ============================================================

SEARCH_GND = ToolDefinition(
    name="search_gnd",
    description="Search local GND database for entries matching a term. Returns GND IDs, titles, descriptions, DDC codes.",
    parameters={
        "type": "object",
        "properties": {
            "term": {"type": "string", "description": "Search term (German or English subject heading)"},
            "min_results": {"type": "integer", "description": "Minimum results to return", "default": 3},
        },
        "required": ["term"],
    },
)

GET_GND_ENTRY = ToolDefinition(
    name="get_gnd_entry",
    description="Get a specific GND entry by its GND ID. Returns title, description, synonyms, DDC codes.",
    parameters={
        "type": "object",
        "properties": {
            "gnd_id": {"type": "string", "description": "GND identifier (e.g. '040128989')"},
        },
        "required": ["gnd_id"],
    },
)

GET_GND_BATCH = ToolDefinition(
    name="get_gnd_batch",
    description="Batch-retrieve multiple GND entries by their IDs. Efficient for loading many entries at once.",
    parameters={
        "type": "object",
        "properties": {
            "gnd_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of GND identifiers",
            },
        },
        "required": ["gnd_ids"],
    },
)

GET_SEARCH_CACHE = ToolDefinition(
    name="get_search_cache",
    description="Get cached search results for a term and suggester type. Avoids redundant web searches.",
    parameters={
        "type": "object",
        "properties": {
            "term": {"type": "string", "description": "Search term"},
            "suggester_type": {"type": "string", "description": "Suggester type: 'lobid', 'swb', or 'biblio'"},
        },
        "required": ["term", "suggester_type"],
    },
)

GET_DK_CACHE = ToolDefinition(
    name="get_dk_cache",
    description="Get cached DK classification results from catalog search for a term.",
    parameters={
        "type": "object",
        "properties": {
            "term": {"type": "string", "description": "Search term for DK classification lookup"},
        },
        "required": ["term"],
    },
)

STORE_SEARCH_RESULT = ToolDefinition(
    name="store_search_result",
    description="Cache a search result for future use. Stores GND IDs and classifications found for a term.",
    parameters={
        "type": "object",
        "properties": {
            "term": {"type": "string", "description": "Search term"},
            "suggester_type": {"type": "string", "description": "Suggester type"},
            "gnd_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Found GND identifiers",
                "default": [],
            },
            "classifications": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Found classifications",
                "default": [],
            },
        },
        "required": ["term", "suggester_type"],
    },
)

GET_CLASSIFICATION = ToolDefinition(
    name="get_classification",
    description="Get a DK or RVK classification entry by code. Returns title, description, parent code.",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Classification code (e.g. '004' for DK, 'ST 250' for RVK)"},
            "classification_type": {"type": "string", "description": "'DK' or 'RVK'", "enum": ["DK", "RVK"]},
        },
        "required": ["code", "classification_type"],
    },
)

GET_DB_STATS = ToolDefinition(
    name="get_db_stats",
    description="Get database statistics: number of GND entries, search mappings, classifications, etc.",
    parameters={
        "type": "object",
        "properties": {},
    },
)


# ============================================================
# Library Server Tools (Web Services)
# ============================================================

SEARCH_LOBID = ToolDefinition(
    name="search_lobid",
    description="Search Lobid.org GND API for subject headings. Returns keywords with GND IDs and DDC codes.",
    parameters={
        "type": "object",
        "properties": {
            "terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of search terms",
            },
        },
        "required": ["terms"],
    },
)

SEARCH_SWB = ToolDefinition(
    name="search_swb",
    description="Search SWB (Südwestdeutscher Bibliotheksverbund) catalog for subject headings and classifications.",
    parameters={
        "type": "object",
        "properties": {
            "terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of search terms",
            },
            "max_pages": {"type": "integer", "description": "Max result pages per term", "default": 5},
        },
        "required": ["terms"],
    },
)

SEARCH_CATALOG = ToolDefinition(
    name="search_catalog",
    description="Search bibliographic catalog via SOAP/SRU for titles and DK classifications.",
    parameters={
        "type": "object",
        "properties": {
            "terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of search terms",
            },
        },
        "required": ["terms"],
    },
)

RESOLVE_DOI = ToolDefinition(
    name="resolve_doi",
    description="Resolve a DOI to metadata and abstract text. Tries Crossref, OpenAlex, DataCite.",
    parameters={
        "type": "object",
        "properties": {
            "doi": {"type": "string", "description": "DOI string (e.g. '10.1234/example')"},
        },
        "required": ["doi"],
    },
)


# ============================================================
# Pipeline Result Tools
# ============================================================

LIST_PIPELINE_RESULTS = ToolDefinition(
    name="list_pipeline_results",
    description="List available saved pipeline result JSON files. Returns filenames, timestamps, and working titles.",
    parameters={
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Max results to return", "default": 20},
            "search": {"type": "string", "description": "Optional filter by title or filename"},
        },
    },
)

GET_PIPELINE_RESULT = ToolDefinition(
    name="get_pipeline_result",
    description="Load a saved pipeline result by filename. Returns the full KeywordAnalysisState with abstract, keywords, classifications.",
    parameters={
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "JSON filename of the pipeline result"},
        },
        "required": ["filename"],
    },
)

GET_PIPELINE_KEYWORDS = ToolDefinition(
    name="get_pipeline_keywords",
    description="Get only the final GND keywords from a pipeline result. Concise view.",
    parameters={
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "JSON filename of the pipeline result"},
        },
        "required": ["filename"],
    },
)

GET_PIPELINE_ABSTRACT = ToolDefinition(
    name="get_pipeline_abstract",
    description="Get only the original abstract text from a pipeline result.",
    parameters={
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "JSON filename of the pipeline result"},
        },
        "required": ["filename"],
    },
)


# ============================================================
# Tool Sets (grouped for agent use)
# ============================================================

KNOWLEDGE_TOOLS = [
    SEARCH_GND, GET_GND_ENTRY, GET_GND_BATCH,
    GET_SEARCH_CACHE, GET_DK_CACHE, STORE_SEARCH_RESULT,
    GET_CLASSIFICATION, GET_DB_STATS,
]

LIBRARY_TOOLS = [
    SEARCH_LOBID, SEARCH_SWB, SEARCH_CATALOG, RESOLVE_DOI,
]

PIPELINE_RESULT_TOOLS = [
    LIST_PIPELINE_RESULTS, GET_PIPELINE_RESULT,
    GET_PIPELINE_KEYWORDS, GET_PIPELINE_ABSTRACT,
]

ALL_TOOLS = KNOWLEDGE_TOOLS + LIBRARY_TOOLS + PIPELINE_RESULT_TOOLS
