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

AGGREGATE_GND_RESULTS = ToolDefinition(
    name="aggregate_gnd_results",
    description=(
        "Aggregate the cached raw responses of the enabled GND-keyword sources "
        "for the given terms into one ranked pool with counter "
        "statistics (Häufigkeit as 'display_count') and provenance ('sources' + "
        "'source_count' = which sources confirmed each keyword). Reads the raw "
        "response cache (single source of truth) — run the search_* tools first to "
        "populate it. Returns {pool:[...], sources:[...], missing:{source:[terms]}}."
    ),
    parameters={
        "type": "object",
        "properties": {
            "terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Search terms to aggregate",
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Provider ids to include, as reported by list_plugins ('type' of an "
                    "instance with the 'gnd_keywords' capability — not the 'source' field "
                    "of a search_* response). Default: all enabled GND-keyword sources."
                ),
                "default": [],
            },
            "search_type": {
                "type": "string",
                "description": "Search mode the raw was fetched with (default 'kw')",
                "default": "kw",
            },
            "max_pages": {
                "type": "integer",
                "description": "swb max_pages the raw was fetched with (default 5)",
                "default": 5,
            },
        },
        "required": ["terms"],
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

ABOUT_ALIMA = ToolDefinition(
    name="about_alima",
    description=(
        "Facts about ALIMA itself: what the name stands for, what the system "
        "does, how mature it is, its two pipeline modes, where it was developed, "
        "the publication that describes it (full citation + DOI), the "
        "repository, the licence, contributors and acknowledgements. "
        "Call this whenever someone asks about ALIMA as a system — 'what is "
        "ALIMA', 'what can you do', 'how does it work', 'how reliable is it', "
        "'who made you', 'is there a paper / how do I cite this', "
        "'which licence'. "
        "Do NOT answer those from memory: a citation invented from training "
        "data looks right and gets the volume, pages or year wrong. "
        "This describes the SYSTEM, not its data — for the active sources use "
        "list_plugins, for runnable orchestrations list_workflows, for the "
        "database contents get_db_stats."
    ),
    parameters={
        "type": "object",
        "properties": {},
    },
)

LIST_PLUGINS = ToolDefinition(
    name="list_plugins",
    description=(
        "List the active ALIMA *plugins* — the configured search providers and "
        "input sources, each with its self-description (what it does + input/output). "
        "Use this to answer 'which plugins/sources are active?' or to choose a source. "
        "NOTE: plugins are NOT workflows — workflows are orchestrations you run "
        "(see list_workflows); plugins are the search/input building blocks."
    ),
    parameters={
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["search_provider", "input_source"],
                "description": "Optional: restrict to one plugin category.",
            },
            "include_disabled": {
                "type": "boolean",
                "default": False,
                "description": "Include disabled instances too (default: only active ones).",
            },
        },
    },
)

RVK_LOOKUP = ToolDefinition(
    name="rvk_lookup",
    description=(
        "Find authoritative, validated RVK (Regensburger Verbundklassifikation) "
        "notations for a set of subject keywords. Runs the catalog RVK search, "
        "validates candidates against the official RVK API, and returns a ranked "
        "shortlist of authority-backed RVK notations with label and hierarchy "
        "path. Use this when RVK classification is appropriate for the work. "
        "Use ONLY the notations returned here — never invent RVK codes."
    ),
    parameters={
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Subject keywords, ideally as 'Term (GND-ID: ...)' like the "
                    "pipeline keyword format."
                ),
            },
            "abstract": {
                "type": "string",
                "description": "The work's abstract — used for thematic RVK scoring.",
                "default": "",
            },
            "dk_codes": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Already-chosen DK and/or DDC codes (e.g. 'DK 330' or "
                    "'DDC 330') — used as an additional thematic hint for RVK "
                    "ranking."
                ),
                "default": [],
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum RVK candidates to return.",
                "default": 8,
            },
        },
        "required": ["keywords"],
    },
)


# ============================================================
# Library Server Tools (Web Services)
# ============================================================

# search_lobid / search_swb / search_catalog / search_catalog_titles / search_finc
# are generated from each provider's ProviderToolSpec (src/core/search/providers),
# registered via ToolRegistry._generated_search_tools(). - Claude Generated

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

SCRAPE_URL = ToolDefinition(
    name="scrape_url",
    description=(
        "Fetch a webpage and return its FULL readable text (only scripts/styles are "
        "removed — nav/header/footer content is kept). Auto-detects PDF Content-Type and "
        "routes to read_pdf. Returns text, title, and full_chars/truncated."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch (e.g. 'https://example.com/article')"},
            "max_chars": {"type": "integer", "default": 0, "description": "Max characters to return; 0 = full page (default). Set only to cap very long pages."},
        },
        "required": ["url"],
    },
)


READ_PDF = ToolDefinition(
    name="read_pdf",
    description=(
        "Extract text from a local PDF file. Reports quality assessment. "
        "Optional LLM-OCR fallback for scanned/low-quality PDFs (requires llm_service + Vision model)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative path to PDF file"},
            "max_chars": {"type": "integer", "default": 0, "description": "Max characters to return; 0 = no limit (default, returns all extracted text)"},
            "ocr_fallback": {"type": "boolean", "default": False, "description": "Use Vision-LLM OCR if text-layer quality is poor (expensive)"},
            "provider": {"type": "string", "description": "Override Vision provider for OCR fallback"},
            "model": {"type": "string", "description": "Override Vision model for OCR fallback"},
        },
        "required": ["path"],
    },
)


ANALYZE_IMAGE = ToolDefinition(
    name="analyze_image",
    description=(
        "Run Vision LLM on a local image file. Default prompt = OCR (extract readable text). "
        "Override prompt for book-cover / table-of-contents classification."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative path to image file (PNG/JPG)"},
            "prompt": {"type": "string", "description": "Custom prompt; default = OCR extraction"},
            "provider": {"type": "string", "description": "Vision provider (e.g. 'ollama', 'openai')"},
            "model": {"type": "string", "description": "Vision model (e.g. 'llava', 'gpt-4o')"},
            "temperature": {"type": "number", "default": 0.7, "description": "Sampling temperature"},
        },
        "required": ["path"],
    },
)


EXPORT_RESULTS = ToolDefinition(
    name="export_results",
    description=(
        "Export pipeline results to disk in the requested format. "
        "Reads from a saved autosave JSON (source='latest' or filename) or absolute path."
    ),
    parameters={
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Source spec: 'latest' (most-recent autosave), filename in autosave dir, or absolute path",
                "default": "latest",
            },
            "format": {
                "type": "string",
                "enum": ["json", "csv", "tex", "marc"],
                "description": "Output format. 'marc' = K10+/WinIBW catalog tags.",
            },
            "output_path": {"type": "string", "description": "Output file path. If omitted: derived from working_title in autosave dir."},
            "validate_rvk": {"type": "boolean", "default": False, "description": "Live-validate RVK codes via official API (JSON only, slow)"},
        },
        "required": ["format"],
    },
)


GENERATE_REPORT = ToolDefinition(
    name="generate_report",
    description=(
        "Generate a LaTeX report from pipeline results using a Jinja2 template. "
        "Optional pdflatex build (silent if pdflatex missing)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "source": {"type": "string", "default": "latest", "description": "Source spec: 'latest', filename in autosave dir, or absolute path"},
            "template": {
                "type": "string",
                "enum": ["ub_freiberg", "short"],
                "description": "Report template name",
            },
            "output_path": {"type": "string", "description": "Output .tex path. If omitted: derived from working_title."},
            "build_pdf": {"type": "boolean", "default": False, "description": "Run pdflatex (two passes) after rendering"},
        },
        "required": ["template"],
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
# Workflow Tools
# ============================================================

LIST_WORKFLOWS = ToolDefinition(
    name="list_workflows",
    description="List available agentic v4 workflows. Returns name, version, description for each.",
    parameters={"type": "object", "properties": {}},
)

SELECT_FROM_GND_POOL = ToolDefinition(
    name="select_from_gnd_pool",
    description=(
        "Select relevant GND keywords from a large candidate pool using chunked LLM filtering. "
        "Given a list of GND entries (title + GND-ID) and an abstract, splits the pool into "
        "chunks, filters each chunk for relevance via LLM, merges and deduplicates results. "
        "Equivalent to the pipeline's selection_chunks step. Use when the agent needs to "
        "filter ~100+ GND candidates down to ~20-30 relevant keywords."
    ),
    parameters={
        "type": "object",
        "properties": {
            "abstract": {
                "type": "string",
                "description": "The work's abstract or description text to filter relevance against.",
            },
            "gnd_entries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "GND subject heading text"},
                        "gnd_id": {"type": "string", "description": "GND identifier"},
                        "count": {"type": "integer", "description": "Hit count (frequency in catalog)", "default": 1},
                    },
                    "required": ["keyword"],
                },
                "description": "List of GND entry dicts with at least 'keyword' and optionally 'gnd_id', 'count'.",
            },
            "chunk_size": {
                "type": "integer",
                "description": "Max entries per LLM call (default: 350). Lower for smaller context windows.",
                "default": 350,
            },
            "max_merged": {
                "type": "integer",
                "description": "Cap on total merged keywords returned (default: 80).",
                "default": 80,
            },
        },
        "required": ["abstract", "gnd_entries"],
    },
)

GET_WORKFLOW = ToolDefinition(
    name="get_workflow",
    description="Load a workflow YAML and return its steps, inputs, outputs, and dependencies.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Workflow name (e.g. 'alima_classic', 'catalog_search')"},
        },
        "required": ["name"],
    },
)

EXECUTE_WORKFLOW = ToolDefinition(
    name="execute_workflow",
    description=(
        "Execute a named agentic v4 workflow. Returns the workflow's final execution report "
        "as a JSON string. Use this to delegate complex, multi-step tasks to specialized workflows."
    ),
    parameters={
        "type": "object",
        "properties": {
            "workflow_id": {
                "type": "string",
                "description": "Name of the workflow (e.g. 'research_deep', 'catalog_search').",
            },
            "inputs": {
                "type": "object",
                "description": "Input parameters to inject into the workflow context. Keys should match the required inputs of the workflow's first steps.",
            },
        },
        "required": ["workflow_id"],
    },
)


# ============================================================
# Tool Sets (grouped for agent use)
# ============================================================

KNOWLEDGE_TOOLS = [
    SEARCH_GND, GET_GND_ENTRY, GET_GND_BATCH,
    GET_SEARCH_CACHE, GET_DK_CACHE, STORE_SEARCH_RESULT,
    GET_CLASSIFICATION, GET_DB_STATS, SELECT_FROM_GND_POOL,
    RVK_LOOKUP, LIST_PLUGINS,
]

def _generated_search_tool_defs():
    """search_* tools generated from provider ProviderToolSpecs - Claude Generated."""
    from src.core.search import provider_tool_specs

    return [
        ToolDefinition(name=s.name, description=s.description, parameters=s.parameters)
        for s in provider_tool_specs()
    ]


LIBRARY_TOOLS = [
    *_generated_search_tool_defs(),
    RESOLVE_DOI, SCRAPE_URL,
    READ_PDF, ANALYZE_IMAGE,
]

PIPELINE_RESULT_TOOLS = [
    LIST_PIPELINE_RESULTS, GET_PIPELINE_RESULT,
    GET_PIPELINE_KEYWORDS, GET_PIPELINE_ABSTRACT,
]

WORKFLOW_TOOLS = [LIST_WORKFLOWS, GET_WORKFLOW]

EXPORT_TOOLS = [EXPORT_RESULTS, GENERATE_REPORT]

ALL_TOOLS = KNOWLEDGE_TOOLS + LIBRARY_TOOLS + PIPELINE_RESULT_TOOLS + WORKFLOW_TOOLS + EXPORT_TOOLS
