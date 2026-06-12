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
            "search_type": {
                "type": "string",
                "enum": ["kw", "title", "freetext"],
                "default": "kw",
                "description": "Query mode: kw=subject/keyword, title=title-only, freetext=any field",
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
            "search_type": {
                "type": "string",
                "enum": ["kw", "title", "freetext"],
                "default": "kw",
                "description": "Query mode: kw=subject (IKT 2074), title=title (IKT 2058), freetext=anyword",
            },
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
            "search_type": {
                "type": "string",
                "enum": ["kw", "title", "freetext"],
                "default": "kw",
                "description": "Query mode: kw=anyword (Libero 'ku'), title=title (Libero 'k'), freetext=anyword",
            },
        },
        "required": ["terms"],
    },
)

SEARCH_CATALOG_TITLES = ToolDefinition(
    name="search_catalog_titles",
    description=(
        "Search bibliographic catalog for book records by title or keyword. "
        "Returns per-query lists of records (rsn, title, authors, year, "
        "dk_codes, rvk_codes, subjects). No GND/SWB/Lobid enrichment — "
        "pure catalog hits intended for title-list workflows. Each record "
        "includes `web_url` (catalog web link for that RSN) when a web "
        "record URL is configured. Always cite records as Markdown links: "
        "[title](web_url). Omit the link only when web_url is absent."
    ),
    parameters={
        "type": "object",
        "properties": {
            "terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of title queries",
            },
            "search_type": {
                "type": "string",
                "default": "title",
                "description": (
                    "Libero use-code or alias: 'title' (ti, default), "
                    "'kw'/'freetext' (ku), or raw codes like 'kb' (author), "
                    "'ke' (combined author), 'sk' (subjects), 'i' (ISBN)."
                ),
            },
            "max_results": {
                "type": "integer",
                "default": 25,
                "description": "Maximum records per query",
            },
        },
        "required": ["terms"],
    },
)

SEARCH_FINC = ToolDefinition(
    name="search_finc",
    description=(
        "Search a finc / VuFind-JSON library catalog (e.g. TU Freiberg finc "
        "solrproxy) for full bibliographic records. Preferred over search_catalog "
        "when the institution runs a finc instance. Choose the search axis via "
        "`search_type`: by subject/keyword, by title (one OR many — pass several "
        "titles in `terms` to look them all up in one call), or by author. "
        "`terms` is searched independently and the results are keyed per term. "
        "Each record has id, title, authors, subjects, formats, languages, series "
        "and web_url. Use `facets` (e.g. [\"udk_raw_de105\",\"rvk_facet\"]) to also "
        "get the DK/RVK classification distribution, and `filters` to scope by "
        "facet (VuFind syntax, e.g. {\"institution\": \"DE-105\"} or "
        "{\"id\": \"<record-id>\"} for one record). Use `availability` to "
        "restrict to physical holdings ('local'), licensed e-resources "
        "('online'), or open access ('free'). "
        "Always cite records as Markdown links: [title](web_url). "
        "Every listed record must include its link when web_url is present."
    ),
    parameters={
        "type": "object",
        "properties": {
            "terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One or more search terms, each searched independently (e.g. several book titles or author names in one call). Wrap a phrase in literal double quotes for an exact match (e.g. \"conrad hübler\"); URL-encoding is handled automatically.",
            },
            "search_type": {
                "type": "string",
                "enum": ["kw", "title", "subject", "author", "freetext", "dk", "rvk"],
                "default": "kw",
                "description": (
                    "Which field to search: subject = controlled subject/keyword "
                    "headings; title = words in the title (use for one or many "
                    "titles); author = author/contributor names; kw/freetext = "
                    "all fields; dk = search directly in the DK/UDK notation field "
                    "(udk_raw_de105, e.g. lookfor='DK 57' or 'qt 000'); rvk = search "
                    "directly in the RVK notation field (rvk_facet). For dk and rvk "
                    "types, udk_raw_de105 and rvk_facet facets are added automatically "
                    "so the classification distribution is always returned."
                ),
            },
            "filters": {
                "type": "object",
                "description": (
                    "Optional facet filters as key→value. Each entry is sent "
                    "as one filter[]=key:\"value\" parameter. Common keys: "
                    "institution (holding library, e.g. DE-105), udk_facet_de105 "
                    "(coarse DK group), rvk_facet, id (single record), format, "
                    "language."
                ),
            },
            "facets": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional facet fields to compute per term; buckets are "
                    "returned under each term's 'facets'. Use udk_raw_de105 for "
                    "numeric DK notations (e.g. 'dk 530.145'), rvk_facet for RVK, "
                    "dewey-hundreds for DDC."
                ),
            },
            "limit": {
                "type": "integer",
                "default": 20,
                "minimum": 0,
                "maximum": 100,
                "description": "Maximum records per term (0..100; 0 = facets only).",
            },
            "availability": {
                "type": "string",
                "enum": ["local", "online", "free"],
                "description": (
                    "Filter by holding type: 'local' = physical copy in the library "
                    "(Präsenzbestand/Ausleihbestand); 'online' = licensed electronic "
                    "resource; 'free' = open access / freely available online. "
                    "Omit to return all holdings."
                ),
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

SCRAPE_URL = ToolDefinition(
    name="scrape_url",
    description=(
        "Fetch a webpage and extract readable text. Removes scripts, styles, nav. "
        "Auto-detects PDF Content-Type and routes to read_pdf. Returns cleaned text and title."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch (e.g. 'https://example.com/article')"},
            "max_chars": {"type": "integer", "default": 10000, "description": "Max characters to return (truncates if longer)"},
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
            "max_chars": {"type": "integer", "default": 20000, "description": "Max characters to return (0 = no limit)"},
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


# ============================================================
# Tool Sets (grouped for agent use)
# ============================================================

KNOWLEDGE_TOOLS = [
    SEARCH_GND, GET_GND_ENTRY, GET_GND_BATCH,
    GET_SEARCH_CACHE, GET_DK_CACHE, STORE_SEARCH_RESULT,
    GET_CLASSIFICATION, GET_DB_STATS, SELECT_FROM_GND_POOL,
]

LIBRARY_TOOLS = [
    SEARCH_LOBID, SEARCH_SWB, SEARCH_CATALOG, SEARCH_CATALOG_TITLES, SEARCH_FINC,
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
