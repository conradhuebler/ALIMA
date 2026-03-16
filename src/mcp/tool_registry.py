"""Tool Registry: Maps tool names to callable handlers - Claude Generated

Dispatches tool calls to existing ALIMA services (UnifiedKnowledgeManager, Suggesters, etc.).
All handlers return JSON-serializable strings for LLM consumption.
"""
import json
import logging
import os
import glob
from typing import Dict, Any, Optional, List, Callable
from dataclasses import asdict

from src.mcp.mcp_types import ToolDefinition
from src.mcp import tool_schemas

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry mapping tool names to handler functions - Claude Generated"""

    def __init__(self, config_manager=None):
        self._tools: Dict[str, ToolDefinition] = {}
        self._handlers: Dict[str, Callable] = {}
        self._config_manager = config_manager
        self._knowledge_manager = None
        self._suggesters_initialized = False
        self._lobid = None
        self._swb = None
        self._biblio = None
        self._resolver = None

    def register(self, tool_def: ToolDefinition, handler: Callable):
        """Register a tool with its handler."""
        self._tools[tool_def.name] = tool_def
        self._handlers[tool_def.name] = handler

    def get_tool_schemas(self, tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get JSON Schema definitions for LLM consumption."""
        if tool_names:
            return [self._tools[n].to_schema() for n in tool_names if n in self._tools]
        return [t.to_schema() for t in self._tools.values()]

    def get_tool_names(self) -> List[str]:
        """Get all registered tool names."""
        return list(self._tools.keys())

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool by name. Returns JSON string."""
        if tool_name not in self._handlers:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            result = self._handlers[tool_name](**arguments)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution error: {e}")
            return json.dumps({"error": str(e)})

    # ---- Lazy initialization of services ----

    def _get_knowledge_manager(self):
        """Lazy-init UnifiedKnowledgeManager singleton."""
        if self._knowledge_manager is None:
            from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
            self._knowledge_manager = UnifiedKnowledgeManager()
        return self._knowledge_manager

    def _init_suggesters(self):
        """Lazy-init suggester instances."""
        if self._suggesters_initialized:
            return
        try:
            from src.utils.suggesters.lobid_suggester import LobidSuggester
            self._lobid = LobidSuggester()
        except Exception as e:
            logger.warning(f"LobidSuggester init failed: {e}")
        try:
            from src.utils.suggesters.swb_suggester import SWBSuggester
            self._swb = SWBSuggester()
        except Exception as e:
            logger.warning(f"SWBSuggester init failed: {e}")
        try:
            from src.utils.suggesters.biblio_suggester import BiblioSuggester
            self._biblio = BiblioSuggester()
        except Exception as e:
            logger.warning(f"BiblioSuggester init failed: {e}")
        self._suggesters_initialized = True

    def _get_resolver(self):
        """Lazy-init DOI resolver."""
        if self._resolver is None:
            from src.utils.doi_resolver import UnifiedResolver
            self._resolver = UnifiedResolver()
        return self._resolver

    def _get_autosave_dir(self) -> str:
        """Get pipeline results directory."""
        try:
            from src.utils.pipeline_defaults import get_autosave_dir
            return str(get_autosave_dir(self._config_manager))
        except ImportError:
            return os.path.expanduser("~/Documents/ALIMA_Results")

    # ============================================================
    # Knowledge Tool Handlers
    # ============================================================

    def _handle_search_gnd(self, term: str, min_results: int = 3) -> str:
        km = self._get_knowledge_manager()
        entries = km.search_local_gnd(term, min_results=min_results)
        results = []
        for e in entries:
            results.append({
                "gnd_id": e.gnd_id,
                "title": e.title,
                "description": e.description,
                "synonyms": e.synonyms,
                "ddcs": e.ddcs,
            })
        return json.dumps({"term": term, "count": len(results), "entries": results}, ensure_ascii=False)

    def _handle_get_gnd_entry(self, gnd_id: str) -> str:
        km = self._get_knowledge_manager()
        entry = km.get_gnd_fact(gnd_id)
        if entry is None:
            return json.dumps({"error": f"GND entry '{gnd_id}' not found"})
        return json.dumps({
            "gnd_id": entry.gnd_id, "title": entry.title,
            "description": entry.description, "synonyms": entry.synonyms, "ddcs": entry.ddcs,
        }, ensure_ascii=False)

    def _handle_get_gnd_batch(self, gnd_ids: List[str]) -> str:
        km = self._get_knowledge_manager()
        entries = km.get_gnd_facts_batch(gnd_ids)
        results = {}
        for gnd_id, entry in entries.items():
            results[gnd_id] = {
                "title": entry.title, "description": entry.description,
                "synonyms": entry.synonyms, "ddcs": entry.ddcs,
            }
        return json.dumps({"count": len(results), "entries": results}, ensure_ascii=False)

    def _handle_get_search_cache(self, term: str, suggester_type: str) -> str:
        km = self._get_knowledge_manager()
        mapping = km.get_search_mapping(term, suggester_type)
        if mapping is None:
            return json.dumps({"cached": False, "term": term, "suggester_type": suggester_type})
        return json.dumps({
            "cached": True,
            "term": mapping.search_term,
            "suggester_type": mapping.suggester_type,
            "gnd_ids": mapping.found_gnd_ids,
            "classifications": mapping.found_classifications,
            "result_count": mapping.result_count,
            "last_updated": mapping.last_updated,
        }, ensure_ascii=False)

    def _handle_get_dk_cache(self, term: str) -> str:
        km = self._get_knowledge_manager()
        result = km.get_catalog_dk_cache(term)
        if result is None:
            return json.dumps({"cached": False, "term": term})
        titles, status, error_msg = result
        return json.dumps({
            "cached": True, "term": term, "status": status,
            "titles": titles, "error": error_msg,
        }, ensure_ascii=False)

    def _handle_store_search_result(self, term: str, suggester_type: str,
                                     gnd_ids: List[str] = None, classifications: List[Dict] = None) -> str:
        km = self._get_knowledge_manager()
        km.update_search_mapping(term, suggester_type,
                                 found_gnd_ids=gnd_ids or [],
                                 found_classifications=classifications or [])
        return json.dumps({"stored": True, "term": term, "suggester_type": suggester_type})

    def _handle_get_classification(self, code: str, classification_type: str) -> str:
        km = self._get_knowledge_manager()
        entry = km.get_classification_fact(code, classification_type)
        if entry is None:
            return json.dumps({"error": f"Classification '{code}' ({classification_type}) not found"})
        return json.dumps({
            "code": entry.code, "type": entry.type,
            "title": entry.title, "description": entry.description,
            "parent_code": entry.parent_code,
        }, ensure_ascii=False)

    def _handle_get_db_stats(self) -> str:
        km = self._get_knowledge_manager()
        stats = km.get_database_stats()
        return json.dumps(stats, ensure_ascii=False)

    # ============================================================
    # Library Tool Handlers
    # ============================================================

    def _handle_search_lobid(self, terms: List[str]) -> str:
        self._init_suggesters()
        if self._lobid is None:
            return json.dumps({"error": "LobidSuggester not available"})
        results = self._lobid.search(terms)
        # Convert sets to lists for JSON
        serializable = {}
        for term, keywords in results.items():
            serializable[term] = {}
            for kw, data in keywords.items():
                serializable[term][kw] = {
                    k: list(v) if isinstance(v, set) else v
                    for k, v in data.items()
                }
        return json.dumps({"source": "lobid", "results": serializable}, ensure_ascii=False)

    def _handle_search_swb(self, terms: List[str], max_pages: int = 5) -> str:
        self._init_suggesters()
        if self._swb is None:
            return json.dumps({"error": "SWBSuggester not available"})
        results = self._swb.search(terms, max_pages=max_pages)
        serializable = {}
        for term, keywords in results.items():
            serializable[term] = {}
            for kw, data in keywords.items():
                serializable[term][kw] = {
                    k: list(v) if isinstance(v, set) else v
                    for k, v in data.items()
                }
        return json.dumps({"source": "swb", "results": serializable}, ensure_ascii=False)

    def _handle_search_catalog(self, terms: List[str]) -> str:
        self._init_suggesters()
        if self._biblio is None:
            return json.dumps({"error": "BiblioSuggester not available"})
        results = self._biblio.search(terms)
        serializable = {}
        for term, keywords in results.items():
            serializable[term] = {}
            for kw, data in keywords.items():
                serializable[term][kw] = {
                    k: list(v) if isinstance(v, set) else v
                    for k, v in data.items()
                }
        return json.dumps({"source": "catalog", "results": serializable}, ensure_ascii=False)

    def _handle_resolve_doi(self, doi: str) -> str:
        resolver = self._get_resolver()
        success, metadata, abstract = resolver.resolve(doi)
        if not success:
            return json.dumps({"success": False, "doi": doi, "error": "Resolution failed"})
        return json.dumps({
            "success": True, "doi": doi,
            "metadata": metadata or {},
            "abstract": abstract or "",
        }, ensure_ascii=False, default=str)

    # ============================================================
    # Pipeline Result Tool Handlers
    # ============================================================

    def _handle_list_pipeline_results(self, limit: int = 20, search: str = None) -> str:
        results_dir = self._get_autosave_dir()
        if not os.path.isdir(results_dir):
            return json.dumps({"error": f"Results directory not found: {results_dir}", "files": []})

        pattern = os.path.join(results_dir, "*.json")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

        entries = []
        for fpath in files[:limit * 2]:  # Read more for filtering
            fname = os.path.basename(fpath)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                title = data.get("working_title", "")
                abstract_preview = (data.get("original_abstract", "") or "")[:150]
                step = data.get("pipeline_step_completed", "")
                timestamp = data.get("timestamp", "")

                if search and search.lower() not in fname.lower() and search.lower() not in (title or "").lower():
                    continue

                entries.append({
                    "filename": fname,
                    "working_title": title,
                    "abstract_preview": abstract_preview,
                    "step_completed": step,
                    "timestamp": timestamp,
                })
                if len(entries) >= limit:
                    break
            except (json.JSONDecodeError, OSError):
                continue

        return json.dumps({"directory": results_dir, "count": len(entries), "files": entries}, ensure_ascii=False)

    def _handle_get_pipeline_result(self, filename: str) -> str:
        results_dir = self._get_autosave_dir()
        fpath = os.path.join(results_dir, filename)
        if not os.path.isfile(fpath):
            return json.dumps({"error": f"File not found: {filename}"})
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return json.dumps(data, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": f"Failed to load {filename}: {e}"})

    def _handle_get_pipeline_keywords(self, filename: str) -> str:
        results_dir = self._get_autosave_dir()
        fpath = os.path.join(results_dir, filename)
        if not os.path.isfile(fpath):
            return json.dumps({"error": f"File not found: {filename}"})
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Extract keywords from final analysis
            keywords = []
            if data.get("final_llm_analysis"):
                keywords = data["final_llm_analysis"].get("extracted_gnd_keywords", [])
            elif data.get("initial_llm_call_details"):
                keywords = data["initial_llm_call_details"].get("extracted_gnd_keywords", [])
            dk = data.get("dk_classifications", [])
            return json.dumps({
                "filename": filename,
                "working_title": data.get("working_title", ""),
                "gnd_keywords": keywords,
                "dk_classifications": dk,
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"Failed to extract keywords from {filename}: {e}"})

    def _handle_get_pipeline_abstract(self, filename: str) -> str:
        results_dir = self._get_autosave_dir()
        fpath = os.path.join(results_dir, filename)
        if not os.path.isfile(fpath):
            return json.dumps({"error": f"File not found: {filename}"})
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return json.dumps({
                "filename": filename,
                "working_title": data.get("working_title", ""),
                "abstract": data.get("original_abstract", ""),
                "input_type": data.get("input_type", ""),
                "source_value": data.get("source_value", ""),
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"Failed to load abstract from {filename}: {e}"})

    # ============================================================
    # Registry Setup
    # ============================================================

    def register_all_tools(self):
        """Register all available tools with their handlers - Claude Generated"""
        # Knowledge tools
        self.register(tool_schemas.SEARCH_GND, self._handle_search_gnd)
        self.register(tool_schemas.GET_GND_ENTRY, self._handle_get_gnd_entry)
        self.register(tool_schemas.GET_GND_BATCH, self._handle_get_gnd_batch)
        self.register(tool_schemas.GET_SEARCH_CACHE, self._handle_get_search_cache)
        self.register(tool_schemas.GET_DK_CACHE, self._handle_get_dk_cache)
        self.register(tool_schemas.STORE_SEARCH_RESULT, self._handle_store_search_result)
        self.register(tool_schemas.GET_CLASSIFICATION, self._handle_get_classification)
        self.register(tool_schemas.GET_DB_STATS, self._handle_get_db_stats)

        # Library tools
        self.register(tool_schemas.SEARCH_LOBID, self._handle_search_lobid)
        self.register(tool_schemas.SEARCH_SWB, self._handle_search_swb)
        self.register(tool_schemas.SEARCH_CATALOG, self._handle_search_catalog)
        self.register(tool_schemas.RESOLVE_DOI, self._handle_resolve_doi)

        # Pipeline result tools
        self.register(tool_schemas.LIST_PIPELINE_RESULTS, self._handle_list_pipeline_results)
        self.register(tool_schemas.GET_PIPELINE_RESULT, self._handle_get_pipeline_result)
        self.register(tool_schemas.GET_PIPELINE_KEYWORDS, self._handle_get_pipeline_keywords)
        self.register(tool_schemas.GET_PIPELINE_ABSTRACT, self._handle_get_pipeline_abstract)

        logger.info(f"Registered {len(self._tools)} MCP tools")

    def register_tool_set(self, tool_set: str):
        """Register a specific tool set: 'knowledge', 'library', 'pipeline', 'all' - Claude Generated"""
        tool_map = {
            "knowledge": [
                (tool_schemas.SEARCH_GND, self._handle_search_gnd),
                (tool_schemas.GET_GND_ENTRY, self._handle_get_gnd_entry),
                (tool_schemas.GET_GND_BATCH, self._handle_get_gnd_batch),
                (tool_schemas.GET_SEARCH_CACHE, self._handle_get_search_cache),
                (tool_schemas.GET_DK_CACHE, self._handle_get_dk_cache),
                (tool_schemas.STORE_SEARCH_RESULT, self._handle_store_search_result),
                (tool_schemas.GET_CLASSIFICATION, self._handle_get_classification),
                (tool_schemas.GET_DB_STATS, self._handle_get_db_stats),
            ],
            "library": [
                (tool_schemas.SEARCH_LOBID, self._handle_search_lobid),
                (tool_schemas.SEARCH_SWB, self._handle_search_swb),
                (tool_schemas.SEARCH_CATALOG, self._handle_search_catalog),
                (tool_schemas.RESOLVE_DOI, self._handle_resolve_doi),
            ],
            "pipeline": [
                (tool_schemas.LIST_PIPELINE_RESULTS, self._handle_list_pipeline_results),
                (tool_schemas.GET_PIPELINE_RESULT, self._handle_get_pipeline_result),
                (tool_schemas.GET_PIPELINE_KEYWORDS, self._handle_get_pipeline_keywords),
                (tool_schemas.GET_PIPELINE_ABSTRACT, self._handle_get_pipeline_abstract),
            ],
        }

        if tool_set == "all":
            self.register_all_tools()
            return

        if tool_set not in tool_map:
            raise ValueError(f"Unknown tool set: {tool_set}. Available: {list(tool_map.keys())}")

        for tool_def, handler in tool_map[tool_set]:
            self.register(tool_def, handler)
