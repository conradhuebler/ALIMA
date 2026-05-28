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


def _parse_llm_json(text: str) -> dict:
    """Best-effort JSON extraction from an LLM response. Claude Generated (P-κ)."""
    import re as _re
    if not text:
        return {}
    # Strip markdown fences
    fence_match = _re.search(r"```(?:json)?\s*\n?(.*?)```", text, _re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    # Extract first balanced {…}
    match = _re.search(r"\{", text)
    if not match:
        return {}
    start = match.start()
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except (json.JSONDecodeError, ValueError):
                    return {}
    return {}


class ToolRegistry:
    """Registry mapping tool names to handler functions - Claude Generated"""

    def __init__(self, config_manager=None, llm_service=None):
        self._tools: Dict[str, ToolDefinition] = {}
        self._handlers: Dict[str, Callable] = {}
        self._config_manager = config_manager
        self._llm_service = llm_service
        self._knowledge_manager = None
        self._suggesters_initialized = False
        self._lobid = None
        self._swb = None
        self._biblio = None
        self._resolver = None
        self._presets: Dict[str, List[str]] = {}
        self._load_default_presets()

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

    def register_preset(self, name: str, tool_names: List[str]) -> None:
        """Register a named tool preset."""
        self._presets[name] = list(tool_names)
        logger.debug(f"Registered tool preset '{name}': {tool_names}")

    def get_preset(self, name: str) -> List[str]:
        """Get tool names for a preset. Returns empty list if unknown."""
        return list(self._presets.get(name, []))

    def list_presets(self) -> List[str]:
        """List available preset names."""
        return sorted(self._presets.keys())

    def _load_default_presets(self) -> None:
        """Load default presets from YAML file."""
        import pathlib
        preset_paths = [
            pathlib.Path("src/mcp/default_presets.yaml"),
            pathlib.Path.home() / ".config" / "alima" / "tool_presets.yaml",
        ]
        for ppath in preset_paths:
            if ppath.exists():
                try:
                    import yaml
                    with open(ppath, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                    for preset_name, tool_names in data.items():
                        if isinstance(tool_names, list):
                            self.register_preset(preset_name, tool_names)
                    logger.info(f"Loaded {len(data)} tool presets from {ppath}")
                except Exception as exc:
                    logger.warning(f"Could not load tool presets from {ppath}: {exc}")

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
            cat_cfg = None
            if self._config_manager is not None:
                try:
                    cat_cfg = self._config_manager.get_catalog_config()
                except Exception as e:
                    logger.debug(f"catalog_config unavailable: {e}")
            if cat_cfg is None:
                try:
                    from src.utils.config_manager import ConfigManager
                    cat_cfg = ConfigManager().get_catalog_config()
                except Exception as e:
                    logger.debug(f"ConfigManager fallback failed: {e}")
            if cat_cfg is not None:
                self._biblio = BiblioSuggester(
                    token=getattr(cat_cfg, "catalog_token", "") or "",
                    catalog_search_url=getattr(cat_cfg, "catalog_search_url", "") or "",
                    catalog_details=getattr(cat_cfg, "catalog_details_url", "") or "",
                )
                try:
                    web_search = getattr(cat_cfg, "catalog_web_search_url", "") or ""
                    web_record = getattr(cat_cfg, "catalog_web_record_url", "") or ""
                    if web_search:
                        self._biblio.extractor.WEB_SEARCH_URL = web_search
                        self._biblio.extractor.enable_web_fallback = True
                    if web_record:
                        self._biblio.extractor.WEB_RECORD_BASE_URL = web_record
                except Exception as e:
                    logger.debug(f"web fallback URL wiring failed: {e}")
            else:
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

    def _get_llm_service(self):
        """Lazy-init LlmService for image/PDF-OCR tools."""
        if self._llm_service is None:
            try:
                from src.llm.llm_service import LlmService
                self._llm_service = LlmService(
                    config_manager=self._config_manager,
                    lazy_initialization=True,
                )
            except Exception as exc:
                logger.warning(f"LlmService init failed: {exc}")
                return None
        return self._llm_service

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

    def _handle_search_lobid(self, terms: List[str], search_type: str = "kw") -> str:
        self._init_suggesters()
        if self._lobid is None:
            return json.dumps({"error": "LobidSuggester not available"})
        results = self._lobid.search(terms, search_type=search_type)
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

    def _handle_search_swb(self, terms: List[str], max_pages: int = 5, search_type: str = "kw") -> str:
        self._init_suggesters()
        if self._swb is None:
            return json.dumps({"error": "SWBSuggester not available"})
        results = self._swb.search(terms, max_pages=max_pages, search_type=search_type)
        serializable = {}
        for term, keywords in results.items():
            serializable[term] = {}
            for kw, data in keywords.items():
                serializable[term][kw] = {
                    k: list(v) if isinstance(v, set) else v
                    for k, v in data.items()
                }
        return json.dumps({"source": "swb", "results": serializable}, ensure_ascii=False)

    def _handle_search_catalog(self, terms: List[str], search_type: str = "kw") -> str:
        self._init_suggesters()
        if self._biblio is None:
            return json.dumps({"error": "BiblioSuggester not available"})
        results = self._biblio.search(terms, search_type=search_type)
        serializable = {}
        for term, keywords in results.items():
            serializable[term] = {}
            for kw, data in keywords.items():
                serializable[term][kw] = {
                    k: list(v) if isinstance(v, set) else v
                    for k, v in data.items()
                }
        return json.dumps({"source": "catalog", "results": serializable}, ensure_ascii=False)

    def _handle_search_catalog_titles(
        self,
        terms: List[str],
        search_type: str = "title",
        max_results: int = 25,
    ) -> str:
        self._init_suggesters()
        if self._biblio is None:
            return json.dumps({"error": "BiblioSuggester not available"})
        results = self._biblio.search_titles(
            terms, search_type=search_type, max_results=max_results
        )
        return json.dumps(
            {"source": "catalog_titles", "results": results},
            ensure_ascii=False,
        )

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

    def _handle_scrape_url(self, url: str, max_chars: int = 10000) -> str:
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            return json.dumps({"error": "requests + beautifulsoup4 required"})
        try:
            resp = requests.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }, timeout=30)
            resp.raise_for_status()
            content_type = (resp.headers.get("Content-Type") or "").lower()
            looks_pdf = "application/pdf" in content_type or url.lower().split("?")[0].endswith(".pdf")
            if looks_pdf:
                import tempfile
                from src.utils.pdf_extractor import extract_text as _pdf_extract
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(resp.content)
                    tmp_path = tmp.name
                try:
                    pdf_result = _pdf_extract(tmp_path, max_chars=max_chars or None)
                    return json.dumps({
                        "url": url,
                        "title": os.path.basename(url.split("?")[0]),
                        "text": pdf_result["text"],
                        "chars": pdf_result["chars"],
                        "source": "pdf",
                        "pdf": {
                            "pages": pdf_result["pages"],
                            "quality": pdf_result["quality"],
                            "extraction_source": pdf_result["source"],
                            "truncated": pdf_result["truncated"],
                        },
                    }, ensure_ascii=False)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            soup = BeautifulSoup(resp.content, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()
            title = (soup.find("title") or soup.find("h1"))
            title_text = title.get_text(strip=True) if title else ""
            main = soup.find("main") or soup.find("article") or soup.find("div", class_="content")
            body = soup.find("body")
            text = ""
            if main:
                text = main.get_text(separator="\n", strip=True)
            elif body:
                text = body.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)
            import re
            text = re.sub(r"\n\s*\n+", "\n\n", text)
            text = re.sub(r" +", " ", text)
            if len(text) > max_chars > 0:
                text = text[:max_chars] + "\n[…truncated]"
            return json.dumps({
                "url": url, "title": title_text,
                "text": text, "chars": len(text),
            }, ensure_ascii=False)
        except requests.RequestException as e:
            return json.dumps({"error": f"Fetch failed: {e}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ============================================================
    # Input-Beschaffung (P-η): PDF + Image
    # ============================================================

    def _handle_read_pdf(
        self,
        path: str,
        max_chars: int = 20000,
        ocr_fallback: bool = False,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        try:
            from src.utils.pdf_extractor import extract_text
        except Exception as exc:
            return json.dumps({"error": f"pdf_extractor import failed: {exc}"})
        llm_service = self._get_llm_service() if ocr_fallback else None
        try:
            result = extract_text(
                path,
                max_chars=max_chars or None,
                ocr_fallback=ocr_fallback,
                llm_service=llm_service,
                provider=provider,
                model=model,
            )
            return json.dumps({"path": path, **result}, ensure_ascii=False)
        except FileNotFoundError as exc:
            return json.dumps({"error": str(exc)})
        except Exception as exc:
            logger.error(f"read_pdf failed: {exc}")
            return json.dumps({"error": str(exc)})

    def _handle_analyze_image(
        self,
        path: str,
        prompt: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        try:
            from src.utils.image_analyzer import analyze, DEFAULT_PROMPT
        except Exception as exc:
            return json.dumps({"error": f"image_analyzer import failed: {exc}"})
        llm_service = self._get_llm_service()
        if llm_service is None:
            return json.dumps({"error": "LlmService unavailable — analyze_image requires Vision provider"})
        try:
            result = analyze(
                path,
                llm_service=llm_service,
                provider=provider,
                model=model,
                prompt=prompt or DEFAULT_PROMPT,
                temperature=temperature,
            )
            return json.dumps({"path": path, **result}, ensure_ascii=False)
        except FileNotFoundError as exc:
            return json.dumps({"error": str(exc)})
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        except Exception as exc:
            logger.error(f"analyze_image failed: {exc}")
            return json.dumps({"error": str(exc)})

    # ============================================================
    # Export & Reporting (P-θ)
    # ============================================================

    def _handle_export_results(
        self,
        format: str,
        source: str = "latest",
        output_path: Optional[str] = None,
        validate_rvk: bool = False,
    ) -> str:
        try:
            from src.utils import exporters
        except Exception as exc:
            return json.dumps({"error": f"exporters import failed: {exc}"})
        autosave_dir = self._get_autosave_dir()
        try:
            state = exporters.load_state(source, autosave_dir=autosave_dir)
        except (FileNotFoundError, ValueError) as exc:
            return json.dumps({"error": str(exc)})
        out_path = output_path or exporters.default_output_path(
            state, format, output_dir=autosave_dir
        )
        try:
            if format.lower() == "json":
                written = exporters.export(state, format, out_path, validate_rvk=validate_rvk)
            else:
                written = exporters.export(state, format, out_path)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        except Exception as exc:
            logger.error(f"export_results failed: {exc}")
            return json.dumps({"error": str(exc)})
        return json.dumps({
            "format": format,
            "output_path": written,
            "source_path": state.get("__source_path__"),
            "bytes": os.path.getsize(written) if os.path.isfile(written) else None,
        }, ensure_ascii=False)

    def _handle_generate_report(
        self,
        template: str,
        source: str = "latest",
        output_path: Optional[str] = None,
        build_pdf: bool = False,
    ) -> str:
        try:
            from src.utils import exporters
            from src.utils import report_renderer
        except Exception as exc:
            return json.dumps({"error": f"renderer import failed: {exc}"})
        autosave_dir = self._get_autosave_dir()
        try:
            state = exporters.load_state(source, autosave_dir=autosave_dir)
        except (FileNotFoundError, ValueError) as exc:
            return json.dumps({"error": str(exc)})
        out_path = output_path or exporters.default_output_path(
            state, "tex", output_dir=autosave_dir
        )
        try:
            result = report_renderer.render(
                template, state, out_path, build_pdf=build_pdf
            )
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        except Exception as exc:
            logger.error(f"generate_report failed: {exc}")
            return json.dumps({"error": str(exc)})
        return json.dumps({
            "template": template,
            "source_path": state.get("__source_path__"),
            **result,
        }, ensure_ascii=False)

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
    # Workflow Tool Handlers
    # ============================================================

    def _handle_list_workflows(self) -> str:
        try:
            from src.core.agents.workflow_loader import find_workflow_file
            workflows = []
            names = ["alima", "alima_classic", "catalog_search", "synonym_expansion", "batch_metadata", "title_list_search"]
            for name in names:
                path = find_workflow_file(name)
                if path:
                    try:
                        import yaml
                        with open(path, "r", encoding="utf-8") as f:
                            data = yaml.safe_load(f)
                        workflows.append({
                            "name": name,
                            "title": data.get("name", name),
                            "version": data.get("version", ""),
                            "description": data.get("description", ""),
                        })
                    except Exception:
                        workflows.append({"name": name, "title": name, "version": "", "description": ""})
            return json.dumps({"workflows": workflows}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _handle_get_workflow(self, name: str) -> str:
        try:
            from src.core.agents.workflow_loader import find_workflow_file
            path = find_workflow_file(name)
            if not path:
                return json.dumps({"error": f"Workflow '{name}' not found"})
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            steps = []
            for step in data.get("steps", []):
                steps.append({
                    "id": step.get("id", ""),
                    "type": step.get("type", ""),
                    "description": step.get("description", ""),
                    "enabled": step.get("enabled", True),
                    "depends_on": step.get("depends_on", []),
                    "inputs": list(step.get("inputs", {}).keys()),
                    "outputs": list(step.get("outputs", {}).keys()),
                })
            return json.dumps({
                "name": data.get("name", name),
                "version": data.get("version", ""),
                "description": data.get("description", ""),
                "steps": steps,
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _handle_select_from_gnd_pool(
        self,
        abstract: str,
        gnd_entries: list,
        chunk_size: int = 350,
        max_merged: int = 80,
    ) -> str:
        """Select relevant GND keywords from a large candidate pool via chunked LLM filtering.

        Splits entries into chunks, filters each for relevance against the abstract,
        merges and deduplicates results. Equivalent to the pipeline's selection_chunks step.
        Claude Generated (P-κ).
        """
        import re as _re

        llm = self._llm_service
        if llm is None:
            return json.dumps({"error": "select_from_gnd_pool requires LLM service"})

        if not gnd_entries:
            return json.dumps({"keywords": [], "count": 0, "chunks_processed": 0})

        # Project entries to minimal fields for token efficiency
        projected = []
        for e in gnd_entries:
            if isinstance(e, dict):
                projected.append({
                    "keyword": e.get("keyword", e.get("title", "")),
                    "gnd_id": e.get("gnd_id", ""),
                    "count": e.get("count", 1),
                })
            elif isinstance(e, str):
                projected.append({"keyword": e, "gnd_id": "", "count": 1})
        if not projected:
            return json.dumps({"keywords": [], "count": 0, "chunks_processed": 0})

        # Sort by count descending (most frequent first)
        projected.sort(key=lambda x: x.get("count", 1), reverse=True)

        # Split into chunks
        chunks = [projected[i:i + chunk_size] for i in range(0, len(projected), chunk_size)]
        merged = []
        seen_keywords = set()
        total_iterations = 0

        system_prompt = (
            "Deine Rolle als Bibliothekar:\n"
            "Du bist ein präziser und selektiver GND-Schlagwort-Experte mit folgenden Kernregeln:\n"
            "1. Strenge Relevanzprüfung: Wähle nur Schlagworte aus, die direkt zum Abstract passen.\n"
            "2. Ignorieren von Nicht-Relevanten: Alle Schlagworte ohne Bezug werden ausgeschlossen.\n"
            "3. Keine Ergänzungen: Nutze nur die vorgegebenen GND-Schlagworte.\n"
            "4. Bevorzuge Einträge mit höherer Trefferzahl (count-Feld) — sie sind katalogverifiziert.\n"
            "5. JSON-Ausgabe: Gib das Ergebnis immer als valides JSON-Objekt aus.\n"
            "6. Keine externen Quellen oder Tools — arbeite nur mit den gegebenen Daten."
        )

        user_template = (
            "Aufgabe: Filtere nur die relevanten Schlagworte aus der Teilliste.\n\n"
            "Abstract:\n{abstract}\n\n"
            "GND-Schlagworte (Teilliste {chunk_index}/{chunk_total}):\n{keywords}\n\n"
            'Ausgabeformat: {{"keywords": [{{"keyword": "...", "gnd_id": "..."}}]}}\n'
            "Keine Diskussion, nur JSON."
        )

        for i, chunk in enumerate(chunks):
            keywords_str = "\n".join(
                f'- {e["keyword"]}' + (f' (GND: {e["gnd_id"]})' if e.get("gnd_id") else '')
                + (f' [count: {e["count"]}]' if e.get("count", 1) > 1 else '')
                for e in chunk
            )
            # Truncate abstract for token control
            abstract_trunc = abstract[:3000] + ("..." if len(abstract) > 3000 else "")
            user_prompt = user_template.format(
                abstract=abstract_trunc,
                keywords=keywords_str,
                chunk_index=i + 1,
                chunk_total=len(chunks),
            )

            try:
                response = llm.generate_response(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    provider="",
                    model="",
                    temperature=0.01,
                    max_tokens=2048,
                )
                total_iterations += 1
            except Exception as exc:
                logger.warning(f"select_from_gnd_pool: chunk {i+1} LLM call failed: {exc}")
                continue

            # Parse JSON response — best-effort extraction
            parsed = _parse_llm_json(response)
            if parsed and "keywords" in parsed:
                for kw in parsed["keywords"]:
                    keyword = kw.get("keyword", "").strip()
                    if not keyword:
                        continue
                    kw_lower = keyword.lower()
                    if kw_lower not in seen_keywords:
                        seen_keywords.add(kw_lower)
                        merged.append({
                            "keyword": keyword,
                            "gnd_id": kw.get("gnd_id", ""),
                        })

        # Cap at max_merged
        if len(merged) > max_merged:
            merged = merged[:max_merged]

        return json.dumps({
            "keywords": merged,
            "count": len(merged),
            "chunks_processed": len(chunks),
            "total_iterations": total_iterations,
        }, ensure_ascii=False)

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
        self.register(tool_schemas.SELECT_FROM_GND_POOL, self._handle_select_from_gnd_pool)

        # Library tools
        self.register(tool_schemas.SEARCH_LOBID, self._handle_search_lobid)
        self.register(tool_schemas.SEARCH_SWB, self._handle_search_swb)
        self.register(tool_schemas.SEARCH_CATALOG, self._handle_search_catalog)
        self.register(tool_schemas.SEARCH_CATALOG_TITLES, self._handle_search_catalog_titles)
        self.register(tool_schemas.RESOLVE_DOI, self._handle_resolve_doi)
        self.register(tool_schemas.SCRAPE_URL, self._handle_scrape_url)
        self.register(tool_schemas.READ_PDF, self._handle_read_pdf)
        self.register(tool_schemas.ANALYZE_IMAGE, self._handle_analyze_image)

        # Export & Reporting tools
        self.register(tool_schemas.EXPORT_RESULTS, self._handle_export_results)
        self.register(tool_schemas.GENERATE_REPORT, self._handle_generate_report)

        # Pipeline result tools
        self.register(tool_schemas.LIST_PIPELINE_RESULTS, self._handle_list_pipeline_results)
        self.register(tool_schemas.GET_PIPELINE_RESULT, self._handle_get_pipeline_result)
        self.register(tool_schemas.GET_PIPELINE_KEYWORDS, self._handle_get_pipeline_keywords)
        self.register(tool_schemas.GET_PIPELINE_ABSTRACT, self._handle_get_pipeline_abstract)

        # Workflow tools
        self.register(tool_schemas.LIST_WORKFLOWS, self._handle_list_workflows)
        self.register(tool_schemas.GET_WORKFLOW, self._handle_get_workflow)

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
                (tool_schemas.SEARCH_CATALOG_TITLES, self._handle_search_catalog_titles),
                (tool_schemas.RESOLVE_DOI, self._handle_resolve_doi),
                (tool_schemas.SCRAPE_URL, self._handle_scrape_url),
                (tool_schemas.READ_PDF, self._handle_read_pdf),
                (tool_schemas.ANALYZE_IMAGE, self._handle_analyze_image),
            ],
            "pipeline": [
                (tool_schemas.LIST_PIPELINE_RESULTS, self._handle_list_pipeline_results),
                (tool_schemas.GET_PIPELINE_RESULT, self._handle_get_pipeline_result),
                (tool_schemas.GET_PIPELINE_KEYWORDS, self._handle_get_pipeline_keywords),
                (tool_schemas.GET_PIPELINE_ABSTRACT, self._handle_get_pipeline_abstract),
            ],
            "export": [
                (tool_schemas.EXPORT_RESULTS, self._handle_export_results),
                (tool_schemas.GENERATE_REPORT, self._handle_generate_report),
            ],
        }

        if tool_set == "all":
            self.register_all_tools()
            return

        if tool_set not in tool_map:
            raise ValueError(f"Unknown tool set: {tool_set}. Available: {list(tool_map.keys())}")

        for tool_def, handler in tool_map[tool_set]:
            self.register(tool_def, handler)
