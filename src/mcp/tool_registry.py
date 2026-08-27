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
from src.core.url_utils import gnd_url, swb_ppn_url
from src.mcp._tool_generation import ToolGenerationMixin

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


class ToolRegistry(ToolGenerationMixin):
    """Registry mapping tool names to handler functions - Claude Generated"""

    def __init__(self, config_manager=None, llm_service=None):
        self._tools: Dict[str, ToolDefinition] = {}
        self._handlers: Dict[str, Callable] = {}
        self._config_manager = config_manager
        self._llm_service = llm_service
        self._knowledge_manager = None
        # All search sources — finc included since WP P2.2 — build through the
        # search factory from PluginInstanceConfig (retires the MetaSuggester
        # primaries + the CatalogConfig mirror). Built providers are memoised per
        # (instance_id, cache) here; ``refresh()`` clears it. - Claude Generated
        self._provider_cache: Dict[tuple, Any] = {}
        # Lookup plugins likewise memoised per instance (webindex e.g. opens its
        # own SQLite store — a fresh build per tool call would open a new
        # connection each time); ``refresh()`` clears it. - Claude Generated
        self._lookup_cache: Dict[str, Any] = {}
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


    def _alima_config(self):
        """Load the full AlimaConfig via the injected config manager (else global).
        Returns None on failure. - Claude Generated"""
        try:
            cm = self._config_manager
            if cm is None:
                from src.utils.config_manager import ConfigManager
                cm = ConfigManager()
            return cm.load_config()
        except Exception:
            return None

    def _instance_for(self, provider_id):
        """Resolve the primary configured instance for a search-provider type
        (synthesising a default when the config knows nothing about it). - Claude Generated"""
        from src.core.search.service import resolve_gnd_instances

        for inst in resolve_gnd_instances([provider_id], config=self._alima_config()):
            if inst.provider_id == provider_id:
                return inst
        return None

    def _provider_for(self, inst, cache=False):
        """Build (and memoise) the factory provider for an instance. - Claude Generated"""
        if getattr(self, "_provider_cache", None) is None:
            self._provider_cache = {}
        key = (getattr(inst, "instance_id", getattr(inst, "provider_id", "?")), bool(cache))
        provider = self._provider_cache.get(key)
        if provider is None:
            from src.core.search.factory import build_provider

            provider = build_provider(inst, cache=cache, ukm=self._get_knowledge_manager())
            self._provider_cache[key] = provider
        return provider

    def _lookup_for(self, inst):
        """Build (and memoise) the lookup plugin for an instance. - Claude Generated"""
        if getattr(self, "_lookup_cache", None) is None:
            self._lookup_cache = {}
        key = getattr(inst, "instance_id", getattr(inst, "provider_id", "?"))
        plugin = self._lookup_cache.get(key)
        if plugin is None:
            from src.core.plugins.category import get_category

            plugin = get_category("lookup").build(inst)
            self._lookup_cache[key] = plugin
        return plugin

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
            results.append(self._gnd_entry_dict(
                e.gnd_id, e.title, e.description, e.synonyms, e.ddcs,
                getattr(e, "ppn", ""),
            ))
        return json.dumps({"term": term, "count": len(results), "entries": results}, ensure_ascii=False)

    @staticmethod
    def _gnd_entry_dict(gnd_id, title, description, synonyms, ddcs, ppn="") -> Dict[str, Any]:
        """Serialise a GND entry with pre-formatted links for the chat agent.

        Claude Generated. Adds a ready ``url`` (d-nb.info, canonical) and, when a
        PPN is stored, an additional ``swb_url`` so the LLM never builds a GND URL
        itself (and never mistakes a GND id for a catalog RSN).
        """
        d: Dict[str, Any] = {
            "gnd_id": gnd_id, "title": title, "description": description,
            "synonyms": synonyms, "ddcs": ddcs,
        }
        _url = gnd_url(gnd_id)
        if _url:
            d["url"] = _url
        _swb = swb_ppn_url(str(ppn or ""))
        if _swb:
            d["swb_url"] = _swb
        return d

    def _handle_get_gnd_entry(self, gnd_id: str) -> str:
        km = self._get_knowledge_manager()
        entry = km.get_gnd_fact(gnd_id)
        if entry is None:
            return json.dumps({"error": f"GND entry '{gnd_id}' not found"})
        return json.dumps(self._gnd_entry_dict(
            entry.gnd_id, entry.title, entry.description, entry.synonyms,
            entry.ddcs, getattr(entry, "ppn", ""),
        ), ensure_ascii=False)

    def _handle_get_gnd_batch(self, gnd_ids: List[str]) -> str:
        km = self._get_knowledge_manager()
        entries = km.get_gnd_facts_batch(gnd_ids)
        results = {}
        for gnd_id, entry in entries.items():
            val = {
                "title": entry.title, "description": entry.description,
                "synonyms": entry.synonyms, "ddcs": entry.ddcs,
            }
            # Claude Generated - pre-formatted links (see _gnd_entry_dict).
            _url = gnd_url(gnd_id)
            if _url:
                val["url"] = _url
            _swb = swb_ppn_url(str(getattr(entry, "ppn", "") or ""))
            if _swb:
                val["swb_url"] = _swb
            results[gnd_id] = val
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

    def _handle_about_alima(self) -> str:
        """Project facts from the single source of truth (``src/core/about.py``).

        Pure lookup, no config and no DB — asking what ALIMA is must work even
        when nothing else is reachable. - Claude Generated
        """
        from src.core.about import about_payload

        return json.dumps(about_payload(), ensure_ascii=False)

    def _handle_list_plugins(self, category: str = None, include_disabled: bool = False) -> str:
        """List active plugins (search providers + input sources) with self-docs.

        Answers "which plugins are active?" from the real plugin registry — this is
        distinct from workflows (``list_workflows``). Secrets are never returned;
        only the *names* of configured non-secret settings. - Claude Generated"""
        import src.core.search  # noqa: F401  register search category
        import src.utils.input_sources  # noqa: F401  register input category
        from src.core.plugins import get_category, list_categories
        from src.utils.config_manager import ConfigManager

        try:
            cfg = (self._config_manager or ConfigManager()).load_config()
        except Exception as e:
            return json.dumps({"error": f"config not loadable: {e}"})

        cats = [category] if category else list_categories()
        out = {}
        for cat in cats:
            try:
                adapter = get_category(cat)
            except KeyError:
                continue
            insts = cfg.instances_for(cat) if include_disabled else cfg.enabled_instances_for(cat)
            items = []
            for p in insts:
                try:
                    meta = adapter.type_meta(p.provider_id)
                except Exception:
                    meta = None
                # Schema-based secret exclusion (ConfigField.secret) with the
                # name-heuristic kept as fallback for unknown keys. - Claude Generated
                secret_keys = {
                    f.key for f in (getattr(meta, "config_fields", []) or [])
                    if getattr(f, "secret", False)
                }
                cfg_keys = [
                    k for k, v in (p.settings or {}).items()
                    if v
                    and k not in secret_keys
                    and not any(s in k.lower() for s in ("token", "key", "secret", "password"))
                ]
                items.append({
                    "instance_id": p.instance_id,
                    "type": p.provider_id,
                    "label": p.display_label(),
                    "enabled": p.enabled,
                    "is_primary": p.is_primary,
                    "usage_hint": p.usage_hint,
                    "capabilities": list(getattr(meta, "capabilities", []) or []),
                    "description": meta.doc.description if meta else "",
                    "input": meta.doc.input if meta else "",
                    "output": meta.doc.output if meta else "",
                    "configured_settings": cfg_keys,
                })
            out[cat] = items
        return json.dumps(
            {
                "plugins": out,
                "note": "Plugins (search providers + input sources) are distinct from "
                        "workflows; use list_workflows for workflows.",
            },
            ensure_ascii=False,
        )

    # ============================================================
    # Library Tool Handlers
    # ============================================================

    @staticmethod
    def _serialize_result_row(data: Dict) -> Dict:
        """JSON-safe copy of one canonical result row: top-level sets → lists,
        nested ``classifications`` code sets → lists. - Claude Generated"""
        row = {k: list(v) if isinstance(v, set) else v for k, v in data.items()}
        if isinstance(row.get("classifications"), dict):
            row["classifications"] = {
                system: list(codes) if isinstance(codes, set) else codes
                for system, codes in row["classifications"].items()
            }
        return row

    @staticmethod
    def _serialize_suggester_results(results: Dict) -> Dict:
        """Convert per-term suggester results (with sets) to JSON-safe dicts - Claude Generated

        Also pre-formats canonical d-nb.info URLs for every GND id (``gnd_urls``)
        so the chat agent links Lobid/SWB hits without constructing a URL itself.
        """
        serializable = {}
        for term, keywords in results.items():
            serializable[term] = {}
            for kw, data in keywords.items():
                row = ToolRegistry._serialize_result_row(data)
                gnd_urls = [
                    u for g in (row.get("gnd_ids") or [])
                    if (u := gnd_url(str(g)))
                ]
                if gnd_urls:
                    row["gnd_urls"] = gnd_urls
                serializable[term][kw] = row
        return serializable

    # search_lobid / search_swb / search_catalog / search_catalog_titles handlers
    # are generated from provider ProviderToolSpecs — see _generated_search_tools().

    def _enabled_input_settings(self):
        """Map ``provider_id -> settings`` for enabled input-source instances."""
        try:
            cm = self._config_manager
            if cm is None:
                from src.utils.config_manager import ConfigManager
                cm = ConfigManager()
            out = {}
            for p in cm.load_config().enabled_instances_for("input_source"):
                out.setdefault(p.provider_id, dict(p.settings or {}))
            return out
        except Exception:
            return {}

    def _handle_resolve_doi(self, doi: str) -> str:
        """Return the COMPLETE metadata from every enabled DOI source, plus the
        first abstract found (convenience). The old handler returned only a curated
        subset from the first source — this forwards all data each source gives. - Claude Generated"""
        from src.utils.input_sources import get_input_source

        settings_map = self._enabled_input_settings()
        doi_types = ["doi_crossref", "doi_openalex", "doi_datacite"]
        active = [t for t in doi_types if t in settings_map] or doi_types

        sources = {}
        best_abstract = ""
        for t in active:
            short = t.replace("doi_", "")
            try:
                src = get_input_source(t)(**settings_map.get(t, {}))
                res = src.mcp_execute(doi)
            except Exception as e:
                res = {"source": short, "success": False, "error": str(e)}
            sources[short] = res
            md = res.get("metadata") or {}
            if not best_abstract:
                best_abstract = md.get("abstract") or ""
        any_ok = any(v.get("success") for v in sources.values())
        return json.dumps(
            {"doi": doi, "success": any_ok, "abstract": best_abstract, "sources": sources},
            ensure_ascii=False, default=str,
        )

    def _handle_scrape_url(self, url: str, max_chars: int = 0) -> str:
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            return json.dumps({"error": "requests + beautifulsoup4 required"})
        try:
            # LLM-supplied URL → the shared guarded fetch (net_guard: strict SSRF
            # guard, redirects re-checked per hop, body capped) used by the
            # url_fetch input source too. - Claude Generated
            from src.utils.input_sources.url_fetch import fetch_guarded_response

            resp = fetch_guarded_response(url, timeout=30)
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
            # Strip only non-content (code); keep the full page text so the tool
            # forwards everything, not just a main-content guess. - Claude Generated
            for tag in soup(["script", "style"]):
                tag.decompose()
            title = (soup.find("title") or soup.find("h1"))
            title_text = title.get_text(strip=True) if title else ""
            body = soup.find("body") or soup
            text = body.get_text(separator="\n", strip=True)
            import re
            text = re.sub(r"\n\s*\n+", "\n\n", text)
            text = re.sub(r" +", " ", text)
            full_chars = len(text)
            truncated = False
            if max_chars and full_chars > max_chars > 0:
                text = text[:max_chars] + "\n[…truncated]"
                truncated = True
            return json.dumps({
                "url": url, "title": title_text,
                "text": text, "chars": len(text),
                "full_chars": full_chars, "truncated": truncated,
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
        max_chars: int = 0,
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

    def _handle_execute_workflow(self, workflow_id: str, inputs: Dict[str, Any] = None) -> str:
        """Executes a named v4 workflow and returns the execution report. - Claude Generated"""
        try:
            from src.core.agents.workflow_loader import load_workflow
            from src.core.agents.workflow_executor import WorkflowExecutor
            from src.core.agents.shared_context import SharedContext

            # 1. Load the workflow definition
            workflow = load_workflow(workflow_id)
            if workflow is None:
                return json.dumps({"error": f"Workflow '{workflow_id}' not found"})

            # 2. Setup context
            context = SharedContext()
            if inputs:
                context.update(inputs)

            # 3. Execute the workflow
            # We reuse the current registry for any tools the sub-workflow needs.
            executor = WorkflowExecutor(
                llm_service=self._llm_service,
                tool_registry=self
            )
            report = executor.run(workflow, context)

            # Return the report. report is an ExecutionReport object.
            from dataclasses import asdict
            return json.dumps(asdict(report), ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"execute_workflow failed for {workflow_id}: {e}")
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

    def _handle_rvk_lookup(
        self,
        keywords: list,
        abstract: str = "",
        dk_codes: Optional[list] = None,
        max_results: int = 8,
    ) -> str:
        """Find validated RVK notations for subject keywords - Claude Generated.

        Stateless wrapper over the pipeline's RVK anchor machinery
        (``PipelineStepExecutor``): derives RVK anchors, runs the catalog RVK
        search + official RVK-API validation, and returns a deterministically
        ranked shortlist of authority-backed RVK candidates. Lets the
        classification LLM pull RVK on demand (e.g. only for WiWi in the
        Freiberg workflow) instead of running it unconditionally in dk_collect.

        Note: runs its own catalog search (independent of dk_collect); the
        catalog layer caches per term. ``alima_manager`` is None → deterministic
        scoring only (no extra LLM call inside the tool).
        """
        clean_keywords = [str(k).strip() for k in (keywords or []) if str(k).strip()]
        if not clean_keywords:
            return json.dumps({"rvk": [], "count": 0})

        try:
            from src.utils.pipeline_utils import PipelineStepExecutor
            from src.utils.config_manager import ConfigManager

            config_manager = self._config_manager or ConfigManager()
            # Real cache_manager (WP P3): the RVK search/validate calls underneath
            # are plugin-routed + cached_call-wrapped, so they now share the WP2 raw
            # cache with the classic pipeline and the rvk_search/rvk_validate agent
            # tools. Safe since P3's prerequisite (the rvk_validate cache-shape
            # collision, C2) is fixed — both writers store the outer dict. - Claude Generated
            executor = PipelineStepExecutor(
                alima_manager=None,
                cache_manager=self._get_knowledge_manager(),
                logger=logger,
                config_manager=config_manager,
            )

            try:
                anchors = executor._derive_rvk_anchor_keywords(
                    clean_keywords,
                    original_abstract=abstract or "",
                )
            except Exception as exc:
                logger.warning(f"rvk_lookup: anchor derivation failed: {exc}")
                anchors = None

            # strict_gnd_validation=False: the calling LLM often passes plain
            # subject terms (no "(GND-ID: …)" suffix). Strict mode would drop
            # all of them → empty search → no RVK. Plain terms are searched
            # directly; the RVK-API fallback still validates the results. - Claude Generated
            dk_result = executor.execute_notation_search(
                keywords=clean_keywords,
                rvk_anchor_keywords=anchors,
                rvk_enabled=True,
                strict_gnd_validation=False,
            )
            prep = executor.prepare_notation_classification_context(
                dk_result.get("classifications", []),
                original_abstract=abstract or "",
                rvk_anchor_keywords=anchors,
                include_rvk=True,
            )

            # Optional DK-context hint for deterministic ranking (no LLM).
            abstract_for_scoring = abstract or ""
            clean_dk = [str(c).strip() for c in (dk_codes or []) if str(c).strip()]
            if clean_dk:
                try:
                    dk_profile = executor._build_dk_semantic_profile(
                        clean_dk, prep["results_with_titles"]
                    )
                    if dk_profile:
                        abstract_for_scoring = f"{abstract_for_scoring}\n\nDK-Profil:\n{dk_profile}"
                except Exception as exc:
                    logger.debug(f"rvk_lookup: dk profile failed: {exc}")

            shortlist = executor._build_rvk_scoring_shortlist(
                prep["results_with_titles"],
                abstract_for_scoring,
                rvk_anchor_keywords=anchors,
                max_standard=max(1, int(max_results or 8)),
            )

            candidates = []
            for cand in shortlist[: max(1, int(max_results or 8))]:
                notation = str(cand.get("dk", "")).strip()
                if not notation:
                    continue
                candidates.append({
                    "notation": f"RVK {notation}",
                    "label": cand.get("label", ""),
                    "ancestor_path": cand.get("ancestor_path", ""),
                    "validation_status": cand.get("rvk_validation_status", "standard"),
                    "source": cand.get("source", "catalog"),
                    "count": int(cand.get("count", 0) or 0),
                    "anchor_hits": int(cand.get("_anchor_hit_count", 0) or 0),
                    "score": int(cand.get("_score", 0) or 0),
                })

            return json.dumps(
                {"rvk": candidates, "count": len(candidates), "anchors": anchors or []},
                ensure_ascii=False,
            )
        except Exception as exc:
            logger.error(f"rvk_lookup failed: {exc}")
            return json.dumps({"error": str(exc), "rvk": [], "count": 0})

    # ============================================================
    # Registry Setup
    # ============================================================

    # ============================================================
    # Library search tools — generated from provider specs (P3)
    # ============================================================
    def _generated_search_tools(self):
        """Build ``(ToolDefinition, handler)`` for every enabled search *instance*.

        Schemas come from each provider's ``ProviderToolSpec``; tools are emitted
        per configured instance (``PluginInstanceConfig``) so several instances of
        one type (e.g. two finc endpoints) each get their own tool. The *primary*
        instance of a type keeps the canonical tool name + handler (behaviour
        unchanged from the single-instance case); additional instances get a
        unique name and a factory-built handler. Each instance's ``usage_hint`` is
        appended to the tool description so an agent can steer between siblings.
        Disabled instances/types are skipped (config-driven selectability). - Claude Generated"""
        from src.core.search import provider_tool_specs

        specs_by_provider = {}
        for spec in provider_tool_specs():
            specs_by_provider.setdefault(spec.provider_id, []).append(spec)

        # instances grouped per provider type, in config order
        by_type = {}
        for inst in self._search_instances():
            by_type.setdefault(inst.provider_id, []).append(inst)

        # Canonical instances get the nuanced handler (gnd_url enrichment,
        # agent_view, non-default raw passthrough) regardless of type: it is
        # factory-backed, so it answers from the instance's *own* provider class —
        # a copied plugin included, finc now too (WP P2.2). - Claude Generated
        tools = []
        used_names = set()
        for provider_id, instances in by_type.items():
            specs = specs_by_provider.get(provider_id)
            if not specs:
                continue  # gnd_local/sru declare no MCP tool
            canonical = self._canonical_instance(instances)
            for inst in instances:
                is_canonical = inst is canonical
                for spec in specs:
                    if is_canonical:
                        name = spec.name
                        handler = self._make_search_handler(spec, inst)
                    else:
                        name = self._instance_tool_name(spec.name, inst)
                        handler = self._make_instance_handler(spec, inst)
                    if name in used_names:
                        # Cross-type collision: a copied code plugin whose
                        # mcp_tool_specs kept the blueprint's tool name must not
                        # shadow the built-in — suffix it instead. - Claude Generated
                        name = self._instance_tool_name(spec.name, inst)
                        handler = self._make_instance_handler(spec, inst)
                        if name in used_names:
                            logger.warning(
                                "Skipping duplicate generated tool '%s' (instance '%s')",
                                name, inst.instance_id,
                            )
                            continue
                    used_names.add(name)
                    tools.append(
                        (
                            ToolDefinition(
                                name=name,
                                description=self._describe_with_hint(spec.description, inst),
                                parameters=spec.parameters,
                            ),
                            handler,
                        )
                    )
        return tools

    def _search_instances(self):
        """Enabled search-provider instances driving tool generation.

        The per-instance ``plugins`` config is the only source; an unreadable
        config yields no search tools. The former ``SearchProviderConfig``-gated
        fallback (one primary per registered type) was unreachable in production
        — a readable config always returned above it — and existed only for
        minimal test stubs, which now bring a real config. - Claude Generated"""
        cfg = self._alima_config()
        return cfg.enabled_instances_for("search_provider") if cfg is not None else []

    @staticmethod
    def _canonical_instance(instances):
        """The instance that owns the canonical tool name: the primary, else the first."""
        for inst in instances:
            if getattr(inst, "is_primary", False):
                return inst
        return instances[0]

    @staticmethod
    def _instance_tool_name(base_name, inst):
        import re
        suffix = re.sub(r"[^a-z0-9]+", "_", str(inst.instance_id).lower()).strip("_")
        return f"{base_name}_{suffix}" if suffix else base_name

    @staticmethod
    def _describe_with_hint(description, inst):
        hint = (getattr(inst, "usage_hint", "") or "").strip()
        return f"{description}\n\nInstanz-Hinweis: {hint}" if hint else description

    # ============================================================
    # Input-source tools — generated per enabled input instance
    # ============================================================
    def _handle_aggregate_gnd_results(self, terms, sources=None, search_type="kw", max_pages=5):
        """Aggregate cached raw responses into a ranked pool (counter + provenance).

        Transform-on-read consumer of the WP2 raw cache: derives the pipeline view
        from raw via each source's transform + gnd_search_core. - Claude Generated
        """
        from src.core.search.aggregate import aggregate_gnd_results
        from src.core.search.factory import enabled_gnd_provider_ids
        from src.core.search.provider import raw_cache_params_for

        derived_default = False
        if not sources:
            # Default = whatever the operator enabled (the Plugins-tab gate that
            # already governs the classic path + tool generation), so an external
            # GND plugin appears in the pool's provenance. ``None`` = config
            # unreadable → keep the legacy list; ``[]`` = every GND source disabled
            # → aggregate nothing (never collapse those two). - Claude Generated
            ids = enabled_gnd_provider_ids(config=self._alima_config())
            sources = ids if ids is not None else ["lobid", "swb", "catalog"]
            derived_default = ids is not None
        km = self._get_knowledge_manager()

        transform_by_source = {}
        params_by_source = {}
        for src in sources:
            transform = self._source_transform(src)
            if transform is None:
                continue
            transform_by_source[src] = transform
            params_by_source[src] = raw_cache_params_for(
                src, search_type=search_type, max_pages=max_pages
            )

        effective = list(transform_by_source.keys())
        if derived_default and effective != ["lobid", "swb", "catalog"]:
            # Provenance/source_count deviates from the historical default — say so
            # once, else a changed ranking is archaeology (the per-source drop above
            # only logs at debug). Silent on a standard config. - Claude Generated
            logger.info("aggregate_gnd_results: config-derived sources %s", effective)

        if not effective:
            # An empty pool here is NOT "nothing found" — there was nothing to
            # aggregate. Returning the plain empty shape lets an agent report
            # "keine Treffer", i.e. a plausible but false answer. Say why.
            # ``pool``/``sources`` stay present so structural readers don't KeyError.
            # - Claude Generated
            reason = (
                "no GND search source is enabled (all search plugins disabled in the "
                "Plugins settings)"
                if derived_default
                else f"none of the requested sources resolved to an enabled provider: {sources}"
            )
            logger.warning("aggregate_gnd_results: nothing to aggregate — %s", reason)
            return json.dumps(
                {"pool": [], "sources": [], "missing": {}, "terms_map": {},
                 "error": f"Cannot aggregate: {reason}. This is NOT a zero-hit result."},
                ensure_ascii=False,
            )

        out = aggregate_gnd_results(
            terms, effective, km, transform_by_source,
            params_by_source=params_by_source,
        )
        return json.dumps(out, ensure_ascii=False, default=str)

    def refresh(self):
        """Rebuild all tools from the *current* config (runtime plugin toggle).

        Clears the tool tables first (so tools of now-disabled plugins actually
        disappear — ``register`` only overwrites), force-reloads the config cache
        (so a GUI/disk edit is picked up), resets lazily-built suggesters (so
        changed endpoints take effect), then re-registers. Lets a running chat
        agent see plugin enable/disable + config changes without a restart. - Claude Generated"""
        try:
            cm = self._config_manager
            if cm is None:
                from src.utils.config_manager import ConfigManager
                cm = ConfigManager()
            cm.load_config(force_reload=True)
        except Exception:
            pass
        self._tools.clear()
        self._handlers.clear()
        self._provider_cache = {}
        self._lookup_cache = {}
        self.register_all_tools()

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
        self.register(tool_schemas.RVK_LOOKUP, self._handle_rvk_lookup)
        self.register(tool_schemas.AGGREGATE_GND_RESULTS, self._handle_aggregate_gnd_results)
        self.register(tool_schemas.LIST_PLUGINS, self._handle_list_plugins)
        self.register(tool_schemas.ABOUT_ALIMA, self._handle_about_alima)

        # Library tools — search tools generated from provider specs (P3)
        for td, handler in self._generated_search_tools():
            self.register(td, handler)
        # Input-source tools generated per instance (e.g. the 3 DOI resolvers,
        # individually callable so an agent can compare sources). - Claude Generated
        for td, handler in self._generated_input_tools():
            self.register(td, handler)
        # Lookup tools (RVK-API …) — the third plugin category. - Claude Generated
        for td, handler in self._generated_lookup_tools():
            self.register(td, handler)
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
        self.register(tool_schemas.EXECUTE_WORKFLOW, self._handle_execute_workflow)

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
                (tool_schemas.RVK_LOOKUP, self._handle_rvk_lookup),
                (tool_schemas.AGGREGATE_GND_RESULTS, self._handle_aggregate_gnd_results),
            ],
            "library": [
                *self._generated_search_tools(),
                (tool_schemas.AGGREGATE_GND_RESULTS, self._handle_aggregate_gnd_results),
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
