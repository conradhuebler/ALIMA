"""Raw-response caching for lookup calls made OUTSIDE the MCP tool handler - Claude Generated.

Direct pipeline/CLI/GUI call sites (e.g. the RVK anchor in ``pipeline_utils``) reuse
the WP2 raw cache the same way the tool handler does
(``src/mcp/tool_registry.py:_make_lookup_handler``): gate on the per-plugin
``cache_responses`` setting + the global ``enable_response_cache``, and key raw JSON
by ``(source, key, params)`` in ``search_response_cache``. This closes the F3 gap
(``rvk_lookup`` bypassed the raw cache).

The cached *value* is whatever the caller's fetch returns (full/raw), so direct
callers keep their own data shape. Using the same ``source`` names as the tools
(``rvk_search`` / ``rvk_validate`` / …) lets pipeline and agent share cache entries.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


def lookup_cache_enabled(config: Any, lookup_id: str) -> bool:
    """Whether raw caching is on for a lookup plugin id, read from an AlimaConfig.

    Mirrors the tool-handler gate: the per-plugin ``cache_responses`` tri-state
    (auto/on/off) resolved against the global ``enable_response_cache``.
    """
    if config is None:
        return False
    try:
        from src.core.plugins.schema import cache_pref_enabled

        global_enabled = bool(
            getattr(getattr(config, "system_config", None), "enable_response_cache", False)
        )
        settings: Dict[str, Any] = {}
        for p in getattr(config, "plugins", []) or []:
            if getattr(p, "category", None) == "lookup" and getattr(p, "provider_id", None) == lookup_id:
                settings = p.settings or {}
                break
        return cache_pref_enabled(settings.get("cache_responses"), global_enabled=global_enabled)
    except Exception as e:
        logger.debug(f"lookup_cache_enabled({lookup_id}) failed: {e}")
        return False


def cached_call(
    km: Any,
    enabled: bool,
    source: str,
    key: Optional[str],
    params: Optional[Dict[str, Any]],
    fetch_fn: Callable[[], Any],
    *,
    loads: Optional[Callable[[str], Any]] = None,
    dumps: Optional[Callable[[Any], str]] = None,
) -> Any:
    """Return ``fetch_fn()``, served from / written to ``search_response_cache`` when
    ``enabled`` and ``key`` are truthy - Claude Generated.

    Best-effort: any cache read/write error falls back to a live fetch, so caching
    can never break the caller. ``source`` should match the plugin tool name so the
    pipeline and the agent share cache entries.
    """
    _dumps = dumps or (lambda v: json.dumps(v, ensure_ascii=False, default=str))
    _loads = loads or json.loads
    use_cache = bool(km is not None and enabled and key)
    if use_cache:
        try:
            hit = km.get_raw_response(source, str(key), params or {})
            if hit:
                return _loads(hit["raw_json"])
        except Exception as e:
            logger.debug(f"lookup cache read failed ({source}/{key}): {e}")
    result = fetch_fn()
    if use_cache:
        try:
            km.store_raw_response(source, str(key), params or {}, _dumps(result))
        except Exception as e:
            logger.debug(f"lookup cache write skipped ({source}/{key}): {e}")
    return result
