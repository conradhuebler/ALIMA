"""Single-instance resolver for lookup plugins - Claude Generated.

Gives non-agent call sites (the classic pipeline, the CLI/GUI batch harvesters, the
GUI DNB-sync) the *same* configured plugin object the MCP tool handler builds
(``ToolRegistry._make_lookup_handler`` → ``get_category("lookup").build(inst)``),
instead of instantiating the underlying client/resolver directly. So there is **one
construction path per source**: pipeline + agent share it and both honor the
operator's per-instance settings (``timeout``, ``cache_dir``, …) and env overrides.

Instance selection mirrors ``ToolRegistry._lookup_instances`` for a single id: the
operator's enabled config instance if present, else a synthetic primary instance
built from the live registry — so a pipeline anchor (e.g. RVK) keeps working even
when the agent-facing instance is disabled.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def resolve_lookup_instance(config: Any, lookup_id: str):
    """Return the ``PluginInstanceConfig`` for ``lookup_id`` - Claude Generated.

    The enabled config instance of that type if present, else a synthetic enabled
    primary instance (empty settings) so lookups without a config section still build.
    """
    import src.utils.lookups  # noqa: F401 — self-registers the category + plugins
    from src.utils.config_models import PluginInstanceConfig

    if config is not None:
        try:
            for inst in config.enabled_instances_for("lookup"):
                if inst.provider_id == lookup_id:
                    return inst
        except Exception as e:  # best-effort; never break the caller
            logger.debug(f"resolve_lookup_instance({lookup_id}) config read failed: {e}")
    return PluginInstanceConfig(
        instance_id=lookup_id, category="lookup", provider_id=lookup_id, is_primary=True
    )


def _load_config_best_effort() -> Optional[Any]:
    """Load the AlimaConfig via the ConfigManager singleton, or None. - Claude Generated"""
    try:
        from src.utils.config_manager import ConfigManager

        return ConfigManager().load_config()
    except Exception as e:
        logger.debug(f"build_lookup: config load failed: {e}")
        return None


def build_lookup(config: Any, lookup_id: str):
    """Build the configured lookup plugin object for ``lookup_id`` - Claude Generated.

    The same object the MCP tool handler builds, so callers honor the per-instance
    settings + env overrides and there is a single construction path per source.
    Pass ``config=None`` to auto-load the AlimaConfig (best-effort) — convenient for
    GUI/worker call sites that do not already hold one.
    """
    import src.utils.lookups  # noqa: F401 — self-registers the category + plugins
    from src.core.plugins.category import get_category

    if config is None:
        config = _load_config_best_effort()
    inst = resolve_lookup_instance(config, lookup_id)
    return get_category("lookup").build(inst)
