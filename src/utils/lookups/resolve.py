"""Single-instance resolver for lookup plugins - Claude Generated.

Gives non-agent call sites (the classic pipeline, the CLI/GUI batch harvesters, the
GUI DNB-sync) the *same* configured plugin object the MCP tool handler builds
(``ToolRegistry._make_lookup_handler`` → ``get_category("lookup").build(inst)``),
instead of instantiating the underlying client/resolver directly. So there is **one
construction path per source**: pipeline + agent share it and both honor the
operator's per-instance settings (``timeout``, ``cache_dir``, …) and env overrides.

Instance selection mirrors ``ToolRegistry._lookup_instances`` for a single id and
follows the same Search-parity disable semantics (WP P5): a *disabled* instance
gates this path too (returns ``None`` → ``build_lookup`` returns ``None``), so
disabling a lookup in the Plugins tab stops both the agent tool and the
pipeline/CLI/GUI callers. A synthetic default is returned only when the config
can't be read at all (anchor safety on a config error), mirroring
``factory.enabled_gnd_provider_ids``' ``None``-vs-``[]`` discipline.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def resolve_lookup_instance(config: Any, lookup_id: str):
    """Return the ``PluginInstanceConfig`` for ``lookup_id``, or ``None`` - Claude Generated.

    * config readable + an **enabled** instance of that type → return it;
    * config readable + the instance **disabled** (or absent) → ``None`` — the
      operator's disable gates this path (Search parity, WP P5);
    * config **unreadable** (raises) → a synthetic enabled primary so a pipeline
      anchor still works despite a config error.

    ``ensure_lookup_instances`` seeds one enabled instance per registered lookup
    type on every load, so a readable config normally *has* the instance — the
    absent case only survives for a genuinely unmigrated config passed directly.
    """
    import src.utils.lookups  # noqa: F401 — self-registers the category + plugins
    from src.utils.config_models import PluginInstanceConfig

    if config is not None:
        try:
            for inst in config.enabled_instances_for("lookup"):
                if inst.provider_id == lookup_id:
                    return inst
            return None  # readable, not enabled → gated (parity with search)
        except Exception as e:  # config read error → anchor safety below
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
    """Build the configured lookup plugin object for ``lookup_id``, or ``None`` - Claude Generated.

    The same object the MCP tool handler builds, so callers honor the per-instance
    settings + env overrides and there is a single construction path per source.
    Pass ``config=None`` to auto-load the AlimaConfig (best-effort) — convenient for
    GUI/worker call sites that do not already hold one.

    Returns ``None`` when the operator has disabled the lookup (Search parity, WP
    P5) — **callers must guard**; a chained ``build_lookup(...).method()`` will
    otherwise ``AttributeError``.
    """
    import src.utils.lookups  # noqa: F401 — self-registers the category + plugins
    from src.core.plugins.category import get_category

    if config is None:
        config = _load_config_best_effort()
    inst = resolve_lookup_instance(config, lookup_id)
    if inst is None:
        return None
    return get_category("lookup").build(inst)
