"""Bridge: discover directory plugins and merge them into the config - Claude Generated.

Ties the generic core loader (:mod:`src.core.plugins.loader`) to the concrete
categories. It ensures the category adapters are registered (by importing the
search + input-source packages), runs the directory scan, and merges the
discovered declarative instances into ``config.plugins`` (dedup by
category + instance_id). Code plugins are gated by the two-tier security in the
loader; the ``approve_cb`` (a GUI dialog / CLI prompt) is injected by the caller.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _ensure_categories() -> None:
    """Import the concrete categories so their adapters self-register."""
    import src.core.search  # noqa: F401  (registers SearchProviderCategory)
    import src.utils.input_sources  # noqa: F401  (registers InputSourceCategory)


def discover_plugins(
    config,
    *,
    plugins_dir: Path,
    approve_cb: Optional[Callable] = None,
    enable_code_plugins: Optional[bool] = None,
    persist: Optional[Callable[[], Any]] = None,
):
    """Discover plugins under ``plugins_dir`` and merge instances into ``config``.

    Returns the loader :class:`LoadResult` (or ``None`` if the dir is absent).
    ``persist`` is called after a fresh code-plugin approval so the caller can save
    the updated ``config.approved_plugins`` ledger.
    """
    plugins_dir = Path(plugins_dir)
    if not plugins_dir.is_dir():
        return None
    _ensure_categories()
    from src.core.plugins.loader import discover

    if enable_code_plugins is None:
        enable_code_plugins = bool(getattr(config.system_config, "enable_code_plugins", False))

    before = dict(config.approved_plugins)
    result = discover(
        plugins_dir,
        approved_plugins=config.approved_plugins,
        approve_cb=approve_cb,
        enable_code_plugins=enable_code_plugins,
    )

    existing = {(p.category, p.instance_id) for p in config.plugins}
    for inst in result.instances:
        key = (inst.category, inst.instance_id)
        if key not in existing:
            config.plugins.append(inst)
            existing.add(key)

    if persist is not None and config.approved_plugins != before:
        try:
            persist()
        except Exception as e:  # pragma: no cover - persistence is best effort
            logger.warning("Could not persist plugin approvals: %s", e)

    return result
