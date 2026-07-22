"""
Pipeline Default Configuration Values
Claude Generated - Central definition of pipeline configuration defaults
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

# Output / Autosave fallback (used when config_manager is unavailable)
DEFAULT_AUTOSAVE_DIR = Path.home() / "Documents" / "ALIMA_Results"
"""Fallback autosave directory. Canonical value lives in SystemConfig.autosave_dir."""


def get_autosave_dir(config_manager=None) -> Path:
    """Return autosave dir from config if available, else DEFAULT_AUTOSAVE_DIR - Claude Generated"""
    if config_manager is not None:
        try:
            cfg = config_manager.load_config()
            return Path(cfg.system_config.autosave_dir)
        except Exception:
            pass  # config unreadable — fall back to DEFAULT_AUTOSAVE_DIR - Claude Generated
    return DEFAULT_AUTOSAVE_DIR


#: Trailing ``_YYYYMMDD_HHMMSS`` — some working titles already carry one (the
#: classic path's ``build_working_title`` appends it), so we don't add a second.
_TIMESTAMP_SUFFIX = re.compile(r"_\d{8}_\d{6}$")


def autosave_filename(working_title: str, when: Optional[datetime] = None) -> str:
    """A collision-free ``.json`` name for one autosaved run.

    The GUI autosave used ``{working_title}.json``. That is fine for the classic
    path, whose ``working_title`` already ends in a timestamp, but the agentic
    path's title is just the LLM working title — so two agentic runs of the same
    document wrote the same file and the second silently overwrote the first (a
    whole run lost). This appends a timestamp when the title lacks one, so every
    run lands in its own file regardless of which path produced it. Idempotent
    on already-timestamped titles (no double stamp). - Claude Generated
    """
    stem = str(working_title or "analysis").strip() or "analysis"
    if not _TIMESTAMP_SUFFIX.search(stem):
        stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
        stem = f"{stem}_{stamp}"
    return f"{stem}.json"

# DK Pipeline Search Configuration
DEFAULT_DK_MAX_RESULTS = 40
"""Maximum number of search results to retrieve per keyword from catalog (default: 20)"""

DEFAULT_CLASSIFICATION_FREQUENCY_THRESHOLD = 1
"""Minimum occurrence count for a catalog classification (DK/DDC) to be included
in LLM analysis. RVK is exempt (validated separately). Only classifications that
appear >= this many times in the catalog are passed to the LLM.
- threshold=1 (default): Include all codes found (maximum coverage, may include noise)
- threshold=2-3: Recommended for general use (balanced precision/coverage)
- threshold=5+: Strict filtering for high-confidence classifications only
Higher thresholds improve classification precision but may reduce coverage."""

# Back-compat alias (general-notation generalization, WS3). - Claude Generated
DEFAULT_DK_FREQUENCY_THRESHOLD = DEFAULT_CLASSIFICATION_FREQUENCY_THRESHOLD
