"""Workflow-discovery endpoint for the webapp (F-6 split).

Claude Generated — moved verbatim from ``app.py``. Exposes ``GET /api/workflows``
plus the discovery helpers. Tests patch
``src.webapp.routers.workflows.{DEFAULT_SEARCH_PATHS, yaml.safe_load,
_discover_workflows}`` (the names are resolved in this module's namespace).
"""

import logging

import yaml
from fastapi import APIRouter

from src.core.agents.workflow_loader import (
    DEFAULT_SEARCH_PATHS,
    discover_workflow_files,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Workflow list order mirrors the Qt6 pipeline tab picker.
_WORKFLOW_ORDER = [
    "alima_v51",
    "alima_v51_105",  # UB Freiberg variant: WiWi-only RVK, DK otherwise
    "__classic__",
    "alima",
    "alima_classic_v51",
    "alima_classic",
    "title_list_search",
    "catalog_search",
    "synonym_expansion",
    "batch_metadata",
]


def _extract_workflow_steps(data: dict) -> list:
    """Reduce a workflow YAML's ``steps:`` block to [{id, label}, …] for the
    frontend pipeline-stepper. Skips non-dict / id-less entries. - Claude Generated
    """
    steps = []
    for entry in data.get("steps", []) or []:
        if not isinstance(entry, dict):
            continue
        step_id = entry.get("id")
        if not step_id:
            continue
        steps.append({"id": str(step_id), "label": str(entry.get("name") or step_id)})
    return steps


# Canonical classic-pipeline steps — authoritative source is
# PipelineManager.step_definitions (src/core/pipeline_manager.py); order matches
# _create_pipeline_steps. German labels for webapp consistency. - Claude Generated
_CLASSIC_STEPS = [
    {"id": "input", "label": "Eingabe"},
    {"id": "initialisation", "label": "Schlagwörter"},
    {"id": "search", "label": "GND-Suche"},
    {"id": "keywords", "label": "Prüfung"},
    {"id": "dk_search", "label": "DK-Suche"},
    {"id": "dk_classification", "label": "Klassifikation"},
]


def _discover_workflows() -> tuple[dict, dict, dict]:
    """Scan DEFAULT_SEARCH_PATHS for v4 YAML workflows.

    Returns (root_stem → version, legacy_stem → version, stem → steps[]). - Claude Generated
    """
    root: dict = {}
    legacy: dict = {}
    steps_by_stem: dict = {}

    # Top-level workflows via the shared discovery (CLI/GUI use the same). - Claude Generated
    files = discover_workflow_files()
    for path in files:
        try:
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            version = str(data.get("version", "?"))
            root[path.stem] = version
            steps_by_stem[path.stem] = _extract_workflow_steps(data)
        except Exception as e:
            logger.warning(f"Could not read workflow {path}: {e}")

    # Legacy subdirs are not part of the shared top-level discovery — scan
    # them here for backward display (deduped against the top-level set).
    seen: set = {p.resolve() for p in files}
    for base in DEFAULT_SEARCH_PATHS:
        legacy_dir = base / "legacy"
        if not legacy_dir.is_dir():
            continue
        for path in sorted(legacy_dir.glob("*.yaml"), key=lambda p: str(p.name)):
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            try:
                with open(path, encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
                version = str(data.get("version", "?"))
                legacy[path.stem] = version
                steps_by_stem[path.stem] = _extract_workflow_steps(data)
            except Exception as e:
                logger.warning(f"Could not read legacy workflow {path}: {e}")

    return root, legacy, steps_by_stem


def _get_configured_default_workflow() -> str:
    """Return SystemConfig.default_workflow, falling back to alima_v51. - Claude Generated"""
    try:
        from src.utils.config_manager import ConfigManager

        cfg = ConfigManager().load_config()
        return getattr(cfg.system_config, "default_workflow", "alima_v51") or "alima_v51"
    except Exception:
        return "alima_v51"


@router.get("/api/workflows")
async def get_available_workflows() -> list:
    """Get available pipeline/agentic workflows for the workflow dropdown. - Claude Generated"""
    try:
        root, legacy, steps_by_stem = _discover_workflows()
        configured_default = _get_configured_default_workflow()

        def _label(stem: str) -> str:
            ver = root.get(stem)
            return f"{stem} (v{ver})" if ver else stem

        items = []
        added: set = set()

        if "alima_v51" in root:
            items.append({
                "label": f"⭐ ALIMA v5.1 — agentisch (v{root['alima_v51']})",
                "value": "alima_v51",
                "agentic": True,
                "steps": steps_by_stem.get("alima_v51", []),
                "default": configured_default == "alima_v51",
            })
            added.add("alima_v51")

        items.append({
            "label": "Klassische Pipeline (nicht agentisch)",
            "value": "__classic__",
            "agentic": False,
            "steps": _CLASSIC_STEPS,
            "default": configured_default == "__classic__",
        })
        added.add("__classic__")

        for stem in _WORKFLOW_ORDER:
            if stem in added or stem not in root:
                continue
            items.append({
                "label": _label(stem),
                "value": stem,
                "agentic": True,
                "steps": steps_by_stem.get(stem, []),
                "default": configured_default == stem,
            })
            added.add(stem)

        for stem in sorted(root):
            if stem in added:
                continue
            items.append({
                "label": _label(stem),
                "value": stem,
                "agentic": True,
                "steps": steps_by_stem.get(stem, []),
                "default": configured_default == stem,
            })
            added.add(stem)

        if legacy:
            items.append({"label": "───────────────", "value": "__separator__",
                          "agentic": False, "steps": []})
            for stem in sorted(legacy):
                items.append({
                    "label": f"{stem} (legacy v{legacy[stem]})",
                    "value": stem,
                    "agentic": True,
                    "steps": steps_by_stem.get(stem, []),
                })

        return items
    except Exception as e:
        logger.error(f"Error getting workflows: {e}")
        return []
