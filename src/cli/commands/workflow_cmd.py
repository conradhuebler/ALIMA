"""Workflow v4 CLI Command - Claude Generated.

Handles ``alima workflow <name>`` and ``alima workflows list`` for the v4
workflow system (generic agents + deterministic steps).

Separate from ``pipeline_cmd`` so the v4 dispatch does not inherit the
AlimaManager/PipelineManager stack.  v4 workflows run against a
SharedContext via :class:`WorkflowExecutor`.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.agents import deterministic_functions as _register_fns  # noqa: F401
from src.core.agents import steps as _register_steps  # noqa: F401
from src.core.agents.shared_context import SharedContext
from src.core.agents.sub_agents import create_caching_registry
from src.core.agents.workflow_executor import WorkflowExecutor
from src.core.agents.workflow_loader import (
    DEFAULT_SEARCH_PATHS,
    discover_workflow_files,
    find_workflow_file,
    load_workflow,
)

logger = logging.getLogger(__name__)


def _discover_workflows() -> List[Path]:
    """Return every ``*.yaml`` under the configured search paths, deduped.

    Thin wrapper over the shared ``workflow_loader.discover_workflow_files``. - Claude Generated
    """
    return discover_workflow_files()


def _load_input_blob(args) -> Dict[str, Any]:
    """Build the input dict from ``--input`` JSON string or ``--input-file``."""
    if getattr(args, "input_file", None):
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif getattr(args, "input", None):
        data = json.loads(args.input)
    else:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Workflow input must be a JSON object")
    return data


def _build_context(input_blob: Dict[str, Any], args) -> SharedContext:
    """Populate a SharedContext.

    Typed ALIMA fields are lifted when the blob contains ``abstract`` or
    ``initial_keywords`` — so ``alima_classic`` can be driven via
    ``--input '{"abstract": "..."}'`` without a YAML glue layer. Everything
    else goes straight into ``extra``.
    """
    ctx = SharedContext(
        abstract=str(input_blob.get("abstract", "") or ""),
        initial_keywords=list(input_blob.get("initial_keywords", []) or []),
        provider=getattr(args, "provider", "") or "",
        model=getattr(args, "model", "") or "",
    )
    if getattr(args, "temperature", None) is not None:
        ctx.temperature = args.temperature
    ctx.extra = {k: v for k, v in input_blob.items() if k not in ("abstract", "initial_keywords")}
    ctx.extra.setdefault("input", dict(input_blob))
    return ctx


def handle_workflows_list(args, log: logging.Logger) -> int:
    """``alima workflows list`` — enumerate discovered workflow YAMLs."""
    files = _discover_workflows()
    if not files:
        print("No workflows found. Searched:")
        for p in DEFAULT_SEARCH_PATHS:
            print(f"  - {p}")
        return 1

    print(f"{'NAME':<30} {'VERSION':<8} DESCRIPTION")
    print("-" * 78)
    for path in files:
        try:
            wf = load_workflow(path, strict=False)
            print(f"{path.stem:<30} {wf.version:<8} {wf.description[:60]}")
        except Exception as e:
            print(f"{path.stem:<30} {'?':<8} <parse error: {e}>")
    return 0


def handle_workflow(args, config_manager, llm_service, log: logging.Logger) -> int:
    """``alima workflow <name>`` — run a single v4 workflow.

    Args:
        args: Parsed CLI namespace (expects ``name``, ``input``/``input_file``,
            optional ``output``, ``provider``, ``model``, ``only_step``).
        config_manager: For CachingToolRegistry init.
        llm_service: Shared LlmService for LLMAgentStep.
        log: Logger.

    Returns:
        Process exit code (0 on success).
    """
    name = args.name
    wf_path = Path(name) if name.endswith((".yaml", ".yml")) and Path(name).exists() \
        else find_workflow_file(name)
    if wf_path is None:
        print(f"❌ Workflow '{name}' not found. Searched:", file=sys.stderr)
        for p in DEFAULT_SEARCH_PATHS:
            print(f"   - {p}", file=sys.stderr)
        return 2

    try:
        wf = load_workflow(wf_path, strict=True)
    except Exception as e:
        print(f"❌ Failed to load {wf_path}: {e}", file=sys.stderr)
        return 2

    try:
        input_blob = _load_input_blob(args)
    except (json.JSONDecodeError, ValueError, OSError) as e:
        print(f"❌ Invalid input: {e}", file=sys.stderr)
        return 2

    ctx = _build_context(input_blob, args)

    tool_registry = create_caching_registry(config_manager=config_manager)

    def _stream(msg: str) -> None:
        sys.stdout.write(msg)
        sys.stdout.flush()

    executor = WorkflowExecutor(
        llm_service=llm_service,
        tool_registry=tool_registry,
        stream_callback=_stream if not getattr(args, "quiet", False) else None,
    )

    only_step: Optional[str] = getattr(args, "only_step", None) or None
    report = executor.run(wf, ctx, only_step=only_step, stop_on_error=True)

    # Generic convention: any workflow that populates extra.report_markdown
    # (e.g. title_list_search's render_report step) gets a human-readable
    # summary printed here, instead of operators having to read the raw JSON
    # dump below. Not workflow-specific — keyed off the field's presence. - Claude Generated
    report_markdown = ctx.extra.get("report_markdown") if isinstance(getattr(ctx, "extra", None), dict) else None
    if report_markdown:
        print("\n" + report_markdown + "\n")

    out = {
        "workflow": wf.name,
        "success": report.success,
        "duration_seconds": round(report.duration_seconds, 3),
        "error": report.error,
        "steps": [r.to_dict() for r in report.step_results],
        "context": ctx.to_dict(),
    }

    if getattr(args, "output", None):
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 Wrote report to {args.output}")
    else:
        print()
        json.dump(out, sys.stdout, ensure_ascii=False, indent=2, default=str)
        print()

    return 0 if report.success else 1
