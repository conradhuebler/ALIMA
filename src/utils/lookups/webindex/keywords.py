"""Keyword-extraction glue for the webindex crawl - Claude Generated.

Turns the per-page keyword extraction (formerly a direct ``generate_response``
call with an inline prompt) into a **workflow**: ``workflows/webindex_keywords.yaml``
holds the editable "standprompt". The indexer stays workflow-agnostic — it just
calls a ``keyword_extractor`` callable ``(text, max_keywords) -> list[str]``; this
module builds that callable by running the tool-less mini-workflow per page.

Model resolution (operator decision: global default, overridable):
  1. instance ``llm_provider`` / ``llm_model`` (set on the plugin instance),
  2. CLI ``--provider`` / ``--model`` flags,
  3. the global ALIMA agentic default (``UnifiedProviderConfig.resolve_default_provider_model``).
  If none yields a provider, the extractor is ``None`` → the crawl falls back to
  deterministic meta/heading keywords only.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)
_WORKFLOW_NAME = "webindex_keywords"


def resolve_crawl_model(
    config: Any,
    instance_settings: Optional[Dict[str, Any]],
    cli_provider: Optional[str] = None,
    cli_model: Optional[str] = None,
) -> Tuple[str, str]:
    """Resolve the (provider, model) the crawl LLM should use - Claude Generated.

    Order: CLI flags → instance settings → global agentic default. Empty strings
    mean "unset"; returning ``("", "")`` signals "no LLM configured → meta-only".
    """
    cli_provider = (cli_provider or "").strip()
    cli_model = (cli_model or "").strip()
    if cli_provider:
        return cli_provider, cli_model
    settings = instance_settings or {}
    inst_provider = str(settings.get("llm_provider") or "").strip()
    inst_model = str(settings.get("llm_model") or "").strip()
    if inst_provider:
        return inst_provider, inst_model
    # Global default.
    try:
        unified = getattr(config, "unified_config", None)
        if unified is not None and hasattr(unified, "resolve_default_provider_model"):
            provider, model = unified.resolve_default_provider_model(
                scope="agentic", fallback_to_first_enabled=True
            )
            if provider:
                return provider, model or ""
    except Exception as e:  # noqa: BLE001 — model resolution must never crash the crawl
        _LOGGER.warning(f"global default model resolution failed: {e}")
    return "", ""


def build_keyword_extractor(
    llm_service: Any,
    provider: str,
    model: str,
    *,
    logger: Optional[logging.Logger] = None,
) -> Optional[Callable[[str, int], List[str]]]:
    """Build a per-page keyword extractor that runs the ``webindex_keywords``
    workflow. Returns ``None`` when the workflow or the provider/model is missing
    (the crawl then falls back to meta/heading keywords). - Claude Generated
    """
    log = logger or _LOGGER
    if llm_service is None or not provider:
        return None
    from src.core.agents.shared_context import SharedContext
    from src.core.agents.workflow_executor import WorkflowExecutor
    from src.core.agents.workflow_loader import find_workflow_file, load_workflow

    path = find_workflow_file(_WORKFLOW_NAME)
    if path is None:
        log.warning(
            f"workflow '{_WORKFLOW_NAME}.yaml' not found (./workflows, "
            f"~/.config/alima/workflows/); LLM keyword extraction disabled."
        )
        return None
    wf = load_workflow(path)
    executor = WorkflowExecutor(
        llm_service=llm_service, tool_registry=None, stream_callback=None
    )

    def extract(text: str, max_keywords: int) -> List[str]:
        ctx = SharedContext(provider=provider, model=model, temperature=0.2)
        ctx.extra = {"page_text": (text or "")[:8000], "max_keywords": int(max_keywords)}
        try:
            report = executor.run(wf, ctx, stop_on_error=True)
        except Exception as e:  # noqa: BLE001 — one page must not kill the crawl
            log.warning(f"keyword workflow failed for a page: {e}")
            return []
        # Prefer the JSON-parsed ``response.keywords``; fall back to parsing the raw
        # text if the model ignored the JSON rule.
        kws = ctx.extra.get("keywords")
        if isinstance(kws, list):
            return [str(k).strip() for k in kws if str(k).strip()]
        try:
            raw = report.step_results[0].data.get("response_text", "") if report.step_results else ""
        except Exception:  # noqa: BLE001
            raw = ""
        if raw:
            from .indexer import _parse_keyword_list

            return _parse_keyword_list(raw, int(max_keywords))
        return []

    return extract