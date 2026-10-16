"""Which model a task really runs on, and where that comes from.

Six settings could once decide it, most of them invisibly. What is left after the
Sep-2026 rework is four, and each is a row here:

* the **general default** — the fallback for everything;
* the **pipeline default** — the classic pipeline, if it deviates;
* the **agentic default** — agentic workflows and the chat;
* the **vision default** — image and OCR analysis, a capability question rather
  than a pipeline step.

Per-task model choices (``task_preferences``) are retired as a live config
surface. They were never part of this resolution hierarchy to begin with — a
separate, dead-end lookup that only read its own first entry and had no
fallback — and the wizards, dialogs and step widgets that wrote or read them
are gone too (Oct 2026). The field and dataclass remain only because
``adopt_vision_task_preference`` still needs something to migrate an old
on-disk vision choice from.

A workflow step does **not** name its own model: ``LLMAgentStep._llm_params``
and ``ReflectionStep.run`` take provider and model from the run context and read
neither from the step's ``llm:`` block (which they do read for temperature,
budget and thinking). These four rows are therefore the whole answer; the claim
is held by ``tests/test_settings_dialog_state.py``. - Claude Generated
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

#: Display label per source key.
SOURCE_LABELS = {
    "vision_default": "Vision-Standard",
    "agentic_default": "Agentic Default",
    "pipeline_default": "Pipeline-Standard",
    "preferred": "Allgemeiner Standard",
    "first_enabled": "erster nutzbarer Anbieter",
    "none": "kein Anbieter konfiguriert",
}


@dataclass
class EffectiveRow:
    """One row of the "what runs where" table."""

    scope: str
    task: str
    provider: str
    model: str
    source: str

    @property
    def source_label(self) -> str:
        return SOURCE_LABELS.get(self.source, self.source)

    @property
    def configured(self) -> bool:
        return bool(self.provider)

    @property
    def is_fallback(self) -> bool:
        """True when nothing was chosen for this scope and it inherited."""
        return self.source in ("preferred", "first_enabled") and self.scope != "general"


def effective_rows(unified_config: Any) -> List[EffectiveRow]:
    """One row per scope, in the order someone reads them. - Claude Generated"""
    from src.utils.config_models import ProviderScope

    plan = [
        (ProviderScope.GENERAL, "general", "Allgemein"),
        (ProviderScope.PIPELINE, "pipeline", "Klassische Pipeline"),
        (ProviderScope.AGENTIC, "agentic", "Agentisch und Chat"),
        (ProviderScope.VISION, "vision", "Bild- und OCR-Analyse"),
    ]
    rows: List[EffectiveRow] = []
    for scope, key, label in plan:
        provider, model, source = unified_config.explain_default_provider_model(
            scope=scope
        )
        rows.append(EffectiveRow(
            scope=key, task=label, provider=provider, model=model, source=source,
        ))
    return rows
