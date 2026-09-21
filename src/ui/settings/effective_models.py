"""Which model a task really runs on, and which settings never get consulted.

Six settings can decide the model. This works out, for each task, the one that
wins and says why — and, just as important, names the entries that look like they
would be used and are not.

The priority list of a task reads like a fallback chain, and the interface used
to present it as one. It is not: ``PipelineConfig.create_from_provider_preferences``
(``pipeline_manager.py``) copies only ``model_priority[0]`` into the step, and at
run time ``PipelineStepExecutor._resolve_provider_smart`` returns an explicit
step provider/model without testing whether it can be reached. So entries 2..n
are never consulted in the classic path, and a first entry that is down makes the
step fail rather than moving on.

Stating that is the point of ``unused``: the table shows it instead of promising
a chain that does not run. - Claude Generated
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Tuple

#: Display label per source key.
SOURCE_LABELS = {
    "task_preference": "Aufgaben-Voreinstellung",
    "pipeline_default_no_step": "Pipeline-Standard",
    "pipeline_default": "Pipeline-Standard",
    "agentic_default": "Agentic Default",
    "preferred": "Allgemeiner Standard",
    "first_enabled": "erster nutzbarer Anbieter",
    "none": "kein Anbieter konfiguriert",
}

#: Why the remaining priority entries do not run.
UNUSED_REASON = (
    "Wird derzeit nicht benutzt: in den Schritt übernommen wird nur der erste "
    "Eintrag, und ein gesetzter Schritt wird ohne Erreichbarkeitsprüfung "
    "verwendet."
)

#: Why a whole priority list does not run.
NO_STEP_REASON = (
    "Für diese Aufgabe gibt es keinen Schritt, in den eine Voreinstellung "
    "übernommen würde; es gilt der Pipeline-Standard."
)

#: The tasks whose preference actually reaches a run. ``create_from_provider_preferences``
#: copies ``model_priority[0]`` onto a step only where the task name *is* a step
#: id, and it builds exactly initialisation / keywords / dk_search /
#: dk_classification. The other task preferences (classification, vision,
#: chunked) have no step to land on, and ``SmartProviderSelector`` — the runtime
#: fallback — does not read task preferences at all: its tiers are explicit
#: override, pipeline default, first available provider.
#: Pinned against the real builder by tests. - Claude Generated
TASK_PREFERENCE_STEPS = frozenset({"initialisation", "keywords", "dk_classification"})


@dataclass
class EffectiveRow:
    """One row of the "what runs where" table."""

    task: str
    provider: str
    model: str
    source: str
    #: (provider, model) pairs that are configured but never consulted.
    unused: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def source_label(self) -> str:
        return SOURCE_LABELS.get(self.source, self.source)

    @property
    def configured(self) -> bool:
        return bool(self.provider)


def _first_entry(task_pref: Any) -> Tuple[str, str] | None:
    for entry in getattr(task_pref, "model_priority", None) or []:
        provider = str(entry.get("provider_name") or "").strip()
        if provider and provider != "auto":
            return provider, str(entry.get("model_name") or "").strip()
    return None


def _rest(task_pref: Any) -> List[Tuple[str, str]]:
    entries = list(getattr(task_pref, "model_priority", None) or [])
    out: List[Tuple[str, str]] = []
    seen_first = False
    for entry in entries:
        provider = str(entry.get("provider_name") or "").strip()
        if not provider or provider == "auto":
            continue
        if not seen_first:
            seen_first = True
            continue
        out.append((provider, str(entry.get("model_name") or "").strip()))
    return out


def effective_rows(unified_config: Any) -> List[EffectiveRow]:
    """One row per LLM task, plus the agentic/chat row. - Claude Generated"""
    from src.utils.config_models import LLM_TASK_DISPLAY_INFO, ProviderScope

    rows: List[EffectiveRow] = []

    pipe_provider, pipe_model, pipe_source = unified_config.explain_default_provider_model(
        scope=ProviderScope.PIPELINE
    )

    for task_type, label, _desc in LLM_TASK_DISPLAY_INFO:
        pref = (unified_config.task_preferences or {}).get(task_type.value)
        first = _first_entry(pref) if pref is not None else None
        reaches_a_step = task_type.value in TASK_PREFERENCE_STEPS

        if first is not None and reaches_a_step:
            rows.append(EffectiveRow(
                task=label, provider=first[0], model=first[1],
                source="task_preference", unused=_rest(pref),
            ))
        elif first is not None:
            # Configured, but nothing consults it — name the whole list as unused
            # rather than showing a model that will not run.
            entries = [first] + _rest(pref)
            rows.append(EffectiveRow(
                task=label, provider=pipe_provider, model=pipe_model,
                source="pipeline_default_no_step", unused=entries,
            ))
        else:
            rows.append(EffectiveRow(
                task=label, provider=pipe_provider, model=pipe_model,
                source=pipe_source if pipe_source != "preferred" else "pipeline_default",
            ))

    ag_provider, ag_model, ag_source = unified_config.explain_default_provider_model(
        scope=ProviderScope.AGENTIC
    )
    rows.append(EffectiveRow(
        task="🤖 Chat und agentische Läufe",
        provider=ag_provider, model=ag_model, source=ag_source,
    ))
    return rows
