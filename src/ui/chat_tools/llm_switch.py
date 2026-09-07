"""Let the chat agent pick its own model — behind a config switch.

Claude Generated.

Some turns want a different model than the one the run is using: a hard
classification benefits from a stronger one, a quick lookup does not need it.
This tool lets the agent make that call itself.

Three deliberate limits, because an agent that changes its own model is exactly
the kind of thing that gets away from you:

* **It is off unless the operator turned it on.** ``ChatConfig.allow_model_switch``
  gates registration, so with the switch off the model never even sees the tool.
* **It only names models that exist.** The provider must be enabled and the model
  must be in that provider's list; anything else is refused with the list of what
  is available, rather than silently breaking the next turn.
* **It lasts for the session, not beyond.** The pick lives on the ``ChatSession``,
  is never written to the config, and the operator's next toolbar change clears
  it. The running turn keeps its model — the loop is already bound to one — so
  the change takes effect from the following turn, and the tool says so.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

from src.ui.chat_tools.base import BaseChatTool
from src.utils.error_visibility import log_caught

logger = logging.getLogger(__name__)


def _enabled_providers() -> List[str]:
    from src.utils.config_manager import ConfigManager

    return [p.name for p in ConfigManager().get_unified_config().get_enabled_providers()]


def _models_for(llm_service: Any, provider: str) -> List[str]:
    """Known models of one provider; empty when it cannot be asked."""
    if llm_service is None:
        return []
    try:
        return list(llm_service.get_available_models(provider) or [])
    except Exception as exc:
        log_caught(logger, exc, f"switch_llm_model: model list for '{provider}'")
        return []


class SwitchLlmModelTool(BaseChatTool):
    name = "switch_llm_model"
    description = (
        "Switch the provider/model you yourself run on for the rest of this "
        "conversation — e.g. to a stronger model for a difficult classification, "
        "or a faster one for simple lookups. The change takes effect from the "
        "NEXT turn; the current answer is finished on the current model. Name "
        "only a provider that is enabled and a model that provider actually has; "
        "call it with no arguments to see what is available. Say in your reply "
        "what you switched to and why — the operator sees the model, not your "
        "reasoning for it."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "provider": {
                "type": "string",
                "description": "Provider name. Omit together with `model` to list what is available.",
            },
            "model": {"type": "string", "description": "Model name as the provider reports it."},
            "reason": {
                "type": "string",
                "description": "Why this model — shown to the operator in the chat.",
            },
        },
    }

    def __init__(self, *, llm_service: Any = None, on_switch: Any = None) -> None:
        self.llm_service = llm_service
        #: Called with ``(provider, model, reason)`` after a successful switch so
        #: the host can tell the operator. Optional.
        self.on_switch = on_switch

    # -- helpers -------------------------------------------------------

    def _catalogue(self) -> Dict[str, List[str]]:
        return {name: _models_for(self.llm_service, name) for name in _enabled_providers()}

    @staticmethod
    def _match(candidates: List[str], wanted: str) -> str:
        """Case-insensitive exact match; returns the canonical spelling."""
        wanted_l = (wanted or "").strip().casefold()
        for candidate in candidates:
            if candidate.casefold() == wanted_l:
                return candidate
        return ""

    def _refuse(self, message: str, catalogue: Dict[str, List[str]]) -> str:
        return json.dumps(
            {"status": "error", "message": message, "available": catalogue},
            ensure_ascii=False,
        )

    # -- execution -----------------------------------------------------

    def execute(self, session: Any, **kwargs: Any) -> str:
        provider = str(kwargs.get("provider") or "").strip()
        model = str(kwargs.get("model") or "").strip()
        reason = str(kwargs.get("reason") or "").strip()

        try:
            catalogue = self._catalogue()
        except Exception as exc:
            log_caught(logger, exc, "switch_llm_model: reading the provider list")
            return json.dumps(
                {"status": "error", "message": "Provider-Liste nicht lesbar."},
                ensure_ascii=False,
            )

        if not provider and not model:
            return json.dumps(
                {"status": "ok", "available": catalogue}, ensure_ascii=False
            )
        if not (provider and model):
            return self._refuse("Provider und Modell müssen beide genannt werden.", catalogue)

        canonical_provider = self._match(list(catalogue), provider)
        if not canonical_provider:
            return self._refuse(f"Provider '{provider}' ist nicht aktiv.", catalogue)

        models = catalogue.get(canonical_provider) or []
        canonical_model = self._match(models, model)
        if not canonical_model:
            # An empty list means the provider could not be asked, not that it
            # has no models — refusing on a guess would be worse than saying so.
            detail = (
                f"Modelle von '{canonical_provider}' konnten nicht abgefragt werden."
                if not models
                else f"Modell '{model}' gibt es bei '{canonical_provider}' nicht."
            )
            return self._refuse(detail, catalogue)

        if session is None:
            return self._refuse("Keine Chat-Sitzung — Wechsel nicht möglich.", catalogue)
        session.requested_provider = canonical_provider
        session.requested_model = canonical_model

        if self.on_switch is not None:
            try:
                self.on_switch(canonical_provider, canonical_model, reason)
            except Exception as exc:
                log_caught(logger, exc, "switch_llm_model: notifying the host")

        return json.dumps(
            {
                "status": "switched",
                "provider": canonical_provider,
                "model": canonical_model,
                "effective": "next_turn",
                "message": (
                    "Gilt ab der nächsten Antwort; diese hier läuft noch auf dem "
                    "bisherigen Modell. Sag dem Nutzer, worauf du gewechselt hast."
                ),
            },
            ensure_ascii=False,
        )


def llm_switch_tools(*, llm_service: Any = None, on_switch: Any = None) -> Tuple[BaseChatTool, ...]:
    """The switch toolset. Register only when the operator enabled it."""
    return (SwitchLlmModelTool(llm_service=llm_service, on_switch=on_switch),)
