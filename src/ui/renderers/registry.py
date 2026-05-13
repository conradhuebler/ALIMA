"""Renderer registry - Claude Generated (WP10 P-α).

Mirror to ``STEP_REGISTRY`` / ``TOOL_FN_REGISTRY`` in
``src/core/agents/registry.py``. Same decorator semantics, same
collision-check, same ``get_*`` / ``list_*`` look-up convention.
"""
from __future__ import annotations

from typing import Dict, Type

from .base import BaseRenderer

RENDERER_REGISTRY: Dict[str, Type[BaseRenderer]] = {}

FALLBACK_SLOT = "slot:raw_json"


def register_renderer(slot: str):
    """Decorator: register a ``BaseRenderer`` subclass under ``slot``.

    Re-registering the same name with a different class raises
    ``ValueError`` (analog ``@register_step`` in
    ``src/core/agents/registry.py``). Re-registering the same class is
    idempotent.
    """

    def _wrap(cls):
        existing = RENDERER_REGISTRY.get(slot)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Renderer slot '{slot}' already registered to "
                f"{existing.__name__}"
            )
        RENDERER_REGISTRY[slot] = cls
        return cls

    return _wrap


def get_renderer(
    slot: str, fallback: str = FALLBACK_SLOT
) -> Type[BaseRenderer]:
    """Look up a renderer class by slot.

    Falls back to ``fallback``-slot if ``slot`` is not registered.
    Raises ``KeyError`` if even the fallback slot is missing.
    """
    if slot in RENDERER_REGISTRY:
        return RENDERER_REGISTRY[slot]
    if fallback in RENDERER_REGISTRY:
        return RENDERER_REGISTRY[fallback]
    raise KeyError(
        f"No renderer for slot '{slot}' and fallback '{fallback}' missing. "
        f"Registered: {sorted(RENDERER_REGISTRY)}"
    )


def list_renderers() -> list[str]:
    """Return sorted list of registered slot names."""
    return sorted(RENDERER_REGISTRY)


def _reset_for_tests() -> None:
    """Clear registry. Used only by test fixtures."""
    RENDERER_REGISTRY.clear()
