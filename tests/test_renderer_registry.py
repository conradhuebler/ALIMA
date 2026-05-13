"""Unit tests for the renderer registry - Claude Generated (WP10 P-α).

Covers WP10 P-α smoke-test criteria:
  * Registration collision-check (ValueError).
  * Lookup fallback to ``slot:raw_json``.
  * raw_json rendering is crash-free and HTML-safe.
  * ``list_renderers`` returns sorted slot names.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import pytest

from src.ui.renderers import (
    RENDERER_REGISTRY,
    BaseRenderer,
    get_renderer,
    list_renderers,
    register_renderer,
)
from src.ui.renderers.raw_json import RawJsonRenderer
from src.ui.renderers.registry import _reset_for_tests


@pytest.fixture(autouse=True)
def _restore_registry():
    """Snapshot and restore the registry around each test."""
    snapshot = dict(RENDERER_REGISTRY)
    yield
    _reset_for_tests()
    RENDERER_REGISTRY.update(snapshot)


# -- Registration & lookup -------------------------------------------------


def test_raw_json_registered_after_import():
    """raw_json must be auto-registered by package __init__."""
    assert "slot:raw_json" in RENDERER_REGISTRY
    assert RENDERER_REGISTRY["slot:raw_json"] is RawJsonRenderer


def test_get_renderer_returns_class():
    cls = get_renderer("slot:raw_json")
    assert cls is RawJsonRenderer


def test_get_renderer_falls_back_to_raw_json_for_unknown():
    cls = get_renderer("slot:does_not_exist")
    assert cls is RawJsonRenderer


def test_get_renderer_raises_if_both_slot_and_fallback_missing():
    _reset_for_tests()
    with pytest.raises(KeyError, match="fallback .* missing"):
        get_renderer("slot:x", fallback="slot:also_missing")


def test_register_collision_raises():
    """Re-registering a slot with a different class raises ValueError."""

    @register_renderer("slot:test_x")
    class FirstRenderer(BaseRenderer):
        output_slot = "slot:test_x"

        def render_html(
            self, data: Any, context: Optional[Dict] = None
        ) -> str:
            return "<p>1</p>"

    with pytest.raises(ValueError, match="already registered"):

        @register_renderer("slot:test_x")
        class SecondRenderer(BaseRenderer):
            output_slot = "slot:test_x"

            def render_html(
                self, data: Any, context: Optional[Dict] = None
            ) -> str:
                return "<p>2</p>"


def test_register_idempotent_with_same_class():
    """Registering the same class twice is allowed (idempotent)."""

    @register_renderer("slot:test_y")
    class IdempotentRenderer(BaseRenderer):
        output_slot = "slot:test_y"

        def render_html(
            self, data: Any, context: Optional[Dict] = None
        ) -> str:
            return "<p/>"

    register_renderer("slot:test_y")(IdempotentRenderer)
    assert RENDERER_REGISTRY["slot:test_y"] is IdempotentRenderer


def test_list_renderers_returns_sorted():
    @register_renderer("slot:zzz")
    class ZRenderer(BaseRenderer):
        output_slot = "slot:zzz"

        def render_html(
            self, data: Any, context: Optional[Dict] = None
        ) -> str:
            return ""

    @register_renderer("slot:aaa")
    class ARenderer(BaseRenderer):
        output_slot = "slot:aaa"

        def render_html(
            self, data: Any, context: Optional[Dict] = None
        ) -> str:
            return ""

    result = list_renderers()
    assert result == sorted(result)
    assert "slot:aaa" in result
    assert "slot:zzz" in result


# -- raw_json rendering ----------------------------------------------------


def test_raw_json_render_html_crash_free():
    renderer = RawJsonRenderer()
    out = renderer.render_html({"a": 1, "b": [2, 3], "c": None})
    assert "<pre" in out
    assert '"a": 1' in out


def test_raw_json_render_html_handles_html_chars():
    """Embedded ``<``, ``>``, ``&`` must be escaped."""
    renderer = RawJsonRenderer()
    out = renderer.render_html({"x": "<script>alert(1)</script>"})
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_raw_json_render_cli_returns_json():
    renderer = RawJsonRenderer()
    out = renderer.render_cli({"k": "v"})
    parsed = json.loads(out)
    assert parsed == {"k": "v"}


# -- BaseRenderer default behavior -----------------------------------------


def test_base_renderer_default_cli_is_json():
    """BaseRenderer default render_cli falls back to json.dumps."""

    class MinimalRenderer(BaseRenderer):
        output_slot = "slot:tmp"

        def render_html(
            self, data: Any, context: Optional[Dict] = None
        ) -> str:
            return "<p/>"

    renderer = MinimalRenderer()
    out = renderer.render_cli({"k": [1, 2]})
    assert json.loads(out) == {"k": [1, 2]}
