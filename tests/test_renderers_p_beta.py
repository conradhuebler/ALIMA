"""Tests for P-β renderers - Claude Generated.

Covers WP10 P-β smoke-test criteria for the 5 renderers
(dk_table, gnd_pool, keyword_chains, duplicate_table, title_list).

Snapshot-Strategie: HTML-Output gegen Fixture in tests/fixtures/p_beta/.
Set ``P_BETA_REGENERATE_SNAPSHOTS=1`` zum (Neu-)Erstellen der Fixtures.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.ui.renderers import (
    RENDERER_REGISTRY,
    get_renderer,
)
from src.ui.renderers.dk_table import DkTableRenderer
from src.ui.renderers.duplicate_table import DuplicateTableRenderer
from src.ui.renderers.gnd_pool import GndPoolRenderer
from src.ui.renderers.keyword_chains import KeywordChainsRenderer
from src.ui.renderers.registry import _reset_for_tests
from src.ui.renderers.title_list import TitleListRenderer

from tests._html_compare import assert_html_equal


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "p_beta"
REGENERATE = os.environ.get("P_BETA_REGENERATE_SNAPSHOTS") == "1"


@pytest.fixture(autouse=True)
def _restore_registry():
    """Snapshot+restore the registry around each test."""
    snapshot = dict(RENDERER_REGISTRY)
    yield
    _reset_for_tests()
    RENDERER_REGISTRY.update(snapshot)


def _snapshot_path(name: str) -> Path:
    return FIXTURE_DIR / name


def _load_or_record(name: str, actual: str) -> str:
    """Return expected snapshot, or write+return ``actual`` if regenerating."""
    path = _snapshot_path(name)
    if REGENERATE or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
        return actual
    return path.read_text(encoding="utf-8")


# -- Test data fixtures ---------------------------------------------------

DK_TABLE_DATA = [
    {"dk": "57.62", "type": "DK", "count": 7, "titles": ["Schwermetalle in Böden"]},
    {"dk": "613.6", "type": "DK", "count": 0, "titles": []},
    {"dk": "504.064", "type": "DK", "count": 25, "titles": [
        "Cadmium-Toxikologie", "Umweltgifte", "Marine Belastung"]},
]

GND_POOL_DATA = [
    {"search_term": "Cadmium", "keyword": "Cadmium", "count": 42, "gnd_id": "4007249-3"},
    {"search_term": "Cadmium", "keyword": "Schwermetall", "count": 8, "gnd_id": "4053458-3"},
]

KEYWORD_CHAINS_DATA = [
    {"chain": ["Cadmium", "Ökotoxikologie"], "reason": "Hauptthema"},
    {"chain": ["Aquatische Systeme"], "reason": "Umfeld"},
]

DUPLICATE_TABLE_DATA = [
    {"input_title": "Cadmium in soils", "status": "duplicate",
     "matches": [{"rsn": "123"}], "reasoning": "ISBN exact match"},
    {"input_title": "Heavy metals revisited", "status": "new",
     "matches": [], "reasoning": "no overlap"},
]

TITLE_LIST_DATA = [
    {"title": "Cadmium in soils", "authors": ["Müller, K."],
     "year": 2024, "isbn": "978-3-123", "dk_codes": ["57.62"]},
    {"title": "Heavy metals", "authors": [], "year": None, "isbn": "",
     "dk_codes": [], "rvk_codes": []},
]


# -- Registration --------------------------------------------------------

def test_all_five_renderers_registered():
    expected_slots = {
        "slot:dk_table",
        "slot:gnd_pool",
        "slot:keyword_chains",
        "slot:duplicate_table",
        "slot:title_list",
    }
    assert expected_slots.issubset(set(RENDERER_REGISTRY))


@pytest.mark.parametrize(
    "slot,expected_cls",
    [
        ("slot:dk_table", DkTableRenderer),
        ("slot:gnd_pool", GndPoolRenderer),
        ("slot:keyword_chains", KeywordChainsRenderer),
        ("slot:duplicate_table", DuplicateTableRenderer),
        ("slot:title_list", TitleListRenderer),
    ],
)
def test_get_renderer_returns_specific_class_no_fallback(slot, expected_cls):
    """Each migrated renderer must be reachable by its own slot, not via fallback."""
    cls = get_renderer(slot)
    assert cls is expected_cls, f"{slot} → {cls.__name__} (expected {expected_cls.__name__})"


# -- HTML snapshots ------------------------------------------------------

def test_dk_table_html_matches_snapshot():
    actual = DkTableRenderer().render_html(DK_TABLE_DATA)
    expected = _load_or_record("dk_table_snapshot.html", actual)
    assert_html_equal(actual, expected)


def test_gnd_pool_html_matches_snapshot():
    actual = GndPoolRenderer().render_html(GND_POOL_DATA)
    expected = _load_or_record("gnd_pool_snapshot.html", actual)
    assert_html_equal(actual, expected)


def test_keyword_chains_html_matches_snapshot():
    actual = KeywordChainsRenderer().render_html(KEYWORD_CHAINS_DATA)
    expected = _load_or_record("keyword_chains_snapshot.html", actual)
    assert_html_equal(actual, expected)


def test_duplicate_table_html_matches_snapshot():
    actual = DuplicateTableRenderer().render_html(DUPLICATE_TABLE_DATA)
    expected = _load_or_record("duplicate_table_snapshot.html", actual)
    assert_html_equal(actual, expected)


def test_title_list_html_matches_snapshot():
    actual = TitleListRenderer().render_html(TITLE_LIST_DATA)
    expected = _load_or_record("title_list_snapshot.html", actual)
    assert_html_equal(actual, expected)


# -- Crash-free on empty input ------------------------------------------

@pytest.mark.parametrize(
    "renderer_cls",
    [
        DkTableRenderer,
        GndPoolRenderer,
        KeywordChainsRenderer,
        DuplicateTableRenderer,
        TitleListRenderer,
    ],
)
def test_render_html_handles_empty_data(renderer_cls):
    """render_html must not crash on empty list / None."""
    renderer = renderer_cls()
    assert isinstance(renderer.render_html([]), str)
    assert isinstance(renderer.render_html(None), str)


@pytest.mark.parametrize(
    "renderer_cls",
    [
        DkTableRenderer,
        GndPoolRenderer,
        KeywordChainsRenderer,
        DuplicateTableRenderer,
        TitleListRenderer,
    ],
)
def test_render_cli_returns_string(renderer_cls):
    """render_cli must return a string for empty + populated input."""
    renderer = renderer_cls()
    assert isinstance(renderer.render_cli([]), str)
    assert isinstance(renderer.render_cli([{"x": 1}]), str)


# -- Content-spot-checks (not snapshot) ---------------------------------

def test_dk_table_html_contains_dk_codes():
    html = DkTableRenderer().render_html(DK_TABLE_DATA)
    for row in DK_TABLE_DATA:
        assert row["dk"] in html


def test_gnd_pool_html_contains_gnd_ids():
    html = GndPoolRenderer().render_html(GND_POOL_DATA)
    for row in GND_POOL_DATA:
        assert row["gnd_id"] in html


def test_keyword_chains_html_contains_arrow():
    html = KeywordChainsRenderer().render_html(KEYWORD_CHAINS_DATA)
    assert "→" in html or "&rarr;" in html
    assert "Hauptthema" in html


def test_duplicate_table_html_color_codes_status():
    html = DuplicateTableRenderer().render_html(DUPLICATE_TABLE_DATA)
    # 'duplicate' status renders with red color (#dc3545)
    assert "#dc3545" in html
    # 'new' status renders with green color (#28a745)
    assert "#28a745" in html


def test_title_list_html_renders_authors():
    html = TitleListRenderer().render_html(TITLE_LIST_DATA)
    assert "Müller, K." in html
    assert "Cadmium in soils" in html


# -- KeywordChains truncation-sentinel handling -------------------------

def test_keyword_chains_handles_truncation_sentinel():
    data = list(KEYWORD_CHAINS_DATA) + [{"_truncated": 5}]
    html = KeywordChainsRenderer().render_html(data)
    assert "+5 more" in html
    # Sentinel itself must not appear as a chain row
    assert "_truncated" not in html


def test_keyword_chains_handles_explicit_truncation_context():
    html = KeywordChainsRenderer().render_html(
        KEYWORD_CHAINS_DATA, context={"truncated": 3}
    )
    assert "+3 more" in html


# -- GndPool fill_table-helper ------------------------------------------

def test_gnd_pool_normalize_accepts_raw_pool_shape():
    """Tolerates {gnd_id, title, count} as well as the AnalysisReviewTab
    normalized shape."""
    renderer = GndPoolRenderer()
    raw_pool = [{"gnd_id": "4007249-3", "title": "Cadmium", "count": 42}]
    html = renderer.render_html(raw_pool)
    assert "4007249-3" in html
    assert "Cadmium" in html
