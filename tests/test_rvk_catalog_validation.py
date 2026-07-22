"""Characterization tests for _validate_catalog_rvk_candidates - Claude Generated (F-13).

The 394-line method that decides which catalog-derived RVK candidates survive:
validate against the RVK API, mark each standard / non-standard / error, drop
artifacts. Untested until now because it is I/O-coupled (the ``rvk_api`` lookup
plugin + the WP2 cache). The three I/O points are mocked here so the DECISION
behavior is reproducible; the newly extracted pure helpers
(``_is_strong_anchor_candidate`` / ``_can_take_branch``, F-14) run inside it.
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from src.utils._pipeline_rvk_scoring import RvkScoringMixin


class _Host(RvkScoringMixin):
    def __init__(self):
        self.logger = Mock(level=100)
        self.cache_manager = Mock()

    def _alima_config_for_cache(self):
        return Mock()


def _kw(keyword, codes):
    """One keyword_results row: a keyword with RVK (or other) classifications."""
    return {
        "keyword": keyword,
        "classifications": [
            {"type": c.get("type", "RVK"), "dk": c["dk"], "count": c.get("count", 1),
             "titles": c.get("titles", [])}
            for c in codes
        ],
    }


def _fake_plugin(statuses):
    """A plugin whose validate_notation returns a canned status per notation.

    ``statuses`` maps a canonicalised notation → status dict. An unknown code
    gets "Notation Not Found" (the non-standard path).
    """
    plugin = Mock()

    def validate(code):
        return statuses.get(
            code, {"status": "not_found", "notation": code, "message": "Notation Not Found"}
        )

    plugin.validate_notation.side_effect = validate
    return plugin


class _ValidationHarness:
    """Patches the three I/O seams so the decision logic runs offline."""

    def __init__(self, plugin):
        self.plugin = plugin
        self._patchers = [
            patch("src.utils.lookups.resolve.build_lookup", return_value=plugin),
            patch("src.utils.lookups.cache.lookup_cache_enabled", return_value=False),
            patch("src.utils.lookups.cache.cached_call",
                  side_effect=lambda km, on, key, param, extra, fn: fn()),
        ]

    def __enter__(self):
        for p in self._patchers:
            p.start()
        return self

    def __exit__(self, *a):
        for p in self._patchers:
            p.stop()
        return False


def _run(host, keyword_results, plugin, **kw):
    with _ValidationHarness(plugin):
        return host._validate_catalog_rvk_candidates(keyword_results, **kw)


def _codes(result):
    """All (keyword, dk, status) triples from a cleaned result."""
    out = []
    for row in result:
        for c in row.get("classifications", []):
            out.append((row["keyword"], c.get("dk"),
                        c.get("rvk_validation_status")))
    return out


class TestCatalogRvkValidation(unittest.TestCase):
    def setUp(self):
        self.host = _Host()

    def test_disabled_plugin_passes_candidates_through_unvalidated(self):
        """rvk_api off → parity: candidates pass through unchanged, not dropped."""
        kw = [_kw("Limnologie", [{"dk": "WI 4700"}])]
        with patch("src.utils.lookups.resolve.build_lookup", return_value=None):
            out = self.host._validate_catalog_rvk_candidates(kw)
        self.assertEqual(out, kw)

    def test_standard_notation_is_kept_and_marked(self):
        plugin = _fake_plugin({
            "WI 4700": {"status": "standard", "notation": "WI 4700",
                        "label": "Limnologie", "ancestor_path": "Bio > Ökologie",
                        "register": ["Seen"], "branch_family": "WI"},
        })
        out = _run(self.host, [_kw("Limnologie", [{"dk": "WI 4700"}])], plugin)
        self.assertEqual(_codes(out), [("Limnologie", "WI 4700", "standard")])

    def test_not_found_but_plausible_is_kept_as_non_standard(self):
        plugin = _fake_plugin({})  # everything → not_found
        out = _run(self.host, [_kw("Limnologie", [{"dk": "WI 4700"}])], plugin)
        self.assertEqual(_codes(out), [("Limnologie", "WI 4700", "non_standard")])

    def test_non_rvk_classification_passes_through_untouched(self):
        plugin = _fake_plugin({})
        kw = [{"keyword": "X", "classifications": [
            {"type": "DK", "dk": "530.1", "count": 1}]}]
        out = _run(self.host, kw, plugin)
        self.assertEqual(out, kw)  # DK row unchanged, no validation status added

    def test_implausible_notation_is_dropped_as_artifact(self):
        """A code that is not a plausible RVK notation is dropped entirely."""
        plugin = _fake_plugin({})
        out = _run(self.host, [_kw("X", [{"dk": "123456789"}])], plugin)
        self.assertEqual(_codes(out), [])

    def test_standard_and_nonstandard_coexist_per_keyword(self):
        plugin = _fake_plugin({
            "WI 4700": {"status": "standard", "notation": "WI 4700"},
        })
        out = _run(self.host, [_kw("Limnologie",
                   [{"dk": "WI 4700"}, {"dk": "RB 1000"}])], plugin)
        got = {(dk, st) for _, dk, st in _codes(out)}
        self.assertIn(("WI 4700", "standard"), got)
        self.assertIn(("RB 1000", "non_standard"), got)

    def test_validation_error_is_kept_marked_not_dropped(self):
        """An API error must not silently drop the candidate."""
        plugin = _fake_plugin({
            "WI 4700": {"status": "validation_error", "notation": "WI 4700",
                        "message": "API down"},
        })
        out = _run(self.host, [_kw("Limnologie", [{"dk": "WI 4700"}])], plugin)
        self.assertEqual(_codes(out), [("Limnologie", "WI 4700", "validation_error")])

    def test_empty_input_yields_empty(self):
        self.assertEqual(_run(self.host, [], _fake_plugin({})), [])


if __name__ == "__main__":
    unittest.main()
