"""P2 regression gate: LobidSuggester fetch/transform split - Claude Generated.

Locks the reduced pool view produced by ``LobidSuggester.transform`` to a golden
value (byte-identical to the pre-split behaviour), and verifies the new
``transform_agent_view`` surfaces the ``member``/``totalItems`` data the reduced
view deliberately drops. ``transform`` is exercised with a hand-built ``self`` so
no network / GND download is needed.
"""

import json
import logging
import os
import unittest

try:
    from src.core.search.providers.lobid.suggester import LobidSuggester
    IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover
    IMPORT_ERROR = exc

_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "lobid_raw_wasser.json")


class _FakeLobid:
    """Minimal stand-in carrying only what ``transform`` reads off ``self``."""

    def __init__(self, gnd_subjects):
        self.gnd_subjects = gnd_subjects
        self.debug = False
        self.logger = logging.getLogger("test_lobid")


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class LobidTransformGateTest(unittest.TestCase):
    def setUp(self):
        with open(_FIXTURE, encoding="utf-8") as fh:
            self.raw = json.load(fh)
        # Two GND IDs map to the same subject "Wasser"; the third is unknown and
        # falls back to the bare GND id.
        self.fake = _FakeLobid({"4064937-4": "Wasser", "4064938-6": "Wasser"})

    def test_reduced_view_is_byte_identical_golden(self):
        got = LobidSuggester.transform(self.fake, self.raw, "kw")
        expected = {
            "Wasser": {
                "count": 50,  # max(47, 50) — dedup keeps the higher count
                "gndid": {"4064937-4", "4064938-6"},
                "ddc": set(),
                "dk": set(),
            },
            "4030550-8": {  # unknown id → subject falls back to the bare id
                "count": 12,
                "gndid": {"4030550-8"},
                "ddc": set(),
                "dk": set(),
            },
        }
        self.assertEqual(got, expected)

    def test_reduced_view_drops_member_and_totalitems(self):
        got = LobidSuggester.transform(self.fake, self.raw, "kw")
        self.assertEqual(set(got.keys()), {"Wasser", "4030550-8"})
        # The reduced view must not leak the record-level data.
        for data in got.values():
            self.assertNotIn("member", data)
            self.assertNotIn("totalItems", data)

    def test_agent_view_surfaces_member_and_totalitems(self):
        view = LobidSuggester.transform_agent_view(self.raw)
        self.assertEqual(view["totalItems"], 1234)
        self.assertEqual(len(view["member"]), 1)
        self.assertEqual(view["member"][0]["title"], "Wasser und Klima")


if __name__ == "__main__":
    unittest.main()
