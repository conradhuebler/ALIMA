"""LobidSuggester lazy subject loading - Claude Generated.

Guards the July 19 change: constructing the suggester must be I/O-free (the
former ``prepare(False)`` in ``__init__`` downloaded + parsed the 25 MB DNB
dump on every first provider build). The dump is loaded lazily by the first
``transform()`` that needs the GND-ID→label table, and the default data dir
lives under the persistent per-user config dir instead of tempdir.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from src.core.search.providers.lobid.suggester import LobidSuggester
    IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover
    IMPORT_ERROR = exc


@unittest.skipIf(IMPORT_ERROR is not None, f"stack unavailable: {IMPORT_ERROR}")
class LobidLazyInitTest(unittest.TestCase):
    def test_construction_does_no_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(
                LobidSuggester, "prepare",
                side_effect=AssertionError("prepare() must not run in __init__"),
            ):
                suggester = LobidSuggester(data_dir=tmp)
            self.assertIsNone(suggester.gnd_subjects)

    def test_first_transform_triggers_lazy_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            suggester = LobidSuggester(data_dir=tmp)

            def fake_prepare(force_gnd_download=False):
                suggester.gnd_subjects = {"4064937-4": "Wasser"}

            raw = {
                "aggregation": {
                    "subject.componentList.id": [
                        {"key": "https://d-nb.info/gnd/4064937-4", "doc_count": 7}
                    ]
                }
            }
            with patch.object(LobidSuggester, "prepare", side_effect=fake_prepare) as prep:
                first = suggester.transform(raw)
                second = suggester.transform(raw)
            prep.assert_called_once()  # memoised: no second load
            self.assertEqual(first, second)
            self.assertEqual(first["Wasser"]["count"], 7)
            self.assertEqual(first["Wasser"]["gnd_ids"], {"4064937-4"})

    def test_default_data_dir_is_persistent_per_user(self):
        default = LobidSuggester.default_data_dir()
        self.assertIn(".config", str(default))
        self.assertEqual(default.name, "lobidsuggester")
        self.assertNotIn(tempfile.gettempdir(), str(default))


if __name__ == "__main__":
    unittest.main()
