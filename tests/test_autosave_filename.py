"""Collision-free autosave filenames - Claude Generated.

The GUI autosave used ``{working_title}.json``. The classic path's working
title ends in a timestamp, so its files are unique; the agentic path's title is
just the LLM working title, so two agentic runs of the SAME document wrote the
same file and the second silently overwrote the first — a whole run lost. The
operator hit this: three runs, only two JSON files. This helper makes every run
land in its own file regardless of path.
"""

from __future__ import annotations

import re
import unittest
from datetime import datetime

from src.utils.pipeline_defaults import autosave_filename

_STAMPED = re.compile(r"_\d{8}_\d{6}\.json$")


class TestAutosaveFilename(unittest.TestCase):
    def setUp(self):
        self.when = datetime(2026, 7, 22, 10, 24, 0)

    def test_agentic_title_gets_a_timestamp(self):
        """The collision case: an agentic title has no timestamp of its own."""
        name = autosave_filename("Jha_Kumar_CadmiumToxicityMitigation", self.when)
        self.assertEqual(name, "Jha_Kumar_CadmiumToxicityMitigation_20260722_102400.json")

    def test_two_agentic_runs_do_not_collide(self):
        a = autosave_filename("Doc", datetime(2026, 7, 22, 10, 24, 0))
        b = autosave_filename("Doc", datetime(2026, 7, 22, 10, 26, 0))
        self.assertNotEqual(a, b)

    def test_already_timestamped_title_is_not_double_stamped(self):
        """The classic path already appends a timestamp — don't add a second."""
        title = "Jha_link_springer_com_20260722_102233"
        self.assertEqual(autosave_filename(title, self.when), f"{title}.json")

    def test_result_always_ends_in_a_timestamp_and_json(self):
        for title in ("Doc", "", None, "   ", "Already_20260101_000000"):
            with self.subTest(title=title):
                self.assertTrue(_STAMPED.search(autosave_filename(title, self.when)))

    def test_blank_title_falls_back_to_analysis(self):
        for title in ("", None, "   "):
            with self.subTest(title=title):
                self.assertTrue(autosave_filename(title, self.when).startswith("analysis_"))

    def test_ends_with_json(self):
        self.assertTrue(autosave_filename("X", self.when).endswith(".json"))


if __name__ == "__main__":
    unittest.main()
