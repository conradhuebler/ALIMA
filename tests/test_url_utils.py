"""Tests for src/core/url_utils.py — canonical GND/SWB URL builders. Claude Generated."""
from __future__ import annotations

import unittest

from src.core.url_utils import gnd_url, swb_ppn_url, extract_urls_from_json


class TestGndUrl(unittest.TestCase):
    def test_subject_id_with_check_digit(self):
        self.assertEqual(gnd_url("4047979-1"), "https://d-nb.info/gnd/4047979-1")

    def test_person_id_plain_digits(self):
        self.assertEqual(gnd_url("118540238"), "https://d-nb.info/gnd/118540238")

    def test_check_char_x(self):
        self.assertEqual(gnd_url("4047979-X"), "https://d-nb.info/gnd/4047979-X")

    def test_whitespace_is_stripped(self):
        self.assertEqual(gnd_url("  4047979-1 "), "https://d-nb.info/gnd/4047979-1")

    def test_invalid_returns_empty(self):
        for bad in ("", "abc", "4047979-12", "4047/979", "DK 530", None):
            self.assertEqual(gnd_url(bad), "", msg=f"{bad!r} should be rejected")


class TestSwbPpnUrl(unittest.TestCase):
    def test_plain_ppn(self):
        self.assertEqual(
            swb_ppn_url("106192760"),
            "https://swb.bsz-bw.de/DB=2.104/PPNSET?PPN=106192760&INDEXSET=21",
        )

    def test_ppn_with_check_char(self):
        self.assertTrue(swb_ppn_url("12345X").endswith("PPN=12345X&INDEXSET=21"))

    def test_invalid_returns_empty(self):
        for bad in ("", "12-3", "abc", "4047979-1", None):
            self.assertEqual(swb_ppn_url(bad), "", msg=f"{bad!r} should be rejected")


class TestExtractUrls(unittest.TestCase):
    def test_recursive_collection(self):
        data = {
            "a": "https://x.de",
            "b": [{"c": "http://y.de"}, "not-a-url", 5],
            "d": {"e": {"f": "https://z.de/path?q=1"}},
        }
        self.assertEqual(
            extract_urls_from_json(data),
            {"https://x.de", "http://y.de", "https://z.de/path?q=1"},
        )

    def test_non_url_strings_ignored(self):
        self.assertEqual(extract_urls_from_json("hello"), set())
        self.assertEqual(extract_urls_from_json(42), set())


if __name__ == "__main__":
    unittest.main()
