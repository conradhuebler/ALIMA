#!/usr/bin/env python3
"""Claude Generated - Tests for the shared classification-system registry."""

import unittest

from src.utils.classification_systems import (
    KNOWN_SYSTEMS,
    classification_system,
    format_classification,
    split_classification_code,
)


class TestClassificationSystems(unittest.TestCase):

    def test_split_dk_ddc_rvk(self):
        self.assertEqual(split_classification_code("DK 530.145"), ("DK", "530.145"))
        self.assertEqual(split_classification_code("DDC 530.1"), ("DDC", "530.1"))
        self.assertEqual(split_classification_code("RVK UC 100"), ("RVK", "UC 100"))

    def test_split_prefix_case_insensitive_code_preserved(self):
        self.assertEqual(split_classification_code("dk 530.145"), ("DK", "530.145"))
        self.assertEqual(split_classification_code("ddc 530"), ("DDC", "530"))
        self.assertEqual(split_classification_code("rvk uc 100"), ("RVK", "uc 100"))

    def test_split_unprefixed_and_empty(self):
        self.assertEqual(split_classification_code("530.145"), ("", "530.145"))
        self.assertEqual(split_classification_code(""), ("", ""))
        self.assertEqual(split_classification_code(None), ("", ""))

    def test_format(self):
        self.assertEqual(format_classification("DDC", "530.1"), "DDC 530.1")
        self.assertEqual(format_classification("ddc", "530.1"), "DDC 530.1")
        self.assertEqual(format_classification("", "530"), "530")
        self.assertEqual(format_classification("FOO", "x"), "x")  # unknown system → code only

    def test_roundtrip(self):
        for s in ("DK 530.145", "DDC 004.43", "RVK ST 250"):
            system, code = split_classification_code(s)
            self.assertEqual(format_classification(system, code), s)

    def test_classification_system_helper(self):
        self.assertEqual(classification_system("DDC 530"), "DDC")
        self.assertEqual(classification_system("plain"), "")

    def test_ddc_is_registered(self):
        self.assertIn("DDC", KNOWN_SYSTEMS)
        # BK joined for the WP-D2 lobid harvest: lobid ships Basisklassifikation
        # notations, and an unregistered system is DROPPED rather than kept, so
        # the registry is what decides whether harvested data survives.
        self.assertEqual(set(KNOWN_SYSTEMS), {"DK", "DDC", "RVK", "BK"})


if __name__ == "__main__":
    unittest.main()
