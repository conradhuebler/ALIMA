"""`log_caught` — telling a defect apart from an unlucky runtime - Claude Generated.

Four July-2026 defects survived behind the same ambiguity: a broad
``except Exception`` caught a programming error and logged it at
``debug``/``warning``, where it read exactly like an expected network or parse
failure. This helper keeps the catching and changes only the volume.
"""

from __future__ import annotations

import logging
import unittest
from unittest.mock import Mock

from src.utils.error_visibility import BUG_SHAPED_ERRORS, is_bug_shaped, log_caught


class TestClassification(unittest.TestCase):
    def test_the_four_real_defect_shapes_are_flagged(self):
        """Exactly the exception types behind the July defects."""
        for exc in (
            AttributeError("no attribute 'update_gnd_entry'"),
            NameError("name 'validation_issues' is not defined"),
            ImportError("No module named 'rdflib'"),
            TypeError("missing 1 required positional argument"),
        ):
            with self.subTest(exc=type(exc).__name__):
                self.assertTrue(is_bug_shaped(exc))

    def test_routine_external_failures_are_not_flagged(self):
        """Flagging these would drown the signal — they are normal for API data."""
        for exc in (
            KeyError("abstract"),
            IndexError("list index out of range"),
            ValueError("invalid literal for int()"),
            OSError("connection refused"),
            RuntimeError("provider down"),
        ):
            with self.subTest(exc=type(exc).__name__):
                self.assertFalse(is_bug_shaped(exc))

    def test_subclasses_count(self):
        """ModuleNotFoundError is an ImportError — the rdflib case exactly."""
        self.assertTrue(is_bug_shaped(ModuleNotFoundError("No module named 'rdflib'")))


class TestLogging(unittest.TestCase):
    def setUp(self):
        self.logger = Mock(spec=logging.Logger)

    def test_defect_goes_to_error(self):
        flagged = log_caught(self.logger, AttributeError("boom"), "Speichern")
        self.assertTrue(flagged)
        self.logger.error.assert_called_once()
        self.logger.warning.assert_not_called()

    def test_expected_failure_uses_the_callers_level(self):
        flagged = log_caught(
            self.logger, OSError("timeout"), "DNB-Abruf", expected_level="debug"
        )
        self.assertFalse(flagged)
        self.logger.debug.assert_called_once()
        self.logger.error.assert_not_called()

    def test_default_level_for_expected_is_warning(self):
        log_caught(self.logger, ValueError("nope"), "Parsen")
        self.logger.warning.assert_called_once()

    def test_message_names_the_type_and_the_context(self):
        log_caught(self.logger, AttributeError("no attr 'x'"), "GND-Sync")
        message = self.logger.error.call_args[0][0]
        self.assertIn("GND-Sync", message)
        self.assertIn("AttributeError", message)
        self.assertIn("no attr 'x'", message)

    def test_detail_is_appended_when_given(self):
        log_caught(self.logger, OSError("x"), "Abruf", detail="gnd_id=4035769-7")
        self.assertIn("4035769-7", self.logger.warning.call_args[0][0])

    def test_missing_logger_is_tolerated(self):
        """Call sites in objects without a logger must stay one-liners."""
        self.assertTrue(log_caught(None, AttributeError("x"), "irgendwo"))
        self.assertFalse(log_caught(None, OSError("x"), "irgendwo"))

    def test_unknown_level_falls_back_to_warning(self):
        log_caught(self.logger, OSError("x"), "y", expected_level="quatsch")
        self.logger.warning.assert_called_once()

    def test_nothing_is_re_raised(self):
        """Adoption must not be able to change control flow — only the log."""
        try:
            log_caught(self.logger, AttributeError("x"), "y")
        except Exception as exc:  # pragma: no cover - would be the failure
            self.fail(f"log_caught raised {exc!r}")


class TestContract(unittest.TestCase):
    def test_the_flagged_set_is_explicit_and_small(self):
        """A creeping list would make ERROR meaningless again."""
        self.assertEqual(
            set(BUG_SHAPED_ERRORS),
            {AttributeError, NameError, ImportError, TypeError},
        )


if __name__ == "__main__":
    unittest.main()
