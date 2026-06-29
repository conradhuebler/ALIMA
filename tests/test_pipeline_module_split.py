"""Regression tests for the pipeline_utils module split - Claude Generated.

These exercise code paths in the extracted modules that depend on stdlib names
which must travel with the moved code (urlparse, datetime, asdict, dataclasses).
The split originally dropped these imports — the failures only surfaced at
runtime on specific inputs (URL/DOI sources, JSON save), not at import time, so
the original suite missed them. Locks them down.
"""

from __future__ import annotations

import unittest

from src.utils.pipeline_text_utils import build_working_title, extract_source_identifier
from src.utils.pipeline_persistence import PipelineJsonManager
from src.core.data_models import TaskState


class TestTextUtilsStdlibPaths(unittest.TestCase):
    def test_extract_source_identifier_url_uses_urlparse(self):
        """URL source → netloc via urlparse (the path that raised NameError)."""
        self.assertEqual(
            extract_source_identifier("url", "https://doi.org/10.1/abc"), "doi.org"
        )

    def test_extract_source_identifier_file_uses_pathlib(self):
        self.assertEqual(
            extract_source_identifier("pdf", "/tmp/Jha_Kumar_Cadmium.pdf"),
            "Jha_Kumar_Cadmium",
        )

    def test_build_working_title_with_llm_title_uses_datetime(self):
        title = build_working_title("Cadmium Toxikologie", "doi.org")
        # llm slug + source + a yyyymmdd_hhmmss timestamp (datetime path)
        self.assertTrue(title.startswith("Cadmium_Toxikologie_doi_org_"))
        self.assertRegex(title, r"\d{8}_\d{6}$")

    def test_build_working_title_fallback_without_llm_title(self):
        title = build_working_title(None, "doi.org")
        self.assertTrue(title.startswith("analysis_doi_org_"))


class TestPersistenceStdlibPaths(unittest.TestCase):
    def test_task_state_to_dict_uses_asdict(self):
        """task_state_to_dict relies on dataclasses.asdict (a moved import)."""
        ts = TaskState(abstract_data=None, analysis_result=None)
        out = PipelineJsonManager.task_state_to_dict(ts)
        self.assertIsInstance(out, dict)
        self.assertIn("status", out)


if __name__ == "__main__":
    unittest.main()
