"""Regression tests for GET /api/session/{id} result exposure - Claude Generated.

Bug: the extract-only frontend flow (DOI/URL/PDF/image) polls
``GET /api/session/{id}`` and reads ``results.original_abstract``. The handler
used to omit ``results`` entirely, so the frontend read ``undefined`` and
reported "Keine Textextraktion möglich" even though extraction succeeded
server-side. Extraction-only results (small) must therefore be included here;
full-pipeline results (potentially large) stay out and are fetched via
``/api/export/{id}``.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class TestSessionResultsExposure(unittest.TestCase):
    def _client(self):
        from fastapi.testclient import TestClient
        from src.webapp import app as appmod
        return TestClient(appmod.app), appmod

    def test_extraction_only_results_present_in_session_poll(self):
        client, appmod = self._client()
        sid = "test-extract-only"
        session = appmod.Session(sid)
        session.status = "completed"
        session.results = {
            "original_abstract": "Cadmium contamination of soil and plants.",
            "input_type": "doi",
            "input_mode": "extraction_only",
            "source_info": "10.1007/978-3-031-47390-6",
            "extraction_method": "text",
        }
        appmod.sessions[sid] = session
        try:
            resp = client.get(f"/api/session/{sid}")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("results", data)
            self.assertEqual(
                data["results"]["original_abstract"],
                "Cadmium contamination of soil and plants.",
            )
            self.assertEqual(data["results"]["extraction_method"], "text")
        finally:
            appmod.sessions.pop(sid, None)

    def test_full_pipeline_results_omitted_from_session_poll(self):
        """Non-extraction results are kept out of the poll response (size)."""
        client, appmod = self._client()
        sid = "test-full-pipeline"
        session = appmod.Session(sid)
        session.status = "completed"
        session.results = {
            "original_abstract": "x" * 10,
            "input_mode": "full_pipeline",
            "final_keywords": ["Cadmium (GND-ID: 4009274-4)"],
        }
        appmod.sessions[sid] = session
        try:
            resp = client.get(f"/api/session/{sid}")
            self.assertEqual(resp.status_code, 200)
            self.assertNotIn("results", resp.json())
        finally:
            appmod.sessions.pop(sid, None)


if __name__ == "__main__":
    unittest.main()
