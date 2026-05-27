"""P-η + P-θ — Input-Beschaffung + Export tool tests. Claude Generated.

Covers:
1. ``pdf_extractor.extract_text`` on a real fixture PDF (quality + truncation).
2. ``image_analyzer.analyze`` with a mock LlmService.
3. ``exporters`` json/csv/tex/marc + ``load_state`` + ``default_output_path``.
4. ``report_renderer.render`` short + ub_freiberg templates (TeX only, no pdflatex).
5. MCP tool dispatch for ``read_pdf``, ``analyze_image``, ``export_results``, ``generate_report``.
6. ``scrape_url`` PDF auto-detect branch (mock requests).
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.utils import exporters, report_renderer
from src.utils.pdf_extractor import extract_text, _assess_text_quality
from src.utils.image_analyzer import analyze, DEFAULT_PROMPT
from src.mcp.tool_registry import ToolRegistry


# ============================================================
# Fixtures
# ============================================================


def _make_sample_pdf(path: str, text_lines: list[str]) -> None:
    """Minimal PDF generator using PyPDF2. Writes plain text on one page."""
    try:
        from reportlab.pdfgen import canvas as rl_canvas
        c = rl_canvas.Canvas(path)
        y = 800
        for line in text_lines:
            c.drawString(72, y, line)
            y -= 18
        c.save()
        return
    except ImportError:
        pass
    # Fallback: minimal hand-written PDF (only readable by PyPDF2 if structure is correct)
    # If reportlab unavailable, skip the PDF tests.
    raise unittest.SkipTest("reportlab not installed; skipping PDF generation")


def _sample_state() -> dict:
    """Canonical export-payload dict for export/report tests."""
    return {
        "session_id": "test-session-123",
        "exported_at": "2026-05-26T12:00:00",
        "status": "completed",
        "current_step": "classification",
        "input": {"type": "text", "text_preview": "Mikroplastik & Marine"},
        "results": {
            "working_title": "Mikroplastik in marinen Ökosystemen",
            "original_abstract": "Marine Ökosysteme sind durch Mikroplastik bedroht...",
            "initial_keywords": ["Mikroplastik", "Marine", "Ökosystem"],
            "final_keywords": ["Mikroplastik (4127527-9)", "Meeresökologie (4038492-7)"],
            "classifications": [
                {"system": "DK", "code": "504.054", "display": "DK 504.054"},
                {"system": "RVK", "code": "WI 4400", "display": "RVK WI 4400"},
            ],
        },
    }


# ============================================================
# PDF extractor
# ============================================================


class TestPdfExtractor(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.pdf_path = os.path.join(self.tmpdir, "sample.pdf")
        _make_sample_pdf(self.pdf_path, [
            "ALIMA Sacherschließung Test-Dokument",
            "Dieses ist ein einfacher Testtext für PyPDF2-Extraktion.",
            "Marine Mikroplastik-Konzentrationen sind seit 2020 deutlich angestiegen.",
            "Die Studie umfasst 12 Stationen entlang der Nordseeküste.",
        ])

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_extract_text_returns_quality_dict(self):
        result = extract_text(self.pdf_path)
        self.assertIn("text", result)
        self.assertIn("quality", result)
        self.assertIn("pages", result)
        self.assertGreater(result["pages"], 0)
        self.assertIn("Mikroplastik", result["text"])

    def test_max_chars_truncates(self):
        result = extract_text(self.pdf_path, max_chars=50)
        self.assertTrue(result["truncated"])
        self.assertLessEqual(len(result["text"]), 100)  # +truncated marker

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            extract_text(os.path.join(self.tmpdir, "nope.pdf"))


class TestQualityAssessment(unittest.TestCase):
    def test_good_text(self):
        text = "\n".join([
            "This is a normal sentence with multiple words and reasonable length.",
            "Marine plastic pollution is a growing concern for coastal ecosystems.",
            "Sample stations were placed along the coastline at regular intervals.",
            "The analysis covered samples collected over a two-year period.",
            "Results show a clear correlation between proximity to urban centers.",
        ])
        result = _assess_text_quality(text)
        self.assertTrue(result["is_good"], result.get("reason"))

    def test_empty_text(self):
        self.assertFalse(_assess_text_quality("")["is_good"])

    def test_too_short(self):
        self.assertFalse(_assess_text_quality("hi")["is_good"])


# ============================================================
# Image analyzer
# ============================================================


class TestImageAnalyzer(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.img_path = os.path.join(self.tmpdir, "sample.png")
        Path(self.img_path).write_bytes(b"fake-png-bytes")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_analyze_calls_llm_service(self):
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "extracted OCR text"
        result = analyze(
            self.img_path,
            llm_service=mock_llm,
            provider="ollama",
            model="llava",
        )
        self.assertEqual(result["text"], "extracted OCR text")
        self.assertEqual(result["provider"], "ollama")
        self.assertEqual(result["model"], "llava")
        mock_llm.generate_response.assert_called_once()
        kwargs = mock_llm.generate_response.call_args.kwargs
        self.assertEqual(kwargs["image"], self.img_path)
        self.assertFalse(kwargs["stream"])

    def test_missing_image_raises(self):
        mock_llm = MagicMock()
        with self.assertRaises(FileNotFoundError):
            analyze("/nonexistent/img.png", llm_service=mock_llm, provider="x", model="y")

    def test_no_llm_service_raises(self):
        with self.assertRaises(ValueError):
            analyze(self.img_path, llm_service=None, provider="x", model="y")

    def test_no_provider_or_model_raises(self):
        mock_llm = MagicMock(spec=[])  # no config_manager attr
        with self.assertRaises(ValueError):
            analyze(self.img_path, llm_service=mock_llm)

    def test_custom_prompt_passed_through(self):
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "x"
        result = analyze(
            self.img_path,
            llm_service=mock_llm,
            provider="p",
            model="m",
            prompt="Beschreibe dieses Bild",
        )
        self.assertEqual(result["prompt_used"], "Beschreibe dieses Bild")


# ============================================================
# Exporters
# ============================================================


class TestExporters(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.state = _sample_state()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_export_json(self):
        out = os.path.join(self.tmpdir, "out.json")
        path = exporters.export_json(self.state, out)
        self.assertEqual(path, out)
        with open(out) as fh:
            data = json.load(fh)
        self.assertEqual(data["session_id"], "test-session-123")

    def test_export_csv(self):
        out = os.path.join(self.tmpdir, "out.csv")
        exporters.export_csv(self.state, out)
        content = Path(out).read_text(encoding="utf-8")
        self.assertIn("kind,value,gnd_id,system,code,source", content)
        self.assertIn("Mikroplastik", content)
        self.assertIn("DK", content)
        self.assertIn("504.054", content)

    def test_export_tex(self):
        out = os.path.join(self.tmpdir, "out.tex")
        exporters.export_tex(self.state, out)
        content = Path(out).read_text(encoding="utf-8")
        self.assertIn("Mikroplastik in marinen", content)
        self.assertIn("\\subsection*{GND-Schlagwörter}", content)
        self.assertIn("Meeresökologie", content)
        # GND-ID should be stripped
        self.assertNotIn("(4038492-7)", content)

    def test_export_marc(self):
        out = os.path.join(self.tmpdir, "out.txt")
        exporters.export_marc(self.state, out)
        lines = Path(out).read_text(encoding="utf-8").splitlines()
        self.assertTrue(any(line.startswith("5550 ") for line in lines))
        self.assertTrue(any(line.startswith("6700 DK 504.054") for line in lines))
        self.assertTrue(any(line.startswith("6700 RVK WI 4400") for line in lines))

    def test_generate_k10plus_lines_with_chains(self):
        state = {
            "results": {
                "keyword_chains": [
                    {"chain": ["Mikroplastik (4127527-9)", "Meeresökologie"]},
                    {"chain": ["Umweltchemie", "Toxikologie"]},
                    ["Einzelschlagwort"],
                ],
                "final_keywords": ["Fallback"],
                "classifications": [
                    {"system": "DK", "code": "504.054", "display": "DK 504.054"},
                ],
            }
        }
        lines = exporters.generate_k10plus_lines(state)
        # First chain → 5550, second → 5551, third → 5552
        self.assertTrue(any(line == "5550 Mikroplastik" for line in lines))
        self.assertTrue(any(line == "5550 Meeresökologie" for line in lines))
        self.assertTrue(any(line == "5551 Umweltchemie" for line in lines))
        self.assertTrue(any(line == "5551 Toxikologie" for line in lines))
        self.assertTrue(any(line == "5552 Einzelschlagwort" for line in lines))
        # Classifications still 6700
        self.assertTrue(any(line.startswith("6700 ") for line in lines))

    def test_generate_k10plus_lines_no_chains_fallback(self):
        lines = exporters.generate_k10plus_lines(self.state)
        # No chains → falls back to flat final_keywords under 5550
        self.assertTrue(any(line == "5550 Mikroplastik" for line in lines))
        self.assertTrue(any(line == "5550 Meeresökologie" for line in lines))

    def test_extract_chains_normalizes_various_shapes(self):
        state = {
            "results": {
                "keyword_chains": [
                    {"chain": ["A", "B"]},
                    ["C", "D"],
                    "ignored_scalar",
                ],
                "final_llm_call_details": {},
            }
        }
        chains = exporters._extract_chains(state)
        self.assertEqual(chains, [["A", "B"], ["C", "D"]])

    def test_extract_chains_from_final_llm_call_details(self):
        state = {
            "results": {
                "final_llm_call_details": {
                    "keyword_chains": [["X", "Y"]],
                },
            }
        }
        chains = exporters._extract_chains(state)
        self.assertEqual(chains, [["X", "Y"]])

    def test_extract_chains_strips_gnd_ids(self):
        state = {
            "results": {
                "keyword_chains": [{"chain": ["Term (123-4)", "Other"]}],
            }
        }
        chains = exporters._extract_chains(state)
        self.assertEqual(chains, [["Term", "Other"]])

    def test_dispatch_unknown_format(self):
        with self.assertRaises(ValueError):
            exporters.export(self.state, "yaml", "/tmp/x")

    def test_load_state_latest(self):
        # Write 2 files, expect the newer one
        old = os.path.join(self.tmpdir, "old.json")
        new = os.path.join(self.tmpdir, "new.json")
        Path(old).write_text(json.dumps({"session_id": "old"}))
        Path(new).write_text(json.dumps({"session_id": "new"}))
        os.utime(old, (1, 1))
        os.utime(new, (1_000_000, 1_000_000))
        loaded = exporters.load_state("latest", autosave_dir=self.tmpdir)
        self.assertEqual(loaded["session_id"], "new")

    def test_default_output_path(self):
        path = exporters.default_output_path(self.state, "tex", output_dir=self.tmpdir)
        self.assertTrue(path.endswith(".tex"))
        self.assertIn("Mikroplastik", path)


# ============================================================
# Report renderer
# ============================================================


class TestReportRenderer(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.state = _sample_state()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_render_short(self):
        out = os.path.join(self.tmpdir, "short.tex")
        result = report_renderer.render("short", self.state, out)
        self.assertEqual(result["tex_path"], out)
        self.assertIsNone(result["pdf_path"])
        content = Path(out).read_text(encoding="utf-8")
        self.assertIn("\\documentclass", content)
        self.assertIn("Mikroplastik", content)
        self.assertIn("Meeresökologie", content)

    def test_render_ub_freiberg(self):
        out = os.path.join(self.tmpdir, "ub.tex")
        report_renderer.render("ub_freiberg", self.state, out)
        content = Path(out).read_text(encoding="utf-8")
        self.assertIn("ALIMA-Sacherschließungsbericht", content)
        self.assertIn("longtable", content)
        self.assertIn("504.054", content)

    def test_unknown_template_raises(self):
        with self.assertRaises(ValueError):
            report_renderer.render("nope", self.state, "/tmp/x.tex")

    def test_tex_escape(self):
        state = {"results": {"working_title": "Test & Co. 50% $value"}}
        out = os.path.join(self.tmpdir, "esc.tex")
        report_renderer.render("short", state, out)
        content = Path(out).read_text(encoding="utf-8")
        self.assertIn("\\&", content)
        self.assertIn("\\%", content)
        self.assertIn("\\$", content)


# ============================================================
# MCP tool dispatch
# ============================================================


class TestMcpToolDispatch(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.state = _sample_state()
        self.json_path = os.path.join(self.tmpdir, "fixture.json")
        Path(self.json_path).write_text(json.dumps(self.state), encoding="utf-8")

        self.registry = ToolRegistry()
        self.registry.register_all_tools()
        # Force autosave dir to tmpdir
        self.registry._get_autosave_dir = lambda: self.tmpdir

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_export_results_json(self):
        out = os.path.join(self.tmpdir, "exp.json")
        result = json.loads(self.registry.execute("export_results", {
            "format": "json",
            "source": "fixture.json",
            "output_path": out,
        }))
        self.assertEqual(result["format"], "json")
        self.assertEqual(result["output_path"], out)
        self.assertTrue(os.path.isfile(out))

    def test_export_results_marc(self):
        out = os.path.join(self.tmpdir, "exp.txt")
        result = json.loads(self.registry.execute("export_results", {
            "format": "marc",
            "source": "fixture.json",
            "output_path": out,
        }))
        self.assertNotIn("error", result)
        content = Path(out).read_text(encoding="utf-8")
        self.assertIn("5550 Mikroplastik", content)

    def test_export_results_latest(self):
        result = json.loads(self.registry.execute("export_results", {
            "format": "csv",
            "source": "latest",
        }))
        self.assertNotIn("error", result)
        self.assertTrue(os.path.isfile(result["output_path"]))

    def test_export_results_unknown_format(self):
        result = json.loads(self.registry.execute("export_results", {
            "format": "xml",
            "source": "fixture.json",
        }))
        self.assertIn("error", result)

    def test_generate_report_short(self):
        out = os.path.join(self.tmpdir, "rep.tex")
        result = json.loads(self.registry.execute("generate_report", {
            "template": "short",
            "source": "fixture.json",
            "output_path": out,
            "build_pdf": False,
        }))
        self.assertEqual(result["tex_path"], out)
        self.assertTrue(os.path.isfile(out))

    def test_read_pdf_missing_file(self):
        result = json.loads(self.registry.execute("read_pdf", {
            "path": "/definitely/not/here.pdf",
        }))
        self.assertIn("error", result)

    def test_analyze_image_without_llm(self):
        # No llm_service injected & no Qt → should fail gracefully
        result = json.loads(self.registry.execute("analyze_image", {
            "path": self.json_path,  # any existing file
        }))
        self.assertIn("error", result)


class TestScrapeUrlPdfBranch(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.registry = ToolRegistry()
        self.registry.register_all_tools()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_pdf_content_type_routes_to_extractor(self):
        # Generate a real PDF, mock requests to return it
        pdf_path = os.path.join(self.tmpdir, "fake.pdf")
        try:
            _make_sample_pdf(pdf_path, ["Header line", "Inhalt der Testseite."])
        except unittest.SkipTest:
            self.skipTest("reportlab not installed")
        pdf_bytes = Path(pdf_path).read_bytes()

        mock_resp = MagicMock()
        mock_resp.content = pdf_bytes
        mock_resp.headers = {"Content-Type": "application/pdf"}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            result = json.loads(self.registry.execute("scrape_url", {
                "url": "https://example.org/paper.pdf",
            }))
        self.assertEqual(result.get("source"), "pdf")
        self.assertIn("Inhalt", result.get("text", ""))
        self.assertIn("pdf", result)
        self.assertEqual(result["pdf"]["extraction_source"], "pypdf2")


if __name__ == "__main__":
    unittest.main()
