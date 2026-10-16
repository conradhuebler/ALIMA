"""ImageAnalysisWorker used to call generate_response without output_format.

output_format=None defaults to JSON-mode on Gemini/OpenAI-compatible/Ollama
(see llm_service.py's three text generators). The tab's own DEFAULT_PROMPT
asks for plain OCR text and the result is displayed as-is
(analysis_finished.emit(response), no JSON parsing anywhere downstream) -
"xml" is this codebase's existing signal for "give me plain text"
(pipeline_input.py). - Claude Generated
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from src.ui.image_analysis_tab import ImageAnalysisWorker


class ImageAnalysisWorkerRequestsPlainTextTest(unittest.TestCase):
    def test_run_passes_output_format_xml(self):
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "extracted OCR text"

        worker = ImageAnalysisWorker(
            mock_llm, provider="ollama", model="llava",
            prompt="Extract text", image_path="/fake/path.png",
        )
        worker.run()

        mock_llm.generate_response.assert_called_once()
        kwargs = mock_llm.generate_response.call_args.kwargs
        self.assertEqual(kwargs["output_format"], "xml")


if __name__ == "__main__":
    unittest.main()
