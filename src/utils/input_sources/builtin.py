"""Built-in input sources: text / file / pdf / image - Claude Generated.

These wrap the extraction helpers that already live in ``pipeline_input`` so the
behaviour is byte-for-byte identical to the former ``execute_input_extraction``
if/elif — this refactor only moves the *dispatch* into the registry (Debt D-11),
not the extraction logic.
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional, Tuple

from src.core.plugins.schema import ConfigField, PluginDoc

from .registry import register_input_source


@register_input_source
class TextInputSource:
    id = "text"
    label = "Direkter Text"

    @classmethod
    def config_fields(cls) -> List[ConfigField]:
        return []

    @classmethod
    def doc(cls) -> PluginDoc:
        return PluginDoc(
            description="Direkter Text ohne Verarbeitung.",
            input="Roher Text.",
            output="Derselbe Text (getrimmt).",
        )

    def can_handle(self, source: str, input_type: str) -> bool:
        return input_type == "text"

    def extract(self, source, *, llm_service=None, stream_callback=None, logger=None, **opts) -> Tuple[str, str, str]:
        return source.strip(), "Direkter Text", "text"


@register_input_source
class FileInputSource:
    id = "file"
    label = "Textdatei"

    @classmethod
    def config_fields(cls) -> List[ConfigField]:
        return []

    @classmethod
    def doc(cls) -> PluginDoc:
        return PluginDoc(
            description="Liest eine Textdatei (utf-8 mit latin-1/cp1252-Fallback).",
            input="Pfad zu einer Textdatei.",
            output="Dateiinhalt als Text.",
        )

    def can_handle(self, source: str, input_type: str) -> bool:
        return input_type == "file" and os.path.isfile(source)

    def extract(self, source, *, llm_service=None, stream_callback=None, logger=None, **opts) -> Tuple[str, str, str]:
        # Verbatim from the former execute_input_extraction 'file' branch.
        try:
            with open(source, "r", encoding="utf-8") as f:
                text = f.read().strip()
                filename = os.path.basename(source)
                return text, f"Textdatei: {filename}", "file_read"
        except UnicodeDecodeError:
            for encoding in ["latin-1", "cp1252"]:
                try:
                    with open(source, "r", encoding=encoding) as f:
                        text = f.read().strip()
                        filename = os.path.basename(source)
                        return text, f"Textdatei: {filename} ({encoding})", "file_read"
                except UnicodeDecodeError:
                    continue
            raise Exception("Datei konnte nicht gelesen werden (Encoding-Problem)")


@register_input_source
class PdfInputSource:
    id = "pdf"
    label = "PDF"

    @classmethod
    def config_fields(cls) -> List[ConfigField]:
        return []

    @classmethod
    def doc(cls) -> PluginDoc:
        return PluginDoc(
            description="Extrahiert Text aus einem PDF (PyPDF2), mit Vision-LLM-OCR-Fallback "
            "bei schlechter Textqualität.",
            input="Pfad zu einer PDF-Datei.",
            output="Extrahierter Volltext.",
        )

    def can_handle(self, source: str, input_type: str) -> bool:
        return input_type == "pdf"

    def extract(self, source, *, llm_service=None, stream_callback=None, logger=None, **opts) -> Tuple[str, str, str]:
        from src.utils.pipeline_input import _extract_from_pdf_pipeline

        return _extract_from_pdf_pipeline(source, llm_service, stream_callback, logger)


@register_input_source
class ImageInputSource:
    id = "image"
    label = "Bild (OCR)"

    @classmethod
    def config_fields(cls) -> List[ConfigField]:
        return []

    @classmethod
    def doc(cls) -> PluginDoc:
        return PluginDoc(
            description="OCR eines Bildes über ein Vision-LLM.",
            input="Pfad zu einer Bilddatei (png/jpg/…).",
            output="Im Bild erkannter Text.",
        )

    def can_handle(self, source: str, input_type: str) -> bool:
        return input_type == "image"

    def extract(self, source, *, llm_service=None, stream_callback=None, logger=None, **opts) -> Tuple[str, str, str]:
        from src.utils.pipeline_input import _extract_from_image_pipeline

        return _extract_from_image_pipeline(source, llm_service, stream_callback, logger)
