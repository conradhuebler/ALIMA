"""PDF text extraction helper — Claude Generated for P-η.

Pure-Python extraction (no Qt). Extracted from
`src/ui/unified_input_widget.py:93-158` so it can be called from
MCP tools, CLI, and headless contexts.

Optional LLM-OCR fallback via `image_analyzer.analyze` when the
raw text-extraction quality is poor.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, Optional

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

logger = logging.getLogger(__name__)


def _assess_text_quality(text: str) -> Dict[str, Any]:
    """Heuristic check whether PDF text-layer extraction is usable."""
    if not text or len(text.strip()) == 0:
        return {"is_good": False, "reason": "Kein Text gefunden"}

    char_count = len(text)
    word_count = len(text.split())

    if char_count < 50:
        return {"is_good": False, "reason": "Text zu kurz"}

    if word_count > 0:
        avg_word_length = char_count / word_count
        if avg_word_length < 2 or avg_word_length > 20:
            return {"is_good": False, "reason": "Ungewöhnliche Wortlängen"}

    special_char_ratio = (
        sum(1 for c in text if not c.isalnum() and c not in " \n\t.,!?;:-()[]")
        / len(text)
    )
    if special_char_ratio > 0.3:
        return {"is_good": False, "reason": "Zu viele Sonderzeichen"}

    lines_with_content = [line for line in text.split("\n") if len(line.strip()) > 5]
    if len(lines_with_content) < max(1, word_count // 20):
        return {"is_good": False, "reason": "Text fragmentiert"}

    return {"is_good": True, "reason": "Text-Qualität ausreichend"}


def _ocr_first_pages(
    path: str,
    page_count: int,
    llm_service: Any,
    provider: Optional[str],
    model: Optional[str],
    max_pages: int = 3,
) -> str:
    """LLM-Vision OCR on first N pages. Returns combined text."""
    try:
        import pdf2image
    except ImportError as exc:
        raise RuntimeError(
            "pdf2image nicht installiert — `pip install pdf2image`"
        ) from exc

    from .image_analyzer import analyze as analyze_image

    images = pdf2image.convert_from_path(
        path, first_page=1, last_page=min(max_pages, page_count), dpi=200
    )
    if not images:
        raise RuntimeError("PDF → Bild Konvertierung lieferte 0 Seiten")

    parts = []
    for idx, img in enumerate(images, start=1):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp.name, "PNG")
            tmp_path = tmp.name
        try:
            result = analyze_image(
                tmp_path,
                llm_service=llm_service,
                provider=provider,
                model=model,
            )
            parts.append(f"[Seite {idx}]\n{result.get('text', '').strip()}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return "\n\n".join(parts)


def extract_text(
    path: str,
    max_chars: Optional[int] = None,
    ocr_fallback: bool = False,
    llm_service: Any = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract text from a local PDF file.

    Returns dict with keys: `text`, `pages`, `quality`, `source`,
    `truncated`. Raises FileNotFoundError if path missing,
    RuntimeError if PyPDF2 absent.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PDF not found: {path}")
    if PyPDF2 is None:
        raise RuntimeError("PyPDF2 nicht installiert — `pip install PyPDF2`")

    with open(path, "rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        page_count = len(reader.pages)
        text_parts = [page.extract_text() or "" for page in reader.pages]

    raw_text = "\n\n".join(text_parts).strip()
    quality = _assess_text_quality(raw_text)
    source = "pypdf2"

    if not quality["is_good"] and ocr_fallback:
        if llm_service is None:
            logger.warning("OCR-Fallback angefordert, aber kein llm_service übergeben")
        else:
            try:
                raw_text = _ocr_first_pages(
                    path, page_count, llm_service, provider, model
                )
                source = "llm_ocr"
                quality = _assess_text_quality(raw_text)
            except Exception as exc:
                logger.error(f"OCR fallback fehlgeschlagen: {exc}")

    truncated = False
    if max_chars and max_chars > 0 and len(raw_text) > max_chars:
        raw_text = raw_text[:max_chars] + "\n[…truncated]"
        truncated = True

    return {
        "text": raw_text,
        "pages": page_count,
        "quality": quality,
        "source": source,
        "truncated": truncated,
        "chars": len(raw_text),
    }
