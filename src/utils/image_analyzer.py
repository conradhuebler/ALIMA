"""Image analysis helper — Claude Generated for P-η.

Synchronous wrapper over `LlmService.generate_response(image=path)`.
Extracted from `src/ui/image_analysis_tab.py::ImageAnalysisWorker`
so the same logic is callable from MCP tools, CLI, and headless
contexts without Qt threading.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "Bitte extrahiere den gesamten lesbaren Text aus diesem Bild. "
    "Gib nur den Text zurück, ohne zusätzliche Formatierung oder Kommentare. "
    "Achte darauf, dass der Text genau so ausgegeben wird, wie er im Bild steht."
)


def _coalesce(response: Any) -> str:
    """Drain generator/iterable LLM responses into a single string."""
    if isinstance(response, str):
        return response
    if hasattr(response, "__iter__"):
        out = []
        for chunk in response:
            if isinstance(chunk, str):
                out.append(chunk)
            elif hasattr(chunk, "text"):
                out.append(chunk.text)
            elif hasattr(chunk, "content"):
                out.append(chunk.content)
            else:
                out.append(str(chunk))
        return "".join(out)
    return str(response)


def analyze(
    path: str,
    llm_service: Any,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    prompt: str = DEFAULT_PROMPT,
    temperature: float = 0.7,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a Vision LLM call against an image file.

    Returns dict with `text`, `prompt_used`, `provider`, `model`,
    `chars`. Raises FileNotFoundError if path missing, ValueError
    if provider/model unresolvable, RuntimeError on LLM failure.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    if llm_service is None:
        raise ValueError("llm_service is required for image analysis")

    if not provider or not model:
        config = getattr(llm_service, "config_manager", None)
        if config is not None:
            if not provider:
                provider = getattr(config, "default_image_provider", None) or getattr(
                    config, "default_provider", None
                )
            if not model:
                model = getattr(config, "default_image_model", None) or getattr(
                    config, "default_model", None
                )
        if not provider or not model:
            raise ValueError(
                "provider and model must be set (no default_image_provider/model found)"
            )

    request_id = str(uuid.uuid4())
    try:
        response = llm_service.generate_response(
            provider=provider,
            model=model,
            prompt=prompt,
            request_id=request_id,
            temperature=temperature,
            seed=seed,
            image=path,
            stream=False,
        )
    except Exception as exc:
        raise RuntimeError(f"LLM-Aufruf fehlgeschlagen: {exc}") from exc

    text = _coalesce(response).strip()
    return {
        "text": text,
        "prompt_used": prompt,
        "provider": provider,
        "model": model,
        "chars": len(text),
    }
