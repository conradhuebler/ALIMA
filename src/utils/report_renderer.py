"""Jinja2 TeX report renderer — Claude Generated for P-θ.

Renders LaTeX report templates from `src/utils/report_templates/`
filled with values from a canonical export payload (the dict
produced by `webapp.result_serialization.build_export_payload`).

Optional `build_pdf=True` runs `pdflatex` twice to produce a PDF.
Missing `pdflatex` is non-fatal: TeX is still written and the
return value carries a warning.

To avoid the `{{ }}` / `{% %}` collision with raw LaTeX, Jinja
uses custom delimiters:
- block:    `((*  *))`
- variable: `(((  )))`
- comment:  `((#  #))`
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import jinja2
except ImportError:
    jinja2 = None  # surfaced lazily so unit tests that don't render still import OK

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent / "report_templates"
KNOWN_TEMPLATES = {"ub_freiberg": "ub_freiberg.tex.j2", "short": "short.tex.j2"}


def _tex_escape(value: Any) -> str:
    if value is None:
        return ""
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    out = str(value)
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def _env() -> "jinja2.Environment":
    if jinja2 is None:
        raise RuntimeError("Jinja2 nicht installiert — `pip install jinja2`")
    env = jinja2.Environment(
        block_start_string="((*",
        block_end_string="*))",
        variable_start_string="(((",
        variable_end_string=")))",
        comment_start_string="((#",
        comment_end_string="#))",
        loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["tex"] = _tex_escape
    return env


def _results(state: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(state.get("results"), dict):
        return state["results"]
    return state


def _strip_gnd_id(keyword: str) -> str:
    if "(" in keyword and ")" in keyword:
        return keyword.split("(")[0].strip()
    return keyword.strip()


def _build_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the export payload into template-friendly vars."""
    results = _results(state)
    inp = state.get("input") or {}

    final_kws_raw: List[str] = [str(k) for k in (results.get("final_keywords") or [])]
    final_kws = [_strip_gnd_id(k) for k in final_kws_raw]

    classifications = []
    for entry in results.get("classifications") or []:
        if isinstance(entry, dict):
            classifications.append({
                "display": entry.get("display") or entry.get("code") or "",
                "system": entry.get("system") or "",
                "code": entry.get("code") or "",
            })
        else:
            text = str(entry)
            from .classification_systems import split_classification_code
            system, code = split_classification_code(text)
            classifications.append({"display": text, "system": system, "code": code})

    return {
        "title": results.get("working_title") or inp.get("text_preview") or "ALIMA-Analyse",
        "abstract": results.get("original_abstract") or "",
        "initial_keywords": [str(k) for k in (results.get("initial_keywords") or [])],
        "final_keywords": final_kws,
        "final_keywords_raw": final_kws_raw,
        "classifications": classifications,
        "session_id": state.get("session_id") or "",
        "exported_at": state.get("exported_at") or "",
        "status": state.get("status") or "",
    }


def _run_pdflatex(tex_path: str) -> Optional[str]:
    """Run pdflatex twice (refs/labels), return PDF path or None on failure."""
    if shutil.which("pdflatex") is None:
        return None
    workdir = os.path.dirname(os.path.abspath(tex_path)) or "."
    base = os.path.splitext(os.path.basename(tex_path))[0]
    pdf_path = os.path.join(workdir, f"{base}.pdf")
    for _ in range(2):
        try:
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", os.path.basename(tex_path)],
                cwd=workdir,
                check=True,
                capture_output=True,
                timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            logger.error(f"pdflatex failed: {exc}")
            return None
    return pdf_path if os.path.isfile(pdf_path) else None


def render(
    template: str,
    state: Dict[str, Any],
    output_path: str,
    build_pdf: bool = False,
) -> Dict[str, Any]:
    """Render a report template with state vars to `output_path` (.tex).

    Returns dict with keys: `tex_path`, `pdf_path` (or None), `warnings`.
    """
    if template not in KNOWN_TEMPLATES:
        raise ValueError(
            f"Unknown template '{template}'. Available: {sorted(KNOWN_TEMPLATES)}"
        )

    env = _env()
    tmpl = env.get_template(KNOWN_TEMPLATES[template])
    ctx = _build_context(state)
    rendered = tmpl.render(**ctx)

    Path(os.path.dirname(os.path.abspath(output_path)) or ".").mkdir(
        parents=True, exist_ok=True
    )
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(rendered)

    warnings: List[str] = []
    pdf_path: Optional[str] = None
    if build_pdf:
        pdf_path = _run_pdflatex(output_path)
        if pdf_path is None:
            warnings.append(
                "pdflatex nicht verfügbar oder Build fehlgeschlagen — TeX wurde geschrieben"
            )

    return {"tex_path": output_path, "pdf_path": pdf_path, "warnings": warnings}
