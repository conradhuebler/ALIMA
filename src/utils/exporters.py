"""Pipeline result exporters — Claude Generated for P-θ.

Format-specific writers operating on the canonical export-payload
dict produced by `src/webapp/result_serialization.build_export_payload`.

All exporters return the written file path on success and raise on
failure. Pure-Python, no Qt.

Supported formats:
- `json`: canonical web/API export payload (same as `pipeline_utils.export_analysis_state_to_file`)
- `csv`: flat row-per-keyword/classification table
- `tex`: minimal stand-alone LaTeX snippet (sections: title, abstract, keywords, classifications)
- `marc`: the WinIBW entry of ``winibw_record.build_winibw_record`` (555x
  chains/single keywords, 1131 form keywords, 6700 notations)
"""

from __future__ import annotations

import csv
import glob
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .winibw_record import build_winibw_record, chains_from_payload
from .winibw_record import strip_gnd_id as _strip_gnd_id

logger = logging.getLogger(__name__)


# ============================================================
# State loading
# ============================================================


def load_state(source: str, autosave_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load an export-payload dict by source spec.

    - `source='latest'`: most-recent JSON in `autosave_dir`.
    - `source=<filename>`: literal file in `autosave_dir`.
    - `source=<absolute_path>`: load that exact file.

    Returns the parsed JSON dict. Raises FileNotFoundError if no
    file resolves.
    """
    if not source:
        raise ValueError("source must be set (use 'latest' or a filename/path)")

    if os.path.isabs(source) and os.path.isfile(source):
        path = source
    else:
        if not autosave_dir:
            raise ValueError("autosave_dir required for non-absolute source")
        if source == "latest":
            files = sorted(
                glob.glob(os.path.join(autosave_dir, "*.json")),
                key=os.path.getmtime,
                reverse=True,
            )
            if not files:
                raise FileNotFoundError(f"No JSON files in {autosave_dir}")
            path = files[0]
        else:
            path = os.path.join(autosave_dir, source)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data["__source_path__"] = path
    return data


# ============================================================
# Helpers
# ============================================================


def _results(state: Dict[str, Any]) -> Dict[str, Any]:
    """Get the `results` sub-dict from an export payload (or the dict itself)."""
    if isinstance(state.get("results"), dict):
        return state["results"]
    return state


def _input(state: Dict[str, Any]) -> Dict[str, Any]:
    return state.get("input") or {}


def _split_classification(value: Any) -> tuple[str, str]:
    """Return (system, code) from a string or dict entry."""
    if isinstance(value, dict):
        return (
            str(value.get("system") or "").strip().upper(),
            str(value.get("code") or "").strip(),
        )
    from .classification_systems import split_classification_code
    return split_classification_code(value)


def _extract_chains(state: Dict[str, Any]) -> List[List[str]]:
    """Keyword chains of the payload as list-of-lists, GND ids stripped.

    Kept for existing callers; the logic lives in
    ``winibw_record.chains_from_payload``. - Claude Generated
    """
    return chains_from_payload(state)


def generate_k10plus_lines(state: Dict[str, Any]) -> List[str]:
    """K10+/WinIBW lines of an export payload dict, blank lines included.

    The lines of ``winibw_record.build_winibw_record``, the one builder every
    output uses (layout and unverified assumptions documented there).
    - Claude Generated
    """
    return list(build_winibw_record(state)["lines"])


# ============================================================
# Exporters
# ============================================================


def export_json(
    state: Dict[str, Any],
    output_path: str,
    validate_rvk: bool = False,
) -> str:
    """Write the canonical export payload as pretty JSON."""
    payload = dict(state)
    payload.pop("__source_path__", None)

    if validate_rvk:
        from ..webapp.result_serialization import prepare_results_for_export

        if isinstance(payload.get("results"), dict):
            payload["results"] = prepare_results_for_export(
                payload["results"], validate_rvk=True
            )

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return output_path


def export_csv(state: Dict[str, Any], output_path: str) -> str:
    """Flat CSV: one row per keyword/classification.

    Columns: kind,value,gnd_id,system,code,source
    """
    results = _results(state)
    rows = []

    for kw in results.get("initial_keywords") or []:
        rows.append(["initial_keyword", str(kw), "", "", "", "initial"])

    for kw in results.get("final_keywords") or []:
        text = str(kw)
        gnd_id = ""
        if "(" in text and ")" in text:
            try:
                gnd_id = text.split("(", 1)[1].split(")")[0].strip()
                text = text.split("(", 1)[0].strip()
            except IndexError:
                pass
        rows.append(["gnd_keyword", text, gnd_id, "", "", "final"])

    for entry in results.get("classifications") or []:
        if isinstance(entry, dict):
            rows.append([
                "classification",
                entry.get("display", "") or "",
                "",
                entry.get("system", "") or "",
                entry.get("code", "") or "",
                entry.get("validation_source", "") or "",
            ])
        else:
            system, code = _split_classification(entry)
            rows.append(["classification", str(entry), "", system, code, ""])

    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["kind", "value", "gnd_id", "system", "code", "source"])
        writer.writerows(rows)
    return output_path


def _tex_escape(text: str) -> str:
    """Minimal TeX escaping for body text."""
    if text is None:
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
    out = str(text)
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def export_tex(state: Dict[str, Any], output_path: str) -> str:
    """Minimal stand-alone LaTeX snippet.

    Not a full report (see report_renderer for that). Useful for
    quick inclusion via \\input{...}.
    """
    results = _results(state)
    working_title = results.get("working_title") or _input(state).get("text_preview") or "ALIMA-Analyse"
    abstract = results.get("original_abstract") or ""

    final_kws = [_strip_gnd_id(str(k)) for k in (results.get("final_keywords") or [])]
    initial_kws = [str(k) for k in (results.get("initial_keywords") or [])]
    classifications = []
    for entry in results.get("classifications") or []:
        if isinstance(entry, dict):
            classifications.append(entry.get("display") or entry.get("code") or "")
        else:
            classifications.append(str(entry))

    buf = io.StringIO()
    buf.write("% ALIMA Export — generated\n")
    buf.write(f"\\section*{{{_tex_escape(working_title)}}}\n\n")

    if abstract:
        buf.write("\\subsection*{Abstract}\n")
        buf.write(_tex_escape(abstract))
        buf.write("\n\n")

    if initial_kws:
        buf.write("\\subsection*{Ursprüngliche Schlagwörter}\n")
        buf.write("\\begin{itemize}\n")
        for kw in initial_kws:
            buf.write(f"  \\item {_tex_escape(kw)}\n")
        buf.write("\\end{itemize}\n\n")

    if final_kws:
        buf.write("\\subsection*{GND-Schlagwörter}\n")
        buf.write("\\begin{itemize}\n")
        for kw in final_kws:
            buf.write(f"  \\item {_tex_escape(kw)}\n")
        buf.write("\\end{itemize}\n\n")

    if classifications:
        buf.write("\\subsection*{Klassifikationen}\n")
        buf.write("\\begin{itemize}\n")
        for cls in classifications:
            buf.write(f"  \\item {_tex_escape(cls)}\n")
        buf.write("\\end{itemize}\n\n")

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(buf.getvalue())
    return output_path


def export_marc(state: Dict[str, Any], output_path: str) -> str:
    """K10+/WinIBW catalog export: the WinIBW entry as plain text, one line per field line."""
    lines = generate_k10plus_lines(state)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        if lines:
            fh.write("\n")
    return output_path


# ============================================================
# Dispatch
# ============================================================


EXPORTERS = {
    "json": export_json,
    "csv": export_csv,
    "tex": export_tex,
    "marc": export_marc,
}


def export(
    state: Dict[str, Any],
    format: str,
    output_path: str,
    **kwargs: Any,
) -> str:
    """Dispatch to format-specific exporter."""
    fmt = format.lower().strip()
    if fmt not in EXPORTERS:
        raise ValueError(
            f"Unknown format '{format}'. Supported: {sorted(EXPORTERS.keys())}"
        )
    writer = EXPORTERS[fmt]
    if fmt == "json":
        return writer(state, output_path, **kwargs)
    return writer(state, output_path)


def default_output_path(
    state: Dict[str, Any],
    format: str,
    output_dir: Optional[str] = None,
) -> str:
    """Build a default output path from working_title + format."""
    results = _results(state)
    base = (
        results.get("working_title")
        or _input(state).get("text_preview")
        or "alima_export"
    )
    base = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(base))[:80]
    base = base.strip("._-") or "alima_export"
    ext = {"json": ".json", "csv": ".csv", "tex": ".tex", "marc": ".txt"}.get(
        format.lower(), f".{format}"
    )
    out_dir = output_dir or os.getcwd()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return os.path.join(out_dir, f"{base}{ext}")
