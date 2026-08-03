"""PipelineResultFormatter — display/serialisation of pipeline results - Claude Generated.

Formats GND search hits, DK/RVK classifications and keyword lists for the GUI,
webapp and CLI. Split out of the former ``pipeline_utils`` god-module; still
re-exported from ``pipeline_utils``. Depends only on the leaf text helpers
(one-way), never on PipelineStepExecutor.
"""

import html
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .pipeline_text_utils import repair_display_text, flatten_keyword_centric_results

logger = logging.getLogger(__name__)

class PipelineResultFormatter:
    """Format pipeline results for display - Claude Generated"""

    @staticmethod
    def format_search_results_for_display(
        search_results: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """Format search results as list of strings for display - Claude Generated"""
        formatted_results = []

        for search_term, results in search_results.items():
            for keyword, data in results.items():
                gnd_ids = data.get("gnd_ids", set())
                for gnd_id in gnd_ids:
                    formatted_results.append(f"{keyword} (GND: {gnd_id})")

        return formatted_results

    @staticmethod
    def format_keywords_for_prompt(search_results: Dict[str, Dict[str, Any]]) -> str:
        """Format search results as text for LLM prompt - Claude Generated"""
        search_results_text = ""

        for search_term, results in search_results.items():
            search_results_text += f"Search Term: {search_term}\n"
            for keyword, data in results.items():
                gnd_ids = ", ".join(data.get("gnd_ids", [])) if data.get("gnd_ids") else ""
                formatted_keyword = f"{keyword} ({gnd_ids})" if gnd_ids else keyword
                search_results_text += f"  - {formatted_keyword}\n"

        return search_results_text

    @staticmethod
    def _filter_placeholder_titles(titles: List[str]) -> List[str]:
        """Filter out placeholder titles from classification cache - Claude Generated"""
        filtered = []
        for title in titles:
            title = repair_display_text(title)
            # Skip placeholder titles that should not be shown to users or LLM
            if title.startswith("Cached Catalog Entry for RSN"):
                continue
            if title == "Cached Author":
                continue
            if not title.strip():  # Skip empty strings
                continue
            filtered.append(title)
        return filtered

    @staticmethod
    def format_notation_results_for_prompt(dk_results: List[Dict[str, Any]], max_results: int = 60) -> str:
        """Format DK/RVK results for LLM prompt or UI display - Claude Generated
        max_results caps total entries to prevent context-length overflow."""
        catalog_results = []
        for result in dk_results[:max_results]:
            # Handle keyword-centric format (fallback) - Claude Generated
            if "keyword" in result and "classifications" in result:
                classifications = result.get("classifications", [])
                for cl in classifications:
                    dk_code = cl.get("dk", "")
                    titles = cl.get("titles", [])
                    classification_type = cl.get("classification_type", "DK")

                    if dk_code:
                        # Filter placeholder titles - Claude Generated
                        filtered_titles = PipelineResultFormatter._filter_placeholder_titles(titles)
                        # Only show titles if available - Claude Generated
                        entry = f"{classification_type}: {dk_code}"
                        if filtered_titles:
                            title_text = " | ".join(filtered_titles[:5]) # Limit to 5 titles for keyword-centric
                            entry += f"\nBeispieltitel: {title_text}"
                        catalog_results.append(entry)
                continue

            # Handle aggregated format from _aggregate_dk_results
            if "dk" in result and "count" in result and "titles" in result:
                # Aggregated format with count and titles
                dk_code = result.get("dk", "")
                count = result.get("count", 0)
                titles = result.get("titles", [])
                matched_keywords = result.get("matched_keywords", [])
                classification_type = result.get("classification_type", "DK")
                source = result.get("source", "")
                label = result.get("label", "")
                ancestor_path = result.get("ancestor_path", "")
                register = result.get("register", [])
                rvk_validation_status = result.get("rvk_validation_status", "")
                validation_message = result.get("validation_message", "")

                if dk_code:
                    keyword_text = ", ".join(
                        cleaned
                        for cleaned in (repair_display_text(item) for item in matched_keywords)
                        if cleaned
                    ) if matched_keywords else "keine"
                    label = repair_display_text(label)
                    ancestor_path = repair_display_text(ancestor_path)
                    register = [cleaned for cleaned in (repair_display_text(item) for item in register) if cleaned]
                    validation_message = repair_display_text(validation_message)
                    if classification_type == "RVK" and rvk_validation_status == "standard":
                        if source == "rvk_api":
                            source_text = "RVK API (autoritaetsbasiert)"
                        elif source == "rvk_gnd_index":
                            source_text = "RVK MarcXML-GND-Index (autoritaetsbasiert)"
                        else:
                            source_text = "Katalog (RVK-API-validiert)"
                        entry = f"{classification_type}: {dk_code}\nKeywords: {keyword_text}\nQuelle: {source_text}"
                        if label:
                            entry += f"\nBenennung: {label}"
                        if ancestor_path:
                            entry += f"\nFachpfad: {ancestor_path}"
                        if register:
                            entry += f"\nRegister: {', '.join(map(str, register[:6]))}"
                    elif source == "rvk_api":
                        entry = f"{classification_type}: {dk_code}\nKeywords: {keyword_text}\nQuelle: RVK API (autoritaetsbasiert)"
                        if label:
                            entry += f"\nBenennung: {label}"
                        if ancestor_path:
                            entry += f"\nFachpfad: {ancestor_path}"
                        if register:
                            entry += f"\nRegister: {', '.join(map(str, register[:6]))}"
                    elif classification_type == "RVK" and rvk_validation_status == "non_standard":
                        entry = f"{classification_type}: {dk_code} (Häufigkeit: {count})\nKeywords: {keyword_text}\nQuelle: Katalog (nicht-standardisiert/lokal)"
                        if validation_message:
                            entry += f"\nHinweis: {validation_message}"
                    elif classification_type == "RVK" and rvk_validation_status == "validation_error":
                        entry = f"{classification_type}: {dk_code} (Häufigkeit: {count})\nKeywords: {keyword_text}\nQuelle: Katalog (RVK-Validierung fehlgeschlagen)"
                        if validation_message:
                            entry += f"\nHinweis: {validation_message}"
                    else:
                        entry = f"{classification_type}: {dk_code} (Häufigkeit: {count})\nKeywords: {keyword_text}"
                    # Filter placeholder titles, cap at 5 to control prompt length - Claude Generated
                    filtered_titles = PipelineResultFormatter._filter_placeholder_titles(titles)
                    if filtered_titles:
                        title_text = " | ".join(filtered_titles[:5])
                        entry += f"\nBeispieltitel: {title_text}"
                    catalog_results.append(entry)
            elif "source_title" in result and "dk" in result:
                # Individual result format
                dk_code = result.get("dk", "")
                title = repair_display_text(result.get("source_title", ""))
                classification_type = result.get("classification_type", "DK")

                if dk_code and title:
                    entry = f"{classification_type}: {dk_code} | Titel: {title}"
                    catalog_results.append(entry)
            else:
                # Legacy format support
                title = repair_display_text(result.get("title", ""))
                subjects = [cleaned for cleaned in (repair_display_text(item) for item in result.get("subjects", [])) if cleaned]
                dk_class = result.get("dk", [])
                rvk_class = result.get("rvk", [])

                if title:
                    entry = f"Titel: {title}"
                    if subjects:
                        entry += f" | Schlagworte: {', '.join(subjects)}"

                    # Handle DK
                    if isinstance(dk_class, str):
                        entry += f" | DK: {dk_class}"
                    elif isinstance(dk_class, list):
                        valid_dk = [dk for dk in dk_class if len(str(dk)) > 1]
                        if valid_dk:
                            entry += f" | DK: {', '.join(map(str, valid_dk))}"

                    # Handle RVK
                    if isinstance(rvk_class, str):
                        entry += f" | RVK: {rvk_class}"
                    elif isinstance(rvk_class, list):
                        entry += f" | RVK: {', '.join(map(str, rvk_class))}"

                    catalog_results.append(entry)

        return "\n---\n".join(catalog_results)

    # ------------------------------------------------------------------
    # Shared display formatters (single source of truth for Pipeline-Tab
    # and Agentic-Chat). Moved here so both GUI surfaces render identical
    # catalog-research and final-notation output. - Claude Generated
    # ------------------------------------------------------------------

    @staticmethod
    def split_classification_code(classification: str) -> Tuple[str, str]:
        """Split a prefixed classification string into (system, code) - Claude Generated

        Delegates to the shared classification-system registry so DK/DDC/RVK are
        recognised uniformly. ``"DK 666.76"`` -> ``("DK", "666.76")``;
        ``"DDC 530.1"`` -> ``("DDC", "530.1")``; unprefixed -> ``("", value)``.
        """
        from .classification_systems import split_classification_code as _split
        return _split(classification)

    @staticmethod
    def get_titles_for_notation_code(
        dk_code: str, dk_search_results: List[Dict[str, Any]]
    ) -> Tuple[List[str], int]:
        """Return ``(titles[:50], total_count)`` for a classification code - Claude Generated

        Matches the flattened catalog-search structure (``{dk, classification_type,
        titles, ...}``). The type prefix (DK/RVK) is honoured when present so a DK
        code does not pick up an RVK entry with the same notation.
        """
        if not dk_search_results:
            return ([], 0)

        # Keyword-centric input (nested classifications) → flatten first,
        # same tolerance as format_dk_search_results_text - Claude Generated
        if any(
            isinstance(r, dict) and "classifications" in r and not r.get("dk")
            for r in dk_search_results
        ):
            dk_search_results = flatten_keyword_centric_results(dk_search_results)

        expected_type, normalized_code = PipelineResultFormatter.split_classification_code(
            dk_code
        )

        for result in dk_search_results:
            result_code = str(result.get("dk", "")).strip()
            result_type = str(result.get("classification_type", "")).strip().upper()
            if result_code == normalized_code and (
                not expected_type or result_type == expected_type
            ):
                titles = result.get("titles", [])
                return (titles[:50], len(titles))

        return ([], 0)

    @staticmethod
    def flatten_gnd_hits(search_results: Any) -> List[Dict[str, Any]]:
        """Flatten GND search results into deduplicated per-GND-ID display rows - Claude Generated

        Accepts any of the shapes the pipeline produces:
        * dict ``{search_term: {label: {gnd_ids: set|list, count, ...}}}`` (classic);
        * ``List[SearchResult]`` (``.search_term`` + ``.results`` dict);
        * a flat ``List[Dict]`` of GND entries with a top-level ``gnd_id`` (agentic
          snapshots' ``gnd_entries``).

        Returns rows ``[{"begriff", "gnd_id", "count", "search_terms": [..]}]``
        deduplicated by GND-ID (counts maxed, search terms merged), sorted by
        descending count then label.
        """
        by_id: Dict[str, Dict[str, Any]] = {}

        def _add(label: str, gnd_id: str, count: Any, term: str,
                 display_count: Any = None) -> None:
            gnd_id = str(gnd_id or "").strip()
            if not gnd_id:
                return
            cnt = int(count) if isinstance(count, (int, float)) else 0
            # F-4: prefer the display-only count (real Häufigkeit restored from the
            # mapping cache) over the pool count, which is 1 for cache hits. This is
            # display-only — it never feeds ranking (see gnd_search_core landmine).
            dc = int(display_count) if isinstance(display_count, (int, float)) else 0
            cnt = max(cnt, dc)
            row = by_id.get(gnd_id)
            if row is None:
                by_id[gnd_id] = {
                    "begriff": repair_display_text(label) or gnd_id,
                    "gnd_id": gnd_id,
                    "count": cnt,
                    "search_terms": [term] if term else [],
                }
            else:
                row["count"] = max(row["count"], cnt)
                if term and term not in row["search_terms"]:
                    row["search_terms"].append(term)

        # --- flat list of GND entries (agentic gnd_entries) ---
        if (
            isinstance(search_results, list)
            and search_results
            and isinstance(search_results[0], dict)
            and "gnd_id" in search_results[0]
        ):
            for entry in search_results:
                if not isinstance(entry, dict):
                    continue
                label = entry.get("keyword") or entry.get("title", "")
                _add(label, entry.get("gnd_id", ""), entry.get("count", 0),
                     entry.get("search_term", ""), entry.get("display_count"))
        else:
            # --- dict form or List[SearchResult] ---
            items: List[Tuple[str, Any]] = []
            if isinstance(search_results, dict):
                items = list(search_results.items())
            elif isinstance(search_results, list):
                for sr in search_results:
                    if isinstance(sr, dict):
                        items.append((sr.get("search_term", ""), sr.get("results", {})))
                    else:
                        items.append(
                            (getattr(sr, "search_term", "") or "",
                             getattr(sr, "results", {}) or {})
                        )

            for term, results in items:
                if not isinstance(results, dict):
                    continue
                for label, data in results.items():
                    if not isinstance(data, dict):
                        continue
                    count = data.get("count", 0)
                    display_count = data.get("display_count")
                    for gnd_id in (data.get("gnd_ids", []) or []):
                        _add(label, gnd_id, count, term, display_count)

        rows = list(by_id.values())
        rows.sort(key=lambda r: (-r["count"], r["begriff"].lower()))
        return rows

    @staticmethod
    def extract_selected_gnd_keys(selected: Any) -> Tuple[Set[str], Set[str]]:
        """Derive ``(gnd_id set, normalized-label set)`` from a final keyword list - Claude Generated

        Handles the inconsistent representations of final/selected keywords:
        dicts with ``gnd_id``/``title``, or strings like ``"Halbleiter (GND-ID: 4129772-7)"``.
        Used to mark which GND hits survived the chunking/selection step.
        """
        import re

        ids: Set[str] = set()
        labels: Set[str] = set()
        for item in (selected or []):
            if isinstance(item, dict):
                gid = str(item.get("gnd_id", "") or "")
                if gid:
                    ids.add(gid)
                text = str(item.get("title", item.get("keyword", "")))
            else:
                text = str(item)
            # GND-IDs embedded in "(GND-ID: x)" / "(GND: x)"
            for match in re.findall(r"GND[^:)]*:\s*([0-9Xx][0-9Xx\-/]*)", text):
                ids.add(match)
            # Normalised label = text without the "(GND…)" suffix
            label = re.sub(r"\s*\(GND[^)]*\)", "", text).strip().lower()
            if label:
                labels.add(label)
        return ids, labels

    @staticmethod
    def select_dk_title_source(
        dk_search_results: Optional[List[Dict[str, Any]]],
        dk_search_results_flattened: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Pick the DK-centric result list that actually carries catalog titles - Claude Generated

        Both pipelines now follow the ``data_models`` contract: the rich
        DK-centric list (per-DK-code, real catalog titles + counts) lives in
        ``dk_search_results_flattened`` and ``dk_search_results`` stays
        keyword-centric. This picker remains defensive — it flattens a
        keyword-centric candidate and returns whichever has the most catalog
        titles — so it also handles legacy exports from pre-convergence agentic
        runs that stored the rich list under ``dk_search_results``.
        """
        def _prep(lst: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
            # A keyword-centric source ({keyword, classifications:[...]}, no
            # top-level dk) hides its catalog titles one level down. The agentic
            # pipeline stores dk_search_results in exactly that shape, so without
            # flattening it scored 0 here and the card fell back to the
            # title-less flattened source — i.e. titles vanished for agentic
            # runs (pipeline-mode dependent, not LLM dependent). Flatten first so
            # the title score and downstream lookup see the real titles. - Claude Generated
            lst = lst or []
            if any(isinstance(r, dict) and "classifications" in r and not r.get("dk") for r in lst):
                return flatten_keyword_centric_results(lst)
            return lst

        candidates = [
            _prep(dk_search_results),
            _prep(dk_search_results_flattened),
        ]

        def _title_score(lst: List[Dict[str, Any]]) -> int:
            return sum(
                len(item.get("titles", []))
                for item in lst
                if isinstance(item, dict) and item.get("dk")
            )

        best = max(candidates, key=_title_score)
        if _title_score(best) > 0:
            return best
        # No titles anywhere — return the first DK-keyed list so codes still resolve.
        for lst in candidates:
            if any(isinstance(item, dict) and item.get("dk") for item in lst):
                return lst
        return best

    @staticmethod
    def format_dk_search_results_text(results: List[Dict[str, Any]]) -> str:
        """Format flattened DK/RVK search results as per-code plain text - Claude Generated

        Mirrors the Pipeline-Tab catalog-research view: one block per DK/RVK code
        with sample titles and frequency. Returns an empty string when nothing has
        titles/count (caller decides on the empty-state placeholder).

        Accepts BOTH formats: DK-centric ({dk, titles, count, ...}) and
        keyword-centric ({keyword, classifications: [...]}) as produced by
        dk_collect — the latter previously fell through silently and the
        Katalog-Recherche view showed nothing.
        """
        if not results:
            return ""

        # Keyword-centric entries (nested classifications, no top-level dk)
        # → flatten with the same dedup logic the pipeline uses
        if any(
            isinstance(r, dict) and "classifications" in r and not r.get("dk")
            for r in results
        ):
            results = flatten_keyword_centric_results(results)

        result_lines = []
        for result in results:
            dk_code = result.get("dk", "")
            count = result.get("count", 0)
            titles = result.get("titles", [])
            # Flattened entries carry "matched_keywords", legacy ones "keywords"
            keywords = result.get("keywords") or result.get("matched_keywords") or []
            classification_type = result.get("classification_type", "DK")

            if not titles or count == 0:
                continue

            sample_titles = titles[:3]
            titles_text = " | ".join(sample_titles)
            if len(titles) > 3:
                titles_text += f" | ... (und {len(titles) - 3} weitere)"

            result_lines.append(
                f"{classification_type}: {dk_code} (Häufigkeit: {count})\n"
                f"Beispieltitel: {titles_text}\n"
                f"Keywords: {', '.join(keywords)}\n"
            )

        return "\n".join(result_lines)

    @staticmethod
    def format_dk_classifications_html(
        dk_classifications: List[str],
        dk_search_results: List[Dict[str, Any]],
        max_titles_per_code: int = 5,
    ) -> str:
        """Format final DK/RVK notations with catalog titles as an HTML fragment - Claude Generated

        Returns a self-contained HTML fragment (no ``<html>/<body>`` wrapper) so it
        renders identically via ``QTextEdit.setHtml`` (Pipeline-Tab) and
        ``QTextCursor.insertHtml`` (Agentic-Chat). Confidence is colour-coded by the
        number of catalog hits.
        """
        if not dk_classifications:
            return "Keine DK/RVK-Klassifikationen generiert"

        html_parts: List[str] = []
        for idx, dk_code in enumerate(dk_classifications, 1):
            titles, total_count = PipelineResultFormatter.get_titles_for_notation_code(
                dk_code, dk_search_results
            )

            # Color-coding based on frequency (confidence)
            if total_count > 50:
                color, bg_color = "#2d5016", "#d4edda"  # Dark green
            elif total_count > 20:
                color, bg_color = "#0c5460", "#d1ecf1"  # Teal
            else:
                color, bg_color = "#664d03", "#fff3cd"  # Brown/Orange

            html_parts.append(
                f"<div style='background-color: {bg_color}; padding: 12px; margin-bottom: 8px; "
                f"border-left: 4px solid {color}; border-radius: 4px;'>"
                f"<h2 style='color: {color}; margin: 0; font-size: 14pt;'>#{idx} {dk_code}</h2>"
            )

            if total_count > 0:
                confidence_bar = "🟩" * min(5, (total_count // 10) + 1)
                html_parts.append(
                    f"<p style='color: {color}; font-weight: bold; margin: 5px 0 2px 0;'>"
                    f"{confidence_bar} {total_count} Katalog-Treffer</p>"
                    f"<p style='color: {color}; font-size: 9pt; opacity: 0.8; margin: 0 0 10px 0;'>"
                    f"📚 Diese Klassifikation wurde in {total_count} Titel{'n' if total_count != 1 else ''} gefunden.</p>"
                )
            html_parts.append("</div>")

            if titles:
                html_parts.append("<ol style='font-size: 9pt; padding-left: 30px;'>")
                for title in titles[:max_titles_per_code]:
                    safe_title = (
                        title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    )
                    html_parts.append(f"<li>{safe_title}</li>")
                html_parts.append("</ol>")

                if total_count > max_titles_per_code:
                    html_parts.append(
                        f"<p style='color: #888; font-style: italic; padding-left: 20px;'>"
                        f"... und {total_count - max_titles_per_code} weitere Titel</p>"
                    )

        return "".join(html_parts)

    @staticmethod
    def _escape_card_html(text: str) -> str:
        """Escape & < > " for trusted-card HTML (matches the GUI renderer)."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    @staticmethod
    def format_dk_search_card_html(output_data: Dict[str, Any]) -> Tuple[str, str]:
        """Build the per-DK-code catalog-research card (HTML, plain_text). - Claude Generated

        Shared producer for the GUI agentic chat (``PipelineChatPanel``) and the
        webapp (WP12). Returns ``("", "")`` when there is nothing to show so the
        caller can skip emitting an empty block.
        """
        flattened = output_data.get(
            "dk_search_results_flattened", output_data.get("dk_search_results", [])
        )
        text = PipelineResultFormatter.format_dk_search_results_text(flattened)
        if not text.strip():
            return "", ""
        body = PipelineResultFormatter._escape_card_html(text).replace("\n", "<br>")
        html = (
            "<div style='font-family: monospace; font-size: 9pt; color: #a8a8a8; "
            "white-space: pre-wrap; margin: 4px 0 4px 8px;'>"
            "<span style='color: #8be9fd;'>📚 Katalog-Recherche (DK/RVK):</span><br>"
            f"{body}</div>"
        )
        return html, text

    @staticmethod
    def _infer_classification_system(display: str) -> str:
        """Best-effort DK/RVK system inference from an unprefixed notation.

        DK is purely numeric/dotted (``614.7``); RVK starts with letters
        (``WD 5000``, ``AK 54000``). Falls back to ``DK`` for digit starts and
        ``RVK`` for letter starts. - Claude Generated
        """
        code = (display or "").strip()
        if not code:
            return "UNKNOWN"
        return "RVK" if code[0].isalpha() else "DK"

    @staticmethod
    def normalize_classifications(
        dk_classifications: Any,
        dk_search_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Normalise raw classifications to badge-card entries. - Claude Generated

        Accepts ``List[str]`` or ``List[Dict]`` (the latter may carry
        ``system`` / ``validation_status`` / ``label`` / ``validation_message``
        from the RVK-validation enrichment). System is inferred when absent;
        catalog titles + total hit count are attached when ``dk_search_results``
        is given. Shared by the GUI and webapp badge card (WP12).
        """
        entries: List[Dict[str, Any]] = []
        for item in dk_classifications or []:
            if isinstance(item, dict):
                display = (
                    str(item.get("display") or "").strip()
                    or f"{item.get('system', '')} {item.get('code', '')}".strip()
                    or str(item.get("code") or "").strip()
                )
                system = str(item.get("system") or "").strip().upper()
                validation_status = item.get("validation_status")
                label = item.get("label")
                validation_message = item.get("validation_message")
            else:
                display = str(item or "").strip()
                system = ""
                validation_status = label = validation_message = None
            if not display:
                continue
            # Honour an inline "DK "/"RVK " prefix; else infer.
            prefix_system, _code = PipelineResultFormatter.split_classification_code(display)
            if not system:
                system = prefix_system or PipelineResultFormatter._infer_classification_system(display)
            titles, total_count = PipelineResultFormatter.get_titles_for_notation_code(
                display, dk_search_results or []
            )
            entries.append({
                "system": system or "UNKNOWN",
                "display": display,
                "validation_status": validation_status,
                "label": label,
                "validation_message": validation_message,
                "titles": titles,
                "total_count": total_count,
            })
        return entries

    @staticmethod
    def format_classification_badge_card_html(
        entries: List[Dict[str, Any]], max_titles_per_code: int = 3
    ) -> str:
        """Render normalised classifications as the shared badge card. - Claude Generated

        One HTML fragment (no ``<html>/<body>``) using the ``classification-*``
        classes styled in the shared ``alima_render.css`` (scoped under
        ``#log``). Ported from the webapp's structured cards so the GUI's
        QWebEngineView log and the webapp render identical chrome (WP12). Empty
        ``entries`` → ``""``.
        """
        if not entries:
            return ""
        esc = PipelineResultFormatter._escape_card_html

        rvk = [e for e in entries if e["system"] == "RVK"]
        std = sum(1 for e in rvk if e["validation_status"] == "standard")
        nonstd = sum(1 for e in rvk if e["validation_status"] == "non_standard")
        err = sum(1 for e in rvk if e["validation_status"] == "validation_error")
        summary = ""
        if std or nonstd or err:
            parts = [
                f'<span class="classification-badge classification-badge--standard">RVK standard: {std}</span>',
                f'<span class="classification-badge classification-badge--non-standard">RVK nicht standard: {nonstd}</span>',
            ]
            if err:
                parts.append(
                    f'<span class="classification-badge classification-badge--unknown">API-Fehler: {err}</span>'
                )
            summary = f'<div class="classification-validation-summary">{"".join(parts)}</div>'

        rows: List[str] = []
        for e in entries:
            system = e["system"]
            sys_class = {
                "RVK": "classification-badge--rvk",
                "DDC": "classification-badge--ddc",
            }.get(system, "classification-badge--dk")
            head = [
                f'<span class="classification-badge {sys_class}">{esc(system)}</span>',
                f'<span class="classification-entry__code">{esc(e["display"])}</span>',
            ]
            vs = e.get("validation_status")
            if system == "RVK" and vs == "standard":
                head.append('<span class="classification-badge classification-badge--standard">standard</span>')
            elif system == "RVK" and vs == "non_standard":
                head.append('<span class="classification-badge classification-badge--non-standard">nicht standard</span>')
            elif system == "RVK" and vs == "validation_error":
                head.append('<span class="classification-badge classification-badge--unknown">API-Fehler</span>')
            total = e.get("total_count") or 0
            if total > 0:
                bar = "🟩" * min(5, (total // 10) + 1)
                head.append(
                    f'<span class="classification-badge classification-badge--standard">'
                    f'{bar} {total} Treffer</span>'
                )

            meta_parts: List[str] = []
            if e.get("label"):
                meta_parts.append(esc(str(e["label"])))
            if e.get("validation_message") and vs != "standard":
                meta_parts.append(esc(str(e["validation_message"])))
            meta = (
                f'<div class="classification-entry__meta">{" · ".join(meta_parts)}</div>'
                if meta_parts else ""
            )

            titles = e.get("titles") or []
            titles_html = ""
            if titles:
                # Show a short preview, then put the rest in a collapsible
                # <details> so the full list is reachable (no hard 3-title cut).
                # Both render surfaces (GUI QWebEngineView log + webapp #log)
                # support <details>. - Claude Generated
                preview, rest = titles[:max_titles_per_code], titles[max_titles_per_code:]
                preview_lis = "".join(f"<li>{esc(str(t))}</li>" for t in preview)
                if rest:
                    rest_lis = "".join(f"<li>{esc(str(t))}</li>" for t in rest)
                    remainder = (
                        f'<li class="classification-entry__meta">… und {total - len(titles)} weitere</li>'
                        if total and total > len(titles) else ""
                    )
                    rest_block = (
                        f'<details class="classification-entry__titles-more">'
                        f'<summary style="cursor:pointer">… {len(rest)} weitere Titel anzeigen</summary>'
                        f'<ol class="classification-entry__titles" start="{len(preview) + 1}">'
                        f'{rest_lis}{remainder}</ol></details>'
                    )
                elif total and total > len(titles):
                    rest_block = (
                        f'<div class="classification-entry__meta">… und {total - len(titles)} weitere</div>'
                    )
                else:
                    rest_block = ""
                titles_html = (
                    f'<ol class="classification-entry__titles">{preview_lis}</ol>{rest_block}'
                )

            rows.append(
                f'<div class="classification-entry">'
                f'<div class="classification-entry__head">{"".join(head)}</div>'
                f"{meta}{titles_html}</div>"
            )

        return (
            '<div class="classification-card-title">🏷 DK/RVK-Klassifikationen</div>'
            f'{summary}'
            f'<div class="classification-entry-list">{"".join(rows)}</div>'
        )

    @staticmethod
    def format_dk_classifications_card_html(analysis_state: Any) -> Tuple[str, str]:
        """Build the final DK-classifications card (HTML, plain_text). - Claude Generated

        Shared producer for the GUI agentic-chat log and the webapp ``#log``
        (WP12). Normalises ``analysis_state.dk_classifications`` (``List[str]`` or
        ``List[Dict]``), attaches catalog titles from the title-carrying source
        (agentic vs classic), and renders the structured **badge card** (ported
        from the webapp). The Pipeline-Tab keeps its own
        ``format_dk_classifications_html`` confidence card. Returns ``("", "")``
        when no classifications exist.
        """
        dk_classifications = getattr(analysis_state, "dk_classifications", None)
        if not dk_classifications:
            return "", ""
        flat = PipelineResultFormatter.select_dk_title_source(
            getattr(analysis_state, "dk_search_results", None),
            getattr(analysis_state, "dk_search_results_flattened", None),
        )
        entries = PipelineResultFormatter.normalize_classifications(dk_classifications, flat)
        html = PipelineResultFormatter.format_classification_badge_card_html(entries)
        return html, ", ".join(e["display"] for e in entries)

    @staticmethod
    def _confidence_bucket(count: int) -> Tuple[str, str]:
        """Confidence label + emoji bar for a hit/title count. - Claude Generated

        Mirrors the thresholds of ``src/ui/styles.get_confidence_style`` inline so
        this shared formatter stays Qt-free (no ``get_colors``/Qt import on the
        webapp/CLI path).
        """
        if count > 50:
            return "Sehr hoch", "\U0001f7e9" * 5
        if count > 20:
            return "Hoch", "\U0001f7e9" * 3
        if count > 5:
            return "Mittel", "\U0001f7e9" * 2
        return "Niedrig", "\U0001f7e9"

    @staticmethod
    def format_dk_auswertung_card_html(analysis_state: Any) -> Tuple[str, str]:
        """Build the DK/RVK frequency-Auswertung + RVK-provenance card. - Claude Generated

        Reintroduces the classic-pipeline RVK analytics ("Auswertung") into the
        shared WP12 render layer. Two stacked tables in one HTML fragment (no
        ``<html>/<body>``), styled by the ``#log``-scoped ``.alima-auswertung``
        rules in ``alima_render.css`` so the GUI QWebEngineView log and the
        webapp ``#log`` render identical chrome:

        * **Auswertung** — top classifications from
          ``analysis_state.dk_statistics["most_frequent"]`` plus a deduplication
          headline and keyword coverage.
        * **RVK-Provenienz** — source breakdown from
          ``analysis_state.rvk_provenance``.

        Returns ``("", "")`` when neither dataset is present (e.g. agentic
        workflows that do not populate ``dk_statistics``).
        """
        stats = getattr(analysis_state, "dk_statistics", None) or {}
        provenance = getattr(analysis_state, "rvk_provenance", None) or {}
        most_frequent = stats.get("most_frequent") or []

        prov_labels = [
            ("catalog_standard", "Katalog (standard)"),
            ("catalog_nonstandard", "Katalog (lokal)"),
            ("rvk_gnd_index", "RVK-GND-Index"),
            ("rvk_api", "RVK-API-Label"),
        ]
        prov_rows = [
            (label, int(provenance.get(key) or 0))
            for key, label in prov_labels
            if int(provenance.get(key) or 0) > 0
        ]

        if not most_frequent and not prov_rows:
            return "", ""

        esc = PipelineResultFormatter._escape_card_html
        sections: List[str] = []
        plain_parts: List[str] = []

        # --- Table A: frequency Auswertung ---
        if most_frequent:
            dedup = stats.get("deduplication_stats") or {}
            total = stats.get("total_classifications")
            original = dedup.get("original_count")
            head_bits: List[str] = []
            if original is not None and total is not None:
                head_bits.append(f"{original} → {total} nach Deduplizierung")
            if dedup.get("deduplication_rate"):
                head_bits.append(f"Dedup-Rate {dedup['deduplication_rate']}")
            if dedup.get("estimated_token_savings"):
                head_bits.append(f"~{dedup['estimated_token_savings']} Token gespart")
            headline = (
                f'<div class="alima-auswertung__sub">{esc(" · ".join(head_bits))}</div>'
                if head_bits
                else ""
            )

            rows: List[str] = []
            for item in most_frequent:
                code = esc(str(item.get("dk", "")))
                ctype = esc(str(item.get("type", "DK")))
                count = int(item.get("count", 0) or 0)
                kws = item.get("keywords") or []
                kw_preview = ", ".join(esc(str(k)) for k in kws[:3])
                if len(kws) > 3:
                    kw_preview += f" (+{len(kws) - 3})"
                conf_count = int(item.get("unique_titles", count) or 0)
                conf_label, bar = PipelineResultFormatter._confidence_bucket(conf_count)
                rows.append(
                    "<tr>"
                    f'<td class="alima-auswertung__code">{code}</td>'
                    f"<td>{ctype}</td>"
                    f'<td class="alima-auswertung__num">{count}</td>'
                    f"<td>{kw_preview}</td>"
                    f'<td class="alima-auswertung__conf">{bar} {conf_label}</td>'
                    "</tr>"
                )
            table_a = (
                '<table class="alima-auswertung-table">'
                "<thead><tr>"
                "<th>Code</th><th>Typ</th><th>Count</th>"
                "<th>Keywords</th><th>Konfidenz</th>"
                "</tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table>"
            )

            coverage = stats.get("keyword_coverage") or {}
            coverage_html = ""
            if coverage:
                cov_bits = [
                    f"{esc(str(kw))} → {esc(', '.join(str(c) for c in (codes or [])[:3]))}"
                    for kw, codes in list(coverage.items())[:8]
                ]
                coverage_html = (
                    '<div class="alima-auswertung__sub">'
                    f'Keyword-Coverage: {" · ".join(cov_bits)}'
                    "</div>"
                )

            sections.append(
                '<div class="classification-card-title">📊 DK/RVK-Auswertung</div>'
                f"{headline}{table_a}{coverage_html}"
            )
            plain_parts.append(
                "DK/RVK-Auswertung: "
                + "; ".join(
                    f"{i.get('dk', '')} ({i.get('count', 0)})" for i in most_frequent
                )
            )

        # --- Table B: RVK provenance ---
        if prov_rows:
            prov_tr = "".join(
                f"<tr><td>{esc(label)}</td>"
                f'<td class="alima-auswertung__num">{count}</td></tr>'
                for label, count in prov_rows
            )
            sections.append(
                '<div class="classification-card-title">🧭 RVK-Provenienz</div>'
                '<table class="alima-auswertung-table alima-auswertung-table--prov">'
                "<thead><tr><th>Quelle</th><th>Anzahl</th></tr></thead>"
                f"<tbody>{prov_tr}</tbody></table>"
            )
            plain_parts.append(
                "RVK-Provenienz: "
                + ", ".join(f"{label} {count}" for label, count in prov_rows)
            )

        html = f'<div class="alima-auswertung">{"".join(sections)}</div>'
        return html, " | ".join(plain_parts)

    @staticmethod
    def parse_notation_results_from_text(text: str) -> List[Dict[str, Any]]:
        """Parse DK/RVK results back from formatted text into dictionary format - Claude Generated"""
        import re
        import logging
        logger = logging.getLogger(__name__)
        results = []

        if not text:
            return results

        # Regex for the main entry line: "[Type]: [DK Code] (Häufigkeit: [Count])"
        # We make it more flexible: optional spaces, optional prefix
        entry_pattern = r'(?:^|\n)\s*(DK|RVK):\s*([A-Z0-9.\-\s/]+?)\s*(?:\(Häufigkeit:\s*(\d+)\))'

        lines = text.split('\n')
        current_entry = None

        logger.debug(f"Parsing DK results from text ({len(text)} chars)")

        # First attempt: Try to parse the structured format with frequencies and titles
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this is a new entry line
            match = re.search(entry_pattern, line)
            if match:
                if current_entry:
                    results.append(current_entry)

                current_entry = {
                    "classification_type": match.group(1),
                    "dk": match.group(2).strip(),
                    "count": int(match.group(3)) if match.group(3) else 1,
                    "matched_keywords": [],
                    "titles": []
                }
            elif current_entry:
                # Check for Keywords line
                if "Keywords:" in line:
                    kw_text = line.split("Keywords:")[1].strip()
                    if kw_text and kw_text.lower() != "keine":
                        # Split by comma, semicolon or pipe
                        current_entry["matched_keywords"] = [k.strip() for k in re.split(r'[;,|]', kw_text) if k.strip()]

                # Check for Beispieltitel line
                elif "Beispieltitel:" in line:
                    title_text = line.split("Beispieltitel:")[1].strip()
                    if title_text:
                        current_entry["titles"] = [t.strip() for t in title_text.split("|") if t.strip()]

        # Don't forget the last entry
        if current_entry:
            results.append(current_entry)

        # Fallback: If no structured entries found, try simple comma-separated DK/RVK codes
        if not results:
            logger.debug("No structured DK entries found, trying simple fallback parser")
            # Look for things like "DK 614.7", "DK: 614.7", "RVK QZ 123"
            simple_pattern = r'(DK|RVK):?\s*([A-Z0-9.\-\s/]+?)(?=[,\n;]|$)'
            matches = re.finditer(simple_pattern, text)
            for match in matches:
                code = match.group(2).strip()
                # Basic validation: code shouldn't be too long and should have some digits/uppercase
                if 1 < len(code) < 30:
                    results.append({
                        "classification_type": match.group(1),
                        "dk": code,
                        "count": 1,
                        "matched_keywords": [],
                        "titles": []
                    })

        if results:
            logger.info(f"✅ Successfully parsed {len(results)} DK entries from text context")
        else:
            logger.warning("⚠️ No DK entries could be parsed from the provided text context")

        return results

    @staticmethod
    def get_gnd_compliant_keywords(
        search_results: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """Extract GND-compliant keywords from search results - Claude Generated"""
        gnd_keywords = []

        for results in search_results.values():
            for keyword, data in results.items():
                gnd_ids = data.get("gnd_ids", set())
                for gnd_id in gnd_ids:
                    gnd_keywords.append(f"{keyword} (GND-ID: {gnd_id})")

        return gnd_keywords
