"""DK classification steps of ``PipelineStepExecutor`` - Claude Generated.

``DkStepsMixin`` carries the DK (Dezimalklassifikation) step logic — context
preparation, LLM classification, catalog DK search and result statistics —
extracted verbatim from ``pipeline_utils.PipelineStepExecutor`` (July 19, 2026;
zero call-site changes, methods reachable via MRO). RVK scoring lives in the
sibling ``_pipeline_rvk_scoring``; cross-calls go through ``self``.
"""

import logging
import re
import time
from typing import List, Tuple, Dict, Any, Optional
from ..core.data_models import AbstractData, LlmKeywordAnalysis
from .pipeline_defaults import DEFAULT_DK_MAX_RESULTS, DEFAULT_DK_FREQUENCY_THRESHOLD
from .gnd_keyword_utils import canonicalize_keyword, extract_gnd_id, canonicalize_rvk_notation, deduplicate_canonical_keywords
from .pipeline_text_utils import flatten_keyword_centric_results
from .pipeline_formatters import PipelineResultFormatter

logger = logging.getLogger(__name__)


class DkStepsMixin:
    """DK classification step methods (mixed into PipelineStepExecutor)."""

    def prepare_dk_classification_context(
        self,
        dk_search_results: List[Dict[str, Any]],
        original_abstract: str,
        dk_frequency_threshold: int = DEFAULT_DK_FREQUENCY_THRESHOLD,
        rvk_anchor_keywords: Optional[List[str]] = None,
        stream_callback=None,
        include_rvk: bool = True,
        record_priors: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Filter and format DK search results for the classification prompt - Claude Generated

        Shared by the classic pipeline (execute_dk_classification) and the
        agentic pipeline (dk_search_agentic) so both build the classification
        context identically: frequency threshold (DK only, RVK exempt),
        institution-library RVK filter, title filter, RVK candidate maps and
        RVK guardrail text prepended to the formatted catalog excerpt.

        ``include_rvk=False`` strips every RVK-typed entry up front, so the
        formatted prompt and the candidate maps stay DK/DDC only. Used by
        workflows that surface RVK out-of-band via the ``rvk_lookup`` tool
        instead of inline in ``dk_collect`` - Claude Generated.

        ``record_priors`` (WP-D1 P2) are the input record's OWN classifications
        in the canonical shape ``{SYSTEM: [{code, origin}]}``: they are surfaced
        to the LLM as a marked authority block prepended to the catalog excerpt
        (inform, never override), and prior RVK codes additionally join
        ``allowed_standard_rvk_map`` (source ``input_record``) so the guardrails
        permit selecting them. With ``record_priors`` falsy the output is
        byte-identical to before.

        Returns:
            Dict with results_with_titles, catalog_text, allowed_standard_rvk_map,
            allowed_nonstandard_rvk_map, rvk_source_map, selected_rvk_meta.
        """
        # Filter results by frequency threshold - Claude Generated
        filtered_results = []
        low_frequency_count = 0
        
        for result in dk_search_results:
            classification_type = str(result.get("classification_type", result.get("type", "DK"))).upper()
            # RVK is validated separately and exempt from the frequency filter;
            # DK and DDC are frequency-filtered below. When include_rvk is False
            # the caller handles RVK out-of-band (rvk_lookup tool) → drop it so
            # the prompt and candidate maps stay DK/DDC only. - Claude Generated
            if classification_type == "RVK":
                if include_rvk:
                    filtered_results.append(result)
                continue

            # Check if result has frequency information and meets threshold
            if "count" in result:
                count = result.get("count", 0)
                if count >= dk_frequency_threshold:
                    filtered_results.append(result)
                else:
                    low_frequency_count += 1
            else:
                # Include results without count information (legacy format)
                filtered_results.append(result)
        
        if stream_callback:
            if low_frequency_count > 0:
                stream_callback(f"Filtere Klassifikationen: {len(filtered_results)} Einträge mit ≥{dk_frequency_threshold} Vorkommen, {low_frequency_count} mit niedrigerer Häufigkeit ausgeschlossen\n", "dk_classification")
            else:
                stream_callback(f"Verwende alle {len(filtered_results)} Klassifikations-Einträge (keine Häufigkeits-Filterung nötig)\n", "dk_classification")

        # Filter out results without titles - Claude Generated
        results_with_titles = []
        titleless_count = 0
        institution_library_rvk_count = 0

        for result in filtered_results:
            classification_type = str(result.get("classification_type", result.get("type", "DK"))).upper()
            if (
                classification_type == "RVK"
                and str(result.get("source", "catalog") or "catalog") == "catalog"
                and self._is_institution_library_rvk(result)
                and not self._matches_specific_library_context(
                    result,
                    original_abstract,
                    result.get("matched_keywords", []) or result.get("keywords", []) or [],
                )
            ):
                institution_library_rvk_count += 1
                continue

            if result.get("source") in {"rvk_api", "rvk_gnd_index"}:
                if result.get("label") or result.get("ancestor_path"):
                    results_with_titles.append(result)
                else:
                    titleless_count += 1
                continue

            # Check if result has titles (aggregated format)
            if "titles" in result:
                if result.get("titles") and any(t.strip() for t in result.get("titles", [])):
                    results_with_titles.append(result)
                else:
                    titleless_count += 1
            # Check if result has source_title (individual format)
            elif "source_title" in result:
                if result.get("source_title", "").strip():
                    results_with_titles.append(result)
                else:
                    titleless_count += 1
            # Check if result has title (legacy format)
            elif "title" in result:
                if result.get("title", "").strip():
                    results_with_titles.append(result)
                else:
                    titleless_count += 1
            else:
                # No title field found - skip this result
                titleless_count += 1

        if stream_callback:
            if titleless_count > 0:
                stream_callback(f"⚠️ Filtere titel-lose Einträge: {len(results_with_titles)} mit Titeln, {titleless_count} ohne Titel ausgeschlossen\n", "dk_classification")
            else:
                stream_callback(f"✅ Alle {len(results_with_titles)} Einträge haben Titel\n", "dk_classification")
            if institution_library_rvk_count > 0:
                stream_callback(
                    f"⚠️ Verwerfe {institution_library_rvk_count} katalogseitige RVK für einzelne Bibliotheken ohne Dokumentbezug\n",
                    "dk_classification",
                )

        if self.logger:
            self.logger.info(f"DK title filter: {len(results_with_titles)} with titles, {titleless_count} without titles excluded")

        allowed_standard_rvk_map = {}
        allowed_nonstandard_rvk_map = {}
        rvk_source_map = {}
        selected_rvk_meta = {}
        anchor_terms = {
            canonicalize_keyword(keyword.split("(GND-ID:")[0].strip())
            for keyword in (rvk_anchor_keywords or [])
            if canonicalize_keyword(keyword.split("(GND-ID:")[0].strip())
        }
        for result in results_with_titles:
            if str(result.get("classification_type", result.get("type", "DK"))).upper() != "RVK":
                continue

            raw_code = result.get("dk", "")
            if not raw_code:
                continue

            normalized = canonicalize_rvk_notation(raw_code)
            if not normalized:
                continue
            rvk_status = result.get("rvk_validation_status")
            rvk_source_map[normalized] = {
                "source": result.get("source", "catalog"),
                "status": rvk_status or "standard",
            }
            matched_keywords = {
                canonicalize_keyword(str(item or ""))
                for item in (result.get("matched_keywords") or [])
                if canonicalize_keyword(str(item or ""))
            }
            selected_rvk_meta[normalized] = {
                "branch": str(result.get("branch_family", "") or ""),
                "depth": len([part for part in str(result.get("ancestor_path", "") or "").split(">") if part.strip()]),
                "anchor_hit_count": len(anchor_terms.intersection(matched_keywords)),
                "source": result.get("source", "catalog"),
            }
            if rvk_status in {"non_standard", "validation_error"}:
                allowed_nonstandard_rvk_map[normalized] = f"RVK {normalized}"
            else:
                allowed_standard_rvk_map[normalized] = f"RVK {normalized}"

        # WP-D1 P2: the input record's own classifications as priors. RVK prior
        # codes join the allowed-standard map (the guardrails are a hard gate —
        # a code the map does not carry cannot survive post-validation), marked
        # source "input_record" so ranking can prefer them. Existing catalog
        # entries for the same code are kept, not overwritten. - Claude Generated
        prior_block = ""
        if record_priors:
            from .classification_systems import KNOWN_SYSTEMS, codes_for_system

            prior_lines = []
            for system in KNOWN_SYSTEMS:
                codes = codes_for_system(record_priors, system)
                if not codes:
                    continue
                if system == "RVK":
                    if not include_rvk:
                        continue
                    kept = []
                    for code in codes:
                        normalized = canonicalize_rvk_notation(code)
                        if not normalized:
                            continue
                        kept.append(normalized)
                        if normalized not in rvk_source_map:
                            rvk_source_map[normalized] = {
                                "source": "input_record",
                                "status": "standard",
                            }
                            selected_rvk_meta[normalized] = {
                                "branch": "",
                                "depth": 0,
                                "anchor_hit_count": 0,
                                "source": "input_record",
                            }
                        # Status describes the NOTATION (is it official RVK?),
                        # not document relevance — a prior must not promote a
                        # known non-standard code to standard. - Claude Generated
                        if normalized not in allowed_nonstandard_rvk_map:
                            allowed_standard_rvk_map.setdefault(normalized, f"RVK {normalized}")
                    codes = kept
                if codes:
                    prior_lines.append(f"- {system}: {'; '.join(codes)}")
            if prior_lines:
                prior_block = (
                    "Der Eingabe-Datensatz trägt bereits eigene Katalog-Klassifikationen "
                    "(Autoritätsangabe des Katalogs — bevorzugt berücksichtigen, sofern "
                    "sie zum Inhalt passen; sie ersetzen die eigene Analyse nicht):\n"
                    + "\n".join(prior_lines)
                    + "\n\n"
                )
                if stream_callback:
                    stream_callback(
                        f"📚 Eingabe-Datensatz liefert eigene Klassifikationen als Prior: "
                        f"{'; '.join(prior_lines)}\n",
                        "dk_classification",
                    )

        # Format catalog results for LLM prompt with aggregated data - Claude Generated
        catalog_text = PipelineResultFormatter.format_dk_results_for_prompt(results_with_titles)
        catalog_text = prior_block + catalog_text

        if allowed_standard_rvk_map or allowed_nonstandard_rvk_map:
            rvk_guardrail = (
                "WICHTIG FÜR RVK:\n"
                "- Erfinde niemals neue RVK-Notationen.\n"
            )
            if allowed_standard_rvk_map:
                rvk_guardrail += (
                    "- Verwende RVK nur aus den unten gelisteten standardisierten Kandidaten.\n"
                    "- Wenn keine standardisierte RVK thematisch passt, gib keine RVK aus.\n"
                )
            else:
                rvk_guardrail += (
                    "- Es liegen keine standardisierten RVK aus dem Katalog vor.\n"
                    "- Bevorzuge standardisierte RVK aus dem RVK-API-Fallback; nur wenn keine solche passt, darf eine explizit als nicht-standardisiert/lokal markierte RVK verwendet werden.\n"
                )
            rvk_guardrail += (
                "- Achte auf den Fachpfad und verwerfe Kandidaten mit unpassendem Oberbereich.\n\n"
            )
            catalog_text = rvk_guardrail + catalog_text

        return {
            "results_with_titles": results_with_titles,
            "catalog_text": catalog_text,
            "allowed_standard_rvk_map": allowed_standard_rvk_map,
            "allowed_nonstandard_rvk_map": allowed_nonstandard_rvk_map,
            "rvk_source_map": rvk_source_map,
            "selected_rvk_meta": selected_rvk_meta,
        }

    def execute_dk_classification(
        self,
        original_abstract: str,
        dk_search_results: List[Dict[str, Any]],
        model: str = None,
        provider: str = None,
        stream_callback: Optional[callable] = None,
        dk_frequency_threshold: int = DEFAULT_DK_FREQUENCY_THRESHOLD,  # Claude Generated - Only pass classifications with >= N occurrences
        rvk_anchor_keywords: Optional[List[str]] = None,
        mode=None,  # <--- NEUER PARAMETER: Pipeline mode for PromptService
        record_priors: Optional[Dict[str, Any]] = None,  # WP-D1 P2: input record's own classifications - Claude Generated
        **kwargs,
    ) -> Tuple[List[str], Optional["LlmKeywordAnalysis"]]:
        """
        Execute LLM-based DK classification using pre-fetched catalog search results with intelligent provider selection - Claude Generated

        Args:
            original_abstract: The original abstract text for analysis
            dk_search_results: List of DK classification results from catalog search
            model: LLM model to use for classification (optional - SmartProvider selection if None)
            provider: LLM provider (optional - SmartProvider selection if None)
            stream_callback: Optional callback for streaming progress updates
            dk_frequency_threshold: Minimum occurrence count for DK classifications to be included.
                                  Only classifications that appear >= this many times in the catalog
                                  will be passed to the LLM for analysis. Default: 10.
                                  This reduces prompt size and focuses on most relevant classifications.
            **kwargs: Additional parameters for LLM (temperature, top_p, etc.)

        Returns:
            Tuple containing:
            - List of selected DK classification codes
            - LlmKeywordAnalysis object with details of the LLM call

        Note:
            The frequency threshold helps manage large result sets by filtering out
            classifications that occur infrequently in the catalog, which are typically
            less relevant for the given abstract.
        """

        # Intelligent provider selection using centralized method - Claude Generated
        provider, model = self._resolve_provider_smart(
            provider=provider,
            model=model,
            task_type="classification",
            prefer_fast=False,  # Classification should prioritize accuracy
            task_name="classification",
            step_id="dk_classification"
        )

        if not dk_search_results:
            if stream_callback:
                stream_callback("Keine DK-Suchergebnisse vorhanden - DK-Klassifikation übersprungen\n", "dk_classification")
            return [], None

        if stream_callback:
            stream_callback(f"Starte DK-Klassifikation mit {len(dk_search_results)} Katalog-Einträgen\n", "dk_classification")

        # Shared filtering/formatting with the agentic pipeline - Claude Generated
        prep = self.prepare_dk_classification_context(
            dk_search_results,
            original_abstract,
            dk_frequency_threshold=dk_frequency_threshold,
            rvk_anchor_keywords=rvk_anchor_keywords,
            stream_callback=stream_callback,
            record_priors=record_priors,
        )
        results_with_titles = prep["results_with_titles"]
        catalog_text = prep["catalog_text"]
        allowed_standard_rvk_map = prep["allowed_standard_rvk_map"]
        allowed_nonstandard_rvk_map = prep["allowed_nonstandard_rvk_map"]
        rvk_source_map = prep["rvk_source_map"]
        selected_rvk_meta = prep["selected_rvk_meta"]

        # Create AbstractData for LLM call
        from ..core.data_models import AbstractData
        abstract_data = AbstractData(
            abstract=original_abstract,
            keywords=catalog_text  # Use catalog results as "keywords" for dk_class prompt
        )

        if stream_callback:
            stream_callback("Starte LLM-basierte DK-Klassifikation...\n", "dk_classification")

        # Create stream callback adapter using centralized method - Claude Generated
        alima_stream_callback = self._create_stream_callback_adapter(
            stream_callback,
            "dk_classification"
        )

        # Filter parameters using centralized method - Claude Generated
        alima_kwargs = self._filter_alima_kwargs(kwargs, exclude_llm_params=True)

        # Execute LLM classification
        try:
            task_state = self.alima_manager.analyze_abstract(
                abstract_data=abstract_data,
                task="dk_classification",
                model=model,
                provider=provider,
                stream_callback=alima_stream_callback,
                mode=mode,  # <--- NEUER PARAMETER: Pass mode to AlimaManager
                **alima_kwargs,
            )

            if task_state.status == "failed":
                if stream_callback:
                    stream_callback(f"LLM-Klassifikation fehlgeschlagen: {task_state.analysis_result.full_text}\n", "dk_classification")
                return [], None

            # Extract DK classifications from LLM response
            response_text = task_state.analysis_result.full_text
            # Pass output_format from prompt_config for JSON extraction - Claude Generated
            _output_format = getattr(task_state.prompt_config, 'output_format', None) if task_state.prompt_config else None
            llm_classifications = self._extract_dk_from_response(response_text, output_format=_output_format)
            llm_classifications = self._filter_final_rvk_classifications(
                llm_classifications,
                allowed_standard_rvk_map,
                allowed_nonstandard_rvk_map,
                stream_callback=stream_callback,
            )
            # Extract analysis text from LLM response - Claude Generated
            from ..core.processing_utils import extract_analyse_text_from_response
            analyse_text = extract_analyse_text_from_response(response_text, output_format=_output_format)

            llm_dk_only = [
                code for code in llm_classifications
                if not str(code or "").strip().upper().startswith("RVK ")
            ]
            llm_rvk_only = [
                code for code in llm_classifications
                if str(code or "").strip().upper().startswith("RVK ")
            ]

            rescored_rvk = self._select_final_rvk_with_dk_context(
                results_with_titles,
                original_abstract,
                llm_dk_only,
                rvk_anchor_keywords=rvk_anchor_keywords,
                model=model,
                provider=provider,
                stream_callback=stream_callback,
                mode=mode,
                llm_kwargs=alima_kwargs,
            )
            if rescored_rvk:
                llm_rvk_only = rescored_rvk

            pruned_llm_rvk = []
            for code in llm_rvk_only:
                normalized = canonicalize_rvk_notation(str(code or "")[4:].strip())
                meta = selected_rvk_meta.get(normalized, {})
                depth = int(meta.get("depth", 0) or 0)
                anchor_hit_count = int(meta.get("anchor_hit_count", 0) or 0)
                source = str(meta.get("source", "catalog") or "catalog")
                branch = str(meta.get("branch", "") or "")

                is_broad_catalog = source == "catalog" and depth <= 2 and anchor_hit_count <= 1
                has_more_specific_peer = any(
                    other_code != normalized
                    and str(other_meta.get("branch", "") or "") == branch
                    and int(other_meta.get("depth", 0) or 0) > depth
                    and int(other_meta.get("anchor_hit_count", 0) or 0) >= anchor_hit_count
                    for other_code, other_meta in selected_rvk_meta.items()
                )

                if is_broad_catalog and has_more_specific_peer:
                    if stream_callback:
                        stream_callback(
                            f"⚠️ Verwerfe zu allgemeine RVK-Auswahl: RVK {normalized}\n",
                            "dk_classification",
                        )
                    continue
                pruned_llm_rvk.append(code)

            llm_rvk_only = pruned_llm_rvk

            max_total_classifications = 10
            max_dk_count = max(0, max_total_classifications - len(llm_rvk_only))
            dk_classifications = list(
                dict.fromkeys(llm_dk_only[:max_dk_count] + llm_rvk_only)
            )

            # Construct LlmKeywordAnalysis object for history/display - Claude Generated
            from ..core.data_models import LlmKeywordAnalysis
            llm_analysis = LlmKeywordAnalysis(
                task_name="dk_classification",
                model_used=task_state.model_used or model or "unknown",
                provider_used=task_state.provider_used or provider or "unknown",
                prompt_template=task_state.prompt_config.prompt if task_state.prompt_config else "",
                filled_prompt="", # We don't store the filled prompt to save space
                temperature=task_state.prompt_config.temp if task_state.prompt_config else 0.7,
                seed=task_state.prompt_config.seed if task_state.prompt_config else 0,
                response_full_text=response_text,
                extracted_gnd_classes=dk_classifications,
                analyse_text=analyse_text  # Analysis text from thought block or JSON - Claude Generated
            )

            final_rvk_sources = {
                "catalog_standard": 0,
                "catalog_nonstandard": 0,
                "rvk_gnd_index": 0,
                "rvk_api": 0,
            }
            if stream_callback:
                stream_callback(
                    f"DK-Klassifikation abgeschlossen: {len(dk_classifications)} Klassifikationscodes extrahiert\n",
                    "dk_classification",
                )
                for code in dk_classifications:
                    clean = str(code or "").strip()
                    if not clean.upper().startswith("RVK "):
                        continue
                    normalized = canonicalize_rvk_notation(clean[4:].strip())
                    source_info = rvk_source_map.get(normalized, {})
                    source = source_info.get("source", "catalog")
                    status = source_info.get("status", "standard")
                    if source == "rvk_gnd_index":
                        final_rvk_sources["rvk_gnd_index"] += 1
                    elif source == "rvk_api":
                        final_rvk_sources["rvk_api"] += 1
                    elif status in {"non_standard", "validation_error"}:
                        final_rvk_sources["catalog_nonstandard"] += 1
                    else:
                        final_rvk_sources["catalog_standard"] += 1

                source_parts = []
                if final_rvk_sources["catalog_standard"]:
                    source_parts.append(f"Katalog standard {final_rvk_sources['catalog_standard']}")
                if final_rvk_sources["catalog_nonstandard"]:
                    source_parts.append(f"Katalog lokal {final_rvk_sources['catalog_nonstandard']}")
                if final_rvk_sources["rvk_gnd_index"]:
                    source_parts.append(f"RVK-GND-Index {final_rvk_sources['rvk_gnd_index']}")
                if final_rvk_sources["rvk_api"]:
                    source_parts.append(f"RVK-API-Label {final_rvk_sources['rvk_api']}")
                if source_parts:
                    stream_callback(
                        f"ℹ️ Finale RVK-Quellen: {', '.join(source_parts)}\n",
                        "dk_classification",
                    )
            else:
                for code in dk_classifications:
                    clean = str(code or "").strip()
                    if not clean.upper().startswith("RVK "):
                        continue
                    normalized = canonicalize_rvk_notation(clean[4:].strip())
                    source_info = rvk_source_map.get(normalized, {})
                    source = source_info.get("source", "catalog")
                    status = source_info.get("status", "standard")
                    if source == "rvk_gnd_index":
                        final_rvk_sources["rvk_gnd_index"] += 1
                    elif source == "rvk_api":
                        final_rvk_sources["rvk_api"] += 1
                    elif status in {"non_standard", "validation_error"}:
                        final_rvk_sources["catalog_nonstandard"] += 1
                    else:
                        final_rvk_sources["catalog_standard"] += 1

            if llm_analysis is not None:
                setattr(llm_analysis, "rvk_provenance", final_rvk_sources)
            setattr(task_state, "rvk_provenance", final_rvk_sources)

            return dk_classifications, llm_analysis

        except Exception as e:
            if self.logger:
                self.logger.error(f"LLM DK classification failed: {e}")
            if stream_callback:
                stream_callback(f"LLM-Klassifikation-Fehler: {str(e)}\n", "dk_classification")
            return [], None

    def _extract_dk_from_response(self, response_text: str, output_format: Optional[str] = None) -> List[str]:
        """Extract DK and RVK classifications from LLM response - Claude Generated

        JSON-first extraction if output_format == "json", then XML fallback.
        PRIMARY: Extract from <final_list> tag (like keywords extraction)
        FALLBACK: Use regex patterns only if <final_list> not found
        """
        import re

        # JSON-first extraction - Claude Generated
        if output_format != "xml":
            from ..core.json_response_parser import parse_json_response, extract_dk_from_json
            data = parse_json_response(response_text)
            if data:
                codes = extract_dk_from_json(data)
                if codes:
                    if self.logger:
                        self.logger.info(f"✅ JSON DK-Extraktion: {len(codes)} Klassifikationen")
                    return codes
            if self.logger:
                self.logger.warning("JSON DK-Parsing fehlgeschlagen, Fallback auf XML")

        classification_codes = []

        # PRIMARY METHOD: Extract from <final_list> tag (preferred and most reliable)
        final_list_match = re.search(r'<final_list>\s*(.*?)\s*</final_list>', response_text, re.DOTALL | re.IGNORECASE)

        if final_list_match:
            # Extract and split by pipe separator
            final_list_content = final_list_match.group(1).strip()
            raw_codes = [code.strip() for code in final_list_content.split('|') if code.strip()]

            if self.logger:
                self.logger.info(f"✅ Extracted {len(raw_codes)} classifications from <final_list>")

            # Parse each code (format: "DK 615.9" or "RVK QC 130")
            for code in raw_codes:
                code_clean = code.strip()
                code_upper = code_clean.upper()

                # Keep DK and RVK prefixes intact
                if code_upper.startswith('DK ') or code_upper.startswith('RVK '):
                    classification_codes.append(code_clean)
                elif re.match(r'^\d+(?:\.\d+)*$', code_clean):
                    # Plain number without prefix -> assume DK
                    classification_codes.append(f"DK {code_clean}")
                elif re.match(r'^[A-Z]{1,2}\s*\d+', code_clean):
                    # Letter-number pattern -> assume RVK
                    classification_codes.append(f"RVK {code_clean}")
                else:
                    # Unknown format, keep as-is and log - Claude Generated
                    if self.logger:
                        self.logger.debug(f"⚠️ Unknown format: '{code_clean}' (not DK/RVK prefixed, not number pattern)")
                    classification_codes.append(code_clean)

            if self.logger:
                self.logger.info(f"✅ Parsed {len(classification_codes)} valid classifications from <final_list>")

            return classification_codes

        # FALLBACK METHOD: Use regex patterns (legacy, less reliable)
        if self.logger:
            self.logger.warning("⚠️  No <final_list> found in DK response, using regex fallback (may produce false positives)")

        # Look for DK patterns explicitly prefixed with "DK" (not arbitrary numbers)
        dk_pattern = r'\bDK\s+(\d{1,3}(?:\.\d+)*)\b'
        dk_matches = re.findall(dk_pattern, response_text, re.IGNORECASE)

        for match in dk_matches:
            classification_codes.append(f"DK {match}")

        # Look for RVK patterns explicitly prefixed with "RVK"
        rvk_pattern = r'\bRVK\s+([A-Z]{1,2}\s*\d{1,4}(?:\s*[A-Z]*)?)\b'
        rvk_matches = re.findall(rvk_pattern, response_text, re.IGNORECASE)

        for match in rvk_matches:
            classification_codes.append(f"RVK {match.strip()}")

        if self.logger and not classification_codes:
            self.logger.warning("⚠️  No classifications found using regex fallback either")

        # Remove duplicates while preserving order
        return list(dict.fromkeys(classification_codes))

    def _is_gnd_validated_keyword(self, keyword: str) -> bool:
        """
        Check if a keyword has been validated against GND system (contains GND-ID)
        Claude Generated - GND validation helper for strict filtering

        Args:
            keyword: Keyword string to validate

        Returns:
            True if keyword contains "(GND-ID:..." format, False for plain text keywords
        """
        return "(GND-ID:" in keyword

    def _flatten_keyword_centric_results(
        self, keyword_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Thin wrapper — logic lives module-level for reuse by formatters - Claude Generated"""
        return flatten_keyword_centric_results(keyword_results)

    def _calculate_dk_statistics(
        self,
        deduplicated_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for DK/RVK classifications - Claude Generated

        Tracks frequency, deduplication metrics, and keyword coverage for classifications.

        Args:
            deduplicated_results: Merged classification results from _flatten_keyword_centric_results()
            keyword_results: Original keyword-centric results from BiblioClient

        Returns:
            Statistics dictionary with frequency data and deduplication metrics
        """
        # Calculate basic metrics
        total_classifications = len(deduplicated_results)
        original_count = sum(len(kr.get("classifications", [])) for kr in keyword_results)
        duplicates_removed = original_count - total_classifications

        # Get top 10 most frequent classifications
        top_10 = sorted(deduplicated_results, key=lambda x: x["count"], reverse=True)[:10]

        # Build keyword coverage map (which keywords led to which DK codes)
        keyword_coverage = {}
        for result in deduplicated_results:
            for keyword in result.get("matched_keywords", []):
                if keyword not in keyword_coverage:
                    keyword_coverage[keyword] = []
                keyword_coverage[keyword].append(result["dk"])

        # Calculate frequency distribution (how many classifications have X occurrences)
        freq_dist = {}
        for result in deduplicated_results:
            count = result["count"]
            freq_dist[count] = freq_dist.get(count, 0) + 1

        # Estimated token savings (rough estimate: ~70 tokens per duplicate entry removed)
        estimated_token_savings = duplicates_removed * 70

        return {
            "total_classifications": total_classifications,
            "total_keywords_searched": len(keyword_results),
            "most_frequent": [
                {
                    "dk": r["dk"],
                    "type": r.get("type", r.get("classification_type", "DK")),
                    "count": r["count"],
                    "keywords": r.get("matched_keywords", []),
                    "unique_titles": len(r.get("titles", []))
                }
                for r in top_10
            ],
            "keyword_coverage": keyword_coverage,
            "frequency_distribution": freq_dist,
            "deduplication_stats": {
                "original_count": original_count,
                "duplicates_removed": duplicates_removed,
                "deduplication_rate": f"{duplicates_removed / original_count * 100:.1f}%" if original_count > 0 else "0%",
                "estimated_token_savings": estimated_token_savings
            }
        }

    @staticmethod
    def _strip_rvk_from_keyword_results(
        keyword_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Drop RVK-typed classifications from keyword-centric results - Claude Generated

        Used when ``rvk_enabled=False``: RVK is handled out-of-band (e.g. via
        the ``rvk_lookup`` tool), so neither the catalog excerpt nor the GUI
        transparency view should carry RVK entries.
        """
        cleaned: List[Dict[str, Any]] = []
        for entry in keyword_results or []:
            if not isinstance(entry, dict):
                cleaned.append(entry)
                continue
            classifications = entry.get("classifications")
            if isinstance(classifications, list):
                new_entry = dict(entry)
                new_entry["classifications"] = [
                    c for c in classifications
                    if str(
                        (c or {}).get("classification_type", (c or {}).get("type", "DK"))
                    ).upper() != "RVK"
                ]
                cleaned.append(new_entry)
            else:
                cleaned.append(entry)
        return cleaned

    def execute_dk_search(
        self,
        keywords: List[str],
        stream_callback: Optional[callable] = None,
        max_results: int = DEFAULT_DK_MAX_RESULTS,
        force_update: bool = False,  # Claude Generated
        strict_gnd_validation: bool = True,  # EXPERT OPTION: Allow disabling strict GND validation
        rvk_anchor_keywords: Optional[List[str]] = None,
        rvk_enabled: bool = True,  # Claude Generated - False skips all RVK anchor/API work
    ) -> List[Dict[str, Any]]:
        """
        Execute catalog search for DK classification data - Claude Generated

        Args:
            keywords: List of keywords to search
            stream_callback: Optional callback for progress updates
            max_results: Maximum results per keyword
            force_update: If True, results will be merged with existing cache (used by store_classification_results)
            strict_gnd_validation: If True (default), only use GND-validated keywords. If False, include plain text keywords.

        Returns:
            List of classification results with titles, counts, and metadata
        """

        # Log force_update status - Claude Generated
        if force_update and self.logger:
            self.logger.info("⚠️ Force update enabled: new titles will be merged with existing")

        # Resolve the DK/RVK extractor (BiblioClient / MarcXmlClient /
        # FincCatalogClient) from the enabled CLASSIFICATION-capable providers. - Claude Generated
        try:
            # DK/RVK source from the enabled CLASSIFICATION providers (finc opt-in →
            # custom plugin → SRU opt-in → Libero default), each built through the
            # factory from its own instance settings (WP P4 — replaces the former
            # 15-kwarg CatalogConfig wall + the catalog_type if-elif). All extractors
            # share the extract_dk_classifications_for_keywords contract, so the
            # per-keyword loop below is unchanged. - Claude Generated
            from src.core.search.factory import resolve_dk_extractor
            from .config_manager import ConfigManager

            try:
                _full_config = ConfigManager().load_config()
            except Exception:
                _full_config = None
            extractor = resolve_dk_extractor(
                config=_full_config,
                logger_=self.logger,
                stream_callback=(lambda m: stream_callback(m, "dk_search")) if stream_callback else None,
                debug=(self.logger.level <= 10) if self.logger else False,
            )
            # Operator hint: the Libero backend with no token falls back to web
            # scraping (finc/SRU backends need no token). - Claude Generated
            if type(extractor).__name__ == "BiblioClient":
                _cat_token = ""
                try:
                    for _i in (_full_config.enabled_instances_for("search_provider") if _full_config else []):
                        if getattr(_i, "provider_id", "") == "catalog":
                            _cat_token = (getattr(_i, "settings", None) or {}).get("token") or ""
                            break
                except Exception:
                    pass
                if not str(_cat_token).strip():
                    if self.logger:
                        self.logger.warning("No catalog token provided - BiblioClient will use web scraping fallback")
                    if stream_callback:
                        stream_callback("Kein Katalog-Token: Web-Fallback wird verwendet\n", "dk_search")
            # Preserve the old per-source progress line. - Claude Generated
            _dk_src_label = {
                "FincCatalogClient": "finc-Katalog (Titelliste + udk_raw pro Titel)",
                "MarcXmlClient": "MARC XML SRU",
                "BiblioClient": "Libero-Katalog",
            }.get(type(extractor).__name__, type(extractor).__name__)
            if self.logger:
                self.logger.info(f"DK source resolved: {type(extractor).__name__}")
            if stream_callback:
                stream_callback(f"Verwende {_dk_src_label} für DK-Suche\n", "dk_search")

        except Exception as e:
            error_msg = f"Catalog client initialization failed: {e}"
            if self.logger:
                self.logger.error(error_msg)
            if stream_callback:
                stream_callback(f"❌ Katalog-Initialisierung fehlgeschlagen: {str(e)}\n", "dk_search")
            # Return structured result with error info - Claude Generated
            return {
                "classifications": [],
                "statistics": {
                    "error": error_msg,
                    "initialization_failed": True,
                    "total_classifications": 0,
                    "total_keywords_searched": 0,
                    "most_frequent": [],
                    "keyword_coverage": {},
                    "frequency_distribution": {},
                    "deduplication_stats": {
                        "original_count": 0,
                        "duplicates_removed": 0,
                        "deduplication_rate": "0%",
                        "estimated_token_savings": 0
                    }
                },
                "keyword_results": []
            }

        # GND VALIDATION FILTERING - Claude Generated
        # Default (strict_gnd_validation=True): Only use keywords with validated GND-IDs
        #   This prevents irrelevant catalog titles from plain text keywords (e.g., "Molekül")
        # Expert mode (strict_gnd_validation=False): Include plain text keywords too
        gnd_validated_keywords = []
        gnd_keyword_entries = []
        plain_keywords = []

        for keyword in keywords:
            if self._is_gnd_validated_keyword(keyword):
                # Extract just the keyword part before (GND-ID:...)
                clean_keyword = keyword.split("(GND-ID:")[0].strip()
                gnd_validated_keywords.append(clean_keyword)
                gnd_id = extract_gnd_id(keyword)
                if gnd_id:
                    gnd_keyword_entries.append({
                        "keyword": clean_keyword,
                        "gnd_id": gnd_id,
                    })
            else:
                # Plain text keyword without GND-ID validation
                plain_keywords.append(keyword)

        # CRITICAL FIX: Deduplicate after GND-ID stripping - Claude Generated
        # Problem: Same keyword from different GND-IDs (e.g. "Cadmium (GND-ID: 123)" and "Cadmium (GND-ID: 456)")
        # both become "Cadmium" after stripping, leading to duplicate searches
        gnd_validated_keywords_before = len(gnd_validated_keywords)
        gnd_validated_keywords = deduplicate_canonical_keywords(gnd_validated_keywords)
        gnd_validated_keywords_after = len(gnd_validated_keywords)
        seen_gnd_ids = set()
        deduplicated_gnd_entries = []
        for entry in gnd_keyword_entries:
            gnd_id = entry.get("gnd_id")
            if not gnd_id or gnd_id in seen_gnd_ids:
                continue
            seen_gnd_ids.add(gnd_id)
            deduplicated_gnd_entries.append(entry)
        gnd_keyword_entries = deduplicated_gnd_entries
        rvk_anchor_entries = gnd_keyword_entries
        rvk_anchor_search_keywords = list(gnd_validated_keywords)
        if rvk_anchor_keywords:
            anchor_term_lookup: Dict[str, str] = {}
            for keyword in rvk_anchor_keywords:
                clean_keyword = keyword.split("(GND-ID:")[0].strip()
                normalized_keyword = canonicalize_keyword(clean_keyword)
                if normalized_keyword and clean_keyword:
                    anchor_term_lookup[normalized_keyword] = clean_keyword
            normalized_anchor_ids = {
                extract_gnd_id(keyword)
                for keyword in rvk_anchor_keywords
                if extract_gnd_id(keyword)
            }
            normalized_anchor_terms = {
                canonicalize_keyword(keyword.split("(GND-ID:")[0].strip())
                for keyword in rvk_anchor_keywords
                if canonicalize_keyword(keyword.split("(GND-ID:")[0].strip())
            }
            filtered_anchor_entries = [
                entry for entry in gnd_keyword_entries
                if (
                    entry.get("gnd_id") in normalized_anchor_ids
                    or canonicalize_keyword(entry.get("keyword", "")) in normalized_anchor_terms
                )
            ]
            filtered_anchor_keywords = [
                keyword for keyword in gnd_validated_keywords
                if canonicalize_keyword(keyword) in normalized_anchor_terms
            ]
            supplemental_anchor_terms = [
                anchor_term_lookup[normalized_term]
                for normalized_term in normalized_anchor_terms
                if normalized_term not in {
                    canonicalize_keyword(keyword) for keyword in filtered_anchor_keywords
                }
            ]
            if filtered_anchor_entries:
                rvk_anchor_entries = filtered_anchor_entries
            if filtered_anchor_keywords or supplemental_anchor_terms:
                rvk_anchor_search_keywords = deduplicate_canonical_keywords(
                    filtered_anchor_keywords + supplemental_anchor_terms
                )

        if gnd_validated_keywords_before != gnd_validated_keywords_after:
            duplicates_removed = gnd_validated_keywords_before - gnd_validated_keywords_after
            if self.logger:
                self.logger.info(
                    f"🔧 GND Keywords Deduplication: {gnd_validated_keywords_before} → "
                    f"{gnd_validated_keywords_after} unique ({duplicates_removed} duplicates removed)"
                )

        # Decide which keywords to use based on strict_gnd_validation setting - Claude Generated
        if strict_gnd_validation:
            final_search_keywords = gnd_validated_keywords
            filtered_keywords = plain_keywords
        else:
            # Combine GND and plain keywords, then deduplicate to avoid searching "Keyword" twice
            combined_keywords = gnd_validated_keywords + plain_keywords
            combined_before = len(combined_keywords)
            final_search_keywords = deduplicate_canonical_keywords(combined_keywords)
            combined_after = len(final_search_keywords)

            if combined_before != combined_after:
                duplicates_removed = combined_before - combined_after
                if self.logger:
                    self.logger.info(
                        f"🔧 Combined Keywords Deduplication: {combined_before} → "
                        f"{combined_after} unique ({duplicates_removed} duplicates removed)"
                    )

            filtered_keywords = []

        # Log filtering results - Claude Generated: Enhanced user feedback
        if filtered_keywords and strict_gnd_validation:
            # Build list of excluded keyword texts for user feedback
            excluded_list = ", ".join([kw[:40] for kw in filtered_keywords[:5]])
            if len(filtered_keywords) > 5:
                excluded_list += f", +{len(filtered_keywords)-5} weitere"

            if self.logger:
                self.logger.info(
                    f"🔍 DK Search (strict GND mode): {len(gnd_validated_keywords)} GND-validated keywords used, "
                    f"{len(filtered_keywords)} plain keywords excluded: {excluded_list}"
                )
            if stream_callback:
                stream_callback(
                    f"⚠️ DK-Suche-Filter: {len(gnd_validated_keywords)} GND-validierte Keywords, "
                    f"{len(filtered_keywords)} ohne GND ausgeschlossen\n   Ausgeschlossen: {excluded_list}\n",
                    "dk_search"
                )

        # Handle edge case: all keywords filtered
        if not final_search_keywords:
            if self.logger:
                self.logger.warning(
                    f"⚠️ DK Search: All {len(keywords)} keywords lack GND validation - skipping catalog search"
                )
            if stream_callback:
                stream_callback(
                    "⚠️ Keine Keywords für DK-Suche vorhanden\n",
                    "dk_search"
                )
            # Return empty results in new format - Claude Generated Step 3
            return {
                "classifications": [],
                "statistics": {
                    "total_classifications": 0,
                    "total_keywords_searched": 0,
                    "most_frequent": [],
                    "keyword_coverage": {},
                    "frequency_distribution": {},
                    "deduplication_stats": {
                        "original_count": 0,
                        "duplicates_removed": 0,
                        "deduplication_rate": "0%",
                        "estimated_token_savings": 0
                    }
                },
                "keyword_results": []
            }

        if stream_callback:
            mode_info = "(strict GND mode)" if strict_gnd_validation else "(including plain keywords)"
            stream_callback(
                f"Suche Katalog-Einträge für {len(final_search_keywords)} Keywords {mode_info} (max {max_results})\n",
                "dk_search"
            )
            if rvk_enabled and (rvk_anchor_keywords or rvk_anchor_entries):
                rvk_anchor_preview_terms = []
                if rvk_anchor_keywords:
                    rvk_anchor_preview_terms = [
                        keyword.split("(GND-ID:")[0].strip()
                        for keyword in rvk_anchor_keywords
                        if keyword.split("(GND-ID:")[0].strip()
                    ]
                if not rvk_anchor_preview_terms:
                    rvk_anchor_preview_terms = [
                        entry.get("keyword", "") for entry in rvk_anchor_entries if entry.get("keyword")
                    ]
                rvk_anchor_preview = ", ".join(rvk_anchor_preview_terms[:6])
                if len(rvk_anchor_preview_terms) > 6:
                    rvk_anchor_preview += f", +{len(rvk_anchor_preview_terms) - 6} weitere"
                if rvk_anchor_preview:
                    stream_callback(
                        f"ℹ️ RVK-Ankerbegriffe: {rvk_anchor_preview}\n",
                        "dk_search"
                    )

        # Execute catalog search with Per-Keyword Feedback - Claude Generated QUICK-FIX
        # IMPORTANT: Loop over keywords individually to provide per-keyword status feedback
        # This replaces the old single-batch call with individual keyword searches
        try:
            import inspect
            # Not every extractor accepts force_update (MarcXmlClient and custom
            # DK plugins don't) — detect once and pass it only when supported, so
            # any CLASSIFICATION-capable provider's extractor works here. - Claude Generated
            try:
                _dk_accepts_force = "force_update" in inspect.signature(
                    extractor.extract_dk_classifications_for_keywords
                ).parameters
            except (TypeError, ValueError):
                _dk_accepts_force = False

            dk_search_results = []
            success_count = 0
            failed_keywords = []

            # Process EACH keyword individually for detailed feedback
            for idx, keyword in enumerate(final_search_keywords, 1):
                # Check circuit breaker status before each search - Claude Generated
                if hasattr(extractor, 'get_circuit_breaker_status'):
                    cb_status = extractor.get_circuit_breaker_status()
                    if cb_status.get('open'):
                        remaining = cb_status.get('remaining_seconds', 60)
                        if stream_callback:
                            stream_callback(
                                f"⏳ Katalog-Server überlastet: Wartezeit {remaining}s...\n",
                                "dk_search"
                            )
                        if self.logger:
                            self.logger.warning(f"Circuit breaker open, waiting {remaining}s")
                        time.sleep(min(remaining + 1, 65))

                # Progress callback with percentage - Claude Generated
                if stream_callback:
                    progress_pct = int((idx / len(final_search_keywords)) * 100)
                    stream_callback(
                        f"[{idx}/{len(final_search_keywords)}] ({progress_pct}%) Suche '{keyword}'...\n",
                        "dk_search"
                    )

                try:
                    # Search THIS keyword only (not all keywords).
                    _dk_kwargs = {"keywords": [keyword], "max_results": max_results}
                    if _dk_accepts_force:
                        _dk_kwargs["force_update"] = force_update
                    kw_results = extractor.extract_dk_classifications_for_keywords(**_dk_kwargs)

                    # Analyze result for THIS keyword
                    if kw_results and len(kw_results) > 0:
                        kw_result = kw_results[0]
                        classifications = kw_result.get("classifications", [])

                        if classifications:
                            # SUCCESS: Keyword found with classifications
                            success_count += 1
                            if stream_callback:
                                stream_callback(
                                    f"  ✅ {keyword}: {len(classifications)} Klassifikationen gefunden\n",
                                    "dk_search"
                                )
                            dk_search_results.append(kw_result)
                        else:
                            # PARTIAL: Keyword found but no classifications
                            failed_keywords.append((keyword, "no_results", "Keine Klassifikationen"))
                            if stream_callback:
                                stream_callback(
                                    f"  ⚠️ {keyword}: Keine Klassifikationen gefunden\n",
                                    "dk_search"
                                )
                    else:
                        # FAILURE: Keyword search completely failed
                        failed_keywords.append((keyword, "error", "Suche fehlgeschlagen"))
                        if stream_callback:
                            stream_callback(
                                f"  ❌ {keyword}: Suche fehlgeschlagen\n",
                                "dk_search"
                            )

                except Exception as kw_error:
                    # Individual keyword error
                    if self.logger:
                        self.logger.error(f"Error searching keyword '{keyword}': {kw_error}")
                    failed_keywords.append((keyword, "error", str(kw_error)))
                    if stream_callback:
                        stream_callback(
                            f"  ❌ {keyword}: Fehler - {str(kw_error)}\n",
                            "dk_search"
                        )

            # Summary callback with complete stats
            if stream_callback:
                stream_callback(
                    f"✅ DK-Suche abgeschlossen: {success_count}/{len(final_search_keywords)} erfolgreich\n",
                    "dk_search"
                )

                # List failed keywords if any
                if failed_keywords:
                    failed_list = ", ".join([k for k, _, _ in failed_keywords[:5]])
                    if len(failed_keywords) > 5:
                        failed_list += f", +{len(failed_keywords) - 5} weitere"

                    stream_callback(
                        f"⚠️ {len(failed_keywords)} fehlgeschlagen: {failed_list}\n",
                        "dk_search"
                    )

            if rvk_enabled:
                dk_search_results = self._validate_catalog_rvk_candidates(
                    dk_search_results,
                    stream_callback=stream_callback,
                    rvk_anchor_keywords=rvk_anchor_keywords,
                )
                dk_search_results = self._inject_rvk_api_fallback(
                    rvk_anchor_search_keywords or final_search_keywords,
                    dk_search_results,
                    gnd_keyword_entries=rvk_anchor_entries,
                    stream_callback=stream_callback,
                )
                self._emit_rvk_source_diagnostics(
                    dk_search_results,
                    stream_callback=stream_callback,
                    step_id="dk_search",
                )
            else:
                # RVK handled out-of-band (rvk_lookup tool) — no anchor
                # validation, no RVK-API calls, no RVK in the result. - Claude Generated
                dk_search_results = self._strip_rvk_from_keyword_results(dk_search_results)

            # Deduplicate and flatten classifications - Claude Generated Step 3
            dk_search_results_flattened = self._flatten_keyword_centric_results(dk_search_results)

            # Calculate comprehensive statistics - Claude Generated Step 3
            dk_statistics = self._calculate_dk_statistics(dk_search_results_flattened, dk_search_results)

            # Return results in new format with statistics and transparency
            return {
                "classifications": dk_search_results_flattened,  # Deduplicated for LLM prompt
                "statistics": dk_statistics,                      # For display/diagnostics
                "keyword_results": dk_search_results              # Original keyword-centric for GUI transparency
            }

        except Exception as e:
            error_msg = f"DK catalog search failed: {e}"
            if self.logger:
                self.logger.error(error_msg)
            if stream_callback:
                stream_callback(f"❌ DK-Suche-Fehler: {str(e)}\n", "dk_search")

            # Return partial results if available - Claude Generated
            # Check if we collected any results before the error occurred
            if 'dk_search_results' in dir() and dk_search_results:
                if stream_callback:
                    stream_callback(
                        f"⚠️ Teilergebnisse: {len(dk_search_results)} Keywords erfolgreich vor Fehler\n",
                        "dk_search"
                    )
                if rvk_enabled:
                    dk_search_results = self._validate_catalog_rvk_candidates(
                        dk_search_results,
                        stream_callback=stream_callback,
                        rvk_anchor_keywords=rvk_anchor_keywords,
                    )
                    dk_search_results = self._inject_rvk_api_fallback(
                        final_search_keywords,
                        dk_search_results,
                        gnd_keyword_entries=gnd_keyword_entries,
                        stream_callback=stream_callback,
                    )
                    self._emit_rvk_source_diagnostics(
                        dk_search_results,
                        stream_callback=stream_callback,
                        step_id="dk_search",
                    )
                else:
                    dk_search_results = self._strip_rvk_from_keyword_results(dk_search_results)
                # Process partial results
                dk_search_results_flattened = self._flatten_keyword_centric_results(dk_search_results)
                dk_statistics = self._calculate_dk_statistics(dk_search_results_flattened, dk_search_results)
                dk_statistics["error"] = error_msg
                dk_statistics["partial_results"] = True
                return {
                    "classifications": dk_search_results_flattened,
                    "statistics": dk_statistics,
                    "keyword_results": dk_search_results
                }

            # Return empty results if no partial data
            return {
                "classifications": [],
                "statistics": {
                    "error": error_msg,
                    "total_classifications": 0,
                    "total_keywords_searched": 0,
                    "most_frequent": [],
                    "keyword_coverage": {},
                    "frequency_distribution": {},
                    "deduplication_stats": {
                        "original_count": 0,
                        "duplicates_removed": 0,
                        "deduplication_rate": "0%",
                        "estimated_token_savings": 0
                    }
                },
                "keyword_results": []
            }


