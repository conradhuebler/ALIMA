"""Pipeline result persistence (JSON save/resume + autosave) - Claude Generated.

``PipelineJsonManager`` (TaskState/KeywordAnalysisState <-> JSON), the canonical
``export_analysis_state_to_file`` writer, and ``AnalysisPersistence`` (autosave).
Split out of the former ``pipeline_utils`` god-module; still re-exported from it.
No dependency on PipelineStepExecutor.
"""

import dataclasses
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.data_models import (
    AbstractData,
    TaskState,
    AnalysisResult,
    KeywordAnalysisState,
    LlmKeywordAnalysis,
    SearchResult,
)

logger = logging.getLogger(__name__)

def export_analysis_state_to_file(
    analysis_state: "KeywordAnalysisState",
    file_path: str,
    input_data: Optional[Dict[str, Any]] = None,
    status: str = "completed",
    current_step: str = "classification",
    session_id: Optional[str] = None,
    created_at: Optional[str] = None,
    exported_at: Optional[str] = None,
    autosave_timestamp: Optional[str] = None,
    validate_rvk: bool = True,
) -> None:
    """Write a KeywordAnalysisState using the canonical web/API export schema."""
    from ..webapp.result_serialization import (
        build_export_payload,
        extract_results_from_analysis_state,
    )

    if input_data is None:
        input_data = {
            "type": "text",
            "text_preview": getattr(analysis_state, "original_abstract", "")[:100],
        }

    results = extract_results_from_analysis_state(analysis_state)
    payload = build_export_payload(
        session_id=session_id,
        created_at=created_at or getattr(analysis_state, "timestamp", None),
        status=status,
        current_step=current_step,
        input_data=input_data,
        results=results,
        autosave_timestamp=autosave_timestamp,
        exported_at=exported_at,
        validate_rvk=validate_rvk,
    )

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


class PipelineJsonManager:
    """JSON serialization/deserialization for pipeline states - Claude Generated"""

    @staticmethod
    def task_state_to_dict(task_state: TaskState) -> dict:
        """Convert TaskState to dictionary for JSON serialization - Claude Generated"""
        task_state_dict = asdict(task_state)

        # Convert nested dataclasses to dicts if they exist
        if task_state_dict.get("abstract_data"):
            task_state_dict["abstract_data"] = asdict(task_state.abstract_data)
        if task_state_dict.get("analysis_result"):
            task_state_dict["analysis_result"] = asdict(task_state.analysis_result)
        if task_state_dict.get("prompt_config"):
            task_state_dict["prompt_config"] = asdict(task_state.prompt_config)

        return task_state_dict

    @staticmethod
    def convert_sets_to_lists(obj):
        """Convert sets to lists for JSON serialization - Claude Generated"""
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, dict):
            return {
                k: PipelineJsonManager.convert_sets_to_lists(v) for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [PipelineJsonManager.convert_sets_to_lists(elem) for elem in obj]
        return obj

    @staticmethod
    def convert_lists_to_sets(obj):
        """Convert known list fields back to sets after JSON loading - Claude Generated

        Enhanced to handle all known set fields: gndid, ddc, dk, missing_concepts
        """
        if isinstance(obj, dict):
            # Known set fields in search results and data models
            SET_FIELDS = {"gndid", "ddc", "dk", "missing_concepts"}

            result = {}
            for key, value in obj.items():
                if key in SET_FIELDS and isinstance(value, list):
                    # Convert known set fields back to sets
                    result[key] = set(value)
                elif isinstance(value, (dict, list)):
                    result[key] = PipelineJsonManager.convert_lists_to_sets(value)
                else:
                    result[key] = value
            return result
        elif isinstance(obj, list):
            return [PipelineJsonManager.convert_lists_to_sets(elem) for elem in obj]
        return obj

    @staticmethod
    def save_analysis_state(analysis_state: KeywordAnalysisState, file_path: str):
        """Save KeywordAnalysisState to JSON file - Claude Generated"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    PipelineJsonManager.convert_sets_to_lists(asdict(analysis_state)),
                    f,
                    ensure_ascii=False,
                    indent=4,
                )
        except Exception as e:
            raise ValueError(f"Error saving analysis state to JSON: {e}")

    @staticmethod
    def load_analysis_state(file_path: str) -> KeywordAnalysisState:
        """Load KeywordAnalysisState from JSON file with deep object reconstruction - Claude Generated"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Detect and unwrap webapp session format - Claude Generated (Support both CLI/GUI and Webapp formats)
            if "session_id" in data and "results" in data:
                logger.info(f"Detected webapp session format (session_id: {data.get('session_id')}), unwrapping results...")
                # Extract the actual analysis state from webapp's session wrapper
                data = data["results"]

                # Map webapp field names to KeywordAnalysisState field names
                if "final_llm_call_details" in data and "final_llm_analysis" not in data:
                    logger.info("Mapping 'final_llm_call_details' → 'final_llm_analysis'")
                    data["final_llm_analysis"] = data.pop("final_llm_call_details")

                # Map webapp LLM field names to LlmKeywordAnalysis field names - Claude Generated
                def fix_llm_field_names(llm_dict):
                    """Fix field names in LLM analysis objects from webapp format"""
                    if not isinstance(llm_dict, dict):
                        return llm_dict

                    # Map provider → provider_used
                    if "provider" in llm_dict and "provider_used" not in llm_dict:
                        llm_dict["provider_used"] = llm_dict.pop("provider")

                    # Map model → model_used
                    if "model" in llm_dict and "model_used" not in llm_dict:
                        llm_dict["model_used"] = llm_dict.pop("model")

                    # Remove webapp-specific fields not in LlmKeywordAnalysis
                    for field in ["extracted_keywords", "token_count"]:
                        llm_dict.pop(field, None)

                    # Fill in missing required fields with defaults (webapp doesn't store these)
                    llm_dict.setdefault("task_name", "webapp-import")
                    llm_dict.setdefault("prompt_template", "")
                    llm_dict.setdefault("filled_prompt", "")
                    llm_dict.setdefault("temperature", 0.7)
                    llm_dict.setdefault("seed", None)

                    return llm_dict

                # Apply mappings to LLM analysis objects
                if isinstance(data.get("initial_llm_call_details"), dict):
                    data["initial_llm_call_details"] = fix_llm_field_names(data["initial_llm_call_details"])

                if isinstance(data.get("final_llm_analysis"), dict):
                    data["final_llm_analysis"] = fix_llm_field_names(data["final_llm_analysis"])

                # Remap property aliases before filtering — Claude Generated
                if "classifications" in data and "dk_classifications" not in data:
                    data["dk_classifications"] = data.pop("classifications")

                # Whitelist filter: keep only KeywordAnalysisState fields — Claude Generated
                # GUI/webapp exports add extra keys (keyword_chains, classifications_deprecated_alias,
                # pipeline_metadata, etc.) that KeywordAnalysisState.__init__() doesn't accept.
                valid_fields = {f.name for f in dataclasses.fields(KeywordAnalysisState)}
                unknown = [k for k in list(data.keys()) if k not in valid_fields]
                if unknown:
                    for k in unknown:
                        data.pop(k)
                    logger.info(f"Removed non-KeywordAnalysisState fields: {', '.join(unknown)}")

                # Fill in missing required fields from webapp format - Claude Generated
                data.setdefault("search_suggesters_used", [])
                data.setdefault("initial_gnd_classes", [])
                data.setdefault("timestamp", datetime.now().isoformat())
                data.setdefault("pipeline_step_completed", "classification")
                data.setdefault("initial_llm_call_details", None)  # May not be in webapp export

            # Deep reconstruction of nested dataclass objects
            from ..core.data_models import SearchResult, LlmKeywordAnalysis

            # Reconstruct SearchResult objects
            if data.get("search_results"):
                reconstructed_search_results = []
                for item in data["search_results"]:
                    # Convert known list fields back to sets (e.g., gndid fields)
                    if "results" in item:
                        item["results"] = PipelineJsonManager.convert_lists_to_sets(item["results"])
                    reconstructed_search_results.append(SearchResult(**item))
                data["search_results"] = reconstructed_search_results

            # Reconstruct LlmKeywordAnalysis objects
            if data.get("initial_llm_call_details"):
                data["initial_llm_call_details"] = LlmKeywordAnalysis(**data["initial_llm_call_details"])

            if data.get("final_llm_analysis"):
                data["final_llm_analysis"] = LlmKeywordAnalysis(**data["final_llm_analysis"])

            if data.get("dk_llm_analysis"):
                data["dk_llm_analysis"] = LlmKeywordAnalysis(**data["dk_llm_analysis"])

            # Ensure list fields are actually lists - Claude Generated (Fix for string parsing bug)
            # This prevents "B, a, t, t, e, r, i, e" issue when JSON contains strings instead of lists
            if "initial_keywords" in data and isinstance(data["initial_keywords"], str):
                # Split comma-separated string back to list
                data["initial_keywords"] = [kw.strip() for kw in data["initial_keywords"].split(",") if kw.strip()]

            if data.get("final_llm_analysis") and hasattr(data["final_llm_analysis"], "extracted_gnd_keywords"):
                if isinstance(data["final_llm_analysis"].extracted_gnd_keywords, str):
                    kw_str = data["final_llm_analysis"].extracted_gnd_keywords
                    data["final_llm_analysis"].extracted_gnd_keywords = [kw.strip() for kw in kw_str.split(",") if kw.strip()]

            if "dk_classifications" in data and isinstance(data["dk_classifications"], str):
                data["dk_classifications"] = [dk.strip() for dk in data["dk_classifications"].split(",") if dk.strip()]

            return KeywordAnalysisState(**data)

        except FileNotFoundError:
            raise ValueError(f"Analysis state file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file: {file_path}. Error: {e}")
        except TypeError as e:
            raise ValueError(f"JSON structure incompatible with KeywordAnalysisState: {e}")
        except Exception as e:
            raise ValueError(f"Error loading analysis state: {e}")

    @staticmethod
    def save_task_state(task_state: TaskState, file_path: str):
        """Save TaskState to JSON file - Claude Generated"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    PipelineJsonManager.task_state_to_dict(task_state),
                    f,
                    ensure_ascii=False,
                    indent=4,
                )
        except Exception as e:
            raise ValueError(f"Error saving task state to JSON: {e}")


class AnalysisPersistence:
    """
    Unified persistence interface for KeywordAnalysisState with Qt dialog integration.
    Eliminates code duplication across GUI components by providing a single API.
    Claude Generated
    """

    @staticmethod
    def save_with_dialog(
        state: "KeywordAnalysisState",
        parent_widget=None,
        default_filename: str = None
    ) -> Optional[str]:
        """
        Save KeywordAnalysisState with Qt file dialog.

        Args:
            state: KeywordAnalysisState object to save
            parent_widget: Qt parent widget for dialog (optional)
            default_filename: Default filename suggestion (optional)

        Returns:
            File path if saved successfully, None if cancelled or failed

        Claude Generated
        """
        try:
            from PyQt6.QtWidgets import QFileDialog, QMessageBox  # pyright: ignore[reportMissingImports]
        except ImportError:
            raise ImportError("PyQt6 required for GUI dialogs. Use PipelineJsonManager directly for CLI.")

        # Generate default filename if not provided
        if not default_filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            default_filename = f"analysis_state_{timestamp}.json"

        # Resolve full path using configured autosave directory
        from .pipeline_defaults import get_autosave_dir
        default_path = str(get_autosave_dir() / default_filename)

        # Open save dialog
        file_path, _ = QFileDialog.getSaveFileName(
            parent_widget,
            "Analyse-Zustand speichern",
            default_path,
            "JSON Files (*.json);;All Files (*)"
        )

        if not file_path:
            return None  # User cancelled

        try:
            # Use canonical web/API export schema for GUI saves
            export_analysis_state_to_file(state, file_path)

            # Success notification
            if parent_widget:
                QMessageBox.information(
                    parent_widget,
                    "Erfolg",
                    f"Analyse-Zustand erfolgreich gespeichert:\n{file_path}"
                )

            return file_path

        except Exception as e:
            # Error notification
            if parent_widget:
                QMessageBox.critical(
                    parent_widget,
                    "Fehler",
                    f"Fehler beim Speichern:\n\n{str(e)}"
                )
            raise

    @staticmethod
    def load_with_dialog(parent_widget=None) -> Optional["KeywordAnalysisState"]:
        """
        Load KeywordAnalysisState with Qt file dialog.

        Args:
            parent_widget: Qt parent widget for dialog (optional)

        Returns:
            KeywordAnalysisState object if loaded successfully, None if cancelled or failed

        Claude Generated
        """
        try:
            from PyQt6.QtWidgets import QFileDialog, QMessageBox  # pyright: ignore[reportMissingImports]
        except ImportError:
            raise ImportError("PyQt6 required for GUI dialogs. Use PipelineJsonManager directly for CLI.")

        # Resolve start directory using configured autosave directory
        from .pipeline_defaults import get_autosave_dir

        # Open load dialog
        file_path, _ = QFileDialog.getOpenFileName(
            parent_widget,
            "Analyse-Zustand laden",
            str(get_autosave_dir()),
            "JSON Files (*.json);;All Files (*)"
        )

        if not file_path:
            return None  # User cancelled

        try:
            # Use PipelineJsonManager for actual load
            state = PipelineJsonManager.load_analysis_state(file_path)

            # Success notification
            if parent_widget:
                QMessageBox.information(
                    parent_widget,
                    "Erfolg",
                    f"Analyse-Zustand erfolgreich geladen:\n{file_path}"
                )

            return state

        except Exception as e:
            # Error notification
            if parent_widget:
                QMessageBox.critical(
                    parent_widget,
                    "Fehler",
                    f"Fehler beim Laden:\n\n{str(e)}"
                )
            return None
