"""
DkAnalysisUnifiedTab - DK LLM Classification Tab for ALIMA
Pure AbstractTab wrapping the dk_classification pipeline step.
UB catalog search and statistics have moved to UBCatalogTab.
Claude Generated
"""

import logging
from typing import Any, Optional

from PyQt6.QtCore import pyqtSlot

from .abstract_tab import AbstractTab
from ..core.alima_manager import AlimaManager
from ..llm.llm_service import LlmService
from ..core.pipeline_manager import PipelineManager
from ..core.unified_knowledge_manager import UnifiedKnowledgeManager
from ..utils.pipeline_utils import PipelineResultFormatter


# ============================================================================
# Unified DK Analysis Tab
# ============================================================================

class DkAnalysisUnifiedTab(AbstractTab):
    """
    DK Analysis Tab for LLM-based classification.
    Receives DK catalog search results (from UBCatalogTab or pipeline) as
    keyword input and runs the dk_classification LLM step.
    Claude Generated
    """

    def __init__(
        self,
        alima_manager: AlimaManager,
        llm_service: LlmService,
        cache_manager: UnifiedKnowledgeManager,
        pipeline_manager: PipelineManager,
        main_window: Optional[object] = None,
        parent: Optional[object] = None,
    ):
        super().__init__(
            alima_manager,
            llm_service,
            cache_manager,
            pipeline_manager,
            main_window,
            parent,
        )

        self.set_task("dk_classification")
        self.need_keywords = True
        self.logger = logging.getLogger(__name__)

        # Update group box title
        self.input_group.setTitle("DK Classification Input")

        # Give keywords area more room for DK results
        self.keywords_edit.setMinimumHeight(150)
        self.keywords_edit.setMaximumHeight(300)

        # Keep a reference to the original input so it can be restored
        self._original_keywords_input = None

    # ------------------------------------------------------------------
    # Keywords handling
    # ------------------------------------------------------------------

    def set_keywords(self, keywords: Any):
        """Enhanced keywords setter: formats list-of-dicts DK results for display."""
        self._original_keywords_input = keywords
        if isinstance(keywords, list) and keywords and isinstance(keywords[0], dict):
            formatted_text = PipelineResultFormatter.format_dk_results_for_prompt(keywords)
            super().set_keywords(formatted_text)
        else:
            super().set_keywords(str(keywords))

    def restore_keywords_input(self):
        """Restore original DK search results after analysis completes."""
        if self._original_keywords_input is not None:
            self.set_keywords(self._original_keywords_input)

    def on_analysis_completed(self, step):
        """Handle completion — restore DK keywords for reuse."""
        super().on_analysis_completed(step)
        self.restore_keywords_input()

    # ------------------------------------------------------------------
    # Pipeline result slots
    # ------------------------------------------------------------------

    @pyqtSlot(object)
    def update_data(self, analysis_state):
        """Alias for receive_pipeline_results — backward compatibility."""
        self.receive_pipeline_results(analysis_state)

    @pyqtSlot(object)
    def receive_pipeline_results(self, analysis_state):
        """Populate DK LLM input from a complete pipeline state."""
        if not analysis_state:
            return

        # Abstract
        if hasattr(analysis_state, 'original_abstract') and analysis_state.original_abstract:
            self.set_abstract(analysis_state.original_abstract)

        # DK catalog search results are the LLM input (keywords field)
        if hasattr(analysis_state, 'dk_search_results_flattened') and analysis_state.dk_search_results_flattened:
            self.set_keywords(analysis_state.dk_search_results_flattened)
        elif hasattr(analysis_state, 'dk_search_results') and analysis_state.dk_search_results:
            self.set_keywords(analysis_state.dk_search_results)

        # Display existing LLM results if available
        if hasattr(analysis_state, 'dk_llm_analysis') and analysis_state.dk_llm_analysis:
            self.display_llm_response(analysis_state.dk_llm_analysis.response_full_text)
            self.add_external_analysis_to_history(analysis_state)

    @pyqtSlot(list)
    def receive_catalog_results(self, results_flattened: list):
        """
        Receive UB catalog search results from UBCatalogTab.
        Sets the flattened DK results as keyword input for LLM classification.
        Claude Generated
        """
        self.set_keywords(results_flattened)

    # ------------------------------------------------------------------
    # Backward compatibility
    # ------------------------------------------------------------------

    def set_abstract(self, abstract: str):
        """Set abstract text."""
        super().set_abstract(abstract)
