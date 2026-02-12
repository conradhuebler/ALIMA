"""
DkAnalysisTab - Derived from AbstractTab for DK classification analysis.
Allows users to see the LLM thought process and final list for the classification step.
"""

from .abstract_tab import AbstractTab
from ..core.alima_manager import AlimaManager
from ..llm.llm_service import LlmService
from ..core.pipeline_manager import PipelineManager
from ..core.unified_knowledge_manager import UnifiedKnowledgeManager
from ..utils.pipeline_utils import PipelineResultFormatter
from typing import Optional, List, Dict, Any
from PyQt6.QtWidgets import QWidget


class DkAnalysisTab(AbstractTab):
    """
    Claude Generated
    Specialized AbstractTab for DK classification analysis.
    """

    def __init__(
        self,
        alima_manager: AlimaManager,
        llm_service: LlmService,
        cache_manager: UnifiedKnowledgeManager,
        pipeline_manager: PipelineManager,
        main_window: Optional[QWidget] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(
            alima_manager,
            llm_service,
            cache_manager,
            pipeline_manager,
            main_window,
            parent,
        )

        # Set task to dk_classification (classification step)
        self.set_task("dk_classification")

        # Classification step needs the extracted keywords from previous steps
        self.need_keywords = True

        # Update group box title
        self.input_group.setTitle("Eingabe für DK-Klassifikation")

        # Increase keywords edit height to accommodate titles - Claude Generated
        self.keywords_edit.setMinimumHeight(200)
        self.keywords_edit.setMaximumHeight(400)

        # Store original keywords input for reuse after analysis - Claude Generated
        self._original_keywords_input = None

    def set_keywords(self, keywords: Any):
        """
        Enhanced keywords setter for DK analysis - Claude Generated
        Stores original input for restore after analysis, then formats for display.
        """
        self._original_keywords_input = keywords
        if isinstance(keywords, list) and len(keywords) > 0 and isinstance(keywords[0], dict):
            formatted_text = PipelineResultFormatter.format_dk_results_for_prompt(keywords)
            super().set_keywords(formatted_text)
        else:
            super().set_keywords(str(keywords))

    def restore_keywords_input(self):
        """Restore original DK search results after analysis — allows reuse - Claude Generated"""
        if self._original_keywords_input is not None:
            self.set_keywords(self._original_keywords_input)

    def on_analysis_completed(self, step):
        """Handle completion — restore DK keywords for reuse - Claude Generated"""
        super().on_analysis_completed(step)
        self.restore_keywords_input()
