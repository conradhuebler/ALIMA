"""Agentic Pipeline Executor - Claude Generated

Executes ALIMA workflows using GenericAgent instances configured from YAML.
Produces identical output to the sequential pipeline, but using LLM-driven agents with MCP tools.
"""
import logging
from typing import Optional, List, Callable
from pathlib import Path

from src.core.agents.base_agent import AgentConfig
from src.core.workflows.workflow_loader import WorkflowLoader
from src.core.workflows.workflow_executor import WorkflowExecutor
from src.core.workflows.workflow_context import WorkflowContext
from src.mcp.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgenticPipelineExecutor:
    """
    Executes ALIMA workflows using GenericAgent instances configured from YAML.

    All workflow steps, prompts, and tools are defined in YAML files,
    making the pipeline fully configurable without code changes.
    """

    def __init__(
        self,
        llm_service,
        config_manager=None,
        stream_callback: Optional[Callable[[str, str], None]] = None,
    ):
        self.llm_service = llm_service
        self.config_manager = config_manager
        self.stream_callback = stream_callback

        # Initialize tool registry
        self.tool_registry = ToolRegistry(config_manager=config_manager)
        self.tool_registry.register_all_tools()

        # Initialize workflow loader
        self.workflow_loader = WorkflowLoader()

    def execute(
        self,
        abstract: str,
        provider: str = "",
        model: str = "",
        temperature: float = 0.3,
        initial_keywords: Optional[List[str]] = None,
        enable_classification: bool = True,
        enable_validation: bool = True,
        max_iterations: int = 20,
        quality_threshold: float = 0.6,
        input_type: str = "text",
        source_value: Optional[str] = None,
        workflow_name: Optional[str] = None,
        custom_workflow_path: Optional[str] = None,
    ):
        """
        Execute full agentic pipeline, return KeywordAnalysisState.

        Args:
            abstract: Input text
            provider: LLM provider name
            model: Model name
            temperature: Sampling temperature
            initial_keywords: Pre-generated keywords
            enable_classification: Run DK classification (deprecated - use workflow)
            enable_validation: Run validation (deprecated - use workflow)
            max_iterations: Max tool-calling iterations per agent
            quality_threshold: Minimum quality score for each agent
            input_type: Source type ('text', 'doi', 'pdf', etc.)
            source_value: Original source (DOI, file path, etc.)
            workflow_name: Name of workflow to use (default: "default_alima")
            custom_workflow_path: Path to custom workflow YAML/JSON file

        Returns:
            KeywordAnalysisState compatible with sequential pipeline output
        """
        logger.info(f"Starting agentic pipeline with {provider}/{model}")

        # Determine workflow name
        if custom_workflow_path:
            workflow_name = None  # Will load from file
        elif workflow_name is None:
            workflow_name = "default_alima"

        # Load workflow
        try:
            if custom_workflow_path:
                workflow = self.workflow_loader.load_from_file(Path(custom_workflow_path))
                logger.info(f"Loaded custom workflow from {custom_workflow_path}")
            else:
                workflow = self.workflow_loader.load(workflow_name)
                logger.info(f"Loaded workflow: {workflow.name}")
        except Exception as e:
            logger.error(f"Failed to load workflow: {e}")
            # Fall back to default workflow
            workflow = self.workflow_loader.create_default_workflow()
            logger.info("Using fallback default workflow")

        # Create workflow context
        context = WorkflowContext(
            abstract=abstract,
            initial_keywords=initial_keywords or [],
            input_type=input_type,
            source_value=source_value,
            provider=provider,
            model=model,
            temperature=temperature,
        )

        # Create agent config
        agent_config = AgentConfig(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tool_iterations=max_iterations,
            quality_threshold=quality_threshold,
        )

        # Adapt stream callback
        def workflow_stream(text: str):
            if self.stream_callback:
                self.stream_callback(text, "workflow")

        # Create executor
        executor = WorkflowExecutor(
            llm_service=self.llm_service,
            tool_registry=self.tool_registry,
            config_manager=self.config_manager,
        )

        # Execute workflow
        context = executor.execute(
            workflow=workflow,
            context=context,
            agent_config=agent_config,
            stream_callback=workflow_stream,
        )

        # Convert to KeywordAnalysisState
        return context.to_keyword_analysis_state()