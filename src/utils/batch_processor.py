"""
Batch Processor - Stapelverarbeitung für ALIMA Pipeline
Claude Generated - Implements batch processing with resume functionality
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from ..core.data_models import KeywordAnalysisState
from .pipeline_utils import PipelineStepExecutor, PipelineJsonManager
from .doi_resolver import resolve_input_to_text
from ..core.unified_knowledge_manager import UnifiedKnowledgeManager


class SourceType(Enum):
    """Supported source types for batch processing - Claude Generated"""
    DOI = "DOI"
    PDF = "PDF"
    TXT = "TXT"
    IMG = "IMG"
    URL = "URL"


@dataclass
class BatchSource:
    """Represents a single source in a batch - Claude Generated"""
    source_type: SourceType
    source_value: str
    custom_name: Optional[str] = None
    step_overrides: Optional[Dict[str, Any]] = None
    line_number: int = 0

    def get_safe_filename(self) -> str:
        """Generate a safe filename for this source - Claude Generated"""
        if self.custom_name:
            base = self.custom_name
        elif self.source_type == SourceType.DOI:
            # Replace / with _ for DOI
            base = self.source_value.replace('/', '_').replace(':', '_')
        else:
            # Use basename for file paths, domain for URLs
            if self.source_type == SourceType.URL:
                from urllib.parse import urlparse
                base = urlparse(self.source_value).netloc
            else:
                base = Path(self.source_value).stem

        # Ensure safe filename
        safe = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in base)
        return f"{safe}.json"


@dataclass
class BatchSourceResult:
    """Result of processing a single batch source - Claude Generated"""
    source: BatchSource
    success: bool
    output_file: Optional[str] = None
    error_message: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    state: Optional[KeywordAnalysisState] = None


@dataclass
class BatchState:
    """Tracks the state of a batch processing job - Claude Generated"""
    batch_file: str
    output_dir: str
    total_sources: int
    processed_sources: List[str] = field(default_factory=list)
    failed_sources: List[Dict[str, Any]] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    pipeline_config: Optional[Dict[str, Any]] = None

    def mark_processed(self, source: BatchSource, success: bool, error: Optional[str] = None):
        """Mark a source as processed - Claude Generated"""
        key = f"{source.source_type.value}:{source.source_value}"
        if key not in self.processed_sources:
            self.processed_sources.append(key)

        if not success and error:
            self.failed_sources.append({
                "source": key,
                "error": error,
                "timestamp": datetime.now().isoformat()
            })

    def is_processed(self, source: BatchSource) -> bool:
        """Check if a source has been processed - Claude Generated"""
        key = f"{source.source_type.value}:{source.source_value}"
        return key in self.processed_sources

    def save(self, filepath: str):
        """Save state to JSON file - Claude Generated"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> 'BatchState':
        """Load state from JSON file - Claude Generated"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


class BatchSourceParser:
    """Parses batch input files - Claude Generated"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def parse_batch_file(self, filepath: str) -> List[BatchSource]:
        """
        Parse a batch file into BatchSource objects

        Format:
        DOI:10.1234/example
        PDF:/path/to/file.pdf
        TXT:/path/to/text.txt
        IMG:/path/to/image.png
        URL:https://example.com/abstract

        Optional extended format:
        SOURCE ; custom_name ; {"step": "override"}

        Claude Generated
        """
        sources = []

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Batch file not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Skip empty lines and comments
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                try:
                    source = self._parse_line(line, line_num)
                    if source:
                        sources.append(source)
                except Exception as e:
                    self.logger.warning(f"Line {line_num}: Failed to parse: {e}")

        self.logger.info(f"Parsed {len(sources)} sources from {filepath}")
        return sources

    def _parse_line(self, line: str, line_num: int) -> Optional[BatchSource]:
        """Parse a single line into a BatchSource - Claude Generated"""
        # Check for extended format with semicolons
        parts = [p.strip() for p in line.split(';')]
        main_part = parts[0]
        custom_name = parts[1] if len(parts) > 1 else None
        step_overrides = None

        if len(parts) > 2:
            try:
                step_overrides = json.loads(parts[2])
            except json.JSONDecodeError:
                self.logger.warning(f"Line {line_num}: Invalid JSON in step overrides")

        # Parse main part: TYPE:VALUE
        if ':' not in main_part:
            raise ValueError(f"Missing ':' separator. Expected format: TYPE:VALUE")

        type_str, value = main_part.split(':', 1)
        type_str = type_str.strip().upper()
        value = value.strip()

        # Validate source type
        try:
            source_type = SourceType[type_str]
        except KeyError:
            raise ValueError(f"Unknown source type: {type_str}. Valid types: {[t.value for t in SourceType]}")

        # Validate value based on type
        if source_type in (SourceType.PDF, SourceType.TXT, SourceType.IMG):
            if not os.path.exists(value):
                self.logger.warning(f"Line {line_num}: File not found: {value}")

        return BatchSource(
            source_type=source_type,
            source_value=value,
            custom_name=custom_name,
            step_overrides=step_overrides,
            line_number=line_num
        )


class BatchProcessor:
    """
    Main batch processing engine - Claude Generated
    Uses PipelineStepExecutor for consistent pipeline execution
    """

    def __init__(
        self,
        pipeline_executor: PipelineStepExecutor,
        cache_manager: UnifiedKnowledgeManager,
        output_dir: str,
        logger: Optional[logging.Logger] = None,
        continue_on_error: bool = True,
    ):
        """
        Initialize BatchProcessor

        Args:
            pipeline_executor: PipelineStepExecutor instance for running pipelines
            cache_manager: UnifiedKnowledgeManager for GND caching
            output_dir: Directory for output JSON files
            logger: Optional logger instance
            continue_on_error: If True, continue processing on error; if False, stop
        """
        self.pipeline_executor = pipeline_executor
        self.cache_manager = cache_manager
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger(__name__)
        self.continue_on_error = continue_on_error

        # Create output directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Parser instance
        self.parser = BatchSourceParser(logger=self.logger)

        # State tracking
        self.batch_state: Optional[BatchState] = None
        self.results: List[BatchSourceResult] = []
        self.pipeline_config_dict: Optional[Dict[str, Any]] = None

        # Callbacks
        self.on_source_start: Optional[Callable[[BatchSource, int, int], None]] = None
        self.on_source_complete: Optional[Callable[[BatchSourceResult], None]] = None
        self.on_batch_complete: Optional[Callable[[List[BatchSourceResult]], None]] = None

    def process_batch_file(
        self,
        batch_file: str,
        pipeline_config: Optional[Dict[str, Any]] = None,
        resume_state: Optional[str] = None,
    ) -> List[BatchSourceResult]:
        """
        Process all sources in a batch file

        Args:
            batch_file: Path to batch file
            pipeline_config: Optional pipeline configuration dict
            resume_state: Optional path to resume state JSON

        Returns:
            List of BatchSourceResult objects

        Claude Generated
        """
        # Store pipeline config for use in processing - Claude Generated
        self.pipeline_config_dict = pipeline_config

        # Load or create batch state
        if resume_state and os.path.exists(resume_state):
            self.logger.info(f"Resuming from state: {resume_state}")
            self.batch_state = BatchState.load(resume_state)
            # Restore pipeline config from state if available
            if self.batch_state.pipeline_config:
                self.pipeline_config_dict = self.batch_state.pipeline_config
        else:
            sources = self.parser.parse_batch_file(batch_file)
            self.batch_state = BatchState(
                batch_file=batch_file,
                output_dir=str(self.output_dir),
                total_sources=len(sources),
                pipeline_config=pipeline_config
            )

        # Parse sources (needed even for resume to get full list)
        sources = self.parser.parse_batch_file(batch_file)

        # Process each source
        self.results = []
        for idx, source in enumerate(sources, 1):
            # Skip if already processed (resume mode)
            if self.batch_state.is_processed(source):
                self.logger.info(f"[{idx}/{len(sources)}] Skipping already processed: {source.source_value}")
                continue

            # Callback: source start
            if self.on_source_start:
                self.on_source_start(source, idx, len(sources))

            # Process source
            result = self._process_single_source(source, idx, len(sources))
            self.results.append(result)

            # Update state
            self.batch_state.mark_processed(source, result.success, result.error_message)

            # Save state after each source (for resume capability)
            state_file = self.output_dir / ".batch_state.json"
            self.batch_state.save(str(state_file))

            # Callback: source complete
            if self.on_source_complete:
                self.on_source_complete(result)

            # Stop on error if configured
            if not result.success and not self.continue_on_error:
                self.logger.error(f"Stopping batch processing due to error: {result.error_message}")
                break

        # Mark batch complete
        self.batch_state.end_time = datetime.now().isoformat()
        state_file = self.output_dir / ".batch_state.json"
        self.batch_state.save(str(state_file))

        # Callback: batch complete
        if self.on_batch_complete:
            self.on_batch_complete(self.results)

        return self.results

    def _process_single_source(
        self,
        source: BatchSource,
        current_idx: int,
        total: int
    ) -> BatchSourceResult:
        """
        Process a single batch source through the pipeline

        Claude Generated
        """
        self.logger.info(f"[{current_idx}/{total}] Processing {source.source_type.value}: {source.source_value}")

        result = BatchSourceResult(
            source=source,
            success=False,
            start_time=datetime.now().isoformat()
        )

        try:
            # Step 1: Resolve input to text
            input_text = self._resolve_source_to_text(source)

            if not input_text:
                raise ValueError(f"Failed to resolve {source.source_type.value} to text")

            # Step 2: Execute complete pipeline - Claude Generated
            self.logger.info(f"[{current_idx}/{total}] Starting pipeline execution for {source.source_type.value}")

            # Create PipelineManager for this source
            from ..core.pipeline_manager import PipelineManager, PipelineConfig

            pipeline_manager = PipelineManager(
                alima_manager=self.pipeline_executor.alima_manager,
                cache_manager=self.cache_manager,
                logger=self.logger,
                config_manager=self.pipeline_executor.config_manager
            )

            # Apply pipeline configuration if provided
            if hasattr(self, 'pipeline_config_dict') and self.pipeline_config_dict:
                # Merge batch pipeline config
                pipeline_manager.config = self._apply_batch_config(
                    pipeline_manager.config,
                    self.pipeline_config_dict
                )

            # Set up completion tracking
            pipeline_completed = False
            pipeline_error = None

            def on_pipeline_complete(analysis_state):
                nonlocal pipeline_completed
                pipeline_completed = True

            def on_step_error(step, error_msg):
                nonlocal pipeline_error
                pipeline_error = error_msg

            pipeline_manager.pipeline_completed_callback = on_pipeline_complete
            pipeline_manager.step_error_callback = on_step_error

            # Start pipeline (synchronous with auto_advance=True)
            pipeline_id = pipeline_manager.start_pipeline(
                input_text=input_text,
                input_type="text"
            )

            # Pipeline runs synchronously through all steps with auto_advance
            # Wait a moment to ensure completion
            import time
            max_wait = 300  # 5 minutes timeout
            wait_time = 0
            while not pipeline_completed and not pipeline_error and wait_time < max_wait:
                time.sleep(0.1)
                wait_time += 0.1

            if pipeline_error:
                raise RuntimeError(f"Pipeline failed: {pipeline_error}")

            if not pipeline_completed:
                raise RuntimeError(f"Pipeline timeout after {max_wait} seconds")

            # Get completed state
            state = pipeline_manager.current_analysis_state

            if not state:
                raise RuntimeError("Pipeline completed but no analysis state available")

            result.state = state

            # Step 3: Save result to JSON using working_title if available - Claude Generated
            if hasattr(state, 'working_title') and state.working_title:
                # Use working title for filename
                output_filename = f"{state.working_title}.json"
            else:
                # Fallback to safe source filename
                output_filename = source.get_safe_filename()

            output_file = self.output_dir / output_filename
            self._save_result(state, str(output_file))

            result.output_file = str(output_file)
            result.success = True

            self.logger.info(f"✓ [{current_idx}/{total}] Completed: {output_file}")

        except Exception as e:
            result.error_message = str(e)
            self.logger.error(f"✗ [{current_idx}/{total}] Failed: {e}")

        result.end_time = datetime.now().isoformat()
        return result

    def _resolve_source_to_text(self, source: BatchSource) -> Optional[str]:
        """
        Resolve a batch source to input text

        Claude Generated
        """
        if source.source_type == SourceType.DOI:
            # Use doi_resolver
            return resolve_input_to_text(source.source_value, self.logger)

        elif source.source_type == SourceType.TXT:
            # Read text file
            with open(source.source_value, 'r', encoding='utf-8') as f:
                return f.read()

        elif source.source_type == SourceType.PDF:
            # PDF extraction with LLM fallback for poor quality - Claude Generated
            try:
                import PyPDF2

                # Try PyPDF2 first (fast method)
                with open(source.source_value, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text_parts = []
                    page_count = len(reader.pages)

                    self.logger.info(f"Extracting text from PDF: {page_count} pages")

                    for page_num, page in enumerate(reader.pages, 1):
                        try:
                            text = page.extract_text()
                            if text:
                                text_parts.append(text)
                        except Exception as e:
                            self.logger.warning(f"Failed to extract page {page_num}: {e}")

                    extracted_text = "\n\n".join(text_parts)

                    # Quality check: if text is very short or poor quality, use LLM OCR
                    if len(extracted_text.strip()) < 100 or self._is_poor_quality_text(extracted_text):
                        self.logger.warning(
                            f"PyPDF2 extraction quality poor ({len(extracted_text)} chars, quality check failed), "
                            "attempting LLM OCR..."
                        )

                        try:
                            from ..utils.pipeline_utils import execute_input_extraction

                            extracted_text, source_info, method = execute_input_extraction(
                                llm_service=self.pipeline_executor.alima_manager.llm_service,
                                input_source=source.source_value,
                                input_type="pdf",
                                stream_callback=None,
                                logger=self.logger
                            )

                            self.logger.info(f"LLM OCR completed: {method}, {len(extracted_text)} characters")

                        except Exception as llm_error:
                            self.logger.warning(f"LLM OCR failed: {llm_error}, using PyPDF2 result")
                            # Fall back to PyPDF2 result even if poor quality
                            if not extracted_text.strip():
                                raise ValueError("Both PyPDF2 and LLM OCR failed to extract text")

                    if not extracted_text.strip():
                        raise ValueError("PDF text extraction resulted in empty text")

                    self.logger.info(f"Extracted {len(extracted_text)} characters from PDF")
                    return extracted_text

            except ImportError:
                # PyPDF2 not available, try LLM OCR as fallback
                self.logger.warning("PyPDF2 not installed, attempting LLM OCR...")

                try:
                    from ..utils.pipeline_utils import execute_input_extraction

                    extracted_text, source_info, method = execute_input_extraction(
                        llm_service=self.pipeline_executor.alima_manager.llm_service,
                        input_source=source.source_value,
                        input_type="pdf",
                        stream_callback=None,
                        logger=self.logger
                    )

                    self.logger.info(f"LLM OCR completed: {method}, {len(extracted_text)} characters")
                    return extracted_text

                except Exception as e:
                    raise NotImplementedError(
                        f"PDF processing requires PyPDF2 OR vision-capable LLM. "
                        f"Install PyPDF2 with: pip install PyPDF2. LLM OCR error: {e}"
                    )

            except Exception as e:
                self.logger.error(f"PDF extraction failed: {e}")
                raise RuntimeError(f"Failed to extract text from PDF: {e}")

        elif source.source_type == SourceType.IMG:
            # Image analysis via LLM Vision models - Claude Generated
            try:
                from ..utils.pipeline_utils import execute_input_extraction

                self.logger.info(f"Analyzing image via Vision model: {source.source_value}")

                # Use existing image extraction pipeline from pipeline_utils
                extracted_text, source_info, extraction_method = execute_input_extraction(
                    llm_service=self.pipeline_executor.alima_manager.llm_service,
                    input_source=source.source_value,
                    input_type="image",
                    stream_callback=None,  # No streaming in batch mode
                    logger=self.logger
                )

                if not extracted_text or not extracted_text.strip():
                    raise ValueError("Image analysis resulted in empty text")

                self.logger.info(f"Image analysis completed: {extraction_method}, {len(extracted_text)} characters")
                return extracted_text

            except Exception as e:
                self.logger.error(f"Image analysis failed: {e}")
                raise RuntimeError(f"Failed to analyze image: {e}")

        elif source.source_type == SourceType.URL:
            # URL web scraping - Claude Generated
            try:
                import requests
                from bs4 import BeautifulSoup

                self.logger.info(f"Fetching URL: {source.source_value}")

                # Fetch with timeout and proper headers
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(source.source_value, timeout=30, headers=headers)
                response.raise_for_status()

                # Parse HTML
                soup = BeautifulSoup(response.content, 'html.parser')

                # Remove unwanted tags
                for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    tag.decompose()

                # Try to find main content (heuristic approach)
                main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')

                if main_content:
                    text = main_content.get_text(separator='\n', strip=True)
                else:
                    # Fallback: get all text from body
                    body = soup.find('body')
                    text = body.get_text(separator='\n', strip=True) if body else soup.get_text(separator='\n', strip=True)

                # Clean up excessive whitespace
                import re
                text = re.sub(r'\n\s*\n+', '\n\n', text)
                text = re.sub(r' +', ' ', text)

                if not text or len(text.strip()) < 50:
                    raise ValueError(f"URL scraping resulted in too little text ({len(text)} chars)")

                self.logger.info(f"URL scraping completed: {len(text)} characters extracted")
                return text

            except ImportError as e:
                self.logger.error("Required libraries not installed. Install with: pip install requests beautifulsoup4")
                raise NotImplementedError("URL processing requires: pip install requests beautifulsoup4")
            except requests.RequestException as e:
                self.logger.error(f"Failed to fetch URL: {e}")
                raise RuntimeError(f"Failed to fetch URL: {e}")
            except Exception as e:
                self.logger.error(f"URL scraping failed: {e}")
                raise RuntimeError(f"Failed to scrape URL: {e}")

        return None

    def _save_result(self, state: KeywordAnalysisState, filepath: str):
        """Save KeywordAnalysisState to JSON file - Claude Generated"""
        state_dict = PipelineJsonManager.task_state_to_dict(state)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, indent=2, ensure_ascii=False)

    def get_batch_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the batch - Claude Generated"""
        if not self.batch_state:
            return {}

        total = self.batch_state.total_sources
        processed = len(self.batch_state.processed_sources)
        failed = len(self.batch_state.failed_sources)
        success = processed - failed

        return {
            "total_sources": total,
            "processed": processed,
            "successful": success,
            "failed": failed,
            "success_rate": (success / processed * 100) if processed > 0 else 0,
            "start_time": self.batch_state.start_time,
            "end_time": self.batch_state.end_time,
            "output_dir": str(self.output_dir),
        }

    def _apply_batch_config(self, base_config, batch_config_dict: Dict[str, Any]):
        """
        Apply batch configuration overrides to base pipeline config

        Claude Generated
        """
        from ..core.pipeline_manager import PipelineConfig
        from ..utils.config_models import PipelineStepConfig, TaskType as UnifiedTaskType

        # If batch config has step_configs, apply them
        if "step_configs" in batch_config_dict:
            for step_id, step_params in batch_config_dict["step_configs"].items():
                if step_id in base_config.step_configs:
                    # Update existing step config
                    step_config = base_config.step_configs[step_id]

                    if "provider" in step_params:
                        step_config.provider = step_params["provider"]
                    if "model" in step_params:
                        step_config.model = step_params["model"]
                    if "task" in step_params:
                        step_config.task = step_params["task"]
                    if "temperature" in step_params:
                        step_config.temperature = step_params["temperature"]
                    if "top_p" in step_params:
                        step_config.top_p = step_params["top_p"]
                    if "enabled" in step_params:
                        step_config.enabled = step_params["enabled"]

        return base_config

    def _is_poor_quality_text(self, text: str) -> bool:
        """
        Check if extracted text is of poor quality - Claude Generated

        Uses simple heuristic: high ratio of non-alphanumeric characters suggests OCR issues
        """
        if not text:
            return True

        # Count alphanumeric and whitespace characters
        alphanumeric_and_space = sum(c.isalnum() or c.isspace() for c in text)
        ratio = alphanumeric_and_space / len(text) if len(text) > 0 else 0

        # If less than 50% alphanumeric+space, consider it poor quality
        return ratio < 0.5
