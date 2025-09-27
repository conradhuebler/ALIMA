#!/usr/bin/env python3
"""
Model Comparison Dialog - A/B Testing for Provider/Model Combinations
Allows users to test and compare different configurations for pipeline steps.
Claude Generated
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QGroupBox,
    QGridLayout, QLabel, QComboBox, QPushButton, QTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar, QMessageBox, QSplitter,
    QCheckBox, QSpinBox, QDoubleSpinBox, QScrollArea, QFrame, QLineEdit
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QFont, QPalette, QTextDocument, QTextCursor

from ..utils.config_models import (
    PipelineStepConfig,
    PipelineMode,
    TaskType as UnifiedTaskType
)
from ..utils.smart_provider_selector import SmartProviderSelector, TaskType
from ..core.alima_manager import AlimaManager
from ..llm.llm_service import LlmService


@dataclass
class ComparisonTest:
    """A single comparison test configuration - Claude Generated"""
    test_id: str
    name: str
    description: str
    configurations: List[PipelineStepConfig]
    test_input: str
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass 
class ComparisonResult:
    """Result of a comparison test - Claude Generated"""
    test_id: str
    config_id: str
    provider: str
    model: str
    input_text: str
    output_text: str
    execution_time: float
    success: bool
    error_message: str = ""
    token_count: Optional[int] = None
    cost_estimate: Optional[float] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class ComparisonTestWorker(QThread):
    """Worker thread for running comparison tests - Claude Generated"""
    
    test_progress = pyqtSignal(str, int, int)  # test_name, current, total
    result_ready = pyqtSignal(str, ComparisonResult)  # config_name, result
    all_tests_completed = pyqtSignal()
    
    def __init__(self, test: ComparisonTest, alima_manager: AlimaManager):
        super().__init__()
        self.test = test
        self.alima_manager = alima_manager
        self.logger = logging.getLogger(__name__)
        self.should_stop = False
    
    def stop_tests(self):
        """Stop running tests - Claude Generated"""
        self.should_stop = True
    
    def run(self):
        """Run comparison tests for all configurations - Claude Generated"""
        total_configs = len(self.test.configurations)
        
        for i, config in enumerate(self.test.configurations):
            if self.should_stop:
                break
                
            config_name = f"{config.provider}/{config.model}" if config.provider else f"Smart Mode"
            self.test_progress.emit(config_name, i + 1, total_configs)
            
            # Run test with this configuration
            result = self._run_single_test(config, config_name)
            self.result_ready.emit(config_name, result)
            
            # Small delay between tests
            self.msleep(500)
        
        if not self.should_stop:
            self.all_tests_completed.emit()
    
    def _run_single_test(self, config: PipelineStepConfig, config_name: str) -> ComparisonResult:
        """Run a single test with given configuration - Claude Generated"""
        start_time = time.time()
        
        try:
            # Convert step config to alima manager parameters
            if config.mode == PipelineMode.SMART:
                # Use smart provider selection
                # For now, simulate smart selection
                provider = "ollama"  # Smart selection result
                model = "cogito:32b"
                
            else:
                # Use manual configuration
                provider = config.provider or "ollama"
                model = config.model or "cogito:32b"
            
            # Simulate LLM call (in real implementation, this would call the actual LLM)
            self.msleep(1000 + (len(self.test.test_input) // 10))  # Simulate processing time
            
            execution_time = time.time() - start_time
            
            # Generate simulated output based on configuration
            output_text = self._generate_simulated_output(provider, model, self.test.test_input)
            
            return ComparisonResult(
                test_id=self.test.test_id,
                config_id=config.step_id,
                provider=provider,
                model=model,
                input_text=self.test.test_input,
                output_text=output_text,
                execution_time=execution_time,
                success=True,
                token_count=len(output_text.split()),
                cost_estimate=self._estimate_cost(provider, len(output_text.split()))
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ComparisonResult(
                test_id=self.test.test_id,
                config_id=config.step_id,
                provider=config.provider or "unknown",
                model=config.model or "unknown", 
                input_text=self.test.test_input,
                output_text="",
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _generate_simulated_output(self, provider: str, model: str, input_text: str) -> str:
        """Generate simulated LLM output for testing - Claude Generated"""
        # This is a simulation - in real implementation, this would call the actual LLM
        base_keywords = ["Bibliotheksmanagement", "Digitale Medien", "Katalogisierung", "Metadaten"]
        
        # Simulate different outputs based on provider/model
        if "gemini" in provider.lower():
            output = f"Gemini-{model} Analysis:\n" + ", ".join(base_keywords[:3])
        elif "cogito:32b" in model.lower():
            output = f"Cogito-32B Analysis:\n" + ", ".join(base_keywords)
        elif "fast" in model.lower() or "14b" in model.lower():
            output = f"Fast Analysis:\n" + ", ".join(base_keywords[:2])
        else:
            output = f"Standard Analysis ({provider}/{model}):\n" + ", ".join(base_keywords[:3])
        
        # Add simulated reasoning
        output += f"\n\nAnalysis Quality: Based on '{input_text[:50]}...'"
        output += f"\nProvider Performance: {provider} with {model}"
        
        return output
    
    def _estimate_cost(self, provider: str, token_count: int) -> float:
        """Estimate cost for API call - Claude Generated"""
        # Simplified cost estimation (in real implementation, use actual pricing)
        cost_per_1k_tokens = {
            "gemini": 0.0015,
            "openai": 0.002,
            "anthropic": 0.003,
            "ollama": 0.0,  # Local
            "chatai": 0.001
        }
        
        base_cost = cost_per_1k_tokens.get(provider.lower(), 0.002)
        return (token_count / 1000) * base_cost


class ModelComparisonDialog(QDialog):
    """
    Model Comparison Dialog for A/B Testing different provider/model configurations
    Claude Generated
    """
    
    def __init__(self, config_manager=None, alima_manager=None, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.alima_manager = alima_manager
        self.logger = logging.getLogger(__name__)
        
        self.setWindowTitle("🧪 Model Comparison & A/B Testing")
        self.setModal(False)  # Allow interaction with main window
        self.resize(1000, 700)
        
        # Test data
        self.current_test: Optional[ComparisonTest] = None
        self.test_results: List[ComparisonResult] = []
        self.test_worker: Optional[ComparisonTestWorker] = None
        
        self._setup_ui()
        self._load_default_configurations()
    
    def _setup_ui(self):
        """Setup the comparison dialog UI - Claude Generated"""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("<h1>🧪 Model Comparison & A/B Testing</h1>")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Main content tabs
        self.main_tabs = QTabWidget()
        layout.addWidget(self.main_tabs)
        
        # Test Setup Tab
        self.setup_tab = self._create_setup_tab()
        self.main_tabs.addTab(self.setup_tab, "🔧 Test Setup")
        
        # Results Tab
        self.results_tab = self._create_results_tab()
        self.main_tabs.addTab(self.results_tab, "📊 Results")
        
        # Analysis Tab
        self.analysis_tab = self._create_analysis_tab()
        self.main_tabs.addTab(self.analysis_tab, "📈 Analysis")
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.run_test_button = QPushButton("🚀 Run Comparison Test")
        self.run_test_button.clicked.connect(self._run_comparison_test)
        
        self.stop_test_button = QPushButton("⏹️ Stop Test")
        self.stop_test_button.clicked.connect(self._stop_test)
        self.stop_test_button.setEnabled(False)
        
        self.export_results_button = QPushButton("💾 Export Results")
        self.export_results_button.clicked.connect(self._export_results)
        
        self.close_button = QPushButton("❌ Close")
        self.close_button.clicked.connect(self.close)
        
        button_layout.addWidget(self.run_test_button)
        button_layout.addWidget(self.stop_test_button)
        button_layout.addStretch()
        button_layout.addWidget(self.export_results_button)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
    
    def _create_setup_tab(self) -> QWidget:
        """Create test setup tab - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Test Information
        info_group = QGroupBox("📋 Test Information")
        info_layout = QGridLayout(info_group)
        
        info_layout.addWidget(QLabel("Test Name:"), 0, 0)
        self.test_name_edit = QLineEdit("Provider Comparison Test")
        info_layout.addWidget(self.test_name_edit, 0, 1)
        
        info_layout.addWidget(QLabel("Description:"), 1, 0)
        self.test_description_edit = QLineEdit("Compare different provider/model combinations")
        info_layout.addWidget(self.test_description_edit, 1, 1)
        
        layout.addWidget(info_group)
        
        # Test Input
        input_group = QGroupBox("📝 Test Input")
        input_layout = QVBoxLayout(input_group)
        
        self.test_input_text = QTextEdit()
        self.test_input_text.setPlaceholderText(
            "Enter the text to be processed by all configurations for comparison.\n\n"
            "Example: 'This research examines the impact of digital transformation "
            "on library management systems and their integration with metadata standards.'"
        )
        self.test_input_text.setMaximumHeight(150)
        input_layout.addWidget(self.test_input_text)
        
        layout.addWidget(input_group)
        
        # Configuration Setup
        config_group = QGroupBox("⚙️ Test Configurations")
        config_layout = QVBoxLayout(config_group)
        
        # Configuration management buttons
        config_buttons = QHBoxLayout()
        
        self.add_smart_config_button = QPushButton("🤖 Add Smart Mode")
        self.add_smart_config_button.clicked.connect(self._add_smart_configuration)
        
        self.add_manual_config_button = QPushButton("⚙️ Add Manual Config")
        self.add_manual_config_button.clicked.connect(self._add_manual_configuration)
        
        self.remove_config_button = QPushButton("🗑️ Remove Selected")
        self.remove_config_button.clicked.connect(self._remove_configuration)
        
        config_buttons.addWidget(self.add_smart_config_button)
        config_buttons.addWidget(self.add_manual_config_button)
        config_buttons.addWidget(self.remove_config_button)
        config_buttons.addStretch()
        
        config_layout.addLayout(config_buttons)
        
        # Configuration table
        self.config_table = QTableWidget()
        self.config_table.setColumnCount(6)
        self.config_table.setHorizontalHeaderLabels([
            "Name", "Mode", "Provider", "Model", "Task", "Parameters"
        ])
        
        # Make table responsive
        header = self.config_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Name
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # Mode
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Provider
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)           # Model
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)  # Task
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)           # Parameters
        
        config_layout.addWidget(self.config_table)
        layout.addWidget(config_group)
        
        return widget
    
    def _create_results_tab(self) -> QWidget:
        """Create results display tab - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Progress display
        progress_group = QGroupBox("🔄 Test Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Ready to run tests")
        
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        
        layout.addWidget(progress_group)
        
        # Results table
        results_group = QGroupBox("📊 Test Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels([
            "Configuration", "Provider", "Model", "Success", "Time (s)", "Tokens", "Cost ($)", "Output Preview"
        ])
        
        # Make results table responsive
        results_header = self.results_table.horizontalHeader()
        results_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Config
        results_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)  # Provider
        results_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Model
        results_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  # Success
        results_header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)  # Time
        results_header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)  # Tokens
        results_header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)  # Cost
        results_header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)           # Output
        
        # Connect selection to detail view
        self.results_table.itemSelectionChanged.connect(self._on_result_selected)
        
        results_layout.addWidget(self.results_table)
        layout.addWidget(results_group)
        
        # Detailed output view
        detail_group = QGroupBox("📄 Detailed Output")
        detail_layout = QVBoxLayout(detail_group)
        
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumHeight(200)
        
        detail_layout.addWidget(self.detail_text)
        layout.addWidget(detail_group)
        
        return widget
    
    def _create_analysis_tab(self) -> QWidget:
        """Create analysis and comparison tab - Claude Generated"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Summary statistics
        summary_group = QGroupBox("📈 Summary Statistics")
        summary_layout = QVBoxLayout(summary_group)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(150)
        
        summary_layout.addWidget(self.summary_text)
        layout.addWidget(summary_group)
        
        # Comparison matrix
        comparison_group = QGroupBox("🔀 Side-by-Side Comparison")
        comparison_layout = QVBoxLayout(comparison_group)
        
        self.comparison_table = QTableWidget()
        comparison_layout.addWidget(self.comparison_table)
        
        layout.addWidget(comparison_group)
        
        # Recommendations
        recommendations_group = QGroupBox("💡 Recommendations")
        recommendations_layout = QVBoxLayout(recommendations_group)
        
        self.recommendations_text = QTextEdit()
        self.recommendations_text.setReadOnly(True)
        self.recommendations_text.setMaximumHeight(100)
        
        recommendations_layout.addWidget(self.recommendations_text)
        layout.addWidget(recommendations_group)
        
        return widget
    
    def _load_default_configurations(self):
        """Load default test configurations - Claude Generated"""
        # Add some default configurations for testing
        default_configs = [
            PipelineStepConfig(
                step_id="smart_config",
                mode=PipelineMode.SMART,
                task_type=UnifiedTaskType.KEYWORDS,
                quality_preference="balanced"
            ),
            PipelineStepConfig(
                step_id="ollama_fast",
                mode=PipelineMode.ADVANCED,
                provider="ollama",
                model="cogito:14b",
                task="keywords"
            ),
            PipelineStepConfig(
                step_id="ollama_quality",
                mode=PipelineMode.ADVANCED,
                provider="ollama", 
                model="cogito:32b",
                task="keywords"
            )
        ]
        
        # Populate configuration table
        self.config_table.setRowCount(len(default_configs))
        for row, config in enumerate(default_configs):
            self._add_config_to_table(row, config)
    
    def _add_config_to_table(self, row: int, config: PipelineStepConfig):
        """Add configuration to table - Claude Generated"""
        # Name
        name = f"Config {row + 1}"
        if config.mode == PipelineMode.SMART:
            name += " (Smart)"
        elif config.provider and config.model:
            name += f" ({config.provider}/{config.model})"
        
        self.config_table.setItem(row, 0, QTableWidgetItem(name))
        
        # Mode
        mode_text = config.mode.value.title()
        self.config_table.setItem(row, 1, QTableWidgetItem(mode_text))
        
        # Provider
        provider_text = config.provider or "Auto"
        self.config_table.setItem(row, 2, QTableWidgetItem(provider_text))
        
        # Model
        model_text = config.model or "Auto"
        self.config_table.setItem(row, 3, QTableWidgetItem(model_text))
        
        # Task
        task_text = config.task or config.task_type.value if config.task_type else "Auto"
        self.config_table.setItem(row, 4, QTableWidgetItem(task_text))
        
        # Parameters
        params = []
        if config.temperature is not None:
            params.append(f"temp={config.temperature}")
        if config.top_p is not None:
            params.append(f"top_p={config.top_p}")
        
        params_text = ", ".join(params) if params else "Default"
        self.config_table.setItem(row, 5, QTableWidgetItem(params_text))
    
    def _add_smart_configuration(self):
        """Add a smart mode configuration - Claude Generated"""
        row = self.config_table.rowCount()
        self.config_table.setRowCount(row + 1)
        
        smart_config = PipelineStepConfig(
            step_id=f"smart_{row}",
            mode=PipelineMode.SMART,
            task_type=UnifiedTaskType.KEYWORDS,
            quality_preference="balanced"
        )
        
        self._add_config_to_table(row, smart_config)
    
    def _add_manual_configuration(self):
        """Add a manual configuration - Claude Generated"""
        # In a real implementation, this would open a configuration dialog
        row = self.config_table.rowCount()
        self.config_table.setRowCount(row + 1)
        
        manual_config = PipelineStepConfig(
            step_id=f"manual_{row}",
            mode=PipelineMode.ADVANCED,
            provider="gemini",
            model="gemini-2.0-flash-exp",
            task="keywords"
        )
        
        self._add_config_to_table(row, manual_config)
    
    def _remove_configuration(self):
        """Remove selected configuration - Claude Generated"""
        current_row = self.config_table.currentRow()
        if current_row >= 0:
            self.config_table.removeRow(current_row)
    
    def _run_comparison_test(self):
        """Run comparison test with all configurations - Claude Generated"""
        if self.config_table.rowCount() == 0:
            QMessageBox.warning(self, "No Configurations", "Please add at least one configuration to test.")
            return
        
        test_input = self.test_input_text.toPlainText().strip()
        if not test_input:
            QMessageBox.warning(self, "No Input", "Please enter test input text.")
            return
        
        # Create test object
        configurations = self._get_configurations_from_table()
        
        self.current_test = ComparisonTest(
            test_id=f"test_{int(time.time())}",
            name=self.test_name_edit.text(),
            description=self.test_description_edit.text(),
            configurations=configurations,
            test_input=test_input
        )
        
        # Setup UI for running test
        self.run_test_button.setEnabled(False)
        self.stop_test_button.setEnabled(True)
        self.main_tabs.setCurrentIndex(1)  # Switch to results tab
        
        # Clear previous results
        self.test_results.clear()
        self.results_table.setRowCount(0)
        self.detail_text.clear()
        
        # Start test worker
        if self.alima_manager:
            self.test_worker = ComparisonTestWorker(self.current_test, self.alima_manager)
        else:
            # Use mock alima manager for testing
            self.test_worker = ComparisonTestWorker(self.current_test, None)
        
        self.test_worker.test_progress.connect(self._on_test_progress)
        self.test_worker.result_ready.connect(self._on_result_ready)
        self.test_worker.all_tests_completed.connect(self._on_tests_completed)
        
        self.test_worker.start()
    
    def _get_configurations_from_table(self) -> List[PipelineStepConfig]:
        """Extract configurations from table - Claude Generated"""
        configurations = []
        
        for row in range(self.config_table.rowCount()):
            # This is simplified - in a real implementation, we'd store the actual configs
            mode_text = self.config_table.item(row, 1).text().lower()
            provider = self.config_table.item(row, 2).text()
            model = self.config_table.item(row, 3).text()
            task = self.config_table.item(row, 4).text()
            
            if mode_text == "smart":
                mode = PipelineMode.SMART
                provider = None
                model = None
            else:
                mode = PipelineMode.ADVANCED
                if provider == "Auto":
                    provider = None
                if model == "Auto":
                    model = None
            
            config = PipelineStepConfig(
                step_id=f"config_{row}",
                mode=mode,
                provider=provider,
                model=model,
                task=task if task != "Auto" else None,
                task_type=UnifiedTaskType.KEYWORDS
            )
            
            configurations.append(config)
        
        return configurations
    
    def _stop_test(self):
        """Stop running test - Claude Generated"""
        if self.test_worker:
            self.test_worker.stop_tests()
            self.test_worker.wait(5000)  # Wait up to 5 seconds
        
        self.run_test_button.setEnabled(True)
        self.stop_test_button.setEnabled(False)
        self.progress_label.setText("Test stopped by user")
    
    def _on_test_progress(self, config_name: str, current: int, total: int):
        """Handle test progress updates - Claude Generated"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"Testing {config_name} ({current}/{total})")
    
    def _on_result_ready(self, config_name: str, result: ComparisonResult):
        """Handle individual test result - Claude Generated"""
        self.test_results.append(result)
        
        # Add to results table
        row = self.results_table.rowCount()
        self.results_table.setRowCount(row + 1)
        
        # Configuration name
        self.results_table.setItem(row, 0, QTableWidgetItem(config_name))
        
        # Provider
        self.results_table.setItem(row, 1, QTableWidgetItem(result.provider))
        
        # Model  
        self.results_table.setItem(row, 2, QTableWidgetItem(result.model))
        
        # Success
        success_text = "✅ Success" if result.success else "❌ Failed"
        self.results_table.setItem(row, 3, QTableWidgetItem(success_text))
        
        # Execution time
        time_text = f"{result.execution_time:.2f}"
        self.results_table.setItem(row, 4, QTableWidgetItem(time_text))
        
        # Token count
        token_text = str(result.token_count) if result.token_count else "N/A"
        self.results_table.setItem(row, 5, QTableWidgetItem(token_text))
        
        # Cost estimate
        cost_text = f"${result.cost_estimate:.4f}" if result.cost_estimate else "N/A"
        self.results_table.setItem(row, 6, QTableWidgetItem(cost_text))
        
        # Output preview
        preview = result.output_text[:100] + "..." if len(result.output_text) > 100 else result.output_text
        preview = preview.replace('\n', ' ')
        self.results_table.setItem(row, 7, QTableWidgetItem(preview))
        
        # Color code based on success
        if not result.success:
            for col in range(8):
                item = self.results_table.item(row, col)
                if item:
                    item.setBackground(QPalette().color(QPalette.ColorRole.Base))
    
    def _on_tests_completed(self):
        """Handle completion of all tests - Claude Generated"""
        self.run_test_button.setEnabled(True)
        self.stop_test_button.setEnabled(False)
        self.progress_label.setText(f"Completed testing {len(self.test_results)} configurations")
        
        # Generate analysis
        self._generate_analysis()
        
        QMessageBox.information(self, "Tests Completed", 
                              f"Successfully completed comparison testing with {len(self.test_results)} configurations.")
    
    def _on_result_selected(self):
        """Handle result selection for detailed view - Claude Generated"""
        current_row = self.results_table.currentRow()
        if current_row >= 0 and current_row < len(self.test_results):
            result = self.test_results[current_row]
            
            detail_text = f"Configuration: {result.provider}/{result.model}\n"
            detail_text += f"Execution Time: {result.execution_time:.2f} seconds\n"
            detail_text += f"Success: {'Yes' if result.success else 'No'}\n"
            
            if result.error_message:
                detail_text += f"Error: {result.error_message}\n"
            
            detail_text += f"\nInput Text:\n{result.input_text}\n"
            detail_text += f"\nOutput Text:\n{result.output_text}"
            
            self.detail_text.setPlainText(detail_text)
    
    def _generate_analysis(self):
        """Generate analysis of test results - Claude Generated"""
        if not self.test_results:
            return
        
        successful_results = [r for r in self.test_results if r.success]
        
        # Summary statistics
        total_tests = len(self.test_results)
        successful_tests = len(successful_results)
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        if successful_results:
            avg_time = sum(r.execution_time for r in successful_results) / len(successful_results)
            avg_tokens = sum(r.token_count for r in successful_results if r.token_count) / len([r for r in successful_results if r.token_count])
            total_cost = sum(r.cost_estimate for r in successful_results if r.cost_estimate)
        else:
            avg_time = 0
            avg_tokens = 0
            total_cost = 0
        
        summary = f"📊 Test Summary:\n"
        summary += f"• Total Configurations: {total_tests}\n"
        summary += f"• Success Rate: {success_rate:.1f}% ({successful_tests}/{total_tests})\n"
        summary += f"• Average Execution Time: {avg_time:.2f} seconds\n"
        summary += f"• Average Token Count: {avg_tokens:.0f}\n"
        summary += f"• Total Estimated Cost: ${total_cost:.4f}\n\n"
        
        # Find best performing configuration
        if successful_results:
            fastest = min(successful_results, key=lambda r: r.execution_time)
            most_tokens = max(successful_results, key=lambda r: r.token_count or 0)
            cheapest = min(successful_results, key=lambda r: r.cost_estimate or float('inf'))
            
            summary += f"🏆 Performance Leaders:\n"
            summary += f"• Fastest: {fastest.provider}/{fastest.model} ({fastest.execution_time:.2f}s)\n"
            summary += f"• Most Detailed: {most_tokens.provider}/{most_tokens.model} ({most_tokens.token_count} tokens)\n"
            summary += f"• Most Cost-Effective: {cheapest.provider}/{cheapest.model} (${cheapest.cost_estimate:.4f})\n"
        
        self.summary_text.setPlainText(summary)
        
        # Generate recommendations
        recommendations = "💡 Recommendations:\n\n"
        
        if successful_results:
            if avg_time > 3.0:
                recommendations += "• Consider using faster models for time-sensitive applications\n"
            if total_cost > 0.01:
                recommendations += "• Consider local models (Ollama) to reduce costs\n"
            if success_rate < 100:
                recommendations += "• Check configurations that failed for compatibility issues\n"
            
            # Quality-based recommendations
            recommendations += "\n📈 Quality Insights:\n"
            recommendations += "• Compare output lengths and detail levels\n"
            recommendations += "• Consider A/B testing with real users for subjective quality\n"
            recommendations += "• Test with different input types and complexity levels\n"
        else:
            recommendations += "❌ All tests failed. Please check:\n"
            recommendations += "• Provider connectivity and API keys\n"
            recommendations += "• Model availability and names\n"
            recommendations += "• Input text format and length\n"
        
        self.recommendations_text.setPlainText(recommendations)
        
        # Switch to analysis tab
        self.main_tabs.setCurrentIndex(2)
    
    def _export_results(self):
        """Export test results to JSON - Claude Generated"""
        if not self.test_results:
            QMessageBox.information(self, "No Results", "No test results to export.")
            return
        
        # Create export data
        export_data = {
            "test_info": asdict(self.current_test) if self.current_test else {},
            "results": [asdict(result) for result in self.test_results],
            "summary": {
                "total_tests": len(self.test_results),
                "successful_tests": len([r for r in self.test_results if r.success]),
                "export_timestamp": datetime.now().isoformat()
            }
        }
        
        # Save to file (simplified - in real implementation, use file dialog)
        filename = f"comparison_results_{int(time.time())}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            QMessageBox.information(self, "Export Successful", 
                                  f"Results exported to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", 
                               f"Failed to export results:\n\n{str(e)}")


# Convenience function for opening comparison dialog
def show_model_comparison_dialog(config_manager=None, alima_manager=None, parent=None):
    """Show model comparison dialog - Claude Generated"""
    dialog = ModelComparisonDialog(config_manager, alima_manager, parent)
    dialog.show()
    return dialog