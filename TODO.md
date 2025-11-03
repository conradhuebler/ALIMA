# ALIMA Project - Architectural Refactoring TODO

## 1. Overall Goal: Flexible GUI/CLI with Interchangeable, JSON-Formatted Steps

The primary objective is to re-architect ALIMA to operate seamlessly across both a Graphical User Interface (GUI) and a Command-Line Interface (CLI). This will be achieved by defining a standardized, JSON-based format for representing and saving the state of each processing step (tasks). This approach will enable:

*   **Interchangeability:** Users can start a task in the GUI, save its state, and resume or continue it from the CLI, and vice-versa.
*   **Automation:** Complex workflows can be defined and executed programmatically via the CLI using the JSON state files.
*   **Modularity:** Each processing step (e.g., `abstract_analysis`, `keywords_verification`, `dk_classification`) will be a self-contained, callable unit, independent of the UI.
*   **Flexibility:** Not every text needs to go through the full classification pipeline. Users can selectively enable/disable specific tasks (e.g., only abstract analysis, or only DK classification) via the CLI.

## 2. Steps for Implementation

### 2.1. Define a Standardized JSON Schema for Task States

*   **Purpose:** To represent the input, output, and configuration of each processing step in a machine-readable and interchangeable format.
*   **Details:** A basic JSON schema for `TaskState` and `AnalysisResult` has been defined in `src/core/data_models.py` to capture input, output, and configuration. This includes fields for `task_name`, `status`, `abstract_data`, `analysis_result` (containing `full_text`, `matched_keywords`, `gnd_systematic`), `prompt_config`, and various chunking parameters.

### 2.2. Refactor Core Logic into Modular, Reusable Components

*   **Purpose:** Decouple the processing logic from the GUI elements.
*   **Details:**
    *   Create a `core_tasks` module (or similar) containing functions/classes for each processing step (e.g., `analyze_abstract`, `verify_keywords`, `classify_dk`).
    *   These functions should accept input data and configuration parameters (potentially loaded from JSON) and return output data (to be saved to JSON).
    *   Ensure these components do not have direct dependencies on PyQt6 UI elements.

### 2.3. Develop a CLI Interface

*   **Purpose:** Enable execution and management of tasks from the command line.
*   **Details:** The main CLI script (`alima_cli.py`) has been refactored to use sub-commands. The `run` command is implemented for executing analysis tasks, and the `load-state` command is implemented for loading and displaying saved task states. Argument parsing for these commands is in place.

### 2.4. Implement Save/Load State Mechanism

*   **Purpose:** Persist and restore the state of a processing session.
*   **Details:** Functions to serialize the current task/workflow state into the defined JSON schema (`_task_state_to_dict` in `alima_cli.py`) and deserialize a JSON state file back into the application's data structures are implemented. The `run` command supports `--output-json` for saving, and a `load-state` command is available for loading.

## 3. Current Status (What Works So Far)

*   **GUI Application:** A functional PyQt6-based GUI is in place.
*   **LLM Integration:** Connection to various LLMs (Ollama, Gemini, OpenAI-compatible) is established.
*   **Core Processing (GUI-driven):**
    *   Abstract/Text input and basic analysis.
    *   Keyword input and initial extraction (including `final_list` and exact matching).
    *   PDF import for text and metadata.
    *   Prompt management: GUI for editing, adding, removing, and saving prompt configurations (`prompts.json`).
    *   Provider and model selection within the GUI.
    *   Basic chunking for large inputs.
*   **Git Update:** Functionality to update the application from Git.
*   **CLI Functionality:**
    *   `run` command for executing analysis tasks with various options.
    *   `load-state` command for loading and displaying saved task states from JSON files.
    *   JSON serialization and deserialization of `TaskState` objects.

## 4. Remaining Tasks (What's Still Needed)

*   **Core Logic Decoupling:**
    *   Extract the core processing logic from `AbstractTab` and other UI-specific classes into a separate, UI-agnostic layer.
    *   Define clear interfaces for these core functions (inputs, outputs).
*   **CLI Implementation:**
    *   Implement `start` and `continue` commands for the CLI.
    *   Implement `list-tasks` and `show-config` commands for the CLI.
*   **Error Handling & Robustness:**
    *   Further refine error handling across the application, especially for API calls and file operations.
    *   Address the reported issue of keywords not being found in `AbstractTab`'s exact match, despite being present in the `keywords_edit` field. This indicates a potential bug in `parse_keywords_from_list` or the exact matching regex.
*   **Testing:**
    *   Develop comprehensive unit and integration tests for both core logic and CLI functionality.
*   **Documentation:**
    *   Update `README.md` and other documentation to reflect the new architecture and usage.
*   **GUI Integration:**
    *   Integrate the new JSON-based state management into the GUI to enable saving and loading tasks from the GUI.

## 5. Reproducibility & Determinism Issues

**Status:** Analysis Complete - [See detailed report: `REPRODUCIBILITY_ANALYSIS.md`](REPRODUCIBILITY_ANALYSIS.md)

### Critical Issues (Must Fix)
1. **Temperature Override Bug** (`src/core/alima_manager.py:93`)
   - Explicit temperature parameters are ignored when using PromptService
   - Hardcoded values from `prompts.json` override all user input
   - Fix: Apply explicit parameters and respect parameter hierarchy

2. **Hardcoded Seed Values** (`prompts.json` - lines 26, 36, 46, 56, 68, 78, 92)
   - All initialization task variants have `"seed": "0"` hardcoded
   - Seeds are not configurable by any parameter
   - Fix: Make seeds configurable, add to configuration UI

3. **Gemini Provider Missing Seed Support** (`src/llm/llm_service.py:1143-1145`)
   - Gemini's `_generate_gemini()` doesn't include seed in generation_config
   - Gemini-based initialization is ALWAYS non-deterministic
   - Fix: Add seed parameter to Gemini generation_config

### High Priority Issues
4. **Dynamic Provider Selection** (`src/utils/pipeline_utils.py:67-97`)
   - SmartProviderSelector routes to different providers based on availability/latency
   - Same input may produce different results on different runs
   - Fix: Add `--deterministic` flag to disable smart selection

5. **Inconsistent Parameter Passing**
   - Different code paths have different parameter precedence
   - Template-path respects temperature, PromptService-path ignores it
   - Fix: Unified parameter handling across all paths

### Provider Reproducibility Matrix
| Provider | Temperature | Top_P | Seed | Status |
|----------|-------------|-------|------|--------|
| Gemini | ✅ | ✅ | ❌ | Non-deterministic |
| Ollama Native | ✅ | ✅ | ✅ | Model-dependent |
| OpenAI Compatible | ✅ | ✅ | ✅ | Newer models only |
| Anthropic | ✅ | ✅ | ✅ | Model-dependent |

### Implementation Priority
- [ ] Fix temperature override bug (alima_manager.py)
- [ ] Make seeds configurable (prompts.json + UI)
- [ ] Add seed support to Gemini provider
- [ ] Implement `--deterministic` mode for pipeline
- [ ] Create reproducibility test suite
