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
*   **Details:**
    *   Each task (e.g., `abstract_analysis`, `keywords_verification`, `dk_classification`) will have its own section.
    *   Include fields for:
        *   `task_name`: Unique identifier for the task.
        *   `status`: (e.g., `pending`, `in_progress`, `completed`, `failed`).
        *   `input_data`: References to or inline content of input (e.g., abstract text, initial keywords).
        *   `output_data`: Results of the task (e.g., extracted keywords, GND IDs, DK classifications).
        *   `configuration`: Parameters used for the task (e.g., selected prompt, model, temperature, chunking settings).
        *   `timestamp`: When the step was last processed.
        *   `log`: Relevant log messages for the step.
    *   Consider a top-level structure for a "project" or "session" that contains a sequence of tasks.

### 2.2. Refactor Core Logic into Modular, Reusable Components

*   **Purpose:** Decouple the processing logic from the GUI elements.
*   **Details:**
    *   Create a `core_tasks` module (or similar) containing functions/classes for each processing step (e.g., `analyze_abstract`, `verify_keywords`, `classify_dk`).
    *   These functions should accept input data and configuration parameters (potentially loaded from JSON) and return output data (to be saved to JSON).
    *   Ensure these components do not have direct dependencies on PyQt6 UI elements.

### 2.3. Develop a CLI Interface

*   **Purpose:** Enable execution and management of tasks from the command line.
*   **Details:**
    *   Implement a main CLI script (`alima_cli.py` or similar).
    *   Commands for:
        *   `alima_cli.py run <json_workflow_file>`: Execute a predefined workflow.
        *   `alima_cli.py start <task_name> --input <file> --config <file>`: Start a specific task.
        *   `alima_cli.py continue <json_state_file>`: Resume a workflow from a saved state.
        *   `alima_cli.py list-tasks`: List available tasks.
        *   `alima_cli.py show-config <task_name>`: Display default/current config for a task.
    *   Input/output handling for CLI (reading from files, printing to console).

### 2.4. Implement Save/Load State Mechanism

*   **Purpose:** Persist and restore the state of a processing session.
*   **Details:**
    *   Functions to serialize the current task/workflow state into the defined JSON schema.
    *   Functions to deserialize a JSON state file back into the application's data structures, allowing the GUI or CLI to pick up where it left off.

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

## 4. Remaining Tasks (What's Still Needed)

*   **Core Logic Decoupling:**
    *   Extract the core processing logic from `AbstractTab` and other UI-specific classes into a separate, UI-agnostic layer.
    *   Define clear interfaces for these core functions (inputs, outputs).
*   **JSON Schema Definition:**
    *   Formalize the JSON schema for representing task states and workflows.
*   **CLI Implementation:**
    *   Develop the `alima_cli.py` script with the commands outlined in section 2.3.
    *   Implement argument parsing for CLI commands.
*   **State Management:**
    *   Implement the JSON serialization/deserialization for saving and loading task states.
    *   Integrate this save/load mechanism into both the GUI and the future CLI.
*   **Error Handling & Robustness:**
    *   Further refine error handling across the application, especially for API calls and file operations.
    *   Address the reported issue of keywords not being found in `AbstractTab`'s exact match, despite being present in the `keywords_edit` field. This indicates a potential bug in `parse_keywords_from_list` or the exact matching regex.
*   **Testing:**
    *   Develop comprehensive unit and integration tests for both core logic and CLI functionality.
*   **Documentation:**
    *   Update `README.md` and other documentation to reflect the new architecture and usage.
