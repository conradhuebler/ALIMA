# Refactoring Plan: Unified Logging

This document outlines the plan to refactor the logging and output mechanisms in the ALIMA application to create a unified, configurable, and consistent logging strategy.

## Current Situation

- **Mixed Output**: The application currently uses a mix of the standard `logging` module and `print()` statements for output. This leads to inconsistent output formatting and makes it difficult to control verbosity.
- **Decentralized Configuration**: Logging is configured independently in `src/alima_gui.py` and `src/alima_cli.py`. Additionally, configuration settings in `config.json` (`debug` and `log_level`) are not fully utilized in a centralized way.
- **Limited Control**: The current system primarily supports `INFO` and `DEBUG` levels, lacking the granular 0-3 verbosity levels requested.
- **GUI Errors**: The GUI uses `QMessageBox` for error popups, which is separate from the logging system.

## Goals

1.  **Unify Output**: Consolidate all application output (status messages, errors, debug information) through the `logging` module.
2.  **Centralize Configuration**: Create a single point of configuration for logging that is respected by both the CLI and the GUI.
3.  **Granular Verbosity Control**: Implement a 0-3 level verbosity system that maps to standard logging levels.
4.  **Separate Results from Logs**: Ensure that at verbosity level 0, only the final, content-rich results are printed to standard output, while all other logging is suppressed.

## Detailed Plan

### Step 1: Create a Central Logging Utility

1.  **Create `src/utils/logging_utils.py`**: This new module will be the central hub for all logging configuration.
2.  **Implement `setup_logging(level: int)` function**:
    - This function will configure the root logger for the entire application.
    - It will accept an integer `level` from 0 to 3.
    - **Level 0 (Quiet)**: Set the root logger's level to `logging.CRITICAL + 1`. This effectively disables all standard logging calls. A separate mechanism will be needed for printing final results (see Step 4).
    - **Level 1 (Normal - Default)**: Set level to `logging.INFO`. This will show informational messages, warnings, and errors.
    - **Level 2 (Debug)**: Set level to `logging.DEBUG`. This will show detailed debugging information from the ALIMA application code.
    - **Level 3 (Verbose Debug)**: Also set level to `logging.DEBUG`, but additionally set the log level for noisy third-party libraries (like `requests`, `urllib3`) to `DEBUG` to trace HTTP requests and other external interactions.
    - The function will configure a `StreamHandler` (for console output) and a `FileHandler` (for `alima.log`).

### Step 2: Integrate the Central Logging Utility

1.  **Refactor `src/alima_cli.py`**:
    - Remove the local `logging.basicConfig()` call.
    - Add a `--log-level` argument that accepts an integer from 0 to 3 (defaulting to 1).
    - At the very beginning of the `main()` function, call `setup_logging()` from the new utility module, passing the value from the `--log-level` argument.

2.  **Refactor `src/alima_gui.py`**:
    - Remove the local `setup_logging()` function and its call.
    - In the `main()` function, call the new central `setup_logging()` function. The level will initially be hardcoded (e.g., to 1) or read from the `config.json`.
    - **Future GUI Enhancement**: A setting will be added to the `ComprehensiveSettingsDialog` to allow the user to change the log level, which will then call `setup_logging()` again to apply the new level dynamically.

### Step 3: Replace `print()` with `logger`

This is the most extensive part of the refactoring.

1.  **Identify Logging `print()`s**: Go through the codebase, particularly in the CLI and core logic, and identify `print()` statements that are used for status updates, progress information, or debugging.
2.  **Replace with `logger` calls**:
    - `print(f"Starting step: {step.name}")` becomes `logger.info(f"Starting step: {step.name}")`.
    - `print(f"Debug info: {data}")` becomes `logger.debug(f"Debug info: {data}")`.
    - `print(f"Error: {e}")` becomes `logger.error(f"Error: {e}")`.
3.  **Keep Result `print()`s**: `print()` statements that are responsible for printing the final, machine-readable or user-facing results (e.g., the list of keywords in the CLI) must be identified and handled separately.

### Step 4: Handle Final Result Output (Quiet Mode)

To achieve the goal of Level 0 (Quiet), where only results are shown, we need to distinguish between logging output and result output.

1.  **Create a dedicated result-printing function**, e.g., `print_result(*args, **kwargs)`.
2.  This function will check the current log level. If the level is 0, it will print to `sys.stdout`. If the level is greater than 0, it could either print or log at an `INFO` level.
3.  Replace all `print()` statements that output the final results with calls to this new `print_result()` function.

By following this plan, we will have a robust, centralized, and flexible logging system that meets the specified requirements.
