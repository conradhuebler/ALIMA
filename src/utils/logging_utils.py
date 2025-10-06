"""
Centralized Logging Configuration for ALIMA Application
Claude Generated - Unified logging system with 4 verbosity levels
"""

import logging
import sys
from typing import Optional


# Global log level tracker for result printing
_current_log_level: int = 1


def setup_logging(level: int = 1, log_file: str = "alima.log") -> None:
    """
    Configure centralized logging for ALIMA application.
    Claude Generated

    Args:
        level: Verbosity level (0-3)
            - 0 (Quiet): Only final results, no logs
            - 1 (Normal): INFO level - default
            - 2 (Debug): DEBUG level for ALIMA code
            - 3 (Verbose): DEBUG + third-party libraries debug
        log_file: Path to log file (default: alima.log)
    """
    global _current_log_level
    _current_log_level = level

    # Map verbosity levels to logging levels
    if level == 0:
        # Quiet mode - disable all standard logging
        log_level = logging.CRITICAL + 1
    elif level == 1:
        # Normal mode - INFO and above
        log_level = logging.INFO
    elif level == 2:
        # Debug mode - ALIMA debug only
        log_level = logging.DEBUG
    elif level == 3:
        # Verbose debug - ALIMA + third-party debug
        log_level = logging.DEBUG
    else:
        # Default to INFO if invalid level
        log_level = logging.INFO
        level = 1
        _current_log_level = 1

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (persistent log)
    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        # If file logging fails, just use console
        print(f"Warning: Could not create log file {log_file}: {e}", file=sys.stderr)

    # Configure third-party library logging for level 3 (Verbose)
    if level < 3:
        # Suppress noisy third-party libraries at levels 0-2
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
    else:
        # At level 3, show everything including third-party DEBUG
        logging.getLogger("requests").setLevel(logging.DEBUG)
        logging.getLogger("urllib3").setLevel(logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logging.getLogger("httpcore").setLevel(logging.DEBUG)

    # Log the logging configuration itself (except in quiet mode)
    if level > 0:
        root_logger.info(f"Logging initialized: Level {level} ({'Quiet' if level == 0 else 'Normal' if level == 1 else 'Debug' if level == 2 else 'Verbose'})")


def print_result(*args, sep: str = " ", end: str = "\n", file=None, flush: bool = False) -> None:
    """
    Print final results to output, respecting quiet mode.
    Claude Generated

    This function should be used for printing final, user-facing results
    (e.g., keyword lists, analysis outputs). It bypasses logging in quiet mode.

    Args:
        *args: Values to print
        sep: String separator between values (default: " ")
        end: String appended after last value (default: "\\n")
        file: File object to write to (default: sys.stdout)
        flush: Force flush output (default: False)
    """
    # Always print results to stdout, regardless of log level
    # Results are what the user explicitly requested
    if file is None:
        file = sys.stdout

    print(*args, sep=sep, end=end, file=file, flush=flush)

    # At non-quiet levels, also log the result for file logging
    if _current_log_level > 0:
        result_str = sep.join(str(arg) for arg in args)
        logging.getLogger("alima.results").info(result_str)


def get_current_log_level() -> int:
    """
    Get the current log level.
    Claude Generated

    Returns:
        Current log level (0-3)
    """
    return _current_log_level


def set_log_level(level: int) -> None:
    """
    Update log level dynamically.
    Claude Generated

    Args:
        level: New verbosity level (0-3)
    """
    setup_logging(level)
