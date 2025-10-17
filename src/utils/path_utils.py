#!/usr/bin/env python3
"""
Path Resolution Utilities - Claude Generated
Provides utilities for resolving absolute and relative paths in ALIMA configuration.
"""

import os
from pathlib import Path
from typing import Union


def get_project_root() -> Path:
    """
    Get the ALIMA project root directory - Claude Generated

    The project root is identified as the directory containing the 'src/' subdirectory.
    This function works regardless of the current working directory.

    Returns:
        Path: Absolute path to project root directory

    Raises:
        RuntimeError: If project root cannot be determined
    """
    # Start from this file's location
    current_file = Path(__file__).resolve()

    # Navigate upwards to find the directory containing 'src'
    # This file is in: <project_root>/src/utils/path_utils.py
    # So we go up 2 levels: utils -> src -> project_root
    project_root = current_file.parent.parent.parent

    # Verify that this is indeed the project root by checking for 'src' directory
    if not (project_root / 'src').is_dir():
        raise RuntimeError(
            f"Could not determine project root. Expected 'src' directory at {project_root}"
        )

    return project_root


def resolve_path(path_str: Union[str, Path]) -> str:
    """
    Resolve a path from configuration to an absolute path - Claude Generated

    Handles three path types:
    1. Absolute paths: Returned unchanged
    2. Home directory paths (~): Expanded to user's home directory
    3. Relative paths: Resolved relative to project root

    Args:
        path_str: Path string from configuration (can be absolute or relative)

    Returns:
        str: Absolute path as string

    Examples:
        >>> resolve_path("/absolute/path/file.db")
        '/absolute/path/file.db'

        >>> resolve_path("~/alima/file.db")
        '/home/user/alima/file.db'

        >>> resolve_path("prompts.json")
        '/path/to/ALIMA/prompts.json'

        >>> resolve_path("data/cache.db")
        '/path/to/ALIMA/data/cache.db'
    """
    if not path_str:
        raise ValueError("Path string cannot be empty")

    # Convert to Path object for consistent handling
    path = Path(path_str)

    # Case 1: Already an absolute path
    if path.is_absolute():
        return str(path)

    # Case 2: Home directory path (starts with ~)
    if str(path_str).startswith('~'):
        return str(Path(path_str).expanduser())

    # Case 3: Relative path - resolve relative to project root
    project_root = get_project_root()
    resolved_path = project_root / path

    return str(resolved_path)


def validate_file_path(path_str: Union[str, Path], must_exist: bool = False) -> tuple[bool, str]:
    """
    Validate a file path and provide feedback - Claude Generated

    Args:
        path_str: Path to validate
        must_exist: If True, path must point to an existing file

    Returns:
        Tuple of (is_valid, message)

    Examples:
        >>> validate_file_path("/tmp/test.json")
        (True, "Path is valid")

        >>> validate_file_path("/tmp/test.json", must_exist=True)
        (False, "File does not exist: /tmp/test.json")
    """
    try:
        resolved = resolve_path(path_str)
        path = Path(resolved)

        # Check if parent directory exists (for new files)
        if not path.parent.exists():
            return False, f"Parent directory does not exist: {path.parent}"

        # Check if file exists (if required)
        if must_exist and not path.exists():
            return False, f"File does not exist: {path}"

        # Check if path is accessible (try to check permissions)
        if path.exists() and not os.access(path, os.R_OK):
            return False, f"File is not readable: {path}"

        return True, "Path is valid"

    except Exception as e:
        return False, f"Path validation error: {e}"


def ensure_directory_exists(path_str: Union[str, Path]) -> bool:
    """
    Ensure that the directory for a file path exists - Claude Generated

    Creates parent directories if they don't exist.

    Args:
        path_str: File path (directory will be created for its parent)

    Returns:
        bool: True if directory exists or was created successfully
    """
    try:
        resolved = resolve_path(path_str)
        path = Path(resolved)

        # Create parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        return True

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to create directory for {path_str}: {e}")
        return False
