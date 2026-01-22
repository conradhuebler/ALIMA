#!/usr/bin/env python3
# ALIMA CLI - Backward Compatibility Wrapper
# Claude Generated - Refactored CLI implementation
"""
ALIMA Command Line Interface - Backward Compatibility Wrapper

This file maintains backward compatibility with the original alima_cli.py entry point.
The actual CLI implementation has been refactored into modular components in src/cli/.

Original monolithic implementation (3062 lines) has been split into:
    - src/cli/main.py: Argument parser and command dispatcher
    - src/cli/commands/: Command handlers organized by functionality
    - src/cli/formatters/: Output formatting utilities

This wrapper simply imports and calls the new modular implementation.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cli.main import main


if __name__ == "__main__":
    main()
