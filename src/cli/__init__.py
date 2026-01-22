# ALIMA CLI Module
# Claude Generated - Modular CLI implementation
"""
ALIMA Command Line Interface - Modular Structure

This package provides a modular CLI for the ALIMA application.
Commands are organized in separate modules for maintainability.

Modules:
    - commands/: Command handlers for each CLI command group
    - formatters/: Output formatting utilities
    - main: Entry point and argument parsing
"""

from .main import main

__all__ = ['main']
