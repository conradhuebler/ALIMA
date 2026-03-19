# Setup Command Handler for ALIMA CLI
# Claude Generated - Extracted from alima_cli.py
"""
Handlers for setup commands:
    - setup: Run first-start setup wizard
"""

import sys
import logging
from pathlib import Path


def handle_setup(args, logger: logging.Logger):
    """Handle 'setup' command - Run CLI setup wizard.

    Args:
        args: Parsed command-line arguments with:
            - skip_gnd: Whether to skip GND database download option
            - force: Force setup wizard even if config exists
        logger: Logger instance
    """
    from src.utils.cli_setup_wizard import run_cli_setup_wizard

    success = run_cli_setup_wizard(force=args.force)
    sys.exit(0 if success else 1)
