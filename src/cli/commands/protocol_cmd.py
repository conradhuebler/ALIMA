# Protocol Display Command Handler for ALIMA CLI
# Claude Generated - Extracted from alima_cli.py
"""
Handler for show-protocol command - displays pipeline results from JSON files.
"""

from typing import List
from src.cli.formatters.protocol_formatters import (
    display_protocol,
    display_protocol_compact,
    display_protocol_k10plus
)


def handle_show_protocol(args):
    """Handle 'show-protocol' command - Display pipeline results from JSON file.

    Args:
        args: Parsed command-line arguments with:
            - json_file: Path to JSON protocol file
            - steps: List of steps to display
            - format: Output format (detailed/compact/k10plus)
            - header: Whether to print CSV header (compact format only)
    """
    # Print CSV header if requested
    if args.header and args.format == "compact":
        print("filename,step,data")

    # Display protocol in requested format
    if args.format == "k10plus":
        display_protocol_k10plus(args.json_file)
    elif args.format == "compact":
        display_protocol_compact(args.json_file, args.steps)
    else:
        display_protocol(args.json_file, args.steps)
