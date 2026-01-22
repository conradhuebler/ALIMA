# Setup and Configuration Import Command Handlers for ALIMA CLI
# Claude Generated - Extracted from alima_cli.py
"""
Handlers for setup and configuration commands:
    - setup: Run first-start setup wizard
    - import-config: Import ALIMA configuration from directory
"""

import sys
import logging
from pathlib import Path


def handle_setup(args, logger: logging.Logger):
    """Handle 'setup' command - Run CLI setup wizard.

    Args:
        args: Parsed command-line arguments with:
            - skip_gnd: Whether to skip GND database download option
        logger: Logger instance
    """
    from src.utils.cli_setup_wizard import run_cli_setup_wizard

    success = run_cli_setup_wizard()
    sys.exit(0 if success else 1)


def handle_import_config(args, logger: logging.Logger):
    """Handle 'import-config' command - Import ALIMA configuration.

    Args:
        args: Parsed command-line arguments with:
            - source: Source directory containing config files
            - no_backup: Whether to skip creating backup
        logger: Logger instance
    """
    from src.utils.config_manager import ConfigManager

    print("🔍 Importing ALIMA configuration...")
    print(f"   Source: {args.source}")

    try:
        config_manager = ConfigManager()

        # Perform import
        create_backup = not args.no_backup
        success, message = config_manager.import_configuration(args.source, create_backup=create_backup)

        if success:
            print()
            print("✅ Configuration import successful!")
            print()
            print(f"📂 Configuration directory: {config_manager.config_file.parent}")
            print(f"📝 Config file: {config_manager.config_file}")
            print()
            print("📋 Imported files:")
            print(f"   * config.json")

            # Check what was imported
            prompts_file = config_manager.config_file.parent / "prompts.json"
            if prompts_file.exists():
                size_kb = prompts_file.stat().st_size / 1024
                print(f"   * prompts.json ({size_kb:.1f} KB)")

            db_file = config_manager.config_file.parent / "alima_knowledge.db"
            if db_file.exists():
                size_mb = db_file.stat().st_size / (1024 * 1024)
                print(f"   * alima_knowledge.db ({size_mb:.1f} MB)")

            print()
            print("🚀 ALIMA is now configured with the imported settings.")
            print("   You can start the application immediately.")
        else:
            print()
            print(message)

    except Exception as e:
        print(f"❌ Configuration import failed: {e}")
        logger.error(f"Configuration import error: {e}", exc_info=True)
