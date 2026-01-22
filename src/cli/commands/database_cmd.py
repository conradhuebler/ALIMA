# Database Management Command Handlers for ALIMA CLI
# Claude Generated - Extracted from alima_cli.py
"""
Handlers for database management commands:
    - db-config: Database configuration management
    - migrate-db: Database migration and backup
    - clear-cache: Clear cache database
    - dnb-import: Import DNB/GND data
"""

import logging
import sys
import os
import time
import tempfile
import gzip
import requests
import json
from pathlib import Path
from PyQt6.QtCore import QCoreApplication

from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
from src.utils.config_manager import ConfigManager


def handle_db_config(args, logger: logging.Logger):
    """Handle 'db-config' command - Database configuration management.

    Args:
        args: Parsed command-line arguments with db_action subcommand
        logger: Logger instance
    """
    # Ensure QCoreApplication exists for QtSql operations
    if not QCoreApplication.instance():
        app = QCoreApplication(sys.argv)

    config_manager = ConfigManager()

    if args.db_action == "show":
        handle_db_show(config_manager, logger)
    elif args.db_action == "paths":
        handle_db_paths(config_manager, logger)
    elif args.db_action == "test":
        handle_db_test(config_manager, logger)
    elif args.db_action == "set-sqlite":
        handle_db_set_sqlite(args, config_manager, logger)
    elif args.db_action == "set-mysql":
        handle_db_set_mysql(args, config_manager, logger)
    else:
        print("❌ No database action specified. Use 'show', 'test', 'set-sqlite', or 'set-mysql'")


def handle_db_show(config_manager: ConfigManager, logger: logging.Logger):
    """Show current database configuration."""
    try:
        config = config_manager.load_config()
        print("📊 Current Database Configuration:")
        print(f"   Type: {config.database.db_type}")
        if config.database.db_type == 'sqlite':
            print(f"   Path: {config.database.sqlite_path}")
        else:
            print(f"   Host: {config.database.host}:{config.database.port}")
            print(f"   Database: {config.database.database}")
            print(f"   Username: {config.database.username}")
            print(f"   Charset: {config.database.charset}")
            print(f"   SSL Disabled: {config.database.ssl_disabled}")
        print(f"   Connection Timeout: {config.database.connection_timeout}s")
        print(f"   Auto Create Tables: {config.database.auto_create_tables}")
    except Exception as e:
        logger.error(f"❌ Error showing database config: {e}")


def handle_db_paths(config_manager: ConfigManager, logger: logging.Logger):
    """Show OS-specific configuration paths."""
    try:
        config_info = config_manager.get_config_info()
        print(f"🖥️  Configuration Paths for {config_info['os']}:")
        print(f"   Project:  {config_info['project_config']}")
        print(f"   User:     {config_info['user_config']}")
        print(f"   System:   {config_info['system_config']}")
        print(f"   Legacy:   {config_info['legacy_config']}")
        print()

        # Show which files exist
        paths = [
            ("Project", config_info['project_config']),
            ("User", config_info['user_config']),
            ("System", config_info['system_config']),
            ("Legacy", config_info['legacy_config'])
        ]

        print("📁 File Status:")
        for name, path in paths:
            exists = Path(path).exists()
            status = "✅ EXISTS" if exists else "❌ NOT FOUND"
            print(f"   {name:8}: {status}")

    except Exception as e:
        logger.error(f"❌ Error showing config paths: {e}")


def handle_db_test(config_manager: ConfigManager, logger: logging.Logger):
    """Test database connection."""
    try:
        success, message = config_manager.test_database_connection()
        print(f"🔌 Database Connection Test: {'✅ SUCCESS' if success else '❌ FAILED'}")
        print(f"   {message}")
    except Exception as e:
        logger.error(f"❌ Error testing database connection: {e}")


def handle_db_set_sqlite(args, config_manager: ConfigManager, logger: logging.Logger):
    """Configure SQLite database."""
    try:
        config = config_manager.load_config()

        # Update database configuration
        config.database.db_type = "sqlite"
        config.database.sqlite_path = args.path

        # Save configuration
        success = config_manager.save_config(config, args.scope)
        if success:
            print(f"✅ SQLite database configuration saved to {args.scope} scope")
            print(f"   Database path: {args.path}")
        else:
            print("❌ Failed to save SQLite configuration")
    except Exception as e:
        logger.error(f"❌ Error configuring SQLite database: {e}")


def handle_db_set_mysql(args, config_manager: ConfigManager, logger: logging.Logger):
    """Configure MySQL/MariaDB database."""
    import getpass

    try:
        config = config_manager.load_config()

        # Get password if not provided
        password = args.password
        if not password:
            password = getpass.getpass("Enter database password: ")

        # Update database configuration
        config.database.db_type = "mysql"
        config.database.host = args.host
        config.database.port = args.port
        config.database.database = args.database
        config.database.username = args.username
        config.database.password = password
        config.database.charset = args.charset
        config.database.ssl_disabled = args.ssl_disabled

        # Test connection first
        print("🔌 Testing MySQL connection...")
        success, message = config_manager.test_database_connection()
        if not success:
            print(f"❌ Connection test failed: {message}")
            print("Configuration not saved.")
            return

        print("✅ Connection test successful!")

        # Save configuration
        success = config_manager.save_config(config, args.scope)
        if success:
            print(f"✅ MySQL database configuration saved to {args.scope} scope")
            print(f"   Host: {args.host}:{args.port}")
            print(f"   Database: {args.database}")
            print(f"   Username: {args.username}")
        else:
            print("❌ Failed to save MySQL configuration")

    except Exception as e:
        logger.error(f"❌ Error configuring MySQL database: {e}")


def handle_migrate_db(args, logger: logging.Logger):
    """Handle 'migrate-db' command - Database migration operations.

    Args:
        args: Parsed command-line arguments with migrate_action subcommand
        logger: Logger instance
    """
    from src.utils.database_migrator import DatabaseMigrator
    from src.core.database_manager import DatabaseManager

    # Ensure QCoreApplication exists
    if not QCoreApplication.instance():
        app = QCoreApplication(sys.argv)

    try:
        config_manager = ConfigManager()

        if args.migrate_action == "export":
            handle_migrate_export(args, config_manager, logger)
        elif args.migrate_action == "import":
            handle_migrate_import(args, config_manager, logger)
        else:
            print("❌ No migration action specified. Use 'export' or 'import'")

    except Exception as e:
        logger.error(f"❌ Migration operation failed: {e}")


def handle_migrate_export(args, config_manager: ConfigManager, logger: logging.Logger):
    """Export database to JSON backup file."""
    from src.utils.database_migrator import DatabaseMigrator
    from src.core.database_manager import DatabaseManager

    try:
        config = config_manager.load_config()
        db_manager = DatabaseManager(config.database)
        migrator = DatabaseMigrator(logger)

        if args.show_info:
            # Show export information without exporting
            print("📊 Database Export Information:")
            print("-" * 40)
            export_info = migrator.get_export_info(db_manager)

            print(f"Database Type: {export_info.get('database_type', 'unknown')}")
            print(f"Total Records: {export_info.get('total_records', 0):,}")
            print("\nTable Breakdown:")
            for table, count in export_info.get('tables', {}).items():
                print(f"  {table}: {count:,} records")

            if 'connection_info' in export_info:
                conn_info = export_info['connection_info']
                if conn_info.get('type') == 'sqlite':
                    size_mb = conn_info.get('file_size_mb', 'unknown')
                    print(f"\nDatabase File Size: {size_mb} MB")
        else:
            # Perform actual export
            print(f"🔄 Exporting database to {args.output}")
            success = migrator.export_database(db_manager, args.output)

            if success:
                print(f"✅ Export completed successfully")
                print(f"📁 Output file: {args.output}")
            else:
                print("❌ Export failed")

    except Exception as e:
        logger.error(f"❌ Export failed: {e}")


def handle_migrate_import(args, config_manager: ConfigManager, logger: logging.Logger):
    """Import database from JSON backup file."""
    from src.utils.database_migrator import DatabaseMigrator
    from src.core.database_manager import DatabaseManager

    try:
        config = config_manager.load_config()
        db_manager = DatabaseManager(config.database)
        migrator = DatabaseMigrator(logger)

        if args.dry_run:
            # Validate import file without importing
            print(f"🔍 Validating import file: {args.input}")

            input_path = Path(args.input)
            if not input_path.exists():
                print(f"❌ Input file not found: {args.input}")
                return

            with open(input_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)

            print("✅ JSON file is valid")
            print(f"   Version: {import_data.get('version', 'unknown')}")
            print(f"   Source DB Type: {import_data.get('source_db_type', 'unknown')}")
            print(f"   Created: {import_data.get('timestamp', 'unknown')}")

            if 'data' in import_data:
                print(f"   Tables: {len(import_data['data'])} found")
                total_records = 0
                for table, records in import_data['data'].items():
                    record_count = len(records) if isinstance(records, list) else 0
                    print(f"     {table}: {record_count:,} records")
                    total_records += record_count
                print(f"   Total Records: {total_records:,}")

            print(f"🎯 Target DB Type: {config.database.db_type}")
            print("✅ Import file validation completed")
        else:
            # Perform actual import
            print(f"🔄 Importing database from {args.input}")
            if args.clear:
                print("⚠️  Warning: Existing data will be cleared before import")

            success = migrator.import_database(db_manager, args.input, args.clear)

            if success:
                print(f"✅ Import completed successfully")
                print(f"📁 Source file: {args.input}")
            else:
                print("❌ Import failed")

    except Exception as e:
        logger.error(f"❌ Import failed: {e}")


def handle_clear_cache(args, logger: logging.Logger):
    """Handle 'clear-cache' command - Clear cache database.

    Args:
        args: Parsed command-line arguments with:
            - type: Type of cache to clear (all/gnd/search/classifications)
            - confirm: Skip confirmation prompt
        logger: Logger instance
    """
    try:
        cache_manager = UnifiedKnowledgeManager()

        if not args.confirm:
            # Show current cache stats
            stats = cache_manager.get_cache_stats()
            print("📊 Current cache statistics:")
            print(f"   GND entries: {stats.get('gnd_entries', 0)}")
            print(f"   Classifications: {stats.get('classification_entries', 0)}")
            print(f"   Search mappings: {stats.get('search_mappings', 0)}")
            print(f"   Database size: {stats.get('size_mb', 0)} MB")
            print()

            # Confirmation prompt
            if args.type == "all":
                confirm_msg = "⚠️  Are you sure you want to clear ALL cache data? This cannot be undone. [y/N]: "
            else:
                confirm_msg = f"⚠️  Are you sure you want to clear {args.type} cache data? This cannot be undone. [y/N]: "

            response = input(confirm_msg).lower().strip()
            if response not in ['y', 'yes']:
                print("❌ Cache clearing cancelled.")
                return

        print(f"🗑️  Clearing {args.type} cache data...")

        if args.type == "all":
            cache_manager.clear_database()
            print("✅ All cache data cleared successfully.")
        else:
            # Selective clearing
            if args.type == "gnd":
                cache_manager.db_manager.execute_query("DELETE FROM gnd_entries")
                print("✅ GND entries cleared successfully.")
            elif args.type == "search":
                cache_manager.db_manager.execute_query("DELETE FROM search_mappings")
                print("✅ Search mappings cleared successfully.")
            elif args.type == "classifications":
                cache_manager.db_manager.execute_query("DELETE FROM classifications")
                print("✅ Classifications cleared successfully.")

    except Exception as e:
        logger.error(f"❌ Error clearing cache: {e}")


def handle_dnb_import(args, logger: logging.Logger):
    """Handle 'dnb-import' command - Import DNB/GND data.

    Args:
        args: Parsed command-line arguments with:
            - force: Force re-download even if data exists
            - debug: Enable debug output
        logger: Logger instance
    """
    try:
        from src.core.gndparser import GNDParser

        url = "https://data.dnb.de/GND/authorities-gnd-sachbegriff_dnbmarc.mrc.xml.gz"

        print("🌐 Starte DNB-Download...")
        print(f"📡 URL: {url}")

        start_time = time.time()

        try:
            # Download file with progress
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Get file size if available
            total_size = int(response.headers.get("content-length", 0))
            if total_size > 0:
                print(f"📦 Dateigröße: {total_size / (1024*1024):.1f} MB")

            # Create temporary files
            temp_dir = tempfile.mkdtemp()
            temp_gz_path = os.path.join(temp_dir, "gnd_data.xml.gz")
            temp_xml_path = os.path.join(temp_dir, "gnd_data.xml")

            # Download with progress
            downloaded = 0
            last_console_percent = 0

            print("⬇️ Download läuft...")
            with open(temp_gz_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        console_percent = (downloaded / total_size) * 100
                        if console_percent - last_console_percent >= 10:
                            print(f"📊 Download: {console_percent:.0f}%")
                            last_console_percent = console_percent

            print("📦 Entpacke GZ-Datei...")

            # Extract gz file
            with gzip.open(temp_gz_path, "rb") as gz_file:
                with open(temp_xml_path, "wb") as xml_file:
                    xml_file.write(gz_file.read())

            print("✅ Download und Entpackung abgeschlossen")

            # Import into cache using GNDParser
            print("🔄 Starte GND-Datenbank Import...")
            print(f"📁 Datei: {temp_xml_path}")

            cache_manager = UnifiedKnowledgeManager()
            parser = GNDParser(cache_manager)

            print("⚙️ Verarbeite XML-Daten...")

            # Process the file
            parser.process_file(temp_xml_path)

            # Clean up temp files
            os.remove(temp_gz_path)
            os.remove(temp_xml_path)
            os.rmdir(temp_dir)

            elapsed = time.time() - start_time
            print(f"✅ DNB-Import erfolgreich abgeschlossen in {elapsed:.2f} Sekunden")

            # Show cache statistics
            stats = cache_manager.get_cache_stats()
            print(f"📊 Cache-Statistiken:")
            print(f"   GND-Einträge: {stats.get('gnd_entries', 0):,}")
            print(f"   Klassifikationen: {stats.get('classification_entries', 0):,}")
            print(f"   Datenbank-Größe: {stats.get('size_mb', 0):.1f} MB")

        except requests.RequestException as e:
            logger.error(f"❌ Download-Fehler: {e}")
            if args.debug:
                raise
        except Exception as e:
            logger.error(f"❌ Import-Fehler: {e}")
            if args.debug:
                raise

    except ImportError as e:
        logger.error(f"❌ Fehlende Module für DNB-Import: {e}")
    except Exception as e:
        logger.error(f"❌ Unerwarteter Fehler beim DNB-Import: {e}")
        if args.debug:
            raise
