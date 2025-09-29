#!/usr/bin/env python3
"""
Database Migration and Backup Tool for ALIMA
Provides functionality to export and import database content between different database types.
Supports SQLite ↔ MariaDB migrations with transaction safety.
Claude Generated
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..core.database_manager import DatabaseManager


class DatabaseMigrator:
    """
    Database migration and backup utility for ALIMA.
    Handles export and import of database content with transaction safety.
    Claude Generated
    """

    # Define tables to migrate in dependency order
    MIGRATION_TABLES = [
        'gnd_entries',
        'classifications',
        'search_mappings'
    ]

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize DatabaseMigrator.
        Claude Generated

        Args:
            logger: Optional logger instance, creates new one if None
        """
        self.logger = logger or logging.getLogger(__name__)

    def export_database(self, db_manager: DatabaseManager, output_file: str) -> bool:
        """
        Export all relevant database data to JSON file.
        Claude Generated

        Args:
            db_manager: DatabaseManager instance for source database
            output_file: Path to output JSON file

        Returns:
            bool: True if export successful

        Raises:
            RuntimeError: If export fails
        """
        try:
            self.logger.info(f"Starting database export to {output_file}")

            # Test database connection first
            if not db_manager.get_connection().isOpen():
                raise RuntimeError("Database connection is not open")

            # Prepare export data structure
            export_data = {
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "source_db_type": db_manager.config.db_type,
                "schema_version": "2.0.0",  # ALIMA database schema version
                "data": {}
            }

            # Export each table
            total_records = 0
            for table in self.MIGRATION_TABLES:
                self.logger.info(f"Exporting table: {table}")

                try:
                    # Get all records from table
                    records = db_manager.fetch_all(f"SELECT * FROM {table}")
                    export_data["data"][table] = records
                    record_count = len(records)
                    total_records += record_count

                    self.logger.info(f"  Exported {record_count} records from {table}")

                except Exception as e:
                    self.logger.warning(f"  Failed to export table {table}: {e}")
                    # Continue with other tables, but set empty list
                    export_data["data"][table] = []

            # Add export statistics
            export_data["statistics"] = {
                "total_records": total_records,
                "tables_exported": len(self.MIGRATION_TABLES),
                "export_timestamp": datetime.now().isoformat()
            }

            # Write to JSON file
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"✅ Database export completed successfully")
            self.logger.info(f"   Exported {total_records} total records")
            self.logger.info(f"   Output file: {output_file} ({output_path.stat().st_size / 1024:.1f} KB)")

            return True

        except Exception as e:
            error_msg = f"Database export failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def import_database(self, db_manager: DatabaseManager, input_file: str,
                       clear_destination: bool = False) -> bool:
        """
        Import database data from JSON file with transaction safety.
        Claude Generated

        Args:
            db_manager: DatabaseManager instance for destination database
            input_file: Path to input JSON file
            clear_destination: If True, clear existing data before import

        Returns:
            bool: True if import successful

        Raises:
            RuntimeError: If import fails
        """
        try:
            self.logger.info(f"Starting database import from {input_file}")

            # Validate input file
            input_path = Path(input_file)
            if not input_path.exists():
                raise RuntimeError(f"Input file not found: {input_file}")

            # Load and validate JSON data
            with open(input_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)

            # Validate JSON structure
            if not self._validate_import_data(import_data):
                raise RuntimeError("Invalid import data format")

            # Test database connection
            if not db_manager.get_connection().isOpen():
                raise RuntimeError("Database connection is not open")

            # Start transaction for atomic import
            self.logger.info("Starting database transaction")
            db_manager.begin_transaction()

            try:
                total_imported = 0

                # Clear destination tables if requested
                if clear_destination:
                    self.logger.info("Clearing destination tables")
                    for table in reversed(self.MIGRATION_TABLES):  # Reverse order for FK constraints
                        try:
                            db_manager.execute_query(f"DELETE FROM {table}")
                            self.logger.info(f"  Cleared table: {table}")
                        except Exception as e:
                            self.logger.warning(f"  Failed to clear table {table}: {e}")

                # Import data for each table
                for table in self.MIGRATION_TABLES:
                    if table not in import_data["data"]:
                        self.logger.warning(f"Table {table} not found in import data, skipping")
                        continue

                    table_data = import_data["data"][table]
                    if not table_data:
                        self.logger.info(f"No data to import for table: {table}")
                        continue

                    self.logger.info(f"Importing table: {table} ({len(table_data)} records)")

                    # Import each record
                    imported_count = 0
                    for record in table_data:
                        if self._import_record(db_manager, table, record):
                            imported_count += 1

                    total_imported += imported_count
                    self.logger.info(f"  Imported {imported_count}/{len(table_data)} records")

                # Commit transaction
                db_manager.commit_transaction()
                self.logger.info("✅ Database transaction committed successfully")

                # Log final statistics
                source_stats = import_data.get("statistics", {})
                self.logger.info(f"✅ Database import completed successfully")
                self.logger.info(f"   Source: {import_data.get('source_db_type', 'unknown')} → {db_manager.config.db_type}")
                self.logger.info(f"   Imported {total_imported} total records")
                self.logger.info(f"   Source timestamp: {import_data.get('timestamp', 'unknown')}")

                return True

            except Exception as e:
                # Rollback transaction on any error
                self.logger.error(f"Import failed, rolling back transaction: {e}")
                try:
                    db_manager.rollback_transaction()
                    self.logger.info("Database transaction rolled back successfully")
                except Exception as rollback_error:
                    self.logger.error(f"Failed to rollback transaction: {rollback_error}")
                raise

        except Exception as e:
            error_msg = f"Database import failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _validate_import_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate import data structure.
        Claude Generated

        Args:
            data: Import data dictionary

        Returns:
            bool: True if data is valid
        """
        required_fields = ["version", "timestamp", "data"]

        for field in required_fields:
            if field not in data:
                self.logger.error(f"Missing required field: {field}")
                return False

        if not isinstance(data["data"], dict):
            self.logger.error("Invalid data section format")
            return False

        self.logger.info(f"Import data validated successfully (version: {data.get('version')})")
        return True

    def _import_record(self, db_manager: DatabaseManager, table: str, record: Dict[str, Any]) -> bool:
        """
        Import a single record into the specified table.
        Claude Generated

        Args:
            db_manager: DatabaseManager instance
            table: Target table name
            record: Record data as dictionary

        Returns:
            bool: True if import successful
        """
        try:
            if not record:
                return False

            # Build dynamic INSERT statement
            columns = list(record.keys())
            values = list(record.values())
            placeholders = ', '.join(['?'] * len(columns))
            columns_str = ', '.join(columns)

            sql = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"

            # Execute the insert
            db_manager.execute_query(sql, values)
            return True

        except Exception as e:
            self.logger.error(f"Failed to import record into {table}: {e}")
            self.logger.debug(f"Record data: {record}")
            return False

    def get_export_info(self, db_manager: DatabaseManager) -> Dict[str, Any]:
        """
        Get information about what would be exported.
        Claude Generated

        Args:
            db_manager: DatabaseManager instance

        Returns:
            Dict with export statistics
        """
        try:
            info = {
                "database_type": db_manager.config.db_type,
                "connection_info": db_manager.get_database_info(),
                "tables": {}
            }

            total_records = 0
            for table in self.MIGRATION_TABLES:
                try:
                    count = db_manager.fetch_scalar(f"SELECT COUNT(*) FROM {table}")
                    info["tables"][table] = count or 0
                    total_records += (count or 0)
                except Exception as e:
                    self.logger.warning(f"Could not get count for table {table}: {e}")
                    info["tables"][table] = 0

            info["total_records"] = total_records
            return info

        except Exception as e:
            self.logger.error(f"Failed to get export info: {e}")
            return {"error": str(e)}