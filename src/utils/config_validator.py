#!/usr/bin/env python3
"""
Configuration Validator - Claude Generated
Utilities for validating ALIMA configuration directories and files.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import logging
import sqlite3


@dataclass
class ConfigDirectoryInfo:
    """Information about a configuration directory - Claude Generated"""
    exists: bool = False  # Required field with default - Claude Generated Fix
    has_config: bool = False  # Required field with default - Claude Generated Fix
    has_prompts: bool = False  # Required field with default - Claude Generated Fix
    has_database: bool = False  # Required field with default - Claude Generated Fix
    config_path: Optional[Path] = None
    prompts_path: Optional[Path] = None
    database_path: Optional[Path] = None
    config_size: int = 0
    prompts_size: int = 0
    database_size: int = 0
    config_valid: bool = False
    config_version: Optional[str] = None
    database_tables: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class ConfigValidator:
    """Validates ALIMA configuration directories - Claude Generated"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def validate_config_directory(self, directory: str) -> Tuple[bool, List[str]]:
        """
        Validate that directory contains valid ALIMA configuration files - Claude Generated

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        path = Path(directory)

        # Check if directory exists
        if not path.is_dir():
            return False, [f"Directory does not exist: {directory}"]

        # Check for config.json (required)
        config_path = path / "config.json"
        if not config_path.exists():
            errors.append(f"Missing config.json in {directory}")
        else:
            # Validate JSON syntax
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON in config.json: {e}")

        # Check for prompts.json (optional but recommended)
        prompts_path = path / "prompts.json"
        if not prompts_path.exists():
            self.logger.warning(f"⚠️ prompts.json not found in {directory}")
        else:
            # Validate JSON syntax
            try:
                with open(prompts_path, 'r', encoding='utf-8') as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON in prompts.json: {e}")

        # Check for database file (optional but recommended)
        db_files = list(path.glob("*.db"))
        if not db_files:
            self.logger.warning(f"⚠️ No .db database file found in {directory}")
        else:
            # Validate SQLite database
            for db_file in db_files:
                is_valid_db = self._validate_sqlite_database(db_file)
                if not is_valid_db:
                    errors.append(f"Invalid SQLite database: {db_file.name}")

        is_valid = len(errors) == 0 or (not any("config.json" in e for e in errors))
        return is_valid, errors

    def _validate_sqlite_database(self, db_path: Path) -> bool:
        """Validate that file is a valid SQLite database - Claude Generated"""
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            cursor.fetchall()
            conn.close()
            return True
        except sqlite3.DatabaseError:
            self.logger.warning(f"Invalid database file: {db_path.name}")
            return False
        except Exception as e:
            self.logger.warning(f"Error checking database: {e}")
            return False

    def get_config_directory_info(self, directory: str) -> ConfigDirectoryInfo:
        """
        Get comprehensive information about a configuration directory - Claude Generated

        Returns:
            ConfigDirectoryInfo object with details
        """
        path = Path(directory)
        info = ConfigDirectoryInfo(exists=path.is_dir())

        if not info.exists:
            info.errors.append(f"Directory does not exist: {directory}")
            return info

        # Check config.json
        config_path = path / "config.json"
        info.has_config = config_path.exists()
        if info.has_config:
            info.config_path = config_path
            info.config_size = config_path.stat().st_size
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    info.config_valid = True
                    info.config_version = config_data.get("config_version", "unknown")
            except Exception as e:
                info.config_valid = False
                info.errors.append(f"Invalid config.json: {e}")

        # Check prompts.json
        prompts_path = path / "prompts.json"
        info.has_prompts = prompts_path.exists()
        if info.has_prompts:
            info.prompts_path = prompts_path
            info.prompts_size = prompts_path.stat().st_size

        # Check database file
        db_files = list(path.glob("*.db"))
        if db_files:
            info.has_database = True
            db_file = db_files[0]  # Use first .db file found
            info.database_path = db_file
            info.database_size = db_file.stat().st_size

            # Count tables
            try:
                conn = sqlite3.connect(str(db_file))
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
                info.database_tables = cursor.fetchone()[0]
                conn.close()
            except Exception as e:
                self.logger.warning(f"Could not read database info: {e}")

        return info

    def compare_configs(self, source_dir: str, target_dir: str) -> Dict[str, Any]:
        """
        Compare two configuration directories - Claude Generated

        Returns:
            Dictionary with comparison results
        """
        source_info = self.get_config_directory_info(source_dir)
        target_info = self.get_config_directory_info(target_dir)

        return {
            "source": {
                "valid": source_info.config_valid,
                "has_prompts": source_info.has_prompts,
                "has_database": source_info.has_database,
                "database_size_mb": source_info.database_size / (1024 * 1024),
            },
            "target": {
                "valid": target_info.config_valid,
                "has_prompts": target_info.has_prompts,
                "has_database": target_info.has_database,
                "database_size_mb": target_info.database_size / (1024 * 1024),
            },
            "files_to_overwrite": {
                "config": target_info.has_config,
                "prompts": target_info.has_prompts,
                "database": target_info.has_database,
            }
        }

    def get_database_schema_info(self, db_path: Path) -> Dict[str, Any]:
        """Get information about database schema - Claude Generated"""
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Get list of tables
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            tables = [row[0] for row in cursor.fetchall()]

            # Get row counts
            table_stats = {}
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    table_stats[table] = count
                except Exception:
                    table_stats[table] = 0

            conn.close()

            return {
                "tables": tables,
                "table_stats": table_stats,
                "total_tables": len(tables),
            }
        except Exception as e:
            self.logger.error(f"Error reading database schema: {e}")
            return {"error": str(e)}

    def format_file_size(self, size_bytes: int) -> str:
        """Format file size to human readable format - Claude Generated"""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
