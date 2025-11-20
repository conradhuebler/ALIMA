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


# ============================================================================
# Provider and Task Preference Validation (P0.3)
# Claude Generated - Validates provider configurations and task preferences
# ============================================================================

@dataclass
class ValidationError:
    """Represents a validation error with severity - Claude Generated"""
    field: str  # Field that failed validation
    message: str  # Human-readable error message
    severity: str  # 'error' or 'warning'
    provider_name: Optional[str] = None  # Optional provider context
    model_name: Optional[str] = None  # Optional model context


@dataclass
class ValidationResult:
    """Result of configuration validation - Claude Generated"""
    valid: bool
    errors: List['ValidationError']
    warnings: List['ValidationError']

    def has_errors(self) -> bool:
        """Check if any errors exist - Claude Generated"""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if any warnings exist - Claude Generated"""
        return len(self.warnings) > 0

    def get_error_messages(self) -> List[str]:
        """Get list of error messages - Claude Generated"""
        return [e.message for e in self.errors]

    def get_warning_messages(self) -> List[str]:
        """Get list of warning messages - Claude Generated"""
        return [w.message for w in self.warnings]

    def __str__(self) -> str:
        """String representation - Claude Generated"""
        if self.valid and not self.has_warnings():
            return "Validation passed"

        lines = []
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"  - {error.message}")

        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  - {warning.message}")

        return "\n".join(lines)


class ProviderConfigValidator:
    """Validates ALIMA provider and task preference configurations - Claude Generated"""

    @staticmethod
    def validate_provider_name_exists(provider_name: str, unified_config) -> ValidationResult:
        """
        Validate that a provider name exists in enabled providers list

        Args:
            provider_name: Provider name to check
            unified_config: UnifiedProviderConfig containing providers

        Returns:
            ValidationResult with error if provider not found

        Claude Generated
        """
        errors = []
        warnings = []

        enabled_providers = unified_config.get_enabled_providers()
        enabled_names = [p.name for p in enabled_providers]

        if provider_name not in enabled_names:
            all_providers = [p.name for p in unified_config.providers]

            if provider_name in all_providers:
                # Provider exists but is disabled
                errors.append(ValidationError(
                    field="provider_name",
                    message=f"Provider '{provider_name}' is disabled. Enable it in settings or select a different provider.",
                    severity="error",
                    provider_name=provider_name
                ))
            else:
                # Provider doesn't exist at all
                error_msg = f"Provider '{provider_name}' not found."
                if enabled_names:
                    error_msg += f" Available providers: {', '.join(enabled_names)}"
                else:
                    error_msg += " No providers configured. Run first-start wizard."

                errors.append(ValidationError(
                    field="provider_name",
                    message=error_msg,
                    severity="error",
                    provider_name=provider_name
                ))

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    @staticmethod
    def validate_model_exists_on_provider(model_name: str, provider) -> ValidationResult:
        """
        Validate that a model exists on the specified provider

        Args:
            model_name: Model name to check
            provider: UnifiedProvider to check against

        Returns:
            ValidationResult with error if model not found

        Claude Generated
        """
        errors = []
        warnings = []

        if not provider.available_models:
            warnings.append(ValidationError(
                field="model_name",
                message=f"Provider '{provider.name}' has no models listed. Model '{model_name}' cannot be validated.",
                severity="warning",
                provider_name=provider.name,
                model_name=model_name
            ))
        elif model_name not in provider.available_models:
            # Try fuzzy matching for suggestions
            suggestions = []
            lower_model = model_name.lower()
            for available_model in provider.available_models[:10]:  # Limit suggestions
                if lower_model in available_model.lower() or available_model.lower() in lower_model:
                    suggestions.append(available_model)

            error_msg = f"Model '{model_name}' not available on provider '{provider.name}'."
            if suggestions:
                error_msg += f" Did you mean: {', '.join(suggestions[:3])}?"
            elif provider.available_models:
                error_msg += f" Available models: {', '.join(provider.available_models[:5])}"
                if len(provider.available_models) > 5:
                    error_msg += f" (and {len(provider.available_models) - 5} more)"

            errors.append(ValidationError(
                field="model_name",
                message=error_msg,
                severity="error",
                provider_name=provider.name,
                model_name=model_name
            ))

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    @staticmethod
    def validate_at_least_one_provider(unified_config) -> ValidationResult:
        """
        Validate that at least one provider is configured

        Args:
            unified_config: UnifiedProviderConfig to check

        Returns:
            ValidationResult with error if no providers

        Claude Generated
        """
        errors = []
        warnings = []

        if not unified_config.providers:
            errors.append(ValidationError(
                field="providers",
                message="No providers configured. Please run the first-start wizard or add a provider in settings.",
                severity="error"
            ))
        else:
            enabled_providers = unified_config.get_enabled_providers()
            if not enabled_providers:
                errors.append(ValidationError(
                    field="providers",
                    message=f"All {len(unified_config.providers)} provider(s) are disabled. Enable at least one provider.",
                    severity="error"
                ))

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    @staticmethod
    def validate_no_duplicate_provider_names(unified_config) -> ValidationResult:
        """
        Validate that no duplicate provider names exist

        Args:
            unified_config: UnifiedProviderConfig to check

        Returns:
            ValidationResult with error if duplicates found

        Claude Generated
        """
        errors = []
        warnings = []

        name_counts: Dict[str, int] = {}
        for provider in unified_config.providers:
            name_counts[provider.name] = name_counts.get(provider.name, 0) + 1

        duplicates = [name for name, count in name_counts.items() if count > 1]
        if duplicates:
            for dup_name in duplicates:
                errors.append(ValidationError(
                    field="provider_name",
                    message=f"Duplicate provider name '{dup_name}' found {name_counts[dup_name]} times. Provider names must be unique.",
                    severity="error",
                    provider_name=dup_name
                ))

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    @staticmethod
    def validate_task_preference(task_preference, unified_config) -> ValidationResult:
        """
        Validate a complete task preference configuration

        Args:
            task_preference: TaskPreference to validate
            unified_config: UnifiedProviderConfig containing providers

        Returns:
            ValidationResult with all validation errors/warnings

        Claude Generated
        """
        errors = []
        warnings = []

        # Check each model priority entry
        for i, priority_entry in enumerate(task_preference.model_priority):
            provider_name = priority_entry.get("provider_name")
            model_name = priority_entry.get("model_name")

            if not provider_name:
                errors.append(ValidationError(
                    field=f"model_priority[{i}].provider_name",
                    message=f"Task '{task_preference.task_type.value}': Missing provider_name in priority entry {i+1}",
                    severity="error"
                ))
                continue

            if not model_name:
                errors.append(ValidationError(
                    field=f"model_priority[{i}].model_name",
                    message=f"Task '{task_preference.task_type.value}': Missing model_name in priority entry {i+1}",
                    severity="error"
                ))
                continue

            # Validate provider exists
            provider_result = ProviderConfigValidator.validate_provider_name_exists(provider_name, unified_config)
            errors.extend(provider_result.errors)
            warnings.extend(provider_result.warnings)

            # Validate model exists on provider (if provider valid)
            if not provider_result.has_errors():
                provider = unified_config.get_provider_by_name(provider_name)
                if provider:
                    model_result = ProviderConfigValidator.validate_model_exists_on_provider(model_name, provider)
                    errors.extend(model_result.errors)
                    warnings.extend(model_result.warnings)

        # Check chunked model priority if present
        if hasattr(task_preference, 'chunked_model_priority') and task_preference.chunked_model_priority:
            for i, priority_entry in enumerate(task_preference.chunked_model_priority):
                provider_name = priority_entry.get("provider_name")
                model_name = priority_entry.get("model_name")

                if provider_name and model_name:
                    provider_result = ProviderConfigValidator.validate_provider_name_exists(provider_name, unified_config)
                    if not provider_result.has_errors():
                        provider = unified_config.get_provider_by_name(provider_name)
                        if provider:
                            model_result = ProviderConfigValidator.validate_model_exists_on_provider(model_name, provider)
                            # Only add warnings for chunked models (not critical)
                            warnings.extend([ValidationError(
                                field=e.field,
                                message=e.message + " (chunked model)",
                                severity="warning",
                                provider_name=e.provider_name,
                                model_name=e.model_name
                            ) for e in model_result.errors])
                            warnings.extend(model_result.warnings)

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    @staticmethod
    def validate_full_config(unified_config) -> ValidationResult:
        """
        Validate complete unified configuration

        Args:
            unified_config: UnifiedProviderConfig to validate

        Returns:
            ValidationResult with all validation errors/warnings

        Claude Generated
        """
        errors = []
        warnings = []

        # Check at least one provider
        provider_result = ProviderConfigValidator.validate_at_least_one_provider(unified_config)
        errors.extend(provider_result.errors)
        warnings.extend(provider_result.warnings)

        # Check no duplicate names
        duplicate_result = ProviderConfigValidator.validate_no_duplicate_provider_names(unified_config)
        errors.extend(duplicate_result.errors)
        warnings.extend(duplicate_result.warnings)

        # Validate all task preferences
        for task_name, task_pref in unified_config.task_preferences.items():
            task_result = ProviderConfigValidator.validate_task_preference(task_pref, unified_config)
            errors.extend(task_result.errors)
            warnings.extend(task_result.warnings)

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
