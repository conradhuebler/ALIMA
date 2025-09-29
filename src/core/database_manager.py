#!/usr/bin/env python3
"""
DatabaseManager - Unified database management with PyQt6.QtSql
Central database management class replacing direct sqlite3 usage.
Provides seamless SQLite ↔ MariaDB switching capability.
Claude Generated
"""

import logging
from typing import Dict, List, Optional, Any, Union
from PyQt6.QtSql import QSqlDatabase, QSqlQuery, QSqlError
from PyQt6.QtCore import QCoreApplication
import json

from ..utils.config_models import DatabaseConfig


class DatabaseManager:
    """
    Unified database manager using PyQt6.QtSql for seamless SQLite/MariaDB support.
    Replaces direct sqlite3 usage with QtSql abstraction layer.
    Claude Generated
    """

    def __init__(self, database_config: DatabaseConfig, connection_name: str = "alima_main"):
        """
        Initialize DatabaseManager with configuration.

        Args:
            database_config: Database configuration object
            connection_name: Unique connection name for thread safety
        """
        self.logger = logging.getLogger(__name__)
        self.config = database_config
        self.connection_name = connection_name
        self._connection: Optional[QSqlDatabase] = None

        # Ensure QCoreApplication exists for QtSql
        if not QCoreApplication.instance():
            self.logger.warning("No QCoreApplication found. Creating minimal QCoreApplication for QtSql.")
            # Create minimal QCoreApplication for CLI mode
            import sys
            self._app = QCoreApplication(sys.argv)
        else:
            self._app = None

    def get_connection(self) -> QSqlDatabase:
        """
        Get or create database connection with appropriate driver.
        Thread-safe with named connections.
        Claude Generated

        Returns:
            QSqlDatabase connection object

        Raises:
            RuntimeError: If connection cannot be established
        """
        if self._connection and self._connection.isOpen():
            return self._connection

        # Remove existing connection if present
        if QSqlDatabase.contains(self.connection_name):
            QSqlDatabase.removeDatabase(self.connection_name)

        # Select appropriate driver
        if self.config.db_type.lower() in ['sqlite', 'sqlite3']:
            driver_name = "QSQLITE"
            self._connection = QSqlDatabase.addDatabase(driver_name, self.connection_name)
            self._connection.setDatabaseName(self.config.sqlite_path)

        elif self.config.db_type.lower() == 'mysql':
            driver_name = "QMYSQL"
            self._connection = QSqlDatabase.addDatabase(driver_name, self.connection_name)
            self._connection.setHostName(self.config.host)
            self._connection.setPort(self.config.port)
            self._connection.setDatabaseName(self.config.database)
            self._connection.setUserName(self.config.username)
            self._connection.setPassword(self.config.password)
            self._connection.setConnectOptions(f"MYSQL_OPT_CONNECT_TIMEOUT={self.config.connection_timeout}")

        elif self.config.db_type.lower() == 'mariadb':
            driver_name = "QMARIADB"
            self._connection = QSqlDatabase.addDatabase(driver_name, self.connection_name)
            self._connection.setHostName(self.config.host)
            self._connection.setPort(self.config.port)
            self._connection.setDatabaseName(self.config.database)
            self._connection.setUserName(self.config.username)
            self._connection.setPassword(self.config.password)
            self._connection.setConnectOptions(f"MYSQL_OPT_CONNECT_TIMEOUT={self.config.connection_timeout}")

        else:
            raise ValueError(f"Unsupported database type: {self.config.db_type}")

        # Attempt to open connection
        if not self._connection.open():
            error = self._connection.lastError()
            error_msg = f"Failed to open database connection: {error.text()}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        self.logger.info(f"✅ Database connection established: {driver_name} ({self.config.db_type})")
        return self._connection

    def execute_query(self, sql: str, params: Optional[List[Any]] = None) -> bool:
        """
        Execute SQL query with parameter binding.
        For INSERT, UPDATE, DELETE operations.
        Claude Generated

        Args:
            sql: SQL query string with ? placeholders
            params: List of parameters to bind

        Returns:
            bool: True if query executed successfully

        Raises:
            RuntimeError: If query execution fails
        """
        connection = self.get_connection()
        query = QSqlQuery(connection)

        # Prepare query
        if not query.prepare(sql):
            error_msg = f"Failed to prepare query: {query.lastError().text()}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Bind parameters
        if params:
            for param in params:
                query.addBindValue(param)

        # Execute query
        if not query.exec():
            error_msg = f"Query execution failed: {query.lastError().text()}"
            self.logger.error(f"SQL: {sql}")
            self.logger.error(f"Params: {params}")
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        return True

    def fetch_one(self, sql: str, params: Optional[List[Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Execute SELECT query and return first row as dictionary.
        Claude Generated

        Args:
            sql: SQL query string with ? placeholders
            params: List of parameters to bind

        Returns:
            Dict with column names as keys, or None if no results
        """
        connection = self.get_connection()
        query = QSqlQuery(connection)

        # Prepare query
        if not query.prepare(sql):
            error_msg = f"Failed to prepare query: {query.lastError().text()}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Bind parameters
        if params:
            for param in params:
                query.addBindValue(param)

        # Execute query
        if not query.exec():
            error_msg = f"Query execution failed: {query.lastError().text()}"
            self.logger.error(f"SQL: {sql}")
            self.logger.error(f"Params: {params}")
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Get first result
        if query.next():
            record = query.record()
            result = {}
            for i in range(record.count()):
                field_name = record.fieldName(i)
                field_value = query.value(i)
                result[field_name] = field_value
            return result

        return None

    def fetch_all(self, sql: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute SELECT query and return all rows as list of dictionaries.
        Claude Generated

        Args:
            sql: SQL query string with ? placeholders
            params: List of parameters to bind

        Returns:
            List of dictionaries with column names as keys
        """
        connection = self.get_connection()
        query = QSqlQuery(connection)

        # Prepare query
        if not query.prepare(sql):
            error_msg = f"Failed to prepare query: {query.lastError().text()}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Bind parameters
        if params:
            for param in params:
                query.addBindValue(param)

        # Execute query
        if not query.exec():
            error_msg = f"Query execution failed: {query.lastError().text()}"
            self.logger.error(f"SQL: {sql}")
            self.logger.error(f"Params: {params}")
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Collect all results
        results = []
        while query.next():
            record = query.record()
            row = {}
            for i in range(record.count()):
                field_name = record.fieldName(i)
                field_value = query.value(i)
                row[field_name] = field_value
            results.append(row)

        return results

    def fetch_scalar(self, sql: str, params: Optional[List[Any]] = None) -> Any:
        """
        Execute SELECT query and return single scalar value.
        Useful for COUNT(*), MAX(), etc. queries.
        Claude Generated

        Args:
            sql: SQL query string with ? placeholders
            params: List of parameters to bind

        Returns:
            Single value from first column of first row, or None
        """
        result = self.fetch_one(sql, params)
        if result:
            # Return first value from the dictionary
            return next(iter(result.values()))
        return None

    def test_connection(self) -> tuple[bool, str]:
        """
        Test database connection and return status.
        Claude Generated

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            connection = self.get_connection()

            # Simple test query
            if self.config.db_type.lower() in ['sqlite', 'sqlite3']:
                test_sql = "SELECT 1"
            else:  # MySQL/MariaDB
                test_sql = "SELECT 1 as test"

            result = self.fetch_scalar(test_sql)

            if result == 1:
                return True, f"✅ {self.config.db_type.upper()} connection successful"
            else:
                return False, f"❌ Unexpected test result: {result}"

        except Exception as e:
            return False, f"❌ Connection failed: {str(e)}"

    def get_database_info(self) -> Dict[str, Any]:
        """
        Get database information and statistics.
        Claude Generated

        Returns:
            Dictionary with database info
        """
        try:
            connection = self.get_connection()
            info = {
                "type": self.config.db_type,
                "connection_name": self.connection_name,
                "is_open": connection.isOpen(),
                "driver": connection.driverName()
            }

            if self.config.db_type.lower() in ['sqlite', 'sqlite3']:
                info["database_file"] = self.config.sqlite_path

                # Get file size if possible
                try:
                    from pathlib import Path
                    db_path = Path(self.config.sqlite_path)
                    if db_path.exists():
                        info["file_size_mb"] = round(db_path.stat().st_size / (1024 * 1024), 2)
                except Exception:
                    info["file_size_mb"] = "unknown"
            else:
                info["host"] = self.config.host
                info["port"] = self.config.port
                info["database"] = self.config.database
                info["username"] = self.config.username

            return info

        except Exception as e:
            return {"error": str(e)}

    def close_connection(self):
        """
        Close database connection and clean up.
        Claude Generated
        """
        if self._connection and self._connection.isOpen():
            self._connection.close()
            self.logger.info(f"Database connection closed: {self.connection_name}")

        if QSqlDatabase.contains(self.connection_name):
            QSqlDatabase.removeDatabase(self.connection_name)

    def begin_transaction(self) -> bool:
        """
        Begin a database transaction.
        Claude Generated

        Returns:
            bool: True if transaction started successfully

        Raises:
            RuntimeError: If transaction cannot be started
        """
        try:
            connection = self.get_connection()
            success = connection.transaction()
            if success:
                self.logger.debug("Database transaction started")
                return True
            else:
                error_msg = f"Failed to start transaction: {connection.lastError().text()}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Error starting transaction: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def commit_transaction(self) -> bool:
        """
        Commit the current database transaction.
        Claude Generated

        Returns:
            bool: True if transaction committed successfully

        Raises:
            RuntimeError: If transaction cannot be committed
        """
        try:
            connection = self.get_connection()
            success = connection.commit()
            if success:
                self.logger.debug("Database transaction committed")
                return True
            else:
                error_msg = f"Failed to commit transaction: {connection.lastError().text()}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Error committing transaction: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def rollback_transaction(self) -> bool:
        """
        Rollback the current database transaction.
        Claude Generated

        Returns:
            bool: True if transaction rolled back successfully

        Raises:
            RuntimeError: If transaction cannot be rolled back
        """
        try:
            connection = self.get_connection()
            success = connection.rollback()
            if success:
                self.logger.debug("Database transaction rolled back")
                return True
            else:
                error_msg = f"Failed to rollback transaction: {connection.lastError().text()}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Error rolling back transaction: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def __del__(self):
        """Clean up connection on destruction - Claude Generated"""
        try:
            self.close_connection()
        except Exception:
            pass  # Ignore errors during cleanup