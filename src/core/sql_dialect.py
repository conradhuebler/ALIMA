"""
SQLDialect - Datenbank-spezifische SQL-Varianten
Unterstützt SQLite und MariaDB/MySQL mit einheitlicher API
Claude Generated
"""

from typing import List, Optional


class SQLDialect:
    """
    Abstraktion für Datenbank-spezifische SQL-Varianten.
    Ermöglicht SQLite und MariaDB/MySQL mit derselben Codebasis.

    Claude Generated
    """

    @staticmethod
    def is_mysql_family(db_type: str) -> bool:
        """Prüft ob DB-Typ zur MySQL-Familie gehört (MySQL, MariaDB)"""
        return db_type.lower() in ('mysql', 'mariadb')

    @staticmethod
    def is_sqlite(db_type: str) -> bool:
        """Prüft ob DB-Typ SQLite ist"""
        return db_type.lower() in ('sqlite', 'sqlite3')

    @staticmethod
    def text_type(db_type: str, max_length: Optional[int] = None) -> str:
        """
        Gibt den passenden TEXT-Typ zurück.

        SQLite: TEXT (unbegrenzt)
        MySQL/MariaDB: TEXT (64KB) oder LONGTEXT (4GB) oder VARCHAR(n)

        Args:
            db_type: Datenbanktyp ('sqlite', 'mysql', 'mariadb')
            max_length: Optionale Maximallänge (für VARCHAR)

        Returns:
            SQL-Typ-String
        """
        if SQLDialect.is_mysql_family(db_type):
            if max_length:
                return f'VARCHAR({max_length})'
            # LONGTEXT für unbegrenzte Inhalte (bis 4GB)
            return 'LONGTEXT'
        # SQLite: TEXT ist unbegrenzt
        return 'TEXT'

    @staticmethod
    def varchar_type(max_length: int) -> str:
        """
        VARCHAR-Typ (identisch auf beiden Plattformen).

        Args:
            max_length: Maximale Länge

        Returns:
            VARCHAR(n) String
        """
        return f'VARCHAR({max_length})'

    @staticmethod
    def timestamp_type(db_type: str) -> str:
        """
        TIMESTAMP/DATETIME-Typ.

        Beide Datenbanken unterstützen DATETIME und TIMESTAMP.
        TIMESTAMP hat automatische DEFAULT CURRENT_TIMESTAMP Unterstützung.
        """
        # Beide unterstützen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        return 'TIMESTAMP'

    @staticmethod
    def primary_key_def(
        db_type: str,
        columns: List[str],
        key_lengths: Optional[dict] = None
    ) -> str:
        """
        PRIMARY KEY Definition.

        SQLite: Akzeptiert TEXT in PRIMARY KEY, ignoriert Key-Length
        MySQL/MariaDB: Benötigt Key-Length für TEXT/VARCHAR-Spalten in PRIMARY KEY

        Args:
            db_type: Datenbanktyp
            columns: Liste der Spaltennamen für den Primary Key
            key_lengths: Dict mit {'spalte': länge} für MySQL (optional)

        Returns:
            PRIMARY KEY (...) String
        """
        if SQLDialect.is_mysql_family(db_type) and key_lengths:
            # MySQL benötigt Key-Length für TEXT/VARCHAR in PRIMARY KEY
            cols = []
            for col in columns:
                if col in key_lengths:
                    cols.append(f"{col}({key_lengths[col]})")
                else:
                    cols.append(col)
            return f"PRIMARY KEY ({', '.join(cols)})"

        # SQLite: Key-Length wird ignoriert, einfach alle Spalten
        return f"PRIMARY KEY ({', '.join(columns)})"

    @staticmethod
    def auto_increment(db_type: str) -> str:
        """
        AUTO_INCREMENT/AUTOINCREMENT Keyword.

        SQLite: AUTOINCREMENT
        MySQL/MariaDB: AUTO_INCREMENT
        """
        if SQLDialect.is_mysql_family(db_type):
            return 'AUTO_INCREMENT'
        return 'AUTOINCREMENT'

    @staticmethod
    def get_table_info_query(db_type: str, table_name: str) -> str:
        """
        Query für Tabellen-Info (Spalten, Typen).

        SQLite: PRAGMA table_info(table_name)
        MySQL/MariaDB: DESCRIBE table_name

        Args:
            db_type: Datenbanktyp
            table_name: Tabellenname

        Returns:
            SQL-Query String
        """
        if SQLDialect.is_mysql_family(db_type):
            return f"DESCRIBE {table_name}"
        return f"PRAGMA table_info({table_name})"

    @staticmethod
    def parse_table_info(db_type: str, rows: List[dict]) -> List[str]:
        """
        Parst Tabellen-Info-Ergebnisse und gibt Spaltennamen zurück.

        Args:
            db_type: Datenbanktyp
            rows: Ergebnis von get_table_info_query

        Returns:
            Liste der Spaltennamen
        """
        if SQLDialect.is_mysql_family(db_type):
            # MySQL DESCRIBE gibt 'Field' zurück
            return [row.get('Field', row.get('field', '')) for row in rows]
        # SQLite PRAGMA gibt 'name' zurück
        return [row.get('name', '') for row in rows]

    @staticmethod
    def alter_table_add_column(
        db_type: str,
        table: str,
        column: str,
        definition: str
    ) -> str:
        """
        ALTER TABLE ADD COLUMN Statement.

        Identisch auf beiden Plattformen.

        Args:
            db_type: Datenbanktyp
            table: Tabellenname
            column: Spaltenname
            definition: Spaltendefinition (Typ, Constraints)

        Returns:
            ALTER TABLE Statement
        """
        return f"ALTER TABLE {table} ADD COLUMN {column} {definition}"

    @staticmethod
    def alter_table_drop_column(
        db_type: str,
        table: str,
        column: str
    ) -> str:
        """
        ALTER TABLE DROP COLUMN Statement.

        SQLite <3.35.0: Nicht unterstützt (Tabelle muss neu erstellt werden)
        MySQL/MariaDB: Vollständig unterstützt

        Args:
            db_type: Datenbanktyp
            table: Tabellenname
            column: Spaltenname

        Returns:
            ALTER TABLE Statement

        Raises:
            NotImplementedError: Für SQLite (Tabelle muss neu erstellt werden)
        """
        if SQLDialect.is_mysql_family(db_type):
            return f"ALTER TABLE {table} DROP COLUMN {column}"
        # SQLite <3.35.0 unterstützt kein DROP COLUMN
        raise NotImplementedError(
            f"SQLite DROP COLUMN requires table rebuild. "
            f"Use rebuild_table_without_column() instead."
        )

    @staticmethod
    def supports_drop_column(db_type: str) -> bool:
        """
        Prüft ob DROP COLUMN direkt unterstützt wird.

        Returns:
            True für MySQL/MariaDB, False für SQLite
        """
        return SQLDialect.is_mysql_family(db_type)

    @staticmethod
    def create_table_rebuild_sql(
        table_name: str,
        columns_def: str,
        select_columns: str,
        drop_columns: List[str]
    ) -> tuple:
        """
        Erstellt SQL für Tabellen-Neuaufbau (SQLite Workaround für DROP COLUMN).

        Args:
            table_name: Tabellenname
            columns_def: Spaltendefinition (CREATE TABLE Teil)
            select_columns: Spalten für SELECT (ohne zu entfernende)
            drop_columns: Zu entfernende Spalten

        Returns:
            Tuple aus (CREATE TABLE, INSERT SELECT, DROP TABLE, RENAME)
        """
        new_table = f"{table_name}_new"

        create_sql = f"CREATE TABLE {new_table} ({columns_def})"
        insert_sql = f"INSERT INTO {new_table} ({select_columns}) SELECT {select_columns} FROM {table_name}"
        drop_sql = f"DROP TABLE {table_name}"
        rename_sql = f"ALTER TABLE {new_table} RENAME TO {table_name}"

        return (create_sql, insert_sql, drop_sql, rename_sql)

    @staticmethod
    def get_version_query(db_type: str) -> str:
        """
        Query für Datenbank-Version.

        Args:
            db_type: Datenbanktyp

        Returns:
            SQL-Query für Version
        """
        if SQLDialect.is_mysql_family(db_type):
            return "SELECT VERSION()"
        return "SELECT sqlite_version()"

    @staticmethod
    def optimize_table_query(db_type: str, table_name: str) -> Optional[str]:
        """
        Query für Tabellen-Optimierung.

        SQLite: VACUUM (global)
        MySQL/MariaDB: OPTIMIZE TABLE (pro Tabelle)

        Args:
            db_type: Datenbanktyp
            table_name: Tabellenname

        Returns:
            SQL-Query oder None
        """
        if SQLDialect.is_mysql_family(db_type):
            return f"OPTIMIZE TABLE {table_name}"
        # SQLite: VACUUM ist global, nicht pro Tabelle
        return None

    @staticmethod
    def escape_identifier(db_type: str, identifier: str) -> str:
        """
        Escaped Identifier (Tabellen-/Spaltennamen).

        SQLite: "identifier"
        MySQL/MariaDB: `identifier`

        Args:
            db_type: Datenbanktyp
            identifier: Zu escapender Bezeichner

        Returns:
            Escaped identifier
        """
        if SQLDialect.is_mysql_family(db_type):
            return f"`{identifier}`"
        return f'"{identifier}"'

    @staticmethod
    def escape_string_literal(db_type: str, value: str) -> str:
        """
        Escaped String-Literal für SQL-Queries.

        Args:
            db_type: Datenbanktyp
            value: Zu escapender String

        Returns:
            Escaped String mit Quotes
        """
        # Beide verwenden Single Quotes, verdoppeln für Escape
        escaped = value.replace("'", "''")
        return f"'{escaped}'"

    @staticmethod
    def get_index_name(table_name: str, columns: List[str]) -> str:
        """
        Generiert konsistenten Index-Namen.

        Args:
            table_name: Tabellenname
            columns: Spaltennamen

        Returns:
            Index-Name (z.B. idx_table_col1_col2)
        """
        cols = '_'.join(columns)
        return f"idx_{table_name}_{cols}"

    @staticmethod
    def create_index_if_not_exists(
        db_type: str,
        index_name: str,
        table_name: str,
        columns: List[str]
    ) -> str:
        """
        CREATE INDEX IF NOT EXISTS Statement.

        Args:
            db_type: Datenbanktyp
            index_name: Index-Name
            table_name: Tabellenname
            columns: Spaltennamen

        Returns:
            CREATE INDEX Statement
        """
        cols = ', '.join(columns)
        return f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({cols})"

    @staticmethod
    def insert_or_replace(
        db_type: str,
        table_name: str,
        columns: List[str],
        values_placeholder: str = "?"
    ) -> str:
        """
        INSERT OR REPLACE / INSERT ... ON DUPLICATE KEY UPDATE Statement.

        SQLite: INSERT OR REPLACE INTO table (cols) VALUES (?)
        MySQL/MariaDB: INSERT INTO table (cols) VALUES (?) ON DUPLICATE KEY UPDATE ...

        Args:
            db_type: Datenbanktyp
            table_name: Tabellenname
            columns: Spaltennamen
            values_placeholder: Placeholder für Werte (Standard: '?')

        Returns:
            INSERT Statement mit Upsert-Logik
        """
        cols_str = ', '.join(columns)
        placeholders = ', '.join([values_placeholder] * len(columns))

        if SQLDialect.is_mysql_family(db_type):
            # MySQL/MariaDB: ON DUPLICATE KEY UPDATE für alle Spalten
            update_clauses = [f"{col} = VALUES({col})" for col in columns]
            update_str = ', '.join(update_clauses)
            return f"INSERT INTO {table_name} ({cols_str}) VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {update_str}"

        # SQLite: INSERT OR REPLACE
        return f"INSERT OR REPLACE INTO {table_name} ({cols_str}) VALUES ({placeholders})"

    @staticmethod
    def insert_or_ignore(
        db_type: str,
        table_name: str,
        columns: List[str],
        values_placeholder: str = "?"
    ) -> str:
        """
        INSERT OR IGNORE / INSERT IGNORE Statement.

        SQLite: INSERT OR IGNORE INTO table (cols) VALUES (?)
        MySQL/MariaDB: INSERT IGNORE INTO table (cols) VALUES (?)

        Args:
            db_type: Datenbanktyp
            table_name: Tabellenname
            columns: Spaltennamen
            values_placeholder: Placeholder für Werte (Standard: '?')

        Returns:
            INSERT Statement mit Ignore-Logik
        """
        cols_str = ', '.join(columns)
        placeholders = ', '.join([values_placeholder] * len(columns))

        if SQLDialect.is_mysql_family(db_type):
            return f"INSERT IGNORE INTO {table_name} ({cols_str}) VALUES ({placeholders})"

        return f"INSERT OR IGNORE INTO {table_name} ({cols_str}) VALUES ({placeholders})"

    @staticmethod
    def convert_insert_or_replace_sql(db_type: str, sql: str) -> str:
        """
        Konvertiert INSERT OR REPLACE SQL zu DB-spezifischem Syntax.

        SQLite: INSERT OR REPLACE INTO ... VALUES ...
        MySQL/MariaDB: INSERT INTO ... VALUES ... ON DUPLICATE KEY UPDATE ...

        Diese Methode erkennt INSERT OR REPLACE und konvertiert es entsprechend.

        Args:
            db_type: Datenbanktyp
            sql: SQL-Statement mit INSERT OR REPLACE

        Returns:
            Konvertiertes SQL-Statement
        """
        if SQLDialect.is_sqlite(db_type):
            # SQLite: INSERT OR REPLACE funktioniert direkt
            return sql

        # MySQL/MariaDB: Konvertiere zu INSERT ... ON DUPLICATE KEY UPDATE
        # Parse: INSERT OR REPLACE INTO table (col1, col2, ...) VALUES (...)
        import re

        # Match INSERT OR REPLACE pattern
        match = re.match(
            r'INSERT\s+OR\s+REPLACE\s+INTO\s+(\w+)\s*\(([^)]+)\)\s*VALUES\s*\(([^)]+)\)',
            sql.strip(),
            re.IGNORECASE | re.DOTALL
        )

        if not match:
            # Fallback: Try REPLACE INTO syntax
            match = re.match(
                r'REPLACE\s+INTO\s+(\w+)\s*\(([^)]+)\)\s*VALUES\s*\(([^)]+)\)',
                sql.strip(),
                re.IGNORECASE | re.DOTALL
            )
            if match:
                # REPLACE INTO works in MySQL/MariaDB
                return sql

            # If we can't parse, return as-is (may fail)
            return sql

        table_name = match.group(1)
        columns_str = match.group(2)
        values_str = match.group(3)

        # Extract column names
        columns = [col.strip() for col in columns_str.split(',')]

        # Build ON DUPLICATE KEY UPDATE clause for all columns
        # Use VALUES(col) for MySQL to reference the value being inserted
        update_clauses = [f"{col} = VALUES({col})" for col in columns]
        update_str = ', '.join(update_clauses)

        # Build new SQL
        new_sql = f"INSERT INTO {table_name} ({columns_str}) VALUES ({values_str}) ON DUPLICATE KEY UPDATE {update_str}"

        return new_sql