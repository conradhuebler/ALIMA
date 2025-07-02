import sqlite3
import json
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any
import os


class CacheManager:
    """Verwaltet das Caching von Suchergebnissen"""

    def __init__(self, db_path: str = "search_cache.db"):
        """
        Initialisiert den Cache-Manager.

        Args:
            db_path: Pfad zur SQLite-Datenbank
        """
        # Logger initialisieren
        self.logger = logging.getLogger(__name__)

        # Stelle sicher, dass der Datenbankpfad existiert
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)

        self.db_path = db_path

        # Create in-memory database
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row

        # Create tables in memory
        self.create_tables()

        # Load data from file database into memory if file exists
        if os.path.exists(db_path):
            self._load_from_file_to_memory()
            # Verify data integrity
            self.verify_data_integrity()

        # Show database statistics
        self.get_memory_database_stats()

    def _load_from_file_to_memory(self):
        """Lädt die komplette Datenbank aus der Datei in den Speicher."""
        try:
            # Connect to file database
            file_conn = sqlite3.connect(self.db_path)
            file_conn.row_factory = sqlite3.Row

            # Get all table names from file database
            cursor = file_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            )
            tables = [row[0] for row in cursor.fetchall()]

            total_records = 0

            for table_name in tables:
                try:
                    # Get table schema from file database
                    schema_cursor = file_conn.execute(
                        f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
                    )
                    schema_result = schema_cursor.fetchone()

                    if schema_result:
                        # Drop table if exists in memory and recreate with correct schema
                        self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                        self.conn.execute(schema_result[0])

                    # Copy all data from file table to memory table
                    data_cursor = file_conn.execute(f"SELECT * FROM {table_name}")
                    rows = data_cursor.fetchall()

                    if rows:
                        # Get column names
                        columns = [
                            description[0] for description in data_cursor.description
                        ]
                        placeholders = ",".join(["?" for _ in columns])

                        # Insert all rows into memory database
                        insert_sql = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})"

                        for row in rows:
                            self.conn.execute(insert_sql, tuple(row))

                        self.conn.commit()
                        total_records += len(rows)
                        self.logger.info(
                            f"Loaded {len(rows)} records from table {table_name}"
                        )

                except Exception as table_error:
                    self.logger.error(
                        f"Error loading table {table_name}: {str(table_error)}"
                    )

            file_conn.close()
            self.logger.info(
                f"Successfully loaded {total_records} total records from {self.db_path} into memory"
            )

        except Exception as e:
            self.logger.error(f"Error loading database into memory: {str(e)}")

    def save_to_file(self):
        """Speichert die In-Memory-Datenbank zurück in die Datei."""
        try:
            # Connect to file database
            file_conn = sqlite3.connect(self.db_path)

            # Backup memory database to file
            self.conn.backup(file_conn)

            file_conn.close()
            self.logger.info(f"In-memory database saved to {self.db_path}")

        except Exception as e:
            self.logger.error(f"Error saving database to file: {str(e)}")

    def verify_data_integrity(self) -> bool:
        """
        Überprüft, ob alle Daten aus der Datei-Datenbank in der In-Memory-Datenbank vorhanden sind.

        Returns:
            bool: True wenn alle Daten vorhanden sind, False sonst
        """
        if not os.path.exists(self.db_path):
            self.logger.info("File database does not exist, skipping verification")
            return True

        try:
            # Connect to file database
            file_conn = sqlite3.connect(self.db_path)
            file_conn.row_factory = sqlite3.Row

            # Get all table names from file database
            file_cursor = file_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            )
            tables = [row[0] for row in file_cursor.fetchall()]

            verification_passed = True

            for table_name in tables:
                # Count entries in file database
                file_cursor = file_conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                file_count = file_cursor.fetchone()[0]

                # Count entries in memory database
                mem_cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                mem_count = mem_cursor.fetchone()[0]

                self.logger.info(
                    f"Table {table_name}: File={file_count}, Memory={mem_count}"
                )

                if file_count != mem_count:
                    self.logger.error(
                        f"Entry count mismatch in table {table_name}: File={file_count}, Memory={mem_count}"
                    )
                    verification_passed = False

                # For detailed verification, check specific entries (example for gnd_entry table)
                if table_name == "gnd_entry" and file_count > 0:
                    file_cursor = file_conn.execute("SELECT gnd_id FROM gnd_entry")
                    file_gnd_ids = set(row[0] for row in file_cursor.fetchall())

                    mem_cursor = self.conn.execute("SELECT gnd_id FROM gnd_entry")
                    mem_gnd_ids = set(row[0] for row in mem_cursor.fetchall())

                    missing_in_memory = file_gnd_ids - mem_gnd_ids
                    extra_in_memory = mem_gnd_ids - file_gnd_ids

                    if missing_in_memory:
                        self.logger.error(
                            f"Missing GND entries in memory: {missing_in_memory}"
                        )
                        verification_passed = False

                    if extra_in_memory:
                        self.logger.warning(
                            f"Extra GND entries in memory: {extra_in_memory}"
                        )

            file_conn.close()

            if verification_passed:
                self.logger.info("Data integrity verification passed")
            else:
                self.logger.error("Data integrity verification failed")

            return verification_passed

        except Exception as e:
            self.logger.error(f"Error during data integrity verification: {str(e)}")
            return False

    def get_memory_database_stats(self) -> Dict[str, int]:
        """
        Gibt Statistiken über die In-Memory-Datenbank zurück.

        Returns:
            Dict mit Tabellennamen und Anzahl der Einträge
        """
        stats = {}
        try:
            # Get all table names
            cursor = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            )
            tables = [row[0] for row in cursor.fetchall()]

            for table_name in tables:
                cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                stats[table_name] = count

            self.logger.info(f"Memory database stats: {stats}")
            return stats

        except Exception as e:
            self.logger.error(f"Error getting database stats: {str(e)}")
            return {}

    def _initialize_connection(self) -> sqlite3.Connection:
        """
        Initialisiert die Datenbankverbindung mit praktischen Defaults.

        Returns:
            SQLite Verbindung
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Ermöglicht dict-ähnlichen Zugriff auf Rows
        return conn

    def create_tables(self) -> None:
        """Erstellt die notwendigen Tabellen in der Datenbank."""
        try:
            with self.conn:
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS gnd_entry (
                        gnd_id TEXT PRIMARY KEY,
                        title TEXT,
                        description TEXT,
                        ddcs TEXT,
                        dks TEXT,
                        gnd_systems TEXT,
                        synonyms TEXT,
                        classification TEXT,
                        ppn TEXT,
                        created_at DATETIME,
                        updated_at DATETIME
                    )
                """
                )

                self.logger.info(
                    "Datenbanktabellen erfolgreich erstellt oder bereits vorhanden"
                )
        except sqlite3.Error as e:
            self.logger.error(f"Fehler beim Erstellen der Tabellen: {e}")
            raise

    def load_entrys(self) -> Dict:
        """
        Lädt alle GND-Einträge aus der Datenbank.

        Returns:
            Dictionary mit GND-Einträgen
        """
        try:
            with self.conn:
                cursor = self.conn.execute("SELECT * FROM gnd_entry")
                results = cursor.fetchall()

                return {row["gnd_id"]: dict(row) for row in results}
        except Exception as e:
            self.logger.error(f"Fehler beim Laden der GND-Einträge: {e}")
            raise

    def gnd_entry_exists(self, gnd_id: str) -> bool:
        """
        Überprüft, ob ein GND-Eintrag in der Datenbank existiert.

        Args:
            gnd_id: GND-ID

        Returns:
            True, wenn der Eintrag existiert, sonst False
        """
        try:
            with self.conn:
                cursor = self.conn.execute(
                    """
                    SELECT COUNT(*) FROM gnd_entry 
                    WHERE gnd_id = ?
                    """,
                    (gnd_id,),
                )
                result = cursor.fetchone()

                return result[0] > 0

        except Exception as e:
            self.logger.error(f"Fehler beim Überprüfen des GND-Eintrags: {e}")
            return False

    def gnd_keyword_exists(self, keyword: str) -> bool:
        """
        Überprüft, ob ein GND-Eintrag mit einem bestimmten Schlagwort in der Datenbank existiert.

        Args:
            keyword: Schlagwort
        """
        try:
            with self.conn:
                cursor = self.conn.execute(
                    """
                    SELECT COUNT(*) FROM gnd_entry 
                    WHERE title LIKE ?
                    """,
                    (f"%{keyword}%",),
                )
                result = cursor.fetchone()

                return result[0] > 0

        except Exception as e:
            self.logger.error(f"Fehler beim Überprüfen des GND-Eintrags: {e}")
            return False

    def insert_gnd_entry(
        self,
        gnd_id: str,
        title: str,
        description: str = "",
        ddcs: str = "",
        dks: str = "",
        gnd_systems: str = "",
        synonyms: str = "",
        classification: str = "",
        ppn: str = "",
    ) -> None:
        """
        Speichert einen GND-Eintrag in der Datenbank.

        Args:
            gnd_id: GND-ID
            title: Titel des Eintrags
            description: Beschreibung
            ddcs: DDCs
            dks: DKS
            gnd_systems: GND-Systematik
            synonyms: Synonyme
        """
        try:
            with self.conn:
                time = datetime.now().isoformat()
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO gnd_entry 
                    (gnd_id, title, description, ddcs, dks, gnd_systems, synonyms, classification, ppn, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        gnd_id,
                        title,
                        description,
                        ddcs,
                        dks,
                        gnd_systems,
                        synonyms,
                        classification,
                        ppn,
                        time,
                        time,
                    ),
                )

            self.logger.info(f"GND-Eintrag '{gnd_id}' erfolgreich gespeichert")

        except Exception as e:
            self.logger.error(f"Fehler beim Speichern des GND-Eintrags: {e}")
            raise

    def get_gnd_entry(self, gnd_id: str) -> Optional[Dict]:
        """
        Holt einen GND-Eintrag aus der Datenbank.

        Args:
            gnd_id: GND-ID
        """
        try:
            cursor = self.conn.execute(
                """
                SELECT * FROM gnd_entry 
                WHERE gnd_id = ?
                """,
                (gnd_id,),
            )
            result = cursor.fetchone()

            if result:
                # Convert Row object to dictionary
                entry = dict(result)
                return entry
            else:
                self.logger.info(f"GND-Eintrag '{gnd_id}' nicht gefunden")
                return None

        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen des GND-Eintrags: {e}")
            return None

    def get_gnd_keyword(self, keyword: str) -> Optional[Dict]:
        """
        Holt einen GND-Eintrag aus der Datenbank.

        Args:
            keyword: Schlagwort
        """
        try:
            with self.conn:
                cursor = self.conn.execute(
                    """
                    SELECT * FROM gnd_entry 
                    WHERE title LIKE ?
                    """,
                    (f"%{keyword}%",),
                )
                result = cursor.fetchone()

                if result:
                    # Spaltennamen abrufen
                    column_names = [
                        description[0] for description in cursor.description
                    ]
                    # Dictionary erstellen
                    entry = dict(zip(column_names, result))
                    return entry
                else:
                    self.logger.info(
                        f"GND-Eintrag mit Schlagwort '{keyword}' nicht gefunden"
                    )
                    return None

        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen des GND-Eintrags: {e}")
            return None

    def update_gnd_entry(
        self,
        gnd_id: str,
        title: str,
        description: str,
        ddcs: str,
        dks: str,
        gnd_systems: str,
        synonyms: str,
        classification: str = "",
        ppn: str = "",
    ) -> None:
        """
        Aktualisiert einen GND-Eintrag in der Datenbank.

        Args:
            gnd_id: GND-ID
            title: Titel des Eintrags
            description: Beschreibung
            ddcs: DDCs
            dks: DKS
            synonyms: Synonyme
        """
        try:
            with self.conn:
                self.conn.execute(
                    """
                    UPDATE gnd_entry 
                    SET title = ?,
                        description = ?,
                        ddcs = ?,
                        dks = ?,
                        gn_systems = ?,
                        synonyms = ?,
                        classification = ?,
                        ppn = ?,
                        updated_at = ?
                    WHERE gnd_id = ?
                    """,
                    (
                        title,
                        description,
                        ddcs,
                        dks,
                        gnd_systems,
                        synonyms,
                        classification,
                        ppn,
                        datetime.now().isoformat(),
                        gnd_id,
                    ),
                )

            self.logger.info(f"GND-Eintrag '{gnd_id}' erfolgreich aktualisiert")

        except Exception as e:
            self.logger.error(f"Fehler beim Aktualisieren des GND-Eintrags: {e}")
            raise

    def update_gnd_entry(
        self, gnd_id: str, title: str, ddcs: str, gnd_systems: str, classification: str
    ) -> None:
        """
        Aktualisiert einen GND-Eintrag in der Datenbank.

        Args:
            gnd_id: GND-ID
            title: Titel des Eintrags
            description: Beschreibung
            ddcs: DDCs
            dks: DKS
            synonyms: Synonyme
        """
        self.logger.info(f"Update GND-Eintrag '{ddcs}'")
        try:
            with self.conn:
                self.conn.execute(
                    """
                        UPDATE gnd_entry 
                        SET title = ?,
                            ddcs = ?,
                            gnd_systems = ?,
                            classification = ?,
                            updated_at = ?
                        WHERE gnd_id = ?
                        """,
                    (
                        title,
                        ddcs,
                        gnd_systems,
                        classification,
                        datetime.now().isoformat(),
                        gnd_id,
                    ),
                )

            self.logger.info(f"GND-Eintrag '{gnd_id}' erfolgreich aktualisiert")

        except Exception as e:
            self.logger.error(f"Fehler beim Aktualisieren des GND-Eintrags: {e}")
            raise
