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
        self.create_tables()  # Ensure tables are created when CacheManager is initialized

    def get_memory_database_stats(self) -> Dict[str, int]:
        """
        Gibt Statistiken über die In-Memory-Datenbank zurück.

        Returns:
            Dict mit Tabellennamen und Anzahl der Einträge
        """
        stats = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                # Get all table names
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table';"
                )
                tables = [row[0] for row in cursor.fetchall()]

                for table_name in tables:
                    cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    stats[table_name] = count

                self.logger.info(f"Memory database stats: {stats}")
                return stats

        except Exception as e:
            self.logger.error(f"Error getting database stats: {str(e)}")
            return {}

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive cache statistics - Claude Generated

        Returns:
            Dict with cache statistics including total entries and size
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM search_cache")
                total_entries = cursor.fetchone()[0]

                # Get database file size
                if os.path.exists(self.db_path):
                    size_bytes = os.path.getsize(self.db_path)
                    size_mb = size_bytes / (1024 * 1024)
                else:
                    size_mb = 0

                return {
                    "total_entries": total_entries,
                    "size_mb": size_mb,
                    "db_path": self.db_path,
                }
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return None

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
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
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

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS search_cache (
                        term TEXT PRIMARY KEY,
                        results TEXT,
                        timestamp DATETIME
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
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM gnd_entry")
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
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
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
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
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
            with sqlite3.connect(self.db_path) as conn:
                time = datetime.now().isoformat()
                conn.execute(
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
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
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
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
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
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE gnd_entry 
                    SET title = ?,
                        description = ?,
                        ddcs = ?,
                        dks = ?,
                        gnd_systems = ?,
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
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
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

    def cache_results(self, term: str, results: Dict) -> None:
        """
        Speichert Suchergebnisse im Cache.

        Args:
            term: Der Suchbegriff.
            results: Die Suchergebnisse als Dictionary.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO search_cache (term, results, timestamp)
                    VALUES (?, ?, ?)
                    """,
                    (term, json.dumps(results), datetime.now().isoformat()),
                )
            self.logger.debug(f"Cached results for term: {term}")
        except Exception as e:
            self.logger.error(f"Error caching results for {term}: {e}")

    def get_cached_results(self, term: str) -> Optional[Dict]:
        """
        Ruft Suchergebnisse aus dem Cache ab.

        Args:
            term: Der Suchbegriff.

        Returns:
            Die gecachten Ergebnisse als Dictionary, oder None wenn nicht gefunden oder abgelaufen.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT results, timestamp FROM search_cache WHERE term = ?",
                    (term,),
                )
                row = cursor.fetchone()
                if row:
                    cached_time = datetime.fromisoformat(row["timestamp"])
                    # Cache-Gültigkeit: 1 Stunde
                    if datetime.now() - cached_time < timedelta(hours=1):
                        return json.loads(row["results"])
                    else:
                        self.logger.debug(f"Cache expired for term: {term}")
                        conn.execute("DELETE FROM search_cache WHERE term = ?", (term,))
                        return None
                return None
        except Exception as e:
            self.logger.error(f"Error retrieving cached results for {term}: {e}")
            return None
