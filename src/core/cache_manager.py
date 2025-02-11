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
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
        
        # Logging konfigurieren
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

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
                self.conn.execute('''
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
                ''')
                
                self.logger.info("Datenbanktabellen erfolgreich erstellt oder bereits vorhanden")
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
                cursor = self.conn.execute('SELECT * FROM gnd_entry')
                results = cursor.fetchall()
                
                return {row['gnd_id']: dict(row) for row in results}
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
                    '''
                    SELECT COUNT(*) FROM gnd_entry 
                    WHERE gnd_id = ?
                    ''',
                    (gnd_id,)
                )
                result = cursor.fetchone()
                
                return result[0] > 0
                
        except Exception as e:
            self.logger.error(f"Fehler beim Überprüfen des GND-Eintrags: {e}")
            return False

    def insert_gnd_entry(self, gnd_id: str, title: str, description: str = "", ddcs: str = "", dks: str = "", gnd_systems : str = "", synonyms: str = "", classification: str = "", ppn : str = "") -> None:
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
                self.conn.execute(
                    '''
                    INSERT OR REPLACE INTO gnd_entry 
                    (gnd_id, title, description, ddcs, dks, gnd_systems, synonyms, classification, ppn, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
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
                        datetime.now().isoformat(),
                        datetime.now().isoformat()
                    )
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
            with self.conn:
                cursor = self.conn.execute(
                    '''
                    SELECT * FROM gnd_entry 
                    WHERE gnd_id = ?
                    ''',
                    (gnd_id,)
                )
                result = cursor.fetchone()
                
                if result:
                    # Spaltennamen abrufen
                    column_names = [description[0] for description in cursor.description]
                    # Dictionary erstellen
                    entry = dict(zip(column_names, result))
                    return entry
                else:
                    self.logger.info(f"GND-Eintrag '{gnd_id}' nicht gefunden")
                    return None
                
        except Exception as e:
            self.logger.error(f"Fehler beim Abrufen des GND-Eintrags: {e}")
            return None
        
    def update_gnd_entry(self, gnd_id: str, title: str, description: str, ddcs: str, dks: str, gnd_systems : str, synonyms: str, classification: str = "", ppn : str = "") -> None:
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
                    '''
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
                    ''',
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
                        gnd_id
                    )
                )
                
            self.logger.info(f"GND-Eintrag '{gnd_id}' erfolgreich aktualisiert")
            
        except Exception as e:
            self.logger.error(f"Fehler beim Aktualisieren des GND-Eintrags: {e}")
            raise


#self.cache_manager.update_gnd_entry(gnd_id, title = term, ddcs = ddc, gnd_systems = gdn_category, classification = category)

    def update_gnd_entry(self, gnd_id: str, title: str, ddcs: str, gnd_systems : str,  classification: str) -> None:
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
                        '''
                        UPDATE gnd_entry 
                        SET title = ?,
                            ddcs = ?,
                            gnd_systems = ?,
                            classification = ?,
                            updated_at = ?
                        WHERE gnd_id = ?
                        ''',
                        (
                            title,
                            ddcs,
                            gnd_systems,
                            classification,
                            datetime.now().isoformat(),
                            gnd_id
                        )
                    )
                    
                self.logger.info(f"GND-Eintrag '{gnd_id}' erfolgreich aktualisiert")
                
            except Exception as e:
                self.logger.error(f"Fehler beim Aktualisieren des GND-Eintrags: {e}")
                raise
