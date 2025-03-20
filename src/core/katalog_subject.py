import requests
from bs4 import BeautifulSoup
import re
import urllib.parse
import logging
import time
from PyQt6.QtCore import QThread, pyqtSignal
import sys

class SubjectExtractor(QThread):
    """
    Extrahiert Schlagwörter aus einem Katalog basierend auf Suchbegriffen.
    Workflow: Suchbegriff -> Suche -> Trefferliste -> Einzelne Records identifizieren -> 
              Titelseite jedes Records öffnen -> Extraktion der Schlagworte
    """
    progress_updated = pyqtSignal(int, int)  # current, total
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    status_updated = pyqtSignal(str)

    def __init__(self, max_results=1):
        """
        Initialisiert den SubjectExtractor.
        
        Args:
            max_results (int): Maximale Anzahl von Ergebnissen, die verarbeitet werden sollen.
        """
        super().__init__()
        self.max_results = max_results
        self.search_term = None  # Wird in run() gesetzt
        self.base_url = "https://katalog.ub.tu-freiberg.de/Search/Results"
        self.record_base_url = "https://katalog.ub.tu-freiberg.de/Record/"
        
        # Logger einrichten
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)
        
        self.results_per_page = 20  # Standardwert des Katalogs
        self.subjects = {}  # Dict: Schlagwort -> Liste von (record_id, titel)

    def set_max_results(self, max_results):
        """Setzt die maximale Anzahl der zu verarbeitenden Ergebnisse"""
        self.max_results = max_results

    def set_search_term(self, search_term):
        """Setzt den Suchbegriff für die Suche"""
        self.search_term = search_term

    def run(self, search_term=None):
        """
        Hauptmethode des Workers, die die Extraktion startet
        
        Args:
            search_term (str): Der Suchbegriff für die Katalogsuche
        """

        self.subjects = {}  # Zurücksetzen bei jedem neuen Suchvorgang
        
        try:
            self.status_updated.emit(f"Starte Suche nach: {self.search_term}")
            self.logger.info(f"Starte Suche nach: {self.search_term}")
            self.extract_subjects()
            self.result_ready.emit(self.subjects)
            return self.subjects
        except Exception as e:
            self.logger.error(f"Fehler bei der Extraktion: {str(e)}", exc_info=True)
            self.error_occurred.emit(f"Fehler bei der Extraktion: {str(e)}")

    def build_search_url(self, page=1):
        """
        Erstellt die URL für die Suchergebnisseite mit dem Suchbegriff
        und weiteren Parametern
        """
        if not self.search_term:
            raise ValueError("Kein Suchbegriff vorhanden")
            
        encoded_search_term = urllib.parse.quote_plus(self.search_term)
        url = (f"{self.base_url}?lookfor={encoded_search_term}&type=AllFields"
               f"&hiddenFilters%5B%5D=institution%3ADE-105"
               f"&hiddenFilters%5B%5D=-format%3AArticle"
               f"&hiddenFilters%5B%5D=-format%3AElectronicArticle"
               f"&hiddenFilters%5B%5D=-source_id%3A172"
               f"&hiddenFilters%5B%5D=-source_id%3A227")
        
        if page > 1:
            url += f"&page={page}"
            
        return url

    def get_total_pages(self, soup):
        """Ermittelt die Gesamtzahl der Suchergebnisseiten"""
        pagination = soup.select_one('ul.pagination')
        if not pagination:
            return 1
            
        # Finde alle Seitenlinks
        page_links = pagination.select('a[href*="page="]')
        if not page_links:
            return 1
            
        max_page = 1
        for link in page_links:
            match = re.search(r'page=(\d+)', link['href'])
            if match:
                page_num = int(match.group(1))
                max_page = max(max_page, page_num)
        
        return max_page

    def extract_record_links(self, soup):
        """
        Extrahiert Record-IDs aus den save-record Links und baut direkt URLs 
        zu den Record-Detailseiten, ohne die Hierarchie zu berücksichtigen
        """
        record_links = []
        
        # Finde alle save-record Links und extrahiere die Record-IDs
        save_links = soup.find_all('a', class_='save-record')
        self.logger.debug(f"Gefundene save-record Links auf der Seite: {len(save_links)}")
        
        if not save_links:
            self.logger.warning("Keine save-record Links auf der Seite gefunden!")
            return record_links
        
        # Zusätzlich alle Titel-Links finden, um sie später zuzuordnen
        title_elements = {}  # Dict zum Speichern von record_id -> (title, url)
        for title_elem in soup.find_all('a', class_='title'):
            if 'href' in title_elem.attrs:
                href = title_elem['href']
                # Extrahiere record_id aus der URL
                record_match = re.search(r'/Record/([^/]+)', href)
                if record_match:
                    record_id = record_match.group(1)
                    title = title_elem.get_text(strip=True)
                    # Erstelle vollständigen URL
                    if href.startswith('/'):
                        record_url = f"https://katalog.ub.tu-freiberg.de{href}"
                    else:
                        record_url = href
                    title_elements[record_id] = (title, record_url)
        
        # Verarbeite die save-record Links
        for idx, link in enumerate(save_links):
            record_id = link.get('data-id')
            if not record_id:
                continue
                
            # Konstruiere den Record-URL direkt aus der record_id
            record_url = f"{self.record_base_url}{record_id}"
            
            # Versuche, den Titel aus den zuvor gefundenen title-Elementen zu erhalten
            if record_id in title_elements:
                title, url = title_elements[record_id]
                # Bevorzuge die URL aus dem title-Element, falls vorhanden
                record_url = url
            else:
                # Fallback: Verwende einen generischen Titel mit der Record-ID
                title = f"Record {record_id}"
                self.logger.debug(f"Kein Titel für Record-ID {record_id} gefunden")
            
            record_links.append((record_id, record_url, title))
            self.logger.debug(f"Gefundener Record {idx+1}: ID={record_id}, Titel={title}")
        
        return record_links



    def extract_subjects_from_record(self, record_id, record_url, title):
        """
        Öffnet eine Record-Detailseite und extrahiert alle Schlagwörter
        """
        self.logger.info(f"Extrahiere Schlagwörter von Record: {record_id} - {title}")
        self.logger.debug(f"Öffne URL: {record_url}")
        
        try:
            # Hole die Record-Detailseite
            response = requests.get(record_url)
            if response.status_code != 200:
                self.logger.warning(f"HTTP {response.status_code} für Record {record_id}: {record_url}")
                return []
                
            # Parse die HTML-Seite
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Finde alle Schlagwort-Links
            subject_links = soup.select('a[href*="type=Subject"]')
            self.logger.debug(f"Gefundene Schlagwort-Links: {len(subject_links)}")
            
            extracted_subjects = []
            
            # Verarbeite jeden Schlagwort-Link
            for link in subject_links:
                href = link.get('href', '')
                self.logger.debug(f"Schlagwort-Link: {href}")
                
                # Extrahiere das Schlagwort aus der URL
                match = re.search(r'lookfor=([^&]+)&type=Subject', href)
                if match:
                    raw_subject = urllib.parse.unquote_plus(match.group(1))
                    # Entferne Anführungszeichen
                    subject = raw_subject.strip('"')
                    self.logger.debug(f"Extrahiertes Schlagwort: {subject}")
                    
                    # Teile mit + verknüpfte Schlagwörter
                    if '+' in subject:
                        for s in subject.split('+'):
                            s = s.strip()
                            if s:  # Ignoriere leere Strings
                                extracted_subjects.append(s)
                    else:
                        extracted_subjects.append(subject)
            
            # Entferne Duplikate
            extracted_subjects = list(set(extracted_subjects))
            self.logger.info(f"Extrahierte Schlagwörter aus Record {record_id}: {', '.join(extracted_subjects)}")
            self.filter_keywords(extracted_subjects)

            return extracted_subjects
            
        except Exception as e:
            self.logger.error(f"Fehler bei Extraktion von Record {record_id}: {str(e)}", exc_info=True)
            return []
        

    def filter_keywords(self, keywords):
        # Sortiere die Schlüsselwörter nach Länge, um die längeren zuerst zu überprüfen
        keywords_sorted = sorted(keywords, key=len, reverse=False)
        self.logger.debug(f"Sortierte Schlüsselwörter: {keywords_sorted}")
        # Initialisiere eine Liste für die endgültigen Schlüsselwörter
        final_keywords = []
        
        # Überprüfe jedes Schlüsselwort
        for keyword in keywords_sorted:
            self.logger.debug(f"Überprüfe Schlüsselwort '{keyword}'")
            # Überprüfe, ob das Schlüsselwort in einem der bereits hinzugefügten Schlüsselwörter vorkommt
            for existing in final_keywords:
                if existing in keyword or keyword in existing:
                    self.logger.debug(f"Schlüsselwort '{keyword}' nicht hinzugefügt, weil")
                    self.logger.debug(f"'{existing}' in '{keyword}' enthalten")
                    keyword.replace(existing, "")
                    keywords_sorted.append(keyword)
                    break
            final_keywords.append(keyword)

        self.logger.debug(f"Endgültige Schlüsselwörter: {final_keywords}")
        return final_keywords

    def extract_subjects(self):
        """
        Hauptmethode: Durchsucht alle Seiten der Suchergebnisse, 
        identifiziert Records und extrahiert deren Schlagwörter
        """
        page = 1
        processed_records = 0
        
        while processed_records < self.max_results:
            # Erstelle die URL für die Suchergebnisseite
            search_url = self.build_search_url(page)
            self.status_updated.emit(f"Verarbeite Suchergebnisseite {page}")
            self.logger.info(f"Verarbeite Suchergebnisseite {page}: {search_url}")
            
            try:
                # Hole die Suchergebnisseite
                response = requests.get(search_url)
                if response.status_code != 200:
                    self.logger.error(f"HTTP {response.status_code} für Suchergebnisseite {page}: {search_url}")
                    self.error_occurred.emit(f"Fehler beim Abrufen der Suchergebnisseite {page}")
                    break
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Beim ersten Durchlauf: Gesamtzahl der Seiten ermitteln
                if page == 1:
                    total_pages = self.get_total_pages(soup)
                    self.logger.info(f"Suchergebnisse umfassen {total_pages} Seiten")
                
                # Extrahiere Record-Links von der Suchergebnisseite
                record_links = self.extract_record_links(soup)
                
                if not record_links:
                    self.logger.info(f"Keine weiteren Records auf Seite {page} gefunden")
                    break
                
                # Verarbeite jeden gefundenen Record
                for record_id, record_url, title in record_links:
                    if processed_records >= self.max_results:
                        break
                    
                    self.status_updated.emit(f"Verarbeite Record {processed_records+1}/{self.max_results}: {title}")
                    
                    # Extrahiere Schlagwörter vom Record
                    subject_list = self.extract_subjects_from_record(record_id, record_url, title)
                    
                    # Speichere die gefundenen Schlagwörter
                    for subject in subject_list:
                        if subject not in self.subjects:
                            self.subjects[subject] = []
                        self.subjects[subject].append((record_id, title))
                    
                    processed_records += 1
                    self.progress_updated.emit(processed_records, self.max_results)
                    
                    # Kurze Pause, um den Server nicht zu überlasten
                    time.sleep(0.2)
                
                # Wenn noch nicht genug Records verarbeitet wurden, gehe zur nächsten Seite
                if processed_records < self.max_results:
                    page += 1
                else:
                    break
            
            except Exception as e:
                self.logger.error(f"Fehler beim Verarbeiten der Suchergebnisseite {page}: {str(e)}", exc_info=True)
                self.error_occurred.emit(f"Fehler beim Verarbeiten der Suchergebnisseite {page}: {str(e)}")
                break
        
        # Zusammenfassung der Ergebnisse
        self.logger.info(f"Extraktion abgeschlossen: {len(self.subjects)} Schlagwörter aus {processed_records} Records")
        self.status_updated.emit(f"Extraktion abgeschlossen: {len(self.subjects)} Schlagwörter aus {processed_records} Records")

################################################################################
if __name__ == "__main__":
    # Process command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--clear-cache":
        clear_cache()
        sys.exit(0)
    
    # Read comma-separated search terms from command line
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} SEARCH_PHRASES [MAX_PAGES]")
        print(f"  or:  {sys.argv[0]} --clear-cache")
        print(f"Examples:")
        print(f"    {sys.argv[0]} Differentialgleichung")
        print(f'    {sys.argv[0]} "Zucker, Protein, Lipid"')
        print(f"    {sys.argv[0]} Kohlenhydrate 10  # Search 10 pages")
        print(f'    {sys.argv[0]} "Supramolekulare Chemie"  # Single result example')
        print()
        sys.exit(1)

    searches = [s.strip() for s in sys.argv[1].split(",")]
    
    # Optional: Maximale Anzahl von Seiten pro Suchbegriff
    max_pages = 5
    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        max_pages = int(sys.argv[2])
    
    # SubjectSuggester als Drop-in-Ersatz verwenden
    suggestor = SubjectExtractor()
    suggestor.set_search_term("Molekulardynamik")
    results = suggestor.run()
    
    # Formatierte Ausgabe wie im Original
    print("\nFinal Results:")
    print(results)

    # Zusätzliche CSV-Ausgabe für einfache Weiterverarbeitung
    print("\nCSV format (subject,gndid,search_term):")
    print("subject,gndid,search_term")
    for search_term, subjects in results.items():
        for subject, data in subjects.items():
            # Extrahiere die GND-ID aus dem Set (nimm das erste Element)
            gnd_id = next(iter(data["gndid"]))
            print(f'"{subject}","{gnd_id}","{search_term}"')