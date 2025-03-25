#!/usr/bin/env python3

import requests
import re
import json
import urllib.parse
import sys
from pathlib import Path
from pprint import pprint
from typing import List, Dict, Any, Optional, Union
import html
import time
from bs4 import BeautifulSoup
from PyQt6.QtCore import QObject, QUrl, QEventLoop, pyqtSignal

################################################################################
def dd(*args):
    """Helper to dump variables and die (inspired by laravel)"""
    for arg in args:
        print()
        print(40 * "=")
        print(f"Type: {type(arg)}")
        pprint(arg)
    sys.exit(1)


################################################################################
class SWBSubjectExtractor(QObject):
    """
    Extracts GND subjects from the SWB interface directly.
    Can be used as an alternative to SubjectSuggester when the lobid.org search
    is not providing the desired results or when the local GND data is outdated.
    """
    currentTerm = pyqtSignal(str)

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        super().__init__()
        """Initialize the extractor with optional cache directory"""
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
        self.cache_filename = "swb_gnd_cache.json"
        self.cache = self._load_cache()
        
        # Für Kompatibilität mit dem originalen SubjectSuggester
        self.gnd_subjects = {}  # Dummy für gnd_subjects
        
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load cache of previously retrieved GND IDs"""
        if not self.cache_dir:
            return {}
        
        cache_file = self.cache_dir / self.cache_filename
        if not cache_file.exists():
            return {}
        
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                # Das Format des Cache ist etwas komplex, da wir gndid als Set speichern
                # Wir müssen die JSON-Sets beim Laden rekonstruieren
                data = json.load(f)
                result = {}
                
                for search_term, subjects in data.items():
                    result[search_term] = {}
                    for subject_name, subject_data in subjects.items():
                        # Konvertiere gndid zurück in ein Set
                        if isinstance(subject_data['gndid'], list):
                            subject_data['gndid'] = set(subject_data['gndid'])
                        elif isinstance(subject_data['gndid'], str):
                            subject_data['gndid'] = {subject_data['gndid']}
                        result[search_term][subject_name] = subject_data
                
                return result
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
            return {}

    def _save_cache(self):
        """Save cache of GND IDs"""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / self.cache_filename
        try:
            # Konvertiere Sets in Listen für JSON-Serialisierung
            serializable_data = {}
            for search_term, subjects in self.cache.items():
                serializable_data[search_term] = {}
                for subject_name, subject_data in subjects.items():
                    serializable_subject = subject_data.copy()
                    if isinstance(subject_data['gndid'], set):
                        serializable_subject['gndid'] = list(subject_data['gndid'])
                    serializable_data[search_term][subject_name] = serializable_subject
            
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def _is_single_result_page(self, content: str) -> bool:
        """
        Detects if the response is a single result page rather than a list of results.
        This happens when the search term matches exactly one GND entry.
        """
        # Typische Merkmale einer Einzelergebnisseite
        signs = [
            # Suche nach GND-Nummer in einer Einzelanzeige
            re.search(r'<div>GND-Nummer:.*?</div>', content, re.DOTALL) is not None,
            # Suche nach typischen Feldern einer Einzelanzeige
            re.search(r'<div>Sachbegriff:.*?</div>', content, re.DOTALL) is not None,
            # Suche nach typischen Navigationslinks, die in Ergebnislisten fehlen
            re.search(r'Oberbegriff:.*?</div>', content, re.DOTALL) is not None,
            # Prüfe, ob keine typischen Listenergebnisse vorhanden sind
            re.search(r'Treffer</span>', content, re.DOTALL) is None
        ]
        
        # Wenn mindestens 3 der 4 Zeichen zutreffen, handelt es sich wahrscheinlich um eine Einzelseite
        return sum(signs) >= 3

    def _extract_details_from_single_result(self, content: str) -> Dict[str, str]:
        """
        Extracts subject information from a single result page 
        where the search directly led to a specific GND entry.
        """
        soup = BeautifulSoup(content, 'html.parser')
        
        # Suche nach dem Sachbegriff über das exakte Tabellenlayout
        subject_name = None
        
        # Suche nach <tr> mit "Sachbegriff:" im Text
        sachbegriff_rows = soup.find_all('tr', string=lambda text: text and "Sachbegriff:" in text)
        
        if not sachbegriff_rows:
            # Alternative: Suche nach dem td mit class "preslabel" und "Sachbegriff:" im Text
            sachbegriff_labels = soup.find_all('td', {'class': 'preslabel'}, 
                                            string=lambda text: text and "Sachbegriff:" in text)
            
            if sachbegriff_labels:
                for label in sachbegriff_labels:
                    row = label.parent  # Das tr-Element
                    if row:
                        sachbegriff_rows.append(row)
        
        if not sachbegriff_rows:
            # Dritter Versuch: Suche nach div mit "Sachbegriff:"
            sachbegriff_divs = soup.find_all('div', string=lambda text: text and "Sachbegriff:" in text)
            for div in sachbegriff_divs:
                parent_td = div.find_parent('td')
                if parent_td:
                    row = parent_td.find_parent('tr')
                    if row:
                        sachbegriff_rows.append(row)
        
        # Versuche den Titel aus den gefundenen Zeilen zu extrahieren
        for row in sachbegriff_rows:
            # Suche nach dem nächsten TD mit class "presvalue"
            value_td = row.find('td', {'class': 'presvalue'})
            if value_td:
                # Suche nach dem Bold-Element (b-Tag)
                bold_elem = value_td.find('b')
                if bold_elem:
                    subject_name = bold_elem.text.strip()
                    break
        
        # Wenn kein Sachbegriff gefunden wurde, suche direkt nach dem Bold-Element in der Nähe von "Sachbegriff:"
        if not subject_name:
            # Suche im Text nach dem Muster "Sachbegriff:"...irgendwas...<b>TITLE</b>
            title_match = re.search(r'Sachbegriff:.*?<b>(.*?)</b>', content, re.DOTALL)
            if title_match:
                subject_name = title_match.group(1).strip()

        # Suche nach der GND-Nummer im Link
        gnd_id = None
        gnd_links = soup.find_all('a', href=lambda href: href and "d-nb.info/gnd/" in href)
        
        for link in gnd_links:
            # Extrahiere die GND-ID aus dem Link
            gnd_match = re.search(r'd-nb\.info/gnd/(\d+-\d+)', link['href'])
            if gnd_match:
                gnd_id = gnd_match.group(1)
                break
        
        # Wenn keine GND-ID im Link gefunden wurde, suche im gesamten Text
        if not gnd_id:
            # Suche nach GND-Nummer: und extrahiere die ID
            gnd_match = re.search(r'GND-Nummer:.*?(\d+-\d+)', content, re.DOTALL)
            if gnd_match:
                gnd_id = gnd_match.group(1)
            else:
                # Versuche, die GND-ID aus einem beliebigen Link zu extrahieren
                gnd_link_match = re.search(r'd-nb\.info/gnd/(\d+-\d+)', content)
                if gnd_link_match:
                    gnd_id = gnd_link_match.group(1)
        
        # Debug-Ausgaben
        print(f"Single result extraction - Subject name: '{subject_name}', GND ID: '{gnd_id}'")
        
        # Wenn sowohl Titel als auch GND-ID gefunden wurden, erstelle einen Eintrag
        if subject_name and gnd_id:
            return {subject_name: gnd_id}
        
        # Wenn etwas fehlt, gib ein leeres Dictionary zurück
        if not subject_name:
            print("Warning: Could not extract subject name from single result page")
        if not gnd_id:
            print("Warning: Could not extract GND ID from single result page")
        
        return {}

    
    def _extract_subjects_from_page(self, content: str) -> Dict[str, str]:
        """
        Extract subjects (Sachbegriffe) and their GND IDs from the page content
        """
        # Zuerst prüfen, ob es sich um eine Einzeltrefferseite handelt
        if self._is_single_result_page(content):
            print("Detected single result page, extracting details...")
            return self._extract_details_from_single_result(content)
        
        # Standardextraktion für Ergebnislisten
        subjects = {}
        
        # Use regex to find all GND ID patterns with surrounding context
        # This pattern looks for the format: >Title</a>...Sachbegriff...(GND / ID)
        pattern = r'>\s*([^<>]+?)\s*<\/a>[^<>]*?(?:Sachbegriff)[^<>]*?\(GND\s*/\s*(\d+-\d+)\)'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            subject_name = match.group(1).strip()
            gnd_id = match.group(2).strip()
            
            if subject_name and gnd_id:
                subjects[subject_name] = gnd_id
                print(f"Found subject: {subject_name} (GND: {gnd_id})")
        
        # If nothing found, try an alternative approach using BeautifulSoup
        if not subjects:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Find all entries with "Sachbegriff" text
            sachbegriff_elements = soup.find_all(string=re.compile(r'Sachbegriff'))
            
            for element in sachbegriff_elements:
                # Find surrounding context
                parent = element.parent
                
                # Look for GND ID
                gnd_match = re.search(r'GND\s*/\s*(\d+-\d+)', str(parent))
                if not gnd_match:
                    # Try looking at siblings
                    for sibling in parent.next_siblings:
                        if isinstance(sibling, str) and 'GND' in sibling:
                            gnd_match = re.search(r'GND\s*/\s*(\d+-\d+)', sibling)
                            if gnd_match:
                                break
                
                if gnd_match:
                    gnd_id = gnd_match.group(1)
                    
                    # Look for title in nearby links
                    link = None
                    if parent.name == 'a':
                        link = parent
                    else:
                        # Search in previous siblings or parent's children
                        for prev_elem in parent.previous_siblings:
                            if prev_elem.name == 'a':
                                link = prev_elem
                                break
                        
                        if not link and parent.parent:
                            link = parent.parent.find('a')
                    
                    if link and link.text.strip():
                        subject_name = link.text.strip()
                        subjects[subject_name] = gnd_id
                        print(f"Found subject via BS4: {subject_name} (GND: {gnd_id})")
        
        # Last resort: direct extraction from HTML with specific pattern
        if not subjects:
            # This pattern extracts the title from the Tip JavaScript function call
            js_pattern = r"Tip\('(?:<a[^>]*>)?\s*([^<>']+)\s*(?:</a>)?[^']*?Sachbegriff[^']*?GND\s*/\s*(\d+-\d+)"
            js_matches = re.finditer(js_pattern, content, re.DOTALL)
            
            for match in js_matches:
                subject_name = match.group(1).strip()
                gnd_id = match.group(2).strip()
                
                if subject_name and gnd_id:
                    subjects[subject_name] = gnd_id
                    print(f"Found subject via JS pattern: {subject_name} (GND: {gnd_id})")
        
        return subjects

    def _get_next_page_url(self, content: str, current_url: str) -> Optional[str]:
        """Extract URL of the next page of results if it exists"""
        # Wenn es eine Einzeltrefferseite ist, gibt es keine nächste Seite
        if self._is_single_result_page(content):
            return None
            
        soup = BeautifulSoup(content, 'html.parser')
        
        # Direkter Ansatz: Suche nach dem "weiter"-Button
        next_buttons = soup.find_all('input', {'value': re.compile(r'weiter', re.IGNORECASE)})
        for button in next_buttons:
            form = button.find_parent('form')
            if form and form.get('action'):
                # Form action könnte die URL zur nächsten Seite sein
                next_url = form['action']
                if next_url:
                    return self._make_absolute_url(next_url, current_url)
        
        # Suche nach Links, die "NXT?FRST=" enthalten (SWB-spezifischer Paging-Mechanismus)
        next_links = soup.find_all('a', href=re.compile(r'NXT\?FRST='))
        for link in next_links:
            # Prüfe, ob es ein "Weiter"- oder "Nächste Seite"-Link ist
            if 'weiter' in link.text.lower() or '>' in link.text or '>>' in link.text:
                return self._make_absolute_url(link['href'], current_url)
                
        # Fallback: Suche im HTML-Code nach Paging-Links
        next_pattern = r'href="([^"]*NXT\?FRST=\d+[^"]*)"[^>]*>(?:[^<]*?nächste|\s*&gt;|\s*weiter)'
        match = re.search(next_pattern, content, re.IGNORECASE)
        if match:
            return self._make_absolute_url(match.group(1), current_url)
            
        return None
    
    def _make_absolute_url(self, url: str, base_url: str) -> str:
        """Convert relative URLs to absolute URLs"""
        if url.startswith('http'):
            return url
            
        # Parse the base URL
        parsed_base = urllib.parse.urlparse(base_url)
        base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"
        
        if url.startswith('/'):
            # Absolute path relative to domain
            return f"{base_domain}{url}"
        else:
            # Relative path
            path_parts = parsed_base.path.split('/')
            # Remove the last part (file/page name)
            if path_parts and path_parts[-1]:
                path_parts = path_parts[:-1]
            base_path = '/'.join(path_parts)
            return f"{base_domain}{base_path}/{url}"

    def extract_gnd_from_swb(self, search_term: str, max_pages: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Extract subject GND IDs from SWB for the given search term,
        scanning multiple result pages if available.
        """
        # Check cache first
        if search_term in self.cache:
            print(f"Using cached results for '{search_term}'")
            return self.cache[search_term]

        # Korrekte URL für die Sachbegriff-Suche
        base_url = "https://swb.bsz-bw.de/DB=2.104/SET=20/TTL=1/CMD"
        
        # Parameter für die Suchanfrage
        params = {
            "RETRACE": "0",
            "TRM_OLD": "",
            "ACT": "SRCHA",
            "IKT": "2074",  # 2074 für Sachbegriffe statt 2072
            "SRT": "RLV",
            "TRM": search_term,
            "MATCFILTER": "N",
            "MATCSET": "N",
            "NOABS": "Y",
            "SHRTST": "50"
        }
        
        # URL zusammenbauen
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        
        print(f"Searching SWB for subject term: {search_term}")
        print(f"URL: {url}")
        
        all_subjects = {}
        current_url = url
        page_count = 0
        
        while current_url and page_count < max_pages:
            page_count += 1
            print(f"\nProcessing page {page_count}: {current_url}")
            
            try:
                response = requests.get(current_url)
                response.raise_for_status()
                
                # HTML decodieren
                content = html.unescape(response.text)
                
                # Debug: Seite speichern
                #debug_filename = f"swb_page_{search_term.replace(' ', '_')}_{page_count}.html"
                #with open(debug_filename, "w", encoding="utf-8") as f:
                #    f.write(content)
                #print(f"Saved page content to {debug_filename} for debugging")
                
                # Prüfe, ob es sich um eine Einzelergebnisseite handelt
                is_single_result = self._is_single_result_page(content)
                if is_single_result:
                    print("Detected single result page - direct match for the search term")
                
                # Sachbegriffe von dieser Seite extrahieren
                page_subjects = self._extract_subjects_from_page(content)
                print(f"Found {len(page_subjects)} subjects on page {page_count}")
                
                # Zu den Gesamtergebnissen hinzufügen
                all_subjects.update(page_subjects)
                
                # Bei Einzelergebnisseite gibt es keine nächste Seite
                if is_single_result:
                    break
                
                # URL der nächsten Seite suchen
                next_url = self._get_next_page_url(content, current_url)
                if next_url:
                    print(f"Found next page URL: {next_url}")
                    current_url = next_url
                    # Kleine Pause, um den Server nicht zu überlasten
                    time.sleep(0.5)
                else:
                    print(f"No more result pages found after page {page_count}")
                    break
                
            except Exception as e:
                print(f"Error processing page {page_count}: {e}")
                break
        
        # Ergebnisse im gewünschten Format bereitstellen
        results = {}
        for subject_name, gnd_id in all_subjects.items():
            # WICHTIG: gndid als Set speichern, um dem Format des originalen SubjectSuggester zu entsprechen
            results[subject_name] = {
                "count": 1,  # Default count
                "gndid": {gnd_id}  # Als SET speichern - genau wie im Original!
            }
        
        # Ergebnisse cachen
        self.cache[search_term] = results
        self._save_cache()
        
        return results

    # Methoden, die im originalen SubjectSuggester existieren
    def prepare(self, force_gnd_download: bool = False) -> None:
        """Dummy-Methode für Kompatibilität mit dem originalen SubjectSuggester"""
        # Tut nichts, da wir keine GND-Daten herunterladen
        pass

    def search(self, searches: List[str], max_pages: int = 5) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Searches for multiple terms and returns results.
        Format: {search_term: {subject: {"count": count, "gndid": {gnd_id}}}}
        
        Parameters:
        - searches: List of search terms
        - max_pages: Maximum number of result pages to scan per search term
        """
        results = {}
        
        for search_term in searches:
            results[search_term] = self.extract_gnd_from_swb(search_term, max_pages)
            
            # Emit Signal für Kompatibilität (eigentlich ein No-op, weil kein richtiges Signal)
            self.currentTerm.emit(search_term)
            
            print(f"\nFound {len(results[search_term])} total subjects for '{search_term}'")
            
            # Debug: Zeige einige Beispielergebnisse
            if results[search_term]:
                print("Sample results:")
                sample_count = min(5, len(results[search_term]))
                for i, (subject, data) in enumerate(list(results[search_term].items())[:sample_count]):
                    # Zeige den Typ der gndid, um sicherzustellen, dass es ein Set ist
                    gnd_type = type(data['gndid']).__name__
                    print(f"{i+1}. '{subject}': {data} (gndid type: {gnd_type})")
        
        return results


################################################################################
class SubjectSuggester(SWBSubjectExtractor):
    """
    Drop-in replacement for the original SubjectSuggester class.
    Uses SWB instead of GND dump files and lobid.org.
    """
    
    def __init__(self, data_dir: str | Path = "") -> None:
        """Initialize with the same signature as the original class"""
        if not data_dir:
            data_dir = Path(sys.argv[0]).parent.resolve() / "swb_cache"
        super().__init__(data_dir)


################################################################################
def clear_cache(cache_dir: str | Path = ""):
    """Clear the cache to force new data extraction"""
    if not cache_dir:
        cache_dir = Path(sys.argv[0]).parent.resolve() / "swb_cache"
    else:
        cache_dir = Path(cache_dir)
        
    cache_file = cache_dir / "swb_gnd_cache.json"
    if cache_file.exists():
        try:
            cache_file.unlink()
            print(f"Cache file {cache_file} deleted.")
        except Exception as e:
            print(f"Error deleting cache file: {e}")
    else:
        print(f"No cache file found at {cache_file}")


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
    suggestor = SubjectSuggester()
    results = suggestor.search(searches, max_pages)
    
    # Formatierte Ausgabe wie im Original
    print("\nFinal Results:")
    pprint(results)

    # Zusätzliche CSV-Ausgabe für einfache Weiterverarbeitung
    print("\nCSV format (subject,gndid,search_term):")
    print("subject,gndid,search_term")
    for search_term, subjects in results.items():
        for subject, data in subjects.items():
            # Extrahiere die GND-ID aus dem Set (nimm das erste Element)
            gnd_id = next(iter(data["gndid"]))
            print(f'"{subject}","{gnd_id}","{search_term}"')
