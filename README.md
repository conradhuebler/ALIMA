# ALIMA

Ein leistungsstarkes Werkzeug zur automatisierten Schlagwortgenerierung und Klassifikation mit KI-Unterstützung entwickelt an der Universitätsbibliothek "Georgius Agricola" der TU Bergakademie Freiberg.

## Überblick

Der ALIMA ist eine Python-basierte Desktop-Anwendung, die fortschrittliche KI-Technologien mit bibliothekarischen Informationssystemen verbindet. Die Anwendung unterstützt bei der Generierung von präzisen GND konformen Schlagwörtern.
ALIMA wurde mit Hilfe von Claude Sonnet (3.5 und 3.7) entwickelt.

## Funktionen
Textanalyse

    - KI-gestützte Inhaltsanalyse: Automatische Zusammenfassung und thematische Analyse von wissenschaftlichen Texten
    - Zeitlich begrenzbare Anfragen: Abbruchmöglichkeit für langläufige KI-Anfragen
    - Streaming-Verarbeitung: Echtzeit-Anzeige der KI-Ausgabe während der Verarbeitung

Schlagwortsuche und -generierung

    - Integrierte Schlagwortsuche: Anbindung an verschiedene Datenquellen (Lobid, SWB, lokaler Katalog (aktuell der - Universitätsbibliothek der TU Freiberg))
    - GND-Schlagwort-Integration: Automatische Verknüpfung mit der Gemeinsamen Normdatei
    - DDC-Filterung: Filtermöglichkeiten nach DDC-Klassifikationen
    - Caching-System: Lokale Speicherung von GND-Einträgen für schnelleren Zugriff

PDF-Verarbeitung

    - PDF-Import: Direkte Extraktion von Text aus PDF-Dokumenten
    - Metadaten-Extraktion: Automatisches Auslesen von Titeln, Autoren und Keywords aus PDF-Metadaten

Flexibles KI-System

    - Multi-Anbieter-Unterstützung: Kompatibilität mit verschiedenen KI-Diensten (OpenAI, Gemini, Anthropic, ChatAI)
    - Anpassbare KI-Parameter: Konfigurierbare Temperatur und Seed-Werte für die KI-Generierung
    - Prompt-Management: Verwaltung von Vorlagen für unterschiedliche Aufgabentypen

## Installation
### Voraussetzungen

    - Python 3.8 oder höher
    - PyQt6
    - Weitere Abhängigkeiten (siehe requirements.txt)

### Installationsschritte

Repository klonen:
```bash
git clone https://github.com/conradhuebler/ALIMA.git
cd ALIMA
```
### Virtuelle Umgebung erstellen und aktivieren:
```bash
python3 -m venv venv
source venv/bin/activate  # Unter Windows: venv\Scripts\activate
```
### Abhängigkeiten installieren:
```
pip install -r requirements.txt
```
### Konfigurationsdatei erstellen:

Erstelle eine Datei ~/.alima_config.json mit folgendem Inhalt:
```json
{
    "providers": ["openai", "chatai", "gemini", "anthropic"],
    "api_keys": {
        "openai": "YOUR_OPENAI_API_KEY",
        "chatai": "YOUR_CHATAI_API_KEY",
        "gemini": "YOUR_GEMINI_API_KEY",
        "anthropic": "YOUR_ANTHROPIC_API_KEY",
        "catalog_token": "YOUR_CATALOG_API_TOKEN",
        "catalog_search_url": "https://liberoserver/libero/LiberoWebServices.CatalogueSearcher.cls",
        "catalog_details" : "https://liberoserver/libero/LiberoWebServices.LibraryAPI.cls"
    }
}
```
Für die Katalogsuche wird libero verwendet, zum Beispiel als liberoserver: ***libero.unibib.tu-edoras.rohan:443***. Die ollama-URL ist aktuell ***http://localhost:11434***, kann aber unter 
```bash
src/llm/llm_interface.py
```
angepasst werden.
### Anwendung starten:
```bash
python3 main.py
```

## Verwendung
### Generierung freier Schlagworte

    - Öffne den "Abstract-Analyse"-Tab
    - Füge Text ein oder importiere eine PDF-Datei
    - Wähle den KI-Provider und das Modell
    - Klicke auf "Analyse starten"
    - Die Ergebnisse werden im unteren Bereich angezeigt

### Suche nach GND-Schlagworten auf der Basis der freien Schlagworte

    - Öffne den "GND-Suche"-Tab
    - Gib Suchbegriffe ein (durch Komma getrennt oder in Anführungszeichen für exakte Phrasen)
    - Wähle die gewünschten Suchquellen (Lobid, SWB, Katalog)
    - Klicke auf "Suche starten"
    - Filtere die Ergebnisse mit den DDC-Filtern nach Bedarf
    - Verwende "Gefilterte Schlagwörter generieren" für eine optimierte Liste

### Vergabe von GND-Schlagworten

    - Öffne den "Verifkation"-Tab
    - Abstrakt/Text und Liste mit GND-Schlagworten sind gefüllt
    - Wähle den KI-Provider und das Modell
    - Klicke auf "Analyse starten"
    - Die Ergebnisse werden im unteren Bereich angezeigt

### Abbrechen laufender Anfragen

    - Klicke auf den "Abbrechen"-Button während einer laufenden KI-Anfrage
    - Die Anwendung wird die Anfrage sofort beenden und den bereits generierten Text anzeigen

## Hinweise und Probleme

    - Gemini-Modelle müssen mit einem Seed von 0 gestartet werden
    - Praktisch getestet wurden Gemini, Ollama und Chatai (GWDG) als LLM-Server
    - Diverse Konfigurationen und Dialoge sind noch nicht eingebaut
    - Das kleinste Modell, das auch im letzten Schritt gute Ergebnisse erzielt, ist Gemma 3 27B

# Lizenz
LGPL v3

# Mitwirkende
Conrad Hübler

# Danksagung
Besten Dank an das Fachreferats- und IT-Team der Universitätsbibliothek. Besonderer Dank an Patrick Reichel für die effiziente Lobid-Abfrage.

# Kontakt
Conrad Hübler
