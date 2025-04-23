# AlIma

Ein leistungsstarkes Werkzeug für Forscher und Bibliothekare zur automatisierten Schlagwortgenerierung, Textanalyse und Metadatenextraktion mit KI-Unterstützung.

## Überblick

Der AlIma ist eine Python-basierte Desktop-Anwendung, die fortschrittliche KI-Technologien mit bibliothekarischen Informationssystemen verbindet. Die Anwendung unterstützt Forscher und Bibliothekare bei der effizienten Analyse von wissenschaftlichen Texten, der Generierung von präzisen GND konformen Schlagwörtern.
AlIma wurde mit Hilfe von Claude Sonnet (3.5 und 3.7) entwickelt.

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
git clone https://github.com/username/ai-research-assistant.git
cd ai-research-assistant
```
### Virtuelle Umgebung erstellen und aktivieren:
```bash
python -m venv venv
source venv/bin/activate  # Unter Windows: venv\Scripts\activate
```
### Abhängigkeiten installieren:
```
pip install -r requirements.txt
```
### Konfigurationsdatei erstellen:

Erstelle eine Datei ~/.llm_config.json mit folgendem Inhalt:
```json
{
    "providers": ["openai", "chatai", "gemini", "anthropic"],
    "api_keys": {
        "openai": "YOUR_OPENAI_API_KEY",
        "chatai": "YOUR_CHATAI_API_KEY",
        "gemini": "YOUR_GEMINI_API_KEY",
        "anthropic": "YOUR_ANTHROPIC_API_KEY",
        "catalog": "YOUR_CATALOG_API_TOKEN"
    },
    "settings": {
        "temperature": 0.7,
        "streaming": true
    }
}
```
### Anwendung starten:
```bash
python main.py
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



# Lizenz
Muss noch entscheiden werden

# Mitwirkende
Conrad Hübler

# Kontakt
Conrad Hübler