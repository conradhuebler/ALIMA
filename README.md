# ALIMA

Ein leistungsstarkes Werkzeug zur automatisierten Schlagwortgenerierung und Klassifikation mit KI-Unterstützung, entwickelt an der Universitätsbibliothek "Georgius Agricola" der TU Bergakademie Freiberg.

## Anmerkung des Autors
ALIMA sowie die Dokumentation sind größtenteils mit Claude erstellt. LLM sind oft übereuphorisch und sehen Dinge als fertig an, auch wenn sie es noch nicht sind. Gleichzeitig ändern sich dank Agentic Coding Dinge schneller, als sie dokumentiert werden können. ALIMA ist demnach noch in der Entwicklung und noch nicht alle dokumentierten Funktionen sind als fertig anzusehen.

## Überblick

ALIMA ist eine Python-basierte Desktop-Anwendung, die fortschrittliche KI-Technologien mit bibliothekarischen Informationssystemen verbindet. Die Anwendung unterstützt bei der Generierung von präzisen, GND-konformen Schlagwörtern und der Zuweisung von DK/RVK-Klassifikationen.

## Hauptfunktionen

*   **Zentraler Pipeline-Workflow:** Eine einheitliche, mehrstufige Pipeline (Input → Initialisierung → GND-Suche → Verifikation → DK-Klassifikation) bildet den Kern der Analyse.
*   **Flexible Dateneingabe:** Unterstützung für Texteingabe, DOI/URL-Auflösung sowie die automatische Texterkennung (OCR) aus PDF- und Bilddateien.
*   **Multi-Provider-Unterstützung:** Kompatibel mit verschiedenen LLM-Anbietern (Ollama, Gemini, OpenAI-kompatible APIs).
*   **Intelligente Provider-Auswahl:** Ein "Smart Mode" wählt automatisch den besten Provider und das beste Modell basierend auf der Aufgabe und benutzerdefinierten Präferenzen.
*   **Stapelverarbeitung:** Möglichkeit zur automatisierten Analyse einer großen Anzahl von Dokumenten über die Kommandozeile.
*   **Interaktive GUI:** Eine auf PyQt6 basierende Oberfläche zur Steuerung der Pipeline, Konfiguration und Überprüfung der Ergebnisse.
*   **WebAPP:** Eine WebAPP, lauffähig auf einem Server als einfach backend. Die Konfiguration für die WebApp ist dieselbe wie für das Qt6-Gui and die Kommandozeile

## Installation

### Voraussetzungen

*   Python 3.8 oder höher
*   PyQt6

### Installationsschritte

1.  **Repository klonen:**
    ```bash
    git clone https://github.com/conradhuebler/ALIMA.git
    cd ALIMA
    ```

2.  **Virtuelle Umgebung erstellen und aktivieren:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Unter Windows: venv\Scripts\activate
    ```

3.  **Abhängigkeiten installieren:**
    ```bash
    pip install -r requirements.txt
    playwright install
    ```

## Konfiguration

Die Konfiguration von ALIMA erfolgt über die Datei `config.json` im `~/.config/alima/`-Verzeichnis (Linux/macOS) oder `%APPDATA%\ALIMA\` (Windows). Die Anwendung bietet einen Einstellungsdialog, um diese Datei komfortabel zu verwalten.

Die Konfiguration der LLM-Provider ist im `unified_config`-Abschnitt zentralisiert und ermöglicht eine detaillierte Steuerung von Providern, Modellen und Aufgaben-Präferenzen.

### Struktur der `config.json`

Die `config.json` ist in mehrere Hauptbereiche unterteilt. Der wichtigste Abschnitt für die Steuerung der KI-Analyse ist `unified_config`.

```json
{
    "database_config": { ... },
    "catalog_config": { ... },
    "prompt_config": { ... },
    "system_config": { ... },
    "unified_config": {
        "providers": [
            {
                "name": "localhost",
                "provider_type": "ollama",
                "enabled": true,
                "api_key": "",
                "base_url": "http://localhost:11434",
                "preferred_model": "gemma:7b",
                "host": "http://localhost",
                "port": 11434,
                "use_ssl": false,
                "connection_type": "native_client"
            },
            {
                "name": "ChatAI",
                "provider_type": "openai_compatible",
                "enabled": true,
                "api_key": "DEIN_API_KEY",
                "base_url": "https://api.gwdg.de/ext/openai/v1",
                "preferred_model": "gpt-4o"
            }
        ],
        "task_preferences": {
            "keywords": {
                "task_type": "keywords",
                "model_priority": [
                    {
                        "provider_name": "localhost",
                        "model_name": "cogito:32b"
                    },
                    {
                        "provider_name": "gemini",
                        "model_name": "gemini-1.5-pro"
                    }
                ]
            },
            "initialisation": { ... }
        },
        "provider_priority": [
            "localhost",
            "gemini",
            "ChatAI"
        ]
    }
}
```

**Erläuterung des `unified_config`-Abschnitts:**

*   **`providers`**: Eine Liste aller konfigurierten LLM-Anbieter. Jeder Anbieter ist ein Objekt mit Typ (`ollama`, `openai_compatible`, `gemini`, etc.), URL, API-Schlüssel und einem optionalen `preferred_model`.
*   **`task_preferences`**: Hier können für spezifische Aufgaben (z.B. `keywords` für die Verifikation oder `initialisation` für die Erst-Analyse) feste Modell-Prioritäten definiert werden. ALIMA wird versuchen, die Modelle in der angegebenen Reihenfolge zu verwenden.
*   **`provider_priority`**: Eine globale Rangfolge der Provider, die verwendet wird, wenn für eine Aufgabe keine spezifische `task_preference` definiert ist.

## Verwendung

ALIMA kann über die grafische Benutzeroberfläche (GUI) oder die Kommandozeile (CLI) genutzt werden.

### 1. Grafische Benutzeroberfläche (GUI)

Starten Sie die Anwendung mit:
```bash
python3 src/alima_gui.py
```
und lassen Sie sich von dem First-Start Wizard die Konfiguration erstellen.

### 1.1 Nutzung der Gui
Der **"🚀 Pipeline"-Tab** ist der zentrale Startpunkt für alle Analysen.

**Dateneingabe im Schritt "📥 Input & Datenquellen"**

Der erste Schritt bietet ein flexibles Eingabefeld mit mehreren Optionen:

*   **Text:** Fügen Sie einen Abstract oder beliebigen Text direkt in das Textfeld ein.
*   **DOI/URL:** Geben Sie eine DOI (z.B. `10.1007/...`) oder eine URL zu einem wissenschaftlichen Artikel ein. ALIMA versucht automatisch, den Inhalt aufzulösen und den Volltext zu extrahieren.
*   **Datei laden (PDF & Bilder):**
    *   Klicken Sie auf den "Datei auswählen"-Button, um eine lokale Datei zu laden.
    *   **PDF-Dateien:** Das System extrahiert automatisch den Text aus der PDF. Bei gescannten Dokumenten oder PDFs ohne Textebene wird eine KI-basierte OCR (Texterkennung) versucht.
    *   **Bild-Dateien:** Bei Bildformaten (PNG, JPG etc.) wird automatisch eine KI-basierte OCR gestartet, um den im Bild enthaltenen Text zu extrahieren.

Nach der erfolgreichen Extraktion der Daten aus einer dieser Quellen können Sie die Analyse mit dem "🚀 Auto-Pipeline"-Button starten.

### 1.2 Nutzung der WebAPP

Für die Nutzung der WebAPP muss die GUI einmal gestartet worden sein, bzw. die Konfiguration erstellt und die Datenbank einmal mit dem GND-Abzug initialisiert werden. Anschließend starten Sie mit
```bash
python3 src/webapp/app.py
```
den Webserver.
```bash
INFO:     Started server process [105998]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Die Pipeline-Ergebnisse können als Json heruntergeladen werden und anschließend in der PyQt6 auch offline betrachtet werden.

### 2. Kommandozeilen-Nutzung (CLI)

Die ALIMA-CLI bietet zwei Hauptmodi für die Analyse: die `pipeline` für Einzelanalysen und `batch` für die Stapelverarbeitung.

#### 2.1. Einzelanalyse (`pipeline`-Befehl)

Der `pipeline`-Befehl führt eine vollständige Analyse für eine einzelne Datenquelle durch.

**Eingabe-Optionen (einer erforderlich):**
*   `--input-text "..."`: Direkte Texteingabe.
*   `--doi "..."`: Eingabe einer DOI oder einer URL (wird automatisch aufgelöst).

**Beispiel:**
```bash
# Führt eine Standard-Analyse für einen Text durch
python3 src/alima_cli.py pipeline --input-text "Ein Text über das Recycling von Lithium-Ionen-Batterien."

# Führt eine Analyse für eine DOI durch und speichert das Ergebnis
python3 src/alima_cli.py pipeline --doi "10.1007/s00442-021-04908-x" --output-json ergebnis.json
```

#### 2.2. Stapelverarbeitung (`batch`-Befehl)

Der `batch`-Befehl verarbeitet eine Liste von Datenquellen nacheinander.

**Argumente:**
*   `--batch-file <pfad>`: Pfad zu einer Textdatei, die die zu verarbeitenden Quellen enthält (eine pro Zeile).
*   `--output-dir <ordner>`: Ordner, in dem die JSON-Ergebnisdateien gespeichert werden.

**Format der Batch-Datei:**
Jede Zeile muss das Format `TYP:WERT` haben. Unterstützte Typen sind `DOI`, `URL`, `PDF`, `IMG`, `TXT`.

```
# Beispiel batch_sources.txt
DOI:10.1007/s00442-021-04908-x
URL:https://www.tagesschau.de/wirtschaft/verbraucher/recycling-lithium-ionen-akkus-100.html
PDF:/home/user/docs/studie.pdf
IMG:/home/user/images/infografik.png
TXT:/home/user/docs/abstract.txt
```

**Beispiel-Aufruf:**
```bash
# Führt eine Batch-Analyse im Smart-Modus durch
python3 src/alima_cli.py batch --batch-file batch_sources.txt --output-dir ./results
```

#### 2.3. Protokoll-Anzeige (`show-protocol`-Befehl)

Der `show-protocol`-Befehl zeigt Pipeline-Ergebnisse aus JSON-Protokolldateien direkt auf der Konsole an – wahlweise detailliert formatiert oder kompakt als CSV für Batch-Analysen mit grep und awk.

**Argumente:**
*   `json_file`: Pfad zur JSON-Protokolldatei (erforderlich)
*   `--format <detailliert|compact>`: Ausgabeformat (Standard: `detailed`)
*   `--steps <step1> <step2>`: Auszugebende Pipeline-Schritte (Standard: alle)
*   `--header`: CSV-Header ausgeben (nur mit `--format compact`)

**Verfügbare Pipeline-Schritte:**
*   `input` – Eingabetext-Verarbeitung (100 Zeichen Preview)
*   `initialisation` – Freie Schlagwort-Extraktion
*   `search` – GND/SWB/LOBID-Suchergebnisse mit Hit-Counts
*   `keywords` – Finale GND-Schlagworte mit GND-IDs
*   `dk_search` – DK-Katalogsuche mit Klassifikationen
*   `dk_classification` – Zugewiesene DK-Codes

**Beispiele:**

**Detaillierte Anzeige (Standard):**
```bash
# Alle Schritte anzeigen
python3 src/alima_cli.py show-protocol ergebnis.json

# Nur finale Keywords und DK-Codes
python3 src/alima_cli.py show-protocol ergebnis.json --steps keywords dk_classification
```

**Kompakte CSV-Ausgabe (für Batch-Analysen):**
```bash
# Einzelne Datei als CSV (eine Zeile pro Step)
python3 src/alima_cli.py show-protocol ergebnis.json --format compact

# Ausgabe:
# ergebnis.json,initialisation,Cadmium|Bodenverschmutzung|Umweltschutz
# ergebnis.json,keywords,Cadmium (4009274-4)|Bodenverschmutzung (4206275-5)
# ergebnis.json,dk_classification,628.5|333.3
```

**Batch-Verarbeitung von 100+ Dateien:**
```bash
# Header + alle Ergebnisse in CSV-Datei
python3 src/alima_cli.py show-protocol datei1.json --format compact --header > tabelle.csv
for json in results/*.json; do
    python3 src/alima_cli.py show-protocol "$json" --format compact >> tabelle.csv
done

# Nur finale Schlagworte extrahieren (grep + cut)
grep ",keywords," tabelle.csv | cut -d, -f3

# DOI → Keywords Tabelle (wenn Dateien DOI-basiert benannt sind)
awk -F, '{gsub(/_/,"/",$1); gsub(/\.json/,""); print $1 " → " $3}' tabelle.csv
```

**CSV-Datenformat:**
```csv
filename,step,data
datei.json,initialisation,Keyword1|Keyword2|Keyword3
datei.json,search,Term1:150|Term2:85
datei.json,keywords,Cadmium (4009274-4)|Bodenverschmutzung (4206275-5)
datei.json,dk_classification,628.5|333.3
```

**Anwendungsfälle:**
*   Schnelle Überprüfung von Analyseergebnissen ohne JSON-Parser
*   Batch-Export aller finalen Schlagworte in Tabellenformat
*   Grep-basierte Suche über hunderte Ergebnisdateien
*   DOI/Keyword-Tabellen für Publikationslisten
*   Pipeline-Debugging mit Step-by-Step-Anzeige

#### 2.4. DK-Klassifikation Transparenz

Die ALIMA-Pipeline zeigt automatisch an, **welche Katalog-Titel** zu jeder DK-Klassifikation führten - für maximale Nachvollziehbarkeit der automatischen Verschlagwortung.

**GUI - Real-time während Pipeline-Ausführung:**
```
[DK_SEARCH] 🔍 DK-Suche: 2 Klassifikationen gefunden (45 Titel)
  📊 DK 628.5 (45 Titel): Cadmium in der Umwelt | Bodenverschmutzung | ...
  📊 DK 333.3 (8 Titel): Umweltschutzmaßnahmen | Nachhaltiger Umgang | ...
```

**GUI - Detaillierte Ansicht im Analysis Review Tab:**
- Tab "DK/RVK-Klassifikationen" zeigt farbcodierte Klassifikationen
- Grün (>50 Titel): Hohe Konfidenz
- Teal (20-50): Mittlere Konfidenz
- Orange (<20): Niedrige Konfidenz
- Expandierbare Titellisten unter jeder Klassifikation

**CLI - Detaillierte Ausgabe:**
```bash
python3 src/alima_cli.py show-protocol ergebnis.json --format detailed --steps dk_search

[STEP: DK_SEARCH]
DK Search Results (2 classifications found):

  📊 DK 628.5
     Keywords: Cadmium, Umweltschutz
     Katalogisiert in 45 Titeln
     Sample Titel (5/5):
       1. Cadmium in der Umwelt: Quellen, Verteilung und Auswirkungen
       2. Bodenverschmutzung durch Schwermetalle: Ein Überblick
       ...
```

**CLI - Kompakte CSV-Ausgabe:**
```bash
python3 src/alima_cli.py show-protocol ergebnis.json --format compact --steps dk_search

file.json,dk_search,"628.5:45:Cadmium in der Umwelt...|Bodenverschmutzung...|..."
```

#### 2.5. K10+/WinIBW Katalog-Export

Direkter Export von Analyseergebnissen im K10+/WinIBW-Format für nahtlose Integration in Bibliothekskataloge.

**Format:**
```
5550 Schlagwort
6700 DK CODE
```

**GUI - K10+ Export Tab:**
Die "Analysis Review" hat einen neuen Tab "K10+ Export" mit:
- Automatisch generierte K10+/WinIBW-konforme Zeilen
- "In Zwischenablage kopieren" Button für direktes Einfügen in WinIBW
- GND-IDs entfernt, nur Begriffe und DK-Codes

**CLI - K10+ Export:**
```bash
# Einfacher Export für direktes Copy-Paste in WinIBW
python3 src/alima_cli.py show-protocol ergebnis.json --format k10plus

5550 Cadmium
5550 Bodenverschmutzung
5550 Umweltverschmutzung
6700 DK 628.5
6700 DK 333.3
```

**Batch-Verarbeitung mit K10+ Export:**
```bash
# Für 100+ Dateien: Alle K10+ Zeilen in eine Textdatei
for json in results/*.json; do
    python3 src/alima_cli.py show-protocol "$json" --format k10plus
done > k10plus_export.txt

# Dann in WinIBW einfügen: Copy → Paste → Speichern
```

**Konfigurierbarkeit:**
Die Tags (5550, 6700) können später in config.json konfiguriert werden:
```json
{
  "k10plus_export": {
    "keyword_tag": "5550",
    "classification_tag": "6700"
  }
}
```

## Lizenz
LGPL v3

## Mitwirkende
Conrad Hübler
Claude und Gemini AI

## Danksagung
Besten Dank an das Fachreferats- und IT-Team der Universitätsbibliothek. Besonderer Dank an Patrick Reichel für die effiziente Lobid-Abfrage.

## Kontakt
Conrad Hübler
