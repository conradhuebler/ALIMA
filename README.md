# ALIMA

Ein leistungsstarkes Werkzeug zur automatisierten Schlagwortgenerierung und Klassifikation mit KI-Unterstützung entwickelt an der Universitätsbibliothek "Georgius Agricola" der TU Bergakademie Freiberg.

## Überblick

Der ALIMA ist eine Python-basierte Desktop-Anwendung, die fortschrittliche KI-Technologien mit bibliothekarischen Informationssystemen verbindet. Die Anwendung unterstützt bei der Generierung von präzisen GND konformen Schlagwörtern.
ALIMA wurde mit Hilfe von Claude Sonnet (3.5 und 3.7) entwickelt.

## Funktionen

### 1. CLI-Nutzung

ALIMA kann über die Kommandozeile für die Analyse von Texten und die Verwaltung von Analyseergebnissen verwendet werden. Die CLI unterstützt das Speichern und Laden von Aufgabenstatus als JSON-Dateien, was die Automatisierung und Wiederaufnahme von Aufgaben ermöglicht.

**1. Analyseaufgabe ausführen (`run` Befehl)**

Um eine Analyseaufgabe auszuführen und optional den Status in einer JSON-Datei zu speichern, verwenden Sie den `run`-Befehl:

```bash
python3 alima_cli.py run <task_name> --abstract "<ihr_abstract_text>" --model <model_name> [OPTIONEN]
```

**Argumente:**

*   `<task_name>`: Die auszuführende Analyseaufgabe (z.B. `abstract`, `keywords`).
*   `--abstract "<ihr_abstract_text>"`: Der zu analysierende Abstract oder Text. In doppelten Anführungszeichen einschließen.
*   `--model <model_name>`: Das für die Analyse zu verwendende LLM-Modell (z.B. `cogito:8b`, `gemini-1.5-flash`).

**Optionen:**

*   `--keywords "<optionale_schlagworte>"`: Optionale Schlagworte, die in die Analyse einbezogen werden sollen. In doppelten Anführungszeichen einschließen.
*   `--provider <provider_name>`: Der zu verwendende LLM-Anbieter (z.B. `ollama`, `gemini`). Standard ist `ollama`.
*   `--ollama-host <host_url>`: Ollama Host-URL. Standard ist `http://localhost`.
*   `--ollama-port <port_number>`: Ollama Port. Standard ist `11434`.
*   `--use-chunking-abstract`: Chunking für den Abstract aktivieren (Flag).
*   `--abstract-chunk-size <größe>`: Chunk-Größe für den Abstract (Ganzzahl). Standard ist `100`.
*   `--use-chunking-keywords`: Chunking für Schlagworte aktivieren (Flag).
*   `--keyword-chunk-size <größe>`: Chunk-Größe für Schlagworte (Ganzzahl). Standard ist `500`.
*   `--output-json <dateipfad>`: Pfad zum Speichern der `TaskState`-JSON-Ausgabe. Wenn angegeben, wird der vollständige Status der Analyse in dieser Datei gespeichert.

**Beispiel:**

```bash
python3 alima_cli.py run abstract \
    --abstract "Dieses Buch behandelt die Kadmiumkontamination von Böden und Pflanzen..." \
    --model cogito:8b \
    --provider ollama \
    --ollama-host http://139.20.140.163 \
    --ollama-port 11434 \
    --use-chunking-abstract \
    --abstract-chunk-size 10 \
    --output-json meine_analyse.json
```

**2. Gespeicherten Analyse-Status laden (`load-state` Befehl)**

Um einen zuvor gespeicherten Analyse-Status aus einer JSON-Datei zu laden und anzuzeigen, verwenden Sie den `load-state`-Befehl:

```bash
python3 alima_cli.py load-state <eingabe_dateipfad>
```

**Argumente:**

*   `<eingabe_dateipfad>`: Pfad zur `TaskState`-JSON-Eingabedatei.

**Beispiel:**

```bash
python3 alima_cli.py load-state meine_analyse.json
```

Dies gibt den `full_text`, `matched_keywords` und `gnd_systematic` aus der geladenen JSON-Datei aus.

**3. Wichtige extrahierte und gespeicherte Informationen**

Wenn `--output-json` verwendet wird, werden die folgenden Informationen in der JSON-Datei gespeichert:

*   `abstract_data`: Der ursprünglich bereitgestellte Abstract und die Schlagworte.
*   `analysis_result`: Enthält die `full_text`-Antwort vom LLM, `matched_keywords` (aus der LLM-Ausgabe extrahiert) und `gnd_systematic` (aus der LLM-Ausgabe extrahiert).
*   `prompt_config`: Die Konfiguration des für die Analyse verwendeten Prompts.
*   `status`: Der Status der Aufgabe (z.B. `completed`, `failed`).
*   `task_name`: Der Name der ausgeführten Aufgabe.
*   `model_used`: Das verwendete LLM-Modell.
*   `provider_used`: Der verwendete LLM-Anbieter.
*   `use_chunking_abstract`, `abstract_chunk_size`, `use_chunking_keywords`, `keyword_chunk_size`: Verwendete Chunking-Einstellungen.

Dies ermöglicht eine reproduzierbare Analyse und die Möglichkeit, Ergebnisse programmgesteuert fortzusetzen oder weiterzuverarbeiten.











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

Die ALIMA-Anwendung kann auf zwei Arten verwendet werden:

### 1. CLI-Nutzung

ALIMA kann über die Kommandozeile für die Analyse von Texten und die Verwaltung von Analyseergebnissen verwendet werden. Die CLI unterstützt das Speichern und Laden von Aufgabenstatus als JSON-Dateien, was die Automatisierung und Wiederaufnahme von Aufgaben ermöglicht.

**1.1. Analyseaufgabe ausführen (`run` Befehl)**

Um eine Analyseaufgabe auszuführen und optional den Status in einer JSON-Datei zu speichern, verwenden Sie den `run`-Befehl:

```bash
python3 alima_cli.py run <task_name> --abstract "<ihr_abstract_text>" --model <model_name> [OPTIONEN]
```

**Argumente:**

*   `<task_name>`: Die auszuführende Analyseaufgabe (z.B. `abstract`, `keywords`).
*   `--abstract "<ihr_abstract_text>"`: Der zu analysierende Abstract oder Text. In doppelten Anführungszeichen einschließen.
*   `--model <model_name>`: Das für die Analyse zu verwendende LLM-Modell (z.B. `cogito:8b`, `gemini-1.5-flash`).

**Optionen:**

*   `--keywords "<optionale_schlagworte>"`: Optionale Schlagworte, die in die Analyse einbezogen werden sollen. In doppelten Anführungszeichen einschließen.
*   `--provider <provider_name>`: Der zu verwendende LLM-Anbieter (z.B. `ollama`, `gemini`). Standard ist `ollama`.
*   `--ollama-host <host_url>`: Ollama Host-URL. Standard ist `http://localhost`.
*   `--ollama-port <port_number>`: Ollama Port. Standard ist `11434`.
*   `--use-chunking-abstract`: Chunking für den Abstract aktivieren (Flag).
*   `--abstract-chunk-size <größe>`: Chunk-Größe für den Abstract (Ganzzahl). Standard ist `100`.
*   `--use-chunking-keywords`: Chunking für Schlagworte aktivieren (Flag).
*   `--keyword-chunk-size <größe>`: Chunk-Größe für Schlagworte (Ganzzahl). Standard ist `500`.
*   `--output-json <dateipfad>`: Pfad zum Speichern der `TaskState`-JSON-Ausgabe. Wenn angegeben, wird der vollständige Status der Analyse in dieser Datei gespeichert.

**Beispiel:**

```bash
python3 alima_cli.py run abstract \
    --abstract "Dieses Buch behandelt die Kadmiumkontamination von Böden und Pflanzen..." \
    --model cogito:8b \
    --provider ollama \
    --ollama-host http://139.20.140.163 \
    --ollama-port 11434 \
    --use-chunking-abstract \
    --abstract-chunk-size 10 \
    --output-json meine_analyse.json
```

**1.2. Gespeicherten Analyse-Status laden (`load-state` Befehl)**

Um einen zuvor gespeicherten Analyse-Status aus einer JSON-Datei zu laden und anzuzeigen, verwenden Sie den `load-state`-Befehl:

```bash
python3 alima_cli.py load-state <eingabe_dateipfad>
```

**Argumente:**

*   `<eingabe_dateipfad>`: Pfad zur `TaskState`-JSON-Eingabedatei.

**Beispiel:**

```bash
python3 alima_cli.py load-state meine_analyse.json
```

Dies gibt den `full_text`, `matched_keywords` und `gnd_systematic` aus der geladenen JSON-Datei aus.

**1.3. Wichtige extrahierte und gespeicherte Informationen**

Wenn `--output-json` verwendet wird, werden die folgenden Informationen in der JSON-Datei gespeichert:

*   `abstract_data`: Der ursprünglich bereitgestellte Abstract und die Schlagworte.
*   `analysis_result`: Enthält die `full_text`-Antwort vom LLM, `matched_keywords` (aus der LLM-Ausgabe extrahiert) und `gnd_systematic` (aus der LLM-Ausgabe extrahiert).
*   `prompt_config`: Die Konfiguration des für die Analyse verwendeten Prompts.
*   `status`: Der Status der Aufgabe (z.B. `completed`, `failed`).
*   `task_name`: Der Name der ausgeführten Aufgabe.
*   `model_used`: Das verwendete LLM-Modell.
*   `provider_used`: Der verwendete LLM-Anbieter.
*   `use_chunking_abstract`, `abstract_chunk_size`, `use_chunking_keywords`, `keyword_chunk_size`: Verwendete Chunking-Einstellungen.

Dies ermöglicht eine reproduzierbare Analyse und die Möglichkeit, Ergebnisse programmgesteuert fortzusetzen oder weiterzuverarbeiten.

### 2. GUI-Nutzung

Die GUI-Anwendung bietet eine interaktive Oberfläche für die Analyse von Texten und die Verwaltung von Schlagworten. Starten Sie die GUI mit `python3 main.py`.









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
