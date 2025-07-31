# ALIMA

Ein leistungsstarkes Werkzeug zur automatisierten Schlagwortgenerierung und Klassifikation mit KI-Unterstützung  (Claude und Gemini) entwickelt an der Universitätsbibliothek "Georgius Agricola" der TU Bergakademie Freiberg.

## Überblick

Der ALIMA ist eine Python-basierte Desktop-Anwendung, die fortschrittliche KI-Technologien mit bibliothekarischen Informationssystemen verbindet. Die Anwendung unterstützt bei der Generierung von präzisen GND konformen Schlagwörtern.
ALIMA wurde mit Hilfe von Claude Sonnet (3.5 und 4.0) und Gemini entwickelt. Ein erheblicher Teil der Dokumentation ist ebenfalls KI generiert, daher eine leichter Übereuphrie über noch nicht getestete Funktionen aber auch frisch implementierte ;-). Das wichtigste habe ich kommentiert.

## Funktionen

### 1. Unified Pipeline (Neu!)

ALIMA bietet jetzt einen einheitlichen Pipeline-Befehl für die vollständige Analyse von Input → Initialisierung → Suche → Verifikation:

**Pipeline-Befehl**

```bash
python3 alima_cli.py pipeline [EINGABE-OPTIONEN] [LLM-OPTIONEN] [AUSGABE-OPTIONEN]
```

**Eingabe-Optionen (einer erforderlich):**

*   `--input-text "Text"`: Direkter Text-Input
*   `--doi "10.1007/123"`: DOI-Eingabe mit automatischer Auflösung (unterstützt Springer und CrossRef)
*   `--url "https://..."`: URL-Eingabe mit Web-Crawling (unterstützt Springer und generische Websites)

**LLM-Optionen:**

*   `--initial-model <model>`: Modell für Initialisierungsschritt (Standard: `cogito:14b`)
*   `--final-model <model>`: Modell für Verifikationsschritt (Standard: `cogito:32b`)
*   `--provider <provider>`: LLM-Anbieter (`ollama`, `gemini`, `openai`, `anthropic`) (Standard: `ollama`)
*   `--initial-task <task>`: Task für Initialisierung (`initialisation`, `keywords`, `rephrase`) (Standard: `initialisation`)
*   `--final-task <task>`: Task für Verifikation (`keywords`, `rephrase`, `keywords_chunked`) (Standard: `keywords`)
*   `--ollama-host <url>`: Ollama Host-URL (Standard: `http://localhost`)
*   `--ollama-port <port>`: Ollama Port (Standard: `11434`)

**Keyword Chunking (Neu!):**

*   `--keyword-chunking-threshold <anzahl>`: Aktiviert Chunking ab dieser Keyword-Anzahl (Standard: `500`)
*   `--chunking-task <task>`: Task für Chunk-Verarbeitung (`keywords_chunked`, `rephrase`) (Standard: `keywords_chunked`)

**Suchkonfiguration:**

*   `--suggesters <suggester1> <suggester2>`: GND-Suchprovider (`lobid`, `swb`) (Standard: `lobid swb`)

**Ausgabe-Optionen:**

*   `--output-json <dateipfad>`: Speichert vollständige Pipeline-Ergebnisse als JSON
*   `--resume-from <dateipfad>`: Lädt Pipeline-Status und setzt fort

**Pipeline-Beispiele:**

```bash
# DOI-basierte Analyse mit Springer-Unterstützung
python3 alima_cli.py pipeline \
  --doi "10.1007/978-3-030-12345-6" \
  --initial-model "cogito:14b" \
  --final-model "cogito:32b" \
  --provider "ollama" \
  --ollama-host "http://139.20.140.163" \
  --ollama-port 11434 \
  --output-json "springer_analysis.json"

# URL-basierte Analyse mit Keyword Chunking
python3 alima_cli.py pipeline \
  --url "https://link.springer.com/book/10.1007/978-3-030-12345-6" \
  --keyword-chunking-threshold 300 \
  --chunking-task "keywords_chunked" \
  --output-json "url_analysis.json"

# Text-basierte Analyse mit benutzerdefinierten Tasks
python3 alima_cli.py pipeline \
  --input-text "Ihre Textanalyse hier..." \
  --initial-task "initialisation" \
  --final-task "rephrase" \
  --output-json "text_analysis.json"
```

### 2. Einzelschritte (Legacy)

ALIMA kann auch für einzelne Analyseschritte verwendet werden:

**Analyseaufgabe ausführen (`run` Befehl)**

```bash
python3 alima_cli.py run <task_name> --abstract "<ihr_abstract_text>" --model <model_name> [OPTIONEN]
```

**Gespeicherten Analyse-Status laden (`load-state` Befehl)**

```bash
python3 alima_cli.py load-state <eingabe_dateipfad>
```

## Installation

### Voraussetzungen

- Python 3.8 oder höher
- PyQt6
- Weitere Abhängigkeiten (siehe requirements.txt)

### Installationsschritte

**Repository klonen:**
```bash
git clone https://github.com/conradhuebler/ALIMA.git
cd ALIMA
```

**Virtuelle Umgebung erstellen und aktivieren:**
```bash
python3 -m venv venv
source venv/bin/activate  # Unter Windows: venv\Scripts\activate
```

**Basis-Abhängigkeiten installieren:**
```bash
pip install -r requirements.txt
```

### 🔧 Optional: Erweiterte Web-Crawling-Funktionen

Für erweiterte URL/DOI-Auflösung mit Springer-Unterstützung wird `crawl4ai` benötigt, auch wenn es erfolgreich via pip installiert ist, muss noch etwas dazu installiert werden. Das habe ich unter Windows nicht weiter getestet.

**Linux/macOS:**
```bash
pip install crawl4ai
```

**Windows (erweiterte Installation erforderlich):**
```bash
# Unter Windows kann crawl4ai zusätzliche Systemabhängigkeiten erfordern
# Bei Installationsproblemen verwenden Sie WSL oder Docker

# Alternative: Basis-Installation ohne crawl4ai (DOI-Funktionalität über CrossRef API verfügbar)
```

**Hinweise zu crawl4ai:**
- **Vollständige Funktionen**: Mit crawl4ai erhalten Sie erweiterte Springer-Unterstützung mit Inhaltsverzeichnis-Extraktion
- **Basis-Funktionalität**: Ohne crawl4ai funktionieren DOI-Auflösungen über CrossRef API weiterhin
- **Windows-Probleme**: Bei Installationsproblemen unter Windows verwenden Sie CrossRef-DOI-Auflösung oder WSL
- **Fallback-Verhalten**: ALIMA erkennt automatisch verfügbare Funktionen und passt sich entsprechend an
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

### 3. GUI-Nutzung

Die GUI-Anwendung bietet eine moderne, tab-basierte Oberfläche für die Analyse von Texten und die Verwaltung von Schlagworten.

**Starten:**
```bash
python3 main.py
```

**Hauptfunktionen:**

**Pipeline-Tab (Neu!):**
- **Auto-Pipeline**: Vollständige Analyse mit einem Klick
- **Vertikaler Workflow**: Chat-ähnliche Darstellung der 5 Pipeline-Schritte
- **Live-Streaming**: Echtzeitanzeige der LLM-Token-Generierung
- **Visuelle Status-Indikatoren**: ▷ (Wartend), ▶ (Läuft), ✓ (Abgeschlossen), ✗ (Fehler)
- **Integrierte Eingabe**: DOI, URL, PDF, Bild oder Text direkt im ersten Schritt
- **Keyword Chunking**: Automatische Aufteilung großer Keyword-Sets (>500 Keywords)
- **JSON Export/Import**: Speichern und Fortsetzen von Pipeline-Zuständen - teilweise getestet

**Eingabemöglichkeiten:**
- **DOI-Eingabe**: Automatische Auflösung von Springer und CrossRef DOIs
- **URL-Eingabe**: Web-Crawling mit verbesserter Springer-Unterstützung - getestet -> es wird nur die ersten Seite des Inhaltsverzeichnis mit übernommen
- **PDF-Analyse**: Texterkennung aus PDF-Dokumenten - geplant
- **Bildanalyse**: KI-basierte Bildbeschreibung und Schlagwortextraktion - geplant
- **Direkteingabe**: Manueller Text oder Clipboard-Import - getestet

**Weitere Tabs:**
- **Schlagwortsuche**: GND-Keyword-Suche und -verwaltung
- **Analyse-Review**: Detaillierte Ergebnisdarstellung und -validierung
- **CrossRef-Suche**: DOI-Metadaten-Lookup
- **UB-Suche**: Universitätsbibliothek-spezifische Suche

**Globale Statusleiste:**
- **Provider-Info**: Aktueller LLM-Provider und Modell
- **Cache-Statistiken**: Live-Anzeige der Datenbankgröße und Einträge
- **Pipeline-Fortschritt**: Echtzeitanzeige des aktuellen Schritts


## 🚀 Aktuelle Pipeline-Funktionen (Version 1.0)

### ✅ Vollständig implementiert und getestet

**Unified Pipeline Architecture:**
- **CLI & GUI Parität**: Beide Interfaces verwenden identische Pipeline-Logik
- **5-Stufen-Workflow**: Input → Initialisierung → Suche → Verifikation → Klassifikation (letzter  Schritt läuft noch nicht)
- **Keyword Chunking**: Intelligente Aufteilung großer Keyword-Sets mit konfigurierbarem Schwellenwert
- **Enhanced DOI/URL Support**: Springer-Crawling mit Inhaltsverzeichnis und CrossRef-API-Integration
- **Real-time Streaming**: Live-Token-Display während LLM-Generierung
- **JSON Persistence**: Vollständiges Speichern/Fortsetzen von Pipeline-Zuständen

**LLM Provider Support:**
- **Ollama**: Lokale Modelle (cogito:14b, cogito:32b, magistral:latest, etc.)
- **Gemini**: Google AI Platform (gemini-1.5-flash, gemini-1.5-pro)
- **OpenAI**: GPT-3.5/4 Modelle - deaktiviert, aber ChatAI von GWDG funktioniert 
- **Anthropic**: Claude-Modelle - deaktiviert
- **Multi-Model Pipelines**: Verschiedene Modelle für verschiedene Schritte

**Input Sources:**
- **Text**: Direkteingabe, Clipboard-Import - getestet
- **DOI**: Automatische Auflösung (10.1007/... für Springer, andere über CrossRef) - getestet
- **URL**: Web-Crawling (Springer-optimiert, generische Websites) - getestet
- **PDF**: Texterkennung (geplant)
- **Images**: KI-basierte Bildbeschreibung (geplant)

### 🔧 Technische Highlights

**Keyword Chunking System:**
- Automatische Erkennung bei >500 Keywords (konfigurierbar)
- Gleichmäßige Chunk-Verteilung (z.B. 1669 Keywords → 4 Chunks [418, 417, 417, 417])
- Enhanced Keyword Recognition mit exakter String- und GND-ID-Erkennung
- Deduplizierung basierend auf Wort- oder GND-ID-Matching
- Finale Konsolidierung aller Chunk-Ergebnisse

**Unified DOI/URL Resolution:**
- **Springer Detection**: Automatische Erkennung von Springer-URLs und DOIs
- **Table of Contents Extraction**: Vollständige ToC-Extraktion für bessere Keyword-Analyse
- **Fallback Mechanisms**: CrossRef API als Backup für nicht-Springer DOIs
- **Error Handling**: Graceful Degradation bei crawl4ai-Problemen

## 📋 Systemstatus

### ✅ Produktionsreif
- **CLI Pipeline Command**: Vollständig funktional mit allen Features
- **GUI Pipeline Tab**: Auto-Pipeline-Button und Echtzeit-Feedback
- **Keyword Chunking**: Getestet mit großen Keyword-Sets (>1500 Keywords)
- **DOI/URL Resolution**: Springer und CrossRef vollständig integriert

### 🔄 In Entwicklung
- **Batch Processing**: Mehrere Inputs parallel verarbeiten
- **Pipeline Templates**: Speichern/Laden verschiedener Workflow-Konfigurationen
- **Advanced Export**: Pipeline-Ergebnisse in verschiedenen Formaten (PDF, Excel)

### ⚠️ Bekannte Einschränkungen
- **crawl4ai unter Windows**: Installationsprobleme aufgrund von Systemabhängigkeiten - gibts auch unter Linux, habe es aber unter Windows nicht weiter testen können
- **Große Keyword-Sets**: Performance-Optimierung bei >2000 Keywords in Arbeit - Chunking sollte das Erledigen oder Gemini-Modelle
- **Gemini Seed Requirements**: Gemini-Modelle müssen mit Seed=0 gestartet werden

### 🧪 Getestete Modelle
- **Optimal**: cogito:32b für finale Keyword-Verifikation
- **Gut**: cogito:14b für Initialisierung, Gemma 3 27B für beide Schritte
- **Experimentell**: magistral:latest, gemini-1.5-flash für spezielle Anwendungen

# Lizenz
LGPL v3

# Mitwirkende
Conrad Hübler
Claude und Gemini AI
# Danksagung
Besten Dank an das Fachreferats- und IT-Team der Universitätsbibliothek. Besonderer Dank an Patrick Reichel für die effiziente Lobid-Abfrage.

# Kontakt
Conrad Hübler
