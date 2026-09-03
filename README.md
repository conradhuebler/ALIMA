# ALIMA — Automated Library Indexing and Metadata Assignment

**Sacherschließung mit Large Language Models.** ALIMA erzeugt GND-konforme Schlagwörter und DK/RVK-Klassifikationen aus Dokumentinhalten.

Entwickelt an der Universitätsbibliothek "Georgius Agricola" der TU Bergakademie Freiberg.

## Publikation

Hübler, Conrad: *ALIMA – Ein RAG-basiertes System zur LLM-gestützten Sacherschließung: Prototypentwicklung und erste Erfahrungen aus der Praxis.* In: Bibliothek Forschung und Praxis 50 (2026), Heft 2, S. 197–216.
DOI: [10.1515/bfp-2026-0014](https://doi.org/10.1515/bfp-2026-0014). Open Access (CC BY 4.0).

## Anmerkung des Autors
ALIMA sowie die Dokumentation sind größtenteils mit Claude erstellt. LLM sind oft übereuphorisch und sehen Dinge als fertig an, auch wenn sie es noch nicht sind. Gleichzeitig ändern sich dank Agentic Coding Dinge schneller, als sie dokumentiert werden können. ALIMA ist demnach noch in der Entwicklung und noch nicht alle dokumentierten Funktionen sind als fertig anzusehen.

## Überblick

ALIMA ist eine Python-basierte Anwendung für **Sacherschließung (Library Indexing)** mit künstlicher Intelligenz. Sie analysiert Dokumente und schlägt standardisierte, GND-konforme Schlagwörter sowie DK/RVK-Klassifikationen vor. Die Vorschläge werden in der Oberfläche geprüft und von dort in den Katalog übernommen.

**Sacherschließung** (auch "Indexierung" oder "Metadaten-Zuweisung" genannt) ist der bibliothekarische Prozess, ein Dokument durch standardisierte Schlagwörter und Klassifikationssysteme zu beschreiben, damit es optimal auffindbar wird. ALIMA erzeugt diese Beschreibung mit großen Sprachmodellen (LLMs).

## Hauptfunktionen

*   **🎯 Sacherschließungsvorschläge:** GND-Schlagwörter und DK/RVK-Klassifikationen aus Dokumentinhalten, in einem Lauf ohne Zwischeneingriff erzeugt
*   **🔄 Klassische 5-Schritt-Pipeline:** `input` → `initialisation` (freie Schlagwörter) → `search` (GND-Suche) → `keywords` (Abgleich gegen die GND) → `dk_classification`
*   **🧩 Agentischer Modus:** statt der festen Schrittfolge ein YAML-Workflow aus `workflows/` (`--agentic`)
*   **📥 Flexible Dateneingabe:** Texte, DOI/URL-Auflösung, PDF-Extraktion, OCR von Bildern und gescannten Dokumenten
*   **🤖 Multi-Provider LLM-Support:** Funktioniert mit Ollama, Claude, Gemini, OpenAI und kompatiblen APIs
*   **⚡ Modellwahl je Aufgabe:** Prioritätenliste pro Pipeline-Schritt (`task_preferences`), Rückfall auf `provider_priority`
*   **📦 Batch-Verarbeitung:** Listen von Quellen nacheinander analysieren, Ergebnisse als JSON ablegen
    *   Batch-Datei: Textdatei mit Quellen (DOI:..., ISBN:..., etc.)
    *   **Manuelle Eingabe:** DOIs/ISBNs direkt per Copy-Paste eingeben
    *   Paketsigel: K10plus-Sigel (`--siegel`), wird zu einer DOI-Liste expandiert
    *   Datei-Auswahl: Einzelne PDFs/Bilder direkt auswählen
    *   Verzeichnis-Scan: Alle Dateien eines Ordners automatisch erfassen
*   **🖥️ GUI (PyQt6):** Desktop-Anwendung mit visuellem Pipeline-Workflow und detaillierter Ergebnisanalyse
*   **🌐 WebAPP:** Server-basierte Schnittstelle mit derselben Konfiguration wie GUI und CLI
*   **💬 Chat-Agent:** derselbe Werkzeugsatz in GUI, CLI (`agent`) und WebAPP; liest den Analysezustand, schlägt Änderungen zur Bestätigung vor, startet Läufe
*   **🔌 Plugin-System:** Suchquellen, Nachschlagedienste und Eingabequellen als konfigurierbare Instanzen im Abschnitt `plugins`
*   **📤 Tagzeilen-Export:** Ergebnisse als PICA-Tagzeilen (Zwischenablage oder Textdatei) zum Einfügen in WinIBW

Hinweis zum Mehrbenutzerbetrieb:
- Für lokale Entwicklung und Einzelplatznutzung reicht SQLite.
- Für mehrere gleichzeitige WebAPP-Nutzer und parallele CLI-Nutzung sollte `mysql` oder `mariadb` als Datenbank konfiguriert werden.
- Die aktuelle WebAPP verwaltet aktive Sessions im Prozessspeicher; ein einzelner Serverprozess kann mehrere Nutzer bedienen, aber ein Multi-Worker-Setup benötigt zusätzlich einen externen Session-Store.

## Installation

### Voraussetzungen

*   Python 3.11 oder 3.12. Die Untergrenze stammt aus den Pins in `requirements.txt` (`numpy` verlangt >= 3.11, `fastapi`/`uvicorn`/`Crawl4AI` >= 3.10). Unter Python 3.13 schlägt das Auslesen von Webseiten über crawl4ai fehl.
*   PyQt6
*   diverse Pakete (openai, ollama, crawl4ai, playwright) siehe requirements.txt

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

Für eine Konfiguration ohne lokalen SRU-Katalog, aber mit **Lobid** für die Schlagwortsuche und **GVK/GBV-SRU** für DK/RVK-Klassifikationen, gibt es ein Beispiel in `config.example.lobid-gbv.json`. Der SRU-Preset heißt in ALIMA `gbv` und verweist auf den GVK-Endpunkt (`https://sru.gbv.de/gvk`). Die Beispieldatei liegt in der älteren `catalog_config`-Form vor und wird beim ersten Laden in Plugin-Instanzen überführt.

### Struktur der `config.json`

Die `config.json` ist in mehrere Hauptbereiche unterteilt. Der wichtigste Abschnitt für die Steuerung der KI-Analyse ist `unified_config`. Suchquellen und Kataloge stehen als Plugin-Instanzen unter `plugins`; Konfigurationen mit dem älteren Abschnitt `catalog_config` werden beim Laden einmalig migriert.

```json
{
    "database_config": { ... },
    "prompt_config": { ... },
    "system_config": { ... },
    "ui_config": { ... },
    "plugins": [ ... ],
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

### Plugins

Suchquellen, Nachschlagedienste und Eingabequellen sind Plugin-Instanzen. Registriert sind drei Kategorien:

*   **search** (`src/core/search/providers/`): `lobid`, `swb`, `catalog`, `finc`, `sru`, `kvk`, `gnd_local`.
*   **lookup** (`src/utils/lookups/`): `k10plus`, `dnb`, `rvk_api`, `webindex`.
*   **input** (`src/utils/input_sources/`): `text`, `file`, `pdf`, `image`, `doi_crossref`, `doi_openalex`, `doi_datacite`, `url_fetch`, `isbn`, `ppn`.

Konfiguriert werden die Instanzen im Abschnitt `plugins` der `config.json` oder in der GUI unter Einstellungen im Tab "🔌 Plugins". Das MCP-Werkzeug `list_plugins` gibt den aktuellen Stand aus. Für die verteilte Installation an einer Einrichtung gibt es `python3 src/alima_cli.py bundle` (Plugins plus Konfigurationsprofil), beschrieben in [`docs/institutional_bundles.md`](docs/institutional_bundles.md).

Plugins mit eigenem Python-Code muss der Betreiber einmal freigeben. Dabei läuft eine AST-Prüfung auf auffällige Konstrukte, und ein SHA-256 über die Plugin-Dateien erzwingt nach jeder Änderung eine erneute Freigabe. Beides sind Hürden, keine Isolierung: ein freigegebenes Plugin wird im selben Prozess mit denselben Rechten wie ALIMA ausgeführt. Einzelheiten in [`docs/plugin_system.md`](docs/plugin_system.md) und [`docs/plugin_authoring.md`](docs/plugin_authoring.md).

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
    *   **PDF-Dateien:** Das System extrahiert automatisch den Text aus der PDF. Bei gescannten Dokumenten oder PDFs ohne Textebene wird eine KI-basierte OCR (Texterkennung) versucht. Dafür werden `pdf2image` und eine installierte Poppler-Umgebung benötigt.
    *   **Bild-Dateien:** Bei Bildformaten (PNG, JPG etc.) wird automatisch eine KI-basierte OCR gestartet, um den im Bild enthaltenen Text zu extrahieren.

Hinweis: Unter macOS kann Poppler z.B. mit `brew install poppler` installiert werden.

Nach der erfolgreichen Extraktion der Daten aus einer dieser Quellen können Sie die Analyse mit dem "🚀 Auto-Pipeline"-Button starten.

### 1.2 Nutzung der WebAPP

Die WebAPP setzt eine vorhandene Konfiguration und eine mit dem GND-Abzug gefüllte Datenbank voraus. Beides erzeugt entweder der First-Start-Wizard der GUI oder die CLI (`python3 src/alima_cli.py setup` und `python3 src/alima_cli.py dnb-import`). Anschließend starten Sie mit
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

Für die Analyse gibt es zwei Befehle: `pipeline` für Einzelanalysen und `batch` für die Stapelverarbeitung. Daneben stellt die CLI Verwaltungsbefehle bereit, unter anderem `setup`, `dnb-import`, `search`, `list-models`, `list-providers`, `test-providers`, `db-config`, `provider` und `clear-cache`. `python3 src/alima_cli.py --help` listet alle auf.

#### 2.1. Einzelanalyse (`pipeline`-Befehl)

Der `pipeline`-Befehl führt eine vollständige Analyse für eine einzelne Datenquelle durch.

**Eingabe-Optionen (einer erforderlich):**
*   `--input-text "..."`: Direkte Texteingabe.
*   `--doi "..."`: Eingabe einer DOI oder einer URL (wird automatisch aufgelöst).
*   `--input-image <pfad>`: Bilddatei, die per OCR ausgelesen wird.

**Beispiel:**
```bash
# Führt eine Standard-Analyse für einen Text durch
python3 src/alima_cli.py pipeline --input-text "Ein Text über das Recycling von Lithium-Ionen-Batterien."

# Führt eine Analyse für eine DOI durch und speichert das Ergebnis
python3 src/alima_cli.py pipeline --doi "10.1007/s00442-021-04908-x" --output-json ergebnis.json
```

**Agentischer Modus:**
Ohne weitere Angabe läuft die klassische Schrittfolge. `--agentic` ersetzt sie
durch einen YAML-Workflow aus `workflows/`:

```bash
python3 src/alima_cli.py pipeline --doi "10.1007/s00442-021-04908-x" --agentic --workflow alima_v51
```

*   `--workflow <name>`: Workflow aus `workflows/` (Standard: `system_config.default_workflow`, ab Werk `alima_v51`).
*   `--custom-workflow <pfad>`: Workflow-Datei außerhalb von `workflows/`.
*   `--override <PROVIDER/MODELL>`: setzt Provider und Modell für alle LLM-Schritte.

#### 2.2. Stapelverarbeitung (`batch`-Befehl)

Der `batch`-Befehl verarbeitet eine Liste von Datenquellen nacheinander.

**Argumente:**
*   `--batch-file <pfad>`: Pfad zu einer Textdatei, die die zu verarbeitenden Quellen enthält (eine pro Zeile).
*   `--output-dir <ordner>`: Ordner, in dem die JSON-Ergebnisdateien gespeichert werden.

**Format der Batch-Datei:**
Jede Zeile muss das Format `TYP:WERT` haben. Unterstützte Typen sind `DOI`, `ISBN`, `PPN`, `URL`, `PDF`, `IMG`, `TXT`. Bare DOIs (z.B. `10.1234/example`) werden automatisch erkannt.

```
# Beispiel batch_sources.txt
DOI:10.1007/s00442-021-04908-x
URL:https://www.tagesschau.de/wirtschaft/verbraucher/recycling-lithium-ionen-akkus-100.html
PDF:/home/user/docs/studie.pdf
IMG:/home/user/images/infografik.png
TXT:/home/user/docs/abstract.txt
```

**Manuelle Eingabe (GUI-Batch):**
Im Batch-Verarbeitungsdialog kann man DOIs und andere Quellen auch direkt per Copy-Paste eingeben:

1. Tab "✏️ Manuelles Eingeben" auswählen
2. DOIs/ISBNs/PPNs in das Textfeld einfügen (eine pro Zeile oder Format `TYP:WERT`)
3. "Vorschau" klicken → Quellen werden geparst
4. Gewünschte Quellen per Checkbox auswählen
5. "Start" klicken

```
# Beispiele für manuelle Eingabe
10.1234/example-paper        # Wird automatisch als DOI erkannt
DOI:10.5678/another-paper    # Explizites DOI-Format
ISBN:9783662123456          # ISBN
PPN:1234567890              # PPN (K10Plus)
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
*   `--format <detailed|compact|k10plus>`: Ausgabeformat (Standard: `detailed`)
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

**Mehrere Dateien in eine CSV:**
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

Die Pipeline weist zu jeder DK-Klassifikation die **Katalog-Titel** aus, die zu ihr geführt haben.

**GUI - Real-time während Pipeline-Ausführung:**
```
[DK_SEARCH] 🔍 DK-Suche: 2 Klassifikationen gefunden (45 Titel)
  📊 DK 628.5 (45 Titel): Cadmium in der Umwelt | Bodenverschmutzung | ...
  📊 DK 333.3 (8 Titel): Umweltschutzmaßnahmen | Nachhaltiger Umgang | ...
```

**GUI - Detaillierte Ansicht im Analysis Review Tab:**
- Tab "DK/RVK" zeigt die Klassifikationen farbcodiert nach Zahl der Titel
- Grün, "Very High": mehr als 50 Titel
- Blau, "High": mehr als 20
- Gelb, "Medium": mehr als 5
- Rot, "Low": 5 und weniger
- Ausklappbare Titellisten unter jeder Klassifikation

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

#### 2.5. Tagzeilen-Export für WinIBW

ALIMA gibt die Analyseergebnisse als PICA-Tagzeilen aus. Die Zeilen gehen über
die Zwischenablage oder eine Textdatei nach WinIBW; eine direkte Verbindung zum
Katalog besteht nicht.

Die Kategorien folgen der Erfassungspraxis der UB Freiberg:

```
5550 Schlagwort
6700 DK CODE
```

GND-Nummern werden entfernt, es bleiben Begriff und Notation.

**GUI:**
Die "Analysis Review" enthält den Tab "K10+ Export" mit den erzeugten Zeilen und
dem Button "📋 In Zwischenablage kopieren". Die Klassifikationszeile behält das
System der Notation (`6700 DK 628.5`, `6700 RVK AR 25140`).

**CLI:**
```bash
python3 src/alima_cli.py show-protocol ergebnis.json --format k10plus

5550 Cadmium
5550 Bodenverschmutzung
5550 Umweltverschmutzung
6700 DK 628.5
6700 DK 333.3
```

Die CLI-Ausgabe setzt vor jede Notation `DK`, unabhängig vom System der
Klassifikation.

**Mehrere Dateien:**
```bash
for json in results/*.json; do
    python3 src/alima_cli.py show-protocol "$json" --format k10plus
done > k10plus_export.txt
```

### 3. Workflows

Die agentischen Läufe sind YAML-Dateien in `workflows/`. Eine Datei beschreibt die Schritte, ihre Prompts, die erlaubten Werkzeuge und die Modellparameter. Schema: [`docs/workflow_yaml_spec.md`](docs/workflow_yaml_spec.md), Ausführung: [`docs/agentic_workflow.md`](docs/agentic_workflow.md).

```bash
python3 src/alima_cli.py workflows list
python3 src/alima_cli.py workflow catalog_search --input-file anfrage.json --output bericht.json
```

`workflow <name>` führt einen Workflow eigenständig aus, unabhängig von der Sacherschließungspipeline; `--only-step` beschränkt den Lauf auf einen Schritt. In der GUI liegt der Editor unter Bearbeiten → "📋 Workflow-Editor" und im Pipeline-Konfigurationsdialog. Die WebAPP liest die Liste über `GET /api/workflows`.

Jede Datei trägt ein Feld `status`, das `workflows list` mit ausgibt. Vergeben wird es vom Betreiber:

*   `tested`: `alima_v51` (allgemein) und `alima_v51_105` (Freiberger Variante, RVK nur für Wirtschaftswissenschaften). Diese beiden tragen die Sacherschließung und sind an echtem Material erprobt.
*   `research`: alle übrigen, darunter `catalog_search`, `synonym_expansion`, `title_list_search`, `webindex_keywords`, `website_rag`, `batch_metadata`, `research_deep`, `main_agent` sowie die älteren Pipeline-Fassungen `alima`, `alima_classic` und `alima_classic_v51`. Sie sind nicht abgenommen; was sie vorhaben, steht in ihrer `description`.

### 4. Chat-Agent

Derselbe `AgentLoop` mit demselben Werkzeugsatz läuft an drei Stellen: im Chat-Panel rechts neben dem Pipeline-Tab der GUI, als CLI-Befehl `agent` und in der WebAPP (`POST /agent/run`, `POST /api/session/{id}/chat`).

```bash
python3 src/alima_cli.py agent --doi "10.1007/s00442-021-04908-x" --prompt "Prüfe die Schlagwörter gegen die GND"
```

*   `--doi`, `--input`, `--input-file`, `--input-image`: Quelle des Werks.
*   `--mode <verschlagwortung|suche|general|auto>`: Systemprompt (Standard: `auto`).
*   `--autonomous`: führt Änderungen und Pipeline-Starts ohne Rückfrage aus.
*   `--output <datei>`: schreibt das JSON-Ergebnis in eine Datei statt nach stdout.
*   `--max-iterations`: Obergrenze der Werkzeugaufrufe.

**Werkzeuge.** Lesend auf den aktuellen Analysezustand (`get_keywords`, `get_gnd_entries`, `get_dk_classifications`, `get_dk_titles_for_code`, `search_in_gnd_pool` und weitere), dazu die MCP-Werkzeuge für Suche und Beschaffung (`search_gnd`, `rvk_lookup`, `resolve_doi`, `read_pdf`, `scrape_url`, `execute_workflow`, `export_results`). Schreibend gibt es `propose_keyword_replacement` und `propose_dk_change`: beide schreiben eine Zeile ins Änderungsprotokoll und warten auf die Zustimmung im Chat, solange `--autonomous` beziehungsweise `chat_config.autonomous_pipeline` nicht gesetzt ist. `run_pipeline` und `rerun_step` starten Läufe.

## Lizenz
LGPL v3

## Mitwirkende
Conrad Hübler
Claude und Gemini AI

## Danksagung
Besten Dank an das Fachreferats- und IT-Team der Universitätsbibliothek. Besonderer Dank an Patrick Reichel für die effiziente Lobid-Abfrage.

## Kontakt
Conrad Hübler
