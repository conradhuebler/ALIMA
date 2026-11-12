# ALIMA Webapp

Web-basierte Schnittstelle für die ALIMA Pipeline.

## Features

✅ **Schrittleiste** - Die Schritte des gewählten Workflows, live hervorgehoben
✅ **Input-Modi** - Text, DOI/URL, PDF, Bilder
✅ **Live-Feedback** - WebSocket-basierte Echtzeitupdates
✅ **Ergebnis-Reiter** - „Log / Chat“, „Ergebnis“ (Abschlusskarte mit WinIBW-Eintrag, lokalen Treffern und Quellen) und „Klassifikation“ (DK/RVK-Karten); ein Reiter füllt den Platz und scrollt in sich
✅ **JSON-Export** - Vollständige Analyseergebnisse als JSON
✅ **Ergebnis laden** - Ein gespeichertes Ergebnis (Export der Webapp oder in GUI/CLI gespeicherter Analysezustand) ansehen, ohne eine Analyse zu starten
✅ **Responsive Design** - Funktioniert auf Desktop und Mobile

## Installation

### Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

FastAPI, Uvicorn und python-multipart werden benötigt.

## Verwendung

### 1. Server starten

```bash
python3 src/alima_webapp.py
```

Der Server läuft dann unter `http://localhost:8000`

### Datenbank-Empfehlung für Mehrbenutzerbetrieb

- `sqlite`: geeignet für Entwicklung, lokale Tests und leichte Einzelprozess-Nutzung
- `mysql`/`mariadb`: empfohlen für mehrere gleichzeitige Webapp-Nutzer und parallele CLI-Nutzung

Wichtig:
- Die Webapp hält aktive Sessions aktuell im Prozessspeicher.
- Ein einzelner Webserver-Prozess kann mehrere gleichzeitige Nutzer bedienen.
- Mehrere Worker/Prozesse hinter einem Load Balancer sind derzeit nicht vollständig unterstützt, weil Session-Status nicht in einem externen Store liegt.

### 2. Webapp öffnen

Öffnen Sie im Browser: `http://localhost:8000`

### 3. Analyse durchführen

1. **Input wählen**: Text, DOI/URL, oder Datei
2. **Analyse starten** klicken
3. **Live-Feedback** im Reiter „Log / Chat“ ansehen (aktualisiert in Echtzeit)
4. Nach dem Lauf zeigt die Kopfzeile der Eingabe „Analyse abgeschlossen“, und die Reiter „Ergebnis“ und „Klassifikation“ sind markiert. Die Ansicht wechselt nicht von selbst.
5. **Speichern** exportiert das Ergebnis als JSON

### 4. Gespeichertes Ergebnis ansehen

„Ergebnis laden…“ in der Kopfzeile der Eingabe öffnet eine JSON-Datei, die ALIMA
gespeichert hat: den Export der Webapp („Speichern“) oder einen in GUI oder CLI
gespeicherten Analysezustand. Das Ergebnis erscheint in den Reitern, gekennzeichnet
mit „Ergebnis geladen“ und dem Dateinamen; es läuft keine Analyse. Kopieren,
Aufklappen und „Speichern“ funktionieren wie nach einem Lauf.

- Höchstens 20 MB; die Datei muss UTF-8-JSON mit einem der beiden Formate sein,
  sonst erscheint eine Meldung neben dem Knopf.
- Während einer Analyse ist Laden gesperrt.
- Ein geladenes Ergebnis ersetzt das bisher angezeigte; eine neue Analyse ersetzt
  das geladene. Das Log nennt jeweils, was ersetzt wurde. Die geladene Datei
  bleibt unverändert; das Ergebnis eines vorigen Laufs bleibt nur erhalten, wenn
  es gespeichert wurde.
- Ein Webapp-Export enthält nicht alles, was ein Lauf zeigt: Es fehlen der
  Auszug einer Zusatzregel in der Ergebniskarte, ein Workflow-Bericht und die
  Liste der Zusatzregeln des Laufs. Ein gespeicherter Analysezustand enthält sie.

## API Endpoints

### Session Management

- `POST /api/session` - Neue Session erstellen
- `GET /api/session/{session_id}` - Session-Status abrufen
- `DELETE /api/session/{session_id}` - Session löschen

### Analyse

- `POST /api/analyze/{session_id}` - Analyse starten
  - `input_type`: "text", "doi", "pdf", "img"
  - `content`: Text/DOI
  - `file`: PDF/Bild Datei

- `GET /api/export/{session_id}?format=json` - JSON exportieren
- `POST /api/session/{session_id}/load` - Gespeichertes Ergebnis anzeigen
  - `file`: JSON-Datei (Export der Webapp oder gespeicherter Analysezustand), höchstens 20 MB
  - Antwort: Render-Ereignisse und Ergebnisse; 413 (zu groß), 422 (kein ALIMA-Ergebnis, mit Meldung), 409 (Analyse läuft)

### WebSocket

- `WS /ws/{session_id}` - Live Progress Updates

## Architektur

### Backend (`app.py`)

- FastAPI Server mit RESTful API
- Führt die Pipeline direkt im Prozess aus
- WebSocket für Live-Streaming
- Session Management

### Frontend (`static/app.js`)

- Vanilla JavaScript (keine Dependencies)
- Tab-Navigation für verschiedene Input-Modi
- Pipeline Visualization
- WebSocket Client für Live Updates
- JSON Export Handler

## Workflow

```
1. Frontend: Benutzer gibt Eingabe ein
   ↓
2. Frontend: POST /api/analyze mit Eingabe
   ↓
3. Backend: Startet PipelineManager direkt
   ↓
4. Backend: Streaming-Output via WebSocket
   ↓
5. Frontend: Visualisiert Steps in Echtzeit
   ↓
6. Benutzer: Exportiert JSON
```

## Entwicklung

### Ports

- **8000** - FastAPI Server
- **WebSocket** - `/ws/{session_id}`

### Logging

Alle requests und Fehler werden in der Console geloggt:

```bash
[INFO] GET /
[INFO] POST /api/session
[INFO] POST /api/analyze/abc123
[INFO] Starting WebSocket for abc123
```

## Konfiguration

Die Webapp nutzt die gleiche `config.json` wie die CLI und GUI:

```bash
~/.config/alima/config.json  # Linux/macOS
%APPDATA%\ALIMA\config.json  # Windows
```

Provider, Modelle und Einstellungen werden automatisch übernommen.

Für produktiven Mehrbenutzerbetrieb sollte die Datenbank in `config.json` auf `mysql` oder `mariadb` gesetzt werden.

## Fehlerbehebung

### Port 8000 bereits belegt

```bash
# Anderen Port verwenden (z.B. 8001)
python3 -c "import uvicorn; uvicorn.run('src.webapp.app:app', host='0.0.0.0', port=8001)"
```

### WebSocket Fehler

- Browser-Console prüfen (F12)
- Stellen Sie sicher, dass Server läuft
- Firewall kann WebSocket blockieren (rare)

### CLI Fehler

Die CLI wird unter `PROJECT_ROOT` ausgeführt. Stellen Sie sicher:
- `alima_cli.py` ist erreichbar
- Alle Config-Dateien sind vorhanden
- Dependencies sind installiert

## Zukunftspläne

- [ ] Batch-Processing UI
- [ ] Analyseverlauf speichern
- [ ] Docker-Support
- [ ] API Authentication
