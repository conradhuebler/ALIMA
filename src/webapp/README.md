# ALIMA Webapp

Web-basierte Schnittstelle für die ALIMA Pipeline.

## Features

✅ **Pipeline Widget** - Visuelle Darstellung der 5 Analyseschritte
✅ **Input-Modi** - Text, DOI/URL, PDF, Bilder
✅ **Live-Feedback** - WebSocket-basierte Echtzeitupdates
✅ **JSON-Export** - Vollständige Analyseergebnisse als JSON
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

### 2. Webapp öffnen

Öffnen Sie im Browser: `http://localhost:8000`

### 3. Analyse durchführen

1. **Input wählen**: Text, DOI/URL, oder Datei
2. **Auto-Pipeline Button** klicken
3. **Live-Feedback** ansehen (aktualisiert in Echtzeit)
4. **Ergebnisse exportieren** als JSON

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

### WebSocket

- `WS /ws/{session_id}` - Live Progress Updates

## Architektur

### Backend (`app.py`)

- FastAPI Server mit RESTful API
- Ruft ALIMA CLI auf für Analysen
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
3. Backend: Ruft alima_cli.py auf
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

- [ ] Drag & Drop für Dateien
- [ ] Webcam-Integration
- [ ] Batch-Processing UI
- [ ] Analyseverlauf speichern
- [ ] Docker-Support
- [ ] API Authentication
