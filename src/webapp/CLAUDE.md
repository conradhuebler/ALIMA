# Webapp Module - Web Interface for ALIMA Pipeline

## Overview

The `src/webapp/` directory provides a FastAPI-based web interface for the ALIMA pipeline, making the pipeline widget accessible via browser without PyQt6.

## Architecture

**Backend** (`app.py`):
- FastAPI REST API with WebSocket support
- Calls ALIMA CLI for analysis execution
- Session management (in-memory for now)
- JSON export of results
- CORS enabled for development

**Frontend** (`static/`):
- Vanilla JavaScript (no dependencies)
- Tab-based input UI (text, DOI, file upload)
- Pipeline step visualization (5-step workflow)
- WebSocket client for live progress
- JSON download handler

**Launcher** (`../alima_webapp.py`):
- Simple entry point: `python3 src/alima_webapp.py`
- Runs on `localhost:8000`

## Current Features

✅ **Pipeline Widget as Webapp** - Full visualization of 5 analysis steps
✅ **Input Modes** - Text, DOI/URL, PDF, Images (uploaded via file input)
✅ **Live Feedback** - WebSocket streaming of progress updates
✅ **Session Management** - Individual analysis sessions with IDs
✅ **JSON Export** - Complete results downloadable
✅ **Responsive Design** - Works on desktop and tablets

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/session` | Create new session |
| GET | `/api/session/{id}` | Get session status |
| POST | `/api/analyze/{id}` | Start analysis with input |
| WS | `/ws/{id}` | WebSocket for live updates |
| GET | `/api/export/{id}` | Download JSON results |
| DELETE | `/api/session/{id}` | Delete session |

## Implementation Notes

- **CLI Integration**: Backend calls `alima_cli.py pipeline` with input arguments
- **Output Capture**: JSON results written to temp file, then returned to frontend
- **Streaming**: WebSocket sends status updates every 500ms during analysis
- **File Handling**: Uploaded files saved to temp directory with cleanup on session delete

## Usage

Start server:
```bash
python3 src/alima_webapp.py
```

Then visit `http://localhost:8000` in browser.

## Future Enhancements

- Drag & drop file upload
- Webcam image capture
- Persistent session storage (database)
- Authentication/user accounts
- Batch processing UI
- Result history
- Docker deployment support
