# Webapp Module - Web Interface for ALIMA Pipeline

## Overview

The `src/webapp/` directory provides a FastAPI-based web interface for the ALIMA pipeline, making the pipeline widget accessible via browser without PyQt6.

## Architecture

**Backend** (`app.py`):
- FastAPI REST API with WebSocket support
- Direct PipelineManager integration (same as CLI/GUI)
- AppContext singleton for lazy service initialization
- Session management (in-memory, per-request lifecycle)
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
✅ **Input Modes** - Text, DOI/URL, PDF, Images, Webcam capture
✅ **Drag & Drop Upload** - Intuitive file upload with visual feedback
✅ **Webcam Integration** - Capture images directly from browser
✅ **Live Feedback** - WebSocket streaming of progress updates with large output window
✅ **Session Management** - Individual analysis sessions with IDs
✅ **JSON Export** - Complete results downloadable
✅ **Responsive Design** - Works on desktop, tablet, and mobile
✅ **Modern UI** - Clean, professional design without unnecessary decorations

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

- **Pipeline Integration**: Direct PipelineManager with callbacks (shared with CLI/GUI)
- **Service Initialization**: AppContext singleton initializes services on first use (lazy init)
- **Initialization Sequence**: ConfigManager → LlmService → PromptService → AlimaManager → UnifiedKnowledgeManager → PipelineManager (6-step pattern)
- **Callbacks**: `step_started`, `step_completed`, `step_error`, `pipeline_completed`, `stream_callback`
- **WebSocket**: Live progress updates via callbacks (with HTTP polling fallback)
- **File Handling**: Uploaded files saved to temp directory with per-session cleanup
- **Drag & Drop**: Uses dragenter/dragover/drop events for file upload with visual feedback
- **Webcam**: getUserMedia API for camera access, canvas for image capture, converted to JPEG
- **Token Streaming**: Tokens buffered per step, transmitted every 500ms without extra whitespace

## Usage

Start server:
```bash
python3 src/alima_webapp.py
```

Then visit `http://localhost:8000` in browser.

## Testing & Deployment

- **Requirements**: Install all dependencies with `pip install -r requirements.txt`
- **Headless Environments**: PyQt6.QtSql requires graphics libraries (libEGL)
  - Run with xvfb: `xvfb-run -a python3 src/alima_webapp.py`
  - Or in environment with display server

## Future Enhancements

- Drag & drop file upload
- Webcam image capture
- Persistent session storage (database backend)
- Authentication/user accounts
- Batch processing UI
- Result history/session recall
- Docker containerization
