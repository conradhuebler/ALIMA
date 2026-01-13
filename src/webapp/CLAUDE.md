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
✅ **Extracted Text Display** - Shows text from PDF/image OCR extraction separately from LLM output
✅ **Live Feedback** - WebSocket streaming of progress updates with large output window
✅ **Session Management** - Individual analysis sessions with IDs
✅ **JSON Export** - Complete results downloadable
✅ **Responsive Design** - Works on desktop, tablet, and mobile
✅ **Modern UI** - Clean, professional German interface without unnecessary decorations
✅ **Auto-Save & Recovery** - Incremental saving after each step, recovery after WebSocket timeout (2026-01-06)
✅ **Extended WebSocket Timeout** - 30-minute timeout (increased from 5 min) with heartbeat mechanism
✅ **Progress Tracking** - Percentage display for long-running DK searches
✅ **Thread-Safe Streaming** - Streaming buffer protected with locks for concurrent access (2026-01-12)
✅ **LLM Queue System** - Max 3 concurrent LLM requests, unlimited direct pipeline execution (2026-01-13)
✅ **Global Queue Status** - Live badge showing LLM + Pipeline activity for all users (2026-01-13)

## LLM Queue System - Claude Generated (2026-01-13)

### Overview
- **Location**: Semaphore in `AlimaManager` class (`src/core/alima_manager.py`)
- **Max Concurrent LLM Calls**: 3 (configurable via code)
- **Scheduling**: Direct execution with blocking acquire on semaphore
- **Pipeline Execution**: Starts immediately, LLM calls may queue
- **Statistics**: Active count, pending count, average duration tracking

### Components
- **`_llm_semaphore`**: `threading.Semaphore(3)` in AlimaManager.__init__()
- **`_llm_queue_stats`**: Dict tracking active, pending, total_completed counts
- **`get_llm_queue_status()`**: Method returning current queue status for API
- **Metrics Endpoint**: `/api/queue/status` returns combined LLM + pipeline stats

### Global Queue Status Badge
- **Location**: Header badge showing "N/M LLM · X Pipelines"
- **Update Frequency**: Every 20 seconds via JavaScript polling
- **Display Logic**: Green (healthy) / Yellow (busy) indicator
- **Tooltip**: Shows LLM active/pending and pipeline count on hover

## Thread-Safety - Claude Generated (2026-01-12)

### Streaming Buffer Protection
- **Lock Type**: `threading.Lock()` on `Session._streaming_lock`
- **Protected Operations**:
  - `add_streaming_token()` - Buffer writes from background thread
  - `get_new_streaming_tokens()` - Buffer reads from WebSocket handler
  - `get_and_clear_streaming_buffer()` - Complete buffer flush
  - `clear()` - Session reset
- **Lock Scope**: Microseconds only (dict operations), NOT network I/O
- **Performance Impact**: < 1% overhead

## Tab Isolation - Claude Generated (2026-01-13)

### Multi-Tab Isolation Strategy (Option C)
- **Problem**: Multiple browser tabs share same HTML → DOM ID conflicts when analyzing simultaneously
- **Solution**: Server generates separate HTML per session with Jinja2 templates
- **Pattern**: Each browser tab gets unique URL `/webapp?session=ABC123` with injected sessionId
- **Isolation**: Each tab = separate HTML document → no DOM ID conflicts possible

### Backend Implementation
- **Route**: `GET /webapp` generates dynamic HTML with Jinja2 templating
- **Session Handling**:
  - No session parameter → generates new UUID
  - With `?session=ABC123` → uses provided session ID
- **Template Injection**: `window.sessionId = "{{ session_id }}"` injected into HTML
- **Stateless**: No session state in backend, each HTML is independent

### Frontend Implementation
- **sessionId Source**: JavaScript reads `window.sessionId` from HTML injection
- **Fallback Mode**: If `window.sessionId` not present, falls back to `/api/session` POST (backward compatibility)
- **No DOM Changes**: Original app.js works unchanged - each tab has its own HTML with original IDs

### Files
- **Backend**: `src/webapp/app.py` - `/webapp` route with Jinja2Templates
- **Template**: `src/webapp/templates/webapp.html` - Dynamic HTML with {{ session_id }} injection
- **Frontend**: `src/webapp/static/app.js` - Detects `window.sessionId` in `createNewSession()`
- **CSS**: `src/webapp/static/styles.css` - Unchanged (standard ID selectors work fine)

### Isolation Guarantees
- ✅ True isolation: each browser tab = separate HTML document
- ✅ No DOM ID conflicts possible (different documents)
- ✅ Unlimited concurrent browser tabs supported
- ✅ Backward compatible - old `/` route still works with `/api/session` fallback
- ✅ Simple to understand and debug - each tab is independent
- ✅ Zero overhead - just template rendering (~1-2ms per request)

### Usage
```
New analysis:     /webapp              → generates random session UUID
With existing ID: /webapp?session=abc  → uses session 'abc'
Old behavior:     /                    → serves index.html, uses /api/session API
```

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/session` | Create new session |
| GET | `/api/session/{id}` | Get session status |
| POST | `/api/analyze/{id}` | Submit analysis (routed through queue, NEW 2026-01-12) |
| GET | `/api/queue/status` | Get global queue statistics (NEW 2026-01-12) |
| GET | `/api/session/{id}/queue` | Get session queue position and ETA (NEW 2026-01-12) |
| WS | `/ws/{id}` | WebSocket for live updates (30 min timeout, heartbeat every 5s) |
| GET | `/api/export/{id}` | Download JSON results |
| GET | `/api/session/{id}/recover` | Recover results from auto-save after timeout (2026-01-06) |
| DELETE | `/api/session/{id}` | Delete session |

## Implementation Notes

- **Pipeline Integration**: Direct PipelineManager with callbacks (shared with CLI/GUI)
- **Service Initialization**: AppContext singleton initializes services on first use (lazy init)
- **Initialization Sequence**: ConfigManager → LlmService → PromptService → AlimaManager → UnifiedKnowledgeManager → PipelineManager (6-step pattern)
- **Callbacks**: `step_started`, `step_completed`, `step_error`, `pipeline_completed`, `stream_callback`
- **Queue Integration** (NEW 2026-01-12): PipelineQueueManager initialized on startup, all `/api/analyze` requests routed through queue
- **Queue Execution**: Queued requests executed when semaphore available, FIFO scheduling with persistent SQLite storage
- **Thread-Safety** (NEW 2026-01-12): `Session._streaming_lock` (threading.Lock) protects streaming_buffer from concurrent access (background thread vs WebSocket handler)
- **WebSocket**: Live progress updates via callbacks (with HTTP polling fallback), connects automatically when request starts executing
- **File Handling**: Uploaded files read immediately in request handler (prevents "read of closed file" error)
- **Drag & Drop**: Uses dragenter/dragover/drop events for file upload with visual feedback
- **Webcam**: getUserMedia API for camera access, canvas for image capture, converted to JPEG
- **Token Streaming**: Tokens buffered per step, transmitted every 500ms without extra whitespace, protected by threading.Lock
- **Extracted Text**: Displayed in separate panel from LLM output, extracted from input step's `original_abstract`
- **Input Type Routing**: Correct input_type (text/doi/pdf/img) passed to pipeline for proper processing
- **Auto-Save System** (2026-01-06): Incremental JSON saving after each pipeline step to `/tmp/alima_webapp_autosave/`
- **Recovery Mechanism**: Automatic detection of WebSocket timeout, recovery button shows to restore results from auto-save
- **Heartbeat Protocol**: WebSocket sends heartbeat every 5 seconds to maintain connection during long DK searches
- **Auto-Cleanup**: Old auto-save files (>24h) automatically removed on webapp startup
- **Queue UI** (NEW 2026-01-12): Shows queue position and ETA with animated status indicator, auto-connects WebSocket when execution starts

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

- Authentication/user accounts (per-user queue limits, session isolation)
- Priority queue support (premium tier users get priority execution)
- Per-user rate limiting (e.g., 10 requests/hour for free tier)
- Result history/session recall (persistent session storage)
- Batch processing UI (process multiple documents in queue)
- Docker containerization
- Dark/light theme toggle
- Monitoring & metrics (queue depth, execution time histograms, error rates)
