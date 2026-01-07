# WebApp Session History - Implementierungsplan

**Feature**: Session History & Recovery Management
**Autor**: Claude (Sonnet 4.5)
**Datum**: 2026-01-06
**Status**: Planung

---

## Übersicht

Die Session History Funktion ermöglicht Benutzern, alle verfügbaren Auto-Save Sessions zu durchsuchen, wiederherzustellen oder zu löschen. Dies erweitert die bestehende Recovery-Funktion um eine vollständige Verwaltungsoberfläche.

### Motivation

**Aktuelles Verhalten:**
- Auto-Save läuft im Hintergrund
- Recovery-Button erscheint nur bei WebSocket-Timeout
- Benutzer wissen nicht, welche Sessions verfügbar sind
- Kein Zugriff auf ältere Sessions (>24h werden gelöscht)

**Gewünschtes Verhalten:**
- Benutzer sehen alle verfügbaren Sessions beim Start
- Alte Analysen können jederzeit wiederhergestellt werden
- Session-Management: Löschen, Umbenennen, Exportieren
- Vorschau der Inhalte (Abstract-Snippet, Keywords-Count)

---

## Architektur-Übersicht

### Komponenten

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (HTML/JS)                    │
├─────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌──────────────────────────────┐  │
│  │ History Modal  │  │  Main Interface              │  │
│  │ - Session List │  │  - Current Analysis          │  │
│  │ - Search/Filter│  │  - Recovery Button           │  │
│  │ - Actions      │  │  - New Analysis              │  │
│  └────────────────┘  └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                           ▲│
                           ││ REST API
                           │▼
┌─────────────────────────────────────────────────────────┐
│                  Backend (FastAPI)                       │
├─────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────┐ │
│  │ Session History Endpoints                          │ │
│  │ GET  /api/sessions/history - List all sessions    │ │
│  │ GET  /api/sessions/{id}/restore - Restore session │ │
│  │ POST /api/sessions/{id}/rename - Rename session   │ │
│  │ DEL  /api/sessions/{id}/delete - Delete session   │ │
│  └────────────────────────────────────────────────────┘ │
│                           │                              │
│  ┌────────────────────────▼──────────────────────────┐  │
│  │ Session History Manager                           │  │
│  │ - Scan auto-save directory                        │  │
│  │ - Load metadata files                             │  │
│  │ - Parse & enrich session data                     │  │
│  │ - Handle CRUD operations                          │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│           Auto-Save Directory (/tmp/...)                 │
├─────────────────────────────────────────────────────────┤
│  session_abc123.json          ← Analysis state          │
│  session_abc123.meta.json     ← Metadata                │
│  session_xyz789.json                                     │
│  session_xyz789.meta.json                                │
└─────────────────────────────────────────────────────────┘
```

---

## Datenmodell

### Session Metadata (erweitert)

**Aktuelle Struktur** (`session_{id}.meta.json`):
```json
{
  "session_id": "abc-123-def",
  "created_at": "2026-01-06T12:00:00",
  "last_step": "classification",
  "status": "completed",
  "autosave_timestamp": "2026-01-06T12:05:00"
}
```

**Erweiterte Struktur** (neue Felder):
```json
{
  "session_id": "abc-123-def",
  "created_at": "2026-01-06T12:00:00",
  "last_step": "classification",
  "status": "completed",
  "autosave_timestamp": "2026-01-06T12:05:00",

  // NEU: Benutzer-definiert
  "user_label": "Quantenchemie Dissertation Kapitel 3",
  "tags": ["dissertation", "quantum", "2026"],

  // NEU: Content Preview
  "abstract_preview": "Quantenmechanische Studien zur Photo...",
  "input_type": "pdf",
  "input_source": "chapter3.pdf",

  // NEU: Result Summary
  "stats": {
    "initial_keywords_count": 15,
    "final_keywords_count": 20,
    "dk_classifications_count": 10,
    "pipeline_steps_completed": 5,
    "total_duration_seconds": 245
  },

  // NEU: File Info
  "file_size_bytes": 125840,
  "analysis_state_file": "session_abc-123-def.json"
}
```

---

## API-Spezifikation

### 1. List Sessions (GET /api/sessions/history)

**Request:**
```http
GET /api/sessions/history?sort=date&order=desc&limit=50&offset=0
```

**Query Parameters:**
| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `sort` | string | `date` | Sort by: `date`, `size`, `keywords`, `status` |
| `order` | string | `desc` | Order: `asc`, `desc` |
| `limit` | int | 50 | Max results per page |
| `offset` | int | 0 | Pagination offset |
| `status` | string | - | Filter by status: `completed`, `error`, `partial` |
| `search` | string | - | Search in labels, abstracts, tags |

**Response:**
```json
{
  "total": 127,
  "limit": 50,
  "offset": 0,
  "sessions": [
    {
      "session_id": "abc-123",
      "created_at": "2026-01-06T12:00:00",
      "user_label": "Quantenchemie Diss.",
      "abstract_preview": "Quantenmechanische Studien...",
      "status": "completed",
      "stats": {
        "final_keywords_count": 20,
        "dk_classifications_count": 10
      },
      "file_size_kb": 123,
      "age_hours": 2
    },
    // ... more sessions
  ]
}
```

---

### 2. Restore Session (GET /api/sessions/{id}/restore)

**Request:**
```http
GET /api/sessions/abc-123/restore
```

**Response:**
```json
{
  "session_id": "abc-123",
  "status": "restored",
  "results": {
    "original_abstract": "...",
    "final_keywords": [...],
    "dk_classifications": [...]
  },
  "metadata": {
    "created_at": "2026-01-06T12:00:00",
    "user_label": "Quantenchemie"
  }
}
```

**Behavior:**
- Lädt analysis_state aus JSON
- Erstellt neue Session mit wiederhergestellten Daten
- Zeigt Ergebnisse in Main Interface
- Erlaubt Export und weitere Verarbeitung

---

### 3. Rename Session (POST /api/sessions/{id}/rename)

**Request:**
```http
POST /api/sessions/abc-123/rename
Content-Type: application/json

{
  "user_label": "Quantum Chemistry Dissertation - Chapter 3",
  "tags": ["dissertation", "quantum", "2026"]
}
```

**Response:**
```json
{
  "session_id": "abc-123",
  "user_label": "Quantum Chemistry Dissertation - Chapter 3",
  "tags": ["dissertation", "quantum", "2026"],
  "updated_at": "2026-01-06T14:30:00"
}
```

---

### 4. Delete Session (DELETE /api/sessions/{id}/delete)

**Request:**
```http
DELETE /api/sessions/abc-123/delete
```

**Response:**
```json
{
  "session_id": "abc-123",
  "deleted": true,
  "files_removed": [
    "session_abc-123.json",
    "session_abc-123.meta.json"
  ]
}
```

---

### 5. Bulk Operations (POST /api/sessions/bulk)

**Request:**
```http
POST /api/sessions/bulk
Content-Type: application/json

{
  "action": "delete",
  "session_ids": ["abc-123", "def-456", "ghi-789"]
}
```

**Actions:**
- `delete` - Löscht mehrere Sessions
- `tag` - Fügt Tags zu mehreren Sessions hinzu
- `export` - Exportiert mehrere Sessions als ZIP

**Response:**
```json
{
  "action": "delete",
  "succeeded": ["abc-123", "def-456"],
  "failed": [
    {
      "session_id": "ghi-789",
      "error": "Session not found"
    }
  ],
  "total": 3,
  "success_count": 2
}
```

---

## Frontend-Komponenten

### History Modal (index.html)

**HTML-Struktur:**
```html
<!-- History Modal - Claude Generated -->
<div id="history-modal" class="modal" style="display: none;">
    <div class="modal-content">
        <div class="modal-header">
            <h2>📚 Analyse-Verlauf</h2>
            <button class="close-btn" onclick="closeHistoryModal()">×</button>
        </div>

        <div class="modal-toolbar">
            <input type="text" id="history-search" placeholder="🔍 Suchen..." />
            <select id="history-sort">
                <option value="date">Nach Datum</option>
                <option value="size">Nach Größe</option>
                <option value="keywords">Nach Keywords</option>
            </select>
            <button onclick="refreshHistory()">🔄 Aktualisieren</button>
        </div>

        <div class="modal-body">
            <table id="history-table" class="history-table">
                <thead>
                    <tr>
                        <th><input type="checkbox" id="select-all" /></th>
                        <th>Datum</th>
                        <th>Label</th>
                        <th>Vorschau</th>
                        <th>Keywords</th>
                        <th>DK</th>
                        <th>Status</th>
                        <th>Größe</th>
                        <th>Aktionen</th>
                    </tr>
                </thead>
                <tbody id="history-tbody">
                    <!-- Dynamically populated -->
                </tbody>
            </table>

            <div id="history-pagination">
                <!-- Pagination controls -->
            </div>
        </div>

        <div class="modal-footer">
            <button class="btn btn-danger" onclick="bulkDelete()">
                🗑️ Ausgewählte löschen
            </button>
            <button class="btn btn-secondary" onclick="bulkExport()">
                📦 Ausgewählte exportieren
            </button>
        </div>
    </div>
</div>

<!-- Trigger Button (neben "Neue Analyse") -->
<button id="show-history-btn" class="btn btn-secondary" onclick="showHistoryModal()">
    📚 Verlauf anzeigen
</button>
```

**CSS-Styling:**
```css
/* History Modal - Claude Generated */
.modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
}

.modal-content {
    background: white;
    width: 90%;
    max-width: 1200px;
    max-height: 90vh;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px;
    border-bottom: 1px solid #ddd;
}

.modal-toolbar {
    display: flex;
    gap: 10px;
    padding: 15px 20px;
    border-bottom: 1px solid #eee;
    background: #f9f9f9;
}

.modal-body {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
}

.history-table {
    width: 100%;
    border-collapse: collapse;
}

.history-table th,
.history-table td {
    padding: 10px;
    text-align: left;
    border-bottom: 1px solid #eee;
}

.history-table th {
    background: #f5f5f5;
    font-weight: 600;
}

.history-table tbody tr:hover {
    background: #f9f9f9;
    cursor: pointer;
}

.session-preview {
    max-width: 300px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: #666;
    font-size: 0.9em;
}

.session-actions {
    display: flex;
    gap: 5px;
}

.session-actions button {
    padding: 5px 10px;
    font-size: 0.85em;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

.btn-restore {
    background: #4caf50;
    color: white;
}

.btn-delete {
    background: #f44336;
    color: white;
}

.btn-edit {
    background: #2196f3;
    color: white;
}
```

---

### JavaScript-Implementierung (app.js)

**Session History Class:**
```javascript
// Session History Manager - Claude Generated
class SessionHistory {
    constructor(webapp) {
        this.webapp = webapp;
        this.sessions = [];
        this.selectedSessions = new Set();
        this.currentPage = 0;
        this.itemsPerPage = 20;
        this.sortBy = 'date';
        this.sortOrder = 'desc';
        this.searchQuery = '';
    }

    // Load session history from backend
    async loadHistory() {
        try {
            const params = new URLSearchParams({
                sort: this.sortBy,
                order: this.sortOrder,
                limit: this.itemsPerPage,
                offset: this.currentPage * this.itemsPerPage,
                search: this.searchQuery
            });

            const response = await fetch(`/api/sessions/history?${params}`);
            if (!response.ok) {
                throw new Error('Failed to load history');
            }

            const data = await response.json();
            this.sessions = data.sessions;
            this.totalSessions = data.total;

            this.renderTable();
            this.renderPagination();
        } catch (error) {
            console.error('History load error:', error);
            this.showError('Fehler beim Laden des Verlaufs');
        }
    }

    // Render session table
    renderTable() {
        const tbody = document.getElementById('history-tbody');
        tbody.innerHTML = '';

        if (this.sessions.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="9" style="text-align: center; padding: 40px; color: #999;">
                        📭 Keine gespeicherten Sessions gefunden
                    </td>
                </tr>
            `;
            return;
        }

        this.sessions.forEach(session => {
            const row = this.createSessionRow(session);
            tbody.appendChild(row);
        });
    }

    // Create table row for session
    createSessionRow(session) {
        const tr = document.createElement('tr');
        tr.dataset.sessionId = session.session_id;

        // Format date
        const date = new Date(session.created_at);
        const dateStr = date.toLocaleString('de-DE', {
            day: '2-digit',
            month: '2-digit',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });

        // Status icon
        const statusIcon = {
            'completed': '✅',
            'error': '❌',
            'partial': '⚠️',
            'running': '🔄'
        }[session.status] || '❓';

        tr.innerHTML = `
            <td>
                <input type="checkbox"
                       onchange="sessionHistory.toggleSelect('${session.session_id}')" />
            </td>
            <td>${dateStr}</td>
            <td>
                <div class="session-label" title="${session.user_label || 'Unbenannt'}">
                    ${session.user_label || '<em>Unbenannt</em>'}
                </div>
                ${session.tags ? `<div class="session-tags">${this.renderTags(session.tags)}</div>` : ''}
            </td>
            <td>
                <div class="session-preview" title="${session.abstract_preview}">
                    ${session.abstract_preview || '-'}
                </div>
            </td>
            <td>${session.stats?.final_keywords_count || 0}</td>
            <td>${session.stats?.dk_classifications_count || 0}</td>
            <td>${statusIcon} ${session.status}</td>
            <td>${this.formatFileSize(session.file_size_kb)}</td>
            <td class="session-actions">
                <button class="btn-restore" onclick="sessionHistory.restore('${session.session_id}')"
                        title="Wiederherstellen">
                    🔄
                </button>
                <button class="btn-edit" onclick="sessionHistory.rename('${session.session_id}')"
                        title="Umbenennen">
                    ✏️
                </button>
                <button class="btn-delete" onclick="sessionHistory.delete('${session.session_id}')"
                        title="Löschen">
                    🗑️
                </button>
            </td>
        `;

        return tr;
    }

    // Restore session
    async restore(sessionId) {
        try {
            const response = await fetch(`/api/sessions/${sessionId}/restore`);
            if (!response.ok) {
                throw new Error('Restore failed');
            }

            const data = await response.json();

            // Close history modal
            this.closeModal();

            // Display results in main interface
            this.webapp.handleAnalysisComplete({
                status: 'completed',
                results: data.results,
                current_step: 'classification'
            });

            // Show success message
            this.webapp.appendStreamText(`\n✅ Session wiederhergestellt: ${data.metadata.user_label || sessionId}\n`);

        } catch (error) {
            console.error('Restore error:', error);
            this.showError('Wiederherstellung fehlgeschlagen');
        }
    }

    // Rename session
    async rename(sessionId) {
        const session = this.sessions.find(s => s.session_id === sessionId);
        const currentLabel = session?.user_label || '';

        const newLabel = prompt('Neuer Name für die Session:', currentLabel);
        if (newLabel === null || newLabel === currentLabel) return;

        try {
            const response = await fetch(`/api/sessions/${sessionId}/rename`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_label: newLabel })
            });

            if (!response.ok) {
                throw new Error('Rename failed');
            }

            // Reload history
            await this.loadHistory();

        } catch (error) {
            console.error('Rename error:', error);
            this.showError('Umbenennung fehlgeschlagen');
        }
    }

    // Delete session
    async delete(sessionId) {
        if (!confirm('Session wirklich löschen?')) return;

        try {
            const response = await fetch(`/api/sessions/${sessionId}/delete`, {
                method: 'DELETE'
            });

            if (!response.ok) {
                throw new Error('Delete failed');
            }

            // Reload history
            await this.loadHistory();

        } catch (error) {
            console.error('Delete error:', error);
            this.showError('Löschen fehlgeschlagen');
        }
    }

    // Bulk delete
    async bulkDelete() {
        if (this.selectedSessions.size === 0) {
            alert('Keine Sessions ausgewählt');
            return;
        }

        if (!confirm(`${this.selectedSessions.size} Sessions löschen?`)) return;

        try {
            const response = await fetch('/api/sessions/bulk', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    action: 'delete',
                    session_ids: Array.from(this.selectedSessions)
                })
            });

            if (!response.ok) {
                throw new Error('Bulk delete failed');
            }

            const data = await response.json();
            alert(`${data.success_count} Sessions gelöscht`);

            // Clear selection and reload
            this.selectedSessions.clear();
            await this.loadHistory();

        } catch (error) {
            console.error('Bulk delete error:', error);
            this.showError('Löschen fehlgeschlagen');
        }
    }

    // Toggle session selection
    toggleSelect(sessionId) {
        if (this.selectedSessions.has(sessionId)) {
            this.selectedSessions.delete(sessionId);
        } else {
            this.selectedSessions.add(sessionId);
        }
    }

    // Format file size
    formatFileSize(kb) {
        if (kb < 1024) return `${kb} KB`;
        return `${(kb / 1024).toFixed(1)} MB`;
    }

    // Render tags
    renderTags(tags) {
        return tags.map(tag =>
            `<span class="tag">${tag}</span>`
        ).join(' ');
    }

    // Show/hide modal
    showModal() {
        document.getElementById('history-modal').style.display = 'flex';
        this.loadHistory();
    }

    closeModal() {
        document.getElementById('history-modal').style.display = 'none';
    }

    // Error handling
    showError(message) {
        alert(`❌ ${message}`);
    }

    // Pagination
    renderPagination() {
        const pagination = document.getElementById('history-pagination');
        const totalPages = Math.ceil(this.totalSessions / this.itemsPerPage);

        pagination.innerHTML = `
            <button ${this.currentPage === 0 ? 'disabled' : ''}
                    onclick="sessionHistory.prevPage()">◀ Zurück</button>
            <span>Seite ${this.currentPage + 1} von ${totalPages}</span>
            <button ${this.currentPage >= totalPages - 1 ? 'disabled' : ''}
                    onclick="sessionHistory.nextPage()">Weiter ▶</button>
        `;
    }

    nextPage() {
        this.currentPage++;
        this.loadHistory();
    }

    prevPage() {
        this.currentPage--;
        this.loadHistory();
    }
}

// Initialize history manager
let sessionHistory;
document.addEventListener('DOMContentLoaded', () => {
    window.alima = new AlimaWebapp();
    sessionHistory = new SessionHistory(window.alima);
});
```

---

## Backend-Implementierung (app.py)

**Session History Manager:**
```python
# Session History Manager - Claude Generated
class SessionHistoryManager:
    """Manages session history and auto-save directory operations"""

    def __init__(self, autosave_dir: Path):
        self.autosave_dir = autosave_dir

    def list_sessions(
        self,
        sort_by: str = "date",
        order: str = "desc",
        limit: int = 50,
        offset: int = 0,
        status_filter: Optional[str] = None,
        search_query: Optional[str] = None
    ) -> dict:
        """List all available sessions with metadata"""

        # Scan directory for session files
        sessions = []
        for meta_file in self.autosave_dir.glob("session_*.meta.json"):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                # Load analysis state for stats
                state_file = meta_file.with_suffix('.json').with_suffix('.json')
                if state_file.exists():
                    stats = self._extract_stats(state_file)
                    metadata['stats'] = stats
                    metadata['file_size_kb'] = state_file.stat().st_size // 1024

                # Calculate age
                created = datetime.fromisoformat(metadata['created_at'])
                age_hours = (datetime.now() - created).total_seconds() / 3600
                metadata['age_hours'] = int(age_hours)

                sessions.append(metadata)

            except Exception as e:
                logger.warning(f"Failed to load session metadata {meta_file}: {e}")
                continue

        # Filter by status
        if status_filter:
            sessions = [s for s in sessions if s.get('status') == status_filter]

        # Search filter
        if search_query:
            query_lower = search_query.lower()
            sessions = [
                s for s in sessions
                if query_lower in s.get('user_label', '').lower()
                or query_lower in s.get('abstract_preview', '').lower()
                or any(query_lower in tag.lower() for tag in s.get('tags', []))
            ]

        # Sort
        reverse = (order == "desc")
        if sort_by == "date":
            sessions.sort(key=lambda s: s.get('created_at', ''), reverse=reverse)
        elif sort_by == "size":
            sessions.sort(key=lambda s: s.get('file_size_kb', 0), reverse=reverse)
        elif sort_by == "keywords":
            sessions.sort(
                key=lambda s: s.get('stats', {}).get('final_keywords_count', 0),
                reverse=reverse
            )

        # Paginate
        total = len(sessions)
        sessions = sessions[offset:offset + limit]

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "sessions": sessions
        }

    def _extract_stats(self, state_file: Path) -> dict:
        """Extract statistics from analysis state file"""
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)

            return {
                "initial_keywords_count": len(state.get('initial_keywords', [])),
                "final_keywords_count": len(state.get('final_keywords', [])),
                "dk_classifications_count": len(state.get('dk_classifications', [])),
                "pipeline_steps_completed": 5 if state.get('pipeline_step_completed') == 'classification' else 0
            }
        except:
            return {}

    def rename_session(self, session_id: str, user_label: str, tags: List[str] = None) -> dict:
        """Rename a session and update metadata"""
        meta_file = self.autosave_dir / f"session_{session_id}.meta.json"

        if not meta_file.exists():
            raise FileNotFoundError("Session not found")

        with open(meta_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        metadata['user_label'] = user_label
        if tags is not None:
            metadata['tags'] = tags
        metadata['updated_at'] = datetime.now().isoformat()

        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return metadata

    def delete_session(self, session_id: str) -> dict:
        """Delete a session and its files"""
        state_file = self.autosave_dir / f"session_{session_id}.json"
        meta_file = self.autosave_dir / f"session_{session_id}.meta.json"

        files_removed = []

        if state_file.exists():
            state_file.unlink()
            files_removed.append(state_file.name)

        if meta_file.exists():
            meta_file.unlink()
            files_removed.append(meta_file.name)

        if not files_removed:
            raise FileNotFoundError("Session not found")

        return {
            "session_id": session_id,
            "deleted": True,
            "files_removed": files_removed
        }


# Initialize history manager
history_manager = SessionHistoryManager(AUTOSAVE_DIR)


# API Endpoints
@app.get("/api/sessions/history")
async def get_session_history(
    sort: str = "date",
    order: str = "desc",
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
    search: Optional[str] = None
) -> dict:
    """Get session history list - Claude Generated"""

    try:
        return history_manager.list_sessions(
            sort_by=sort,
            order=order,
            limit=limit,
            offset=offset,
            status_filter=status,
            search_query=search
        )
    except Exception as e:
        logger.error(f"History list error: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to load history: {str(e)}")


@app.get("/api/sessions/{session_id}/restore")
async def restore_session_from_history(session_id: str) -> dict:
    """Restore a session from history - Claude Generated"""

    # This is identical to the recovery endpoint, just with different semantics
    return await recover_session(session_id)


@app.post("/api/sessions/{session_id}/rename")
async def rename_session_endpoint(
    session_id: str,
    data: dict
) -> dict:
    """Rename a session - Claude Generated"""

    try:
        user_label = data.get('user_label', '')
        tags = data.get('tags', [])

        metadata = history_manager.rename_session(session_id, user_label, tags)

        return metadata

    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    except Exception as e:
        logger.error(f"Rename error: {e}", exc_info=True)
        raise HTTPException(500, f"Rename failed: {str(e)}")


@app.delete("/api/sessions/{session_id}/delete")
async def delete_session_endpoint(session_id: str) -> dict:
    """Delete a session - Claude Generated"""

    try:
        return history_manager.delete_session(session_id)

    except FileNotFoundError:
        raise HTTPException(404, "Session not found")
    except Exception as e:
        logger.error(f"Delete error: {e}", exc_info=True)
        raise HTTPException(500, f"Delete failed: {str(e)}")


@app.post("/api/sessions/bulk")
async def bulk_operations(data: dict) -> dict:
    """Bulk operations on sessions - Claude Generated"""

    action = data.get('action')
    session_ids = data.get('session_ids', [])

    if not action or not session_ids:
        raise HTTPException(400, "Missing action or session_ids")

    succeeded = []
    failed = []

    for session_id in session_ids:
        try:
            if action == 'delete':
                history_manager.delete_session(session_id)
            # Add more actions here (tag, export, etc.)

            succeeded.append(session_id)

        except Exception as e:
            failed.append({
                "session_id": session_id,
                "error": str(e)
            })

    return {
        "action": action,
        "succeeded": succeeded,
        "failed": failed,
        "total": len(session_ids),
        "success_count": len(succeeded)
    }
```

---

## Implementierungs-Reihenfolge

### Phase 1: Backend Foundation (2-3h)
- [ ] `SessionHistoryManager` Klasse implementieren
- [ ] API-Endpoints hinzufügen
- [ ] Metadata-Schema erweitern
- [ ] Testing mit Mock-Daten

### Phase 2: Frontend UI (2-3h)
- [ ] History Modal HTML/CSS
- [ ] Session Table Rendering
- [ ] Search & Filter Logik
- [ ] Pagination

### Phase 3: Actions & Integration (1-2h)
- [ ] Restore-Funktion
- [ ] Rename-Funktion
- [ ] Delete-Funktion
- [ ] Bulk-Operations

### Phase 4: Polish & Testing (1h)
- [ ] Error-Handling
- [ ] Loading States
- [ ] Deutsche Übersetzungen
- [ ] End-to-End Testing

**Total: 6-9 Stunden**

---

## Testing-Strategie

### Unit Tests
- `test_list_sessions()` - Session-Listing
- `test_rename_session()` - Umbenennung
- `test_delete_session()` - Löschung
- `test_bulk_operations()` - Bulk-Actions

### Integration Tests
- `test_restore_flow()` - Kompletter Restore-Workflow
- `test_search_filter()` - Suche und Filter
- `test_pagination()` - Seitennavigation

### Manual Tests
- Create 50+ sessions und teste Performance
- Teste Sort/Filter/Search Kombinationen
- Teste Bulk-Delete mit 20+ Sessions

---

## Performance-Überlegungen

### Optimierungen
1. **Lazy Loading**: Nur Metadaten laden, State-Files on-demand
2. **Caching**: Session-Liste im Memory cachen (5min TTL)
3. **Pagination**: Max 50 Sessions pro Request
4. **Index Files**: Optional: SQLite-Index für große Histories (1000+ Sessions)

### Skalierung
- **< 100 Sessions**: Filesystem-Scan ist schnell genug
- **100-1000 Sessions**: Caching empfohlen
- **> 1000 Sessions**: SQLite-Index notwendig

---

## Zukunfts-Features

### V1.1 - Advanced Features
- **Tags & Categories**: Farbige Tags, Kategorien
- **Export to ZIP**: Multiple Sessions als ZIP exportieren
- **Search Improvements**: Fuzzy-Search, Advanced Filters
- **Thumbnails**: Preview-Images für Image-Inputs

### V1.2 - Collaboration
- **Shared Sessions**: Sessions mit anderen Usern teilen
- **Comments**: Notizen zu Sessions hinzufügen
- **Version History**: Änderungen tracken

### V2.0 - Analytics
- **Dashboard**: Statistiken über alle Sessions
- **Charts**: Keyword-Trends, Classification-Verteilung
- **Export Reports**: PDF-Reports generieren

---

## Security & Privacy

### Überlegungen
1. **Access Control**: Aktuell keine User-Auth → alle Sessions öffentlich
2. **File Permissions**: Auto-Save Directory sollte User-restricted sein
3. **Sensitive Data**: Keine PII in Abstracts speichern
4. **GDPR**: Daten-Löschung auf Anfrage ermöglichen

### Empfehlungen
- Optional: Sessions mit Passwort schützen
- Optional: Encryption für Auto-Save Files
- Optional: User-Auth mit Session-Ownership

---

## Dokumentation Updates

Nach Implementierung aktualisieren:
- `src/webapp/CLAUDE.md` - Neue Endpoints
- `AIChangelog.md` - Session History Feature
- `README.md` - User Guide für History

---

**Ende des Planungsdokuments**
