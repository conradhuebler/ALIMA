/**
 * ALIMA Webapp Frontend - Claude Generated
 * Handles UI interactions and WebSocket communication
 */

/**
 * ThemeManager — Dark/light mode with localStorage persistence and system-pref auto-detect.
 * Runs synchronously before the DOM renders to prevent FOUC. — Claude Generated
 */
const ThemeManager = {
    STORAGE_KEY: 'alima_theme',

    /** Returns 'dark' or 'light' based on localStorage or system preference */
    getEffective() {
        const stored = localStorage.getItem(this.STORAGE_KEY);
        if (stored === 'dark' || stored === 'light') return stored;
        return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    },

    /** Applies theme to <html> and updates toggle button emoji */
    apply(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        const btn = document.getElementById('theme-toggle');
        if (btn) btn.textContent = theme === 'dark' ? '☀️' : '🌙';
    },

    /** Cycles between dark and light, persists choice */
    toggle() {
        const current = document.documentElement.getAttribute('data-theme') || this.getEffective();
        const next = current === 'dark' ? 'light' : 'dark';
        localStorage.setItem(this.STORAGE_KEY, next);
        this.apply(next);
    },

    /** Initialize: apply saved/system theme and listen for system changes */
    init() {
        this.apply(this.getEffective());
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            // Only follow system if user has not made a manual choice
            if (!localStorage.getItem(this.STORAGE_KEY)) {
                this.apply(e.matches ? 'dark' : 'light');
            }
        });
    },
};

// Run synchronously — prevents flash of wrong theme
ThemeManager.init();

class AlimaWebapp {
    constructor() {
        this.sessionId = null;
        this.isAnalyzing = false;
        this.currentStep = 0;
        this.ws = null;
        this.pollInterval = null;
        this.cameraStream = null;
        this.capturedCameraImage = null;
        this.cameraBlob = null;
        this.pendingSourceType = 'text';   // Source type for working title / filename - Claude Generated
        this.pendingInputSource = '';       // DOI, URL, or filename for working title - Claude Generated

        // WP12: the shared alima_render.js auto-scrolls its host window; in the
        // webapp the render region is embedded, so disable it to avoid hijacking
        // the page scroll. _lastRenderSeq dedups render events by their monotonic
        // server seq, so a WS reconnect replay / polling re-delivery never
        // double-appends an append-only block.
        if (typeof window !== 'undefined') window.__autoscroll = false;
        this._lastRenderSeq = -1;

        // Pipeline-stepper state: per-workflow step lists + the active list. - Claude Generated
        this.workflowSteps = {};
        this._stepperSteps = [];

        this.setupEventListeners();
        this.initializeSession();
    }

    // Initialize session when app loads - Claude Generated
    async initializeSession() {
        await this.createNewSession();
        await this.loadWorkflows();
        await this.loadModelOverrides();
        // Check if current URL session is already active (page refresh scenario)
        const reconnectedCurrent = await this.checkCurrentSessionState();
        // If not, check localStorage for a different running session
        if (!reconnectedCurrent) await this.checkForRunningSession();
        console.log('Ready for analysis');
    }

    // Load available workflows into the header dropdown - Claude Generated
    async loadWorkflows() {
        try {
            const response = await fetch('/api/workflows');
            if (!response.ok) return;
            const workflows = await response.json();
            const select = document.getElementById('workflow-select');
            if (!select) return;
            // Keep placeholder option
            select.innerHTML = '<option value="">— Workflow laden … —</option>';
            this.workflowSteps = {};
            workflows.forEach(wf => {
                const option = document.createElement('option');
                option.value = wf.value;
                option.textContent = wf.label;
                if (wf.value === '__separator__') {
                    option.disabled = true;
                }
                // Cache the per-workflow step list for the pipeline-stepper. - Claude Generated
                this.workflowSteps[wf.value] = Array.isArray(wf.steps) ? wf.steps : [];
                // Pre-select the configured default workflow unless the user already chose one.
                if (wf.default && !select.dataset.userSelected) {
                    option.selected = true;
                }
                select.appendChild(option);
            });
            console.log(`Loaded ${workflows.length} workflows`);
            // Render the stepper for whatever workflow is now selected. - Claude Generated
            this.renderStepperForSelected();
        } catch (e) {
            console.error('Failed to load workflows:', e);
        }
    }

    // Check if the current session (from URL) is already running or completed - Claude Generated
    async checkCurrentSessionState() {
        if (!this.sessionId) return false;
        try {
            const resp = await fetch(`/api/session/${this.sessionId}`);
            if (!resp.ok) return false;
            const data = await resp.json();
            if (data.status === 'running') {
                this.isAnalyzing = true;
                this.updateButtonState();
                this.setResultsPanelState('running');
                this.enableExportButton(true);
                localStorage.setItem('alima_running_session', this.sessionId);
                this.appendLocalNotice(`🔌 Wiederverbunden mit laufender Analyse …`);
                this.connectWebSocket();
                return true;
            } else if (data.status === 'completed') {
                let results = {};
                try {
                    const exportResp = await fetch(`/api/export/${this.sessionId}`);
                    if (exportResp.ok) results = await exportResp.json();
                } catch (e) { console.warn('Could not fetch export results:', e); }
                this.handleAnalysisComplete({
                    status: 'completed',
                    results,
                    current_step: data.current_step || 'classification'
                });
                return true;
            }
        } catch (e) { /* fresh session, ignore */ }
        return false;
    }

    // Check localStorage for a previously running session and offer reconnect - Claude Generated
    async checkForRunningSession() {
        const savedId = localStorage.getItem('alima_running_session');
        if (!savedId || savedId === this.sessionId) return;

        try {
            const resp = await fetch(`/api/session/${savedId}`);
            if (!resp.ok) {
                localStorage.removeItem('alima_running_session');
                return;
            }
            const data = await resp.json();

            if (data.status === 'running') {
                this.showReconnectBanner(savedId, 'running', data);
            } else if (data.status === 'completed') {
                this.showReconnectBanner(savedId, 'completed', data);
            } else {
                localStorage.removeItem('alima_running_session');
            }
        } catch (e) {
            console.warn('Could not check saved session:', e);
            localStorage.removeItem('alima_running_session');
        }
    }

    // Show banner offering reconnect to saved session - Claude Generated
    showReconnectBanner(savedId, status, data) {
        const banner = document.getElementById('reconnect-banner');
        const msg = document.getElementById('reconnect-message');
        const reconnectBtn = document.getElementById('reconnect-btn');
        const dismissBtn = document.getElementById('reconnect-dismiss');
        if (!banner || !msg) return;

        const shortId = savedId.substring(0, 8);
        if (status === 'running') {
            msg.textContent = `⚡ Pipeline läuft noch (Session ${shortId}…, Schritt: ${data.current_step || '?'}) — Wiederverbinden?`;
            reconnectBtn.textContent = '🔌 Wiederverbinden';
        } else {
            msg.textContent = `✅ Abgeschlossene Analyse gefunden (Session ${shortId}…) — Ergebnisse anzeigen?`;
            reconnectBtn.textContent = '📂 Ergebnisse anzeigen';
        }

        banner.style.display = 'flex';

        reconnectBtn.onclick = () => {
            banner.style.display = 'none';
            this.reconnectToSession(savedId, status, data);
        };
        dismissBtn.onclick = () => {
            banner.style.display = 'none';
            localStorage.removeItem('alima_running_session');
        };
    }

    // Reconnect to a saved session (running or completed) - Claude Generated
    reconnectToSession(savedId, status, data) {
        // Switch to saved session
        this.sessionId = savedId;
        const url = new URL(window.location);
        url.searchParams.set('session', savedId);
        window.history.replaceState(null, '', url);

        if (status === 'running') {
            this.isAnalyzing = true;
            this.updateButtonState();
            this.setResultsPanelState('running');
            this.enableExportButton(true);
            this.appendLocalNotice(`🔌 Wiederverbunden mit laufender Analyse (${savedId.substring(0, 8)}…)`);
            this.connectWebSocket();
        } else {
            // Completed: fetch results from export endpoint (session poll no longer includes them).
            fetch(`/api/export/${savedId}`)
                .then(r => r.ok ? r.json() : {})
                .catch(() => ({}))
                .then(results => {
                    this.handleAnalysisComplete({
                        status: 'completed',
                        results,
                        current_step: data.current_step || 'classification'
                    });
                    this.appendLocalNotice(`📂 Ergebnisse der abgeschlossenen Analyse wiederhergestellt.`);
                    localStorage.removeItem('alima_running_session');
                });
        }
    }

    // Load available provider/model overrides into separate dropdowns - Claude Generated
    async loadModelOverrides({ force = false } = {}) {
        try {
            const response = force
                ? await fetch('/api/models/refresh', { method: 'POST' })
                : await fetch('/api/models');
            if (!response.ok) return;
            const models = await response.json();

            const providerSelect = document.getElementById('provider-override');
            const modelSelect = document.getElementById('model-override');
            if (!providerSelect || !modelSelect) return;

            // Preserve the current pick across a refresh (mirrors Qt6 ProviderModelSelector)
            const prevProvider = providerSelect.value;
            const prevModel = modelSelect.value;

            const providers = [...new Set(models.map(m => m.provider))].sort();
            providerSelect.innerHTML = '<option value="">— Provider —</option>';
            providers.forEach(p => {
                const option = document.createElement('option');
                option.value = p;
                option.textContent = p;
                providerSelect.appendChild(option);
            });
            if (prevProvider && providers.includes(prevProvider)) {
                providerSelect.value = prevProvider;
            }

            this._availableModels = models;
            this._refreshModelOverrideOptions();
            if (prevModel && models.some(m => m.provider === providerSelect.value && m.model === prevModel)) {
                modelSelect.value = prevModel;
            }

            if (!this._modelOverrideListenerBound) {
                providerSelect.addEventListener('change', () => this._refreshModelOverrideOptions());
                this._modelOverrideListenerBound = true;
            }

            console.log(`Loaded ${models.length} models across ${providers.length} providers`);
        } catch (e) {
            console.error('Failed to load models:', e);
        }
    }

    async refreshModels() {
        const btn = document.getElementById('refresh-models-btn');
        const original = btn ? btn.textContent : null;
        if (btn) {
            btn.disabled = true;
            btn.textContent = '⏳';
        }
        try {
            await this.loadModelOverrides({ force: true });
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.textContent = original;
            }
        }
    }

    _refreshModelOverrideOptions() {
        const providerSelect = document.getElementById('provider-override');
        const modelSelect = document.getElementById('model-override');
        if (!providerSelect || !modelSelect || !this._availableModels) return;
        const provider = providerSelect.value;
        modelSelect.innerHTML = '<option value="">— Modell —</option>';
        const filtered = provider
            ? this._availableModels.filter(m => m.provider === provider)
            : this._availableModels;
        filtered.forEach(m => {
            const option = document.createElement('option');
            option.value = m.model;
            option.textContent = m.model;
            modelSelect.appendChild(option);
        });
    }

    // Setup event listeners
    setupEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // Analyze button (full pipeline)
        document.getElementById('analyze-btn').addEventListener('click', () => {
            this.startAnalysis();
        });

        // Refresh models button (re-detect providers/models without restart) - Claude Generated
        document.getElementById('refresh-models-btn')?.addEventListener('click', () => {
            this.refreshModels();
        });

        // Clear text button - Claude Generated
        document.getElementById('clear-text-btn').addEventListener('click', () => {
            document.getElementById('text-input').value = '';
            this.pendingSourceType = 'text';  // Reset source tracking - Claude Generated
            this.pendingInputSource = '';
        });

        // DOI/URL Resolve button - Claude Generated
        document.getElementById('doi-resolve-btn').addEventListener('click', () => {
            this.processDoiUrl();
        });

        // DOI/URL Enter key - Claude Generated
        document.getElementById('doi-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.processDoiUrl();
            }
        });

        // DOI/URL Open in browser button - Claude Generated
        document.getElementById('doi-open-btn').addEventListener('click', () => {
            this.openDoiUrl();
        });

        // Export button
        document.getElementById('export-btn').addEventListener('click', () => {
            this.exportResults();
        });

        // Clear button (clear results panel)
        document.getElementById('clear-btn').addEventListener('click', () => {
            this.clearSession();
        });

        // Title override field - Claude Generated
        document.getElementById('title-override').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.applyTitleOverride();
            }
        });
        document.getElementById('title-override').addEventListener('blur', () => {
            this.applyTitleOverride();
        });

        // Cancel button (cancel running pipeline)
        document.getElementById('cancel-btn').addEventListener('click', () => {
            this.cancelAnalysis();
        });

        // Abort-step button (stop LLM call, pipeline continues) - Claude Generated
        document.getElementById('abort-step-btn').addEventListener('click', () => {
            this.abortCurrentStep();
        });

        // Clear logs button
        document.getElementById('clear-logs-btn').addEventListener('click', () => {
            this.clearStreamText();
        });

        // File input
        document.getElementById('file-input').addEventListener('change', (e) => {
            const fileName = e.target.files[0]?.name || '';
            document.getElementById('file-name').textContent = fileName ? `✓ ${fileName}` : '';
            // Auto-process file on selection - Claude Generated
            if (e.target.files[0]) {
                this.processFileInput(e.target.files[0]);
            }
        });

        // Theme toggle button — Claude Generated
        const themeBtn = document.getElementById('theme-toggle');
        if (themeBtn) themeBtn.addEventListener('click', () => ThemeManager.toggle());

        // Workflow dropdown: remember user choice and auto-select agentic default on first load
        const workflowSelect = document.getElementById('workflow-select');
        if (workflowSelect) {
            workflowSelect.addEventListener('change', () => {
                workflowSelect.dataset.userSelected = 'true';
                this.renderStepperForSelected();  // Rebuild stepper for the chosen workflow - Claude Generated
            });
        }

        // Input-zone collapse toggle (manual) - Claude Generated
        const inputZoneToggle = document.getElementById('input-zone-toggle');
        if (inputZoneToggle) {
            inputZoneToggle.addEventListener('click', () => this.toggleInputZone());
        }

        // Chat input: Enter sends, Shift+Enter newline
        const chatInput = document.getElementById('chat-input');
        const chatSendBtn = document.getElementById('chat-send-btn');
        if (chatInput) {
            chatInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendChatMessage();
                }
            });
        }
        if (chatSendBtn) {
            chatSendBtn.addEventListener('click', () => this.sendChatMessage());
        }

        // Chat reply-language toggle (DE/EN) - Claude Generated
        if (this.chatLanguage === undefined) this.chatLanguage = 'de';
        const chatLangBtn = document.getElementById('chat-lang-btn');
        if (chatLangBtn) {
            chatLangBtn.addEventListener('click', () => {
                this.chatLanguage = this.chatLanguage === 'de' ? 'en' : 'de';
                chatLangBtn.textContent = this.chatLanguage.toUpperCase();
                const ind = document.getElementById('chat-lang-indicator');
                if (ind) ind.textContent = this.chatLanguage === 'de' ? 'Deutsch' : 'English';
            });
        }

        // Drag and drop
        this.setupDragAndDrop();

        // Camera controls
        this.setupCamera();
    }

    setupDragAndDrop() {
        const uploadArea = document.getElementById('file-upload-area');
        const fileInput = document.getElementById('file-input');

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.remove('dragover');
            });
        });

        uploadArea.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                const event = new Event('change', { bubbles: true });
                fileInput.dispatchEvent(event);
            }
        });

        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });
    }

    setupCamera() {
        const startBtn = document.getElementById('camera-start-btn');
        const captureBtn = document.getElementById('camera-capture-btn');
        const stopBtn = document.getElementById('camera-stop-btn');
        const confirmBtn = document.getElementById('camera-confirm-btn');
        const retakeBtn = document.getElementById('camera-retake-btn');
        const previewActions = document.getElementById('camera-preview-actions');
        const video = document.getElementById('camera-video');
        const canvas = document.getElementById('camera-canvas');

        // Check if browser supports camera API - Claude Generated (Defensive)
        const hasCameraSupport = navigator && navigator.mediaDevices && navigator.mediaDevices.getUserMedia;
        if (!hasCameraSupport) {
            startBtn.disabled = true;
            startBtn.textContent = '❌ Kamera nicht unterstützt';
            const errorMsg = window.location.protocol === 'http:'
                ? 'Kamera benötigt HTTPS (Sicherheit)'
                : 'Ihr Browser unterstützt keine Kamera-API';
            console.warn('Camera not available:', errorMsg);
            return;
        }

        startBtn.addEventListener('click', async () => {
            try {
                // Try to get camera stream with better error handling - Claude Generated
                const constraints = {
                    video: {
                        facingMode: 'environment',
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    },
                    audio: false
                };

                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                video.srcObject = stream;
                // Ensure video plays immediately (fallback if autoplay attribute isn't honored)
                try {
                    await video.play();
                } catch (playError) {
                    console.warn('Video.play() failed, relying on autoplay attribute:', playError);
                }
                video.style.display = 'block';
                this.cameraStream = stream;
                startBtn.style.display = 'none';
                captureBtn.style.display = 'block';
                stopBtn.style.display = 'block';
            } catch (error) {
                // Provide helpful error messages - Claude Generated
                let errorMsg = 'Kamera nicht verfügbar: ' + error.message;

                if (error.name === 'NotAllowedError') {
                    errorMsg = 'Kamera-Zugriff wurde verweigert. Bitte Berechtigung erteilen.';
                } else if (error.name === 'NotFoundError') {
                    errorMsg = 'Keine Kamera auf diesem Gerät gefunden.';
                } else if (error.name === 'NotReadableError') {
                    errorMsg = 'Kamera wird bereits von einer anderen Anwendung verwendet.';
                } else if (window.location.protocol === 'http:') {
                    errorMsg = 'Kamera benötigt HTTPS (Sicherheit). Bitte verwende https://.';
                }

                console.error('Camera error:', error);
                alert(errorMsg);
            }
        });

        captureBtn.addEventListener('click', async () => {
            const ctx = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);
            const imageData = canvas.toDataURL('image/jpeg', 0.8);

            document.getElementById('camera-image').src = imageData;
            document.getElementById('camera-preview').style.display = 'flex';
            this.capturedCameraImage = imageData;

            // Convert data URL to Blob for file submission - Claude Generated
            try {
                const response = await fetch(imageData);
                const blob = await response.blob();
                this.cameraBlob = blob;

                // Auto-extract text from camera image and fill textfield - Claude Generated
                await this.extractAndFillTextField('img', null, blob);
            } catch (error) {
                console.error('Error processing camera image:', error);
            }

            // Hide live camera controls, show preview actions (Option A - Quick Retake Flow)
            video.style.display = 'none';
            captureBtn.style.display = 'none';
            stopBtn.style.display = 'none';
            previewActions.style.display = 'flex';  // Show confirm/retake buttons
        });

        // STAGE 2: Stop button (only shown during live camera, not preview)
        stopBtn.addEventListener('click', () => {
            // Stop camera and return to STAGE 1
            if (this.cameraStream) {
                this.cameraStream.getTracks().forEach(track => track.stop());
            }
            video.style.display = 'none';
            video.srcObject = null;
            this.capturedCameraImage = null;
            this.cameraBlob = null;

            startBtn.style.display = 'block';
            captureBtn.style.display = 'none';
            stopBtn.style.display = 'none';
            previewActions.style.display = 'none';
        });

        // STAGE 3: Confirm button (accept photo and stop camera)
        confirmBtn.addEventListener('click', () => {
            // Stop camera and reset to initial state
            if (this.cameraStream) {
                this.cameraStream.getTracks().forEach(track => track.stop());
            }
            video.srcObject = null;
            video.style.display = 'none';
            document.getElementById('camera-preview').style.display = 'none';

            // Reset to STAGE 1
            startBtn.style.display = 'block';
            previewActions.style.display = 'none';
            captureBtn.style.display = 'none';
            stopBtn.style.display = 'none';

            // Keep the captured image and blob for analysis
            // (already in this.capturedCameraImage and this.cameraBlob)
        });

        // STAGE 3: Retake button (go back to live camera without restart)
        retakeBtn.addEventListener('click', () => {
            // Hide preview, show live feed again (camera still running!)
            document.getElementById('camera-preview').style.display = 'none';
            video.style.display = 'block';

            // Back to STAGE 2 (live camera)
            previewActions.style.display = 'none';
            captureBtn.style.display = 'block';
            stopBtn.style.display = 'block';

            // Clear previous capture for new one
            this.capturedCameraImage = null;
            this.cameraBlob = null;
        });
    }

    // Switch input tabs
    switchTab(tabId) {
        // Update button states
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabId}"]`).classList.add('active');

        // Update content visibility
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(tabId).classList.add('active');
    }

    // Create new session
    async createNewSession() {
        try {
            // Check if session ID was injected by server (Option C: Multi-tab isolation) - Claude Generated (2026-01-13)
            if (window.sessionId) {
                this.sessionId = window.sessionId;
                console.log('Using server-injected session ID:', this.sessionId);

                // Update URL to include session ID for bookmarking/refreshing - Claude Generated (2026-01-13)
                const currentUrl = new URL(window.location);
                if (!currentUrl.searchParams.has('session')) {
                    currentUrl.searchParams.set('session', this.sessionId);
                    window.history.replaceState(null, '', currentUrl);
                    console.log(`URL updated to: ${currentUrl}`);
                }
                return;
            }

            // Fallback: create a new session via API when the template injected no sessionId
            const response = await fetch('/api/session', { method: 'POST' });
            const data = await response.json();
            this.sessionId = data.session_id;
            console.log('Session created via API:', this.sessionId);
        } catch (error) {
            console.error('Error creating session:', error);
            this.appendLocalNotice(`❌ Session konnte nicht erstellt werden: ${error.message}`, 'error');
        }
    }

    // Start analysis
    async startAnalysis() {
        if (!this.sessionId) {
            alert(alimaT('js.alert.no_session', 'Session nicht initialisiert. Bitte Seite neu laden.'));
            return;
        }

        if (this.isAnalyzing) {
            alert(alimaT('js.alert.already_running', 'Analyse läuft bereits'));
            return;
        }

        // Read ALWAYS from the main text field - Claude Generated
        const textContent = document.getElementById('text-input').value.trim();
        if (!textContent) {
            alert(alimaT('js.alert.need_input', 'Bitte geben Sie Text ein oder laden Sie eine Quelle'));
            return;
        }

        // Request notification permission on first run (user gesture required) - Claude Generated
        await this.requestNotificationPermission();

        // If no source tracked yet, read doi-input directly — handles manual paste without "Laden" - Claude Generated
        let sourceType = this.pendingSourceType;
        let sourceValue = this.pendingInputSource;
        if (sourceType === 'text') {
            const doiVal = document.getElementById('doi-input').value.trim();
            if (doiVal) {
                sourceType = doiVal.startsWith('http://') || doiVal.startsWith('https://') ? 'url' : 'doi';
                sourceValue = doiVal;
            }
        }

        // Always submit text content; pass source metadata separately for filename/working title - Claude Generated
        await this.submitAnalysis('text', textContent, null, sourceType, sourceValue);
    }

    // Send a chat message to the session agent - Claude Generated
    async sendChatMessage() {
        const chatInput = document.getElementById('chat-input');
        if (!chatInput) return;
        const message = chatInput.value.trim();
        if (!message) return;
        if (!this.sessionId) {
            alert(alimaT('js.alert.no_session', 'Session nicht initialisiert. Bitte Seite neu laden.'));
            return;
        }

        // The backend will render the user bubble into the shared render buffer
        // and stream it to us via WebSocket so the log stays the single source of truth.
        chatInput.value = '';
        chatInput.style.height = 'auto';

        // Collapse the input zone on the first chat send so the chat/log area gets the
        // full height — mirrors the auto-collapse on analysis start (updateButtonState).
        // One-shot, so a later manual re-expand by the operator is respected. - Claude Generated
        if (!this._inputZoneAutoCollapsed) {
            this.setInputZoneCollapsed(true);
            this._inputZoneAutoCollapsed = true;
        }

        const providerSelect = document.getElementById('provider-override');
        const modelSelect = document.getElementById('model-override');
        const thinkSelect = document.getElementById('think-override');
        const body = {
            message,
            provider: providerSelect?.value || null,
            model: modelSelect?.value || null,
            // Reply language + thinking override for the chat agent - Claude Generated
            language: this.chatLanguage || 'de',
            think: thinkSelect?.value || null,
        };

        try {
            const response = await fetch(`/api/session/${this.sessionId}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            if (!response.ok) {
                const err = await response.text();
                throw new Error(`HTTP ${response.status}: ${err}`);
            }
            const data = await response.json();
            console.log('Chat started:', data);
            // Make sure the WebSocket is open so backend-rendered events reach the UI.
            if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
                this.connectWebSocket();
            }
        } catch (error) {
            console.error('Chat error:', error);
            this.appendSystemMessage(`❌ Chat-Fehler: ${error.message}`);
        }
    }

    // Append a user chat bubble to the unified log region - Claude Generated
    appendUserBubble(text) {
        if (typeof appendBlock !== 'function') return;
        const html = `
            <div class="user-bubble">
                <span class="user-bubble-inner">${this.escapeHtml(text)}</span>
            </div>`;
        appendBlock(html);
    }

    // Append a system/info line to the unified log region - Claude Generated
    appendSystemMessage(text) {
        if (typeof appendBlock !== 'function') return;
        const html = `<div class="system-message">${this.escapeHtml(text)}</div>`;
        appendBlock(html);
    }

    // Submit analysis request
    async submitAnalysis(inputType, content, file, sourceType = null, sourceValue = null) {
        try {
            this.isAnalyzing = true;
            this.updateButtonState();
            this.clearStreamText();
            this.resetResultsPanelContent();
            this.hideRecoveryOption();
            this.setResultsPanelState('running');
            // The summary card is NOT shown yet: it now sits outside the
            // collapsing input body, and an empty "Analyse läuft" card would
            // only steal height from the chat during the run. It appears in
            // handleAnalysisComplete, when there is something to read. - Claude Generated
            this.enableExportButton(true);

            // Create FormData for multipart request
            const formData = new FormData();
            formData.append('input_type', inputType);
            if (content) {
                formData.append('content', content);
            }
            if (file) {
                formData.append('file', file);
            } else if (this.cameraBlob) {
                formData.append('file', this.cameraBlob, 'camera_photo.jpg');
                this.cameraBlob = null;
            }

            // Pass source origin metadata for working title / JSON filename - Claude Generated
            if (sourceType && sourceType !== 'text') {
                formData.append('source_type', sourceType);
            }
            if (sourceValue) {
                formData.append('source_value', sourceValue);
            }

            // Add workflow selection - Claude Generated
            const workflowSelect = document.getElementById('workflow-select');
            if (workflowSelect && workflowSelect.value) {
                formData.append('workflow', workflowSelect.value);
            }

            // Add global model override if selected - Claude Generated
            const providerSelect = document.getElementById('provider-override');
            const modelSelect = document.getElementById('model-override');
            if (providerSelect && providerSelect.value && modelSelect && modelSelect.value) {
                formData.append('global_override', `${providerSelect.value}|${modelSelect.value}`);
            }

            // Add global thinking override if not "default" - Claude Generated
            const thinkSelect = document.getElementById('think-override');
            if (thinkSelect && thinkSelect.value) {
                formData.append('think_override', thinkSelect.value);
            }

            // Add the token budget if not "Standard" — it outranks the
            // workflow YAML's per-step max_tokens. - Claude Generated
            const budgetSelect = document.getElementById('max-tokens-override');
            if (budgetSelect && budgetSelect.value) {
                formData.append('max_tokens_override', budgetSelect.value);
            }

            const response = await fetch(`/api/analyze/${this.sessionId}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('Analysis started:', data);

            // Persist session ID so page reload can reconnect - Claude Generated
            localStorage.setItem('alima_running_session', this.sessionId);

            // Connect WebSocket for live updates
            this.connectWebSocket();

        } catch (error) {
            console.error('Analysis error:', error);
            this.appendLocalNotice(`❌ Fehler: ${error.message}`, 'error');
            this.isAnalyzing = false;
            this.updateButtonState();
        }
    }

    // Connect via Polling (fallback from WebSocket) - Claude Generated
    connectViaPolling() {
        console.log('Using polling instead of WebSocket');

        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }

        let lastStep = null;
        let pollCount = 0;
        const maxPolls = 2400; // 20 minutes max (2400 * 0.5s)

        this.pollInterval = setInterval(async () => {
            pollCount++;

            try {
                const response = await fetch(`/api/session/${this.sessionId}`);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);

                const data = await response.json();
                console.log('Poll response:', data);

                // Simulate WebSocket message format (Claude Generated - include streaming tokens)
                const msg = {
                    type: 'status',
                    status: data.status,
                    current_step: data.current_step,
                    results: {},  // results excluded from poll response; fetched separately on completion
                    streaming_tokens: data.streaming_tokens || {},  // Include tokens from polling
                    render_events: data.render_events || []  // WP12: shared chrome events
                };

                if (data.status === 'running') {
                    this.updatePipelineStatus(msg);
                    lastStep = data.current_step;
                } else if (data.status === 'completed' || data.status === 'error') {
                    // Tokens render via the seq-deduped render events (Chat-UX
                    // 5/9); streaming_tokens frames are no longer displayed.
                    clearInterval(this.pollInterval);
                    this.pollInterval = null;

                    // Fetch full results separately so the final poll response stays small.
                    let results = {};
                    if (data.status === 'completed') {
                        try {
                            const exportResp = await fetch(`/api/export/${this.sessionId}`);
                            if (exportResp.ok) results = await exportResp.json();
                        } catch (e) {
                            console.warn('Could not fetch export results:', e);
                        }
                    }

                    this.handleAnalysisComplete({
                        type: 'complete',
                        status: data.status,
                        results: results,
                        error: data.error_message,
                        current_step: data.current_step,
                        render_events: data.render_events || []  // WP12
                    });
                }
            } catch (error) {
                console.error('Poll error:', error);
                this.appendLocalNotice(`⚠️ Poll-Fehler: ${error.message}`, 'error');
            }

            // Timeout after max polls
            if (pollCount > maxPolls) {
                clearInterval(this.pollInterval);
                this.pollInterval = null;
                console.warn('Polling timeout after', maxPolls, 'attempts');
                this.isAnalyzing = false;
                this.updateButtonState();
            }
        }, 500); // Poll every 500ms
    }

    // Try WebSocket, fallback to polling - Claude Generated
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`;

        console.log(`Trying WebSocket: ${wsUrl}`);

        this.ws = new WebSocket(wsUrl);
        let wsConnected = false;

        // Set timeout for WebSocket connection attempt
        const wsTimeout = setTimeout(() => {
            if (!wsConnected) {
                console.log('WebSocket timeout, falling back to polling');
                try {
                    this.ws.close();
                } catch (e) {
                    // Ignore
                }
                this.connectViaPolling();
            }
        }, 2000); // 2 second timeout

        this.ws.onopen = () => {
            wsConnected = true;
            clearTimeout(wsTimeout);
            this.hideRecoveryOption();
            console.log('WebSocket connected');
        };

        this.ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);

            // Ignore heartbeat messages in console and display - Claude Generated
            if (msg.type === 'heartbeat') {
                console.debug('Heartbeat:', msg.timestamp);
                return; // Don't display heartbeats
            }

            console.log('WebSocket message:', msg);

            if (msg.type === 'status') {
                this.updatePipelineStatus(msg);
            } else if (msg.type === 'complete') {
                this.handleAnalysisComplete(msg);
            } else if (msg.type === 'error') {
                // Server-side timeout or error - fall back to polling - Claude Generated
                console.warn('Server WS error:', msg.error);
                if (this.isAnalyzing) {
                    this.showRecoveryOption();
                    this.connectViaPolling();
                }
            }
        };

        this.ws.onerror = (error) => {
            clearTimeout(wsTimeout);
            console.error('WebSocket error (wasConnected=' + wsConnected + '):', error);
            if (wsConnected) {
                // Error on an established connection - show recovery and fall back
                if (this.isAnalyzing) {
                    this.showRecoveryOption();
                    this.appendLocalNotice(`⚠️ WebSocket-Fehler, wechsle zu Polling …`);
                    this.connectViaPolling();
                }
            } else {
                // Connection attempt failed - silent fallback, no user-visible noise
                this.connectViaPolling();
            }
        };

        this.ws.onclose = (event) => {
            console.log('WebSocket closed (code=' + event.code + ', wasConnected=' + wsConnected + ')');

            // Only react to abnormal closure if WS was actually established - Claude Generated
            // Code 1006 can also fire when the 2s timeout calls this.ws.close() before connection
            if ((event.code === 1006 || event.code === 1011) && wsConnected && this.isAnalyzing) {
                this.showRecoveryOption();
                this.appendLocalNotice(`⚠️ Verbindung unterbrochen, wechsle zu Polling …`);
                this.connectViaPolling();
            }
        };
    }

    // WP12: dispatch shared chrome render events to the alima_render.js
    // dispatcher (the same funcs the GUI drives). Dedup by monotonic seq so a
    // WS-reconnect replay / poll re-delivery never double-appends a block.
    dispatchRenderEvents(events) {
        if (!Array.isArray(events) || events.length === 0) return;
        if (typeof appendBlock !== 'function') {
            // alima_render.js failed to load (cache? wrong path?) — make it loud
            // once instead of silently dropping the shared chrome.
            if (!this._renderWarned) {
                console.error('WP12: alima_render.js not loaded — render events dropped. '
                    + 'Hard-refresh (Ctrl+Shift+R) to clear a cached page.');
                this._renderWarned = true;
            }
            return;
        }
        console.debug(`WP12: dispatching ${events.length} render event(s)`);
        // Reveal the results panel so the embedded #log region is visible.
        const panel = document.getElementById('results-panel');
        if (panel) panel.style.display = '';
        for (const ev of events) {
            if (typeof ev.seq === 'number') {
                if (ev.seq <= this._lastRenderSeq) continue;  // already applied
                this._lastRenderSeq = ev.seq;
            }
            try {
                this.dispatchRenderEvent(ev);
            } catch (e) {
                console.error('WP12 render event failed:', ev, e);
            }
        }
    }

    dispatchRenderEvent(ev) {
        switch (ev.type) {
            case 'block':
                if (ev.kind === 'proposal') return;  // GUI-only — Tier-3 webapp ignores
                appendBlock(ev.html);
                break;
            case 'collapsible': appendCollapsible(ev.id, ev.summary, ev.body, ev.open, ev.kind); break;
            case 'collapsible_update': updateCollapsible(ev.id, ev.summary, ev.body, ev.kind, ev.open); break;
            case 'collapsible_append': appendToCollapsible(ev.id, ev.text); break;
            case 'assistant_open': openAssistant(ev.header); break;
            case 'assistant_token': appendToken(ev.text); break;
            case 'assistant_finalize': finalizeAssistant(ev.html); break;
            case 'stream_open': openStreamBlock(ev.id, ev.summary); break;
            case 'stream_token': appendStreamBlock(ev.text); break;
            case 'stream_close': closeStreamBlock(ev.id, ev.summary, ev.collapse); break;
            case 'typing':
                if (ev.active) showTyping(ev.model);
                else hideTyping();
                break;
            case 'clear': clearLog(); this._lastRenderSeq = -1; break;
            default: break;  // unknown/ignorable type — forward-compatible
        }
    }

    // ─── Pipeline-Stepper (bottom bar) ─── Claude Generated ───

    // Rebuild the stepper from the currently-selected workflow's step list.
    renderStepperForSelected() {
        const select = document.getElementById('workflow-select');
        const val = select ? select.value : '';
        this.renderStepper((this.workflowSteps && this.workflowSteps[val]) || []);
    }

    // Render the step nodes; empty list hides the stepper (CSS :empty).
    renderStepper(steps) {
        const el = document.getElementById('pipeline-stepper');
        if (!el) return;
        this._stepperSteps = Array.isArray(steps) ? steps : [];
        el.innerHTML = '';
        this._stepperSteps.forEach((s, i) => {
            const node = document.createElement('div');
            node.className = 'step-node';
            node.dataset.stepId = s.id;
            const dot = document.createElement('span');
            dot.className = 'step-dot';
            dot.textContent = String(i + 1);
            const label = document.createElement('span');
            label.className = 'step-label';
            label.textContent = s.label || s.id;
            node.appendChild(dot);
            node.appendChild(label);
            el.appendChild(node);
        });
    }

    // Highlight progress. status 'completed' marks the step done and the next
    // one active (running); anything else marks the matched step active. Unknown
    // step ids (e.g. the final alias 'classification') are ignored gracefully.
    updateStepper(currentStep, status) {
        const el = document.getElementById('pipeline-stepper');
        if (!el || !this._stepperSteps || !this._stepperSteps.length || !currentStep) return;
        const ids = this._stepperSteps.map(s => s.id);
        const idx = ids.indexOf(currentStep);
        if (idx === -1) return;
        const done = status === 'completed';
        // Compute each node's state directly from idx/status in a single pass so
        // the "next active on completed" isn't clobbered by a later iteration.
        el.querySelectorAll('.step-node').forEach((n, i) => {
            n.classList.remove('is-done', 'is-active');
            let cls = null;
            if (done) {
                if (i <= idx) cls = 'is-done';
                else if (i === idx + 1) cls = 'is-active';
            } else {
                if (i < idx) cls = 'is-done';
                else if (i === idx) cls = 'is-active';
            }
            if (cls) n.classList.add(cls);
        });
    }

    // Mark every step done (called on pipeline completion).
    markStepperComplete() {
        const el = document.getElementById('pipeline-stepper');
        if (!el) return;
        el.querySelectorAll('.step-node').forEach(n => {
            n.classList.remove('is-active');
            n.classList.add('is-done');
        });
    }

    // ─── Input-zone collapse ─── Claude Generated ───

    setInputZoneCollapsed(collapsed) {
        const zone = document.getElementById('input-zone');
        const toggle = document.getElementById('input-zone-toggle');
        if (!zone) return;
        zone.classList.toggle('is-collapsed', collapsed);
        if (toggle) toggle.setAttribute('aria-expanded', String(!collapsed));
    }

    toggleInputZone() {
        const zone = document.getElementById('input-zone');
        if (!zone) return;
        this.setInputZoneCollapsed(!zone.classList.contains('is-collapsed'));
    }

    // Update pipeline status from WebSocket - Claude Generated
    updatePipelineStatus(msg) {
        // WP12: render shared chrome events (DK/GND cards) if present.
        this.dispatchRenderEvents(msg.render_events);

        // Display working title if available - Claude Generated
        if (msg.results && msg.results.working_title) {
            this.displayWorkingTitle(msg.results.working_title);
        }

        // Update auto-save indicator - Claude Generated (2026-01-06)
        if (msg.autosave_timestamp) {
            this.updateAutosaveStatus(msg.autosave_timestamp);
        }

        if (msg.current_step) {
            this.hideRecoveryOption();
            console.log(`📊 Step update: ${msg.current_step}`);

            // Advance the bottom-bar pipeline-stepper. - Claude Generated
            this.updateStepper(msg.current_step, msg.current_step_status);

            // DK search progress: in-place pipeline-bar element (Chat-UX 5/9).
            if (msg.current_step === 'dk_search' || msg.current_step === 'search') {
                this.updateDkProgress(msg.dk_search_progress || null);
            } else {
                this.updateDkProgress(null);
            }
        }

        // LLM tokens arrive as seq-deduped render events (stream blocks in the
        // shared #log, Chat-UX 5/9); streaming_tokens frames stay for the
        // polling API but are no longer rendered here.

        // NOTE: Results are displayed in handleAnalysisComplete() only, not during polling
        // This prevents duplicate display of extracted text - Claude Generated
    }

    // Handle analysis completion
    handleAnalysisComplete(msg) {
        console.log('Analysis complete:', msg);
        this.updateDkProgress(null);

        // WP12: flush any final shared chrome events (e.g. the DK card);
        // final tokens are inside these events too (Chat-UX 5/9).
        this.dispatchRenderEvents(msg.render_events);

        if (msg.status === 'completed') {
            this.setResultsPanelState('completed');
            this.markStepperComplete();  // All steps done - Claude Generated

            // Display working title if available - Claude Generated
            if (msg.results && msg.results.working_title) {
                this.displayWorkingTitle(msg.results.working_title);
            }

            // Check if this is input extraction only or full pipeline - Claude Generated
            const isExtractionOnly = msg.results && msg.results.input_mode === 'extraction_only';

            if (isExtractionOnly) {
                this.appendLocalNotice(`✅ Text erfolgreich extrahiert!`);
            } else {
                this.appendLocalNotice(`✅ Analyse erfolgreich abgeschlossen!`);
            }

            // Display extracted text if available (from input step) - Claude Generated
            if (msg.results && msg.results.original_abstract) {
                document.getElementById('text-input').value = msg.results.original_abstract;
            }

            // Show results panel for both extraction-only and full pipeline - Claude Generated
            // The input zone STAYS collapsed after an analysis: the summary sits
            // outside .input-zone-body, and re-opening the input here pushed the
            // result out of view — the operator had to collapse it again to read
            // what came out. Extraction-only is the exception: its whole point is
            // the extracted text in the input body, which is next to be reviewed
            // and started. - Claude Generated
            if (isExtractionOnly) this.setInputZoneCollapsed(false);
            this.showResultsPanel();

            // For extraction-only, display simplified results - Claude Generated
            if (isExtractionOnly && msg.results) {
                const resultsSummary = document.getElementById('results-summary');
                if (resultsSummary) {
                    const summaryHTML = `
                        <div class="result-item">
                            <strong>Eingabemethode:</strong> ${msg.results.input_type || 'unbekannt'}
                        </div>
                        <div class="result-item">
                            <strong>Extraktionsmethode:</strong> ${msg.results.extraction_method || 'text'}
                        </div>
                        <div class="result-item">
                            <strong>Textlänge:</strong> ${msg.results.original_abstract?.length || 0} Zeichen
                        </div>
                    `;
                    resultsSummary.innerHTML = summaryHTML;
                }
            } else if (msg.results) {
                // Render the completed pipeline payload into the stream and summary panel.
                this.displayResults(msg.results);
            }
        } else if (msg.status === 'error') {
            this.appendLocalNotice(`❌ Fehler: ${msg.error}`, 'error');
        }

        this.isAnalyzing = false;
        this.updateButtonState();
        this.hideRecoveryOption();

        // Clear persisted session - pipeline is done - Claude Generated
        localStorage.removeItem('alima_running_session');

        // Update export button to "completed" state - Claude Generated (2026-01-06)
        if (msg.status === 'completed') {
            this.enableExportButton(false); // false = completed state
        }

        if (this.ws) {
            this.ws.close();
        }
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }

        // Browser notification on completion - Claude Generated
        const title = msg.results?.working_title || null;
        this.showPipelineNotification(msg.status, title, msg.error);
    }

    // Request browser notification permission once, on first pipeline start - Claude Generated
    async requestNotificationPermission() {
        if (!('Notification' in window)) return;
        if (Notification.permission === 'default') {
            await Notification.requestPermission();
        }
    }

    // Show browser notification for pipeline end - Claude Generated
    showPipelineNotification(status, workingTitle, errorMsg) {
        if (!('Notification' in window) || Notification.permission !== 'granted') return;

        if (status === 'completed') {
            const body = workingTitle
                ? `„${workingTitle}" — Schlagwörter & Klassifikation fertig`
                : 'Sacherschließung abgeschlossen';
            new Notification('ALIMA ✅', { body });
        } else if (status === 'error') {
            const body = errorMsg
                ? `Fehler: ${errorMsg}`
                : 'Pipeline-Fehler aufgetreten';
            new Notification('ALIMA ❌', { body });
        }
    }

    // Show extracted text from input step - Claude Generated
    showExtractedText(text) {
        const section = document.getElementById('extracted-text-section');
        const textEl = document.getElementById('extracted-text');

        if (text && text.trim()) {
            textEl.textContent = text;
            section.style.display = 'block';
            console.log(`Extracted text shown: ${text.substring(0, 100)}...`);
        }
    }

    // Display working title after initialisation - Claude Generated
    displayWorkingTitle(workingTitle) {
        if (workingTitle) {
            const titleLabelSection = document.getElementById('title-label-section');
            const titleDisplay = document.getElementById('title-display');
            const titleOverride = document.getElementById('title-override');

            titleDisplay.textContent = workingTitle;
            titleLabelSection.style.display = 'block';  // Show only the label section

            // Pre-fill input field with current title if not already filled by user
            if (!titleOverride.value.trim()) {
                titleOverride.value = workingTitle;
            }

            this.currentWorkingTitle = workingTitle;
        }
    }

    // Apply title override - Claude Generated
    applyTitleOverride() {
        const override = document.getElementById('title-override').value.trim();
        if (override && this.sessionId) {
            // Send override to backend via fetch
            fetch(`/api/session/${this.sessionId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ working_title: override })
            }).catch(e => console.warn('Could not save title override:', e));

            document.getElementById('title-display').textContent = override;
        }
    }

    escapeHtml(value) {
        return String(value ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    normalizeList(value) {
        if (Array.isArray(value)) return value;
        if (typeof value === 'string') {
            return value.split(',').map(item => item.trim()).filter(Boolean);
        }
        return [];
    }

    normalizeClassifications(value, legacyValue) {
        if (Array.isArray(value)) {
            return value.map(item => {
                if (typeof item === 'string') {
                    return { display: item, system: '', code: item };
                }
                if (item && typeof item === 'object') {
                    const display = item.display || [item.system, item.code].filter(Boolean).join(' ').trim() || item.code || '';
                    return {
                        display,
                        system: item.system || '',
                        code: item.code || display,
                        validation_status: item.validation_status || null,
                        is_standard: Object.prototype.hasOwnProperty.call(item, 'is_standard') ? item.is_standard : null,
                        canonical_code: item.canonical_code || item.code || display,
                        label: item.label || null,
                        validation_message: item.validation_message || null,
                        validation_source: item.validation_source || null,
                    };
                }
                return null;
            }).filter(Boolean);
        }

        return this.normalizeList(legacyValue).map(item => ({
            display: item,
            system: '',
            code: item,
            validation_status: null,
            is_standard: null,
            canonical_code: item,
            label: null,
            validation_message: null,
            validation_source: null,
        }));
    }

    getRvkValidationSummary(classifications, backendSummary = null) {
        const rvkEntries = classifications.filter(cls => cls.system === 'RVK');
        if (backendSummary && typeof backendSummary === 'object') {
            return {
                total: backendSummary.rvk_total || rvkEntries.length,
                standard: backendSummary.rvk_standard || 0,
                nonStandard: backendSummary.rvk_non_standard || 0,
                errors: backendSummary.rvk_validation_errors || 0,
            };
        }

        return {
            total: rvkEntries.length,
            standard: rvkEntries.filter(cls => cls.validation_status === 'standard').length,
            nonStandard: rvkEntries.filter(cls => cls.validation_status === 'non_standard').length,
            errors: rvkEntries.filter(cls => cls.validation_status === 'validation_error').length,
        };
    }

    // Display results in stream (Claude Generated - Updated for full results)
    displayResults(results) {
        if (!results) return;

        // Chat-UX 5/9: the old free-text recap into the stream region is
        // gone — the server-side render cards (dk_search /
        // dk_classifications / dk_statistics) already carry that content in
        // the shared #log; the compact overview lives in the summary panel.
        if (results.original_abstract) {
            // Update input text field with extracted text - Claude Generated
            document.getElementById('text-input').value = results.original_abstract;
        }

        // Populate summary panel
        this.populateSummary(results);
    }

    populateSummary(results) {
        const summaryDiv = document.getElementById('results-summary');
        if (!summaryDiv) return;

        summaryDiv.innerHTML = '';

        const finalKeywords = this.normalizeList(results.final_keywords);
        const classifications = this.normalizeClassifications(results.classifications, results.dk_classifications);
        const initialKeywords = this.normalizeList(results.initial_keywords);
        const rvkSummary = this.getRvkValidationSummary(classifications, results.classification_validation);

        if (finalKeywords.length > 0) {
            const item = document.createElement('div');
            item.className = 'results-summary-item keyword';
            item.style.maxHeight = '100px';
            item.style.overflowY = 'auto';
            item.style.wordWrap = 'break-word';
            item.style.whiteSpace = 'normal';
            // Add verification badge if available - Claude Generated
            const verificationBadge = (results.verification && results.verification.stats)
                ? ` <span style="color: #4caf50; font-size: 0.85em;">(${results.verification.stats.verified_count}/${results.verification.stats.total_extracted} GND-verifiziert)</span>`
                : '';
            item.innerHTML = `<strong>GND-Schlagworte:</strong>${verificationBadge} ${finalKeywords.join(', ')}`;
            summaryDiv.appendChild(item);
        }

        if (classifications.length > 0) {
            const item = document.createElement('div');
            item.className = 'results-summary-item classification';
            item.style.maxHeight = '220px';
            item.style.overflowY = 'auto';
            item.style.wordWrap = 'break-word';
            item.style.whiteSpace = 'normal';
            const validationSummary = rvkSummary.total > 0
                ? `<div class="classification-validation-summary">
                    <span class="classification-badge classification-badge--standard">RVK standard: ${rvkSummary.standard}</span>
                    <span class="classification-badge classification-badge--non-standard">RVK nicht standard: ${rvkSummary.nonStandard}</span>
                    ${rvkSummary.errors > 0 ? `<span class="classification-badge classification-badge--unknown">API-Fehler: ${rvkSummary.errors}</span>` : ''}
                </div>`
                : '';

            const itemsHtml = classifications.map(cls => {
                const systemClass = cls.system === 'RVK'
                    ? 'classification-badge classification-badge--rvk'
                    : cls.system === 'DDC'
                    ? 'classification-badge classification-badge--ddc'
                    : 'classification-badge classification-badge--dk';

                let validationHtml = '';
                if (cls.system === 'RVK') {
                    if (cls.validation_status === 'standard') {
                        validationHtml = '<span class="classification-badge classification-badge--standard">standard</span>';
                    } else if (cls.validation_status === 'non_standard') {
                        validationHtml = '<span class="classification-badge classification-badge--non-standard">nicht standard</span>';
                    } else if (cls.validation_status === 'validation_error') {
                        validationHtml = '<span class="classification-badge classification-badge--unknown">API-Fehler</span>';
                    }
                }

                const metaParts = [];
                if (cls.system === 'RVK' && cls.label) {
                    metaParts.push(this.escapeHtml(cls.label));
                }
                if (cls.system === 'RVK' && cls.validation_message && cls.validation_status !== 'standard') {
                    metaParts.push(this.escapeHtml(cls.validation_message));
                }

                return `<div class="classification-entry">
                    <div class="classification-entry__head">
                        <span class="${systemClass}">${this.escapeHtml(cls.system || 'Code')}</span>
                        <span class="classification-entry__code">${this.escapeHtml(cls.display)}</span>
                        ${validationHtml}
                    </div>
                    ${metaParts.length > 0 ? `<div class="classification-entry__meta">${metaParts.join(' · ')}</div>` : ''}
                </div>`;
            }).join('');

            item.innerHTML = `<strong>Klassifikationen:</strong>${validationSummary}<div class="classification-entry-list">${itemsHtml}</div>`;
            summaryDiv.appendChild(item);
        }

        if (initialKeywords.length > 0) {
            const item = document.createElement('div');
            item.className = 'results-summary-item';
            item.style.maxHeight = '100px';
            item.style.overflowY = 'auto';
            item.style.wordWrap = 'break-word';
            item.style.whiteSpace = 'normal';
            item.innerHTML = `<strong>Initiale Schlagworte:</strong> ${initialKeywords.join(', ')}`;
            summaryDiv.appendChild(item);
        }
    }

    // Enable export button with dynamic text - Claude Generated (2026-01-06)
    enableExportButton(isRunning = false) {
        const exportBtn = document.getElementById('export-btn');
        if (!exportBtn) return;

        exportBtn.disabled = false;

        if (isRunning) {
            exportBtn.textContent = '💾 Stand speichern';
            exportBtn.title = 'Exportiert den aktuellen Fortschritt als JSON (kann unvollständig sein)';
        } else {
            exportBtn.textContent = '📥 Speichern';
            exportBtn.title = 'Exportiert die vollständigen Ergebnisse als JSON';
        }
    }

    // Export results as JSON
    async exportResults() {
        if (!this.sessionId) return;

        try {
            const response = await fetch(`/api/export/${this.sessionId}?format=json`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            // Get filename from Content-Disposition header - Claude Generated
            // Supports RFC 5987 format (filename*=UTF-8''encoded) and standard format (filename="name")
            let filename = 'alima_analysis.json';
            const contentDisposition = response.headers.get('content-disposition');
            if (contentDisposition) {
                // Try RFC 5987 format first: filename*=UTF-8''encoded_name
                const rfc5987Match = contentDisposition.match(/filename\*=(?:UTF-8''|utf-8'')([^;\s]+)/i);
                if (rfc5987Match) {
                    try {
                        filename = decodeURIComponent(rfc5987Match[1]);
                    } catch (e) {
                        console.warn('Failed to decode RFC 5987 filename:', e);
                    }
                }
                // Fallback to standard format: filename="name" or filename=name
                if (filename === 'alima_analysis.json') {
                    const standardMatch = contentDisposition.match(/filename=["']?([^"';\s]+)["']?/i);
                    if (standardMatch) {
                        filename = standardMatch[1];
                    }
                }
            }

            // Download file
            const blob = await response.blob();
            if (blob.size === 0) {
                throw new Error('Exportierte Datei ist leer');
            }

            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

            this.appendLocalNotice(`💾 Exportiert: ${filename}`);

        } catch (error) {
            console.error('Export error:', error);
            alert(alimaT('js.alert.export_failed', 'Export fehlgeschlagen: {msg}').replace('{msg}', error.message));
        }
    }

    // Clear results
    async clearResults() {
        this.clearStreamText();
        this.hideResultsPanel();
        this.resetSharedRender();  // WP12: empty the shared render region
        this.setInputZoneCollapsed(false);  // Re-open input for a fresh run - Claude Generated
        this._inputZoneAutoCollapsed = false;  // Re-arm one-shot chat auto-collapse - Claude Generated
        this.renderStepperForSelected();     // Reset stepper to pending - Claude Generated

        // Hide extracted text section - Claude Generated
        const extractedSection = document.getElementById('extracted-text-section');
        if (extractedSection) {
            extractedSection.style.display = 'none';
            document.getElementById('extracted-text').textContent = '';
        }

        // Clear persisted session on explicit "Neue Analyse" - Claude Generated
        localStorage.removeItem('alima_running_session');

        // Cleanup old session
        if (this.sessionId) {
            await fetch(`/api/session/${this.sessionId}`, { method: 'DELETE' });
        }

        // Create new session
        await this.createNewSession();
        this.updateButtonState();
    }

    // Stream text manipulation
    // Reliable scroll to bottom - Claude Generated (2026-01-13)
    scrollToBottom(element) {
        // Method 1: Direct parent scroll
        if (element.parentElement) {
            element.parentElement.scrollTop = element.parentElement.scrollHeight;
        }

        // Method 2: Try container scroll (in case of nested containers)
        const container = element.closest('.stream-output');
        if (container) {
            container.scrollTop = container.scrollHeight;
        }

        // Method 3: Force scroll with requestAnimationFrame for smoothness
        requestAnimationFrame(() => {
            if (element.parentElement) {
                element.parentElement.scrollTop = element.parentElement.scrollHeight;
            }
            if (container) {
                container.scrollTop = container.scrollHeight;
            }
        });
    }

    clearStreamText() {
        // Chat-UX 5/9: the legacy #stream-text region is gone — everything
        // renders in the shared #log; this clears it plus the seq cursor.
        if (typeof clearLog === 'function') {
            clearLog();
        }
        this._lastRenderSeq = -1;
        this.updateDkProgress(null);
    }

    // Client-generated transient notice into the shared #log (reconnect,
    // fallback, validation, abort …) — replaces the legacy #stream-text
    // region (Chat-UX 5/9). Server-side content arrives as render events.
    appendLocalNotice(text, level = 'info') {
        const esc = (typeof _alimaEscapeHtml === 'function')
            ? _alimaEscapeHtml : (s) => String(s);
        const cls = level === 'error'
            ? 'system-message system-message--error' : 'system-message';
        if (typeof appendBlock === 'function') {
            appendBlock(`<div class="${cls}">${esc(text)}</div>`);
        } else {
            console.log(`[notice:${level}]`, text);
        }
    }

    // DK-search progress as an in-place pipeline-bar element instead of one
    // log line per poll frame. Pass null to clear.
    updateDkProgress(progress) {
        const el = document.getElementById('dk-progress');
        if (!el) return;
        el.textContent = progress
            ? `DK-Suche ${progress.current}/${progress.total} (${progress.percent}%)` : '';
    }

    resetResultsPanelContent() {
        const summaryDiv = document.getElementById('results-summary');
        if (summaryDiv) {
            summaryDiv.innerHTML = '';
        }
        this.resetSharedRender();
    }

    // WP12: clear the shared render region and the seq cursor so the next run
    // (server buffer restarts at seq 0) renders from a clean slate.
    resetSharedRender() {
        this._lastRenderSeq = -1;
        if (typeof clearLog === 'function') clearLog();
    }

    // Results panel
    showResultsPanel() {
        document.getElementById('results-panel').style.display = 'flex';
    }

    hideResultsPanel() {
        document.getElementById('results-panel').style.display = 'none';
        this.setResultsPanelState('idle');
    }

    setResultsPanelState(state = 'idle') {
        const panel = document.getElementById('results-panel');
        const title = document.getElementById('results-panel-title');
        if (!panel || !title) return;

        panel.classList.remove('card--success', 'card--warning');
        title.classList.remove('card-label--success', 'card-label--warning');

        if (state === 'running') {
            panel.classList.add('card--warning');
            title.classList.add('card-label--warning');
            title.textContent = '▶ Analyse läuft';
            return;
        }

        if (state === 'completed') {
            panel.classList.add('card--success');
            title.classList.add('card-label--success');
            title.textContent = '✓ Analyse abgeschlossen';
            return;
        }

        title.textContent = 'Analyse';
    }

    // Update button state
    updateButtonState() {
        document.getElementById('analyze-btn').disabled = this.isAnalyzing;
        document.getElementById('analyze-btn').textContent = this.isAnalyzing ? 'Wird analysiert...' : 'Analyse starten';

        // Show/hide cancel button - Claude Generated
        document.getElementById('cancel-btn').style.display = this.isAnalyzing ? 'block' : 'none';

        // Show/hide abort-step button - Claude Generated
        document.getElementById('abort-step-btn').style.display = this.isAnalyzing ? 'block' : 'none';

        // Auto-collapse the input zone when a run starts so the chat/log becomes
        // the focal area. Add-only: the operator re-opens it via the chevron. - Claude Generated
        if (this.isAnalyzing) this.setInputZoneCollapsed(true);
    }

    // Clear session (rename of clearResults) - Claude Generated
    async clearSession() {
        await this.clearResults();
    }

    // Cancel running analysis - Claude Generated
    async cancelAnalysis() {
        if (!this.isAnalyzing || !this.sessionId) {
            alert(alimaT('js.alert.no_analysis', 'Keine Analyse läuft'));
            return;
        }

        try {
            // Request cancellation from backend
            const response = await fetch(`/api/session/${this.sessionId}/cancel`, {
                method: 'POST'
            });

            if (!response.ok) {
                throw new Error(`Failed to cancel: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('Cancellation response:', data);
            this.appendLocalNotice('❌ Analyse durch Benutzer abgebrochen', 'error');

            // Stop polling
            this.isAnalyzing = false;
            this.updateButtonState();

        } catch (error) {
            console.error('Error cancelling analysis:', error);
            alert(alimaT('js.alert.abort_failed', 'Fehler beim Abbrechen: {msg}').replace('{msg}', error.message));
        }
    }

    // Abort only the current LLM step; pipeline continues - Claude Generated
    async abortCurrentStep() {
        if (!this.isAnalyzing || !this.sessionId) return;
        try {
            const response = await fetch(`/api/session/${this.sessionId}/abort_step`, {
                method: 'POST'
            });
            const data = await response.json();
            console.log('Step-abort response:', data);
            this.appendLocalNotice('🛑 Schritt abgebrochen – Pipeline läuft weiter');
        } catch (error) {
            console.error('Error aborting step:', error);
            // Silent failure OK - step may have already finished
        }
    }

    // Open DOI or URL in browser tab - Claude Generated
    openDoiUrl() {
        const input = document.getElementById('doi-input').value.trim();
        if (!input) {
            this.appendLocalNotice('⚠️ Bitte geben Sie eine DOI oder URL ein');
            return;
        }
        let url;
        if (input.startsWith('http://') || input.startsWith('https://')) {
            url = input;
        } else if (input.includes('doi.org/')) {
            const doi = input.split('doi.org/').pop();
            url = `https://doi.org/${doi}`;
        } else {
            // Bare DOI or doi:10.x/y
            const doi = input.replace(/^doi:/, '').trim();
            url = `https://doi.org/${doi}`;
        }
        window.open(url, '_blank', 'noopener,noreferrer');
    }

    // Process DOI/URL input and run initialization - Claude Generated
    async processDoiUrl() {
        const doiUrl = document.getElementById('doi-input').value.trim();

        // Validation only in tab context - Claude Generated
        if (!doiUrl) {
            this.appendLocalNotice(`⚠️ Bitte geben Sie eine DOI oder URL ein`);
            return;
        }

        console.log(`Extracting text from DOI/URL: ${doiUrl}`);
        await this.extractAndFillTextField('doi', doiUrl, null);
    }

    // Process file input and extract text to textfield - Claude Generated
    async processFileInput(file) {
        // Validation only in tab context - Claude Generated
        if (!file) {
            this.appendLocalNotice(`⚠️ Bitte wählen Sie eine Datei aus`);
            return;
        }

        // Determine input type
        let inputType = 'txt';
        if (file.type.includes('pdf')) {
            inputType = 'pdf';
        } else if (file.type.includes('image')) {
            inputType = 'img';
        }

        console.log(`Extracting text from file: ${file.name} (${inputType})`);
        await this.extractAndFillTextField(inputType, null, file);
    }

    // Extract text from various sources and fill the main text field - Claude Generated
    async extractAndFillTextField(inputType, content, file) {
        try {
            // Show extraction progress in stream
            this.appendLocalNotice(`🔄 Extrahiere Text aus ${inputType === 'doi' ? 'DOI/URL' : inputType} …`);

            // Create FormData for multipart request
            const formData = new FormData();
            formData.append('input_type', inputType);
            if (content) {
                formData.append('content', content);
            }
            if (file) {
                formData.append('file', file);
            }

            // Use /api/input endpoint for text extraction only (not full pipeline) - Claude Generated
            const response = await fetch(`/api/input/${this.sessionId}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('Text extraction started:', data);

            // Wait for extraction to complete and capture the session data
            const sessionData = await this.waitForExtractionCompletion();

            // Get the extracted text from session results (not the POST response)
            if (sessionData.results && sessionData.results.original_abstract) {
                // Fill the main text field with extracted text - Claude Generated
                document.getElementById('text-input').value = sessionData.results.original_abstract;
                this.appendLocalNotice(`✅ Text erfolgreich extrahiert (${sessionData.results.extraction_method})`);
                // Track source origin for working title / JSON filename - Claude Generated
                this.pendingSourceType = inputType;
                this.pendingInputSource = content || (file ? file.name : '');
            } else {
                throw new Error('Keine Textextraktion möglich');
            }

            // Clear extraction-specific UI
            this.isAnalyzing = false;
            this.updateButtonState();

        } catch (error) {
            console.error('Extraction error:', error);
            this.appendLocalNotice(`❌ Fehler bei der Textextraktion: ${error.message}`, 'error');
            this.isAnalyzing = false;
            this.updateButtonState();
        }
    }

    // Wait for extraction to complete - Claude Generated
    async waitForExtractionCompletion() {
        return new Promise((resolve, reject) => {
            let attempts = 0;
            const maxAttempts = 60; // 30 seconds max (60 * 500ms)

            const checkStatus = async () => {
                try {
                    const response = await fetch(`/api/session/${this.sessionId}`);
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);

                    const data = await response.json();

                    if (data.status === 'completed' || data.status === 'error') {
                        resolve(data);
                    } else if (attempts < maxAttempts) {
                        attempts++;
                        setTimeout(checkStatus, 500);
                    } else {
                        reject(new Error('Extraction timeout'));
                    }
                } catch (error) {
                    reject(error);
                }
            };

            checkStatus();
        });
    }

    // Get current time string
    getTime() {
        const now = new Date();
        return now.toLocaleTimeString('de-DE');
    }

    // Update auto-save status indicator - Claude Generated (2026-01-06)
    updateAutosaveStatus(timestamp) {
        const indicator = document.getElementById('autosave-status');
        if (!indicator) return;

        // Show indicator
        indicator.style.display = 'inline';

        // Calculate time ago
        const saveTime = new Date(timestamp);
        const now = new Date();
        const secondsAgo = Math.floor((now - saveTime) / 1000);

        let timeText = 'gerade eben';
        if (secondsAgo > 60) {
            const minutesAgo = Math.floor(secondsAgo / 60);
            timeText = `vor ${minutesAgo} Min`;
        } else if (secondsAgo > 5) {
            timeText = `vor ${secondsAgo}s`;
        }

        indicator.textContent = `💾 Gespeichert ${timeText}`;
        indicator.style.color = '#4caf50';  // Green for success

        // Fade back to gray after 3 seconds
        setTimeout(() => {
            indicator.style.color = '#888';
        }, 3000);
    }

    // Show recovery option on WebSocket error/close - Claude Generated
    showRecoveryOption() {
        const recoveryBtn = document.getElementById('recovery-btn');
        const recoveryMsg = document.getElementById('recovery-message');

        if (recoveryBtn) {
            recoveryBtn.style.display = 'inline-block';
            recoveryBtn.onclick = () => this.recoverResults();
        }

        if (recoveryMsg) {
            recoveryMsg.style.display = 'inline';
            recoveryMsg.textContent = 'Verbindung unterbrochen. Ergebnisse können wiederhergestellt werden.';
        }
    }

    hideRecoveryOption() {
        const recoveryBtn = document.getElementById('recovery-btn');
        const recoveryMsg = document.getElementById('recovery-message');

        if (recoveryBtn) {
            recoveryBtn.style.display = 'none';
            recoveryBtn.disabled = false;
        }

        if (recoveryMsg) {
            recoveryMsg.style.display = 'none';
            recoveryMsg.textContent = '';
            recoveryMsg.style.color = '';
        }
    }

    // Attempt recovery - Claude Generated
    async recoverResults() {
        const recoveryBtn = document.getElementById('recovery-btn');
        const recoveryMsg = document.getElementById('recovery-message');

        if (recoveryBtn) recoveryBtn.disabled = true;
        if (recoveryMsg) recoveryMsg.textContent = '🔄 Wiederherstellung läuft...';

        try {
            const response = await fetch(`/api/session/${this.sessionId}/recover`);

            if (!response.ok) {
                // Better error messages based on status code - Claude Generated
                let errorMsg = '❌ Wiederherstellung fehlgeschlagen';
                if (response.status === 404) {
                    errorMsg = '❌ Keine gespeicherten Ergebnisse gefunden';
                } else if (response.status === 422) {
                    errorMsg = '❌ Gespeicherte Datei beschädigt';
                } else if (response.status === 500) {
                    errorMsg = '❌ Server-Fehler bei Wiederherstellung';
                }
                throw new Error(errorMsg);
            }

            const data = await response.json();

            if (data.status === 'recovered') {
                console.log('✓ Recovery successful:', data.metadata);

                // Display recovered results
                this.handleAnalysisComplete({
                    status: 'completed',
                    results: data.results,
                    current_step: 'classification'
                });

                // Enable export button in completed state - Claude Generated (2026-01-06)
                this.enableExportButton(false); // false = completed state

                // Hide recovery UI with success message
                if (recoveryBtn) recoveryBtn.style.display = 'none';
                if (recoveryMsg) {
                    recoveryMsg.textContent = '✅ Ergebnisse erfolgreich wiederhergestellt!';
                    recoveryMsg.style.color = '#4caf50';
                    setTimeout(() => {
                        recoveryMsg.style.display = 'none';
                    }, 5000);
                }

                // Show friendly notification
                this.appendLocalNotice('✅ Analyse erfolgreich wiederhergestellt!');
            }
        } catch (error) {
            console.error('Recovery error:', error);
            if (recoveryMsg) {
                recoveryMsg.textContent = error.message || '❌ Wiederherstellung fehlgeschlagen';
                recoveryMsg.style.color = '#f44336';
            }
            if (recoveryBtn) recoveryBtn.disabled = false;

            // Show detailed error in stream
            this.appendLocalNotice(`⚠️ ${error.message}`, 'error');
        }
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.alima = new AlimaWebapp();
});
