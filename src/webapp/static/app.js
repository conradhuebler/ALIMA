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
        this.streamRawBuffer = '';
        this.streamRenderPending = false;

        this.setupPipelineSteps();
        this.setupEventListeners();
        this.initializeSession();
    }

    // Initialize session when app loads - Claude Generated
    async initializeSession() {
        await this.createNewSession();
        await this.loadModelOverrides();
        // Check if current URL session is already active (page refresh scenario)
        const reconnectedCurrent = await this.checkCurrentSessionState();
        // If not, check localStorage for a different running session
        if (!reconnectedCurrent) await this.checkForRunningSession();
        console.log('Ready for analysis');
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
                this.showResultsPanel();
                this.enableExportButton(true);
                localStorage.setItem('alima_running_session', this.sessionId);
                this.appendStreamText(`🔌 Wiederverbunden mit laufender Analyse…\n`);
                this.connectWebSocket();
                return true;
            } else if (data.status === 'completed' && data.results && Object.keys(data.results).length > 0) {
                this.handleAnalysisComplete({
                    status: 'completed',
                    results: data.results,
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

        this.showResultsPanel();

        if (status === 'running') {
            this.isAnalyzing = true;
            this.updateButtonState();
            this.setResultsPanelState('running');
            this.enableExportButton(true);
            this.appendStreamText(`🔌 Wiederverbunden mit laufender Analyse (${savedId.substring(0, 8)}…)\n`);
            this.connectWebSocket();
        } else {
            // Completed: fetch results directly from session and display
            this.handleAnalysisComplete({
                status: 'completed',
                results: data.results,
                current_step: data.current_step || 'classification'
            });
            this.appendStreamText(`📂 Ergebnisse der abgeschlossenen Analyse wiederhergestellt.\n`);
            localStorage.removeItem('alima_running_session');
        }
    }

    // Load available models for override dropdown - Claude Generated
    async loadModelOverrides() {
        try {
            const response = await fetch('/api/models');
            if (!response.ok) return;
            const models = await response.json();
            const select = document.getElementById('model-override');
            if (!select) return;
            models.forEach(m => {
                const option = document.createElement('option');
                option.value = m.value;
                option.textContent = `${m.provider} | ${m.model}`;
                select.appendChild(option);
            });
            console.log(`Loaded ${models.length} models for override dropdown`);
        } catch (e) {
            console.error('Failed to load models:', e);
        }
    }

    // Pipeline step definitions
    setupPipelineSteps() {
        this.steps = [
            { id: 'input', name: 'Eingabe', description: 'Verarbeitung' },
            { id: 'initialisation', name: 'Initialisierung', description: 'Schlagworte' },
            { id: 'search', name: 'Katalogsuche', description: 'GND/SWB' },
            { id: 'keywords', name: 'Erschließung', description: 'Finale Worte' },
            { id: 'classification', name: 'Klassifikation', description: 'Codes' }
        ];

        this.renderPipelineSteps();
    }

    // Render pipeline steps
    renderPipelineSteps() {
        const container = document.getElementById('pipeline-steps');
        container.innerHTML = '';

        this.steps.forEach((step) => {
            const stepEl = document.createElement('div');
            stepEl.className = 'step pending';
            stepEl.id = `step-${step.id}`;
            stepEl.dataset.step = step.id;
            stepEl.innerHTML = `
                <div class="step-status">▷</div>
                <div class="step-content">
                    <div class="step-name">${step.name}</div>
                    <div class="step-info" data-default-info="${step.description}">${step.description}</div>
                </div>
            `;
            container.appendChild(stepEl);
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

            // Fallback: Create new session via API (old behavior for index.html)
            const response = await fetch('/api/session', { method: 'POST' });
            const data = await response.json();
            this.sessionId = data.session_id;
            console.log('Session created via API:', this.sessionId);
        } catch (error) {
            console.error('Error creating session:', error);
            this.appendStreamText(`❌ Error creating session: ${error.message}`);
        }
    }

    // Start analysis
    async startAnalysis() {
        if (!this.sessionId) {
            alert('Session not initialized. Please refresh the page.');
            return;
        }

        if (this.isAnalyzing) {
            alert('Analysis is already running');
            return;
        }

        // Read ALWAYS from the main text field - Claude Generated
        const textContent = document.getElementById('text-input').value.trim();
        if (!textContent) {
            alert('Bitte geben Sie Text ein oder laden Sie eine Quelle');
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

    // Submit analysis request
    async submitAnalysis(inputType, content, file, sourceType = null, sourceValue = null) {
        try {
            this.isAnalyzing = true;
            this.updateButtonState();
            this.clearStreamText();
            this.resetSteps();
            this.resetResultsPanelContent();
            this.hideRecoveryOption();
            this.setResultsPanelState('running');
            this.showResultsPanel();
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

            // Add global model override if selected - Claude Generated
            const overrideSelect = document.getElementById('model-override');
            if (overrideSelect && overrideSelect.value) {
                formData.append('global_override', overrideSelect.value);
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
            this.appendStreamText(`❌ Error: ${error.message}`);
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
                    results: data.results,
                    streaming_tokens: data.streaming_tokens || {}  // Include tokens from polling
                };

                if (data.status === 'running') {
                    this.updatePipelineStatus(msg);
                    lastStep = data.current_step;
                } else if (data.status === 'completed' || data.status === 'error') {
                    // Display final streaming tokens before completing (Claude Generated)
                    if (data.streaming_tokens && Object.keys(data.streaming_tokens).length > 0) {
                        for (const [stepId, tokens] of Object.entries(data.streaming_tokens)) {
                            if (Array.isArray(tokens) && tokens.length > 0) {
                                // Add step separator between different steps - Claude Generated
                                if (stepId && stepId !== 'input') {
                                    this.appendStreamText(`\n───────────────────\n[${stepId}]\n───────────────────`);
                                }
                                this.appendStreamToken(tokens.join(''));
                            }
                        }
                    }

                    this.handleAnalysisComplete({
                        type: 'complete',
                        status: data.status,
                        results: data.results,
                        error: data.error_message,
                        current_step: data.current_step
                    });
                    clearInterval(this.pollInterval);
                    this.pollInterval = null;
                }
            } catch (error) {
                console.error('Poll error:', error);
                this.appendStreamText(`⚠️ Poll error: ${error.message}`);
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
                    this.appendStreamText(`\n⚠️ WebSocket-Fehler, wechsle zu Polling…\n`);
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
                this.appendStreamText(`\n⚠️ Verbindung unterbrochen, wechsle zu Polling…\n`);
                this.connectViaPolling();
            }
        };
    }

    // Update pipeline status from WebSocket - Claude Generated
    updatePipelineStatus(msg) {
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

            // Map backend step names to frontend
            const stepMap = {
                'initialisation': 'initialisation',
                'search': 'search',
                'dk_search': 'classification',
                'keywords': 'keywords',
                'classification': 'classification',
                'dk_classification': 'classification',
            };
            const stepInfoMap = {
                'initialisation': 'Schlagworte',
                'search': 'GND/SWB',
                'dk_search': 'Katalogabgleich',
                'keywords': 'Finale Worte',
                'classification': 'Codeauswahl',
                'dk_classification': 'Codeauswahl',
            };

            const displayStep = stepMap[msg.current_step] || msg.current_step;
            const stepStatus = msg.current_step_status || 'running';  // Claude Generated
            this.updateStepStatus(displayStep, stepStatus);
            this.updateStepInfo(displayStep, stepInfoMap[msg.current_step]);

            // Mark previous steps as completed
            const stepIndex = this.steps.findIndex(s => s.id === displayStep);
            const upTo = stepStatus === 'completed' ? stepIndex : stepIndex - 1;

            for (let i = 0; i <= upTo; i++) {
                this.updateStepStatus(this.steps[i].id, 'completed');
            }

            // Display DK search progress if available - Claude Generated
            if (msg.dk_search_progress && (msg.current_step === 'dk_search' || msg.current_step === 'search')) {
                const progress = msg.dk_search_progress;
                const progressText = `(${progress.current}/${progress.total} - ${progress.percent}%)`;

                // Find step info element and add progress
                const stepElement = document.querySelector(`.step[data-step="${displayStep}"]`);
                if (stepElement) {
                    const infoElement = stepElement.querySelector('.step-info');
                    if (infoElement) {
                        const baseText = stepInfoMap[msg.current_step] || infoElement.dataset.defaultInfo || infoElement.textContent.split('(')[0].trim();
                        infoElement.textContent = `${baseText} ${progressText}`;
                    }
                }
            }
        }

        // Display streaming tokens (Claude Generated - Real-time LLM output)
        if (msg.streaming_tokens && Object.keys(msg.streaming_tokens).length > 0) {
            // Track last displayed step to add separators - Claude Generated
            if (!this.lastDisplayedStep) {
                this.lastDisplayedStep = null;
            }

            for (const [stepId, tokens] of Object.entries(msg.streaming_tokens)) {
                if (Array.isArray(tokens) && tokens.length > 0) {
                    // Add step separator if step changed - Claude Generated
                    if (stepId && stepId !== this.lastDisplayedStep && stepId !== 'input') {
                        this.appendStreamText(`\n═══ [${stepId}] ═══`);
                        this.lastDisplayedStep = stepId;
                    }

                    // Concatenate and display tokens for this step (no extra newlines)
                    const tokenText = tokens.join('');
                    this.appendStreamToken(tokenText);
                }
            }
        }

        // NOTE: Results are displayed in handleAnalysisComplete() only, not during polling
        // This prevents duplicate display of extracted text - Claude Generated
    }

    // Handle analysis completion
    handleAnalysisComplete(msg) {
        console.log('Analysis complete:', msg);

        if (msg.status === 'completed') {
            this.setResultsPanelState('completed');

            // Display working title if available - Claude Generated
            if (msg.results && msg.results.working_title) {
                this.displayWorkingTitle(msg.results.working_title);
            }

            // Check if this is input extraction only or full pipeline - Claude Generated
            const isExtractionOnly = msg.results && msg.results.input_mode === 'extraction_only';

            if (isExtractionOnly) {
                // Only mark input step as completed for extraction-only
                this.updateStepStatus('input', 'completed');
                this.appendStreamText(`\n✅ Text erfolgreich extrahiert!`);
            } else {
                // Mark all steps as completed for full pipeline
                this.steps.forEach(step => {
                    this.updateStepStatus(step.id, 'completed');
                });
                this.appendStreamText(`\n✅ Analyse erfolgreich abgeschlossen!`);
            }

            // Display extracted text if available (from input step) - Claude Generated
            if (msg.results && msg.results.original_abstract) {
                document.getElementById('text-input').value = msg.results.original_abstract;
            }

            // Show results panel for both extraction-only and full pipeline - Claude Generated
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
            this.appendStreamText(`\n❌ Fehler: ${msg.error}`);
            this.updateStepStatus(msg.current_step, 'error');
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

    // Update step status
    updateStepStatus(stepId, status) {
        const stepEl = document.getElementById(`step-${stepId}`);
        if (!stepEl) return;

        // Update class
        stepEl.className = `step ${status}`;

        // Update status icon
        const statusEl = stepEl.querySelector('.step-status');
        const icons = {
            pending: '▷',
            running: '▶',
            completed: '✓',
            error: '✗'
        };
        statusEl.textContent = icons[status] || '◆';
    }

    updateStepInfo(stepId, text) {
        const stepEl = document.getElementById(`step-${stepId}`);
        if (!stepEl) return;

        const infoEl = stepEl.querySelector('.step-info');
        if (!infoEl) return;

        infoEl.textContent = text || infoEl.dataset.defaultInfo || '';
    }

    // Reset all steps
    resetSteps() {
        this.steps.forEach(step => {
            this.updateStepStatus(step.id, 'pending');
            this.updateStepInfo(step.id, step.description);
        });
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

        const initialKeywords = this.normalizeList(results.initial_keywords);
        const finalKeywords = this.normalizeList(results.final_keywords);
        const classifications = this.normalizeClassifications(results.classifications, results.dk_classifications);
        const rvkSummary = this.getRvkValidationSummary(classifications, results.classification_validation);

        // Display original abstract
        if (results.original_abstract) {
            // Update input text field with extracted text - Claude Generated
            document.getElementById('text-input').value = results.original_abstract;

            this.appendStreamText(`\n[${this.getTime()}] Originalabstract:`);
            this.appendStreamText(`  ${results.original_abstract.substring(0, 150)}${results.original_abstract.length > 150 ? '...' : ''}`);
        }

        // Display initial keywords
        if (initialKeywords.length > 0) {
            this.appendStreamText(`\n[${this.getTime()}] Initiale Schlagworte (frei):`);
            initialKeywords.forEach(kw => {
                this.appendStreamText(`  • ${kw}`);
            });
        }

        // Display final GND-compliant keywords with verification status - Claude Generated
        if (finalKeywords.length > 0) {
            this.appendStreamText(`\n[${this.getTime()}] GND-Schlagworte:`);
            finalKeywords.forEach(kw => {
                this.appendStreamText(`  ✓ ${kw}`);
            });

            // Display verification summary - Claude Generated
            if (results.verification && results.verification.stats) {
                const stats = results.verification.stats;
                this.appendStreamText(`\n[${this.getTime()}] GND-Verifikation: ${stats.verified_count}/${stats.total_extracted} verifiziert`);
                if (results.verification.rejected && results.verification.rejected.length > 0) {
                    const rejectedNames = results.verification.rejected.map(r => r.split('(')[0].trim());
                    this.appendStreamText(`  ⚠️ ${stats.rejected_count} entfernt: ${rejectedNames.join(', ')}`);
                }
            }
        }

        // Display DK/RVK classifications
        if (classifications.length > 0) {
            this.appendStreamText(`\n[${this.getTime()}] DK/RVK Klassifikationen:`);
            classifications.forEach(cls => {
                this.appendStreamText(`  ${cls.display}`);
            });

            if (rvkSummary.nonStandard > 0) {
                this.appendStreamText(`  ⚠️ ${rvkSummary.nonStandard} RVK-Notation(en) sind nicht standardisiert`);
            } else if (rvkSummary.total > 0 && rvkSummary.errors === 0) {
                this.appendStreamText(`  ✓ RVK-Prüfung: ${rvkSummary.standard}/${rvkSummary.total} standardisiert`);
            }

            if (rvkSummary.errors > 0) {
                this.appendStreamText(`  ⚠️ RVK-Prüfung unvollständig: ${rvkSummary.errors} API-Fehler`);
            }

            const provenance = results.rvk_provenance || {};
            const provenanceParts = [];
            if (provenance.catalog_standard > 0) provenanceParts.push(`Katalog standard ${provenance.catalog_standard}`);
            if (provenance.catalog_nonstandard > 0) provenanceParts.push(`Katalog lokal ${provenance.catalog_nonstandard}`);
            if (provenance.rvk_gnd_index > 0) provenanceParts.push(`RVK-GND-Index ${provenance.rvk_gnd_index}`);
            if (provenance.rvk_api > 0) provenanceParts.push(`RVK-API-Label ${provenance.rvk_api}`);
            if (provenanceParts.length > 0) {
                this.appendStreamText(`  ℹ️ RVK-Quellen: ${provenanceParts.join(', ')}`);
            }
        }

        // Display DK search results summary
        if (results.dk_search_results && results.dk_search_results.length > 0) {
            this.appendStreamText(`\n[${this.getTime()}] DK-Suche:`);
            results.dk_search_results.forEach(result => {
                const keyword = result.keyword || 'unbekannt';
                const classifications = Array.isArray(result.classifications) ? result.classifications : [];
                const titleSet = new Set();
                classifications.forEach(cls => {
                    const titles = Array.isArray(cls.titles) ? cls.titles : [];
                    titles.forEach(title => {
                        const clean = String(title || '').trim();
                        if (clean) titleSet.add(clean);
                    });
                });

                if (titleSet.size > 0) {
                    this.appendStreamText(`  ${keyword}: ${titleSet.size} Titel`);
                } else {
                    this.appendStreamText(`  ${keyword}: ${classifications.length} Klassifikationen`);
                }
            });
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
            exportBtn.textContent = '💾 Aktuellen Stand exportieren';
            exportBtn.title = 'Exportiert den aktuellen Fortschritt (kann unvollständig sein)';
        } else {
            exportBtn.textContent = '📥 JSON Exportieren';
            exportBtn.title = 'Exportiert die vollständigen Ergebnisse';
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

            this.appendStreamText(`Exportiert: ${filename}`);

        } catch (error) {
            console.error('Export error:', error);
            alert(`Export fehlgeschlagen: ${error.message}`);
        }
    }

    // Clear results
    async clearResults() {
        this.clearStreamText();
        this.resetSteps();
        this.hideResultsPanel();

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
    appendStreamText(text) {
        if (this.streamRawBuffer && !this.streamRawBuffer.endsWith('\n')) {
            this.streamRawBuffer += '\n';
        }
        this.streamRawBuffer += `${text}\n`;
        this.scheduleStreamRender();
    }

    appendStreamToken(text) {
        this.streamRawBuffer += text;
        this.scheduleStreamRender();
    }

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
        this.streamRawBuffer = '';
        this.streamRenderPending = false;
        const streamEl = document.getElementById('stream-text');
        if (streamEl) {
            streamEl.innerHTML = '';
        }
        this.lastDisplayedStep = null;  // Reset step tracking - Claude Generated
    }

    scheduleStreamRender() {
        if (this.streamRenderPending) return;
        this.streamRenderPending = true;
        requestAnimationFrame(() => {
            this.streamRenderPending = false;
            this.renderStreamBuffer();
        });
    }

    renderStreamBuffer() {
        const streamEl = document.getElementById('stream-text');
        if (!streamEl) return;
        streamEl.innerHTML = this.streamBufferToHtml(this.streamRawBuffer);
        this.scrollToBottom(streamEl);
    }

    streamBufferToHtml(text) {
        const lines = String(text || '').replace(/\r\n/g, '\n').split('\n');
        const html = [];
        let index = 0;

        while (index < lines.length) {
            const line = this.normalizeSpecialLogLine(lines[index]);
            const trimmed = line.trim();

            if (!trimmed) {
                html.push('<div class="stream-log-blank"></div>');
                index += 1;
                continue;
            }

            if (trimmed === '## Analyse') {
                html.push('<h2 class="stream-md-heading stream-md-heading--2">Analyse</h2>');
                index += 1;
                continue;
            }

            if (this.isStructuredSectionLabel(trimmed)) {
                html.push(`<h3 class="stream-md-heading stream-md-heading--3">${this.renderInlineMarkdown(trimmed.replace(/:$/, ''))}</h3>`);
                index += 1;
                continue;
            }

            if (trimmed.startsWith('ℹ️ RVK-Zweitranking')) {
                html.push(`<div class="stream-log-line stream-log-line--info">${this.renderInlineMarkdown(trimmed)}</div>`);
                index += 1;
                continue;
            }

            if (this.isMarkdownTableStart(lines, index)) {
                const tableResult = this.renderMarkdownTable(lines, index);
                html.push(tableResult.html);
                index = tableResult.nextIndex;
                continue;
            }

            if (this.isPipeRecordBlockStart(lines, index)) {
                const tableResult = this.renderPipeRecordTable(lines, index);
                html.push(tableResult.html);
                index = tableResult.nextIndex;
                continue;
            }

            if (/^#{1,3}\s+/.test(trimmed)) {
                const level = Math.min(3, trimmed.match(/^#+/)[0].length);
                const content = trimmed.replace(/^#{1,3}\s+/, '');
                html.push(`<h${level} class="stream-md-heading stream-md-heading--${level}">${this.renderInlineMarkdown(content)}</h${level}>`);
                index += 1;
                continue;
            }

            if (/^\s*[-*]\s+/.test(line)) {
                const items = [];
                while (index < lines.length && /^\s*[-*]\s+/.test(lines[index])) {
                    items.push(lines[index].replace(/^\s*[-*]\s+/, ''));
                    index += 1;
                }
                const itemsHtml = items
                    .map(item => `<li>${this.renderInlineMarkdown(item)}</li>`)
                    .join('');
                html.push(`<ul class="stream-md-list">${itemsHtml}</ul>`);
                continue;
            }

            html.push(`<div class="stream-log-line">${this.renderInlineMarkdown(line)}</div>`);
            index += 1;
        }

        return html.join('');
    }

    normalizeSpecialLogLine(line) {
        let normalized = String(line || '');
        normalized = normalized.replace(/<\|begin_of_thought\|>/g, '## Analyse');
        normalized = normalized.replace(/<\|end_of_thought\|>/g, '').trimEnd();
        return normalized;
    }

    isStructuredSectionLabel(line) {
        const trimmed = String(line || '').trim();
        return [
            'DK-Profil für RVK-Zweitranking:',
            'RVK-Kandidaten für DK-basiertes Zweitranking:',
            'RVK-Bewertung aus DK-basiertem Zweitranking:',
            'RVK-Auswahl nach DK-basiertem Zweitranking:',
        ].includes(trimmed);
    }

    isMarkdownTableStart(lines, index) {
        if (index + 1 >= lines.length) return false;
        return this.isPotentialMarkdownTableRow(lines[index]) && this.isMarkdownTableSeparator(lines[index + 1]);
    }

    isPotentialMarkdownTableRow(line) {
        const trimmed = String(line || '').trim();
        if (!trimmed || !trimmed.includes('|')) return false;
        const pipeCount = (trimmed.match(/\|/g) || []).length;
        if (pipeCount < 2) return false;
        if (trimmed.startsWith('🔍 ') || trimmed.startsWith('✅ ') || trimmed.startsWith('⚠️ ') || trimmed.startsWith('❌ ')) {
            return false;
        }
        return true;
    }

    isMarkdownTableSeparator(line) {
        const trimmed = String(line || '').trim();
        if (!trimmed.includes('-')) return false;
        const normalized = trimmed.replace(/^\|/, '').replace(/\|$/, '').trim();
        const cells = normalized.split('|').map(cell => cell.trim()).filter(Boolean);
        return cells.length >= 2 && cells.every(cell => /^:?-{3,}:?$/.test(cell));
    }

    parseMarkdownTableRow(line) {
        const trimmed = String(line || '').trim().replace(/^\|/, '').replace(/\|$/, '');
        return trimmed.split('|').map(cell => cell.trim());
    }

    isPipeRecordBlockStart(lines, index) {
        if (index + 1 >= lines.length) return false;
        const current = String(lines[index] || '').trim();
        const next = String(lines[index + 1] || '').trim();
        return this.isPipeRecordLine(current) && this.isPipeRecordLine(next);
    }

    isPipeRecordLine(line) {
        const trimmed = String(line || '').trim();
        if (!trimmed || this.isMarkdownTableSeparator(trimmed)) {
            return false;
        }
        if (trimmed.startsWith('═══') || trimmed.startsWith('[') || trimmed.startsWith('ℹ️ ') || trimmed.startsWith('✅ ') || trimmed.startsWith('⚠️ ') || trimmed.startsWith('❌ ') || trimmed.startsWith('🔎 ')) {
            return false;
        }
        const pipeCount = (trimmed.match(/\|/g) || []).length;
        return pipeCount >= 2;
    }

    splitPipeRecordLine(line) {
        return String(line || '').split('|').map(cell => cell.trim()).filter(Boolean);
    }

    renderPipeRecordTable(lines, startIndex) {
        const rows = [];
        let index = startIndex;
        const keyValueRows = [];
        let renderAsKeyValue = true;

        while (index < lines.length && this.isPipeRecordLine(lines[index])) {
            const row = this.splitPipeRecordLine(lines[index]);
            if (row.length >= 2) {
                rows.push(row);
                const head = row[0] || '';
                const detailCells = row.slice(1);
                const details = [];
                for (const cell of detailCells) {
                    const match = cell.match(/^([^:]+):\s*(.+)$/);
                    if (match) {
                        details.push({
                            label: match[1].trim(),
                            value: match[2].trim(),
                        });
                    } else {
                        renderAsKeyValue = false;
                    }
                }
                if (details.length > 0) {
                    keyValueRows.push({ head, details });
                } else {
                    renderAsKeyValue = false;
                }
            }
            index += 1;
        }

        if (renderAsKeyValue && keyValueRows.length > 0) {
            const bodyHtml = keyValueRows
                .map(row => {
                    const detailHtml = row.details
                        .map(detail => `<div class="stream-record-detail"><span class="stream-record-detail__label">${this.renderInlineMarkdown(detail.label)}</span><span class="stream-record-detail__value">${this.renderInlineMarkdown(detail.value)}</span></div>`)
                        .join('');
                    return `<tr><th class="stream-record-head">${this.renderInlineMarkdown(row.head)}</th><td>${detailHtml}</td></tr>`;
                })
                .join('');

            return {
                html: `<table class="stream-md-table stream-md-table--records stream-md-table--keyvalue"><tbody>${bodyHtml}</tbody></table>`,
                nextIndex: index,
            };
        }

        let maxColumns = 0;
        for (const row of rows) {
            maxColumns = Math.max(maxColumns, row.length);
        }
        const bodyHtml = rows
            .map(row => {
                const padded = Array.from({ length: maxColumns }, (_, cellIndex) => row[cellIndex] || '');
                return `<tr>${padded.map(cell => `<td>${this.renderInlineMarkdown(cell)}</td>`).join('')}</tr>`;
            })
            .join('');

        return {
            html: `<table class="stream-md-table stream-md-table--records"><tbody>${bodyHtml}</tbody></table>`,
            nextIndex: index,
        };
    }

    renderMarkdownTable(lines, startIndex) {
        const headers = this.parseMarkdownTableRow(lines[startIndex]);
        const rows = [];
        let index = startIndex + 2;

        while (index < lines.length && this.isPotentialMarkdownTableRow(lines[index])) {
            rows.push(this.parseMarkdownTableRow(lines[index]));
            index += 1;
        }

        const headerHtml = headers
            .map(cell => `<th>${this.renderInlineMarkdown(cell)}</th>`)
            .join('');
        const bodyHtml = rows
            .map(row => {
                const cells = headers.map((_, cellIndex) => row[cellIndex] || '');
                return `<tr>${cells.map(cell => `<td>${this.renderInlineMarkdown(cell)}</td>`).join('')}</tr>`;
            })
            .join('');

        return {
            html: `<table class="stream-md-table"><thead><tr>${headerHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>`,
            nextIndex: index,
        };
    }

    renderInlineMarkdown(text) {
        let html = this.escapeHtml(text);
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
        return html;
    }

    resetResultsPanelContent() {
        const summaryDiv = document.getElementById('results-summary');
        if (summaryDiv) {
            summaryDiv.innerHTML = '';
        }
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
    }

    // Clear session (rename of clearResults) - Claude Generated
    async clearSession() {
        await this.clearResults();
    }

    // Cancel running analysis - Claude Generated
    async cancelAnalysis() {
        if (!this.isAnalyzing || !this.sessionId) {
            alert('Keine Analyse läuft');
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
            this.appendStreamText('\n❌ Analyse durch Benutzer abgebrochen\n');

            // Stop polling
            this.isAnalyzing = false;
            this.updateButtonState();

        } catch (error) {
            console.error('Error cancelling analysis:', error);
            alert('Fehler beim Abbrechen: ' + error.message);
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
            this.appendStreamText('\n🛑 Schritt abgebrochen – Pipeline läuft weiter\n');
        } catch (error) {
            console.error('Error aborting step:', error);
            // Silent failure OK - step may have already finished
        }
    }

    // Open DOI or URL in browser tab - Claude Generated
    openDoiUrl() {
        const input = document.getElementById('doi-input').value.trim();
        if (!input) {
            this.appendStreamText('⚠️ Bitte geben Sie eine DOI oder URL ein');
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
            this.appendStreamText(`⚠️ Bitte geben Sie eine DOI oder URL ein`);
            return;
        }

        console.log(`Extracting text from DOI/URL: ${doiUrl}`);
        await this.extractAndFillTextField('doi', doiUrl, null);
    }

    // Process file input and extract text to textfield - Claude Generated
    async processFileInput(file) {
        // Validation only in tab context - Claude Generated
        if (!file) {
            this.appendStreamText(`⚠️ Bitte wählen Sie eine Datei aus`);
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
            this.appendStreamText(`\n🔄 Extrahiere Text aus ${inputType === 'doi' ? 'DOI/URL' : inputType}...`);

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
                this.appendStreamText(`✅ Text erfolgreich extrahiert (${sessionData.results.extraction_method})`);
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
            this.appendStreamText(`❌ Fehler bei der Textextraktion: ${error.message}`);
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
                this.appendStreamText('\n✅ Analyse erfolgreich wiederhergestellt!\n');
            }
        } catch (error) {
            console.error('Recovery error:', error);
            if (recoveryMsg) {
                recoveryMsg.textContent = error.message || '❌ Wiederherstellung fehlgeschlagen';
                recoveryMsg.style.color = '#f44336';
            }
            if (recoveryBtn) recoveryBtn.disabled = false;

            // Show detailed error in stream
            this.appendStreamText(`\n⚠️ ${error.message}\n`);
        }
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.alima = new AlimaWebapp();
});
