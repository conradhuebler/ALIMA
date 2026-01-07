/**
 * ALIMA Webapp Frontend - Claude Generated
 * Handles UI interactions and WebSocket communication
 */

class AlimaWebapp {
    constructor() {
        this.sessionId = null;
        this.isAnalyzing = false;
        this.currentStep = 0;
        this.ws = null;
        this.cameraStream = null;
        this.capturedCameraImage = null;
        this.cameraBlob = null;

        this.setupPipelineSteps();
        this.setupEventListeners();
        this.initializeSession();
    }

    // Initialize session when app loads - Claude Generated
    async initializeSession() {
        await this.createNewSession();
        console.log('Ready for analysis');
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
            stepEl.innerHTML = `
                <div class="step-status">▷</div>
                <div class="step-content">
                    <div class="step-name">${step.name}</div>
                    <div class="step-info">${step.description}</div>
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

        // Clear logs button
        document.getElementById('clear-logs-btn').addEventListener('click', () => {
            document.getElementById('stream-text').textContent = '';
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
            const response = await fetch('/api/session', { method: 'POST' });
            const data = await response.json();
            this.sessionId = data.session_id;
            console.log('Session created:', this.sessionId);
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

        // Always use text input type for analysis - Claude Generated
        // The text field is the primary source for analysis
        // Input methods (DOI, File, Webcam) just populate this field
        await this.submitAnalysis('text', textContent, null);
    }

    // Submit analysis request
    async submitAnalysis(inputType, content, file) {
        try {
            this.isAnalyzing = true;
            this.updateButtonState();
            this.clearStreamText();
            this.resetSteps();

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

            const response = await fetch(`/api/analyze/${this.sessionId}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('Analysis started:', data);

            // Show results panel immediately - Claude Generated (2026-01-06)
            this.showResultsPanel();

            // Enable export button immediately - Claude Generated (2026-01-06)
            this.enableExportButton(true); // true = running state

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

        let lastStep = null;
        let pollCount = 0;
        const maxPolls = 300; // 2.5 minutes max (300 * 0.5s)

        const pollInterval = setInterval(async () => {
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
                    clearInterval(pollInterval);
                }
            } catch (error) {
                console.error('Poll error:', error);
                this.appendStreamText(`⚠️ Poll error: ${error.message}`);
            }

            // Timeout after max polls
            if (pollCount > maxPolls) {
                clearInterval(pollInterval);
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
            }
        };

        this.ws.onerror = (error) => {
            wsConnected = true; // Prevent timeout from firing
            clearTimeout(wsTimeout);
            console.error('WebSocket error:', error);
            this.appendStreamText(`⚠️ WebSocket error, using polling...`);
            this.showRecoveryOption(); // Show recovery button on error - Claude Generated
            this.connectViaPolling();
        };

        this.ws.onclose = (event) => {
            console.log('WebSocket closed:', event.code);

            // Abnormal closure (timeout or error) - Claude Generated
            if (event.code === 1006 || event.code === 1011) {
                this.showRecoveryOption();
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
            console.log(`📊 Step update: ${msg.current_step}`);

            // Map backend step names to frontend
            const stepMap = {
                'initialisation': 'initialisation',
                'search': 'search',
                'dk_search': 'search',  // dk_search maps to search visually
                'keywords': 'keywords',
                'classification': 'classification'
            };

            const displayStep = stepMap[msg.current_step] || msg.current_step;
            this.updateStepStatus(displayStep, 'running');

            // Mark previous steps as completed
            const stepIndex = this.steps.findIndex(s => s.id === displayStep);

            for (let i = 0; i < stepIndex; i++) {
                this.updateStepStatus(this.steps[i].id, 'completed');
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
            }
        } else if (msg.status === 'error') {
            this.appendStreamText(`\n❌ Fehler: ${msg.error}`);
            this.updateStepStatus(msg.current_step, 'error');
        }

        this.isAnalyzing = false;
        this.updateButtonState();

        // Update export button to "completed" state - Claude Generated (2026-01-06)
        if (msg.status === 'completed') {
            this.enableExportButton(false); // false = completed state
        }

        if (this.ws) {
            this.ws.close();
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

    // Reset all steps
    resetSteps() {
        this.steps.forEach(step => {
            this.updateStepStatus(step.id, 'pending');
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

    // Display results in stream (Claude Generated - Updated for full results)
    displayResults(results) {
        if (!results) return;

        // Display original abstract
        if (results.original_abstract) {
            // Update input text field with extracted text - Claude Generated
            document.getElementById('text-input').value = results.original_abstract;

            this.appendStreamText(`\n[${this.getTime()}] Originalabstract:`);
            this.appendStreamText(`  ${results.original_abstract.substring(0, 150)}${results.original_abstract.length > 150 ? '...' : ''}`);
        }

        // Display initial keywords
        if (results.initial_keywords && results.initial_keywords.length > 0) {
            this.appendStreamText(`\n[${this.getTime()}] Initiale Schlagworte (frei):`);
            results.initial_keywords.forEach(kw => {
                this.appendStreamText(`  • ${kw}`);
            });
        }

        // Display final GND-compliant keywords
        if (results.final_keywords && results.final_keywords.length > 0) {
            this.appendStreamText(`\n[${this.getTime()}] GND-Schlagworte:`);
            results.final_keywords.forEach(kw => {
                this.appendStreamText(`  ✓ ${kw}`);
            });
        }

        // Display DK/RVK classifications
        if (results.dk_classifications && results.dk_classifications.length > 0) {
            this.appendStreamText(`\n[${this.getTime()}] DK/RVK Klassifikationen:`);
            results.dk_classifications.forEach(cls => {
                this.appendStreamText(`  ${cls}`);
            });
        }

        // Display DK search results summary
        if (results.dk_search_results && results.dk_search_results.length > 0) {
            this.appendStreamText(`\n[${this.getTime()}] DK-Suche:`);
            results.dk_search_results.forEach(result => {
                const keyword = result.keyword || 'unbekannt';
                const count = result.count || 0;
                this.appendStreamText(`  ${keyword}: ${count} Titel`);
            });
        }

        // Populate summary panel
        this.populateSummary(results);
    }

    populateSummary(results) {
        const summaryDiv = document.getElementById('results-summary');
        if (!summaryDiv) return;

        summaryDiv.innerHTML = '';

        if (results.final_keywords && results.final_keywords.length > 0) {
            const item = document.createElement('div');
            item.className = 'results-summary-item keyword';
            item.style.maxHeight = '100px';
            item.style.overflowY = 'auto';
            item.style.wordWrap = 'break-word';
            item.style.whiteSpace = 'normal';
            item.innerHTML = `<strong>GND-Schlagworte:</strong> ${results.final_keywords.join(', ')}`;
            summaryDiv.appendChild(item);
        }

        if (results.dk_classifications && results.dk_classifications.length > 0) {
            const item = document.createElement('div');
            item.className = 'results-summary-item classification';
            item.style.maxHeight = '120px';
            item.style.overflowY = 'auto';
            item.style.wordWrap = 'break-word';
            item.style.whiteSpace = 'normal';
            item.innerHTML = `<strong>Klassifikationen:</strong> ${results.dk_classifications.join(', ')}`;
            summaryDiv.appendChild(item);
        }

        if (results.initial_keywords && results.initial_keywords.length > 0) {
            const item = document.createElement('div');
            item.className = 'results-summary-item';
            item.style.maxHeight = '100px';
            item.style.overflowY = 'auto';
            item.style.wordWrap = 'break-word';
            item.style.whiteSpace = 'normal';
            item.innerHTML = `<strong>Initiale Schlagworte:</strong> ${results.initial_keywords.join(', ')}`;
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

            // Get filename from Content-Disposition header
            const filename = response.headers
                .get('content-disposition')
                ?.split('filename=')[1]
                ?.replace(/"/g, '') || 'alima_analysis.json';

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
        const streamEl = document.getElementById('stream-text');
        streamEl.textContent += text + '\n';
        streamEl.parentElement.scrollTop = streamEl.parentElement.scrollHeight;
    }

    appendStreamToken(text) {
        // Append token without adding newline (for streaming output)
        const streamEl = document.getElementById('stream-text');
        streamEl.textContent += text;
        streamEl.parentElement.scrollTop = streamEl.parentElement.scrollHeight;
    }

    clearStreamText() {
        document.getElementById('stream-text').textContent = '';
        this.lastDisplayedStep = null;  // Reset step tracking - Claude Generated
    }

    // Results panel
    showResultsPanel() {
        document.getElementById('results-panel').style.display = 'flex';
    }

    hideResultsPanel() {
        document.getElementById('results-panel').style.display = 'none';
    }

    // Update button state
    updateButtonState() {
        document.getElementById('analyze-btn').disabled = this.isAnalyzing;
        document.getElementById('analyze-btn').textContent = this.isAnalyzing ? 'Wird analysiert...' : 'Analyse starten';

        // Show/hide cancel button - Claude Generated
        document.getElementById('cancel-btn').style.display = this.isAnalyzing ? 'block' : 'none';
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
