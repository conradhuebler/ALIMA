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
            { id: 'input', name: 'Input', description: 'Processing' },
            { id: 'initialisation', name: 'Init', description: 'Keywords' },
            { id: 'search', name: 'Search', description: 'Catalog' },
            { id: 'keywords', name: 'Refine', description: 'Subjects' },
            { id: 'classification', name: 'Classify', description: 'Codes' }
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

        // Analyze button
        document.getElementById('analyze-btn').addEventListener('click', () => {
            this.startAnalysis();
        });

        // Export button
        document.getElementById('export-btn').addEventListener('click', () => {
            this.exportResults();
        });

        // Clear button
        document.getElementById('clear-btn').addEventListener('click', () => {
            this.clearResults();
        });

        // File input
        document.getElementById('file-input').addEventListener('change', (e) => {
            const fileName = e.target.files[0]?.name || '';
            document.getElementById('file-name').textContent = fileName ? `✓ ${fileName}` : '';
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
        const video = document.getElementById('camera-video');
        const canvas = document.getElementById('camera-canvas');

        startBtn.addEventListener('click', async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
                video.srcObject = stream;
                video.style.display = 'block';
                this.cameraStream = stream;
                startBtn.style.display = 'none';
                captureBtn.style.display = 'block';
                stopBtn.style.display = 'block';
            } catch (error) {
                alert('Kamera nicht verfügbar: ' + error.message);
            }
        });

        captureBtn.addEventListener('click', () => {
            const ctx = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);
            const imageData = canvas.toDataURL('image/jpeg', 0.8);

            document.getElementById('camera-image').src = imageData;
            document.getElementById('camera-preview').style.display = 'flex';
            this.capturedCameraImage = imageData;

            video.style.display = 'none';
            captureBtn.style.display = 'none';
            stopBtn.textContent = 'Neue Aufnahme';
        });

        stopBtn.addEventListener('click', () => {
            if (this.cameraStream) {
                this.cameraStream.getTracks().forEach(track => track.stop());
            }
            video.style.display = 'none';
            video.srcObject = null;
            document.getElementById('camera-preview').style.display = 'none';
            this.capturedCameraImage = null;
            startBtn.style.display = 'block';
            captureBtn.style.display = 'none';
            stopBtn.style.display = 'none';
            stopBtn.textContent = 'Beenden';
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

        // Get active tab and input
        const activeTab = document.querySelector('.tab-content.active');
        let inputType, content, file;

        if (activeTab.id === 'text-tab') {
            inputType = 'text';
            content = document.getElementById('text-input').value;
            if (!content.trim()) {
                alert('Please enter some text');
                return;
            }
        } else if (activeTab.id === 'doi-tab') {
            inputType = 'doi';
            content = document.getElementById('doi-input').value;
            if (!content.trim()) {
                alert('Please enter a DOI or URL');
                return;
            }
        } else if (activeTab.id === 'file-tab') {
            const fileInput = document.getElementById('file-input');
            if (!fileInput.files.length) {
                alert('Please select a file');
                return;
            }
            file = fileInput.files[0];
            if (file.type.includes('pdf')) {
                inputType = 'pdf';
            } else if (file.type.includes('image')) {
                inputType = 'img';
            } else {
                inputType = 'txt';
            }
        } else if (activeTab.id === 'camera-tab') {
            if (!this.capturedCameraImage) {
                alert('Please take a photo first');
                return;
            }
            inputType = 'img';
            // Convert data URL to Blob
            const response = await fetch(this.capturedCameraImage);
            const blob = await response.blob();
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(new File([blob], 'camera_photo.jpg', { type: 'image/jpeg' }));
            file = dataTransfer.files[0];
        }

        await this.submitAnalysis(inputType, content, file);
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
            this.connectViaPolling();
        };

        this.ws.onclose = () => {
            console.log('WebSocket closed');
        };
    }

    // Update pipeline status from WebSocket - Claude Generated
    updatePipelineStatus(msg) {
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
            for (const [stepId, tokens] of Object.entries(msg.streaming_tokens)) {
                if (Array.isArray(tokens) && tokens.length > 0) {
                    // Concatenate and display tokens for this step (no extra newlines)
                    const tokenText = tokens.join('');
                    this.appendStreamToken(tokenText);
                }
            }
        }

        // Update stream with results
        if (msg.results && Object.keys(msg.results).length > 0) {
            console.log('Results available:', Object.keys(msg.results));
            this.displayResults(msg.results);
        }
    }

    // Handle analysis completion
    handleAnalysisComplete(msg) {
        console.log('Analysis complete:', msg);

        if (msg.status === 'completed') {
            // Mark all steps as completed
            this.steps.forEach(step => {
                this.updateStepStatus(step.id, 'completed');
            });

            this.appendStreamText(`\n✅ Analysis completed successfully!`);
            this.showResultsPanel();
        } else if (msg.status === 'error') {
            this.appendStreamText(`\n❌ Error: ${msg.error}`);
            this.updateStepStatus(msg.current_step, 'error');
        }

        this.isAnalyzing = false;
        this.updateButtonState();

        if (this.ws) {
            this.ws.close();
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

    // Display results in stream (Claude Generated - Updated for full results)
    displayResults(results) {
        if (!results) return;

        // Display original abstract
        if (results.original_abstract) {
            this.appendStreamText(`\n[${this.getTime()}] 📄 Original Abstract:`);
            this.appendStreamText(`  ${results.original_abstract.substring(0, 150)}${results.original_abstract.length > 150 ? '...' : ''}`);
        }

        // Display initial keywords
        if (results.initial_keywords && results.initial_keywords.length > 0) {
            this.appendStreamText(`\n[${this.getTime()}] 🔤 Initial Keywords (free):`);
            results.initial_keywords.forEach(kw => {
                this.appendStreamText(`  • ${kw}`);
            });
        }

        // Display final GND-compliant keywords
        if (results.final_keywords && results.final_keywords.length > 0) {
            this.appendStreamText(`\n[${this.getTime()}] 📚 Final GND Keywords:`);
            results.final_keywords.forEach(kw => {
                this.appendStreamText(`  ✓ ${kw}`);
            });
        }

        // Display DK/RVK classifications
        if (results.dk_classifications && results.dk_classifications.length > 0) {
            this.appendStreamText(`\n[${this.getTime()}] 📊 DK/RVK Classifications:`);
            results.dk_classifications.forEach(cls => {
                this.appendStreamText(`  📋 ${cls}`);
            });
        }

        // Display DK search results summary
        if (results.dk_search_results && results.dk_search_results.length > 0) {
            this.appendStreamText(`\n[${this.getTime()}] 🔍 DK Search Results:`);
            results.dk_search_results.forEach(result => {
                const keyword = result.keyword || 'unknown';
                const count = result.count || 0;
                this.appendStreamText(`  🔎 ${keyword}: ${count} titles`);
            });
        }
    }

    // Export results as JSON
    async exportResults() {
        if (!this.sessionId) return;

        try {
            const response = await fetch(`/api/export/${this.sessionId}?format=json`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            // Get filename from Content-Disposition header
            const filename = response.headers
                .get('content-disposition')
                ?.split('filename=')[1]
                ?.replace(/"/g, '') || 'alima_analysis.json';

            // Download file
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

            this.appendStreamText(`✅ Exported: ${filename}`);

        } catch (error) {
            console.error('Export error:', error);
            alert(`Export failed: ${error.message}`);
        }
    }

    // Clear results
    async clearResults() {
        this.clearStreamText();
        this.resetSteps();
        this.hideResultsPanel();

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
        document.getElementById('analyze-btn').textContent = this.isAnalyzing ? '⏳ Analyzing...' : '🚀 Auto-Pipeline';
    }

    // Get current time string
    getTime() {
        const now = new Date();
        return now.toLocaleTimeString('de-DE');
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.alima = new AlimaWebapp();
});
