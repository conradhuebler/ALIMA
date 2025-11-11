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

        this.setupPipelineSteps();
        this.setupEventListeners();
        this.createNewSession();
    }

    // Pipeline step definitions
    setupPipelineSteps() {
        this.steps = [
            { id: 'input', name: '📥 Input & Datenquellen', description: 'Eingabe verarbeiten' },
            { id: 'initialisation', name: '🔍 Initialisierung', description: 'Freie Schlagworte generieren' },
            { id: 'search', name: '🔎 GND/SWB Suche', description: 'Katalogsuche durchführen' },
            { id: 'keywords', name: '📚 Verbale Erschließung', description: 'Finale GND-Schlagworte' },
            { id: 'classification', name: '📊 Klassifikation', description: 'DK/RVK-Codes zuweisen' }
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
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            this.appendStreamText(`[${this.getTime()}] Started analysis...`);

            // Connect WebSocket for live updates
            this.connectWebSocket();

        } catch (error) {
            console.error('Analysis error:', error);
            this.appendStreamText(`❌ Error: ${error.message}`);
            this.isAnalyzing = false;
            this.updateButtonState();
        }
    }

    // Connect WebSocket for live updates
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);

            if (msg.type === 'status') {
                this.updatePipelineStatus(msg);
            } else if (msg.type === 'complete') {
                this.handleAnalysisComplete(msg);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.appendStreamText(`⚠️ Connection error`);
        };

        this.ws.onclose = () => {
            console.log('WebSocket closed');
        };
    }

    // Update pipeline status from WebSocket
    updatePipelineStatus(msg) {
        if (msg.current_step) {
            this.updateStepStatus(msg.current_step, 'running');

            // Mark previous steps as completed
            const stepIndex = this.steps.findIndex(s => s.id === msg.current_step);
            for (let i = 0; i < stepIndex; i++) {
                this.updateStepStatus(this.steps[i].id, 'completed');
            }
        }

        // Update stream with results
        if (msg.results) {
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

    // Display results in stream
    displayResults(results) {
        if (!results) return;

        // Display keywords
        if (results.keywords && results.keywords.length > 0) {
            this.appendStreamText(`\n[${this.getTime()}] Keywords found:`);
            results.keywords.forEach(kw => {
                this.appendStreamText(`  • ${kw}`);
            });
        }

        // Display classifications
        if (results.dk_classification && results.dk_classification.length > 0) {
            this.appendStreamText(`\n[${this.getTime()}] Classifications:`);
            results.dk_classification.forEach(cls => {
                this.appendStreamText(`  📊 ${cls}`);
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
