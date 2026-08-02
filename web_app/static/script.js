document.addEventListener('DOMContentLoaded', () => {
    
    // --- Model Tab Selection Logic ---
    const modelTabs = document.querySelectorAll('.model-tab-card');
    modelTabs.forEach(tab => {
        tab.addEventListener('click', function() {
            modelTabs.forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            const radio = this.querySelector('input[type="radio"]');
            if (radio) radio.checked = true;
        });
    });

    // --- Image Analysis Page Logic ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('image-preview');
    const previewImg = document.getElementById('preview-img');
    const previewFilename = document.getElementById('preview-filename');
    const removeBtn = document.getElementById('remove-img');
    const analyzeBtn = document.getElementById('analyze-btn');
    const form = document.getElementById('image-upload-form');
    
    if (dropZone && fileInput) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
        });

        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            handleFiles(files);
        });

        dropZone.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', function() {
            handleFiles(this.files);
        });

        removeBtn.addEventListener('click', () => {
            fileInput.value = '';
            previewContainer.classList.add('hidden');
            dropZone.classList.remove('hidden');
            analyzeBtn.disabled = true;
            document.getElementById('result-container').classList.add('hidden');
        });

        function handleFiles(files) {
            if (files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('image/')) {
                    if (fileInput.files !== files) {
                        const dataTransfer = new DataTransfer();
                        dataTransfer.items.add(file);
                        fileInput.files = dataTransfer.files;
                    }
                    
                    if (previewFilename) {
                        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
                        previewFilename.textContent = `${file.name} (${sizeMB} MB)`;
                    }

                    const reader = new FileReader();
                    reader.readAsDataURL(file);
                    reader.onload = () => {
                        previewImg.src = reader.result;
                        previewContainer.classList.remove('hidden');
                        dropZone.classList.add('hidden');
                        analyzeBtn.disabled = false;
                    }
                }
            }
        }

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const btnText = analyzeBtn.querySelector('.btn-text');
            const spinner = analyzeBtn.querySelector('.spinner');
            const resultContainer = document.getElementById('result-container');
            const predictionText = document.getElementById('prediction-text');
            const confidenceBar = document.getElementById('confidence-bar');
            const confidenceText = document.getElementById('confidence-text');
            
            btnText.classList.add('hidden');
            spinner.classList.remove('hidden');
            analyzeBtn.disabled = true;
            resultContainer.classList.add('hidden');
            
            try {
                const formData = new FormData(form);
                const response = await fetch('http://localhost:8000/predict/image', {
                    method: 'POST',
                    body: formData
                });
                
                let data;
                try {
                    data = await response.json();
                } catch (jsonError) {
                    throw new Error(`Server returned non-JSON response (Status: ${response.status})`);
                }
                
                if (response.ok) {
                    predictionText.textContent = data.prediction;
                    const confidencePercent = (data.confidence * 100).toFixed(1);
                    confidenceText.textContent = `${confidencePercent}%`;
                    
                    resultContainer.classList.remove('hidden');
                    
                    void confidenceBar.offsetWidth;
                    confidenceBar.style.width = `${confidencePercent}%`;
                    
                    if (data.confidence < 0.6) {
                        confidenceBar.style.background = 'linear-gradient(90deg, #f59e0b, #ef4444)';
                        predictionText.style.color = '#f59e0b';
                    } else {
                        confidenceBar.style.background = 'linear-gradient(90deg, var(--primary), var(--secondary))';
                        predictionText.style.color = 'var(--secondary)';
                    }
                    
                    const tipsContainer = document.getElementById('tips-container');
                    renderTips(data.tips, tipsContainer);

                    // Scroll smoothly to results
                    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                } else {
                    alert(`Error: ${data.detail || 'Failed to analyze image'}`);
                }
            } catch (error) {
                alert(`Error connecting to server: ${error.message}`);
            } finally {
                btnText.classList.remove('hidden');
                spinner.classList.add('hidden');
                analyzeBtn.disabled = false;
            }
        });
    }

    // --- NLP Symptom Page Logic ---
    const nlpForm = document.getElementById('nlp-form');
    const symptomsInput = document.getElementById('symptoms-input');
    const charCounter = document.getElementById('char-counter');

    if (symptomsInput && charCounter) {
        symptomsInput.addEventListener('input', () => {
            const count = symptomsInput.value.length;
            charCounter.textContent = `${count} chars`;
        });
    }

    // Symptom Preset Chips
    const symptomChips = document.querySelectorAll('.symptom-chip');
    symptomChips.forEach(chip => {
        chip.addEventListener('click', function() {
            const symptomText = this.getAttribute('data-symptom');
            if (symptomsInput && symptomText) {
                symptomsInput.value = symptomText;
                symptomsInput.dispatchEvent(new Event('input'));
                symptomsInput.focus();
            }
        });
    });

    if (nlpForm) {
        const analyzeNlpBtn = document.getElementById('analyze-nlp-btn');
        
        nlpForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const textValue = symptomsInput.value;
            const btnText = analyzeNlpBtn.querySelector('.btn-text');
            const spinner = analyzeNlpBtn.querySelector('.spinner');
            const resultContainer = document.getElementById('nlp-result-container');
            const predictionText = document.getElementById('nlp-prediction-text');
            const confidenceBar = document.getElementById('nlp-confidence-bar');
            const confidenceText = document.getElementById('nlp-confidence-text');
            
            btnText.classList.add('hidden');
            spinner.classList.remove('hidden');
            analyzeNlpBtn.disabled = true;
            resultContainer.classList.add('hidden');
            
            try {
                const response = await fetch('http://localhost:8000/predict/nlp', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ symptoms: textValue })
                });
                
                let data;
                try {
                    data = await response.json();
                } catch (jsonError) {
                    throw new Error(`Server returned non-JSON response (Status: ${response.status})`);
                }
                
                if (response.ok) {
                    predictionText.textContent = data.prediction;
                    const confidencePercent = (data.confidence * 100).toFixed(1);
                    confidenceText.textContent = `${confidencePercent}%`;
                    
                    resultContainer.classList.remove('hidden');
                    
                    void confidenceBar.offsetWidth;
                    confidenceBar.style.width = `${confidencePercent}%`;

                    const nlpTipsContainer = document.getElementById('nlp-tips-container');
                    renderTips(data.tips, nlpTipsContainer);

                    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                } else {
                    alert(`Error: ${data.detail || 'Failed to analyze symptoms'}`);
                }
            } catch (error) {
                alert(`Error connecting to server: ${error.message}`);
            } finally {
                btnText.classList.remove('hidden');
                spinner.classList.add('hidden');
                analyzeNlpBtn.disabled = false;
            }
        });
    }

    // --- Copy Report Feature ---
    const copyReportBtn = document.getElementById('copy-report-btn');
    if (copyReportBtn) {
        copyReportBtn.addEventListener('click', () => {
            const pred = document.getElementById('prediction-text')?.textContent || '';
            const conf = document.getElementById('confidence-text')?.textContent || '';
            const reportText = `[Medbuddy Diagnostic Report]\nDiagnosis: ${pred}\nConfidence: ${conf}\nDate: ${new Date().toLocaleString()}`;
            copyToClipboard(reportText, copyReportBtn);
        });
    }

    const copyNlpReportBtn = document.getElementById('copy-nlp-report-btn');
    if (copyNlpReportBtn) {
        copyNlpReportBtn.addEventListener('click', () => {
            const pred = document.getElementById('nlp-prediction-text')?.textContent || '';
            const conf = document.getElementById('nlp-confidence-text')?.textContent || '';
            const reportText = `[Medbuddy Symptom Triage Assessment]\nPossible Condition: ${pred}\nConfidence: ${conf}\nDate: ${new Date().toLocaleString()}`;
            copyToClipboard(reportText, copyNlpReportBtn);
        });
    }

    function copyToClipboard(text, btnElement) {
        navigator.clipboard.writeText(text).then(() => {
            const originalHTML = btnElement.innerHTML;
            btnElement.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
            btnElement.classList.add('btn-copied');
            setTimeout(() => {
                btnElement.innerHTML = originalHTML;
                btnElement.classList.remove('btn-copied');
            }, 2000);
        }).catch(err => {
            alert('Failed to copy report: ' + err);
        });
    }

    // --- Helper function to render disease tips & advice ---
    function renderTips(tips, containerElement) {
        if (!tips || !containerElement) {
            if (containerElement) containerElement.classList.add('hidden');
            return;
        }

        let html = '<div class="tips-card-inner">';

        // Severity Estimate Banner
        if (tips.severity_estimate) {
            html += `
                <div class="tip-banner severity-banner">
                    <i class="fa-solid fa-triangle-exclamation tip-banner-icon"></i>
                    <div>
                        <strong class="tip-banner-title">Severity Estimate</strong>
                        <p class="tip-banner-desc">${escapeHtml(tips.severity_estimate)}</p>
                    </div>
                </div>`;
        }

        // Treatment & Care Suggestions
        if (tips.treatment_suggestions && tips.treatment_suggestions.length > 0) {
            html += `
                <div class="tip-section">
                    <h4 class="tip-section-title"><i class="fa-solid fa-notes-medical"></i> Treatment & Care Suggestions</h4>
                    <ul class="tip-list">
                        ${tips.treatment_suggestions.map(t => `<li><i class="fa-solid fa-check tip-icon-check"></i> <span>${escapeHtml(t)}</span></li>`).join('')}
                    </ul>
                </div>`;
        }

        // Prevention Tips
        if (tips.prevention_tips && tips.prevention_tips.length > 0) {
            html += `
                <div class="tip-section">
                    <h4 class="tip-section-title"><i class="fa-solid fa-shield-heart"></i> Prevention & Management</h4>
                    <ul class="tip-list">
                        ${tips.prevention_tips.map(t => `<li><i class="fa-solid fa-lightbulb tip-icon-bulb"></i> <span>${escapeHtml(t)}</span></li>`).join('')}
                    </ul>
                </div>`;
        }

        // Common Symptoms
        if (tips.common_symptoms && tips.common_symptoms.length > 0) {
            html += `
                <div class="tip-section">
                    <h4 class="tip-section-title"><i class="fa-solid fa-stethoscope"></i> Common Associated Symptoms</h4>
                    <ul class="tip-list">
                        ${tips.common_symptoms.map(s => `<li><i class="fa-solid fa-angle-right tip-icon-arrow"></i> <span>${escapeHtml(s)}</span></li>`).join('')}
                    </ul>
                </div>`;
        }

        // When to see a doctor
        if (tips.when_to_see_a_doctor) {
            html += `
                <div class="tip-banner doctor-banner">
                    <i class="fa-solid fa-user-doctor tip-banner-icon"></i>
                    <div>
                        <strong class="tip-banner-title">When to See a Doctor</strong>
                        <p class="tip-banner-desc">${escapeHtml(tips.when_to_see_a_doctor)}</p>
                    </div>
                </div>`;
        }

        html += '</div>';
        containerElement.innerHTML = html;
        containerElement.classList.remove('hidden');
    }

    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }
});
