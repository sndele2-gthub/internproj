<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Feedback Classification System</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary-color: #2563eb;
            --primary-hover: #1d4ed8;
            --secondary-color: #f8fafc;
            --accent-color: #0ea5e9;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
            --critical-color: #dc2626;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --border-color: #e2e8f0;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
            --gradient-bg: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--gradient-bg);
            min-height: 100vh;
            line-height: 1.6;
            color: var(--text-primary);
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }

        .header {
            text-align: center;
            margin-bottom: 3rem;
            color: white;
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .main-card {
            background: white;
            border-radius: 16px;
            padding: 2rem;
            box-shadow: var(--shadow-lg);
            margin-bottom: 2rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .form-group {
            margin-bottom: 1.5rem;
        }

        label {
            display: block;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: var(--text-primary);
        }

        .textarea-container {
            position: relative;
        }

        textarea {
            width: 100%;
            min-height: 150px;
            padding: 1rem;
            border: 2px solid var(--border-color);
            border-radius: 12px;
            font-size: 1rem;
            transition: all 0.3s ease;
            resize: vertical;
            background: var(--secondary-color);
        }

        textarea:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
            background: white;
        }

        .char-counter {
            position: absolute;
            bottom: 0.5rem;
            right: 1rem;
            font-size: 0.875rem;
            color: var(--text-secondary);
            background: rgba(255, 255, 255, 0.9);
            padding: 0.25rem 0.5rem;
            border-radius: 6px;
        }

        .button-group {
            display: flex;
            gap: 1rem;
            justify-content: center;
            flex-wrap: wrap;
        }

        .btn {
            padding: 0.875rem 2rem;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            text-decoration: none;
            position: relative;
            overflow: hidden;
        }

        .btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }

        .btn:hover::before {
            left: 100%;
        }

        .btn-primary {
            background: var(--primary-color);
            color: white;
        }

        .btn-primary:hover {
            background: var(--primary-hover);
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }

        .btn-secondary {
            background: var(--secondary-color);
            color: var(--text-primary);
            border: 2px solid var(--border-color);
        }

        .btn-secondary:hover {
            background: white;
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .loading {
            display: inline-block;
            width: 1rem;
            height: 1rem;
            border: 2px solid transparent;
            border-top: 2px solid currentColor;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .results-container {
            display: none;
            animation: slideUp 0.5s ease-out;
        }

        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .result-card {
            background: var(--secondary-color);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid var(--border-color);
            transition: all 0.3s ease;
        }

        .result-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }

        .result-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }

        .result-icon {
            width: 2.5rem;
            height: 2.5rem;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            color: white;
        }

        .category-icon {
            background: var(--accent-color);
        }

        .priority-icon.Low {
            background: var(--success-color);
        }

        .priority-icon.Medium {
            background: var(--warning-color);
        }

        .priority-icon.High {
            background: var(--error-color);
        }

        .priority-icon.Critical {
            background: var(--critical-color);
        }

        .result-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .result-value {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .confidence-bar {
            background: var(--border-color);
            height: 6px;
            border-radius: 3px;
            overflow: hidden;
            margin-bottom: 0.5rem;
        }

        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--error-color) 0%, var(--warning-color) 50%, var(--success-color) 100%);
            transition: width 0.8s ease;
            border-radius: 3px;
        }

        .details-section {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 1.5rem;
        }

        .details-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .keyword-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .tag {
            background: var(--primary-color);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 500;
        }

        .tag.critical {
            background: var(--critical-color);
        }

        .tag.high {
            background: var(--error-color);
        }

        .tag.medium {
            background: var(--warning-color);
        }

        .factors-list {
            list-style: none;
        }

        .factors-list li {
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .factors-list li:last-child {
            border-bottom: none;
        }

        .escalation-alert {
            background: linear-gradient(135deg, #fee2e2, #fecaca);
            border: 2px solid var(--critical-color);
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .escalation-alert .icon {
            color: var(--critical-color);
            font-size: 1.5rem;
        }

        .stats-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 16px;
            padding: 2rem;
            color: white;
            margin-top: 2rem;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-top: 1rem;
        }

        .stat-item {
            text-align: center;
        }

        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .stat-label {
            opacity: 0.8;
            font-size: 0.9rem;
        }

        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }

            .header h1 {
                font-size: 2rem;
            }

            .main-card {
                padding: 1.5rem;
            }

            .button-group {
                flex-direction: column;
            }

            .btn {
                width: 100%;
                justify-content: center;
            }

            .results-grid {
                grid-template-columns: 1fr;
            }
        }

        .error-message {
            background: #fee2e2;
            color: #dc2626;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1><i class="fas fa-brain"></i> Enhanced Feedback Classification</h1>
            <p>AI-powered feedback analysis with intelligent priority detection and duplicate escalation</p>
        </header>

        <div class="main-card">
            <form id="classificationForm">
                <div class="form-group">
                    <label for="feedbackText">
                        <i class="fas fa-comment-alt"></i> Enter your feedback or concern
                    </label>
                    <div class="textarea-container">
                        <textarea 
                            id="feedbackText" 
                            name="text" 
                            placeholder="Describe your safety concern, equipment issue, improvement idea, or other feedback..."
                            maxlength="5000"
                            required
                        ></textarea>
                        <div class="char-counter">
                            <span id="charCount">0</span>/5000
                        </div>
                    </div>
                </div>

                <div class="button-group">
                    <button type="submit" class="btn btn-primary" id="classifyBtn">
                        <i class="fas fa-search"></i>
                        <span>Classify Feedback</span>
                    </button>
                    <button type="button" class="btn btn-secondary" id="clearBtn">
                        <i class="fas fa-eraser"></i>
                        Clear Form
                    </button>
                    <button type="button" class="btn btn-secondary" id="statsBtn">
                        <i class="fas fa-chart-bar"></i>
                        View Stats
                    </button>
                </div>
            </form>

            <div id="errorContainer"></div>
        </div>

        <div id="resultsContainer" class="results-container">
            <div class="main-card">
                <div id="escalationAlert" class="escalation-alert" style="display: none;">
                    <i class="fas fa-exclamation-triangle icon"></i>
                    <div>
                        <strong>Priority Escalated!</strong>
                        <div id="escalationMessage"></div>
                    </div>
                </div>

                <div class="results-grid">
                    <div class="result-card">
                        <div class="result-header">
                            <div class="result-icon category-icon">
                                <i class="fas fa-tag"></i>
                            </div>
                            <div>
                                <div class="result-title">Category</div>
                                <div class="result-value" id="categoryResult">-</div>
                            </div>
                        </div>
                    </div>

                    <div class="result-card">
                        <div class="result-header">
                            <div class="result-icon priority-icon" id="priorityIcon">
                                <i class="fas fa-flag"></i>
                            </div>
                            <div>
                                <div class="result-title">Priority</div>
                                <div class="result-value" id="priorityResult">-</div>
                            </div>
                        </div>
                    </div>

                    <div class="result-card">
                        <div class="result-header">
                            <div class="result-icon" style="background: var(--accent-color);">
                                <i class="fas fa-percentage"></i>
                            </div>
                            <div style="flex: 1;">
                                <div class="result-title">Confidence</div>
                                <div class="result-value" id="confidenceResult">-</div>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" id="confidenceFill" style="width: 0%"></div>
                                </div>
                                <small class="text-secondary" id="confidenceLabel">Classification accuracy</small>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="details-section">
                    <div class="details-title">
                        <i class="fas fa-info-circle"></i>
                        Analysis Details
                    </div>
                    
                    <div id="keywordsSection" style="margin-bottom: 1.5rem;">
                        <h4 style="margin-bottom: 0.5rem;">Detected Keywords</h4>
                        <div id="keywordTags" class="keyword-tags"></div>
                    </div>

                    <div id="factorsSection">
                        <h4 style="margin-bottom: 0.5rem;">Priority Factors</h4>
                        <ul id="factorsList" class="factors-list"></ul>
                    </div>
                </div>
            </div>
        </div>

        <div id="statsContainer" class="stats-card" style="display: none;">
            <h3><i class="fas fa-chart-line"></i> System Statistics</h3>
            <div class="stats-grid" id="statsGrid">
                <!-- Stats will be populated here -->
            </div>
        </div>
    </div>

    <script>
        const API_BASE_URL = 'http://localhost:5000'; // Update this to match your Flask server
        
        // DOM elements
        const form = document.getElementById('classificationForm');
        const textarea = document.getElementById('feedbackText');
        const charCount = document.getElementById('charCount');
        const classifyBtn = document.getElementById('classifyBtn');
        const clearBtn = document.getElementById('clearBtn');
        const statsBtn = document.getElementById('statsBtn');
        const resultsContainer = document.getElementById('resultsContainer');
        const errorContainer = document.getElementById('errorContainer');
        const statsContainer = document.getElementById('statsContainer');

        // Character counter
        textarea.addEventListener('input', function() {
            const count = this.value.length;
            charCount.textContent = count;
            
            if (count > 4500) {
                charCount.style.color = 'var(--error-color)';
            } else if (count > 4000) {
                charCount.style.color = 'var(--warning-color)';
            } else {
                charCount.style.color = 'var(--text-secondary)';
            }
        });

        // Form submission
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            await classifyFeedback();
        });

        // Clear button
        clearBtn.addEventListener('click', function() {
            textarea.value = '';
            charCount.textContent = '0';
            charCount.style.color = 'var(--text-secondary)';
            resultsContainer.style.display = 'none';
            errorContainer.innerHTML = '';
        });

        // Stats button
        statsBtn.addEventListener('click', async function() {
            await loadStats();
        });

        async function classifyFeedback() {
            const text = textarea.value.trim();
            if (!text) return;

            // Show loading state
            const originalContent = classifyBtn.innerHTML;
            classifyBtn.innerHTML = '<div class="loading"></div> Analyzing...';
            classifyBtn.disabled = true;
            errorContainer.innerHTML = '';

            try {
                const response = await fetch(`${API_BASE_URL}/classify`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const result = await response.json();
                displayResults(result);
                
            } catch (error) {
                showError(`Failed to classify feedback: ${error.message}. Make sure the Flask server is running on ${API_BASE_URL}`);
            } finally {
                classifyBtn.innerHTML = originalContent;
                classifyBtn.disabled = false;
            }
        }

        function displayResults(result) {
            // Update category
            document.getElementById('categoryResult').textContent = result.category;

            // Update priority with icon
            const priorityIcon = document.getElementById('priorityIcon');
            priorityIcon.className = `result-icon priority-icon ${result.priority}`;
            document.getElementById('priorityResult').textContent = result.priority;

            // Update confidence
            const confidence = Math.round(result.confidence * 100);
            document.getElementById('confidenceResult').textContent = `${confidence}%`;
            document.getElementById('confidenceFill').style.width = `${confidence}%`;
            
            // Update confidence label
            let confidenceLabel = 'Low accuracy';
            if (confidence >= 80) confidenceLabel = 'High accuracy';
            else if (confidence >= 60) confidenceLabel = 'Good accuracy';
            else if (confidence >= 40) confidenceLabel = 'Moderate accuracy';
            document.getElementById('confidenceLabel').textContent = confidenceLabel;

            // Handle escalation alert
            const escalationAlert = document.getElementById('escalationAlert');
            const escalationMessage = document.getElementById('escalationMessage');
            
            if (result.duplicate_analysis && result.duplicate_analysis.escalation_applied) {
                escalationAlert.style.display = 'flex';
                escalationMessage.innerHTML = `
                    This issue has been escalated to <strong>CRITICAL</strong> priority due to 
                    ${result.duplicate_analysis.similar_count} similar submissions detected.
                    <br><small>Original priority: ${result.duplicate_analysis.original_priority}</small>
                `;
            } else {
                escalationAlert.style.display = 'none';
            }

            // Display keywords
            const keywordTags = document.getElementById('keywordTags');
            keywordTags.innerHTML = '';
            
            if (result.matched_keywords) {
                ['critical', 'high', 'medium'].forEach(level => {
                    if (result.matched_keywords[level] && result.matched_keywords[level].length > 0) {
                        result.matched_keywords[level].forEach(keyword => {
                            const tag = document.createElement('span');
                            tag.className = `tag ${level}`;
                            tag.textContent = keyword;
                            keywordTags.appendChild(tag);
                        });
                    }
                });
            }

            // Display priority factors
            const factorsList = document.getElementById('factorsList');
            factorsList.innerHTML = '';
            
            if (result.priority_factors) {
                result.priority_factors.forEach(factor => {
                    const li = document.createElement('li');
                    li.innerHTML = `<i class="fas fa-check-circle" style="color: var(--success-color);"></i> ${factor}`;
                    factorsList.appendChild(li);
                });
            }

            // Show results
            resultsContainer.style.display = 'block';
            resultsContainer.scrollIntoView({ behavior: 'smooth' });
        }

        async function loadStats() {
            try {
                const response = await fetch(`${API_BASE_URL}/duplicate_stats`);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const stats = await response.json();
                displayStats(stats);
                
            } catch (error) {
                showError(`Failed to load stats: ${error.message}`);
            }
        }

        function displayStats(stats) {
            const statsGrid = document.getElementById('statsGrid');
            statsGrid.innerHTML = `
                <div class="stat-item">
                    <div class="stat-value">${stats.total_submissions_retained}</div>
                    <div class="stat-label">Total Submissions</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${stats.escalated_critical_in_memory}</div>
                    <div class="stat-label">Escalated to Critical</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${stats.retention_hours}h</div>
                    <div class="stat-label">Retention Period</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${Math.round(stats.similarity_threshold * 100)}%</div>
                    <div class="stat-label">Similarity Threshold</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${stats.escalation_threshold}</div>
                    <div class="stat-label">Escalation Threshold</div>
                </div>
            `;

            statsContainer.style.display = 'block';
            statsContainer.scrollIntoView({ behavior: 'smooth' });
        }

        function showError(message) {
            errorContainer.innerHTML = `
                <div class="error-message">
                    <i class="fas fa-exclamation-circle"></i>
                    ${message}
                </div>
            `;
        }

        // Auto-focus textarea on load
        document.addEventListener('DOMContentLoaded', function() {
            textarea.focus();
        });
    </script>
</body>
</html>
