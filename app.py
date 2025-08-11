import os
import re
import difflib
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from flask import Flask, request, jsonify, render_template_string
from datetime import datetime, timedelta
import hashlib

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
app = Flask(__name__)

# --- Enums and Data Structures ---
class Priority(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

@dataclass
class SubmissionRecord:
    text: str
    category: str
    priority: str
    timestamp: datetime
    submission_hash: str
    is_escalated: bool = False

@dataclass
class ClassificationResult:
    category: str
    priority: str
    confidence: float
    matched_keywords: Dict[str, List[str]] = field(default_factory=dict)
    priority_factors: List[str] = field(default_factory=list)
    is_duplicate: bool = False
    escalation_applied: bool = False

# --- Knowledge Base and Constants ---
IMPACT_LEVELS = {"Minimal": Priority.LOW, "Moderate": Priority.MEDIUM, "Significant": Priority.HIGH, "Critical": Priority.CRITICAL}
MAX_TEXT_LENGTH = 5000

KNOWLEDGE_BASE = {
    "Safety Concern": {
        "critical": {"emergency", "fire", "explosion", "fatal"},
        "high": {"danger", "hazard", "unsafe", "injury", "accident"},
        "medium": {"safety", "risk", "warning", "slippery"},
        "negation": {"not", "no", "without", "lacking"},
    },
    "Machine/Equipment Issue": {
        "critical": {"complete failure", "shutdown", "catastrophic", "unusable"},
        "high": {"broken", "malfunction", "down", "stopped", "leaking"},
        "medium": {"noise", "vibration", "loose", "maintenance"},
        "negation": {"not", "no", "without"},
    },
    "Process Improvement Idea": {
        "high": {"automate", "streamline", "optimize"},
        "medium": {"improve", "efficiency", "better", "process"},
        "negation": [],
    },
    "Other": {
        "medium": {"supplies", "lighting", "parking", "temperature"},
        "negation": [],
    }
}

# --- Core Logic Functions ---
class ClassifierLogic:
    def __init__(self):
        self.submissions = []
        self.retention_hours = 168
        self.similarity_threshold = 0.6
        self.escalation_threshold = 2

    def _normalize_text(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def _calculate_scores(self, tokens: List[str]) -> Tuple[Dict, Dict]:
        category_scores, matched_keywords = defaultdict(float), defaultdict(lambda: defaultdict(list))
        for cat, data in KNOWLEDGE_BASE.items():
            for i, token in enumerate(tokens):
                score = 0
                level = None
                if token in data["critical"]: score, level = 3.0, "critical"
                elif token in data["high"]: score, level = 2.0, "high"
                elif token in data["medium"]: score, level = 1.0, "medium"

                if score > 0:
                    is_negated = any(t in data["negation"] for t in tokens[max(0, i-3):i+4])
                    if not is_negated:
                        category_scores[cat] += score
                        matched_keywords[cat][level].append(token)
        return dict(category_scores), dict(matched_keywords)

    def _determine_priority(self, scores: Dict, text: str) -> Tuple[Priority, List[str]]:
        factors = []
        text_lower = text.lower()
        
        explicit_level = None
        match = re.search(r"impact level: (\w+)", text_lower)
        if match and match.group(1).capitalize() in IMPACT_LEVELS:
            explicit_level = match.group(1).capitalize()
            factors.append(f"Explicit Impact Level: {explicit_level}")

        critical_score = scores.get("Safety Concern", 0) + scores.get("Machine/Equipment Issue", 0)
        
        if explicit_level == "Critical" or critical_score >= 3.0:
            if critical_score >= 3.0: factors.append("Direct Critical indicators detected.")
            return Priority.CRITICAL, factors
        elif explicit_level == "Significant" or critical_score >= 2.0:
            if critical_score >= 2.0: factors.append("High severity indicators.")
            return Priority.HIGH, factors
        elif explicit_level == "Moderate" or any(s > 0 for cat, s in scores.items() if cat != "Other"):
            if explicit_level == "Moderate": factors.append("Explicit Impact Level: Moderate")
            return Priority.MEDIUM, factors
        else:
            factors.append("Low severity or general suggestion.")
            return Priority.LOW, factors

    def _analyze_duplicates(self, text: str, category: str, priority: str) -> Tuple[bool, bool, int, str]:
        self.submissions = [s for s in self.submissions if s.timestamp > datetime.now() - timedelta(hours=self.retention_hours)]
        
        new_hash = hashlib.md5(text.lower().encode()).hexdigest()
        similar_count = 0
        
        for s in self.submissions:
            is_match = (s.submission_hash == new_hash) or \
                       (difflib.SequenceMatcher(None, text.lower(), s.text.lower()).ratio() > self.similarity_threshold)
            if is_match and s.category == category:
                similar_count += 1

        is_duplicate = similar_count > 0
        escalation = False
        final_priority = priority
        
        if priority in (Priority.LOW.value, Priority.MEDIUM.value) and similar_count >= self.escalation_threshold:
            escalation = True
            final_priority = Priority.CRITICAL.value
            
        self.submissions.append(SubmissionRecord(text=text, category=category, priority=final_priority, timestamp=datetime.now(), submission_hash=new_hash, is_escalated=escalation))
        return is_duplicate, escalation, similar_count, final_priority

    def classify_and_process(self, text: str) -> ClassificationResult:
        if not text or len(text) > MAX_TEXT_LENGTH:
            return ClassificationResult("Other", Priority.LOW.value, 0.1, {}, ["Invalid input"])
        
        tokens = self._normalize_text(text)
        scores, keywords = self._calculate_scores(tokens)
        
        best_category = max(scores, key=scores.get, default="Other")
        if scores.get(best_category, 0) == 0:
            best_category = "Other"
            
        initial_priority, factors = self._determine_priority(scores, text)
        is_duplicate, escalation, similar_count, final_priority = self._analyze_duplicates(text, best_category, initial_priority.value)
        
        if escalation:
            factors.append(f"ESCALATED TO CRITICAL: {similar_count} similar reports found.")
        
        confidence = 0.5 + (scores.get(best_category, 0) / (sum(scores.values()) + 1) * 0.5)
        
        return ClassificationResult(best_category, final_priority, round(confidence, 3), keywords.get(best_category, {}), factors, is_duplicate, escalation)

classifier_logic = ClassifierLogic()

# --- Flask Routes and UI ---
@app.route("/", methods=["GET"])
def home():
    html_content = """
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
            <form id="classificationForm" onsubmit="return false;">
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
                </div>
        </div>
    </div>

    <script>
        const API_BASE_URL = ''; // Now relative to the server
        
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
                showError(`Failed to classify feedback: ${error.message}. Make sure the Flask server is running on the correct port.`);
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
            
            if (result.escalation_applied) {
                escalationAlert.style.display = 'flex';
                escalationMessage.innerHTML = `
                    This issue has been escalated to <strong>CRITICAL</strong> priority due to 
                    a duplicate submission.
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
"""
    return render_template_string(html_content)

@app.route("/classify", methods=["POST"])
def classify_route():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "Missing 'text' in request body"}), 400
        
        result = classifier_logic.classify_and_process(text)
        
        # Prepare the response with data the front end expects
        response_data = {
            "category": result.category,
            "priority": result.priority,
            "confidence": result.confidence,
            "matched_keywords": result.matched_keywords,
            "priority_factors": result.priority_factors,
            "is_duplicate": result.is_duplicate,
            "escalation_applied": result.escalation_applied,
        }
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error during classification: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/duplicate_stats", methods=["GET"])
def stats_route():
    try:
        # Calculate stats from the in-memory list
        total_submissions = len(classifier_logic.submissions)
        escalated_count = sum(1 for s in classifier_logic.submissions if s.is_escalated)
        
        stats = {
            "total_submissions_retained": total_submissions,
            "escalated_critical_in_memory": escalated_count,
            "retention_hours": classifier_logic.retention_hours,
            "similarity_threshold": classifier_logic.similarity_threshold,
            "escalation_threshold": classifier_logic.escalation_threshold
        }
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error generating stats: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
