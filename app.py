import re
import difflib
import logging
from collections import defaultdict
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from flask import Flask, request, jsonify, render_template_string
from datetime import datetime, timedelta
import hashlib

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

# --- Data Models ---
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
    original_priority: str = "N/A"
    similar_count: int = 0

# --- Knowledge Base & Constants ---
MAX_TEXT_LENGTH = 5000
KNOWLEDGE_BASE = {
    "Safety Concern": { "critical": {"emergency", "fire", "explosion", "fatal"}, "high": {"danger", "hazard", "unsafe", "injury", "accident"}, "medium": {"safety", "risk", "warning", "slippery"}, "negation": {"not", "no", "without", "lacking"}, },
    "Machine/Equipment Issue": { "critical": {"complete failure", "shutdown", "catastrophic", "unusable"}, "high": {"broken", "malfunction", "down", "stopped", "leaking"}, "medium": {"noise", "vibration", "loose", "maintenance"}, "negation": {"not", "no", "without"}, },
    "Process Improvement Idea": { "high": {"automate", "streamline", "optimize"}, "medium": {"improve", "efficiency", "better", "process"}, "negation": [], },
    "Other": { "medium": {"supplies", "lighting", "parking", "temperature"}, "negation": [], }
}

# --- Core Logic ---
class ClassifierLogic:
    def __init__(self):
        self.submissions: List[SubmissionRecord] = []
        self.retention_hours = 168
        self.similarity_threshold = 0.6
        self.escalation_threshold = 2

    def _normalize_text(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def _calculate_scores(self, tokens: List[str]) -> Tuple[Dict, Dict]:
        cat_scores, matched_keys = defaultdict(float), defaultdict(lambda: defaultdict(list))
        for cat, data in KNOWLEDGE_BASE.items():
            for i, token in enumerate(tokens):
                score = 0
                level = None
                if token in data["critical"]: score, level = 3.0, "critical"
                elif token in data["high"]: score, level = 2.0, "high"
                elif token in data["medium"]: score, level = 1.0, "medium"
                if score > 0 and not any(t in data["negation"] for t in tokens[max(0, i-3):i+4]):
                    cat_scores[cat] += score
                    matched_keys[cat][level].append(token)
        return dict(cat_scores), dict(matched_keys)

    def _determine_priority(self, scores: Dict, text: str) -> Tuple[Priority, List[str]]:
        factors, text_lower = [], text.lower()
        explicit_level_match = re.search(r"impact level:\s*(minimal|moderate|significant|critical)", text_lower)
        explicit_level = explicit_level_match.group(1).capitalize() if explicit_level_match else None

        if explicit_level and explicit_level in ["Minimal", "Moderate", "Significant", "Critical"]:
            factors.append(f"Explicit Impact Level: {explicit_level}")

        critical_score = scores.get("Safety Concern", 0) + scores.get("Machine/Equipment Issue", 0)
        
        if explicit_level == "Critical" or critical_score >= 3.0:
            return Priority.CRITICAL, factors + (["Direct Critical indicators detected."] if critical_score >= 3.0 else [])
        elif explicit_level == "Significant" or critical_score >= 2.0:
            return Priority.HIGH, factors + (["High severity indicators."] if critical_score >= 2.0 else [])
        elif explicit_level == "Moderate" or any(s > 0 for cat, s in scores.items() if cat != "Other"):
            return Priority.MEDIUM, factors + (["Explicit Impact Level: Moderate"] if explicit_level == "Moderate" else [])
        else:
            return Priority.LOW, factors + ["Low severity or general suggestion."]

    def _analyze_duplicates(self, text: str, category: str, priority: str) -> Tuple[bool, bool, int, str, str]:
        self.submissions = [s for s in self.submissions if s.timestamp > datetime.now() - timedelta(hours=self.retention_hours)]
        new_hash, similar_count = hashlib.md5(text.lower().encode()).hexdigest(), 0
        
        for s in self.submissions:
            is_match = (s.submission_hash == new_hash) or \
                       (difflib.SequenceMatcher(None, text.lower(), s.text.lower()).ratio() > self.similarity_threshold and s.category == category)
            if is_match:
                similar_count += 1
        
        is_dup, escalated = similar_count > 0, False
        final_prio, original_prio = priority, priority
        
        if original_prio in (Priority.LOW.value, Priority.MEDIUM.value) and similar_count >= self.escalation_threshold:
            escalated, final_prio = True, Priority.CRITICAL.value
        elif original_prio == Priority.CRITICAL.value and similar_count > 0:
            escalated = True
        
        self.submissions.append(SubmissionRecord(text=text, category=category, priority=final_prio, timestamp=datetime.now(), submission_hash=new_hash, is_escalated=escalated))
        return is_dup, escalated, similar_count, final_prio, original_prio

    def classify_and_process(self, text: str) -> ClassificationResult:
        if not text or len(text) > MAX_TEXT_LENGTH:
            return ClassificationResult("Other", Priority.LOW.value, 0.1, {}, ["Invalid input"])
        
        tokens = self._normalize_text(text)
        scores, keywords = self._calculate_scores(tokens)
        
        best_category = max(scores, key=scores.get, default="Other")
        if scores.get(best_category, 0) == 0:
            best_category = "Other"
            
        initial_priority, factors = self._determine_priority(scores, text)
        is_dup, escalated, similar_count, final_prio, original_prio = self._analyze_duplicates(text, best_category, initial_priority.value)
        
        if escalated:
            factors.append(f"System-determined priority: {original_prio} -> {final_prio} (Escalated)")
            if similar_count > 0:
                factors.append(f"Reason: {similar_count} similar reports found.")
        
        confidence = 0.5 + (scores.get(best_category, 0) / (sum(scores.values()) + 1) * 0.5)
        
        return ClassificationResult(best_category, final_prio, round(confidence, 3), keywords.get(best_category, {}), factors, is_dup, escalated, original_prio, similar_count)

classifier_logic = ClassifierLogic()

# --- Flask Routes ---
@app.route("/", methods=["GET"])
def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Feedback Classifier</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            body { 
                font-family: 'Inter', sans-serif; 
                margin: 0; 
                padding: 40px; 
                background-color: #f0f2f5; 
                color: #333; 
                display: flex; 
                flex-direction: column; 
                align-items: center; 
                min-height: 100vh;
            }
            .container { 
                max-width: 600px; 
                width: 100%;
                background-color: #ffffff; 
                padding: 30px; 
                border-radius: 10px; 
                box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
                text-align: center; 
                box-sizing: border-box; /* Ensures padding doesn't increase total width */
            }
            h1 { 
                color: #2c3e50; 
                margin-bottom: 25px; 
                font-size: 2em; 
                font-weight: 700;
            }
            textarea { 
                width: 100%; 
                height: 120px; 
                margin-bottom: 20px; 
                padding: 15px; 
                border: 1px solid #e0e0e0; 
                border-radius: 8px; 
                font-size: 1em; 
                resize: vertical; 
                box-sizing: border-box;
                font-family: 'Inter', sans-serif;
            }
            button { 
                padding: 12px 25px; 
                margin: 5px; 
                border: none; 
                border-radius: 8px; 
                cursor: pointer; 
                background-color: #3498db; 
                color: white; 
                font-size: 1.1em; 
                font-weight: 600;
                transition: background-color 0.3s ease;
            }
            button:hover { 
                background-color: #2980b9; 
            }
            .results { 
                margin-top: 30px; 
                text-align: left; 
                padding-top: 20px;
                border-top: 1px solid #e0e0e0;
            }
            .results div { 
                margin-bottom: 10px; 
                font-size: 1em;
            }
            .results strong {
                color: #555;
            }
            .priority-critical { color: #e74c3c; font-weight: bold; } /* Red */
            .priority-high { color: #f39c12; font-weight: bold; }    /* Orange */
            .priority-medium { color: #3498db; font-weight: bold; }  /* Blue */
            .priority-low { color: #2ecc77; font-weight: bold; }     /* Green */
            .error-message { 
                color: #e74c3c; 
                margin-top: 15px; 
                font-weight: 600;
                background-color: #ffe0e0;
                padding: 10px;
                border-radius: 5px;
            }
            .loading { 
                display: inline-block; 
                width: 15px; 
                height: 15px; 
                border: 2px solid rgba(255,255,255,.5); 
                border-top-color: #fff; 
                border-radius: 50%; 
                animation: spin 1s linear infinite; 
                vertical-align: middle;
                margin-left: 8px;
            }
            @keyframes spin { 
                to { transform: rotate(360deg); } 
            }
            #escalationMessage {
                margin-top: 10px;
                padding: 10px;
                background-color: #fff3cd; /* Light warning yellow */
                color: #856404;
                border: 1px solid #ffeeba;
                border-radius: 5px;
                font-weight: bold;
            }

            /* Responsive adjustments */
            @media (max-width: 600px) {
                body {
                    padding: 20px;
                }
                .container {
                    padding: 20px;
                }
                h1 {
                    font-size: 1.8em;
                }
                button {
                    width: calc(100% - 10px); /* Account for margin */
                    margin: 5px 0;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Feedback Classifier</h1>
            <textarea id="feedbackText" placeholder="Describe the issue or suggestion..."></textarea>
            <button id="classifyBtn">Classify</button>
            <button id="clearBtn">Clear</button>
            <div id="errorContainer" class="error-message" style="display:none;"></div>
            <div id="resultsContainer" class="results" style="display:none;">
                <h2>Classification Results:</h2>
                <div><strong>Category:</strong> <span id="categoryResult"></span></div>
                <div><strong>Priority:</strong> <span id="priorityResult"></span></div>
                <div><strong>Confidence:</strong> <span id="confidenceResult"></span></div>
                <div><strong>Factors:</strong> <ul id="factorsList"></ul></div>
                <div><strong>Duplicate Status:</strong> <span id="duplicateStatus"></span></div>
                <div id="escalationMessage" style="display:none;"></div>
            </div>
        </div>

        <script>
            const API_BASE_URL = ''; // Relative URL for deployment
            const feedbackText = document.getElementById('feedbackText');
            const classifyBtn = document.getElementById('classifyBtn');
            const clearBtn = document.getElementById('clearBtn');
            const resultsContainer = document.getElementById('resultsContainer');
            const errorContainer = document.getElementById('errorContainer');
            const categoryResult = document.getElementById('categoryResult');
            const priorityResult = document.getElementById('priorityResult');
            const confidenceResult = document.getElementById('confidenceResult');
            const factorsList = document.getElementById('factorsList');
            const duplicateStatus = document.getElementById('duplicateStatus');
            const escalationMessage = document.getElementById('escalationMessage');

            // --- UI Interaction Logic ---
            classifyBtn.addEventListener('click', async () => {
                const text = feedbackText.value.trim();
                if (!text) {
                    errorContainer.textContent = "Please enter some feedback.";
                    errorContainer.style.display = 'block';
                    return;
                }
                errorContainer.style.display = 'none';
                resultsContainer.style.display = 'none';
                escalationMessage.style.display = 'none'; // Hide escalation message on new classification
                classifyBtn.disabled = true;
                classifyBtn.innerHTML = 'Classifying... <div class="loading"></div>';

                try {
                    const response = await fetch(`${API_BASE_URL}/classify`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: text })
                    });

                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
                    }

                    const result = await response.json();
                    
                    categoryResult.textContent = result.category;
                    priorityResult.textContent = result.priority;
                    priorityResult.className = `priority-${result.priority.toLowerCase()}`;
                    confidenceResult.textContent = `${Math.round(result.confidence * 100)}%`;
                    
                    factorsList.innerHTML = '';
                    if (result.priority_factors && result.priority_factors.length > 0) {
                        result.priority_factors.forEach(factor => {
                            const li = document.createElement('li');
                            li.textContent = factor;
                            factorsList.appendChild(li);
                        });
                    } else {
                        const li = document.createElement('li');
                        li.textContent = "No specific factors identified.";
                        factorsList.appendChild(li);
                    }

                    let dupMsg = result.is_duplicate ? "Yes" : "No";
                    duplicateStatus.textContent = dupMsg;

                    if (result.escalation_applied) {
                        escalationMessage.textContent = `Priority was escalated from ${result.original_priority} to ${result.priority} due to ${result.similar_count} similar report(s).`;
                        escalationMessage.style.display = 'block';
                    } else {
                        escalationMessage.style.display = 'none';
                    }
                    
                    resultsContainer.style.display = 'block';

                } catch (error) {
                    errorContainer.textContent = `Error: ${error.message}`;
                    errorContainer.style.display = 'block';
                } finally {
                    classifyBtn.disabled = false;
                    classifyBtn.innerHTML = 'Classify';
                }
            });

            clearBtn.addEventListener('click', () => {
                feedbackText.value = '';
                resultsContainer.style.display = 'none';
                errorContainer.style.display = 'none';
                escalationMessage.style.display = 'none';
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
            logging.warning("Missing 'text' in request body for /classify endpoint.")
            return jsonify({"error": "Missing 'text' in request body"}), 400
        
        result = classifier_logic.classify_and_process(text)
        
        formatted_keywords = {
            "critical": result.matched_keywords.get("critical", []),
            "high": result.matched_keywords.get("high", []),
            "medium": result.matched_keywords.get("medium", [])
        }

        response_data = {
            "category": result.category,
            "priority": result.priority,
            "confidence": result.confidence,
            "matched_keywords": formatted_keywords,
            "priority_factors": result.priority_factors,
            "is_duplicate": result.is_duplicate,
            "escalation_applied": result.escalation_applied,
            "original_priority": result.original_priority,
            "similar_count": result.similar_count
        }
        
        return jsonify(response_data)
    except Exception as e:
        logging.error(f"Server-side error during classification: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred during classification."}), 500

# This endpoint is no longer directly used by the minimalistic UI, but kept for API completeness
@app.route("/stats", methods=["GET"]) 
def stats_route():
    try:
        classifier_logic.submissions = [s for s in classifier_logic.submissions if s.timestamp > datetime.now() - timedelta(hours=classifier_logic.retention_hours)]
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
        logging.error(f"Server-side error generating stats: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred while fetching statistics."}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
