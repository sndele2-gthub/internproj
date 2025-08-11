import os
import re
import difflib
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from flask import Flask, request, render_template_string
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
@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    if request.method == "POST":
        text = request.form.get("feedback_text", "")
        results = classifier_logic.classify_and_process(text)

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Feedback Classifier</title>
        <style>
            body {
                font-family: 'Segoe UI', Arial, sans-serif;
                background-color: #f4f7f6;
                color: #333;
                margin: 0;
                padding: 40px;
                display: flex;
                flex-direction: column;
                align-items: center;
                min-height: 100vh;
            }
            .container {
                background: #ffffff;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                width: 100%;
                max-width: 650px;
                margin-top: 20px;
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 25px;
                font-weight: 600;
            }
            form {
                display: flex;
                flex-direction: column;
            }
            textarea {
                width: 100%;
                height: 150px;
                padding: 15px;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                font-size: 16px;
                margin-bottom: 20px;
                resize: vertical;
                box-sizing: border-box;
            }
            button {
                padding: 12px 20px;
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 18px;
                cursor: pointer;
                transition: background-color 0.3s, transform 0.2s;
                font-weight: 600;
            }
            button:hover {
                background-color: #2980b9;
                transform: translateY(-2px);
            }
            .results {
                margin-top: 30px;
                border-top: 1px solid #e0e0e0;
                padding-top: 25px;
            }
            .results h2 {
                color: #2c3e50;
                margin-bottom: 20px;
                font-weight: 600;
            }
            .result-item {
                margin-bottom: 15px;
                display: flex;
                flex-direction: column;
            }
            .result-item strong {
                font-size: 1.1em;
                color: #555;
                margin-bottom: 5px;
            }
            .result-item span, .result-item ul {
                font-size: 1em;
                color: #333;
                margin: 0;
            }
            .result-item ul {
                list-style-type: disc;
                padding-left: 25px;
                margin-top: 5px;
            }
            .priority-critical { color: #e74c3c; font-weight: bold; }
            .priority-high { color: #f39c12; font-weight: bold; }
            .priority-medium { color: #3498db; font-weight: bold; }
            .priority-low { color: #2ecc71; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>AI Feedback Classification</h1>
        <div class="container">
            <form method="post">
                <textarea name="feedback_text" placeholder="Describe the issue or suggestion..."></textarea>
                <button type="submit">Classify Feedback</button>
            </form>
            {% if results %}
            <div class="results">
                <h2>Classification Results</h2>
                <div class="result-item">
                    <strong>Category:</strong>
                    <span>{{ results.category }}</span>
                </div>
                <div class="result-item">
                    <strong>Priority:</strong>
                    <span class="priority-{{ results.priority.lower() }}">{{ results.priority }}</span>
                </div>
                <div class="result-item">
                    <strong>Confidence:</strong>
                    <span>{{ results.confidence }}</span>
                </div>
                {% if results.priority_factors %}
                <div class="result-item">
                    <strong>Factors:</strong>
                    <ul>
                        {% for factor in results.priority_factors %}
                        <li>{{ factor }}</li>
                        {% endfor %}
                    </ul>
                </div>
                {% endif %}
                <div class="result-item">
                    <strong>Duplicate Status:</strong>
                    <span>{{ "Duplicate" if results.is_duplicate else "Not a Duplicate" }} (Escalation Applied: {{ "Yes" if results.escalation_applied else "No" }})</span>
                </div>
            </div>
            {% endif %}
        </div>
    </body>
    </html>
    """
    return render_template_string(html_content, results=results)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
