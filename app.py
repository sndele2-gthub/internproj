import re
import difflib
import logging
from collections import defaultdict
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from flask import Flask, request, jsonify # Removed render_template_string as HTML is gone
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
        # This regex looks for "impact level: " followed by one of the known levels
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
        # Clean old submissions
        self.submissions = [s for s in self.submissions if s.timestamp > datetime.now() - timedelta(hours=self.retention_hours)]
        new_hash, similar_count = hashlib.md5(text.lower().encode()).hexdigest(), 0
        
        for s in self.submissions:
            # Check for exact hash match or high text similarity within the same category
            is_match = (s.submission_hash == new_hash) or \
                       (difflib.SequenceMatcher(None, text.lower(), s.text.lower()).ratio() > self.similarity_threshold and s.category == category)
            if is_match:
                similar_count += 1
        
        is_dup, escalated = similar_count > 0, False
        final_prio, original_prio = priority, priority
        
        # Escalation logic for LOW/MEDIUM original priorities
        if original_prio in (Priority.LOW.value, Priority.MEDIUM.value) and similar_count >= self.escalation_threshold:
            escalated, final_prio = True, Priority.CRITICAL.value
        elif original_prio == Priority.CRITICAL.value and similar_count > 0:
            # If already critical and is a duplicate, mark as re-occurrence
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

# --- Flask API Endpoints ---
# The root route now just returns a simple confirmation that the API is running
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Feedback Classification API is running. Use /classify for classification and /stats for statistics."})

@app.route("/classify", methods=["POST"])
def classify_route():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            logging.warning("Missing 'text' in request body for /classify endpoint.")
            return jsonify({"error": "Missing 'text' in request body"}), 400
        
        result = classifier_logic.classify_and_process(text)
        
        # Format matched_keywords as expected (e.g., {"critical": [...], "high": [...], "medium": [...]})
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

@app.route("/stats", methods=["GET"]) # Changed to /stats for clarity
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
