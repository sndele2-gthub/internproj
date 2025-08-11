import re
import difflib
import logging
from collections import defaultdict
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from flask import Flask, request, jsonify
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
        """Normalizes text by converting to lowercase and finding all word characters."""
        return re.findall(r'\b\w+\b', text.lower())

    def _calculate_scores(self, tokens: List[str]) -> Tuple[Dict, Dict]:
        """Calculates keyword scores for each category based on normalized tokens."""
        cat_scores, matched_keys = defaultdict(float), defaultdict(lambda: defaultdict(list))
        for cat, data in KNOWLEDGE_BASE.items():
            for i, token in enumerate(tokens):
                score = 0
                level = None
                # Safely get keyword sets, defaulting to an empty set if the key doesn't exist
                critical_keywords = data.get("critical", set())
                high_keywords = data.get("high", set())
                medium_keywords = data.get("medium", set())

                if token in critical_keywords: score, level = 3.0, "critical"
                elif token in high_keywords: score, level = 2.0, "high"
                elif token in medium_keywords: score, level = 1.0, "medium"
                
                # Apply negation check: if a negation word is near a keyword, reduce its score
                if score > 0 and not any(t in data.get("negation", set()) for t in tokens[max(0, i-3):i+4]):
                    cat_scores[cat] += score
                    matched_keys[cat][level].append(token)
        return dict(cat_scores), dict(matched_keys)

    def _determine_priority(self, scores: Dict, text: str) -> Tuple[Priority, List[str]]:
        """Determines the overall priority based on calculated scores and explicit mentions."""
        factors, text_lower = [], text.lower()
        
        # Check for explicit impact level mention in the text (e.g., "impact level: critical")
        explicit_level_match = re.search(r"impact level:\s*(minimal|moderate|significant|critical)", text_lower)
        explicit_level = explicit_level_match.group(1).capitalize() if explicit_level_match else None

        if explicit_level and explicit_level in ["Minimal", "Moderate", "Significant", "Critical"]:
            factors.append(f"Explicit Impact Level: {explicit_level}")

        # Combine critical scores from relevant categories (e.g., Safety and Equipment)
        critical_score = scores.get("Safety Concern", 0) + scores.get("Machine/Equipment Issue", 0)
        
        # Determine final priority based on a hierarchy of conditions
        if explicit_level == "Critical" or critical_score >= 3.0:
            return Priority.CRITICAL, factors + (["Direct Critical indicators detected."] if critical_score >= 3.0 else [])
        elif explicit_level == "Significant" or critical_score >= 2.0:
            return Priority.HIGH, factors + (["High severity indicators."] if critical_score >= 2.0 else [])
        elif explicit_level == "Moderate" or any(s > 0 for cat, s in scores.items() if cat != "Other"):
            return Priority.MEDIUM, factors + (["Explicit Impact Level: Moderate"] if explicit_level == "Moderate" else [])
        else:
            return Priority.LOW, factors + ["Low severity or general suggestion."]

    def _analyze_duplicates(self, text: str, category: str, priority: str) -> Tuple[bool, bool, int, str, str]:
        """Manages submission history to detect and potentially escalate duplicate issues."""
        # Remove old submissions from memory
        self.submissions = [s for s in self.submissions if s.timestamp > datetime.now() - timedelta(hours=self.retention_hours)]
        
        new_hash, similar_count = hashlib.md5(text.lower().encode()).hexdigest(), 0
        
        # Check for existing similar submissions
        for s in self.submissions:
            # Match by hash (exact duplicate) or by semantic similarity within the same category
            is_match = (s.submission_hash == new_hash) or \
                       (difflib.SequenceMatcher(None, text.lower(), s.text.lower()).ratio() > self.similarity_threshold and s.category == category)
            if is_match:
                similar_count += 1
        
        is_dup, escalated = similar_count > 0, False
        final_prio, original_prio = priority, priority # Store original priority before potential escalation
        
        # Logic for auto-escalation of Low/Medium issues based on recurrence
        if original_prio in (Priority.LOW.value, Priority.MEDIUM.value) and similar_count >= self.escalation_threshold:
            escalated, final_prio = True, Priority.CRITICAL.value
        elif original_prio == Priority.CRITICAL.value and similar_count > 0:
            # If an issue is already critical and re-occurs, mark it as escalated for tracking
            escalated = True
        
        # Store the current submission with its determined (or escalated) priority
        self.submissions.append(SubmissionRecord(text=text, category=category, priority=final_prio, timestamp=datetime.now(), submission_hash=new_hash, is_escalated=escalated))
        return is_dup, escalated, similar_count, final_prio, original_prio

    def classify_and_process(self, text: str) -> ClassificationResult:
        """Main classification pipeline: tokenization -> scoring -> priority -> duplicate analysis."""
        if not text or len(text) > MAX_TEXT_LENGTH:
            return ClassificationResult("Other", Priority.LOW.value, 0.1, {}, ["Invalid input: Text is empty or too long."])
        
        tokens = self._normalize_text(text)
        scores, keywords = self._calculate_scores(tokens)
        
        # Determine the best category based on highest score
        best_category = max(scores, key=scores.get, default="Other")
        if scores.get(best_category, 0) == 0: # Default to "Other" if no strong category match
            best_category = "Other"
            
        initial_priority, factors = self._determine_priority(scores, text)
        is_dup, escalated, similar_count, final_prio, original_prio = self._analyze_duplicates(text, best_category, initial_priority.value)
        
        if escalated:
            factors.append(f"System-determined priority: {original_prio} -> {final_prio} (Escalated)")
            if similar_count > 0:
                factors.append(f"Reason: {similar_count} similar reports found.")
        
        # Calculate a simple confidence score based on the best category's score
        confidence = 0.5 + (scores.get(best_category, 0) / (sum(scores.values()) + 1) * 0.5)
        
        return ClassificationResult(
            best_category, final_prio, round(confidence, 3), 
            keywords.get(best_category, {}), factors, 
            is_dup, escalated, original_prio, similar_count
        )

classifier_logic = ClassifierLogic()

# --- Flask API Endpoints ---
# The root route provides a simple message indicating the API is running.
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Feedback Classification API is running. Use /classify for classification and /stats for statistics."})

@app.route("/classify", methods=["POST"])
def classify_route():
    """Endpoint for classifying feedback text. Expects a JSON payload with a 'text' field."""
    try:
        # Use silent=True to prevent Flask from automatically raising an error
        # if the Content-Type is application/json but the body is malformed.
        data = request.get_json(silent=True)

        if data is None:
            raw_data = request.data.decode('utf-8', errors='ignore')
            logging.error(f"Request body is not valid JSON or is empty. Raw data received: '{raw_data[:200]}...'")
            return jsonify({"error": "Request body must be valid JSON."}), 400
        
        text = data.get("text")

        # Explicitly check if 'text' is a string and not None
        if not isinstance(text, str):
            logging.warning(f"Invalid 'text' field type: {type(text)}. Expected string. Received data: {data}")
            return jsonify({"error": "Invalid or missing 'text' field in JSON payload. Expected a string."}), 400
        
        if not text.strip():
            logging.warning("Received empty 'text' field after stripping whitespace.")
            return jsonify({"error": "Text field cannot be empty or just whitespace."}), 400

        logging.info(f"Received text for classification: '{text[:100]}...'") # Log the received text
        result = classifier_logic.classify_and_process(text)
        
        # Format matched_keywords for consistent JSON output (even if empty)
        formatted_keywords = {
            "critical": result.matched_keywords.get("critical", []),
            "high": result.matched_keywords.get("high", []),
            "medium": result.matched_keywords.get("medium", [])
        }

        # Construct the response payload
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
        # Catch any unexpected errors during processing and log them with traceback
        logging.error(f"Server-side error during classification: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred during classification. Please check server logs."}), 500

@app.route("/stats", methods=["GET"])
def stats_route():
    """Endpoint for retrieving duplicate detection statistics."""
    try:
        # Ensure submissions list is cleaned before calculating stats
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
        return jsonify({"error": "An internal server error occurred while fetching statistics. Please check server logs."}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
