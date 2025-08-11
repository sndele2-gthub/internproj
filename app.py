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
    confidence: int # Score 0-10
    matched_keywords: Dict[str, List[str]] = field(default_factory=dict)
    priority_factors: List[str] = field(default_factory=list)
    is_duplicate: bool = False
    escalation_applied: bool = False
    original_priority: str = "N/A"
    similar_count: int = 0

# --- Knowledge Base & Constants ---
MAX_TEXT_LENGTH = 5000
KNOWLEDGE_BASE = {
    "Safety Concern": {
        "critical": {"emergency", "fire", "explosion", "fatal", "dangerous", "imminent danger", "life-threatening", "critical injury", "collapse", "toxic leak", "electrocution"},
        "high": {"hazard", "unsafe", "accident", "injury risk", "fall risk", "structural damage", "chemical spill", "electrical issue", "no safety gear", "blocked exit", "gas leak"},
        "medium": {"safety concern", "risk identified", "warning sign", "slippery floor", "trip hazard", "poor visibility", "loud noise", "minor injury", "first aid needed"},
        "negation": {"not", "no", "without", "lacking", "un-", "non-", "safe", "clear", "insignificant"},
    },
    "Machine/Equipment Issue": {
        "critical": {"complete failure", "total shutdown", "catastrophic", "unusable", "major breakdown", "production halt", "burst pipe", "electrical short"},
        "high": {"malfunction", "down", "stopped working", "leaking fluid", "faulty", "error code", "damaged", "overheating", "smoking", "intermittent failure", "broken part"},
        "medium": {"noise", "vibration", "loose part", "maintenance needed", "humming", "grinding", "stuck", "press issue", "adjustment required", "defect", "calibration", "worn out", "slow performance"},
        "negation": {"not", "no", "without", "un-", "non-", "working", "functional", "repaired"},
    },
    "Process Improvement Idea": {
        "critical": {"bottleneck", "critical delay", "major inefficiency", "costly error", "legal non-compliance"},
        "high": {"automate", "streamline", "optimize", "reduce waste", "cost saving", "quality improvement", "significant inefficiency", "redundant steps", "data inaccuracy"},
        "medium": {"improve", "suggestion", "idea", "process enhancement", "workflow improvement", "efficiency gain", "better method", "new system", "simplify", "training need", "communication gap"},
        "negation": {"not", "no", "without", "current process is fine", "working well"},
    },
    "Facility/Environment Issue": {
        "critical": {"structural integrity", "major leak", "fire hazard", "mold infestation", "pest infestation", "unsafe air quality", "sewage backup"},
        "high": {"extreme temperature", "poor ventilation", "odor", "dirty environment", "security breach", "blocked access", "water damage", "power outage", "broken fixture"},
        "medium": {"supplies low", "lighting issue", "parking problem", "temperature uncomfortable", "cold office", "warm office", "comfort issue", "HVAC problem", "messy area", "restroom issue", "cleaning needed", "broken furniture", "noise disturbance", "wifi issue"},
        "negation": {"not", "no", "without", "clean", "comfortable", "functional", "acceptable"},
    }
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
                negation_keywords = data.get("negation", set())

                if token in critical_keywords: score, level = 3.0, "critical"
                elif token in high_keywords: score, level = 2.0, "high"
                elif token in medium_keywords: score, level = 1.0, "medium"
                
                # Apply negation check: if a negation word is near a keyword, reduce its score
                if score > 0 and not any(t in negation_keywords for t in tokens[max(0, i-3):i+4]):
                    cat_scores[cat] += score
                    matched_keys[cat][level].append(token)
        return dict(cat_scores), dict(matched_keys)

    def _determine_priority(self, scores: Dict, text: str) -> Tuple[Priority, List[str]]:
        """
        Determines the overall priority based on calculated scores and explicit mentions.
        Prioritizes explicit impact levels, then critical keywords, then other matched keywords.
        """
        factors, text_lower = [], text.lower()
        
        # 1. Check for explicit impact level mention
        explicit_level_match = re.search(r"impact level:\s*(minimal|moderate|significant|critical)", text_lower)
        explicit_level = explicit_level_match.group(1).capitalize() if explicit_level_match else None

        if explicit_level:
            factors.append(f"Explicit Impact Level: {explicit_level}")
            if explicit_level == "Critical": return Priority.CRITICAL, factors
            if explicit_level == "Significant": return Priority.HIGH, factors
            if explicit_level == "Moderate": return Priority.MEDIUM, factors
            if explicit_level == "Minimal": return Priority.LOW, factors

        # 2. Evaluate critical and high scores from Safety/Equipment concerns (as these are generally high priority)
        # Sum scores from categories most likely to indicate high severity if no explicit level
        critical_concern_score = scores.get("Safety Concern", 0) * 1.5 + scores.get("Machine/Equipment Issue", 0) # Safety weighted slightly higher
        
        if critical_concern_score >= 3.0: # A strong critical indicator
            factors.append("Direct critical keywords detected in high-impact categories.")
            return Priority.CRITICAL, factors
        elif critical_concern_score >= 1.5: # A strong high indicator or multiple medium safety/equipment
            factors.append("High severity indicators detected in high-impact categories.")
            return Priority.HIGH, factors

        # 3. If no explicit levels or strong critical/high indicators, check for any positive category score
        # This covers cases like "Process Improvement Idea" or general "Facility/Environment Issue" with medium keywords
        # Also ensures that if any category has a score, it's at least Medium
        if any(s > 0 for s in scores.values()): # Check if any category has a score (even 'Facility/Environment Issue')
            factors.append("General issue indicators detected across categories.")
            return Priority.MEDIUM, factors
        
        # 4. Default to Low priority if no other conditions met
        factors.append("No specific severity indicators found. Defaulting to Low.")
        return Priority.LOW, factors


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
            # If input is invalid, set confidence to 0
            return ClassificationResult("Invalid Input", Priority.LOW.value, 0, {}, ["Invalid input: Text is empty or too long."])
        
        tokens = self._normalize_text(text)
        scores, keywords = self._calculate_scores(tokens)
        
        # Determine the best category based on highest score.
        # Prioritize categories with actual matched keywords.
        categories_with_scores = {cat: score for cat, score in scores.items() if score > 0}

        if categories_with_scores:
            # Find the best category among those with scores
            best_category = max(categories_with_scores, key=categories_with_scores.get)
        else:
            # If no specific keywords matched any category with a positive score,
            # assign a default category for general feedback.
            best_category = "Facility/Environment Issue" # Default if no specific match

            
        initial_priority, factors = self._determine_priority(scores, text)
        is_dup, escalated, similar_count, final_prio, original_prio = self._analyze_duplicates(text, best_category, initial_priority.value)
        
        if escalated:
            factors.append(f"System-determined priority: {original_prio} -> {final_prio} (Escalated)")
            if similar_count > 0:
                factors.append(f"Reason: {similar_count} similar reports found.")
        
        # Calculate confidence as an integer between 0 and 10
        # The score is based on the best category's score relative to a conceptual max score.
        # A simple max score could be 3 (critical keyword) * number of keywords if they all matched.
        # Let's use a dynamic max_score_potential based on the category's keyword presence.
        
        # If the best category has actual scores, normalize against its potential maximum.
        # A simple approach: sum of max scores (3+2+1=6) for a category, assuming it contains all levels.
        max_score_for_chosen_category = 0
        if best_category in KNOWLEDGE_BASE:
            # Assuming max score is when one critical, one high, one medium keyword hit.
            # This is a simplification; a more complex model might count multiple hits.
            max_score_for_chosen_category += 3.0 if KNOWLEDGE_BASE[best_category].get("critical") else 0
            max_score_for_chosen_category += 2.0 if KNOWLEDGE_BASE[best_category].get("high") else 0
            max_score_for_chosen_category += 1.0 if KNOWLEDGE_BASE[best_category].get("medium") else 0
        
        # If best_category has no score (e.g., initial "Invalid Input" or "Other" with no specific hits),
        # set confidence to 0 to avoid division by zero or inflated confidence.
        if scores.get(best_category, 0) == 0 or max_score_for_chosen_category == 0:
            confidence_score_0_10 = 0
        else:
            normalized_score = scores.get(best_category, 0) / max_score_for_chosen_category
            confidence_score_0_10 = int(round(normalized_score * 10))
            # Clamp between 0 and 10
            confidence_score_0_10 = max(0, min(10, confidence_score_0_10))

        return ClassificationResult(
            best_category, final_prio, confidence_score_0_10,
            keywords.get(best_category, {}), factors,
            is_dup, escalated, original_prio, similar_count
        )

classifier_logic = ClassifierLogic()

# --- Flask API Endpoints ---
@app.route("/", methods=["GET"])
def home():
    """Returns a simple message indicating the API is running."""
    return jsonify({"message": "Feedback Classification API is running. Use /classify for classification and /stats for statistics."})

@app.route("/classify", methods=["POST"])
def classify_route():
    """Endpoint for classifying feedback text. Expects a JSON payload with a 'text' field."""
    try:
        data = request.get_json(silent=True)

        if data is None:
            raw_data = request.data.decode('utf-8', errors='ignore')
            logging.error(f"Request body is not valid JSON or is empty. Raw data received: '{raw_data[:200]}...'")
            return jsonify({"error": "Request body must be valid JSON."}), 400
        
        text = data.get("text")

        if not isinstance(text, str):
            logging.warning(f"Invalid 'text' field type: {type(text)}. Expected string. Received data: {data}")
            return jsonify({"error": "Invalid or missing 'text' field in JSON payload. Expected a string."}), 400
        
        if not text.strip():
            logging.warning("Received empty 'text' field after stripping whitespace.")
            return jsonify({"error": "Text field cannot be empty or just whitespace."}), 400

        logging.info(f"Received text for classification: '{text[:100]}...'")
        result = classifier_logic.classify_and_process(text)
        
        # Format matched_keywords for consistent JSON output (even if empty)
        formatted_keywords = {
            "critical": result.matched_keywords.get("critical", []),
            "high": result.matched_keywords.get("high", []),
            "medium": result.matched_keywords.get("medium", [])
        }

        response_data = {
            "category": result.category,
            "priority": result.priority,
            "confidence": result.confidence, # Now an integer between 0-10
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
        return jsonify({"error": "An internal server error occurred during classification. Please check server logs."}), 500

@app.route("/stats", methods=["GET"])
def stats_route():
    """Endpoint for retrieving duplicate detection statistics."""
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
        return jsonify({"error": "An internal server error occurred while fetching statistics. Please check server logs."}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
