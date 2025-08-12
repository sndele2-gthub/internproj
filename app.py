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
        "critical": {"emergency", "fire", "explosion", "fatal", "dangerous", "imminent danger", "life-threatening", "critical injury", "collapse", "toxic leak", "electrocution", "unsafe structure", "structural failure", "immediate danger"},
        "high": {"hazard", "unsafe", "accident", "injury risk", "fall risk", "structural damage", "chemical spill", "electrical issue", "no safety gear", "blocked exit", "gas leak", "safety violation", "exposed wiring"},
        "medium": {"safety concern", "risk identified", "warning sign", "slippery floor", "trip hazard", "poor visibility", "loud noise", "minor injury", "first aid needed", "broken glass", "spill", "unsecured item", "light out"}, # Tweak: Added "light out"
        "negation": {"not", "no", "without", "lacking", "un-", "non-", "safe", "clear", "insignificant", "minimal risk"},
    },
    "Machine/Equipment Issue": {
        "critical": {"complete failure", "total shutdown", "catastrophic", "unusable", "major breakdown", "production halt", "burst pipe", "electrical short", "machine dead", "severely damaged", "irreparable"},
        "high": {"malfunction", "down", "stopped working", "leaking fluid", "faulty", "error code", "damaged", "overheating", "smoking", "intermittent failure", "broken part", "system crash", "no power", "offline"},
        "medium": {"noise", "vibration", "loose part", "maintenance needed", "humming", "grinding", "stuck", "press issue", "adjustment required", "defect", "calibration", "worn out", "slow performance", "clogged", "filter", "compressor", "engine", "pump", "robot", "conveyor", "temperature", "cold", "warm", "comfort", "HVAC", "ventilation", "drafty", "not working"}, # Tweak: Added "not working"
        "negation": {"not", "no", "without", "un-", "non-", "working", "functional", "repaired", "fixed", "normal operation"},
    },
    "Process Improvement Idea": {
        "critical": {"bottleneck", "critical delay", "major inefficiency", "costly error", "legal non-compliance", "regulatory violation", "audit failure", "severe waste"},
        "high": {"automate", "streamline", "optimize", "reduce waste", "cost saving", "quality improvement", "significant inefficiency", "redundant steps", "data inaccuracy", "improve workflow", "new procedure", "expedite", "better method"},
        "medium": {"improve", "suggestion", "idea", "process enhancement", "workflow improvement", "efficiency gain", "better method", "new system", "simplify", "training need", "communication gap", "feedback mechanism", "documentation"},
        "negation": {"not", "no", "without", "current process is fine", "working well", "efficient", "optimal"},
    },
    # The "Other" category is now a fallback for anything that doesn't fit the main three well.
    "Other": {
        "medium": {"general inquiry", "feedback", "suggestion", "question", "comment", "miscellaneous", "not listed", "supplies", "lighting", "parking"}, # General terms
        "negation": [],
    }
}

# --- Core Logic ---
class ClassifierLogic:
    def __init__(self):
        self.submissions: List[SubmissionRecord] = []
        self.retention_hours = 168
        self.similarity_threshold = 0.65 # Adjusted threshold for improved duplicate detection
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
        # This covers cases like "Process Improvement Idea" or general "Other" with medium keywords
        # Also ensures that if any category has a score, it's at least Medium
        if any(s > 0 for s in scores.values()): # Check if any category has a score (even 'Other')
            factors.append("General issue indicators detected across categories.")
            return Priority.MEDIUM, factors
        
        # 4. Default to Low priority if no other conditions met
        factors.append("No specific severity indicators found. Defaulting to Low.")
        return Priority.LOW, factors


    def _analyze_duplicates(self, text: str, category: str, priority: str) -> Tuple[bool, bool, int, str, str]:
        """Manages submission history to detect and potentially escalate duplicate issues."""
        # Remove old submissions from memory that are outside the retention window
        self.submissions = [s for s in self.submissions if s.timestamp > datetime.now() - timedelta(hours=self.retention_hours)]
        
        new_hash = hashlib.md5(text.lower().encode()).hexdigest()
        similar_count = 0
        
        logging.info(f"Analyzing duplicates for new submission (Category: {category}, Priority: {priority}, Text: '{text[:50]}...')")
        
        for s in self.submissions:
            # Check for exact hash match (strongest form of duplicate)
            if s.submission_hash == new_hash:
                similar_count += 1
                logging.info(f"  Exact hash match found for previous submission '{s.text[:50]}...'. Current count: {similar_count}")
                continue # Move to next submission if exact match found
            
            # Perform fuzzy matching only if categories are the same
            if s.category == category:
                similarity_ratio = difflib.SequenceMatcher(None, text.lower(), s.text.lower()).ratio()
                if similarity_ratio > self.similarity_threshold:
                    similar_count += 1
                    logging.info(f"  Fuzzy match found (Ratio: {similarity_ratio:.2f} > {self.similarity_threshold}) for previous submission '{s.text[:50]}...' (Category: {s.category}). Current count: {similar_count}")
            else:
                logging.info(f"  Skipping fuzzy match for '{s.text[:50]}...' due to category mismatch ({s.category} != {category})")

        is_dup, escalated = similar_count > 0, False
        final_prio, original_prio = priority, priority # Store original priority before potential escalation
        
        # Logic for auto-escalation of Low/Medium issues based on recurrence
        if original_prio in (Priority.LOW.value, Priority.MEDIUM.value) and similar_count >= self.escalation_threshold:
            escalated, final_prio = True, Priority.CRITICAL.value
            logging.info(f"ESCALATION: Priority for '{text[:50]}...' escalated from {original_prio} to {final_prio} due to {similar_count} similar entries (threshold: {self.escalation_threshold}).")
        elif original_prio == Priority.CRITICAL.value and similar_count > 0:
            # If an issue is already critical and re-occurs, mark it as escalated for tracking
            escalated = True
            logging.info(f"RE-OCCURRENCE: Critical issue '{text[:50]}...' is a duplicate (similar_count: {similar_count}), indicating recurrence.")
        
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
            # assign to "Other" as a general fallback category.
            best_category = "Other"

            
        initial_priority, factors = self._determine_priority(scores, text)
        is_dup, escalated, similar_count, final_prio, original_prio = self._analyze_duplicates(text, best_category, initial_priority.value)
        
        if escalated:
            factors.append(f"System-determined priority: {original_prio} -> {final_prio} (Escalated)")
            if similar_count > 0:
                factors.append(f"Reason: {similar_count} similar reports found.")
        
        # Calculate confidence as an integer between 0 and 10
        score_for_confidence = scores.get(best_category, 0)
        
        # --- NEW Confidence Calculation Logic (More granular mapping) ---
        # Map raw score to 0-10 confidence. This mapping is subjective and can be fine-tuned.
        if score_for_confidence <= 0.5: # Very weak or no matches
            confidence_score_0_10 = 0
        elif score_for_confidence < 1.0: # Single medium keyword, or very weak sum
            confidence_score_0_10 = 1
        elif score_for_confidence < 2.0: # Moderate sum, maybe couple of mediums
            confidence_score_0_10 = 3
        elif score_for_confidence < 3.0: # Strong medium, or one low high keyword
            confidence_score_0_10 = 5
        elif score_for_confidence < 4.5: # One critical keyword, or multiple high
            confidence_score_0_10 = 7
        elif score_for_confidence < 6.0: # Strong critical + some others
            confidence_score_0_10 = 9
        else: # score >= 6.0 (multiple strong matches)
            confidence_score_0_10 = 10

        # Ensure the score is clamped between 0 and 10
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
