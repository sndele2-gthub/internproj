import os
import re
import difflib
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import math
import hashlib

# Configure logging to display INFO messages and above
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- ENHANCED CONSTANTS AND KNOWLEDGE BASE ---

class Priority(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

# Impact levels from the user form
IMPACT_LEVELS = {
    "Minimal": Priority.LOW,
    "Moderate": Priority.MEDIUM,
    "Significant": Priority.HIGH,
    "Critical": Priority.CRITICAL,
}

# Adjusted knowledge base for better separation of concerns and more granular scoring
# Using a flat structure for keywords for easier lookup
KNOWLEDGE_BASE = {
    "Safety Concern": {
        "critical_keywords": {"emergency", "fire", "explosion", "toxic", "unconscious", "trapped", "evacuation", "fatal", "imminent danger", "life threatening"},
        "high_keywords": {"danger", "hazard", "unsafe", "injury", "accident", "hurt", "injured", "fall", "cut", "burn", "electrical", "shock", "blocked", "spill", "leak"},
        "medium_keywords": {"safety", "risk", "ppe", "protective", "guard", "warning", "caution", "helmet", "gloves", "training"},
        "negation_words": {"not", "no", "without", "lacking", "need", "should", "could", "want", "wish", "suggest", "could be", "might be", "if"},
    },
    "Machine/Equipment Issue": {
        "critical_keywords": {"explosion", "fire", "complete failure", "shutdown", "total loss", "major breakdown", "catastrophic", "halted", "broken down", "unusable", "seized", "burnt out"},
        "high_keywords": {"broken", "malfunction", "down", "stopped", "jam", "stuck", "overheating", "failure", "error", "crash", "damaged", "faulty", "leaking"},
        "medium_keywords": {"machine", "equipment", "conveyor", "motor", "pump", "repair", "maintenance", "noise", "vibration", "rattling", "squeaking", "loose", "worn", "calibration", "slow", "hesitates", "clicking"},
        "negation_words": {"not", "no", "without", "need", "should", "could", "want", "suggest"},
    },
    "Process Improvement Idea": {
        "high_keywords": {"automate", "streamline", "optimize", "revolutionize"},
        "medium_keywords": {"improve", "efficiency", "productivity", "enhance", "better", "workflow", "process", "reduce waste", "faster"},
        "negation_words": [],
    },
    "Other": {
        "medium_keywords": {"supplies", "training", "lighting", "parking", "temperature", "breakroom", "facilities", "heating", "cooling", "air conditioning", "smell", "unhygienic", "pest"},
        "negation_words": [],
    }
}

MAX_TEXT_LENGTH = 5000

# --- DATA STRUCTURES ---

@dataclass
class SubmissionRecord:
    """Stores a record of a submission for duplicate detection."""
    text: str
    category: str
    priority: str
    timestamp: datetime
    submission_hash: str
    is_escalated: bool = False  # Track if this submission was an escalation

@dataclass
class DuplicateAnalysis:
    """Encapsulates the result of duplicate detection."""
    is_duplicate: bool
    similar_count: int
    escalation_applied: bool
    original_priority: str
    escalated_priority: str

@dataclass
class ClassificationResult:
    """Comprehensive result of the classification process."""
    category: str
    priority: str
    confidence: float
    priority_score: float
    matched_keywords: Dict[str, List[str]] = field(default_factory=dict)
    priority_factors: List[str] = field(default_factory=list)
    duplicate_analysis: Optional[DuplicateAnalysis] = None
    error: Optional[str] = None

# --- CORE LOGIC CLASSES ---

class DuplicateDetector:
    """Manages submission history and detects duplicates for priority escalation."""
    def __init__(self, similarity_threshold=0.6, escalation_threshold=2, retention_hours=168):
        self.similarity_threshold = similarity_threshold
        self.escalation_threshold = escalation_threshold
        self.retention_hours = retention_hours
        self.submissions: List[SubmissionRecord] = []

    def clean_old_submissions(self):
        """Removes submissions older than the retention period."""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        self.submissions = [s for s in self.submissions if s.timestamp > cutoff_time]
        logger.info(f"Cleaned old submissions. Current count: {len(self.submissions)}")

    def generate_content_hash(self, text: str) -> str:
        """Generates a hash for exact duplicate detection."""
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculates similarity using a weighted combination of SequenceMatcher and Jaccard similarity."""
        if not text1 or not text2:
            return 0.0
        norm1 = re.sub(r'\s+', ' ', text1.lower().strip())
        norm2 = re.sub(r'\s+', ' ', text2.lower().strip())

        sequence_sim = difflib.SequenceMatcher(None, norm1, norm2).ratio()
        
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        if not words1 and not words2:
            word_sim = 1.0
        elif not words1 or not words2:
            word_sim = 0.0
        else:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            word_sim = intersection / union
        
        return (sequence_sim * 0.4) + (word_sim * 0.6)

    def categories_compatible(self, cat1: str, cat2: str) -> bool:
        """Determines if two categories are related for duplicate detection."""
        if cat1 == cat2:
            return True
        # Safety and equipment issues are often related in practical scenarios
        safety_equipment = {"Safety Concern", "Machine/Equipment Issue"}
        return cat1 in safety_equipment and cat2 in safety_equipment

    def analyze_and_store(self, text: str, category: str, priority: str) -> DuplicateAnalysis:
        """Analyzes a new submission for duplicates, applies escalation, and stores it."""
        self.clean_old_submissions()
        
        similar_submissions = []
        new_hash = self.generate_content_hash(text)
        
        # Find similar submissions that match or are compatible in category
        for existing in self.submissions:
            if new_hash == existing.submission_hash or \
               (self.categories_compatible(category, existing.category) and \
               self.calculate_similarity(text, existing.text) >= self.similarity_threshold):
                similar_submissions.append(existing)
        
        similar_low_medium_count = len([s for s in similar_submissions if s.priority in (Priority.LOW.value, Priority.MEDIUM.value)])
        
        escalation_applied = False
        escalated_priority = priority
        
        if Priority(priority) in (Priority.LOW, Priority.MEDIUM) and similar_low_medium_count >= self.escalation_threshold:
            escalation_applied = True
            escalated_priority = Priority.CRITICAL.value
            logger.info(f"Escalating from {priority} to {escalated_priority} due to {similar_low_medium_count} similar submissions.")
        elif Priority(priority) == Priority.CRITICAL and len(similar_submissions) > 0:
             # If it's already critical, just note that it was a repeated critical issue
             escalation_applied = True
             logger.info(f"Submission already Critical. Similar submissions found: {len(similar_submissions)}. Marking as escalation_applied.")
        
        # Store the new submission
        submission = SubmissionRecord(
            text=text,
            category=category,
            priority=escalated_priority,
            timestamp=datetime.now(),
            submission_hash=new_hash,
            is_escalated=escalation_applied
        )
        self.submissions.append(submission)
        logger.info(f"New submission added. Total submissions: {len(self.submissions)}")
        
        return DuplicateAnalysis(
            is_duplicate=len(similar_submissions) > 0,
            similar_count=len(similar_submissions),
            escalation_applied=escalation_applied,
            original_priority=priority,
            escalated_priority=escalated_priority
        )

class FeedbackClassifier:
    """Main classifier class to determine category and priority."""
    def __init__(self):
        self.knowledge_base = KNOWLEDGE_BASE
        self.duplicate_detector = DuplicateDetector()

    def normalize_text(self, text: str) -> List[str]:
        """Tokenizes and normalizes text for analysis."""
        return re.findall(r'\b\w+\b', text.lower())

    def calculate_category_score(self, text_tokens: List[str], category: str) -> Tuple[float, Dict[str, List[str]]]:
        """Calculates a category score based on keyword matches and context."""
        category_data = self.knowledge_base[category]
        all_keywords = {
            "critical": category_data.get("critical_keywords", set()),
            "high": category_data.get("high_keywords", set()),
            "medium": category_data.get("medium_keywords", set()),
        }
        negation_words = category_data.get("negation_words", set())

        scores = {"critical": 0, "high": 0, "medium": 0}
        matched_keywords = {"critical": [], "high": [], "medium": []}

        for i, token in enumerate(text_tokens):
            weight = 0
            level = None
            if token in all_keywords["critical"]:
                weight = 3.0
                level = "critical"
            elif token in all_keywords["high"]:
                weight = 2.0
                level = "high"
            elif token in all_keywords["medium"]:
                weight = 1.0
                level = "medium"

            if weight > 0:
                is_negated = any(text_tokens[j] in negation_words for j in range(max(0, i-3), min(len(text_tokens), i+4)))
                if not is_negated:
                    scores[level] += weight
                    matched_keywords[level].append(token)
        
        # Calculate a total score, but also keep track of individual levels
        total_score = scores["critical"] + scores["high"] + scores["medium"]
        
        return total_score, matched_keywords

    def determine_priority(self, total_scores: Dict[str, float], matched_keywords: Dict[str, List[str]], text: str) -> Tuple[Priority, float, List[str]]:
        """Determines the final priority based on scores and explicit text."""
        priority_factors = []
        normalized_text = " ".join(self.normalize_text(text))
        
        # Check for explicit user-submitted impact level (from the prompt image)
        explicit_impact_level = None
        impact_match = re.search(r'impact level: (\w+)', normalized_text, re.IGNORECASE)
        if impact_match and impact_match.group(1).capitalize() in IMPACT_LEVELS:
            explicit_impact_level = impact_match.group(1).capitalize()
            priority_factors.append(f"Explicit Impact Level: {explicit_impact_level}")

        # Check for direct, high-impact keywords
        critical_keyword_score = total_scores["Safety Concern"]["critical"] + total_scores["Machine/Equipment Issue"]["critical"]
        high_keyword_score = total_scores["Safety Concern"]["high"] + total_scores["Machine/Equipment Issue"]["high"]
        
        final_priority = Priority.LOW
        priority_score = 1.0

        if explicit_impact_level == "Critical" or critical_keyword_score >= 3.0:
            final_priority = Priority.CRITICAL
            priority_score = 4.5
            if critical_keyword_score >= 3.0: priority_factors.append("Direct Critical indicators detected.")
        elif explicit_impact_level == "Significant" or high_keyword_score >= 2.0:
            final_priority = Priority.HIGH
            priority_score = 3.5
            if high_keyword_score >= 2.0: priority_factors.append("High severity indicators.")
        elif explicit_impact_level == "Moderate" or total_scores.get("Process Improvement Idea", {}).get("high", 0) > 0 or total_scores.get("Other", {}).get("medium", 0) > 0:
            final_priority = Priority.MEDIUM
            priority_score = 2.5
            if explicit_impact_level == "Moderate": priority_factors.append("Explicit Impact Level: Moderate")
            if total_scores.get("Process Improvement Idea", {}).get("high", 0) > 0: priority_factors.append("High-level process improvement keywords found.")
            if total_scores.get("Other", {}).get("medium", 0) > 0: priority_factors.append("Facilities issue keywords found.")
        else:
            final_priority = Priority.LOW
            priority_score = 1.5
            priority_factors.append("Low severity or general suggestion.")
        
        return final_priority, priority_score, priority_factors

    def calculate_confidence(self, category_scores: Dict[str, float], best_category: str) -> float:
        """Calculates classification confidence based on score separation."""
        sorted_scores = sorted(category_scores.values(), reverse=True)
        best_score = sorted_scores[0] if sorted_scores else 0
        
        if best_score == 0:
            return 0.1
        
        second_best_score = sorted_scores[1] if len(sorted_scores) > 1 else 0
        
        # Confidence is a function of both the absolute best score and the gap to the second best
        if best_score > 0 and best_score != second_best_score:
            separation_factor = (best_score - second_best_score) / best_score
            confidence = min(max(separation_factor * 0.75 + (best_score / max(sum(category_scores.values()), 1)) * 0.25, 0.1), 1.0)
        else:
            confidence = min(max(best_score * 0.8, 0.1), 1.0)
            
        return confidence
        
    def classify(self, text: str) -> ClassificationResult:
        """The main classification pipeline."""
        if not text or not text.strip():
            logger.warning("Classification error: Empty text provided.")
            return ClassificationResult(category="Other", priority=Priority.LOW.value, confidence=0.1, priority_score=1.0, error="Empty text provided")

        try:
            text_tokens = self.normalize_text(text)
            category_scores = {}
            all_matched_keywords = {}

            # Calculate scores for all categories
            for category in self.knowledge_base.keys():
                score, matches = self.calculate_category_score(text_tokens, category)
                category_scores[category] = score
                all_matched_keywords[category] = matches
            
            # Find the best category
            best_category = max(category_scores, key=category_scores.get, default="Other")
            
            # Determine initial priority based on the text content
            initial_priority, initial_priority_score, priority_factors = self.determine_priority(category_scores, all_matched_keywords, text)
            
            # Check for duplicates and apply escalation logic
            duplicate_analysis = self.duplicate_detector.analyze_and_store(text, best_category, initial_priority.value)
            
            final_priority = duplicate_analysis.escalated_priority
            
            if duplicate_analysis.escalation_applied:
                priority_factors.append(f"ESCALATED TO CRITICAL: {duplicate_analysis.similar_count} similar submissions detected.")
                priority_factors.append(f"Original Priority: {duplicate_analysis.original_priority} -> Final Priority: {final_priority}")
            
            confidence = self.calculate_confidence(category_scores, best_category)
            
            result = ClassificationResult(
                category=best_category,
                priority=final_priority,
                confidence=round(confidence, 3),
                priority_score=round(initial_priority_score, 2),
                matched_keywords=all_matched_keywords.get(best_category, {}),
                priority_factors=priority_factors,
                duplicate_analysis=duplicate_analysis
            )
            
            logger.info(f"Classification Result: Category={result.category}, Priority={result.priority}, Confidence={result.confidence}, Escalation_Applied={result.duplicate_analysis.escalation_applied}")
            return result
            
        except Exception as e:
            logger.error(f"Classification error: {e}", exc_info=True)
            return ClassificationResult(
                category="Other", priority=Priority.LOW.value, confidence=0.1,
                priority_score=1.0, error=str(e)
            )

classifier = FeedbackClassifier()

# --- FLASK API ENDPOINTS ---

@app.route("/classify", methods=['POST'])
def handle_classify():
    data = request.get_json()
    if not data or 'text' not in data:
        logger.warning("Missing 'text' field in request.")
        return jsonify({"error": "Missing 'text' field"}), 400
    
    text = data.get('text', '')
    if len(text) > MAX_TEXT_LENGTH:
        logger.warning(f"Text exceeds MAX_TEXT_LENGTH ({MAX_TEXT_LENGTH} chars).")
        return jsonify({"error": f"Text exceeds {MAX_TEXT_LENGTH} characters"}), 400
    
    result = classifier.classify(text)
    
    response = {
        "category": result.category,
        "autocategory": result.category,
        "priority": result.priority,
        "autopriority": result.priority,
        "confidence": result.confidence,
        "priority_score": result.priority_score,
        "matched_keywords": result.matched_keywords,
        "priority_factors": result.priority_factors
    }
    
    if result.duplicate_analysis:
        response["duplicate_analysis"] = {
            "is_duplicate": result.duplicate_analysis.is_duplicate,
            "similar_count": result.duplicate_analysis.similar_count,
            "escalation_applied": result.duplicate_analysis.escalation_applied,
            "original_priority": result.duplicate_analysis.original_priority,
            "escalated_priority": result.duplicate_analysis.escalated_priority
        }
    
    logger.info(f"API Response: {response}")
    return jsonify(response)

@app.route("/duplicate_stats", methods=['GET'])
def get_duplicate_stats():
    classifier.duplicate_detector.clean_old_submissions()
    submissions = classifier.duplicate_detector.submissions
    
    escalated_critical_count = len([s for s in submissions if s.is_escalated])

    response = {
        "total_submissions_retained": len(submissions),
        "escalated_critical_in_memory": escalated_critical_count,
        "retention_hours": classifier.duplicate_detector.retention_hours,
        "similarity_threshold": classifier.duplicate_detector.similarity_threshold,
        "escalation_threshold": classifier.duplicate_detector.escalation_threshold
    }
    logger.info(f"Duplicate Stats: {response}")
    return jsonify(response)

@app.route("/")
def home():
    return """
    <!DOCTYPE html>
    <html><head><title>Enhanced Feedback Classification API</title>
    </head><body>
    <h1>Enhanced Feedback Classification API</h1>
    <p>Use the <strong>POST /classify</strong> endpoint to classify text.</p>
    <p>Use the <strong>GET /duplicate_stats</strong> endpoint for duplicate detection statistics.</p>
    </body></html>
    """

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
