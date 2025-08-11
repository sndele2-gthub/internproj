import os
import re
import difflib
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import math
import hashlib

# Configure logging to display INFO messages and above
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class Priority(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

# Enhanced knowledge base with weighted keywords and context indicators
KNOWLEDGE_BASE = {
    "Safety Concern": {
        "keywords": {
            # Critical safety indicators (weight: 3.0) - More specific to actual crisis
            "emergency": 3.0, "fire": 3.0, "explosion": 3.0, "toxic": 3.0, "unconscious": 3.0, 
            "trapped": 3.0, "evacuation": 3.0, "severe": 3.0, "blood": 3.0, "collapsed": 3.0,
            "fatal": 3.0, "imminent danger": 3.0, "life threatening": 3.0, "immediate hazard": 3.0,
            
            # High priority safety (weight: 2.0) - Clear and present danger, but not immediate crisis
            "danger": 2.0, "hazard": 2.0, "unsafe": 2.0, "injury": 2.0, "accident": 2.0, 
            "hurt": 2.0, "injured": 2.0, "fall": 2.0, "cut": 2.0, "burn": 2.0, 
            "electrical": 2.0, "shock": 2.0, "blocked": 2.0, "spill": 2.0, "leak": 2.0,
            "critical issue": 2.0, "urgent": 2.0, # Adding these from original critical list, but with 2.0 weight for direct mention
            "risk of injury": 2.0, "violation": 2.0,
            
            # Medium priority safety (weight: 1.0) - Potential issues, suggestions, minor concerns
            "safety": 1.0, "risk": 1.0, "ppe": 1.0, "protective": 1.0, "guard": 1.0, 
            "warning": 1.0, "caution": 1.0, "helmet": 1.0, "gloves": 1.0, "training": 1.0,
            "policy": 1.0, "procedure": 1.0, "inspection": 1.0, "audit": 1.0, "slippery": 1.0,
            "trip hazard": 1.0, "concern": 1.0
        },
        "negation_words": ["not", "no", "without", "lacking", "need", "should", "could", "want", "wish", "suggest", "could be", "might be", "if"],
        "context_reducers": ["suggestion", "idea", "recommend", "propose", "think", "maybe", "could", "should", "future"]
    },
    
    "Machine/Equipment Issue": {
        "keywords": {
            # Critical equipment issues (weight: 3.0) - System down, major failure
            "explosion": 3.0, "fire": 3.0, "complete failure": 3.0, "shutdown": 3.0, 
            "total loss": 3.0, "major breakdown": 3.0, "catastrophic": 3.0, "halted": 3.0,
            "broken down": 3.0, "unusable": 3.0, "emergency": 3.0, "line stopped": 3.0, "production halted": 3.0,
            "seized": 3.0, "burnt out": 3.0, "failed": 3.0,
            
            # High priority equipment (weight: 2.0) - Significant problem, affecting production
            "broken": 2.0, "malfunction": 2.0, "down": 2.0, "stopped": 2.0, "jam": 2.0,
            "stuck": 2.0, "overheating": 2.0, "failure": 2.0, "error": 2.0, "crash": 2.0,
            "damaged": 2.0, "faulty": 2.0, "leaking": 2.0, "no power": 2.0, "critical": 2.0, "urgent": 2.0,
            "major defect": 2.0, "intermittent shutdown": 2.0,
            
            # Medium priority equipment (weight: 1.0) - Minor issues, noises, potential problems
            "machine": 1.0, "equipment": 1.0, "conveyor": 1.0, "motor": 1.0, "pump": 1.0,
            "repair": 1.0, "maintenance": 1.0, "noise": 1.0, "vibration": 1.0,
            "rattling": 1.0, "squeaking": 1.0, "loose": 1.0, "worn": 1.0, "calibration": 1.0,
            "slow": 1.0, "hesitates": 1.0, "clicking": 1.0, "hums": 1.0, "drifting": 1.0
        },
        "negation_words": ["not", "no", "without", "need", "should", "could", "want", "suggest", "could be", "might be", "if"],
        "context_reducers": ["suggestion", "idea", "recommend", "propose", "schedule", "plan", "future"]
    },
    
    "Process Improvement Idea": {
        "keywords": {
            "automate": 2.0, "streamline": 2.0, "optimize": 2.0, "revolutionize": 2.0,
            "improve": 1.0, "efficiency": 1.0, "productivity": 1.0, "enhance": 1.0,
            "suggestion": 0.5, "idea": 0.5, "recommend": 0.5, "think": 0.5,
            "better": 1.0, "workflow": 1.0, "process": 1.0, "reduce waste": 1.0,
            "streamline": 1.0, "faster": 1.0, "reduce time": 1.0, "systematic": 1.0,
            "discrepancy": 1.0, "backlog": 1.0, "error prone": 1.0
        },
        "negation_words": [], # less strict for ideas
        "context_reducers": [] # ideas are inherently suggestions
    },
    
    "Other": {
        "keywords": {
            "supplies": 1.0, "training": 1.0, "lighting": 1.0, "parking": 1.0, 
            "temperature": 1.0, "breakroom": 1.0, "lunchroom": 1.0, "bathroom": 1.0, 
            "coffee": 1.0, "clean": 1.0, "organize": 1.0, "facilities": 1.0,
            "heating": 1.0, "cooling": 1.0, "air conditioning": 1.0, "ventilation": 1.0,
            "desk": 1.0, "chair": 1.0, "office": 1.0, "smell": 1.0, "unhygienic": 1.0,
            "pest": 1.0, "water cooler": 1.0
        },
        "negation_words": ["urgent", "immediate", "critical", "emergency", "dangerous", "hazard", "unsafe", "major", "severe"],
        "context_reducers": []
    }
}

MAX_TEXT_LENGTH = 5000

@dataclass
class SubmissionRecord:
    text: str
    category: str
    priority: str
    timestamp: datetime
    submission_hash: str = ""

@dataclass
class DuplicateAnalysis:
    is_duplicate: bool
    similar_count: int
    escalation_applied: bool
    original_priority: str
    escalated_priority: str

@dataclass
class ClassificationResult:
    category: str
    priority: str
    confidence: float
    priority_score: float
    matched_keywords: List[str]
    priority_factors: List[str]
    duplicate_analysis: Optional[DuplicateAnalysis] = None
    error: Optional[str] = None

class DuplicateDetector:
    def __init__(self, similarity_threshold=0.6, escalation_threshold=2, retention_hours=168): # Adjusted for easier testing
        self.similarity_threshold = similarity_threshold # Lowered threshold to detect more duplicates
        self.escalation_threshold = escalation_threshold # Lowered to trigger escalation faster
        self.retention_hours = retention_hours
        self.submissions: List[SubmissionRecord] = []
        
        # Escalation rules: Only LOW/MEDIUM originally can escalate to CRITICAL
        self.escalation_rules = {
            Priority.LOW: Priority.CRITICAL,
            Priority.MEDIUM: Priority.CRITICAL,
            # Priority.HIGH: Priority.CRITICAL, # Removed direct HIGH to CRITICAL escalation here
            Priority.CRITICAL: Priority.CRITICAL # If already critical, stays critical
        }
    
    def clean_old_submissions(self):
        """Remove submissions older than retention period"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        self.submissions = [s for s in self.submissions if s.timestamp > cutoff_time]
        logger.info(f"Cleaned old submissions. Current submission count: {len(self.submissions)}")
    
    def generate_content_hash(self, text: str) -> str:
        """Generate hash for exact duplicate detection"""
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using a combination of SequenceMatcher and Jaccard similarity"""
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts for comparison
        norm1 = re.sub(r'\s+', ' ', text1.lower().strip())
        norm2 = re.sub(r'\s+', ' ', text2.lower().strip())
        
        # Use SequenceMatcher for overall sequence similarity
        sequence_sim = difflib.SequenceMatcher(None, norm1, norm2).ratio()
        
        # Use word overlap (Jaccard similarity) for concept overlap
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        if not words1 and not words2:
            word_sim = 1.0 # Both empty, consider them identical
        elif not words1 or not words2:
            word_sim = 0.0 # One empty, one not, consider them dissimilar
        else:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            word_sim = intersection / union
            
        # Weighted combination - adjust weights if one method proves more reliable
        combined_similarity = (sequence_sim * 0.4) + (word_sim * 0.6)
        logger.debug(f"Similarity between '{text1}' and '{text2}': {combined_similarity:.2f}")
        return combined_similarity
    
    def find_similar_submissions(self, new_text: str, new_category: str) -> List[SubmissionRecord]:
        """Find submissions similar to the new one within the retention period."""
        similar = []
        new_hash = self.generate_content_hash(new_text)
        
        for existing in self.submissions:
            # Check exact match first (hash)
            if new_hash == existing.submission_hash:
                similar.append(existing)
                logger.debug(f"Exact hash match found for: '{new_text}'")
                continue
            
            # Skip if categories are not compatible (e.g., Safety vs. Process Improvement)
            if not self.categories_compatible(new_category, existing.category):
                continue
            
            # Calculate similarity for non-exact matches within compatible categories
            similarity = self.calculate_similarity(new_text, existing.text)
            if similarity >= self.similarity_threshold:
                similar.append(existing)
                logger.debug(f"Similar submission found (Similarity: {similarity:.2f}): '{existing.text}'")
        
        return similar
    
    def categories_compatible(self, cat1: str, cat2: str) -> bool:
        """Determines if two categories are related enough for duplicate detection."""
        if cat1 == cat2:
            return True
        # Safety and equipment issues are often related in practical scenarios
        safety_equipment = {"Safety Concern", "Machine/Equipment Issue"}
        return cat1 in safety_equipment and cat2 in safety_equipment

    def analyze_submission(self, text: str, category: str, priority: str) -> DuplicateAnalysis:
        """Analyze submission for duplicates and apply escalation rules."""
        self.clean_old_submissions() # Clean before analysis
        
        similar_submissions = self.find_similar_submissions(text, category)
        
        # Count submissions that are relevant for escalating existing low/medium priority issues
        # Now specifically counting LOW/MEDIUM that are still in memory
        escalatable_count = len([s for s in similar_submissions 
                                 if Priority(s.priority) in [Priority.LOW, Priority.MEDIUM]])
        
        should_escalate = False
        escalated_priority = priority # Start with the current determined priority
        
        # Only escalate if the original priority was LOW or MEDIUM and threshold is met
        if Priority(priority) in [Priority.LOW, Priority.MEDIUM] and escalatable_count >= self.escalation_threshold:
            should_escalate = True
            escalated_priority = Priority.CRITICAL.value
            logger.info(f"Escalating from {priority} to {escalated_priority} due to {escalatable_count} similar (original LOW/MEDIUM) submissions.")
        elif Priority(priority) == Priority.CRITICAL:
             # If it's already critical, it remains critical and counts as an "escalation applied" to signify repeated critical
             should_escalate = len(similar_submissions) > 0 # If there are any similar, mark as escalation applied
             escalated_priority = Priority.CRITICAL.value
             logger.info(f"Submission already Critical. Similar submissions found: {len(similar_submissions)}. Marking as escalation_applied: {should_escalate}")
        
        # Store this new submission for future duplicate checks
        submission = SubmissionRecord(
            text=text,
            category=category,
            priority=escalated_priority, # Store the escalated priority if it was applied
            timestamp=datetime.now(),
            submission_hash=self.generate_content_hash(text)
        )
        self.submissions.append(submission)
        logger.info(f"New submission added to detector. Total submissions in memory: {len(self.submissions)}")
        
        return DuplicateAnalysis(
            is_duplicate=len(similar_submissions) > 0,
            similar_count=len(similar_submissions),
            escalation_applied=should_escalate,
            original_priority=priority, # This is the priority *before* escalation by the detector
            escalated_priority=escalated_priority # This is the final priority after detector logic
        )

class FeedbackClassifier:
    def __init__(self):
        self.knowledge_base = KNOWLEDGE_BASE
        self.duplicate_detector = DuplicateDetector()
    
    def normalize_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text.lower().strip())
    
    def calculate_category_score(self, text: str, category: str) -> Tuple[float, List[str]]:
        """Calculate score for a category with context awareness"""
        normalized = self.normalize_text(text)
        words = normalized.split()
        
        category_data = self.knowledge_base[category]
        keywords = category_data["keywords"]
        negation_words = set(category_data.get("negation_words", []))
        context_reducers = set(category_data.get("context_reducers", []))
        
        total_score = 0.0
        matched_keywords = []
        
        for i, word in enumerate(words):
            if word in keywords:
                weight = keywords[word]
                
                # Check for negation context (within 3 words before or 4 words after)
                negation_factor = 1.0
                for j in range(max(0, i-3), min(len(words), i+4)):
                    if words[j] in negation_words:
                        negation_factor = 0.3 # Reduce score significantly if negated
                        break
                
                # Check for context reducers (anywhere in the text)
                context_factor = 1.0
                if any(reducer in normalized for reducer in context_reducers):
                    context_factor = 0.6 # Reduce score slightly if suggestion-like context
                
                adjusted_weight = weight * negation_factor * context_factor
                total_score += adjusted_weight
                matched_keywords.append(word)
        
        # Normalize score relative to maximum possible score for this category
        # Using a fixed divisor for normalization can sometimes lead to very low scores
        # Let's try normalizing based on the number of matched keywords instead
        max_possible_score_for_category = sum(keywords.values()) # Max possible raw score
        
        # If no keywords matched, score is 0. If there are keywords, normalize based on total possible.
        normalized_score = 0.0
        if total_score > 0 and max_possible_score_for_category > 0:
            normalized_score = total_score / max_possible_score_for_category
        
        return normalized_score, matched_keywords
    
    def determine_priority(self, text: str, category: str, category_score: float) -> Tuple[Priority, float, List[str]]:
        """Determine priority based on content analysis, with refined thresholds and ImpactLevel consideration."""
        normalized = self.normalize_text(text)
        factors = []
        
        # Extract ImpactLevel if present in the text (e.g., "Impact Level: Critical")
        impact_level_match = re.search(r"impact level:\s*(Minimal|Moderate|Significant|Critical)", normalized)
        explicit_impact_level = impact_level_match.group(1).capitalize() if impact_level_match else None
        
        # Check for absolute critical indicators
        # Added "critical" and "urgent" to keywords with weight 3.0 for strong direct matches
        # The direct keywords in the KNOWLEDGE_BASE for CRITICAL will now directly influence this.
        absolute_critical_score = 0
        for keyword, weight in KNOWLEDGE_BASE.get(category, {}).get("keywords", {}).items():
            if weight >= 3.0 and keyword in normalized: # Only consider high-weight keywords for this score
                absolute_critical_score += weight
        
        # Category-specific keyword priority score (from all weights)
        category_data = self.knowledge_base[category]
        keyword_priority_score = 0
        for word in normalized.split():
            if word in category_data["keywords"]:
                keyword_priority_score += category_data["keywords"][word]
        
        current_priority = Priority.LOW
        current_priority_score = 1.0 # Base score for LOW
        
        # Strongest indicators for priority: explicit text or very high keyword scores
        if explicit_impact_level == "Critical" or absolute_critical_score >= 3.0: # Check for critical keyword or explicit impact
            current_priority = Priority.CRITICAL
            current_priority_score = 4.5
            if explicit_impact_level == "Critical": factors.append(f"Explicit Impact Level: Critical")
            if absolute_critical_score >= 3.0: factors.append(f"Direct Critical indicators detected")
        elif explicit_impact_level == "Significant" or keyword_priority_score >= 4.0: # High priority for significant keywords
            current_priority = Priority.HIGH
            current_priority_score = 3.5
            if explicit_impact_level == "Significant": factors.append(f"Explicit Impact Level: Significant")
            if keyword_priority_score >= 4.0: factors.append(f"High severity indicators")
        elif explicit_impact_level == "Moderate" or keyword_priority_score >= 2.0: # Medium priority for some keywords
            current_priority = Priority.MEDIUM
            current_priority_score = 2.5
            if explicit_impact_level == "Moderate": factors.append(f"Explicit Impact Level: Moderate")
            if keyword_priority_score >= 2.0: factors.append(f"Moderate severity indicators")
        else: # Default or very low keyword match, or Minimal explicit impact
            current_priority = Priority.LOW
            current_priority_score = 1.5 # Still use 1.5 as base score for low
            if explicit_impact_level == "Minimal": factors.append(f"Explicit Impact Level: Minimal")
            factors.append(f"Low severity or general suggestion")
        
        return current_priority, current_priority_score, factors
    
    def calculate_confidence(self, category_scores: Dict[str, float], best_category: str) -> float:
        """Calculate classification confidence based on score separation and magnitude."""
        sorted_scores = sorted(category_scores.values(), reverse=True)
        
        # If no scores, or only one very low score
        if not sorted_scores or sorted_scores[0] == 0:
            return 0.1 # Very low confidence
        
        best_score = sorted_scores[0]
        
        # If only one category was considered or only one non-zero score
        if len(sorted_scores) < 2 or sorted_scores[1] == 0:
            # If the best score is high, it can be high confidence even without a competitor
            return min(best_score * 0.8 + 0.2, 1.0) # Scale magnitude, minimum 0.2
        
        # Confidence based on separation between top two scores AND magnitude of best score
        second_best_score = sorted_scores[1]
        
        # Calculate separation (relative difference)
        separation_factor = (best_score - second_best_score) / best_score
        
        # Combine separation and magnitude for a more nuanced confidence
        confidence = (separation_factor * 0.6) + (best_score * 0.4) # Adjust weights as needed
        
        return min(max(confidence, 0.1), 1.0) # Ensure confidence is between 0.1 and 1.0
    
    def classify(self, text: str) -> ClassificationResult:
        if not text or not text.strip():
            logger.warning("Classification error: Empty text provided.")
            return ClassificationResult(
                category="Other", priority=Priority.LOW.value, confidence=0.1,
                priority_score=1.0, matched_keywords=[], priority_factors=["Empty input"],
                error="Empty text provided"
            )
        
        try:
            # Calculate scores for all categories
            category_scores = {}
            all_matches = {}
            
            for category in self.knowledge_base.keys():
                score, matches = self.calculate_category_score(text, category)
                category_scores[category] = score
                all_matches[category] = matches
            
            # Find best category
            best_category = max(category_scores.keys(), key=lambda k: category_scores[k])
            
            # Re-evaluate best category if top score is very low, using simpler keyword inference
            if category_scores[best_category] < 0.05: # Even lower threshold for inference
                normalized = self.normalize_text(text)
                inferred_category = "Other" # Default if no strong keywords
                if any(word in normalized for word in ["safe", "danger", "hazard", "injury", "guard", "slippery"]):
                    inferred_category = "Safety Concern"
                elif any(word in normalized for word in ["machine", "equipment", "broken", "repair", "motor", "conveyor", "press"]):
                    inferred_category = "Machine/Equipment Issue"
                elif any(word in normalized for word in ["improve", "suggest", "idea", "better", "process", "workflow", "automate", "efficiency"]):
                    inferred_category = "Process Improvement Idea"
                best_category = inferred_category
                
            best_matches = all_matches.get(best_category, []) # Get matches for the final best category
            
            # Determine initial priority (before duplicate analysis)
            initial_priority_enum, initial_priority_score, priority_factors = self.determine_priority(
                text, best_category, category_scores.get(best_category, 0.0)
            )
            
            # Check for duplicates and potential escalation
            duplicate_analysis = self.duplicate_detector.analyze_submission(
                text, best_category, initial_priority_enum.value # Pass the initial priority value
            )
            
            # Apply escalation if needed (final priority comes from duplicate_analysis)
            final_priority = duplicate_analysis.escalated_priority
            
            # Adjust priority score and factors if escalation was applied
            if duplicate_analysis.escalation_applied and Priority(final_priority) == Priority.CRITICAL:
                # Set a high priority score for escalated criticals
                priority_score = 4.8 if initial_priority_enum != Priority.CRITICAL else initial_priority_score
                priority_factors.append(f"ESCALATED TO CRITICAL: {duplicate_analysis.similar_count} similar submissions detected within retention.")
                priority_factors.append(f"Original Priority: {duplicate_analysis.original_priority} → Final Priority: {final_priority}")
            else:
                priority_score = initial_priority_score # Use initial score if no escalation
                
            # Calculate final confidence
            confidence = self.calculate_confidence(category_scores, best_category)
            
            result = ClassificationResult(
                category=best_category,
                priority=final_priority,
                confidence=round(confidence, 3),
                priority_score=round(priority_score, 2),
                matched_keywords=best_matches[:6],
                priority_factors=priority_factors,
                duplicate_analysis=duplicate_analysis
            )
            logger.info(f"Classification Result: Category={result.category}, Priority={result.priority}, Confidence={result.confidence}, Duplicate_Applied={result.duplicate_analysis.escalation_applied if result.duplicate_analysis else 'N/A'}")
            return result
            
        except Exception as e:
            logger.error(f"Classification error: {e}", exc_info=True) # Log full traceback
            return ClassificationResult(
                category="Other", priority=Priority.LOW.value, confidence=0.1,
                priority_score=1.0, matched_keywords=[], priority_factors=["Error occurred"],
                error=str(e)
            )

classifier = FeedbackClassifier()

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
    if result.error:
        logger.warning(f"Classification warning: {result.error}")
    
    # Scale confidence to 1-10 range as requested
    confidence_10_scale = max(1, min(10, round(result.confidence * 10)))
    
    response = {
        "category": result.category,
        "autocategory": result.category,
        "priority": result.priority,
        "autopriority": result.priority,
        "confidence": confidence_10_scale,
        "confidence_score": confidence_10_scale, # Providing both for flexibility
        "priority_score": result.priority_score,
        "matched_keywords": result.matched_keywords,
        "priority_factors": result.priority_factors
    }
    
    # Add duplicate analysis if present
    if result.duplicate_analysis:
        response["duplicate_analysis"] = {
            "is_duplicate": result.duplicate_analysis.is_duplicate,
            "similar_count": result.duplicate_analysis.similar_count,
            "escalation_applied": result.duplicate_analysis.escalation_applied,
            "original_priority": result.duplicate_analysis.original_priority,
            "escalated_priority": result.duplicate_analysis.escalated_priority
        }
    logger.info(f"API Response: {response}") # Log the full response being sent
    return jsonify(response)

@app.route("/duplicate_stats", methods=['GET'])
def get_duplicate_stats():
    """Get statistics about submissions and duplicates"""
    classifier.duplicate_detector.clean_old_submissions() # Clean before reporting
    submissions = classifier.duplicate_detector.submissions
    
    # Count how many of the currently retained submissions are critical due to escalation
    # This requires re-classifying which might be inefficient for a large list,
    # but for debugging it's fine. For production, store escalation flag with submission.
    escalated_critical_count = 0
    for s in submissions:
        # Check if the stored priority is Critical AND if it was marked as escalated
        # We're relying on the 'priority_factors' string to indicate if it was an *escalation* to critical
        # rather than just being classified critical initially.
        temp_result = classifier.classify(s.text) # Re-classify to get factors
        if temp_result.priority == Priority.CRITICAL.value and \
           temp_result.duplicate_analysis and \
           temp_result.duplicate_analysis.escalation_applied:
            escalated_critical_count += 1
    
    response = {
        "total_submissions_retained": len(submissions),
        "escalated_critical_in_memory": escalated_critical_count,
        "retention_hours": classifier.duplicate_detector.retention_hours,
        "similarity_threshold": classifier.duplicate_detector.similarity_threshold,
        "escalation_threshold": classifier.duplicate_detector.escalation_threshold
    }
    logger.info(f"Duplicate Stats: {response}")
    return jsonify(response)

# Keep the existing home route and error handlers from original code
@app.route("/")
def home():
    return """
    <!DOCTYPE html>
    <html><head><title>Enhanced Feedback Classification API with Duplicate Detection</title>
    <style>body{font-family:Arial;margin:40px;background:#f5f5f5}
    .card{background:white;padding:30px;margin:20px 0;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,0.1)}
    h1{color:#333;text-align:center}h2{color:#4CAF50}
    .endpoint{background:#f8f8f8;padding:15px;border-left:4px solid #4CAF50;margin:10px 0}
    .feature{background:#e8f5e8;padding:10px;margin:5px 0;border-radius:5px}
    </style></head><body>
    <h1>🧠 Enhanced Feedback Classification API</h1>
    <div class="card"><h2>🆕 NEW: Duplicate Detection & Auto-Escalation</h2>
    <div class="feature">✨ Detects similar/duplicate submissions automatically</div>
    <div class="feature">⬆️ Escalates LOW/MEDIUM issues to CRITICAL when reported 2+ times</div>
    <div class="feature">🔍 Uses advanced text similarity matching (60% threshold)</div>
    <div class="feature">⏰ Tracks submissions for 1 week (configurable)</div>
    </div>
    
    <div class="card"><h2>📡 API Endpoints</h2>
    <div class="endpoint"><strong>POST /classify</strong><br>
    Submit feedback text for classification with duplicate detection<br>
    <code>{"text": "The conveyor belt is broken again"}</code></div>
    
    <div class="endpoint"><strong>GET /duplicate_stats</strong><br>
    Get statistics about duplicate detection and escalations</div>
    </div>
    
    <div class="card"><h2>🎯 How Duplicate Detection Works</h2>
    <p><strong>Step 1:</strong> System analyzes text similarity using advanced algorithms</p>
    <p><strong>Step 2:</strong> If 2+ similar submissions found with LOW/MEDIUM priority</p>
    <p><strong>Step 3:</strong> Automatically escalates to CRITICAL priority</p>
    <p><strong>Step 4:</strong> Returns detailed analysis of duplicate detection</p>
    </div>
    
    <div class="card"><h2>📊 Enhanced Response Format</h2>
    <div class="endpoint">{<br>
    &nbsp;&nbsp;"priority": "Critical",<br>
    &nbsp;&nbsp;"priority_factors": ["ESCALATED TO CRITICAL: 3 similar submissions detected"],<br>
    &nbsp;&nbsp;"duplicate_analysis": {<br>
    &nbsp;&nbsp;&nbsp;&nbsp;"is_duplicate": true,<br>
    &nbsp;&nbsp;&nbsp;&nbsp;"similar_count": 3,<br>
    &nbsp;&nbsp;&nbsp;&nbsp;"escalation_applied": true,<br>
    &nbsp;&nbsp;&nbsp;&nbsp;"original_priority": "Low",<br>
    &nbsp;&nbsp;&nbsp;&nbsp;"escalated_priority": "Critical"<br>
    &nbsp;&nbsp;}<br>
    }</div>
    </div>
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
