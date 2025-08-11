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

logging.basicConfig(level=logging.INFO)
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
            # Critical safety indicators (weight: 3.0)
            "emergency": 3.0, "fire": 3.0, "explosion": 3.0, "toxic": 3.0, "unconscious": 3.0, 
            "trapped": 3.0, "evacuation": 3.0, "severe": 3.0, "blood": 3.0, "collapsed": 3.0,
            
            # High priority safety (weight: 2.0)
            "danger": 2.0, "hazard": 2.0, "unsafe": 2.0, "injury": 2.0, "accident": 2.0, 
            "hurt": 2.0, "injured": 2.0, "fall": 2.0, "cut": 2.0, "burn": 2.0, 
            "electrical": 2.0, "shock": 2.0, "blocked": 2.0, "spill": 2.0, "leak": 2.0,
            
            # Medium priority safety (weight: 1.0)
            "safety": 1.0, "risk": 1.0, "ppe": 1.0, "protective": 1.0, "guard": 1.0, 
            "warning": 1.0, "caution": 1.0, "helmet": 1.0, "gloves": 1.0, "training": 1.0,
        },
        "negation_words": ["not", "no", "without", "lacking", "need", "should", "could", "want", "wish", "suggest"],
        "context_reducers": ["suggestion", "idea", "recommend", "propose", "think", "maybe", "could", "should"]
    },
    
    "Machine/Equipment Issue": {
        "keywords": {
            # Critical equipment issues (weight: 3.0)
            "explosion": 3.0, "fire": 3.0, "complete failure": 3.0, "shutdown": 3.0, 
            "total loss": 3.0, "major breakdown": 3.0, "catastrophic": 3.0,
            
            # High priority equipment (weight: 2.0)
            "broken": 2.0, "malfunction": 2.0, "down": 2.0, "stopped": 2.0, "jam": 2.0,
            "stuck": 2.0, "overheating": 2.0, "failure": 2.0, "error": 2.0, "crash": 2.0,
            
            # Medium priority equipment (weight: 1.0)
            "machine": 1.0, "equipment": 1.0, "conveyor": 1.0, "motor": 1.0, "pump": 1.0,
            "repair": 1.0, "maintenance": 1.0, "noise": 1.0, "vibration": 1.0,
        },
        "negation_words": ["not", "no", "without", "need", "should", "could", "want", "suggest"],
        "context_reducers": ["suggestion", "idea", "recommend", "propose", "schedule", "plan"]
    },
    
    "Process Improvement Idea": {
        "keywords": {
            "automate": 2.0, "streamline": 2.0, "optimize": 2.0, "revolutionize": 2.0,
            "improve": 1.0, "efficiency": 1.0, "productivity": 1.0, "enhance": 1.0,
            "suggestion": 0.5, "idea": 0.5, "recommend": 0.5, "think": 0.5,
        },
        "negation_words": [],
        "context_reducers": []
    },
    
    "Other": {
        "keywords": {
            "supplies": 1.0, "training": 1.0, "lighting": 1.0, "parking": 1.0, 
            "temperature": 1.0, "break": 1.0, "lunch": 1.0, "bathroom": 1.0, 
            "coffee": 1.0, "clean": 1.0, "organize": 1.0, "facilities": 1.0
        },
        "negation_words": ["urgent", "immediate", "critical", "emergency"],
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
    def __init__(self, similarity_threshold=0.75, escalation_threshold=3, retention_hours=168):
        self.similarity_threshold = similarity_threshold
        self.escalation_threshold = escalation_threshold
        self.retention_hours = retention_hours
        self.submissions: List[SubmissionRecord] = []
        
        # Escalation rules: LOW/MEDIUM → CRITICAL when repeated
        self.escalation_rules = {
            Priority.LOW: Priority.CRITICAL,
            Priority.MEDIUM: Priority.CRITICAL,
            Priority.HIGH: Priority.CRITICAL,
            Priority.CRITICAL: Priority.CRITICAL
        }
    
    def clean_old_submissions(self):
        """Remove submissions older than retention period"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        self.submissions = [s for s in self.submissions if s.timestamp > cutoff_time]
    
    def generate_content_hash(self, text: str) -> str:
        """Generate hash for exact duplicate detection"""
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        norm1 = re.sub(r'\s+', ' ', text1.lower().strip())
        norm2 = re.sub(r'\s+', ' ', text2.lower().strip())
        
        # Use sequence matcher for overall similarity
        sequence_sim = difflib.SequenceMatcher(None, norm1, norm2).ratio()
        
        # Use word overlap (Jaccard similarity)
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
        
        # Weighted combination
        return (sequence_sim * 0.4) + (word_sim * 0.6)
    
    def find_similar_submissions(self, new_text: str, new_category: str) -> List[SubmissionRecord]:
        """Find submissions similar to the new one"""
        similar = []
        new_hash = self.generate_content_hash(new_text)
        
        for existing in self.submissions:
            # Check exact match first
            if new_hash == existing.submission_hash:
                similar.append(existing)
                continue
            
            # Skip if completely different categories (unless safety/equipment)
            if not self.categories_compatible(new_category, existing.category):
                continue
            
            # Calculate similarity
            similarity = self.calculate_similarity(new_text, existing.text)
            if similarity >= self.similarity_threshold:
                similar.append(existing)
        
        return similar
    
    def categories_compatible(self, cat1: str, cat2: str) -> bool:
        """Check if categories are compatible for duplicate detection"""
        if cat1 == cat2:
            return True
        # Safety and equipment issues can be related
        safety_equipment = ["Safety Concern", "Machine/Equipment Issue"]
        return cat1 in safety_equipment and cat2 in safety_equipment
    
    def analyze_submission(self, text: str, category: str, priority: str) -> DuplicateAnalysis:
        """Analyze submission for duplicates and escalation"""
        self.clean_old_submissions()
        
        # Find similar submissions
        similar_submissions = self.find_similar_submissions(text, category)
        
        # Count submissions that should trigger escalation
        escalatable_count = len([s for s in similar_submissions 
                               if s.priority in [Priority.LOW.value, Priority.MEDIUM.value]])
        
        # Determine if escalation is needed
        should_escalate = escalatable_count >= self.escalation_threshold
        escalated_priority = priority
        
        if should_escalate and Priority(priority) in self.escalation_rules:
            escalated_priority = Priority.CRITICAL.value
        
        # Store this submission
        submission = SubmissionRecord(
            text=text,
            category=category,
            priority=escalated_priority,
            timestamp=datetime.now(),
            submission_hash=self.generate_content_hash(text)
        )
        self.submissions.append(submission)
        
        return DuplicateAnalysis(
            is_duplicate=len(similar_submissions) > 0,
            similar_count=len(similar_submissions),
            escalation_applied=should_escalate,
            original_priority=priority,
            escalated_priority=escalated_priority
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
                
                # Check for negation context
                negation_factor = 1.0
                for j in range(max(0, i-3), min(len(words), i+4)):
                    if words[j] in negation_words:
                        negation_factor = 0.3
                        break
                
                # Check for context reducers
                context_factor = 1.0
                if any(reducer in normalized for reducer in context_reducers):
                    context_factor = 0.6
                
                adjusted_weight = weight * negation_factor * context_factor
                total_score += adjusted_weight
                matched_keywords.append(word)
        
        # Normalize score
        max_possible = sum(keywords.values()) * 0.3
        normalized_score = min(total_score / max_possible, 1.0) if max_possible > 0 else 0.0
        
        return normalized_score, matched_keywords
    
    def determine_priority(self, text: str, category: str, category_score: float) -> Tuple[Priority, float, List[str]]:
        """Determine priority based on content analysis"""
        normalized = self.normalize_text(text)
        factors = []
        
        # Check for critical indicators
        critical_words = ["emergency", "immediate", "urgent", "fire", "explosion", "dangerous", "fatal"]
        critical_score = sum(1 for word in critical_words if word in normalized)
        
        # Check category-specific priority indicators
        category_data = self.knowledge_base[category]
        keyword_priority_score = 0
        for word in normalized.split():
            if word in category_data["keywords"]:
                keyword_priority_score += category_data["keywords"][word]
        
        if critical_score > 0:
            priority = Priority.CRITICAL
            priority_score = 4.5
            factors.append(f"Critical indicators detected")
        elif keyword_priority_score >= 4.0:
            priority = Priority.HIGH
            priority_score = 3.5
            factors.append(f"High severity indicators")
        elif keyword_priority_score >= 2.0:
            priority = Priority.MEDIUM
            priority_score = 2.5
            factors.append(f"Moderate severity indicators")
        else:
            priority = Priority.LOW
            priority_score = 1.5
            factors.append(f"Low severity or suggestion")
        
        return priority, priority_score, factors
    
    def calculate_confidence(self, category_scores: Dict[str, float], best_category: str) -> float:
        """Calculate classification confidence"""
        sorted_scores = sorted(category_scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] > 0:
            separation = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
            return min(max(separation * category_scores[best_category], 0.1), 1.0)
        return 0.3
    
    def classify(self, text: str) -> ClassificationResult:
        if not text or not text.strip():
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
            best_matches = all_matches[best_category]
            
            # Handle low scores with keyword inference
            if category_scores[best_category] < 0.15:
                normalized = self.normalize_text(text)
                if any(word in normalized for word in ["safe", "danger", "hazard", "injury"]):
                    best_category = "Safety Concern"
                elif any(word in normalized for word in ["machine", "equipment", "broken", "repair"]):
                    best_category = "Machine/Equipment Issue"
                elif any(word in normalized for word in ["improve", "suggest", "idea", "better"]):
                    best_category = "Process Improvement Idea"
                else:
                    best_category = "Other"
            
            # Determine initial priority
            priority, priority_score, priority_factors = self.determine_priority(
                text, best_category, category_scores[best_category]
            )
            
            # Check for duplicates and potential escalation
            duplicate_analysis = self.duplicate_detector.analyze_submission(
                text, best_category, priority.value
            )
            
            # Apply escalation if needed
            final_priority = duplicate_analysis.escalated_priority
            if duplicate_analysis.escalation_applied:
                priority_score = 4.8  # Critical priority score
                priority_factors.append(f"ESCALATED TO CRITICAL: {duplicate_analysis.similar_count} similar submissions detected")
                priority_factors.append(f"Original: {duplicate_analysis.original_priority} → Final: {final_priority}")
            
            # Calculate confidence
            confidence = self.calculate_confidence(category_scores, best_category)
            
            return ClassificationResult(
                category=best_category,
                priority=final_priority,
                confidence=round(confidence, 3),
                priority_score=round(priority_score, 2),
                matched_keywords=best_matches[:6],
                priority_factors=priority_factors,
                duplicate_analysis=duplicate_analysis
            )
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
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
        return jsonify({"error": "Missing 'text' field"}), 400
    
    text = data.get('text', '')
    if len(text) > MAX_TEXT_LENGTH:
        return jsonify({"error": f"Text exceeds {MAX_TEXT_LENGTH} characters"}), 400
    
    result = classifier.classify(text)
    if result.error:
        logger.warning(f"Classification warning: {result.error}")
    
    confidence_10_scale = max(1, min(10, round(result.confidence * 10)))
    
    response = {
        "category": result.category,
        "autocategory": result.category,
        "priority": result.priority,
        "autopriority": result.priority,
        "confidence": confidence_10_scale,
        "confidence_score": confidence_10_scale,
        "priority_score": result.priority_score,
        "matched_keywords": result.matched_keywords,
        "priority_factors": result.priority_factors
    }
    
    # Add duplicate analysis
    if result.duplicate_analysis:
        response["duplicate_analysis"] = {
            "is_duplicate": result.duplicate_analysis.is_duplicate,
            "similar_count": result.duplicate_analysis.similar_count,
            "escalation_applied": result.duplicate_analysis.escalation_applied,
            "original_priority": result.duplicate_analysis.original_priority,
            "escalated_priority": result.duplicate_analysis.escalated_priority
        }
    
    return jsonify(response)

@app.route("/duplicate_stats", methods=['GET'])
def get_duplicate_stats():
    """Get statistics about submissions and duplicates"""
    classifier.duplicate_detector.clean_old_submissions()
    submissions = classifier.duplicate_detector.submissions
    
    return jsonify({
        "total_submissions": len(submissions),
        "escalated_count": len([s for s in submissions if s.priority == "Critical"]),
        "retention_hours": classifier.duplicate_detector.retention_hours,
        "similarity_threshold": classifier.duplicate_detector.similarity_threshold,
        "escalation_threshold": classifier.duplicate_detector.escalation_threshold
    })

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
    <div class="feature">⬆️ Escalates LOW/MEDIUM issues to CRITICAL when reported 3+ times</div>
    <div class="feature">🔍 Uses advanced text similarity matching (75% threshold)</div>
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
    <p><strong>Step 2:</strong> If 3+ similar submissions found with LOW/MEDIUM priority</p>
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
    logger.error(f"Internal error: {error}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
