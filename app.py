import os
import re
import math
import difflib
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

# ==============================================================================
# CONFIGURATION AND DATA
# ==============================================================================

class Priority(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

# Enhanced knowledge base with more comprehensive keywords and patterns
KNOWLEDGE_BASE = [
    {
        "category": "Safety Concern",
        "keywords": ["safety", "emergency", "danger", "hazard", "risk", "injury", "accident", "unsafe", "spill", "leak", "fire", "toxic", "chemical", "guard", "protective", "first aid", "evacuation", "blocked", "exit", "warning", "slip", "fall", "cut", "burn", "electrical", "shock", "ppe", "helmet", "gloves", "goggles", "mask", "emergency stop", "guard rail", "loose", "unstable", "tripping", "crush", "exposed", "sharp", "hot", "hurt", "injured", "pain", "blood", "broken bone", "unconscious", "trapped", "smoking", "sparks", "gas", "fumes", "ventilation", "lockout", "tagout", "confined space"],
        "examples": ["Emergency exit is blocked", "Chemical spill in storage area", "Guard rail is loose and unsafe", "Workers not wearing proper PPE", "Electrical cord is frayed", "Floor is wet without warning signs", "Safety valve is not working", "Emergency stop button broken"]
    },
    {
        "category": "Machine/Equipment Issue", 
        "keywords": ["machine", "equipment", "conveyor", "motor", "pump", "compressor", "hydraulic", "mechanical", "broken", "malfunction", "jam", "stuck", "slow", "fast", "vibration", "noise", "leak", "pressure", "temperature", "maintenance", "repair", "belt", "chain", "gear", "bearing", "sensor", "overheating", "grinding", "slipping", "down", "stopped", "failure", "error", "alarm", "warning light", "calibration", "adjustment", "replacement", "part", "component", "system", "control", "panel", "display", "indicator", "gauge", "meter", "valve", "cylinder", "pneumatic", "electrical", "wiring", "connection", "coupling", "shaft", "pulley", "roller", "switch", "button", "lever", "handle", "tool", "die", "mold", "cutting", "drilling", "welding", "assembly", "production line", "workstation"],
        "examples": ["Conveyor belt is slipping badly", "Machine making loud grinding noise", "Equipment keeps jamming frequently", "Motor is overheating during operation", "Hydraulic system has slow response", "Sensor readings seem inaccurate", "Production line is down", "Control panel showing errors"]
    },
    {
        "category": "Process Improvement Idea",
        "keywords": ["improve", "optimize", "efficiency", "productivity", "streamline", "automate", "reduce", "increase", "faster", "better", "easier", "suggestion", "idea", "recommend", "proposal", "enhancement", "upgrade", "workflow", "process", "lean", "5s", "kaizen", "training", "schedule", "organize", "standardize", "best practice", "innovation", "quality", "cost", "saving", "time", "method", "procedure", "system", "layout", "design", "ergonomic", "user friendly", "simplify", "consolidate", "eliminate", "waste", "bottleneck", "flow", "throughput", "cycle time", "setup", "changeover", "cross train", "multiskill", "flexibility", "capacity", "utilization"],
        "examples": ["Could we automate this repetitive task", "Suggest implementing 5S system", "Optimize the loading dock schedule", "Cross-train operators for flexibility", "Streamline the packaging process", "What if we rearranged workspace layout", "Implement lean manufacturing principles", "Reduce setup time for efficiency"]
    },
    {
        "category": "Other",
        "keywords": ["supplies", "training", "lighting", "breakroom", "microwave", "parking", "cafeteria", "recognition", "materials", "facilities", "comfort", "ergonomic", "temperature", "air conditioning", "bathroom", "cleanliness", "coffee", "water", "snacks", "chair", "desk", "computer", "software", "phone", "communication", "meeting", "policy", "procedure", "documentation", "forms", "paperwork", "inventory", "storage", "organization", "housekeeping", "maintenance", "janitorial", "landscaping", "security", "access", "badge", "uniform", "locker", "shift", "overtime", "schedule", "break", "lunch", "benefits", "payroll", "hr", "human resources"],
        "examples": ["Break room needs more supplies", "Parking lot has too many potholes", "Could we get better coffee", "Bathroom needs more frequent cleaning", "Temperature is too cold in office", "Need ergonomic assessment for workstation", "Request for additional training", "Improve communication between shifts"]
    }
]

# Enhanced priority keywords with more specific safety and equipment terms
PRIORITY_KEYWORDS = {
    "critical": ["emergency", "fire", "explosion", "toxic", "gas leak", "electrical shock", "trapped", "unconscious", "severe injury", "ambulance", "evacuation", "hazmat", "complete failure", "plant shutdown", "production stopped", "major breakdown", "system failure", "immediate danger", "life threatening", "critical failure", "total loss", "complete stoppage"],
    "high": ["danger", "hazard", "unsafe", "blocked exit", "exposed wire", "overheating", "smoking", "broken", "won't start", "major leak", "production stopped", "emergency stop", "high pressure", "high temperature", "significant risk", "equipment down", "major malfunction", "urgent repair", "safety risk", "injury risk", "broken guard", "missing safety", "sparks", "unusual noise", "vibration", "unstable"],
    "medium": ["worn", "loose", "slow", "inconsistent", "needs repair", "minor leak", "calibration", "training needed", "efficiency", "productivity", "maintenance due", "adjustment needed", "replacement due", "intermittent", "degraded performance", "reduced capacity", "quality issue", "process variation", "minor problem", "scheduled maintenance"],
    "low": ["suggestion", "idea", "recommend", "could we", "supplies", "comfort", "convenience", "nice to have", "future", "enhancement", "improvement", "optimize", "better way", "consider", "maybe", "perhaps", "eventually", "when possible", "minor issue", "cosmetic"]
}

# Lower similarity threshold for better detection
SIMILARITY_THRESHOLD = 0.08
MAX_TEXT_LENGTH = 5000

# ==============================================================================
# CORE CLASSES  
# ==============================================================================

@dataclass
class ClassificationResult:
    category: str
    priority: str
    confidence: float
    priority_score: float
    matched_example: Optional[str]
    keyword_matches: List[str]
    priority_factors: List[str]
    similarity_scores: Dict[str, float]
    error: Optional[str] = None

class TextProcessor:
    @staticmethod
    def normalize_text(text: str) -> str:
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\-\']', ' ', text)
        return text

    @staticmethod
    def extract_keywords(text: str) -> Set[str]:
        words = TextProcessor.normalize_text(text).split()
        stop_words = {'the', 'is', 'at', 'on', 'and', 'a', 'to', 'it', 'in', 'for', 'of', 'i', 'you', 'he', 'she', 'we', 'they', 'this', 'that', 'with', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'an', 'as', 'be', 'by', 'from', 'up', 'out', 'so', 'if', 'no', 'not', 'only', 'own', 'same', 'such', 'than', 'too', 'very', 'just', 'now', 'also', 'its', 'our'}
        return {word for word in words if len(word) > 2 and word not in stop_words}

    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        # Enhanced similarity calculation with fuzzy matching
        seq_ratio = difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        
        words1 = set(TextProcessor.normalize_text(text1).split())
        words2 = set(TextProcessor.normalize_text(text2).split())
        if not words1 or not words2:
            return seq_ratio
        
        # Jaccard similarity
        jaccard = len(words1.intersection(words2)) / len(words1.union(words2))
        
        # Fuzzy word matching for partial matches
        fuzzy_matches = 0
        for word1 in words1:
            for word2 in words2:
                if len(word1) > 3 and len(word2) > 3:
                    word_sim = difflib.SequenceMatcher(None, word1, word2).ratio()
                    if word_sim > 0.8:
                        fuzzy_matches += 1
                        break
        
        fuzzy_score = fuzzy_matches / max(len(words1), len(words2)) if max(len(words1), len(words2)) > 0 else 0
        
        return (seq_ratio * 0.3) + (jaccard * 0.5) + (fuzzy_score * 0.2)

class PriorityEngine:
    def __init__(self):
        self.priority_keywords = PRIORITY_KEYWORDS

    def calculate_priority(self, text: str, category: str, text_keywords: Set[str]) -> Tuple[Priority, float, List[str]]:
        normalized_text = text.lower()
        base_score = 0.3  # Lower starting point
        priority_factors = []
        
        # Enhanced keyword matching with partial matching
        critical_matches = self._find_matches_enhanced(normalized_text, text_keywords, self.priority_keywords["critical"])
        if critical_matches:
            base_score += 5.0  # Strong boost for critical
            priority_factors.extend([f"Critical: {match}" for match in critical_matches[:3]])
        
        high_matches = self._find_matches_enhanced(normalized_text, text_keywords, self.priority_keywords["high"])
        if high_matches:
            base_score += 3.0  # Good boost for high
            priority_factors.extend([f"High: {match}" for match in high_matches[:3]])
        
        medium_matches = self._find_matches_enhanced(normalized_text, text_keywords, self.priority_keywords["medium"])
        if medium_matches:
            base_score += 1.8  # Moderate boost for medium
            priority_factors.extend([f"Medium: {match}" for match in medium_matches[:2]])
        
        low_matches = self._find_matches_enhanced(normalized_text, text_keywords, self.priority_keywords["low"])
        if low_matches:
            base_score += 0.5  # Small boost for low
            priority_factors.extend([f"Low: {match}" for match in low_matches[:2]])

        # Enhanced category multipliers
        category_multiplier = self._get_category_multiplier(category)
        if category_multiplier != 1.0:
            base_score *= category_multiplier
            priority_factors.append(f"Category: {category} (×{category_multiplier})")

        # Check urgency indicators
        urgency_multiplier = self._check_urgency(normalized_text)
        if urgency_multiplier != 1.0:
            base_score *= urgency_multiplier
            priority_factors.append(f"Urgency (×{urgency_multiplier})")

        # Enhanced minimum scores for safety and equipment
        if category == "Safety Concern" and base_score < 2.5:
            base_score = 2.5
            priority_factors.append("Minimum priority for safety concern")
        elif category == "Machine/Equipment Issue" and base_score < 1.8:
            base_score = 1.8
            priority_factors.append("Minimum priority for equipment issue")

        if not priority_factors:
            priority_factors.append("Default priority assignment")

        return self._score_to_priority(base_score), round(base_score, 2), priority_factors

    def _find_matches_enhanced(self, text: str, text_keywords: Set[str], priority_keywords: List[str]) -> List[str]:
        matches = []
        for keyword in priority_keywords:
            # Direct text match
            if keyword in text:
                matches.append(keyword)
                continue
            
            # Keyword intersection
            if any(word in text_keywords for word in keyword.split()):
                matches.append(keyword)
                continue
            
            # Fuzzy matching for important terms
            for word in keyword.split():
                if len(word) > 4:  # Only fuzzy match longer words
                    for text_word in text_keywords:
                        if len(text_word) > 4 and difflib.SequenceMatcher(None, word, text_word).ratio() > 0.85:
                            matches.append(keyword)
                            break
        return matches

    def _get_category_multiplier(self, category: str) -> float:
        multipliers = {
            "Safety Concern": 1.8,      # Higher for safety
            "Machine/Equipment Issue": 1.4,  # Higher for equipment 
            "Process Improvement Idea": 0.8,
            "Other": 0.7
        }
        return multipliers.get(category, 1.0)

    def _check_urgency(self, text: str) -> float:
        immediate_indicators = ["now", "immediately", "asap", "urgent", "emergency", "right away", "critical", "stop", "shutdown"]
        soon_indicators = ["soon", "quickly", "today", "this week", "priority", "important"]
        
        if any(indicator in text for indicator in immediate_indicators):
            return 1.3
        elif any(indicator in text for indicator in soon_indicators):
            return 1.15
        return 1.0

    def _score_to_priority(self, score: float) -> Priority:
        # Adjusted thresholds for better distribution
        if score >= 5.0:
            return Priority.CRITICAL
        elif score >= 3.5:
            return Priority.HIGH
        elif score >= 2.0:
            return Priority.MEDIUM
        else:
            return Priority.LOW

class FeedbackClassifier:
    def __init__(self, knowledge_base: List[Dict], similarity_threshold: float = SIMILARITY_THRESHOLD):
        self.knowledge_base = knowledge_base
        self.similarity_threshold = similarity_threshold
        self.text_processor = TextProcessor()
        self.priority_engine = PriorityEngine()

    def classify(self, text: str) -> ClassificationResult:
        if not text or not text.strip():
            return ClassificationResult(
                category="Other", priority=Priority.LOW.value, confidence=0.0,
                priority_score=0.0, matched_example=None, keyword_matches=[],
                priority_factors=[], similarity_scores={}, error="Empty text provided"
            )

        try:
            text_keywords = self.text_processor.extract_keywords(text)
            scores = {}
            best_examples = {}

            # Enhanced scoring for each category
            for item in self.knowledge_base:
                category = item['category']
                
                # Enhanced keyword matching with partial matching
                category_keywords = set(item.get('keywords', []))
                keyword_matches = text_keywords.intersection(category_keywords)
                
                # Fuzzy keyword matching for better detection
                fuzzy_keyword_matches = self._fuzzy_keyword_match(text_keywords, category_keywords)
                all_matches = keyword_matches.union(fuzzy_keyword_matches)
                
                # Improved keyword scoring
                keyword_score = len(all_matches) / max(len(category_keywords), 1) if category_keywords else 0.0
                
                # Boost keyword score for safety and equipment categories
                if category in ["Safety Concern", "Machine/Equipment Issue"] and keyword_score > 0:
                    keyword_score *= 1.3
                
                # Example similarity score
                example_scores = [self.text_processor.calculate_similarity(text, example) 
                                for example in item['examples']]
                best_example_score = max(example_scores) if example_scores else 0.0
                best_example = item['examples'][example_scores.index(best_example_score)] if example_scores else None
                
                # Enhanced combined score calculation
                if keyword_score > 0 and best_example_score > 0:
                    # Both keyword and example matches - strong indicator
                    combined_score = (keyword_score * 0.6) + (best_example_score * 0.4)
                    combined_score *= 1.4  # Boost for having both types of matches
                elif keyword_score > 0:
                    # Keyword matches only
                    combined_score = keyword_score * 0.8
                elif best_example_score > 0:
                    # Example matches only
                    combined_score = best_example_score * 0.7
                else:
                    combined_score = 0.0
                
                scores[category] = combined_score
                best_examples[category] = best_example

            # Find best category with improved logic
            best_category = max(scores.keys(), key=lambda k: scores[k])
            confidence = scores[best_category]

            # Enhanced confidence and category assignment
            if confidence < self.similarity_threshold:
                # Check if it's a safety or equipment issue that got missed
                safety_indicators = ["safety", "danger", "hazard", "unsafe", "emergency", "injury", "accident", "risk"]
                equipment_indicators = ["machine", "equipment", "broken", "malfunction", "down", "stopped", "repair", "maintenance"]
                
                text_lower = text.lower()
                if any(indicator in text_lower for indicator in safety_indicators):
                    category_name = "Safety Concern"
                    confidence = 0.6  # Assign reasonable confidence for safety
                elif any(indicator in text_lower for indicator in equipment_indicators):
                    category_name = "Machine/Equipment Issue"
                    confidence = 0.6  # Assign reasonable confidence for equipment
                else:
                    category_name = "Other"
                    confidence = min(0.5, confidence + 0.2)
                
                matched_example = None
                keyword_matches = []
            else:
                category_name = best_category
                matched_example = best_examples[best_category]
                item = next(i for i in self.knowledge_base if i['category'] == category_name)
                
                # Get both exact and fuzzy keyword matches
                exact_matches = text_keywords.intersection(set(item.get('keywords', [])))
                fuzzy_matches = self._fuzzy_keyword_match(text_keywords, set(item.get('keywords', [])))
                keyword_matches = list(exact_matches.union(fuzzy_matches))
                
                # Enhanced confidence boost
                if confidence > 0.2:
                    confidence = min(1.0, confidence * 1.5)

            # Calculate priority
            priority, priority_score, priority_factors = self.priority_engine.calculate_priority(
                text, category_name, text_keywords
            )

            return ClassificationResult(
                category=category_name,
                priority=priority.value,
                confidence=round(confidence, 3),
                priority_score=priority_score,
                matched_example=matched_example,
                keyword_matches=keyword_matches[:5],  # Limit to top 5 matches
                priority_factors=priority_factors,
                similarity_scores={k: round(v, 3) for k, v in scores.items()}
            )

        except Exception as e:
            logger.error(f"Classification error: {e}", exc_info=True)
            return ClassificationResult(
                category="Other", priority=Priority.LOW.value, confidence=0.0,
                priority_score=0.0, matched_example=None, keyword_matches=[],
                priority_factors=[], similarity_scores={}, 
                error="Classification error occurred"
            )

    def _fuzzy_keyword_match(self, text_keywords: Set[str], category_keywords: Set[str]) -> Set[str]:
        """Find fuzzy matches between text keywords and category keywords"""
        fuzzy_matches = set()
        for text_word in text_keywords:
            for cat_word in category_keywords:
                if len(text_word) > 3 and len(cat_word) > 3:
                    similarity = difflib.SequenceMatcher(None, text_word, cat_word).ratio()
                    if similarity > 0.8:
                        fuzzy_matches.add(cat_word)
        return fuzzy_matches

# ==============================================================================
# FLASK ROUTES
# ==============================================================================

classifier = FeedbackClassifier(KNOWLEDGE_BASE)

def validate_request_data(data: Optional[Dict]) -> Tuple[bool, str]:
    if not data:
        return False, "No JSON data provided"
    if 'text' not in data:
        return False, "Missing 'text' field"
    text = data.get('text', '')
    if not isinstance(text, str):
        return False, "'text' must be a string"
    if len(text) > MAX_TEXT_LENGTH:
        return False, f"Text exceeds {MAX_TEXT_LENGTH} characters"
    return True, ""

@app.route("/classify", methods=['POST'])
def handle_classify():
    data = request.get_json()
    is_valid, error_message = validate_request_data(data)
    if not is_valid:
        return jsonify({"error": error_message}), 400

    result = classifier.classify(data.get("text", ""))
    if result.error:
        return jsonify({"error": result.error}), 500

    # Enhanced confidence to 0-10 scale with better distribution
    confidence_raw = result.confidence
    if confidence_raw >= 0.8:
        confidence_int = 10
    elif confidence_raw >= 0.6:
        confidence_int = 8 + int((confidence_raw - 0.6) * 10)
    elif confidence_raw >= 0.4:
        confidence_int = 6 + int((confidence_raw - 0.4) * 10)
    elif confidence_raw >= 0.2:
        confidence_int = 4 + int((confidence_raw - 0.2) * 10)
    else:
        confidence_int = max(1, int(confidence_raw * 20))

    return jsonify({
        "category": result.category,
        "autocategory": result.category,
        "priority": result.priority,
        "autopriority": result.priority,
        "confidence": confidence_int,
        "confidence_score": confidence_int,
        "priority_score": result.priority_score,
        "matched_example": result.matched_example,
        "keyword_matches": result.keyword_matches,
        "priority_factors": result.priority_factors,
        "similarity_scores": result.similarity_scores,
        "error": result.error,
    })

@app.route("/")
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Feedback Classification API</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                line-height: 1.6;
                color: #1f2937;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 2rem;
            }
            
            .header {
                text-align: center;
                color: white;
                margin-bottom: 3rem;
            }
            
            .header h1 {
                font-size: 3rem;
                font-weight: 700;
                margin-bottom: 1rem;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
            
            .header p {
                font-size: 1.25rem;
                opacity: 0.9;
                font-weight: 300;
            }
            
            .status-badge {
                display: inline-block;
                background: rgba(255,255,255,0.2);
                backdrop-filter: blur(10px);
                padding: 0.75rem 1.5rem;
                border-radius: 50px;
                margin-top: 1.5rem;
                border: 1px solid rgba(255,255,255,0.3);
                font-weight: 500;
            }
            
            .card {
                background: rgba(255,255,255,0.95);
                backdrop-filter: blur(20px);
                border-radius: 20px;
                padding: 2.5rem;
                margin-bottom: 2rem;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                border: 1px solid rgba(255,255,255,0.2);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 30px 60px rgba(0,0,0,0.15);
            }
            
            .card h2 {
                color: #4f46e5;
                font-size: 1.875rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }
            
            .card h3 {
                color: #374151;
                font-size: 1.25rem;
                font-weight: 600;
                margin: 1.5rem 0 1rem 0;
            }
            
            .endpoint {
                background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
                color: white;
                padding: 1rem 1.5rem;
                border-radius: 12px;
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 1.1rem;
                margin: 1rem 0;
                display: inline-block;
                font-weight: 500;
            }
            
            .code-block {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 0.9rem;
                overflow-x: auto;
                line-height: 1.6;
            }
            
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 1.5rem;
                margin: 2rem 0;
            }
            
            .priority-card {
                padding: 1.5rem;
                border-radius: 16px;
                text-align: center;
                transition: transform 0.2s ease;
                position: relative;
                overflow: hidden;
            }
            
            .priority-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: var(--accent-color);
            }
            
            .priority-card:hover {
                transform: translateY(-3px);
            }
            
            .critical {
                background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
                --accent-color: #dc2626;
                color: #7f1d1d;
            }
            
            .high {
                background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
                --accent-color: #d97706;
                color: #92400e;
            }
            
            .medium {
                background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
                --accent-color: #16a34a;
                color: #14532d;
            }
            
            .low {
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                --accent-color: #64748b;
                color: #334155;
            }
            
            .category-card {
                background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
                padding: 1.5rem;
                border-radius: 16px;
                text-align: center;
                border: 2px solid #cbd5e1;
                transition: all 0.3s ease;
            }
            
            .category-card:hover {
                border-color: #4f46e5;
                transform: translateY(-3px);
                box-shadow: 0 10px 25px rgba(79, 70, 229, 0.1);
            }
            
            .icon {
                font-size: 1.5rem;
                margin-right: 0.5rem;
            }
            
            footer {
                text-align: center;
                padding: 2rem;
                color: rgba(255,255,255,0.8);
                font-size: 0.9rem;
                margin-top: 3rem;
            }
            
            /* Hidden watermark */
            .watermark {
                position: fixed;
                bottom: 5px;
                right: 10px;
                font-size: 8px;
                color: rgba(255,255,255,0.1);
                font-family: monospace;
                z-index: 1000;
                pointer-events: none;
                user-select: none;
                transform: rotate(-90deg);
                transform-origin: bottom right;
            }
            
            @media (max-width: 768px) {
                .header h1 { font-size: 2rem; }
                .card { padding: 1.5rem; }
                .container { padding: 1rem; }
                .watermark { display: none; }
            }
        </style>
    </head>
    <body>
        <div class="watermark">Made by Sean N</div>
        <div class="container">
            <header class="header">
                <h1>🔍 Feedback Classification API</h1>
                <p>Enhanced intelligent categorization and priority assessment for manufacturing feedback</p>
                <div class="status-badge">✅ API Status: Online & Ready</div>
            </header>

            <div class="card">
                <h2><span class="icon">🚀</span>API Endpoint</h2>
                <div class="endpoint">POST /classify</div>
                <p>Submit feedback text for enhanced intelligent classification and priority assessment with improved accuracy.</p>
                
                <h3>📝 Request Format</h3>
                <div class="code-block">{
    "text": "Emergency exit is blocked by equipment and workers can't get out"
}</div>
                
                <h3>📊 Response Format</h3>
                <div class="code-block">{
    "category": "Safety Concern",
    "autocategory": "Safety Concern", 
    "priority": "Critical",
    "autopriority": "Critical",
    "confidence": 9,
    "confidence_score": 9,
    "priority_score": 6.8,
    "matched_example": "Emergency exit is blocked",
    "keyword_matches": ["emergency", "exit", "blocked", "safety"],
    "priority_factors": ["Critical: emergency", "Critical: blocked exit", "Category: Safety Concern (×1.8)"]
}</div>
            </div>

            <div class="card">
                <h2><span class="icon">📂</span>Enhanced Categories</h2>
                <p>Improved classification with expanded keyword detection and fuzzy matching for better accuracy.</p>
                <div class="grid">
                    <div class="category-card">
                        <h4>🛡️ Safety Concern</h4>
                        <p>Safety hazards, emergencies, and risks with enhanced detection</p>
                    </div>
                    <div class="category-card">
                        <h4>⚙️ Machine/Equipment Issue</h4>
                        <p>Equipment problems, malfunctions, and maintenance needs</p>
                    </div>
                    <div class="category-card">
                        <h4>💡 Process Improvement Idea</h4>
                        <p>Optimization suggestions and efficiency improvements</p>
                    </div>
                    <div class="category-card">
                        <h4>📋 Other</h4>
                        <p>General feedback, facilities, and workplace requests</p>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2><span class="icon">⚡</span>Enhanced Priority Levels</h2>
                <p>Improved priority scoring with balanced thresholds and category-specific adjustments.</p>
                <div class="grid">
                    <div class="priority-card critical">
                        <h4>🚨 Critical</h4>
                        <p>Immediate safety threats, emergencies, or complete production stoppage</p>
                    </div>
                    <div class="priority-card high">
                        <h4>🔥 High</h4>
                        <p>Urgent safety risks, major equipment failures requiring prompt attention</p>
                    </div>
                    <div class="priority-card medium">
                        <h4>⚠️ Medium</h4>
                        <p>Important issues, maintenance needs, and process problems to address</p>
                    </div>
                    <div class="priority-card low">
                        <h4>📝 Low</h4>
                        <p>Suggestions, minor issues, and non-urgent improvement ideas</p>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2><span class="icon">🎯</span>Key Improvements</h2>
                <div class="grid">
                    <div class="category-card">
                        <h4>🔍 Enhanced Detection</h4>
                        <p>Lower similarity threshold and fuzzy matching for better category detection</p>
                    </div>
                    <div class="category-card">
                        <h4>🛡️ Safety Priority</h4>
                        <p>Automatic safety concern detection with minimum priority levels</p>
                    </div>
                    <div class="category-card">
                        <h4>📊 Improved Confidence</h4>
                        <p>Better confidence scoring with enhanced distribution (1-10 scale)</p>
                    </div>
                    <div class="category-card">
                        <h4>⚙️ Equipment Focus</h4>
                        <p>Enhanced equipment issue detection with expanded keyword matching</p>
                    </div>
                </div>
            </div>

            <footer>
                <p>Enhanced Feedback Classification API • Optimized for manufacturing environments with improved accuracy and reliability</p>
            </footer>
        </div>
    </body>
    </html>
    """

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({"error": "Internal server error"}), 500

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
