from flask import Flask, request, jsonify
import difflib
import logging
import re
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class Priority(Enum):
    """Priority levels for feedback items."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# --- Enhanced Knowledge Base with Keywords and Priority Rules ---
KNOWLEDGE_BASE = [
    {
        "category": "Safety Concern",
        "keywords": [
            "safety", "emergency", "danger", "hazard", "risk", "injury", "accident", 
            "unsafe", "spill", "leak", "fire", "toxic", "chemical", "guard", "protective",
            "first aid", "evacuation", "blocked", "exit", "warning", "caution", "slip",
            "fall", "cut", "burn", "electrical", "shock", "noise", "loud", "ppe",
            "helmet", "gloves", "goggles", "mask", "respirator"
        ],
        "examples": [
            "Emergency stop failed; risk of injury",
            "Hazardous spill in aisle",
            "Guard rail is loose",
            "Forklift has worn tires",
            "Chemical leak in storage area",
            "Missing safety signs",
            "Blocked emergency exit",
            "Frayed electrical cord",
            "Noise levels exceeding limits",
            "Spill kit missing",
            "First aid kit expired items",
            "Ladder rungs showing wear",
            "Workers not wearing proper PPE",
            "Wet floor without warning signs",
            "Sharp edges on equipment"
        ]
    },
    {
        "category": "Machine/Equipment Issue",
        "keywords": [
            "machine", "equipment", "conveyor", "motor", "pump", "compressor", "hydraulic",
            "mechanical", "broken", "malfunction", "jam", "stuck", "slow", "fast",
            "vibration", "noise", "leak", "pressure", "temperature", "calibration",
            "maintenance", "repair", "replace", "belt", "chain", "gear", "bearing",
            "sensor", "control", "display", "gauge", "meter", "alarm", "error",
            "fault", "overheating", "smoking", "grinding"
        ],
        "examples": [
            "Conveyor belt is slipping",
            "Paper web keeps breaking",
            "Banding machine jamming",
            "Quality control scanner malfunctioning",
            "Loading dock hydraulics slow",
            "Corrugator heating elements inconsistent",
            "Pallet jack wheels sticking",
            "Air compressor running continuously",
            "Conveyor speed control erratic",
            "Scale calibration seems off",
            "Pressure gauge readings inconsistent",
            "Cooling system not maintaining temp",
            "Motor vibrations increasing",
            "Dock leveler hydraulics leaking",
            "Machine making strange grinding noise",
            "Equipment overheating frequently"
        ]
    },
    {
        "category": "Process Improvement Idea",
        "keywords": [
            "improve", "optimize", "efficiency", "productivity", "streamline", "automate",
            "reduce", "increase", "faster", "better", "easier", "suggestion", "idea",
            "recommend", "proposal", "enhancement", "upgrade", "modification", "change",
            "implement", "lean", "5s", "kaizen", "workflow", "process", "procedure",
            "method", "technique", "system", "digital", "technology", "training",
            "cross-train", "schedule", "organize", "standardize", "quality", "metrics"
        ],
        "examples": [
            "Suggest 5S to reduce downtime",
            "Optimize loading dock schedule",
            "Reduce setup time by organizing tools",
            "Implement 5S system in work area",
            "Streamline packaging process",
            "Cross-train operators for flexibility",
            "Digital work order system",
            "Barcode scanning for inventory",
            "Preventive maintenance schedule",
            "Quality metrics dashboard",
            "Energy-saving lighting upgrade",
            "Automated quality inspection",
            "Supplier quality scorecards",
            "Could we automate this repetitive task?",
            "What if we rearranged the workspace layout?"
        ]
    },
    {
        "category": "Other",
        "keywords": [
            "supplies", "training", "lighting", "breakroom", "microwave", "parking",
            "time clock", "cafeteria", "menu", "recognition", "assessment", "materials",
            "outdated", "program", "variety", "potholes", "glitches", "suggestion box",
            "empty", "request", "need", "want", "facilities", "comfort", "ergonomic",
            "chair", "desk", "temperature", "air conditioning", "heating", "bathroom",
            "cleanliness", "janitorial", "supplies", "coffee", "water", "snacks"
        ],
        "examples": [
            "Need more breakroom supplies",
            "Training request for new equipment",
            "Request for additional lighting",
            "Suggestion box needs emptying",
            "Break room microwave not working",
            "Request ergonomic assessment",
            "Training materials outdated",
            "Employee recognition program",
            "Cafeteria menu variety",
            "Parking lot potholes",
            "Time clock system glitches",
            "Could we get better coffee in the break room?",
            "The bathroom needs more frequent cleaning",
            "Parking spaces are too narrow"
        ]
    }
]

# Priority rules based on keywords and severity indicators
PRIORITY_RULES = {
    # Critical priority keywords - immediate safety risks or production shutdown
    "critical_keywords": [
        "emergency", "fire", "explosion", "toxic", "chemical leak", "gas leak",
        "electrical shock", "electrocuted", "collapsed", "trapped", "unconscious",
        "bleeding", "severe injury", "broken bone", "ambulance", "hospital",
        "shutdown", "complete failure", "total breakdown", "stopped production",
        "line down", "plant shutdown", "evacuation", "hazmat"
    ],
    
    # High priority keywords - serious safety concerns or major equipment issues
    "high_keywords": [
        "danger", "hazard", "unsafe", "blocked exit", "missing guard", "exposed wire",
        "frayed cord", "overheating", "smoking", "sparking", "grinding noise",
        "violent vibration", "pressure buildup", "steam leak", "oil leak",
        "broken", "malfunction", "jam", "stuck", "won't start", "keeps stopping",
        "major issue", "production impact", "quality problem", "customer complaint"
    ],
    
    # Medium priority keywords - moderate concerns requiring attention
    "medium_keywords": [
        "worn", "loose", "slow", "inconsistent", "needs repair", "needs maintenance",
        "replace soon", "calibration", "adjustment", "minor leak", "slight noise",
        "efficiency", "productivity", "improvement", "optimize", "streamline",
        "training needed", "outdated", "upgrade", "modification"
    ],
    
    # Low priority keywords - general requests and minor issues
    "low_keywords": [
        "suggestion", "idea", "recommend", "could we", "what if", "maybe",
        "convenience", "comfort", "supplies", "breakroom", "parking",
        "lighting", "temperature", "coffee", "menu", "recognition",
        "nice to have", "future", "eventually", "when possible"
    ]
}

# Urgency multipliers based on text indicators
URGENCY_INDICATORS = {
    "immediate": ["now", "immediately", "asap", "urgent", "emergency", "right away", "critical"],
    "soon": ["soon", "quickly", "fast", "today", "this week", "needs attention"],
    "eventually": ["eventually", "future", "someday", "when possible", "nice to have"]
}

# Configuration
SIMILARITY_THRESHOLD = 0.35
KEYWORD_THRESHOLD = 0.6
DEFAULT_CATEGORY = "General_Feedback"
MAX_TEXT_LENGTH = 5000

@dataclass
class ClassificationResult:
    """Data class for classification results."""
    category: str
    priority: str
    confidence: float
    priority_score: float
    matched_example: Optional[str]
    keyword_matches: List[str]
    priority_factors: List[str]
    similarity_scores: Dict[str, float]
    error: Optional[str] = None

class PriorityEngine:
    """Handles priority assignment based on content analysis."""
    
    def __init__(self):
        self.priority_rules = PRIORITY_RULES
        self.urgency_indicators = URGENCY_INDICATORS
    
    def calculate_priority(self, text: str, category: str, text_keywords: Set[str]) -> Tuple[Priority, float, List[str]]:
        """
        Calculate priority based on text content, category, and keywords.
        Returns: (Priority, priority_score, priority_factors)
        """
        priority_factors = []
        base_score = 0.0
        
        # Normalize text for analysis
        normalized_text = text.lower()
        
        # 1. Check for critical keywords first
        critical_matches = self._check_keyword_matches(normalized_text, text_keywords, 
                                                     self.priority_rules["critical_keywords"])
        if critical_matches:
            priority_factors.extend([f"Critical: {match}" for match in critical_matches])
            base_score += 4.0
        
        # 2. Check for high priority keywords
        high_matches = self._check_keyword_matches(normalized_text, text_keywords,
                                                  self.priority_rules["high_keywords"])
        if high_matches:
            priority_factors.extend([f"High: {match}" for match in high_matches])
            base_score += 3.0
        
        # 3. Check for medium priority keywords
        medium_matches = self._check_keyword_matches(normalized_text, text_keywords,
                                                    self.priority_rules["medium_keywords"])
        if medium_matches:
            priority_factors.extend([f"Medium: {match}" for match in medium_matches])
            base_score += 2.0
        
        # 4. Check for low priority keywords
        low_matches = self._check_keyword_matches(normalized_text, text_keywords,
                                                 self.priority_rules["low_keywords"])
        if low_matches:
            priority_factors.extend([f"Low: {match}" for match in low_matches])
            base_score += 1.0
        
        # 5. Apply category-based priority adjustments
        category_multiplier = self._get_category_multiplier(category)
        base_score *= category_multiplier
        if category_multiplier != 1.0:
            priority_factors.append(f"Category adjustment: {category} (x{category_multiplier})")
        
        # 6. Check for urgency indicators
        urgency_multiplier = self._check_urgency_indicators(normalized_text)
        base_score *= urgency_multiplier
        if urgency_multiplier != 1.0:
            priority_factors.append(f"Urgency indicator (x{urgency_multiplier})")
        
        # 7. Check for quantity/severity indicators
        severity_multiplier = self._check_severity_indicators(normalized_text)
        base_score *= severity_multiplier
        if severity_multiplier != 1.0:
            priority_factors.append(f"Severity indicator (x{severity_multiplier})")
        
        # 8. Determine final priority level
        priority = self._score_to_priority(base_score)
        
        # Ensure we have at least one priority factor
        if not priority_factors:
            priority_factors.append(f"Default categorization based on content analysis")
        
        return priority, round(base_score, 2), priority_factors
    
    def _check_keyword_matches(self, text: str, text_keywords: Set[str], priority_keywords: List[str]) -> List[str]:
        """Check for matches between text and priority keywords."""
        matches = []
        
        for priority_keyword in priority_keywords:
            # Check for exact phrase match in text
            if priority_keyword in text:
                matches.append(priority_keyword)
                continue
            
            # Check for individual word matches in extracted keywords
            priority_words = set(priority_keyword.split())
            if priority_words.intersection(text_keywords):
                matches.append(priority_keyword)
        
        return matches
    
    def _get_category_multiplier(self, category: str) -> float:
        """Get priority multiplier based on category."""
        category_multipliers = {
            "Safety Concern": 1.5,  # Safety issues get higher priority
            "Machine/Equipment Issue": 1.3,  # Equipment issues impact production
            "Process Improvement Idea": 0.8,  # Improvements are generally lower priority
            "Other": 0.7,  # General requests are typically lower priority
            "General_Feedback": 0.5
        }
        return category_multipliers.get(category, 1.0)
    
    def _check_urgency_indicators(self, text: str) -> float:
        """Check for urgency indicators in text."""
        for urgency_level, indicators in self.urgency_indicators.items():
            for indicator in indicators:
                if indicator in text:
                    if urgency_level == "immediate":
                        return 1.5
                    elif urgency_level == "soon":
                        return 1.2
                    elif urgency_level == "eventually":
                        return 0.8
        return 1.0
    
    def _check_severity_indicators(self, text: str) -> float:
        """Check for severity indicators that might affect priority."""
        severity_indicators = {
            "multiple": ["multiple", "several", "many", "various", "numerous"],
            "recurring": ["keeps", "always", "constantly", "repeatedly", "continuous"],
            "worsening": ["getting worse", "deteriorating", "failing", "breaking down"],
            "complete": ["completely", "totally", "entirely", "won't work", "dead"]
        }
        
        multiplier = 1.0
        for severity_type, indicators in severity_indicators.items():
            for indicator in indicators:
                if indicator in text:
                    if severity_type in ["complete", "worsening"]:
                        multiplier *= 1.3
                    elif severity_type in ["recurring", "multiple"]:
                        multiplier *= 1.2
                    break
        
        return multiplier
    
    def _score_to_priority(self, score: float) -> Priority:
        """Convert numerical score to priority level."""
        if score >= 6.0:
            return Priority.CRITICAL
        elif score >= 4.0:
            return Priority.HIGH
        elif score >= 2.0:
            return Priority.MEDIUM
        else:
            return Priority.LOW

class TextProcessor:
    """Handles text preprocessing and normalization."""

    # Common misspellings and variations
    SPELLING_CORRECTIONS = {
        'equipement': 'equipment',
        'machien': 'machine',
        'maintenence': 'maintenance',
        'occured': 'occurred',
        'recieve': 'receive',
        'seperate': 'separate',
        'definately': 'definitely',
        'occassion': 'occasion',
        'neccessary': 'necessary',
    }

    # Word variations and synonyms
    WORD_VARIATIONS = {
        'broke': ['broken', 'damaged', 'failed'],
        'fix': ['repair', 'maintenance', 'service'],
        'loud': ['noisy', 'noise', 'sound'],
        'slow': ['sluggish', 'delayed', 'lagging'],
        'fast': ['quick', 'rapid', 'speedy'],
        'hot': ['heating', 'overheating', 'warm'],
        'cold': ['cooling', 'freezing', 'chilled'],
        'dirty': ['unclean', 'messy', 'contaminated'],
        'idea': ['suggestion', 'proposal', 'recommendation'],
        'problem': ['issue', 'trouble', 'concern'],
    }

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for better matching."""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower().strip()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\-\']', ' ', text)

        # Apply spelling corrections
        words = text.split()
        corrected_words = []
        for word in words:
            if word in TextProcessor.SPELLING_CORRECTIONS:
                corrected_words.append(TextProcessor.SPELLING_CORRECTIONS[word])
            else:
                corrected_words.append(word)

        return ' '.join(corrected_words)

    @staticmethod
    def extract_keywords(text: str) -> Set[str]:
        """Extract meaningful keywords from text."""
        normalized = TextProcessor.normalize_text(text)
        words = normalized.split()

        # Filter out common stop words
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'are', 'as', 'was',
            'with', 'for', 'by', 'an', 'be', 'this', 'that', 'it', 'not', 'or',
            'have', 'from', 'they', 'we', 'been', 'had', 'their', 'said', 'each',
            'but', 'do', 'can', 'could', 'should', 'would', 'will', 'there', 'here'
        }

        keywords = set()
        for word in words:
            if len(word) > 2 and word not in stop_words:
                keywords.add(word)
                # Add word variations
                if word in TextProcessor.WORD_VARIATIONS:
                    keywords.update(TextProcessor.WORD_VARIATIONS[word])

        return keywords

    @staticmethod
    def calculate_cosine_similarity(text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        words1 = Counter(TextProcessor.normalize_text(text1).split())
        words2 = Counter(TextProcessor.normalize_text(text2).split())

        # Get intersection of words
        intersection = set(words1.keys()) & set(words2.keys())

        if not intersection:
            return 0.0

        # Calculate dot product
        dot_product = sum(words1[word] * words2[word] for word in intersection)

        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(count ** 2 for count in words1.values()))
        magnitude2 = math.sqrt(sum(count ** 2 for count in words2.values()))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

class EnhancedFeedbackClassifier:
    """Enhanced feedback classifier with priority assignment."""

    def __init__(self, knowledge_base: List[Dict], 
                 similarity_threshold: float = SIMILARITY_THRESHOLD,
                 keyword_threshold: float = KEYWORD_THRESHOLD):
        self.knowledge_base = knowledge_base
        self.similarity_threshold = similarity_threshold
        self.keyword_threshold = keyword_threshold
        self.text_processor = TextProcessor()
        self.priority_engine = PriorityEngine()
        logger.info(f"Initialized enhanced classifier with {len(knowledge_base)} categories and priority engine")

    def _calculate_keyword_score(self, text_keywords: Set[str], category_keywords: List[str]) -> Tuple[float, List[str]]:
        """Calculate keyword matching score."""
        if not text_keywords or not category_keywords:
            return 0.0, []

        matched_keywords = []
        fuzzy_matches = 0

        for text_keyword in text_keywords:
            for cat_keyword in category_keywords:
                # Exact match
                if text_keyword == cat_keyword:
                    matched_keywords.append(text_keyword)
                    continue

                # Fuzzy match using sequence matcher
                similarity = difflib.SequenceMatcher(None, text_keyword, cat_keyword).ratio()
                if similarity > 0.8:  # High similarity threshold for keywords
                    matched_keywords.append(f"{text_keyword}~{cat_keyword}")
                    fuzzy_matches += similarity

                # Substring match
                elif (len(text_keyword) > 3 and text_keyword in cat_keyword) or \
                     (len(cat_keyword) > 3 and cat_keyword in text_keyword):
                    matched_keywords.append(f"{text_keyword}*{cat_keyword}")

        # Calculate score based on matches
        exact_score = len([k for k in matched_keywords if '~' not in k and '*' not in k])
        fuzzy_score = fuzzy_matches
        partial_score = len([k for k in matched_keywords if '*' in k]) * 0.5

        total_score = (exact_score + fuzzy_score + partial_score) / len(category_keywords)
        return min(total_score, 1.0), matched_keywords

    def _calculate_example_similarity(self, text: str, examples: List[str]) -> Tuple[float, Optional[str]]:
        """Calculate similarity with examples using multiple algorithms."""
        best_score = 0.0
        best_example = None

        normalized_text = self.text_processor.normalize_text(text)

        for example in examples:
            normalized_example = self.text_processor.normalize_text(example)

            # Use multiple similarity measures
            sequence_ratio = difflib.SequenceMatcher(None, normalized_text, normalized_example).ratio()
            cosine_sim = self.text_processor.calculate_cosine_similarity(normalized_text, normalized_example)

            # Weighted combination of similarity measures
            combined_score = (sequence_ratio * 0.6) + (cosine_sim * 0.4)

            if combined_score > best_score:
                best_score = combined_score
                best_example = example

        return best_score, best_example

    def classify(self, text: str) -> ClassificationResult:
        """
        Enhanced classification with priority assignment.
        """
        if not text or not text.strip():
            return ClassificationResult(
                category=DEFAULT_CATEGORY,
                priority=Priority.LOW.value,
                confidence=0.0,
                priority_score=0.0,
                matched_example=None,
                keyword_matches=[],
                priority_factors=[],
                similarity_scores={},
                error="Empty text provided"
            )

        try:
            # Extract keywords from input text
            text_keywords = self.text_processor.extract_keywords(text)

            category_scores = {}
            all_keyword_matches = {}
            all_example_matches = {}

            # Check each category
            for item in self.knowledge_base:
                category = item['category']

                # Calculate keyword score
                keyword_score, keyword_matches = self._calculate_keyword_score(
                    text_keywords, item.get('keywords', [])
                )

                # Calculate example similarity score
                example_score, best_example = self._calculate_example_similarity(
                    text, item['examples']
                )

                # Combine scores with weights
                combined_score = (keyword_score * 0.4) + (example_score * 0.6)

                category_scores[category] = combined_score
                all_keyword_matches[category] = keyword_matches
                all_example_matches[category] = (example_score, best_example)

            # Find best matching category
            best_category = max(category_scores.items(), key=lambda x: x[1])
            category_name, confidence = best_category

            # Get details for best category
            keyword_matches = all_keyword_matches[category_name]
            example_score, matched_example = all_example_matches[category_name]

            # Apply threshold
            if confidence < self.similarity_threshold:
                category_name = DEFAULT_CATEGORY
                matched_example = None
                keyword_matches = []

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
                keyword_matches=keyword_matches,
                priority_factors=priority_factors,
                similarity_scores={k: round(v, 3) for k, v in category_scores.items()},
                error=None
            )

        except Exception as e:
            logger.error(f"Error during enhanced classification: {e}")
            return ClassificationResult(
                category=DEFAULT_CATEGORY,
                priority=Priority.LOW.value,
                confidence=0.0,
                priority_score=0.0,
                matched_example=None,
                keyword_matches=[],
                priority_factors=[],
                similarity_scores={},
                error="Classification error occurred"
            )

# Initialize enhanced classifier
classifier = EnhancedFeedbackClassifier(KNOWLEDGE_BASE)

def validate_request_data(data: Optional[Dict]) -> tuple[bool, str]:
    """Validate incoming request data."""
    if not data:
        return False, "No JSON data provided"

    if 'text' not in data:
        return False, "Missing 'text' field in request"

    text = data.get('text', '')
    if not isinstance(text, str):
        return False, "'text' field must be a string"

    if len(text) > MAX_TEXT_LENGTH:
        return False, f"Text length exceeds maximum allowed ({MAX_TEXT_LENGTH} characters)"

    return True, ""

@app.errorhandler(400)
def bad_request(error):
    """Handle bad request errors."""
    return jsonify({"error": "Bad request", "message": str(error)}), 400

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

@app.route("/")
def home():
    """Home page with API documentation styled after Smurfit WestRock."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Feedback Classification & Prioritization API - Smurfit WestRock</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                min-height: 100vh;
            }

            .header {
                background: linear-gradient(135deg, #1a5490 0%, #2d6aa3 100%);
                color: white;
                padding: 2rem 0;
                box-shadow: 0 4px 20px rgba(26, 84, 144, 0.3);
            }

            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 2rem;
            }

            .header-content {
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 2rem;
            }

            .logo-section {
                display: flex;
                align-items: center;
                gap: 1rem;
            }

            .logo {
                width: 60px;
                height: 60px;
                background: white;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                color: #1a5490;
                font-size: 1.2rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }

            .title-section h1 {
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
                text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }

            .subtitle {
                font-size: 1.1rem;
                opacity: 0.9;
                font-weight: 300;
            }

            .status-badge {
                background: rgba(255,255,255,0.2);
                padding: 0.5rem 1rem;
                border-radius: 25px;
                font-size: 0.9rem;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.3);
            }

            .main-content {
                padding: 3rem 0;
            }

            .section {
                background: white;
                margin: 2rem 0;
                padding: 2.5rem;
                border-radius: 16px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.08);
                border: 1px solid rgba(26, 84, 144, 0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }

            .section:hover {
                transform: translateY(-4px);
                box-shadow: 0 12px 40px rgba(0,0,0,0.12);
            }

            .section h2 {
                color: #1a5490;
                font-size: 1.8rem;
                margin-bottom: 1.5rem;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            .section h3 {
                color: #2d6aa3;
                font-size: 1.3rem;
                margin: 1.5rem 0 1rem 0;
                font-weight: 500;
            }

            .icon {
                width: 24px;
                height: 24px;
                background: #1a5490;
                border-radius: 50%;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                font-size: 0.8rem;
            }

            .priority-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1.5rem;
                margin: 2rem 0;
            }

            .priority-card {
                padding: 1.5rem;
                border-radius: 12px;
                border-left: 4px solid;
                transition: all 0.3s ease;
            }

            .priority-card:hover {
                transform: translateX(4px);
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            }

            .critical {
                border-color: #e63946;
                background: #fff0f0;
            }
            .high {
                border-color: #ffba08;
                background: #fffbe6;
            }
            .medium {
                border-color: #43aa8b;
                background: #f0fff5;
            }
            .low {
                border-color: #577590;
                background: #f3f9ff;
            }
        </style>
    </head>
    <body>
        <header class="header">
            <div class="container">
                <div class="header-content">
                    <div class="logo-section">
                        <div class="logo">SW</div>
                        <div class="title-section">
                            <h1>Feedback Classification & Prioritization API</h1>
                            <p class="subtitle">Advanced Classification &amp; Prioritization • Built for Manufacturing Excellence</p>
                        </div>
                    </div>
                    <div class="status-badge">
                        ✅ System Online
                    </div>
                </div>
            </div>
        </header>
        <main class="main-content">
            <div class="container">
                <div class="section">
                    <h2><span class="icon">🎯</span> Overview</h2>
                    <p>This API automatically categorizes and prioritizes workplace feedback for manufacturing environments, assigning clear categories and urgency levels for better decision making.</p>
                </div>
            </div>
        </main>
        <footer class="footer">
            <div class="container">
                <p>&copy; 2025 Smurfit WestRock • Advanced Feedback Classification System</p>
                <p>Supporting operational excellence in sustainable packaging solutions</p>
                <p style="font-size: 0.8rem; opacity: 0.7; margin-top: 1rem;">Built by Sean</p>
            </div>
        </footer>
    </body>
    </html>
    """

