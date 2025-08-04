import os
import re
import math
import difflib
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum

from flask import Flask, request, jsonify

# ==============================================================================
# 1. FLASK APP INITIALIZATION & LOGGING
# ==============================================================================
# Configure logging to see output in the deployment environment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Flask application
app = Flask(__name__)


# ==============================================================================
# 2. KNOWLEDGE BASE, RULES, AND CONSTANTS
# ==============================================================================

class Priority(Enum):
    """Priority levels for feedback items."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

# This knowledge base contains the categories, keywords, and examples for classification.
KNOWLEDGE_BASE = [
    {
        "category": "Safety Concern",
        "keywords": ["safety", "emergency", "danger", "hazard", "risk", "injury", "accident", "unsafe", "spill", "leak", "fire", "toxic", "chemical", "guard", "protective", "first aid", "evacuation", "blocked", "exit", "warning", "caution", "slip", "fall", "cut", "burn", "electrical", "shock", "noise", "loud", "ppe", "helmet", "gloves", "goggles", "mask", "respirator"],
        "examples": ["Emergency stop failed; risk of injury", "Hazardous spill in aisle", "Guard rail is loose", "Forklift has worn tires", "Chemical leak in storage area", "Missing safety signs", "Blocked emergency exit", "Frayed electrical cord", "Noise levels exceeding limits", "Spill kit missing", "First aid kit expired items", "Ladder rungs showing wear", "Workers not wearing proper PPE", "Wet floor without warning signs", "Sharp edges on equipment"]
    },
    {
        "category": "Machine/Equipment Issue",
        "keywords": ["machine", "equipment", "conveyor", "motor", "pump", "compressor", "hydraulic", "mechanical", "broken", "malfunction", "jam", "stuck", "slow", "fast", "vibration", "noise", "leak", "pressure", "temperature", "calibration", "maintenance", "repair", "replace", "belt", "chain", "gear", "bearing", "sensor", "control", "display", "gauge", "meter", "alarm", "error", "fault", "overheating", "smoking", "grinding"],
        "examples": ["Conveyor belt is slipping", "Paper web keeps breaking", "Banding machine jamming", "Quality control scanner malfunctioning", "Loading dock hydraulics slow", "Corrugator heating elements inconsistent", "Pallet jack wheels sticking", "Air compressor running continuously", "Conveyor speed control erratic", "Scale calibration seems off", "Pressure gauge readings inconsistent", "Cooling system not maintaining temp", "Motor vibrations increasing", "Dock leveler hydraulics leaking", "Machine making strange grinding noise", "Equipment overheating frequently"]
    },
    {
        "category": "Process Improvement Idea",
        "keywords": ["improve", "optimize", "efficiency", "productivity", "streamline", "automate", "reduce", "increase", "faster", "better", "easier", "suggestion", "idea", "recommend", "proposal", "enhancement", "upgrade", "modification", "change", "implement", "lean", "5s", "kaizen", "workflow", "process", "procedure", "method", "technique", "system", "digital", "technology", "training", "cross-train", "schedule", "organize", "standardize", "quality", "metrics"],
        "examples": ["Suggest 5S to reduce downtime", "Optimize loading dock schedule", "Reduce setup time by organizing tools", "Implement 5S system in work area", "Streamline packaging process", "Cross-train operators for flexibility", "Digital work order system", "Barcode scanning for inventory", "Preventive maintenance schedule", "Quality metrics dashboard", "Energy-saving lighting upgrade", "Automated quality inspection", "Supplier quality scorecards", "Could we automate this repetitive task?", "What if we rearranged the workspace layout?"]
    },
    {
        "category": "Other",
        "keywords": ["supplies", "training", "lighting", "breakroom", "microwave", "parking", "time clock", "cafeteria", "menu", "recognition", "assessment", "materials", "outdated", "program", "variety", "potholes", "glitches", "suggestion box", "empty", "request", "need", "want", "facilities", "comfort", "ergonomic", "chair", "desk", "temperature", "air conditioning", "heating", "bathroom", "cleanliness", "janitorial", "supplies", "coffee", "water", "snacks"],
        "examples": ["Need more breakroom supplies", "Training request for new equipment", "Request for additional lighting", "Suggestion box needs emptying", "Break room microwave not working", "Request ergonomic assessment", "Training materials outdated", "Employee recognition program", "Cafeteria menu variety", "Parking lot potholes", "Time clock system glitches", "Could we get better coffee in the break room?", "The bathroom needs more frequent cleaning", "Parking spaces are too narrow"]
    }
]

# Rules for determining priority based on keywords.
PRIORITY_RULES = {
    "critical_keywords": ["emergency", "fire", "explosion", "toxic", "chemical leak", "gas leak", "electrical shock", "electrocuted", "collapsed", "trapped", "unconscious", "bleeding", "severe injury", "broken bone", "ambulance", "hospital", "shutdown", "complete failure", "total breakdown", "stopped production", "line down", "plant shutdown", "evacuation", "hazmat"],
    "high_keywords": ["danger", "hazard", "unsafe", "blocked exit", "missing guard", "exposed wire", "frayed cord", "overheating", "smoking", "sparking", "grinding noise", "violent vibration", "pressure buildup", "steam leak", "oil leak", "broken", "malfunction", "jam", "stuck", "won't start", "keeps stopping", "major issue", "production impact", "quality problem", "customer complaint"],
    "medium_keywords": ["worn", "loose", "slow", "inconsistent", "needs repair", "needs maintenance", "replace soon", "calibration", "adjustment", "minor leak", "slight noise", "efficiency", "productivity", "improvement", "optimize", "streamline", "training needed", "outdated", "upgrade", "modification"],
    "low_keywords": ["suggestion", "idea", "recommend", "could we", "what if", "maybe", "convenience", "comfort", "supplies", "breakroom", "parking", "lighting", "temperature", "coffee", "menu", "recognition", "nice to have", "future", "eventually", "when possible"]
}

# Rules for adjusting priority based on words indicating urgency.
URGENCY_INDICATORS = {
    "immediate": ["now", "immediately", "asap", "urgent", "emergency", "right away", "critical"],
    "soon": ["soon", "quickly", "fast", "today", "this week", "needs attention"],
    "eventually": ["eventually", "future", "someday", "when possible", "nice to have"]
}

# Global configuration constants
SIMILARITY_THRESHOLD = 0.20
DEFAULT_CATEGORY = "General_Feedback"
MAX_TEXT_LENGTH = 5000

# ==============================================================================
# 3. DATA CLASSES AND CORE LOGIC CLASSES
#     --- All classes must be defined before they are used ---
# ==============================================================================

@dataclass
class ClassificationResult:
    """A data class to hold the results of a classification task."""
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
    """Handles the logic for assigning a priority level to a piece of text."""
    def __init__(self):
        self.priority_rules = PRIORITY_RULES
        self.urgency_indicators = URGENCY_INDICATORS

    def calculate_priority(self, text: str, category: str, text_keywords: Set[str]) -> Tuple[Priority, float, List[str]]:
        priority_factors, base_score = [], 0.0
        normalized_text = text.lower()

        # Check for keywords from each priority level
        if self._check_keyword_matches(normalized_text, text_keywords, self.priority_rules["critical_keywords"]):
            base_score += 4.0
            priority_factors.extend([f"Critical: {match}" for match in self._check_keyword_matches(normalized_text, text_keywords, self.priority_rules["critical_keywords"])])
        if self._check_keyword_matches(normalized_text, text_keywords, self.priority_rules["high_keywords"]):
            base_score += 3.0
            priority_factors.extend([f"High: {match}" for match in self._check_keyword_matches(normalized_text, text_keywords, self.priority_rules["high_keywords"])])
        if self._check_keyword_matches(normalized_text, text_keywords, self.priority_rules["medium_keywords"]):
            base_score += 2.0
            priority_factors.extend([f"Medium: {match}" for match in self._check_keyword_matches(normalized_text, text_keywords, self.priority_rules["medium_keywords"])])
        if self._check_keyword_matches(normalized_text, text_keywords, self.priority_rules["low_keywords"]):
            base_score += 1.0
            priority_factors.extend([f"Low: {match}" for match in self._check_keyword_matches(normalized_text, text_keywords, self.priority_rules["low_keywords"])])

        # Apply multipliers based on category, urgency, and severity
        category_multiplier = self._get_category_multiplier(category)
        base_score *= category_multiplier
        if category_multiplier != 1.0: priority_factors.append(f"Category adjustment: {category} (x{category_multiplier})")
        
        urgency_multiplier = self._check_urgency_indicators(normalized_text)
        base_score *= urgency_multiplier
        if urgency_multiplier != 1.0: priority_factors.append(f"Urgency indicator (x{urgency_multiplier})")

        if not priority_factors: priority_factors.append("Default categorization")
        
        return self._score_to_priority(base_score), round(base_score, 2), priority_factors

    def _check_keyword_matches(self, text: str, text_keywords: Set[str], priority_keywords: List[str]) -> List[str]:
        matches = [pk for pk in priority_keywords if pk in text or set(pk.split()).intersection(text_keywords)]
        return list(set(matches))

    def _get_category_multiplier(self, category: str) -> float:
        return {"Safety Concern": 1.5, "Machine/Equipment Issue": 1.3, "Process Improvement Idea": 0.8, "Other": 0.7, "General_Feedback": 0.5}.get(category, 1.0)

    def _check_urgency_indicators(self, text: str) -> float:
        for urgency_level, indicators in self.urgency_indicators.items():
            if any(indicator in text for indicator in indicators):
                if urgency_level == "immediate": return 1.5
                elif urgency_level == "soon": return 1.2
                elif urgency_level == "eventually": return 0.8
        return 1.0

    def _score_to_priority(self, score: float) -> Priority:
        if score >= 6.0: return Priority.CRITICAL
        elif score >= 4.0: return Priority.HIGH
        elif score >= 2.0: return Priority.MEDIUM
        else: return Priority.LOW

class TextProcessor:
    """Handles text cleaning, normalization, and keyword extraction."""
    @staticmethod
    def normalize_text(text: str) -> str:
        if not text: return ""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\-\']', ' ', text)
        return text

    @staticmethod
    def extract_keywords(text: str) -> Set[str]:
        words = TextProcessor.normalize_text(text).split()
        stop_words = {'the', 'is', 'at', 'on', 'and', 'a', 'to', 'it', 'in', 'for', 'of', 'i', 'you', 'he', 'she'}
        return {word for word in words if len(word) > 2 and word not in stop_words}

    @staticmethod
    def calculate_cosine_similarity(text1: str, text2: str) -> float:
        words1, words2 = Counter(TextProcessor.normalize_text(text1).split()), Counter(TextProcessor.normalize_text(text2).split())
        intersection = set(words1.keys()) & set(words2.keys())
        if not intersection: return 0.0
        dot_product = sum(words1[word] * words2[word] for word in intersection)
        mag1, mag2 = math.sqrt(sum(c**2 for c in words1.values())), math.sqrt(sum(c**2 for c in words2.values()))
        return dot_product / (mag1 * mag2) if mag1 and mag2 else 0.0

class EnhancedFeedbackClassifier:
    """The main classifier class that orchestrates the entire process."""
    def __init__(self, knowledge_base: List[Dict], similarity_threshold: float = SIMILARITY_THRESHOLD):
        self.knowledge_base = knowledge_base
        self.similarity_threshold = similarity_threshold
        self.text_processor = TextProcessor()
        self.priority_engine = PriorityEngine()
        logger.info(f"Classifier initialized with {len(knowledge_base)} categories.")

    def _calculate_example_similarity(self, text: str, examples: List[str]) -> Tuple[float, Optional[str]]:
        best_score, best_example = 0.0, None
        norm_text = self.text_processor.normalize_text(text)
        for ex in examples:
            norm_ex = self.text_processor.normalize_text(ex)
            seq_ratio = difflib.SequenceMatcher(None, norm_text, norm_ex).ratio()
            cos_sim = self.text_processor.calculate_cosine_similarity(norm_text, norm_ex)
            score = (seq_ratio * 0.6) + (cos_sim * 0.4)
            if score > best_score:
                best_score, best_example = score, ex
        return best_score, best_example

    def classify(self, text: str) -> ClassificationResult:
        if not text or not text.strip():
            return ClassificationResult(category=DEFAULT_CATEGORY, priority=Priority.LOW.value, confidence=0.0, priority_score=0.0, matched_example=None, keyword_matches=[], priority_factors=[], similarity_scores={}, error="Empty text provided")
        
        try:
            text_keywords = self.text_processor.extract_keywords(text)
            scores = {}
            for item in self.knowledge_base:
                category = item['category']
                kw_matches = text_keywords.intersection(set(item.get('keywords', [])))
                kw_score = len(kw_matches) / len(item.get('keywords', [])) if item.get('keywords') else 0.0
                ex_score, _ = self._calculate_example_similarity(text, item['examples'])
                scores[category] = (kw_score * 0.4) + (ex_score * 0.6)
            
            best_cat, confidence = max(scores.items(), key=lambda x: x[1])
            
            if confidence < self.similarity_threshold:
                category_name, kw_matches, matched_ex = DEFAULT_CATEGORY, [], None
            else:
                category_name = best_cat
                item = next(i for i in self.knowledge_base if i['category'] == category_name)
                kw_matches = list(text_keywords.intersection(set(item.get('keywords', []))))
                _, matched_ex = self._calculate_example_similarity(text, item['examples'])

            priority, p_score, p_factors = self.priority_engine.calculate_priority(text, category_name, text_keywords)
            
            return ClassificationResult(
                category=category_name,
                priority=priority.value,
                confidence=round(confidence, 3),
                priority_score=p_score,
                matched_example=matched_ex,
                keyword_matches=kw_matches,
                priority_factors=p_factors,
                similarity_scores={k: round(v, 3) for k, v in scores.items()}
            )
        except Exception as e:
            logger.error(f"Classification error: {e}", exc_info=True)
            return ClassificationResult(category=DEFAULT_CATEGORY, priority=Priority.LOW.value, confidence=0.0, priority_score=0.0, matched_example=None, keyword_matches=[], priority_factors=[], similarity_scores={}, error="An internal classification error occurred")

# ==============================================================================
# 4. HELPER FUNCTIONS & APP SETUP
# ==============================================================================

def validate_request_data(data: Optional[Dict]) -> tuple[bool, str]:
    """Validates the incoming JSON data for the /classify endpoint."""
    if not data: return False, "No JSON data provided"
    if 'text' not in data: return False, "Missing 'text' field in request"
    text = data.get('text', '')
    if not isinstance(text, str): return False, "'text' field must be a string"
    if len(text) > MAX_TEXT_LENGTH: return False, f"Text length exceeds maximum of {MAX_TEXT_LENGTH} characters"
    return True, ""

# This line MUST come after the class `EnhancedFeedbackClassifier` is defined.
classifier = EnhancedFeedbackClassifier(KNOWLEDGE_BASE)

# ==============================================================================
# 5. FLASK ROUTES AND ERROR HANDLERS
# ==============================================================================

@app.errorhandler(400)
def bad_request_error(error):
    return jsonify({"error": "Bad Request", "message": str(error)}), 400

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Not Found", "message": "The requested URL was not found on the server."}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal Server Error: {error}")
    return jsonify({"error": "Internal Server Error", "message": "An internal error occurred."}), 500

@app.route("/classify", methods=['POST'])
def handle_classify():
    """API endpoint for classifying and prioritizing feedback."""
    data = request.get_json()
    is_valid, error_message = validate_request_data(data)
    if not is_valid:
        return jsonify({"error": error_message}), 400

    result = classifier.classify(data.get("text", ""))
    if result.error:
        return jsonify({"error": result.error}), 500

    # Convert confidence to an integer on a scale of 0-10
    confidence_int = int(result.confidence * 10)

    # Manually construct the response dictionary to include duplicate keys
    result_dict = {
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
    }
    return jsonify(result_dict)

@app.route("/")
def home():
    """Serves the main documentation page."""
    return """
    <!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Feedback Classification API</title><link href="https://fonts.googleapis.com/css2?family=Segoe+UI:wght@300;400;600;700&display=swap" rel="stylesheet"><style>*{margin:0;padding:0;box-sizing:border-box}body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;line-height:1.7;color:#333;background-color:#f8f9fa}.header{background:linear-gradient(135deg,#1a5490 0%,#2d6aa3 100%);color:white;padding:3rem 1rem;text-align:center;box-shadow:0 4px 12px rgba(0,0,0,.1)}.header h1{font-size:2.8rem;font-weight:700;margin-bottom:.5rem}.header p{font-size:1.2rem;opacity:.9;font-weight:300}.status-badge{display:inline-block;background:rgba(255,255,255,.15);padding:.5rem 1.2rem;border-radius:25px;font-size:.9rem;margin-top:1.5rem;border:1px solid rgba(255,255,255,.3);font-weight:600}.container{max-width:960px;margin:0 auto;padding:2rem 1rem}.section{background:white;margin-bottom:2.5rem;padding:2.5rem;border-radius:12px;box-shadow:0 6px 24px rgba(0,0,0,.07);border:1px solid #e9ecef}.section h2{color:#1a5490;font-size:2rem;font-weight:600;margin-bottom:1.5rem;padding-bottom:.8rem;border-bottom:2px solid #e9ecef}h3{color:#2d6aa3;font-size:1.4rem;margin:1.5rem 0 1rem 0;font-weight:600}p,li{color:#555;font-size:1rem}ul{list-style-position:inside;padding-left:1rem}code.inline{background:#e9ecef;padding:.2em .4em;margin:0;font-size:85%;border-radius:6px;font-family:'SFMono-Regular',Consolas,'Liberation Mono',Menlo,Courier,monospace;color:#c7254e}pre{background:#212529;color:#f8f9fa;padding:1.5rem;border-radius:8px;white-space:pre-wrap;word-wrap:break-word;font-family:'SFMono-Regular',Consolas,'Liberation Mono',Menlo,Courier,monospace;font-size:.9rem;margin:1rem 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1.5rem}.card{padding:1.5rem;border-radius:8px;border-left:5px solid;transition:transform .2s ease,box-shadow .2s ease}.card:hover{transform:translateY(-3px);box-shadow:0 8px 20px rgba(0,0,0,.1)}.critical{border-color:#d90429;background:#fff1f2;color:#8c1c13}.high{border-color:#ffb703;background:#fffbeb;color:#b56213}.medium{border-color:#2a9d8f;background:#f0fdfa;color:#1e655d}.low{border-color:#6c757d;background:#f8f9fa;color:#343a40}footer{text-align:center;padding:2rem;color:#6c757d;font-size:.9rem}</style></head><body><header class="header"><h1>Feedback Classification & Prioritization API</h1><p>Advanced Classification & Prioritization &bull; Built for Manufacturing Excellence</p><div class="status-badge">API Status: Operational</div></header><main class="container"><section class="section"><h2>API Endpoint</h2><h3><code class="inline">POST /classify</code></h3><p>This endpoint analyzes the provided text and returns a detailed classification and priority assessment.</p><h3>Request Body</h3><p>The request must be a JSON object with a single key, <code class="inline">text</code>.</p><pre><code>{\n    "text": "The conveyor belt near station 4 is making a loud grinding noise and seems to be running slower than usual."\n}</code></pre><h3>Success Response (200 OK)</h3><p>A successful request returns a JSON object with the classification details, including duplicate keys for compatibility.</p><pre><code>{\n    "category": "Machine/Equipment Issue",\n    "autocategory": "Machine/Equipment Issue",\n    "priority": "high",\n    "autopriority": "high",\n    "confidence": 0.875,\n    "confidence_score": 0.875,\n    "priority_score": 5.85,\n    ...\n}</code></pre></section><section class="section"><h2>Possible Categories</h2><div class="grid"><div class="card low"><h4>Safety Concern</h4></div><div class="card low"><h4>Machine/Equipment Issue</h4></div><div class="card low"><h4>Process Improvement Idea</h4></div><div class="card low"><h4>Other</h4></div></div></section><section class="section"><h2>Priority Levels</h2><div class="grid"><div class="card critical"><h4>Critical</h4><p>Immediate threats to safety or production.</p></div><div class="card high"><h4>High</h4><p>Serious issues requiring urgent attention.</p></div><div class="card medium"><h4>Medium</h4><p>Moderate issues that should be addressed soon.</p></div><div class="card low"><h4>Low</h4><p>Suggestions and non-urgent requests.</p></div></div></section></main><footer>Feedback Classification API &copy; 2025</footer></body></html>
    """

# ==============================================================================
# 6. MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    # Get port from environment variable or default to 8080 for local development
    port = int(os.environ.get("PORT", 8080))
    # Run the app, listening on all network interfaces
    # Set debug=False for production environments
    app.run(host="0.0.0.0", port=port, debug=False)
