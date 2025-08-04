from flask import Flask, request, jsonify
import difflib
import logging
import re
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Enhanced Knowledge Base with Keywords ---
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

# Configuration
SIMILARITY_THRESHOLD = 0.35  # Lowered for more fuzzy matching
KEYWORD_THRESHOLD = 0.6
DEFAULT_CATEGORY = "General_Feedback"
MAX_TEXT_LENGTH = 5000

@dataclass
class ClassificationResult:
    """Data class for classification results."""
    category: str
    confidence: float
    matched_example: Optional[str]
    keyword_matches: List[str]
    similarity_scores: Dict[str, float]
    error: Optional[str] = None

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
    """Enhanced feedback classifier with multiple similarity algorithms."""

    def __init__(self, knowledge_base: List[Dict], 
                 similarity_threshold: float = SIMILARITY_THRESHOLD,
                 keyword_threshold: float = KEYWORD_THRESHOLD):
        self.knowledge_base = knowledge_base
        self.similarity_threshold = similarity_threshold
        self.keyword_threshold = keyword_threshold
        self.text_processor = TextProcessor()
        logger.info(f"Initialized enhanced classifier with {len(knowledge_base)} categories")

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
        Enhanced classification with multiple algorithms and fuzzy matching.
        """
        if not text or not text.strip():
            return ClassificationResult(
                category=DEFAULT_CATEGORY,
                confidence=0.0,
                matched_example=None,
                keyword_matches=[],
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

            return ClassificationResult(
                category=category_name,
                confidence=round(confidence, 3),
                matched_example=matched_example,
                keyword_matches=keyword_matches,
                similarity_scores={k: round(v, 3) for k, v in category_scores.items()},
                error=None
            )

        except Exception as e:
            logger.error(f"Error during enhanced classification: {e}")
            return ClassificationResult(
                category=DEFAULT_CATEGORY,
                confidence=0.0,
                matched_example=None,
                keyword_matches=[],
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
        <title>Feedback Classification API - Smurfit WestRock</title>
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

            .categories-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 1.5rem;
                margin: 2rem 0;
            }

            .category-card {
                background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                padding: 1.5rem;
                border-radius: 12px;
                border-left: 4px solid #1a5490;
                transition: all 0.3s ease;
            }

            .category-card:hover {
                transform: translateX(4px);
                box-shadow: 0 4px 20px rgba(26, 84, 144, 0.15);
            }

            .category-card h4 {
                color: #1a5490;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }

            .category-card p {
                color: #666;
                font-size: 0.9rem;
            }

            .features-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1.5rem;
                margin: 2rem 0;
            }

            .feature-item {
                display: flex;
                align-items: flex-start;
                gap: 1rem;
                padding: 1rem;
                background: rgba(26, 84, 144, 0.05);
                border-radius: 8px;
                transition: background 0.3s ease;
            }

            .feature-item:hover {
                background: rgba(26, 84, 144, 0.1);
            }

            .feature-icon {
                width: 40px;
                height: 40px;
                background: #1a5490;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                flex-shrink: 0;
            }

            .code-block {
                background: #2d3748;
                color: #e2e8f0;
                padding: 1.5rem;
                border-radius: 8px;
                margin: 1rem 0;
                font-family: 'Courier New', monospace;
                overflow-x: auto;
                border-left: 4px solid #1a5490;
            }

            .endpoints-list {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin: 1.5rem 0;
            }

            .endpoint-item {
                background: linear-gradient(135deg, #1a5490 0%, #2d6aa3 100%);
                color: white;
                padding: 1rem;
                border-radius: 8px;
                text-align: center;
                font-weight: 500;
                transition: transform 0.3s ease;
            }

            .endpoint-item:hover {
                transform: scale(1.05);
            }

            .endpoint-item code {
                display: block;
                font-size: 1.1rem;
                margin-bottom: 0.5rem;
            }

            .footer {
                background: #1a5490;
                color: white;
                text-align: center;
                padding: 2rem 0;
                margin-top: 3rem;
            }

            .sustainability-note {
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 12px;
                margin: 2rem 0;
                text-align: center;
            }

            @media (max-width: 768px) {
                .header-content {
                    text-align: center;
                }

                .title-section h1 {
                    font-size: 2rem;
                }

                .container {
                    padding: 0 1rem;
                }

                .section {
                    padding: 1.5rem;
                }
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
                            <h1>Feedback Classification API</h1>
                            <p class="subtitle">Advanced Classification • Built for Manufacturing Excellence</p>
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
                <div class="sustainability-note">
                    <h3>🌱 Supporting Sustainable Operations</h3>
                    <p>This system helps optimize workplace feedback processing, supporting our commitment to operational efficiency and continuous improvement in sustainable packaging solutions.</p>
                </div>

                <div class="section">
                    <h2><span class="icon">🎯</span> Classification Categories</h2>
                    <p>Our advanced system automatically categorizes workplace feedback into four key areas:</p>
                    <div class="categories-grid">
                        <div class="category-card">
                            <h4>🛡️ Safety Concern</h4>
                            <p>Issues related to workplace safety, hazards, emergency equipment, and protective measures</p>
                        </div>
                        <div class="category-card">
                            <h4>⚙️ Machine/Equipment Issue</h4>
                            <p>Problems with machinery, equipment malfunctions, maintenance needs, and technical issues</p>
                        </div>
                        <div class="category-card">
                            <h4>💡 Process Improvement Idea</h4>
                            <p>Suggestions for optimizing processes, increasing efficiency, and implementing best practices</p>
                        </div>
                        <div class="category-card">
                            <h4>📋 Other</h4>
                            <p>General requests, facility issues, training needs, and other workplace feedback</p>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <h2><span class="icon">🚀</span> Advanced Features</h2>
                    <div class="features-grid">
                        <div class="feature-item">
                            <div class="feature-icon">🔤</div>
                            <div>
                                <h4>Fuzzy Matching</h4>
                                <p>Handles misspellings and variations in language</p>
                            </div>
                        </div>
                        <div class="feature-item">
                            <div class="feature-icon">🧠</div>
                            <div>
                                <h4>Smart Recognition</h4>
                                <p>Uses multiple sophisticated algorithms for accurate classification</p>
                            </div>
                        </div>
                        <div class="feature-item">
                            <div class="feature-icon">📊</div>
                            <div>
                                <h4>Confidence Scoring</h4>
                                <p>Provides detailed confidence metrics for each classification</p>
                            </div>
                        </div>
                        <div class="feature-item">
                            <div class="feature-icon">🔍</div>
                            <div>
                                <h4>Keyword Analysis</h4>
                                <p>Identifies and matches relevant keywords and phrases</p>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <h2><span class="icon">🔧</span> API Usage</h2>
                    <h3>Making a Classification Request</h3>
                    <p>Send a POST request to <code>/classify</code> with your feedback text:</p>
                    <div class="code-block">
{
  "text": "The conveyor belt is making strange noises and needs maintenance"
}</div>

                    <h3>Response Format</h3>
                    <p>The API returns detailed classification results:</p>
                    <div class="code-block">
{
  "category": "Machine/Equipment Issue",
  "confidence": 0.847,
  "matched_example": "Conveyor belt is slipping",
  "keyword_matches": ["conveyor", "maintenance", "noise"],
  "similarity_scores": {
    "Safety Concern": 0.234,
    "Machine/Equipment Issue": 0.847,
    "Process Improvement Idea": 0.156,
    "Other": 0.089
  }
}</div>
                </div>

                <div class="section">
                    <h2><span class="icon">🌐</span> Available Endpoints</h2>
                    <div class="endpoints-list">
                        <div class="endpoint-item">
                            <code>/classify</code>
                            <div>Classify feedback text</div>
                        </div>
                        <div class="endpoint-item">
                            <code>/categories</code>
                            <div>View all categories</div>
                        </div>
                        <div class="endpoint-item">
                            <code>/test</code>
                            <div>Test the classifier</div>
                        </div>
                        <div class="endpoint-item">
                            <code>/health</code>
                            <div>System health check</div>
                        </div>
                    </div>
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

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "categories_available": len(KNOWLEDGE_BASE),
        "version": "2.0.0",
        "features": [
            "fuzzy_matching",
            "spelling_correction", 
            "keyword_extraction",
            "multiple_similarity_algorithms"
        ]
    })

@app.route("/categories", methods=["GET"])
def get_categories():
    """Get available categories and their examples."""
    categories = []
    for item in KNOWLEDGE_BASE:
        categories.append({
            "category": item["category"],
            "keyword_count": len(item.get("keywords", [])),
            "example_count": len(item["examples"]),
            "sample_keywords": item.get("keywords", [])[:5],
            "sample_examples": item["examples"][:3]
        })
    return jsonify({"categories": categories})

@app.route("/test", methods=["GET"])
def test_classifier():
    """Test endpoint with sample inputs."""
    test_cases = [
        "The conveyor belt is making weird noises",
        "We need better safety equipment",
        "Could we automate this process?",
        "The break room coffee machine is broken",
        "Emergency exit is blocked by boxes",
        "Machien keeps jamming every few hours",  # Intentional misspelling
        "Suggest implementing lean manufacturing"
    ]

    results = []
    for test_text in test_cases:
        result = classifier.classify(test_text)
        results.append({
            "input": test_text,
            "classification": {
                "category": result.category,
                "confidence": result.confidence,
                "keyword_matches": result.keyword_matches[:3],  # Limit for readability
                "matched_example": result.matched_example
            }
        })

    return jsonify({"test_results": results})

@app.route("/classify", methods=["POST"])
def classify():
    """Classify feedback text with enhanced algorithms."""
    try:
        # Get and validate request data
        data = request.get_json()
        is_valid, error_message = validate_request_data(data)

        if not is_valid:
            logger.warning(f"Invalid request: {error_message}")
            return jsonify({"error": error_message}), 400

        text = data["text"].strip()
        logger.info(f"Classifying feedback: '{text[:50]}...' ({len(text)} chars)")

        # Classify the text
        result = classifier.classify(text)

        # Log the classification result
        logger.info(f"Classification result: {result.category} (confidence: {result.confidence})")

        # Return detailed result
        response = {
            "category": result.category,
            "confidence": result.confidence,
            "matched_example": result.matched_example,
            "keyword_matches": result.keyword_matches,
            "similarity_scores": result.similarity_scores,
            "error": result.error
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Unexpected error in classify endpoint: {e}")
        return jsonify({
            "error": "Internal server error",
            "category": DEFAULT_CATEGORY,
            "confidence": 0.0,
            "matched_example": None,
            "keyword_matches": [],
            "similarity_scores": {}
        }), 500

if __name__ == "__main__":
    logger.info("Starting Enhanced Feedback Classification API...")
    app.run(host="0.0.0.0", port=5000, debug=False)
    app.run(host="0.0.0.0", port=81, debug=False)
