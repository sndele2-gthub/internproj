import os
import re
import difflib
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)

class Priority(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

# Streamlined knowledge base focused on key indicators
KNOWLEDGE_BASE = {
    "Safety Concern": {
        "keywords": ["safety", "danger", "hazard", "unsafe", "emergency", "injury", "accident", "risk", "fire", "toxic", "chemical", "spill", "leak", "blocked", "exit", "fall", "cut", "burn", "electrical", "shock", "ppe", "hurt", "injured", "blood", "unconscious", "trapped", "evacuation"],
        "critical_words": ["emergency", "fire", "explosion", "toxic", "unconscious", "trapped", "evacuation", "severe", "major", "critical"],
        "high_words": ["danger", "hazard", "unsafe", "blocked exit", "injury", "accident", "electrical", "shock", "burn"],
        "medium_words": ["safety", "risk", "ppe", "protective", "guard", "warning"],
        "examples": ["Emergency exit blocked", "Chemical spill", "Worker injured", "Electrical hazard"]
    },
    "Machine/Equipment Issue": {
        "keywords": ["machine", "equipment", "conveyor", "motor", "pump", "broken", "malfunction", "jam", "stuck", "down", "stopped", "repair", "maintenance", "overheating", "noise", "vibration", "pressure", "temperature", "failure", "error"],
        "critical_words": ["complete failure", "shutdown", "explosion", "major breakdown", "total loss"],
        "high_words": ["broken", "malfunction", "down", "stopped", "overheating", "major", "urgent"],
        "medium_words": ["repair", "maintenance", "noise", "vibration", "slow", "adjustment"],
        "examples": ["Conveyor belt broken", "Motor overheating", "Equipment malfunction", "Machine down"]
    },
    "Process Improvement Idea": {
        "keywords": ["improve", "optimize", "efficiency", "productivity", "suggestion", "idea", "recommend", "better", "faster", "automate", "streamline", "reduce", "increase", "enhance", "upgrade"],
        "critical_words": [],
        "high_words": ["major improvement", "significant", "urgent change"],
        "medium_words": ["improve", "optimize", "efficiency", "productivity", "enhance"],
        "examples": ["Automate packaging", "Optimize workflow", "Improve efficiency", "Streamline process"]
    },
    "Other": {
        "keywords": ["supplies", "training", "lighting", "parking", "temperature", "break", "lunch", "bathroom", "coffee", "clean", "organize"],
        "critical_words": [],
        "high_words": ["urgent", "immediate"],
        "medium_words": ["training", "supplies", "facilities"],
        "examples": ["Need supplies", "Training request", "Parking issues", "Temperature control"]
    }
}

# Priority determination keywords
PRIORITY_INDICATORS = {
    "critical": ["critical", "emergency", "immediate", "urgent", "asap", "now", "stop", "shutdown", "danger", "life", "severe", "major", "complete", "total"],
    "high": ["high", "important", "soon", "broken", "down", "unsafe", "hazard", "significant"],
    "medium": ["medium", "moderate", "repair", "maintenance", "improve", "fix"],
    "low": ["low", "minor", "suggestion", "idea", "when possible", "future", "eventually"]
}

MAX_TEXT_LENGTH = 5000

@dataclass
class ClassificationResult:
    category: str
    priority: str
    confidence: float
    priority_score: float
    matched_keywords: List[str]
    priority_factors: List[str]
    error: Optional[str] = None

class FeedbackClassifier:
    def __init__(self):
        self.knowledge_base = KNOWLEDGE_BASE
    
    def normalize_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text.lower().strip())
    
    def extract_category_from_form(self, text: str) -> Optional[str]:
        """Check if the form explicitly mentions a category"""
        normalized = self.normalize_text(text)
        
        # Look for explicit category mentions in form submissions
        if any(word in normalized for word in ["safety concern", "safety issue", "safety problem"]):
            return "Safety Concern"
        elif any(word in normalized for word in ["equipment issue", "machine problem", "equipment problem"]):
            return "Machine/Equipment Issue"
        elif any(word in normalized for word in ["improvement", "suggestion", "optimize", "idea"]):
            return "Process Improvement Idea"
        
        return None
    
    def calculate_category_score(self, text: str, category: str) -> Tuple[float, List[str]]:
        """Calculate how well text matches a category"""
        normalized = self.normalize_text(text)
        words = set(normalized.split())
        
        category_data = self.knowledge_base[category]
        category_keywords = set(category_data["keywords"])
        
        # Find matching keywords
        matches = words.intersection(category_keywords)
        
        # Add fuzzy matching for important terms
        fuzzy_matches = []
        for word in words:
            if len(word) > 4:
                for keyword in category_keywords:
                    if len(keyword) > 4 and difflib.SequenceMatcher(None, word, keyword).ratio() > 0.85:
                        fuzzy_matches.append(keyword)
                        matches.add(keyword)
        
        # Calculate score based on keyword density
        if len(category_keywords) == 0:
            score = 0.0
        else:
            score = len(matches) / len(category_keywords)
        
        # Boost score for safety and equipment categories
        if category in ["Safety Concern", "Machine/Equipment Issue"] and score > 0:
            score *= 1.5
        
        return min(score, 1.0), list(matches)
    
    def determine_priority(self, text: str, category: str, matched_keywords: List[str]) -> Tuple[Priority, float, List[str]]:
        """Determine priority based on text content and category"""
        normalized = self.normalize_text(text)
        factors = []
        base_score = 1.0
        
        category_data = self.knowledge_base[category]
        
        # Check for priority-specific keywords in the category
        critical_found = any(word in normalized for word in category_data["critical_words"])
        high_found = any(word in normalized for word in category_data["high_words"])
        medium_found = any(word in normalized for word in category_data["medium_words"])
        
        # Check general priority indicators
        general_critical = any(word in normalized for word in PRIORITY_INDICATORS["critical"])
        general_high = any(word in normalized for word in PRIORITY_INDICATORS["high"])
        general_medium = any(word in normalized for word in PRIORITY_INDICATORS["medium"])
        general_low = any(word in normalized for word in PRIORITY_INDICATORS["low"])
        
        # Determine priority based on findings
        if critical_found or general_critical:
            priority = Priority.CRITICAL
            base_score = 5.0
            factors.append("Critical keywords detected")
        elif high_found or general_high:
            priority = Priority.HIGH
            base_score = 4.0
            factors.append("High priority keywords detected")
        elif medium_found or general_medium:
            priority = Priority.MEDIUM
            base_score = 2.5
            factors.append("Medium priority keywords detected")
        elif general_low:
            priority = Priority.LOW
            base_score = 1.0
            factors.append("Low priority keywords detected")
        else:
            # Default based on category
            if category == "Safety Concern":
                priority = Priority.HIGH
                base_score = 3.5
                factors.append("Default high priority for safety concerns")
            elif category == "Machine/Equipment Issue":
                priority = Priority.MEDIUM
                base_score = 2.0
                factors.append("Default medium priority for equipment issues")
            else:
                priority = Priority.LOW
                base_score = 1.0
                factors.append("Default low priority")
        
        return priority, base_score, factors
    
    def classify(self, text: str) -> ClassificationResult:
        if not text or not text.strip():
            return ClassificationResult(
                category="Other", priority=Priority.LOW.value, confidence=0.0,
                priority_score=0.0, matched_keywords=[], priority_factors=[],
                error="Empty text provided"
            )
        
        try:
            # First, check if category is explicitly mentioned in form
            explicit_category = self.extract_category_from_form(text)
            
            if explicit_category:
                category = explicit_category
                _, matched_keywords = self.calculate_category_score(text, category)
                confidence = 0.9  # High confidence for explicit mentions
            else:
                # Calculate scores for all categories
                scores = {}
                all_matches = {}
                
                for cat in self.knowledge_base.keys():
                    score, matches = self.calculate_category_score(text, cat)
                    scores[cat] = score
                    all_matches[cat] = matches
                
                # Find best category
                category = max(scores.keys(), key=lambda k: scores[k])
                confidence = scores[category]
                matched_keywords = all_matches[category]
                
                # If no good match found, default intelligently
                if confidence < 0.1:
                    normalized = self.normalize_text(text)
                    if any(word in normalized for word in ["safe", "danger", "hazard", "injury"]):
                        category = "Safety Concern"
                        confidence = 0.6
                    elif any(word in normalized for word in ["machine", "equipment", "broken", "repair"]):
                        category = "Machine/Equipment Issue"
                        confidence = 0.6
                    else:
                        category = "Other"
                        confidence = 0.5
            
            # Determine priority
            priority, priority_score, priority_factors = self.determine_priority(text, category, matched_keywords)
            
            return ClassificationResult(
                category=category,
                priority=priority.value,
                confidence=round(confidence, 3),
                priority_score=priority_score,
                matched_keywords=matched_keywords[:5],
                priority_factors=priority_factors
            )
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return ClassificationResult(
                category="Other", priority=Priority.LOW.value, confidence=0.0,
                priority_score=0.0, matched_keywords=[], priority_factors=[],
                error="Classification error occurred"
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
        return jsonify({"error": result.error}), 500
    
    # Convert confidence to 1-10 scale
    confidence_int = max(1, min(10, int(result.confidence * 10) + 1))
    
    return jsonify({
        "category": result.category,
        "autocategory": result.category,
        "priority": result.priority,
        "autopriority": result.priority,
        "confidence": confidence_int,
        "confidence_score": confidence_int,
        "priority_score": result.priority_score,
        "matched_keywords": result.matched_keywords,
        "priority_factors": result.priority_factors
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
                font-family: 'Inter', sans-serif;
                line-height: 1.6;
                color: #1f2937;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                position: relative;
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
                transition: transform 0.3s ease;
            }
            
            .card:hover {
                transform: translateY(-5px);
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
            
            .endpoint {
                background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
                color: white;
                padding: 1rem 1.5rem;
                border-radius: 12px;
                font-family: 'Monaco', monospace;
                font-size: 1.1rem;
                margin: 1rem 0;
                display: inline-block;
            }
            
            .code-block {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 1.5rem;
                margin: 1rem 0;
                font-family: 'Monaco', monospace;
                font-size: 0.9rem;
                overflow-x: auto;
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
            }
            
            .watermark {
                position: fixed;
                bottom: 10px;
                right: 15px;
                font-size: 10px;
                color: rgba(255,255,255,0.4);
                font-family: monospace;
                z-index: 1000;
                pointer-events: none;
                user-select: none;
                background: rgba(0,0,0,0.1);
                padding: 2px 6px;
                border-radius: 3px;
                backdrop-filter: blur(5px);
            }
            
            footer {
                text-align: center;
                padding: 2rem;
                color: rgba(255,255,255,0.8);
                font-size: 0.9rem;
                margin-top: 3rem;
            }
            
            @media (max-width: 768px) {
                .header h1 { font-size: 2rem; }
                .card { padding: 1.5rem; }
                .container { padding: 1rem; }
            }
        </style>
    </head>
    <body>
        <div class="watermark">Made by Sean N</div>
        <div class="container">
            <header class="header">
                <h1>🔍 Feedback Classification API</h1>
                <p>Smart categorization and priority assessment for manufacturing feedback</p>
                <div class="status-badge">✅ API Status: Online & Ready</div>
            </header>

            <div class="card">
                <h2><span>🚀</span>API Endpoint</h2>
                <div class="endpoint">POST /classify</div>
                <p>Submit feedback text for intelligent classification and priority assessment.</p>
                
                <h3>📝 Request Format</h3>
                <div class="code-block">{
    "text": "Safety concern: Emergency exit is blocked by equipment"
}</div>
                
                <h3>📊 Response Format</h3>
                <div class="code-block">{
    "category": "Safety Concern",
    "priority": "Critical",
    "confidence": 9,
    "priority_score": 5.0,
    "matched_keywords": ["safety", "emergency", "exit", "blocked"],
    "priority_factors": ["Critical keywords detected"]
}</div>
            </div>

            <div class="card">
                <h2><span>📂</span>Categories</h2>
                <div class="grid">
                    <div class="category-card">
                        <h4>🛡️ Safety Concern</h4>
                        <p>Safety hazards, emergencies, and workplace risks</p>
                    </div>
                    <div class="category-card">
                        <h4>⚙️ Machine/Equipment Issue</h4>
                        <p>Equipment problems and maintenance needs</p>
                    </div>
                    <div class="category-card">
                        <h4>💡 Process Improvement Idea</h4>
                        <p>Suggestions and efficiency improvements</p>
                    </div>
                    <div class="category-card">
                        <h4>📋 Other</h4>
                        <p>General feedback and facility requests</p>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2><span>⚡</span>Priority Levels</h2>
                <div class="grid">
                    <div class="priority-card critical">
                        <h4>🚨 Critical</h4>
                        <p>Immediate threats requiring urgent action</p>
                    </div>
                    <div class="priority-card high">
                        <h4>🔥 High</h4>
                        <p>Important issues needing prompt attention</p>
                    </div>
                    <div class="priority-card medium">
                        <h4>⚠️ Medium</h4>
                        <p>Issues to address in reasonable timeframe</p>
                    </div>
                    <div class="priority-card low">
                        <h4>📝 Low</h4>
                        <p>Suggestions and non-urgent improvements</p>
                    </div>
                </div>
            </div>

            <footer>
                <p>Enhanced Feedback Classification API • Streamlined for manufacturing environments</p>
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

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
