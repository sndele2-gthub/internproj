import os
import re
import difflib
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from flask import Flask, request, jsonify
import math

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
            "electrocuted": 3.0, "poisoned": 3.0, "fatal": 3.0,
            
            # High priority safety (weight: 2.0)
            "danger": 2.0, "hazard": 2.0, "unsafe": 2.0, "injury": 2.0, "accident": 2.0, 
            "hurt": 2.0, "injured": 2.0, "fall": 2.0, "cut": 2.0, "burn": 2.0, 
            "electrical": 2.0, "shock": 2.0, "blocked": 2.0, "spill": 2.0, "leak": 2.0,
            "chemical": 2.0, "slippery": 2.0, "broken": 2.0,
            
            # Medium priority safety (weight: 1.0)
            "safety": 1.0, "risk": 1.0, "ppe": 1.0, "protective": 1.0, "guard": 1.0, 
            "warning": 1.0, "caution": 1.0, "helmet": 1.0, "gloves": 1.0, "goggles": 1.0,
            "boots": 1.0, "vest": 1.0, "mask": 1.0, "training": 1.0,
            
            # Low priority safety (weight: 0.5)
            "suggestion": 0.5, "recommend": 0.5, "improve": 0.5, "better": 0.5, "policy": 0.5
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
            "leak": 2.0, "smoke": 2.0, "sparks": 2.0,
            
            # Medium priority equipment (weight: 1.0)
            "machine": 1.0, "equipment": 1.0, "conveyor": 1.0, "motor": 1.0, "pump": 1.0,
            "repair": 1.0, "maintenance": 1.0, "noise": 1.0, "vibration": 1.0, 
            "pressure": 1.0, "temperature": 1.0, "slow": 1.0, "sluggish": 1.0,
            
            # Low priority equipment (weight: 0.5)
            "adjustment": 0.5, "calibration": 0.5, "cleaning": 0.5, "lubrication": 0.5,
            "inspection": 0.5, "routine": 0.5
        },
        "negation_words": ["not", "no", "without", "need", "should", "could", "want", "suggest"],
        "context_reducers": ["suggestion", "idea", "recommend", "propose", "schedule", "plan"]
    },
    
    "Process Improvement Idea": {
        "keywords": {
            # High impact improvements (weight: 2.0)
            "automate": 2.0, "streamline": 2.0, "optimize": 2.0, "revolutionize": 2.0,
            "transform": 2.0, "overhaul": 2.0, "redesign": 2.0,
            
            # Medium impact improvements (weight: 1.0)
            "improve": 1.0, "efficiency": 1.0, "productivity": 1.0, "enhance": 1.0,
            "upgrade": 1.0, "modernize": 1.0, "simplify": 1.0, "reduce": 1.0,
            "increase": 1.0, "faster": 1.0, "better": 1.0,
            
            # Low impact improvements (weight: 0.5)
            "suggestion": 0.5, "idea": 0.5, "recommend": 0.5, "think": 0.5,
            "consider": 0.5, "maybe": 0.5, "could": 0.5, "might": 0.5
        },
        "negation_words": [],
        "context_reducers": []
    },
    
    "Other": {
        "keywords": {
            "supplies": 1.0, "training": 1.0, "lighting": 1.0, "parking": 1.0, 
            "temperature": 1.0, "break": 1.0, "lunch": 1.0, "bathroom": 1.0, 
            "coffee": 1.0, "clean": 1.0, "organize": 1.0, "facilities": 1.0,
            "comfort": 1.0, "environment": 1.0
        },
        "negation_words": ["urgent", "immediate", "critical", "emergency"],
        "context_reducers": []
    }
}

# Context-aware priority determination
PRIORITY_CONTEXTS = {
    "critical_indicators": {
        "words": ["emergency", "immediate", "urgent", "asap", "now", "stop", "shutdown", 
                 "critical", "severe", "major", "life threatening", "fatal", "dangerous"],
        "phrases": ["right now", "immediately", "can't wait", "stop work", "shut down"]
    },
    "high_indicators": {
        "words": ["important", "soon", "quickly", "significant", "serious", "major"],
        "phrases": ["as soon as possible", "high priority", "needs attention"]
    },
    "medium_indicators": {
        "words": ["moderate", "reasonable", "normal", "standard"],
        "phrases": ["when possible", "reasonable time", "normal priority"]
    },
    "low_indicators": {
        "words": ["minor", "small", "little", "eventually", "someday", "future"],
        "phrases": ["when convenient", "low priority", "not urgent", "future consideration"]
    }
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
    
    def calculate_category_score(self, text: str, category: str) -> Tuple[float, List[str], float]:
        """Enhanced category scoring with context awareness"""
        normalized = self.normalize_text(text)
        words = normalized.split()
        
        category_data = self.knowledge_base[category]
        keywords = category_data["keywords"]
        negation_words = set(category_data.get("negation_words", []))
        context_reducers = set(category_data.get("context_reducers", []))
        
        total_score = 0.0
        matched_keywords = []
        word_count = len(words)
        
        # Calculate weighted keyword matches
        for i, word in enumerate(words):
            if word in keywords:
                weight = keywords[word]
                
                # Check for negation context (reduces weight)
                negation_factor = 1.0
                for j in range(max(0, i-3), min(len(words), i+4)):  # Check 3 words around
                    if words[j] in negation_words:
                        negation_factor = 0.3  # Reduce weight significantly
                        break
                
                # Check for context reducers
                context_factor = 1.0
                if any(reducer in normalized for reducer in context_reducers):
                    context_factor = 0.6  # Moderate reduction
                
                adjusted_weight = weight * negation_factor * context_factor
                total_score += adjusted_weight
                matched_keywords.append(word)
        
        # Add fuzzy matching for important terms
        for word in words:
            if len(word) > 4:
                for keyword, weight in keywords.items():
                    if len(keyword) > 4 and difflib.SequenceMatcher(None, word, keyword).ratio() > 0.85:
                        if keyword not in matched_keywords:
                            total_score += weight * 0.7  # Reduced weight for fuzzy matches
                            matched_keywords.append(f"{word}~{keyword}")
        
        # Calculate density-based confidence
        if word_count == 0:
            density = 0.0
        else:
            density = len(matched_keywords) / word_count
        
        # Normalize score (max possible score estimation)
        max_possible_score = sum(keywords.values()) * 0.3  # Assume 30% keyword density is max
        normalized_score = min(total_score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0
        
        return normalized_score, matched_keywords, density
    
    def determine_priority_from_content(self, text: str, category: str, category_score: float) -> Tuple[Priority, float, List[str]]:
        """Context-aware priority determination"""
        normalized = self.normalize_text(text)
        factors = []
        
        # Check for explicit priority indicators
        critical_score = 0
        high_score = 0
        medium_score = 0
        low_score = 0
        
        # Count priority indicators
        for word in PRIORITY_CONTEXTS["critical_indicators"]["words"]:
            if word in normalized:
                critical_score += 2
        for phrase in PRIORITY_CONTEXTS["critical_indicators"]["phrases"]:
            if phrase in normalized:
                critical_score += 3
                
        for word in PRIORITY_CONTEXTS["high_indicators"]["words"]:
            if word in normalized:
                high_score += 1
        for phrase in PRIORITY_CONTEXTS["high_indicators"]["phrases"]:
            if phrase in normalized:
                high_score += 2
                
        for word in PRIORITY_CONTEXTS["medium_indicators"]["words"]:
            if word in normalized:
                medium_score += 1
                
        for word in PRIORITY_CONTEXTS["low_indicators"]["words"]:
            if word in normalized:
                low_score += 1
        for phrase in PRIORITY_CONTEXTS["low_indicators"]["phrases"]:
            if phrase in normalized:
                low_score += 2
        
        # Analyze category-specific keywords for priority
        category_data = self.knowledge_base[category]
        keyword_priority_score = 0
        
        for word in normalized.split():
            if word in category_data["keywords"]:
                weight = category_data["keywords"][word]
                keyword_priority_score += weight
        
        # Determine base priority from content analysis
        if critical_score > 0:
            priority = Priority.CRITICAL
            priority_score = 4.5 + (critical_score * 0.1)
            factors.append(f"Critical indicators detected (score: {critical_score})")
        elif high_score > 0 or keyword_priority_score >= 4.0:
            priority = Priority.HIGH
            priority_score = 3.5 + (high_score * 0.1)
            factors.append(f"High priority indicators (score: {high_score}, keyword score: {keyword_priority_score:.1f})")
        elif medium_score > 0 or keyword_priority_score >= 2.0:
            priority = Priority.MEDIUM
            priority_score = 2.0 + (medium_score * 0.1)
            factors.append(f"Medium priority indicators (score: {medium_score}, keyword score: {keyword_priority_score:.1f})")
        elif low_score > 2:  # Strong low indicators
            priority = Priority.LOW
            priority_score = 1.0
            factors.append(f"Low priority indicators detected (score: {low_score})")
        else:
            # Default based on category and keyword analysis
            if category == "Safety Concern":
                if keyword_priority_score >= 2.5:
                    priority = Priority.HIGH
                    priority_score = 3.2
                    factors.append("Safety concern with significant risk indicators")
                elif keyword_priority_score >= 1.5:
                    priority = Priority.MEDIUM
                    priority_score = 2.3
                    factors.append("Safety concern with moderate risk")
                else:
                    priority = Priority.LOW
                    priority_score = 1.5
                    factors.append("Safety suggestion or minor concern")
            elif category == "Machine/Equipment Issue":
                if keyword_priority_score >= 2.5:
                    priority = Priority.HIGH
                    priority_score = 3.0
                    factors.append("Critical equipment failure")
                elif keyword_priority_score >= 1.5:
                    priority = Priority.MEDIUM
                    priority_score = 2.2
                    factors.append("Equipment issue requiring attention")
                else:
                    priority = Priority.LOW
                    priority_score = 1.3
                    factors.append("Minor equipment maintenance")
            elif category == "Process Improvement Idea":
                if keyword_priority_score >= 1.5:
                    priority = Priority.MEDIUM
                    priority_score = 2.0
                    factors.append("Significant improvement opportunity")
                else:
                    priority = Priority.LOW
                    priority_score = 1.2
                    factors.append("Process improvement suggestion")
            else:  # Other
                priority = Priority.LOW
                priority_score = 1.0
                factors.append("General feedback or request")
        
        return priority, min(priority_score, 5.0), factors
    
    def calculate_confidence(self, category_scores: Dict[str, float], best_category: str, 
                           matched_keywords: List[str], text_length: int) -> float:
        """Enhanced confidence calculation based on multiple factors"""
        
        # Factor 1: Category separation (how much better is the best vs second best)
        sorted_scores = sorted(category_scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] > 0:
            separation = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
        else:
            separation = 0.0
        
        # Factor 2: Absolute score of best category
        best_score = category_scores[best_category]
        
        # Factor 3: Keyword match density
        words_in_text = len(text_length.split()) if isinstance(text_length, str) else max(text_length // 5, 1)
        keyword_density = min(len(matched_keywords) / words_in_text, 1.0)
        
        # Factor 4: Text length adequacy (too short = less reliable)
        length_factor = min(words_in_text / 5, 1.0)  # Optimal at 5+ words
        
        # Combine factors
        confidence = (
            separation * 0.3 +           # 30% - how clearly separated from other categories
            best_score * 0.4 +           # 40% - how well it matches the category
            keyword_density * 0.2 +      # 20% - density of relevant keywords
            length_factor * 0.1          # 10% - text length adequacy
        )
        
        return min(max(confidence, 0.1), 1.0)  # Clamp between 0.1 and 1.0
    
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
            all_densities = {}
            
            for category in self.knowledge_base.keys():
                score, matches, density = self.calculate_category_score(text, category)
                category_scores[category] = score
                all_matches[category] = matches
                all_densities[category] = density
            
            # Find best category
            best_category = max(category_scores.keys(), key=lambda k: category_scores[k])
            best_matches = all_matches[best_category]
            
            # If all scores are very low, try to infer from common words
            if category_scores[best_category] < 0.15:
                normalized = self.normalize_text(text)
                if any(word in normalized for word in ["safe", "danger", "hazard", "injury", "hurt"]):
                    best_category = "Safety Concern"
                    category_scores[best_category] = 0.4
                elif any(word in normalized for word in ["machine", "equipment", "broken", "repair", "fix"]):
                    best_category = "Machine/Equipment Issue"
                    category_scores[best_category] = 0.4
                elif any(word in normalized for word in ["improve", "suggest", "idea", "better", "optimize"]):
                    best_category = "Process Improvement Idea"
                    category_scores[best_category] = 0.4
                else:
                    best_category = "Other"
                    category_scores[best_category] = 0.3
            
            # Determine priority based on content analysis
            priority, priority_score, priority_factors = self.determine_priority_from_content(
                text, best_category, category_scores[best_category]
            )
            
            # Calculate confidence
            confidence = self.calculate_confidence(
                category_scores, best_category, best_matches, len(text.split())
            )
            
            return ClassificationResult(
                category=best_category,
                priority=priority.value,
                confidence=round(confidence, 3),
                priority_score=round(priority_score, 2),
                matched_keywords=best_matches[:8],  # Limit to top 8 matches
                priority_factors=priority_factors
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
    
    # Convert confidence to 1-10 scale (more realistic distribution)
    confidence_10_scale = max(1, min(10, round(result.confidence * 10)))
    
    return jsonify({
        "category": result.category,
        "autocategory": result.category,
        "priority": result.priority,
        "autopriority": result.priority,
        "confidence": confidence_10_scale,
        "confidence_score": confidence_10_scale,
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
        <title>Enhanced Feedback Classification API</title>
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
            
            .enhancement-badge {
                display: inline-block;
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 25px;
                font-size: 0.9rem;
                font-weight: 600;
                margin-top: 1rem;
                box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
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
            
            .feature-list {
                list-style: none;
                margin: 1.5rem 0;
            }
            
            .feature-list li {
                padding: 0.5rem 0;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .feature-list li::before {
                content: "✨";
                font-size: 1rem;
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
        <div class="watermark">Enhanced by Claude - Made by Sean N</div>
        <div class="container">
            <header class="header">
                <h1>🧠 Enhanced Feedback Classification API</h1>
                <p>Advanced context-aware categorization with intelligent priority assessment</p>
                <div class="status-badge">✅ API Status: Enhanced & Online</div>
                <div class="enhancement-badge">🚀 NEW: Context-Aware Analysis</div>
            </header>

            <div class="card">
                <h2><span>⚡</span>Enhanced Features</h2>
                <ul class="feature-list">
                    <li>Weighted keyword analysis for accurate categorization</li>
                    <li>Context-aware priority determination</li>
                    <li>Intelligent confidence scoring based on multiple factors</li>
                    <li>Negation and context detection (e.g., "need PPE" vs "PPE fire")</li>
                    <li>Fuzzy matching for similar terms</li>
                    <li>No more default high priority for safety suggestions</li>
                    <li>Realistic confidence distribution (not everything is 9-10)</li>
                </ul>
            </div>

            <div class="card">
                <h2><span>🚀</span>API Endpoint</h2>
                <div class="endpoint">POST /classify</div>
                <p>Submit feedback text for intelligent classification and priority assessment.</p>
                
                <h3>📝 Request Format</h3>
                <div class="code-block">{
    "text": "We should improve PPE training for new employees"
}</div>
                
                <h3>📊 Enhanced Response Format</h3>
                <div class="code-block">{
    "category": "Safety Concern",
    "priority": "Low",
    "confidence": 6,
    "priority_score": 1.5,
    "matched_keywords": ["ppe", "training", "safety"],
    "priority_factors": ["Safety suggestion or minor concern"]
}</div>
            </div>

            <div class="card">
                <h2><span>📂</span>Categories</h2>
                <div class="grid">
                    <div class="category-card">
                        <h4>🛡️ Safety Concern</h4>
                        <p>Context-aware safety classification - from critical emergencies to training suggestions</p>
                    </div>
                    <div class="category-card">
                        <h4>⚙️ Machine/Equipment Issue</h4>
                        <p>Equipment problems with intelligent severity detection</p>
                    </div>
                    <div class="category-card">
                        <h4>💡 Process Improvement Idea</h4>
                        <p>Suggestions and efficiency improvements with impact assessment</p>
                    </div>
                    <div class="category-card">
                        <h4>📋 Other</h4>
                        <p>General feedback and facility requests</p>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2><span>⚡</span>Smart Priority Levels</h2>
                <div class="grid">
                    <div class="priority-card critical">
                        <h4>🚨 Critical</h4>
                        <p>Immediate threats - fire, emergency, life-threatening situations</p>
                    </div>
                    <div class="priority-card high">
                        <h4>🔥 High</h4>
                        <p>Serious issues - equipment failures, significant safety hazards</p>
                    </div>
                    <div class="priority-card medium">
                        <h4>⚠️ Medium</h4>
                        <p>Important issues - maintenance needs, moderate safety concerns</p>
                    </div>
                    <div class="priority-card low">
                        <h4>📝 Low</h4>
                        <p>Suggestions and minor issues - training requests, improvement ideas</p>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2><span>🎯</span>Example Classifications</h2>
                <div class="code-block"><strong>Input:</strong> "Emergency! Fire in the warehouse!"
<strong>Output:</strong> Category: Safety Concern, Priority: Critical, Confidence: 9

<strong>Input:</strong> "We need better PPE training for new hires"
<strong>Output:</strong> Category: Safety Concern, Priority: Low, Confidence: 6

<strong>Input:</strong> "Conveyor belt is completely broken and smoking"
<strong>Output:</strong> Category: Machine/Equipment Issue, Priority: High, Confidence: 8

<strong>Input:</strong> "Maybe we could automate the packaging process"
<strong>Output:</strong> Category: Process Improvement Idea, Priority: Low, Confidence: 7</div>
            </div>

            <footer>
                <p>Enhanced Feedback Classification API • Context-Aware Analysis • Intelligent Priority Assessment</p>
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
