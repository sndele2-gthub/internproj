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

# (All other code is unchanged...)

# ... (Keep all your classes and functions here, unchanged, as before) ...

# Initialize enhanced classifier
classifier = EnhancedFeedbackClassifier(KNOWLEDGE_BASE)

def validate_request_data(data: Optional[Dict]) -> tuple[bool, str]:
    # (Unchanged...)

@app.errorhandler(400)
def bad_request(error):
    # (Unchanged...)

@app.errorhandler(500)
def internal_error(error):
    # (Unchanged...)

@app.route("/")
def home():
    # (Unchanged...)

# ---- UPDATED /classify ENDPOINT ----
@app.route("/classify", methods=["POST"])
def classify():
    """API endpoint for classifying and prioritizing feedback."""
    data = request.get_json(force=True, silent=True)
    is_valid, error_message = validate_request_data(data)
    if not is_valid:
        return jsonify({"error": error_message}), 400

    text = data.get("text", "")
    result = classifier.classify(text)

    # Add duplicate keys for compatibility with Power Automate and Excel
    result_dict = {
        "category": result.category,
        "autocategory": result.category,          # duplicate field
        "priority": result.priority,
        "autopriority": result.priority,          # duplicate field
        "confidence": result.confidence,
        "confidence_score": result.confidence,    # duplicate field
        "priority_score": result.priority_score,
        "matched_example": result.matched_example,
        "keyword_matches": result.keyword_matches,
        "priority_factors": result.priority_factors,
        "similarity_scores": result.similarity_scores,
        "error": result.error,
    }
    return jsonify(result_dict), 200

if __name__ == "__main__":
    logger.info("Starting Enhanced Feedback Classification API...")
    app.run(host="0.0.0.0", port=5000, debug=False)
