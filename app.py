import re
import difflib
import logging
from collections import defaultdict
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from flask import Flask, request, jsonify, render_template_string
from datetime import datetime, timedelta
import hashlib

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

# --- Data Models ---
class Priority(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

@dataclass
class SubmissionRecord:
    text: str
    category: str
    priority: str
    timestamp: datetime
    submission_hash: str
    is_escalated: bool = False
    confidence_score: int = 0

@dataclass
class ClassificationResult:
    category: str
    priority: str
    confidence: int
    matched_keywords: Dict[str, List[str]] = field(default_factory=dict)
    priority_factors: List[str] = field(default_factory=list)
    is_duplicate: bool = False
    escalation_applied: bool = False
    original_priority: str = "N/A"
    similar_count: int = 0

# --- Enhanced Knowledge Base ---
MAX_TEXT_LENGTH = 5000
KNOWLEDGE_BASE = {
    "Safety Concern": {
        "critical": {
            "emergency","fire","explosion","fatal","dangerous","imminent danger","life-threatening",
            "critical injury","collapse","toxic leak","electrocution","unsafe structure","structural failure",
            "immediate danger","chemical exposure","gas leak","confined space","fall from height"
        },
        "high": {
            "hazard","unsafe","accident","injury risk","fall risk","structural damage","chemical spill",
            "electrical issue","no safety gear","blocked exit","safety violation","exposed wiring","lockout failure",
            "ppe missing","guardrail damaged","near miss","serious injury","osha violation"
        },
        "medium": {
            "safety concern","risk identified","warning sign","slippery floor","trip hazard","poor visibility",
            "loud noise","minor injury","first aid needed","broken glass","spill","unsecured item","light out",
            "ergonomic issue","housekeeping","access issue","signage missing"
        },
        "negation": {"not","no","without","lacking","un-","non-","safe","clear","resolved","fixed","repaired","insignificant","minimal risk","controlled"},
    },
    "Machine/Equipment Issue": {
        "critical": {
            "complete failure","total shutdown","catastrophic","unusable","major breakdown","production halt",
            "burst pipe","electrical short","machine dead","severely damaged","irreparable","system crash",
            "emergency stop","production line down","critical equipment failure"
        },
        "high": {
            "malfunction","down","stopped working","leaking fluid","faulty","error code","damaged","overheating",
            "smoking","intermittent failure","broken part","no power","offline","bearing failure",
            "motor burned","hydraulic failure","control system down","production impact"
        },
        "medium": {
            "noise","vibration","loose part","maintenance needed","humming","grinding","stuck","press issue",
            "adjustment required","defect","calibration","worn out","slow performance","clogged","filter",
            "not working","sensor issue","alignment","belt loose","preventive maintenance"
        },
        "negation": {"not","no","without","un-","non-","working","functional","repaired","fixed","normal operation","running smoothly","operating normally","restored"},
    },
    "Process Improvement": {
        "critical": {
            "bottleneck","critical delay","major inefficiency","costly error","legal non-compliance","regulatory violation",
            "audit failure","severe waste","compliance breach","quality failure","customer complaint","product recall",
            "safety violation","environmental violation"
        },
        "high": {
            "automate","streamline","optimize","reduce waste","cost saving","quality improvement","significant inefficiency",
            "redundant steps","data inaccuracy","improve workflow","new procedure","expedite","better method",
            "eliminate waste","lean implementation","six sigma opportunity"
        },
        "medium": {
            "improve","suggestion","idea","process enhancement","workflow improvement","better method","new system",
            "simplify","training need","communication gap","feedback mechanism","documentation","best practice",
            "continuous improvement","kaizen","procedure update"
        },
        "negation": {"not","no","without","current process is fine","working well","efficient","optimal","no improvement needed","satisfactory","adequate"},
    },
    "Quality Issue": {
        "critical": {"product defect","quality failure","out of spec","customer reject","recall","major nonconformance","critical defect","safety defect","regulatory failure"},
        "high": {"quality issue","defective","rework","scrap","inspection failure","tolerance exceeded","material defect","process variation","quality alert"},
        "medium": {"minor defect","cosmetic issue","quality observation","improvement opportunity","quality suggestion","process monitoring","documentation issue"},
        "negation": {"quality good","meets spec","acceptable quality","no defects","passed inspection"},
    },
    "Other": {
        "medium": {"general inquiry","feedback","suggestion","question","comment","miscellaneous","supplies","lighting","parking","facility","administrative","communication"},
        "negation": set(),  # ensure it's a set for checks below
    }
}

# --- Enhanced Core Logic ---
class ClassifierLogic:
    def __init__(self):
        self.submissions: List[SubmissionRecord] = []
        self.retention_hours = 168
        self.similarity_threshold = 0.70
        self.escalation_threshold = 2

    def _normalize_text(self, text: str) -> List[str]:
        """Enhanced text normalization."""
        text = re.sub(r'[^\w\s\-\.]', ' ', text.lower())
        tokens = re.findall(r'\b\w+(?:-\w+)?\b', text)
        return [token for token in tokens if len(token) > 2]

    def _extract_context_clues(self, text: str) -> Dict:
        """Extract urgency and impact indicators."""
        context = {"urgency_indicators": [], "impact_scale": []}
        text_lower = text.lower()

        urgency_patterns = [
            r"\b(urgent|asap|immediate|emergency|critical|now)\b",
            r"\b(today|this morning|right now|immediately)\b"
        ]
        impact_patterns = [
            r"\b(affects?\s+\w+\s+people|multiple|several|many|all)\b",
            r"\b(department|shift|team|entire|whole)\b"
        ]

        for pattern in urgency_patterns:
            context["urgency_indicators"].extend(re.findall(pattern, text_lower))
        for pattern in impact_patterns:
            context["impact_scale"].extend(re.findall(pattern, text_lower))

        return context

    def _calculate_scores(self, tokens: List[str], context: Dict) -> Tuple[Dict, Dict]:
        """Enhanced scoring with context awareness."""
        cat_scores, matched_keys = defaultdict(float), defaultdict(lambda: defaultdict(list))

        for cat, data in KNOWLEDGE_BASE.items():
            negation_keywords = set(data.get("negation", set()))
            for i, token in enumerate(tokens):
                score, level = 0.0, None

                if token in data.get("critical", set()):
                    score, level = 3.0, "critical"
                elif token in data.get("high", set()):
                    score, level = 2.0, "high"
                elif token in data.get("medium", set()):
                    score, level = 1.0, "medium"

                if score > 0:
                    # Context boosters
                    if context["urgency_indicators"] and level in ("critical", "high"):
                        score *= 1.3
                    if context["impact_scale"] and cat in ("Safety Concern", "Machine/Equipment Issue"):
                        score *= 1.2

                    # Negation window
                    window = tokens[max(0, i - 3): i + 4]
                    if not any(t in negation_keywords for t in window):
                        cat_scores[cat] += score
                        matched_keys[cat][level].append(token)

        return dict(cat_scores), dict(matched_keys)

    def _determine_priority(self, scores: Dict, text: str, context: Dict) -> Tuple[Priority, List[str]]:
        """Enhanced priority determination."""
        factors, text_lower = [], text.lower()

        # Explicit/implicit priority patterns
        explicit_patterns = [
            (r"impact level:\s*(critical|severe)", Priority.CRITICAL),
            (r"impact level:\s*(significant|high)", Priority.HIGH),
            (r"impact level:\s*(moderate|medium)", Priority.MEDIUM),
            (r"impact level:\s*(minimal|low)", Priority.LOW),
            (r"\b(life.?threatening|immediate.?danger)\b", Priority.CRITICAL),
            (r"\b(critical|emergency|urgent)\b", Priority.HIGH),
        ]
        for pattern, priority in explicit_patterns:
            if re.search(pattern, text_lower):
                factors.append(f"Explicit priority indicator: {pattern}")
                return priority, factors

        # Context multipliers
        urgency_mult = 1.4 if context["urgency_indicators"] else 1.0
        impact_mult = 1.2 if context["impact_scale"] else 1.0
        if context["urgency_indicators"]:
            factors.append(f"Urgency indicators: {context['urgency_indicators']}")
        if context["impact_scale"]:
            factors.append(f"Impact scale indicators: {context['impact_scale']}")

        # Category scoring
        safety_equipment = (
            scores.get("Safety Concern", 0) * 1.6 +
            scores.get("Machine/Equipment Issue", 0) * 1.3 +
            scores.get("Quality Issue", 0) * 1.1
        ) * urgency_mult * impact_mult

        if safety_equipment >= 4.0:
            factors.append(f"Critical safety/equipment score: {safety_equipment:.2f}")
            return Priority.CRITICAL, factors
        elif safety_equipment >= 2.5:
            factors.append(f"High priority indicators: {safety_equipment:.2f}")
            return Priority.HIGH, factors
        elif any(s > 0.8 for s in scores.values()):
            factors.append("Medium priority indicators detected")
            return Priority.MEDIUM, factors
        else:
            factors.append("Low priority - minimal indicators")
            return Priority.LOW, factors

    def _calculate_confidence(self, scores: Dict, best_category: str, context: Dict, matched_keywords: Dict) -> int:
        """Enhanced confidence calculation."""
        base = scores.get(best_category, 0.0)
        if base <= 0.3: conf = 1
        elif base <= 0.8: conf = 2
        elif base <= 1.5: conf = 4
        elif base <= 2.5: conf = 6
        elif base <= 4.0: conf = 8
        else: conf = 9

        boosts = 0
        if len([v for v in matched_keywords.values() if v]) >= 2: boosts += 1
        if (len(context["urgency_indicators"]) + len(context["impact_scale"])) >= 2: boosts += 1
        if best_category != "Other": boosts += 1
        if any("critical" in str(v) for v in matched_keywords.values()): boosts += 2

        return max(1, min(10, conf + boosts))

    def _analyze_duplicates(self, text: str, category: str, priority: str, confidence: int) -> Tuple[bool, bool, int, str, str]:
        """Enhanced duplicate analysis."""
        # Trim old
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        self.submissions = [s for s in self.submissions if s.timestamp > cutoff]

        new_hash = hashlib.md5(text.lower().encode()).hexdigest()
        similar_count = 0
        for s in self.submissions:
            if s.submission_hash == new_hash:
                similar_count += 1
                continue
            if s.category == category:
                ratio = difflib.SequenceMatcher(None, text.lower(), s.text.lower()).ratio()
                if ratio > self.similarity_threshold:
                    similar_count += 1

        is_dup = similar_count > 0
        escalated = False
        final_prio = priority
        original_prio = priority

        if priority in (Priority.LOW.value, Priority.MEDIUM.value) and similar_count >= self.escalation_threshold:
            escalated = True
            final_prio = Priority.HIGH.value if priority == Priority.LOW.value else Priority.CRITICAL.value
        elif priority == Priority.CRITICAL.value and similar_count > 0:
            escalated = True

        self.submissions.append(
            SubmissionRecord(
                text=text,
                category=category,
                priority=final_prio,
                timestamp=datetime.now(),
                submission_hash=new_hash,
                is_escalated=escalated,
                confidence_score=confidence,
            )
        )
        return is_dup, escalated, similar_count, final_prio, original_prio

    def classify_and_process(self, text: str) -> ClassificationResult:
        """Main enhanced classification pipeline."""
        if not text or not text.strip() or len(text) > MAX_TEXT_LENGTH:
            return ClassificationResult("Invalid Input", Priority.LOW.value, 0, {}, ["Invalid input"])

        tokens = self._normalize_text(text)
        context = self._extract_context_clues(text)
        scores, keywords = self._calculate_scores(tokens, context)

        categories_with_scores = {cat: score for cat, score in scores.items() if score > 0}
        best_category = max(categories_with_scores, key=categories_with_scores.get) if categories_with_scores else "Other"

        initial_priority, factors = self._determine_priority(scores, text, context)
        confidence = self._calculate_confidence(scores, best_category, context, keywords.get(best_category, {}))
        is_dup, escalated, similar_count, final_prio, original_prio = self._analyze_duplicates(
            text, best_category, initial_priority.value, confidence
        )

        if escalated:
            factors.append(f"Priority escalated: {original_prio} \u2192 {final_prio}")

        return ClassificationResult(
            best_category, final_prio, confidence,
            keywords.get(best_category, {}), factors,
            is_dup, escalated, original_prio, similar_count
        )

# 🔧 INSTANTIATE the classifier (this was missing and caused runtime errors)
classifier_logic = ClassifierLogic()

# --- HTML Interface Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Smurfit WestRock - Feedback Classification</title>
    <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root{--primary:#0066CC;--secondary:#004B99;--success:#00A651;--gray:#F5F5F5;--text:#333;--critical:#DC3545;--high:#FFA500;--medium:#17A2B8;--low:#6C757D}
        *{margin:0;padding:0;box-sizing:border-box}body{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,var(--gray),#fff);color:var(--text);line-height:1.6}
        .header{background:linear-gradient(135deg,var(--primary),var(--secondary));color:#fff;padding:1.5rem 0;box-shadow:0 4px 12px rgba(0,102,204,0.2)}
        .header-content{max-width:1200px;margin:0 auto;padding:0 2rem;display:flex;align-items:center;gap:1rem}
        .logo{width:50px;height:50px;background:#fff;border-radius:8px;display:flex;align-items:center;justify-content:center;color:var(--primary);font-size:1.5rem;font-weight:bold}
        .container{max-width:1200px;margin:0 auto;padding:2rem;display:grid;grid-template-columns:1fr 400px;gap:2rem}
        .card{background:#fff;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.08);overflow:hidden}
        .card-header{padding:1.5rem;font-weight:600;display:flex;align-items:center;gap:0.75rem}
        .card-content{padding:2rem}.form-group{margin-bottom:1.5rem}.label{display:block;font-weight:600;margin-bottom:0.5rem}
        .textarea{width:100%;min-height:180px;padding:1rem;border:2px solid #E0E0E0;border-radius:8px;font-family:inherit;resize:vertical}
        .textarea:focus{outline:none;border-color:var(--primary);box-shadow:0 0 0 3px rgba(0,102,204,0.1)}
        .btn{padding:0.875rem 2rem;border:none;border-radius:8px;font-weight:600;cursor:pointer;transition:all 0.3s;display:inline-flex;align-items:center;gap:0.5rem}
        .btn-primary{background:linear-gradient(135deg,var(--primary),var(--secondary));color:#fff}.btn-primary:hover{transform:translateY(-2px);box-shadow:0 6px 20px rgba(0,102,204,0.3)}
        .btn-secondary{background:#E0E0E0;color:var(--text)}.btn:disabled{opacity:0.6;cursor:not-allowed;transform:none!important}
        .button-group{display:flex;gap:1rem;margin-top:1.5rem}.results{display:flex;flex-direction:column;gap:1.5rem}
        .metric-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1rem 0}
        .metric{text-align:center;padding:1rem;background:var(--gray);border-radius:8px;border-left:4px solid var(--primary)}
        .metric-value{font-size:1.5rem;font-weight:bold;color:var(--primary);display:block}.metric-label{font-size:0.85rem;color:#666;margin-top:0.25rem}
        .tag-container{display:flex;flex-wrap:wrap;gap:0.5rem;margin:0.75rem 0}
        .tag{padding:0.25rem 0.75rem;border-radius:16px;font-size:0.8rem;font-weight:500}
        .tag-critical{background:rgba(220,53,69,0.1);color:var(--critical);border:1px solid rgba(220,53,69,0.3)}
        .tag-high{background:rgba(255,165,0,0.1);color:var(--high);border:1px solid rgba(255,165,0,0.3)}
        .tag-medium{background:rgba(23,162,184,0.1);color:var(--medium);border:1px solid rgba(23,162,184,0.3)}
        .priority-critical{background:var(--critical);color:#fff}.priority-high{background:var(--high);color:#fff}
        .priority-medium{background:var(--medium);color:#fff}.priority-low{background:var(--low);color:#fff}
        .alert{padding:1rem;border-radius:8px;margin:1rem 0;border:1px solid}
        .alert-error{background:rgba(220,53,69,0.1);border-color:var(--critical);color:var(--critical)}
        .alert-success{background:rgba(40,167,69,0.1);border-color:var(--success);color:var(--success)}
        .confidence-bar{width:100%;height:8px;background:#E0E0E0;border-radius:4px;margin:0.5rem 0}
        .confidence-fill{height:100%;background:linear-gradient(90deg,var(--critical) 0%,var(--high) 50%,var(--success) 100%);transition:width 0.6s ease;border-radius:4px}
        .loading{display:inline-block;width:20px;height:20px;border:3px solid #E0E0E0;border-radius:50%;border-top-color:var(--primary);animation:spin 1s ease-in-out infinite}
        @keyframes spin{to{transform:rotate(360deg)}}
        .fade-in{animation:fadeIn 0.5s ease-in}@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
        @media (max-width:768px){.container{grid-template-columns:1fr;padding:1rem}.header-content{padding:0 1rem}.button-group{flex-direction:column}}
        .footer{margin-top:2rem;padding:1rem;text-align:center;color:#666;font-size:0.9rem}
    </style>
</head>
<body>
    <header class="header">
        <div class="header-content">
            <div class="logo">SW</div>
            <div><h1>Feedback Classification System</h1><p>Smurfit WestRock Industrial Intelligence</p></div>
        </div>
    </header>
    
    <div class="container">
        <div class="card">
            <div class="card-header" style="background:var(--primary);color:#fff">📝 Submit Feedback</div>
            <div class="card-content">
                <form id="form">
                    <div class="form-group">
                        <label class="label">Describe your feedback:</label>
                        <textarea id="text" class="textarea" placeholder="Provide detailed information about your feedback, including location, urgency, and context..." maxlength="5000"></textarea>
                        <div style="text-align:right;font-size:0.85rem;color:#666;margin-top:0.5rem"><span id="count">0</span>/5000</div>
                    </div>
                    <div class="button-group">
                        <button type="submit" class="btn btn-primary" id="classify">🧠 Classify</button>
                        <button type="button" class="btn btn-secondary" id="clear">🗑️ Clear</button>
                    </div>
                </form>
                <div id="error" class="alert alert-error" style="display:none"><span id="errorText"></span></div>
            </div>
        </div>
        
        <div class="results">
            <div class="card" id="resultCard" style="display:none">
                <div class="card-header" id="resultHeader">📊 Results</div>
                <div class="card-content">
                    <div class="metric-grid">
                        <div class="metric"><span class="metric-value" id="category">-</span><div class="metric-label">Category</div></div>
                        <div class="metric"><span class="metric-value" id="priority">-</span><div class="metric-label">Priority</div></div>
                        <div class="metric"><span class="metric-value" id="confidence">-</span><div class="metric-label">Confidence</div></div>
                    </div>
                    <div class="confidence-bar"><div class="confidence-fill" id="bar" style="width:0%"></div></div>
                    <div><h4>🔑 Keywords</h4><div id="keywords" class="tag-container"></div></div>
                    <div id="duplicate" style="margin-top:1rem"></div>
                    <div id="factors" style="margin-top:1rem"></div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">📈 Statistics</div>
                <div class="card-content">
                    <div class="metric-grid">
                        <div class="metric"><span class="metric-value" id="total">-</span><div class="metric-label">Total</div></div>
                        <div class="metric"><span class="metric-value" id="escalated">-</span><div class="metric-label">Escalated</div></div>
                        <div class="metric"><span class="metric-value" id="avgConf">-</span><div class="metric-label">Avg Conf</div></div>
                    </div>
                    <button class="btn btn-secondary" id="refresh" style="width:100%;margin-top:1rem">🔄 Refresh</button>
                </div>
            </div>
        </div>
    </div>
    
    <footer class="footer">© 2025 Smurfit WestRock - Feedback Classification System v2.0</footer>
    
    <script>
        const API='/';const text=document.getElementById('text'),count=document.getElementById('count'),form=document.getElementById('form'),
        classifyBtn=document.getElementById('classify'),clearBtn=document.getElementById('clear'),error=document.getElementById('error'),
        errorText=document.getElementById('errorText'),resultCard=document.getElementById('resultCard'),
        resultHeader=document.getElementById('resultHeader'),refreshBtn=document.getElementById('refresh');
        
        document.addEventListener('DOMContentLoaded',()=>{loadStats();setupEvents()});
        
        function setupEvents(){
            text.addEventListener('input',()=>{count.textContent=text.value.length;count.style.color=text.value.length>4500?'#DC3545':text.value.length>4000?'#FFA500':'#666'});
            form.addEventListener('submit',handleSubmit);clearBtn.addEventListener('click',clearForm);refreshBtn.addEventListener('click',loadStats)
        }
        
        function handleSubmit(e){e.preventDefault();const t=text.value.trim();if(!t){showError('Please enter feedback');return}
            if(t.length>5000){showError('Text too long');return}hideError();classifyFeedback(t)}
        
        function clearForm(){text.value='';count.textContent='0';count.style.color='#666';hideError();resultCard.style.display='none'}
        
        async function classifyFeedback(t){setLoading(true);try{
            const response=await fetch(API+'classify',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})});
            if(!response.ok)throw new Error('Classification failed');const data=await response.json();displayResults(data);setTimeout(loadStats,1000)
        }catch(err){showError(err.message)}finally{setLoading(false)}}
        
        function displayResults(data){document.getElementById('category').textContent=data.category;
            document.getElementById('priority').textContent=data.priority;document.getElementById('confidence').textContent=data.confidence+'/10';
            document.getElementById('bar').style.width=(data.confidence*10)+'%';
            resultHeader.className='card-header priority-'+data.priority.toLowerCase();displayKeywords(data.matched_keywords);
            displayDuplicate(data);displayFactors(data.priority_factors);resultCard.style.display='block';resultCard.classList.add('fade-in')}
        
        function displayKeywords(kw){const container=document.getElementById('keywords');container.innerHTML='';let hasKw=false;
            ['critical','high','medium'].forEach(level=>{if(kw[level]&&kw[level].length){kw[level].forEach(k=>{
                const tag=document.createElement('span');tag.className='tag tag-'+level;tag.textContent=k;container.appendChild(tag);hasKw=true})}});
            if(!hasKw)container.innerHTML='<span class="tag" style="background:#F5F5F5;color:#666">No keywords</span>'}
        
        function displayDuplicate(data){const dup=document.getElementById('duplicate');
            if(data.is_duplicate){dup.innerHTML='<div class="alert" style="background:rgba(255,165,0,0.1);border-color:#FFA500;color:#FFA500">⚠️ <strong>Duplicate:</strong> '+data.similar_count+' similar found'+(data.escalation_applied?' (Priority escalated)':'')+'</div>'}
            else{dup.innerHTML='<div class="alert alert-success">✅ <strong>New submission</strong></div>'}}
        
        function displayFactors(factors){const f=document.getElementById('factors');
            if(factors&&factors.length){f.innerHTML='<h4>📋 Priority Factors:</h4><ul style="margin:0.5rem 0;padding-left:1.5rem">'+factors.map(factor=>'<li>'+factor+'</li>').join('')+'</ul>'}else{f.innerHTML=''}}
        
        async function loadStats(){try{const response=await fetch(API+'stats');if(!response.ok)throw new Error('Failed');
            const stats=await response.json();document.getElementById('total').textContent=stats.system_info.total_submissions_retained;
            document.getElementById('escalated').textContent=stats.submission_stats.escalated_count;
            document.getElementById('avgConf').textContent=stats.submission_stats.average_confidence||'0'
        }catch(err){document.getElementById('total').textContent='Error';document.getElementById('escalated').textContent='Error';document.getElementById('avgConf').textContent='Error'}}
        
        function setLoading(loading){classifyBtn.disabled=loading;classifyBtn.innerHTML=loading?'<span class="loading"></span> Classifying...':'🧠 Classify'}
        function showError(msg){errorText.textContent=msg;error.style.display='block'}function hideError(){error.style.display='none'}
    </script>
</body>
</html>"""

# --- Flask Routes ---
@app.route("/", methods=["GET"])
def home():
    """Serve the HTML interface."""
    return render_template_string(HTML_TEMPLATE)

@app.route("/classify", methods=["POST"])
def classify_route():
    """Enhanced classification endpoint."""
    try:
        data = request.get_json(silent=True)
        if not data or not isinstance(data.get("text"), str) or not data.get("text").strip():
            return jsonify({"error": "Invalid text input"}), 400

        result = classifier_logic.classify_and_process(data["text"])

        # Ensure matched_keywords always has the expected keys
        mk = result.matched_keywords or {}
        formatted_keywords = {
            "critical": mk.get("critical", []),
            "high": mk.get("high", []),
            "medium": mk.get("medium", []),
        }

        return jsonify({
            "category": result.category,
            "priority": result.priority,
            "confidence": result.confidence,
            "matched_keywords": formatted_keywords,
            "priority_factors": result.priority_factors,
            "is_duplicate": result.is_duplicate,
            "escalation_applied": result.escalation_applied,
            "original_priority": result.original_priority,
            "similar_count": result.similar_count
        })

    except Exception as e:
        logging.error(f"Classification error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/stats", methods=["GET"])
def stats_route():
    """Enhanced statistics endpoint."""
    try:
        classifier_logic.submissions = [
            s for s in classifier_logic.submissions
            if s.timestamp > datetime.now() - timedelta(hours=classifier_logic.retention_hours)
        ]

        total = len(classifier_logic.submissions)
        escalated = sum(1 for s in classifier_logic.submissions if s.is_escalated)

        categories = defaultdict(int)
        priorities = defaultdict(int)
        avg_confidence = 0

        for s in classifier_logic.submissions:
            categories[s.category] += 1
            priorities[s.priority] += 1
            avg_confidence += s.confidence_score

        if total > 0:
            avg_confidence = round(avg_confidence / total, 2)

        return jsonify({
            "system_info": {
                "total_submissions_retained": total,
                "retention_hours": classifier_logic.retention_hours,
                "similarity_threshold": classifier_logic.similarity_threshold,
                "escalation_threshold": classifier_logic.escalation_threshold
            },
            "submission_stats": {
                "escalated_count": escalated,
                "average_confidence": avg_confidence,
                "category_breakdown": dict(categories),
                "priority_breakdown": dict(priorities)
            }
        })

    except Exception as e:
        logging.error(f"Stats error: {e}", exc_info=True)
        return jsonify({"error": "Unable to generate statistics"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "submissions_in_memory": len(classifier_logic.submissions),
        "categories_supported": len(KNOWLEDGE_BASE)
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
