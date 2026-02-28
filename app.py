from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# ===============================
# LOAD MODEL
# ===============================

model = joblib.load("saved_models/best_model.pkl")
scaler = joblib.load("saved_models/scaler.pkl")
label_encoder = joblib.load("saved_models/label_encoder.pkl")

# ===============================
# SKILL LIST
# ===============================

SKILL_NAMES = [
    "Python","Java","JavaScript","SQL",
    "DSA","Cloud","DevOps","Networking",
    "DataAnalysis","Communication"
]

# ===============================
# FULL ROLE REQUIREMENTS (ALL 8 ROLES)
# ===============================

ROLE_REQUIREMENTS = {

    "Backend Developer": {
        "core": {"Python":4,"DSA":4,"SQL":4},
        "important": {"Cloud":3,"DevOps":3,"Networking":3},
        "support": {
            "Java":3,"JavaScript":2,
            "DataAnalysis":2,"Communication":3
        }
    },

    "Frontend Developer": {
        "core": {"JavaScript":5,"Communication":4},
        "important": {"DSA":3,"SQL":2,"Java":4},
        "support": {
            "Python":2,"Cloud":2,"DevOps":1,
            "Networking":1,"DataAnalysis":1
        }
    },

    "Data Analyst": {
        "core": {"SQL":5,"DataAnalysis":5},
        "important": {"Python":4,"Communication":3},
        "support": {
            "DSA":3,"Cloud":2,"Networking":2,
            "DevOps":1,"Java":1,"JavaScript":2
        }
    },

    "Data Scientist": {
        "core": {"Python":5,"DataAnalysis":5,"DSA":4},
        "important": {"SQL":4,"Cloud":3},
        "support": {
            "DevOps":2,"Networking":2,
            "Communication":3,"Java":1,"JavaScript":2
        }
    },

    "Cloud Engineer": {
        "core": {"Cloud":5,"Networking":5},
        "important": {"DevOps":4,"Python":3},
        "support": {
            "SQL":3,"DSA":2,"Java":2,
            "JavaScript":1,"DataAnalysis":2,"Communication":3
        }
    },

    "DevOps Engineer": {
        "core": {"DevOps":5,"Cloud":4},
        "important": {"Networking":4,"Python":3},
        "support": {
            "SQL":3,"DSA":2,"Java":2,
            "JavaScript":2,"DataAnalysis":2,"Communication":3
        }
    },

    "Mobile Developer": {
        "core": {"Java":5,"DSA":4},
        "important": {"SQL":3,"Communication":3},
        "support": {
            "Python":2,"JavaScript":2,"Cloud":2,
            "DevOps":2,"Networking":2,"DataAnalysis":1
        }
    },

    "Cybersecurity Analyst": {
        "core": {"Networking":5,"Python":4},
        "important": {"Cloud":3,"DevOps":3},
        "support": {
            "DSA":3,"SQL":3,"Communication":3,
            "Java":2,"JavaScript":2,"DataAnalysis":2
        }
    }
}

# ===============================
# PRACTICE PROBLEMS
# ===============================

PROBLEMS = {
    "Backend Developer": [
        {"title": "Two Sum", "difficulty": "Easy", "link": "https://leetcode.com/problems/two-sum/"},
        {"title": "LRU Cache", "difficulty": "Hard", "link": "https://leetcode.com/problems/lru-cache/"}
    ],
    "Frontend Developer": [
        {"title": "Valid Parentheses", "difficulty": "Easy", "link": "https://leetcode.com/problems/valid-parentheses/"}
    ],
    "Data Analyst": [
        {"title": "SQL Practice", "difficulty": "Medium", "link": "https://leetcode.com/problemset/database/"}
    ],
    "Data Scientist": [
        {"title": "Kth Largest Element", "difficulty": "Medium", "link": "https://leetcode.com/problems/kth-largest-element-in-an-array/"}
    ],
    "Cloud Engineer": [
        {"title": "Linux Basics", "difficulty": "Easy", "link": "https://www.hackerrank.com/domains/shell"}
    ],
    "DevOps Engineer": [
        {"title": "Shell Scripting", "difficulty": "Medium", "link": "https://www.hackerrank.com/domains/shell"}
    ],
    "Mobile Developer": [
        {"title": "Reverse Linked List", "difficulty": "Easy", "link": "https://leetcode.com/problems/reverse-linked-list/"}
    ],
    "Cybersecurity Analyst": [
        {"title": "Basic Cryptography", "difficulty": "Medium", "link": "https://www.hackerrank.com"}
    ]
}

# ===============================
# HOME ROUTE
# ===============================

@app.route("/")
def home():
    return render_template("form.html", roles=ROLE_REQUIREMENTS.keys())

# ===============================
# MAIN ANALYSIS ROUTE
# ===============================

@app.route("/analyze", methods=["POST"])
def analyze():

    mode = request.form["mode"]

    features = [float(request.form[skill]) for skill in SKILL_NAMES]
    skill_dict = dict(zip(SKILL_NAMES, features))

    # ===============================
    # MODE 1 - ML RECOMMENDATION
    # ===============================

    if mode == "recommend":

        # Prepare input for ML
        input_array = np.array([features + [0,0,0,0,0]])
        scaled_input = scaler.transform(input_array)

        probabilities = model.predict_proba(scaled_input)[0]
        top_indices = probabilities.argsort()[-3:][::-1]

        top_roles = []
        readiness_map = {}

        for idx in top_indices:
            role_name = label_encoder.inverse_transform([idx])[0]
            prob_percent = round(probabilities[idx] * 100, 2)

            top_roles.append((role_name, prob_percent))

             # -----------------------------
             # CALCULATE READINESS FOR EACH
             # -----------------------------

            requirements_temp = ROLE_REQUIREMENTS.get(role_name, {})
            core_temp = requirements_temp.get("core", {})
            important_temp = requirements_temp.get("important", {})
            support_temp = requirements_temp.get("support", {})

            weighted_score = 0
            weighted_max = 0
            core_gap = False

            # Core weight 3
            for skill, req in core_temp.items():
                weighted_max += 3
                if skill_dict[skill] >= req:
                    weighted_score += 3
                else:
                    core_gap = True

            # Important weight 2
            for skill, req in important_temp.items():
                weighted_max += 2
                if skill_dict[skill] >= req:
                    weighted_score += 2

            # Support weight 1
            for skill, req in support_temp.items():
                weighted_max += 1
                if skill_dict[skill] >= req:
                    weighted_score += 1

            readiness_temp = (weighted_score / weighted_max) * 100 if weighted_max > 0 else 0

            if core_gap:
                readiness_temp *= 0.6

            readiness_map[role_name] = round(readiness_temp, 2)

        # Primary recommended role
        role = top_roles[0][0]
        confidence = top_roles[0][1]
        readiness = readiness_map[role]

    else:
        role = request.form["target_role"]
        confidence = None
        top_roles = None
        readiness_map = None

    # ===============================
    # GET REQUIREMENTS
    # ===============================

    requirements = ROLE_REQUIREMENTS.get(role, {})
    core = requirements.get("core", {})
    important = requirements.get("important", {})
    support = requirements.get("support", {})

    # ===============================
    # BUILD RADAR REQUIRED SKILLS
    # ===============================

    required_skills = []

    for skill in SKILL_NAMES:
        if skill in core:
            required_skills.append(core[skill])
        elif skill in important:
            required_skills.append(important[skill])
        elif skill in support:
            required_skills.append(support[skill])
        else:
            required_skills.append(1)

    # ===============================
    # GAP ANALYSIS
    # ===============================

    gaps = []
    all_requirements = {**core, **important, **support}

    for skill, req in all_requirements.items():
        if skill_dict[skill] < req:
            gaps.append((skill, skill_dict[skill], req))

    # ===============================
    # WEIGHTED READINESS
    # ===============================

    weighted_score = 0
    weighted_max = 0
    core_gap = False

    # Core (weight 3)
    for skill, req in core.items():
        weighted_max += 3
        if skill_dict[skill] >= req:
            weighted_score += 3
        else:
            core_gap = True

    # Important (weight 2)
    for skill, req in important.items():
        weighted_max += 2
        if skill_dict[skill] >= req:
            weighted_score += 2

    # Support (weight 1)
    for skill, req in support.items():
        weighted_max += 1
        if skill_dict[skill] >= req:
            weighted_score += 1

    readiness = (weighted_score / weighted_max) * 100 if weighted_max > 0 else 0

    # Core penalty
    if core_gap:
        readiness *= 0.6

    readiness = round(readiness,2)

    explanation = None

    if mode == "recommend":

        requirements_main = ROLE_REQUIREMENTS.get(role, {})
        core_main = requirements_main.get("core", {})

        strong_core_skills = []

        for skill, req in core_main.items():
            if skill_dict[skill] >= req:
                strong_core_skills.append(skill)

        if strong_core_skills:
            explanation = (
                f"This role is recommended because your strong skills in "
                f"{', '.join(strong_core_skills[:2])} align with its core requirements."
            )
        else:
            explanation = (
                "This role matches your overall profile, but improvement in core skills is recommended."
            )
    # ===============================

    return render_template(
    "analysis_result.html",
    mode=mode,
    role=role,
    confidence=confidence,
    top_roles=top_roles,
    readiness=readiness,
    readiness_map=readiness_map,
    user_skills=features,
    required_skills=required_skills,
    gaps=gaps,
    explanation=explanation,
    problems=PROBLEMS.get(role, [])
)

@app.route("/complete", methods=["POST"])
def complete():

    completed = request.form.getlist("completed")
    role = request.form["role"]

    total = len(PROBLEMS.get(role, []))
    done = len(completed)

    progress = (done / total) * 100 if total > 0 else 0

    return f"""
    <h2>{role} Practice Progress</h2>
    <h3>{round(progress,2)}% Completed</h3>
    <p>You completed {done} out of {total} problems.</p>
    <br><a href="/">Go Back</a>
    """

# ===============================
# RUN
# ===============================

if __name__ == "__main__":
    app.run(debug=True)