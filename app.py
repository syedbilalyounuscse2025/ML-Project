from flask import (
    Flask,
    render_template,
    request,
    send_file
)

import pandas as pd
import joblib
import sqlite3
import json
import os

from datetime import datetime


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "model/best_model.pkl"

METRICS_PATH = "model/metrics.json"

DATABASE = "predictions.db"


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# MODEL COMPONENTS
# ============================================================

preprocessor = (
    model
    .named_steps["preprocessor"]
)

classifier = (
    model
    .named_steps["classifier"]
)


feature_names = (
    preprocessor
    .get_feature_names_out()
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importances():

    # Random Forest / Decision Tree

    if hasattr(
        classifier,
        "feature_importances_"
    ):

        return classifier.feature_importances_


    # Logistic Regression

    elif hasattr(
        classifier,
        "coef_"
    ):

        return abs(
            classifier.coef_[0]
        )


    return [
        0
        for _ in feature_names
    ]


feature_importances = (
    get_feature_importances()
)


# ============================================================
# DATABASE
# ============================================================

def init_db():

    conn = sqlite3.connect(
        DATABASE
    )

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            date TEXT,

            branch TEXT,

            college_tier TEXT,

            cgpa REAL,

            backlogs INTEGER,

            coding_skills REAL,

            dsa_score REAL,

            aptitude_score REAL,

            communication_skills REAL,

            internships INTEGER,

            certifications INTEGER,

            probability REAL,

            prediction INTEGER

        )
    """)

    conn.commit()

    conn.close()


init_db()


# ============================================================
# FEATURE IMPORTANCE HELPERS
# ============================================================

def clean_feature_name(name):

    return (
        name
        .replace("num__", "")
        .replace("cat__", "")
        .replace("_", " ")
        .title()
    )


# ============================================================
# EXPLAIN PREDICTION
# ============================================================

def explain_prediction(student):

    transformed_student = (
        preprocessor.transform(student)
    )


    if hasattr(
        transformed_student,
        "toarray"
    ):

        transformed_student = (
            transformed_student.toarray()
        )


    values = (
        transformed_student[0]
    )


    explanation = []


    for feature, value, importance in zip(

        feature_names,

        values,

        feature_importances

    ):

        contribution = (
            abs(value)
            * importance
        )


        explanation.append({

            "feature":
                feature,

            "importance":
                float(importance),

            "contribution":
                float(contribution)

        })


    explanation = sorted(

        explanation,

        key=lambda x:
            x["contribution"],

        reverse=True

    )


    return explanation[:5]


# ============================================================
# CAREER ROLE RECOMMENDATION
# ============================================================

def recommend_roles(

    coding,
    dsa,
    aptitude,
    communication,
    cgpa,
    internships

):

    roles = []


    if coding >= 8 and dsa >= 7:

        roles.append(
            "Software Developer"
        )


    if (
        coding >= 7
        and dsa >= 7
        and cgpa >= 7
    ):

        roles.append(
            "Software Engineer"
        )


    if (
        aptitude >= 75
        and communication >= 7
    ):

        roles.append(
            "Data / Business Analyst"
        )


    if (
        coding >= 7
        and aptitude >= 65
    ):

        roles.append(
            "QA / Automation Engineer"
        )


    if (
        communication >= 8
        and aptitude >= 70
    ):

        roles.append(
            "Technical Support Engineer"
        )


    if not roles:

        roles.append(
            "General Entry-Level IT Role"
        )


    return roles


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )

# ==========================================
# DASHBOARD
# ==========================================

@app.route("/dashboard")
def dashboard():

    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    # Total predictions
    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM predictions
    """)

    total_predictions = cursor.fetchone()["total"]

    # Placed students
    cursor.execute("""
        SELECT COUNT(*) AS placed
        FROM predictions
        WHERE prediction = 1
    """)

    placed_students = cursor.fetchone()["placed"]

    # Less likely students
    cursor.execute("""
        SELECT COUNT(*) AS not_placed
        FROM predictions
        WHERE prediction = 0
    """)

    not_placed_students = cursor.fetchone()["not_placed"]

    # Average placement probability
    cursor.execute("""
        SELECT AVG(probability) AS average_probability
        FROM predictions
    """)

    result = cursor.fetchone()

    average_probability = result["average_probability"]

    if average_probability is None:
        average_probability = 0

    # Recent predictions
    cursor.execute("""
        SELECT *
        FROM predictions
        ORDER BY id DESC
        LIMIT 10
    """)

    recent_predictions = cursor.fetchall()

    conn.close()

    return render_template(
        "dashboard.html",

        total_predictions=total_predictions,

        placed_students=placed_students,

        not_placed_students=not_placed_students,

        average_probability=
            round(average_probability, 2),

        recent_predictions=recent_predictions
    )
# ============================================================
# PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
def predict():

    if request.method == "GET":

        return render_template(
            "index.html"
        )


    # ========================================================
    # READ FORM DATA
    # ========================================================

    try:

        branch = request.form["branch"]

        college_tier = request.form[
            "college_tier"
        ]

        cgpa = float(
            request.form["cgpa"]
        )

        backlogs = int(
            request.form["backlogs"]
        )

        coding_skills = float(
            request.form["coding_skills"]
        )

        dsa_score = float(
            request.form["dsa_score"]
        )

        aptitude_score = float(
            request.form["aptitude_score"]
        )

        communication_skills = float(
            request.form[
                "communication_skills"
            ]
        )

        internships = int(
            request.form["internships"]
        )

        certifications = int(
            request.form[
                "certifications"
            ]
        )

    except (
        ValueError,
        KeyError
    ):

        return (
            "Invalid input. "
            "Please check all fields."
        )


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    student = pd.DataFrame([{

        "branch":
            branch,

        "college_tier":
            college_tier,

        "cgpa":
            cgpa,

        "backlogs":
            backlogs,

        "coding_skills":
            coding_skills,

        "dsa_score":
            dsa_score,

        "aptitude_score":
            aptitude_score,

        "communication_skills":
            communication_skills,

        "internships":
            internships,

        "certifications":
            certifications

    }])


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    probabilities = (
        model.predict_proba(student)[0]
    )

    prediction = int(
        model.predict(student)[0]
    )


    placement_probability = (
        probabilities[1] * 100
    )

    not_placement_probability = (
        probabilities[0] * 100
    )


    # ========================================================
    # PROFILE LEVEL
    # ========================================================

    if placement_probability >= 80:

        profile_level = "Excellent"

    elif placement_probability >= 65:

        profile_level = "Strong"

    elif placement_probability >= 50:

        profile_level = "Average"

    else:

        profile_level = "Needs Improvement"


    # ========================================================
    # TOP FACTORS
    # ========================================================

    top_factors = explain_prediction(
        student
    )


    # ========================================================
    # CAREER ROLES
    # ========================================================

    recommended_roles = recommend_roles(

        coding_skills,

        dsa_score,

        aptitude_score,

        communication_skills,

        cgpa,

        internships

    )


    # ========================================================
    # STRENGTHS
    # ========================================================

    strengths = []

    improvements = []


    if cgpa >= 8:

        strengths.append(
            "Strong academic performance"
        )

    elif cgpa < 7:

        improvements.append(
            "Improve CGPA"
        )


    if coding_skills >= 8:

        strengths.append(
            "Strong coding skills"
        )

    elif coding_skills < 6:

        improvements.append(
            "Improve coding skills"
        )


    if dsa_score >= 8:

        strengths.append(
            "Strong DSA performance"
        )

    elif dsa_score < 6:

        improvements.append(
            "Practice DSA regularly"
        )


    if aptitude_score >= 75:

        strengths.append(
            "Good aptitude performance"
        )

    elif aptitude_score < 60:

        improvements.append(
            "Improve aptitude preparation"
        )


    if communication_skills >= 8:

        strengths.append(
            "Good communication skills"
        )

    elif communication_skills < 6:

        improvements.append(
            "Improve communication skills"
        )


    if internships >= 2:

        strengths.append(
            "Good internship experience"
        )

    elif internships == 0:

        improvements.append(
            "Consider completing an internship"
        )


    if certifications >= 3:

        strengths.append(
            "Good number of certifications"
        )

    elif certifications < 2:

        improvements.append(
            "Consider earning relevant certifications"
        )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    recommendations = []


    if cgpa < 8:

        recommendations.append(
            "Maintain or improve academic performance."
        )


    if coding_skills < 8:

        recommendations.append(
            "Practice programming and problem solving."
        )


    if dsa_score < 8:

        recommendations.append(
            "Practice DSA regularly for technical interviews."
        )


    if internships < 2:

        recommendations.append(
            "Try to gain practical internship experience."
        )


    if certifications < 3:

        recommendations.append(
            "Complete relevant technical certifications."
        )


    recommendations.append(
        "Build practical projects for your portfolio."
    )


    # ========================================================
    # SAVE TO DATABASE
    # ========================================================

    conn = sqlite3.connect(
        DATABASE
    )

    cursor = conn.cursor()


    cursor.execute("""
        INSERT INTO predictions (

            date,
            branch,
            college_tier,
            cgpa,
            backlogs,
            coding_skills,
            dsa_score,
            aptitude_score,
            communication_skills,
            internships,
            certifications,
            probability,
            prediction

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (

        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        branch,

        college_tier,

        cgpa,

        backlogs,

        coding_skills,

        dsa_score,

        aptitude_score,

        communication_skills,

        internships,

        certifications,

        placement_probability,

        prediction

    ))


    conn.commit()

    conn.close()


    # ========================================================
    # RESULT
    # ========================================================

    return render_template(

        "result.html",

        prediction=prediction,

        placement_probability=
            round(
                placement_probability,
                2
            ),

        not_placement_probability=
            round(
                not_placement_probability,
                2
            ),

        profile_level=
            profile_level,

        recommended_roles=
            recommended_roles,

        branch=
            branch,

        college_tier=
            college_tier,

        cgpa=
            cgpa,

        backlogs=
            backlogs,

        coding_skills=
            coding_skills,

        dsa_score=
            dsa_score,

        aptitude_score=
            aptitude_score,

        communication_skills=
            communication_skills,

        internships=
            internships,

        certifications=
            certifications,

        strengths=
            strengths,

        improvements=
            improvements,

        recommendations=
            recommendations,

        top_factors=
            top_factors

    )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():

    conn = sqlite3.connect(
        DATABASE
    )

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()


    cursor.execute("""
        SELECT *
        FROM predictions
        ORDER BY id DESC
    """)


    records = cursor.fetchall()

    conn.close()


    return render_template(

        "history.html",

        records=records

    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

@app.route(
    "/feature-importance"
)
def feature_importance():

    feature_data = []


    for name, importance in zip(

        feature_names,

        feature_importances

    ):

        feature_data.append({

            "feature":
                name,

            "display_name":
                clean_feature_name(name),

            "importance":
                float(importance)

        })


    feature_data.sort(

        key=lambda x:
            x["importance"],

        reverse=True

    )


    feature_data = feature_data[:10]


    return render_template(

        "feature_importance.html",

        features=feature_data

    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

@app.route(
    "/model-performance"
)
def model_performance():

    if not os.path.exists(
        METRICS_PATH
    ):

        return (
            "metrics.json not found. "
            "Please run train_model.py first."
        )


    with open(
        METRICS_PATH,
        "r"
    ) as file:

        metrics = json.load(file)


    return render_template(

        "model_performance.html",

        metrics=metrics

    )


# ============================================================
# VISUALIZATIONS
# ============================================================

@app.route(
    "/visualizations"
)
def visualizations():

    return render_template(
        "visualizations.html"
    )


# ============================================================
# HOW IT WORKS
# ============================================================

@app.route(
    "/how-it-works"
)
def how_it_works():

    best_model_name = "Machine Learning Model"


    if os.path.exists(
        METRICS_PATH
    ):

        with open(
            METRICS_PATH,
            "r"
        ) as file:

            metrics = json.load(file)

            best_model_name = metrics.get(
                "best_model",
                "Machine Learning Model"
            )


    return render_template(

        "how_it_works.html",

        best_model=best_model_name

    )


# ============================================================
# ABOUT
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# CSV UPLOAD
# ============================================================

@app.route(
    "/upload",
    methods=["GET", "POST"]
)
def upload():

    if request.method == "GET":

        return render_template(
            "upload.html"
        )


    file = request.files.get(
        "file"
    )


    if file is None:

        return (
            "No file uploaded."
        )


    if file.filename == "":

        return (
            "Please select a CSV file."
        )


    if not file.filename.lower().endswith(
        ".csv"
    ):

        return (
            "Only CSV files are allowed."
        )


    try:

        df = pd.read_csv(file)

    except Exception as e:

        return (
            f"Error reading CSV: {e}"
        )


    required_columns = [

        "branch",

        "college_tier",

        "cgpa",

        "backlogs",

        "coding_skills",

        "dsa_score",

        "aptitude_score",

        "communication_skills",

        "internships",

        "certifications"

    ]


    missing_columns = [

        column

        for column in required_columns

        if column not in df.columns

    ]


    if missing_columns:

        return (

            "Missing columns: "

            + ", ".join(
                missing_columns
            )

        )


    student_data = df[
        required_columns
    ].copy()


    try:

        predictions = model.predict(
            student_data
        )

        probabilities = (
            model
            .predict_proba(
                student_data
            )[:, 1]
        )

    except Exception as e:

        return (
            f"Prediction error: {e}"
        )


    df[
        "placement_probability"
    ] = (

        probabilities * 100

    ).round(2)


    df[
        "prediction"
    ] = predictions


    df["status"] = df[
        "prediction"
    ].apply(

        lambda x:

        "Likely to be Placed"

        if x == 1

        else

        "Less Likely to be Placed"

    )


    records = df.to_dict(
        orient="records"
    )


    return render_template(

        "batch_results.html",

        records=records

    )


# ============================================================
# DOWNLOAD REPORT
# ============================================================

@app.route(
    "/download-report"
)
def download_report():

    report = f"""
STUDENT PLACEMENT PREDICTION REPORT
====================================

Student Profile
----------------

Branch:
{request.args.get("branch")}

College Tier:
{request.args.get("college_tier")}

CGPA:
{request.args.get("cgpa")}

Backlogs:
{request.args.get("backlogs")}

Coding Skills:
{request.args.get("coding_skills")}/10

DSA Score:
{request.args.get("dsa_score")}/10

Aptitude Score:
{request.args.get("aptitude_score")}/100

Communication Skills:
{request.args.get("communication_skills")}/10

Internships:
{request.args.get("internships")}

Certifications:
{request.args.get("certifications")}


Prediction
----------

Placement Probability:
{request.args.get("probability")}%

Status:
{request.args.get("status")}


====================================
Generated by PlacementAI
"""


    file_path = "placement_report.txt"


    with open(
        file_path,
        "w"
    ) as file:

        file.write(report)


    return send_file(

        file_path,

        as_attachment=True,

        download_name=
            "placement_prediction_report.txt"

    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )