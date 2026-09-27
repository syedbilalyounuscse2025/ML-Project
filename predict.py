import joblib
import pandas as pd


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(
    "model/best_model.pkl"
)


print(
    "\n===== Student Placement Prediction =====\n"
)


# ==========================================
# INPUT
# ==========================================

branch = input(
    "Branch (CSE/ECE/IT/MECH/CIVIL): "
)


college_tier = input(
    "College Tier (Tier1/Tier2/Tier3): "
)


cgpa = float(
    input("CGPA: ")
)


backlogs = int(
    input("Backlogs: ")
)


coding_skills = float(
    input("Coding Skills (1-10): ")
)


dsa_score = float(
    input("DSA Score (1-10): ")
)


aptitude_score = float(
    input("Aptitude Score (20-100): ")
)


communication_skills = float(
    input("Communication Skills (1-10): ")
)


internships = int(
    input("Internships: ")
)


certifications = int(
    input("Certifications: ")
)


# ==========================================
# DATAFRAME
# ==========================================

student = pd.DataFrame([{

    "branch": branch,

    "college_tier": college_tier,

    "cgpa": cgpa,

    "backlogs": backlogs,

    "coding_skills": coding_skills,

    "dsa_score": dsa_score,

    "aptitude_score": aptitude_score,

    "communication_skills":
        communication_skills,

    "internships": internships,

    "certifications": certifications

}])


# ==========================================
# PREDICTION
# ==========================================

prediction = model.predict(
    student
)[0]


probability = model.predict_proba(
    student
)[0]


placement_probability = (
    probability[1] * 100
)


print(
    "\n=============================="
)


if prediction == 1:

    print(
        "Prediction : STUDENT IS LIKELY TO BE PLACED"
    )

else:

    print(
        "Prediction : STUDENT IS LESS LIKELY TO BE PLACED"
    )


print(
    f"Placement Probability : "
    f"{placement_probability:.2f}%"
)


print(
    "\nSuggestions:"
)


if cgpa < 7:

    print(
        "- Improve your CGPA."
    )


if coding_skills < 7:

    print(
        "- Practice coding regularly."
    )


if dsa_score < 7:

    print(
        "- Practice DSA regularly."
    )


if internships == 0:

    print(
        "- Complete at least one internship."
    )


if certifications < 2:

    print(
        "- Earn more certifications."
    )


print(
    "=============================="
)