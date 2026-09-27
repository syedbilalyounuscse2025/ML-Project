# 🎓 Student Placement Prediction System

> **An intelligent Machine Learning web application that predicts a student's placement probability, explains the factors behind the prediction, and provides personalized recommendations for improvement.**

---

## 🚀 About the Project

The **Student Placement Prediction System** is a Machine Learning-based web application designed to help students understand their placement readiness.

Instead of simply predicting **Placed** or **Not Placed**, the system analyzes a student's academic performance, technical skills, aptitude, communication skills, internships, and certifications to generate a **placement probability**.

The application also provides:

- 🎯 Placement probability
- 📊 Student profile assessment
- 💪 Key strengths
- 📈 Areas for improvement
- 💼 Recommended job roles
- 🧠 Prediction explanations
- 📉 Model performance statistics
- 📂 Batch CSV prediction
- 🕒 Prediction history

The goal is to transform a basic ML classification model into a practical **placement-readiness assistant for students**.

---

## ✨ Key Features

### 🤖 Placement Prediction

Students enter their academic and skill-related information, and the trained ML model estimates their probability of getting placed.

The system classifies the profile into:

| Probability | Profile Level |
|---|---|
| ≥ 80% | 🌟 Excellent |
| 65% – 79% | 💪 Strong |
| 50% – 64% | 📊 Average |
| < 50% | 📈 Needs Improvement |

---

### 🧠 Explainable Predictions

A prediction should not just say **YES** or **NO**.

The application identifies the most influential features contributing to an individual prediction using the feature importance values from the trained Random Forest model.

This makes the result easier to understand and provides more transparency.

---

### 💡 Personalized Recommendations

Based on the student's profile, the application automatically identifies:

**Strengths**
- Strong CGPA
- Good coding skills
- Strong DSA knowledge
- High aptitude
- Good communication
- Internship experience
- Certifications

**Areas for Improvement**
- Improve coding skills
- Practice DSA
- Increase aptitude score
- Improve communication
- Gain internship experience
- Earn relevant certifications
- Improve academic performance

---

## 💼 Career Role Recommendations

The system recommends possible entry-level career paths based on the student's skills.

Possible recommendations include:

- 💻 Software Developer
- 👨‍💻 Software Engineer
- 📊 Data / Business Analyst
- 🧪 QA / Automation Engineer
- 🛠️ Technical Support Engineer
- 🌐 General Entry-Level IT Roles

The recommendations are generated using the student's coding, DSA, aptitude, communication, CGPA, internship, and certification profile.

---

## 📊 Dataset

The model was developed using a student placement dataset containing approximately:

```text
100,000 Student Records
18 Features
```

### Target Variable

```text
placement_status
```

Dataset distribution:

```text
Placed Students     : 68,475
Not Placed Students : 31,525
```

### Major Input Features

| Feature | Description |
|---|---|
| 🏫 Branch | Student's engineering branch |
| 🎓 College Tier | Tier 1 / Tier 2 / Tier 3 |
| 📚 CGPA | Academic performance |
| ⚠️ Backlogs | Number of academic backlogs |
| 💻 Coding Skills | Coding proficiency |
| 🧩 DSA Score | Data Structures & Algorithms knowledge |
| 🧠 Aptitude Score | Quantitative/reasoning ability |
| 🗣️ Communication Skills | Communication proficiency |
| 💼 Internships | Number of internships |
| 📜 Certifications | Number of certifications |

---

## 🔄 Machine Learning Workflow

```text
        📂 Student Dataset
                │
                ▼
        🧹 Data Preprocessing
                │
                ▼
      🔢 Feature Transformation
                │
                ▼
      🧠 Model Training & Testing
                │
                ▼
       📊 Model Evaluation
                │
                ▼
      🌲 Random Forest Pipeline
                │
                ▼
        🌐 Flask Web App
                │
                ▼
      🎯 Placement Prediction
                │
        ┌───────┴────────┐
        ▼                ▼
 📈 Probability      💡 Suggestions
        │                │
        └───────┬────────┘
                ▼
       💼 Career Recommendations
```

---

## 🧠 Machine Learning Models

Multiple classification algorithms were explored during development.

### 1️⃣ Logistic Regression

```text
Accuracy : 63.23%
F1 Score : 64.50%
ROC-AUC  : 68.56%
```

### 2️⃣ Decision Tree

```text
Accuracy : 59.25%
F1 Score : 60.68%
ROC-AUC  : 63.23%
```

### 3️⃣ Random Forest 🌲

Random Forest is integrated into the final prediction pipeline because it works effectively with the project's mixed feature set and provides **feature importance**, which is also used by the application for prediction explanations.

---

## ⚙️ Data Preprocessing

Categorical and numerical features are handled using a Scikit-learn `ColumnTransformer`.

### Categorical Features

```text
branch
college_tier
```

Processed using:

```python
OneHotEncoder(handle_unknown="ignore")
```

### Numerical Features

```text
cgpa
backlogs
coding_skills
dsa_score
aptitude_score
communication_skills
internships
certifications
```

The preprocessing stage and classifier are combined into a single Scikit-learn **Pipeline**.

---

## 🛠️ Technology Stack

### 👨‍💻 Programming

![Python](https://img.shields.io/badge/Python-Machine%20Learning-blue?logo=python)

### 🧠 Machine Learning

![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)

- Scikit-learn
- Pandas
- NumPy
- Joblib

### 🌐 Web Development

![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?logo=flask)
![HTML](https://img.shields.io/badge/HTML5-Frontend-orange?logo=html5)
![CSS](https://img.shields.io/badge/CSS3-Styling-blue?logo=css3)
![JavaScript](https://img.shields.io/badge/JavaScript-Interactive-yellow?logo=javascript)

### 🗄️ Database

![SQLite](https://img.shields.io/badge/SQLite-Database-blue?logo=sqlite)

### 🧰 Development Tools

![VS Code](https://img.shields.io/badge/VS%20Code-IDE-blue?logo=visualstudiocode)
![GitHub](https://img.shields.io/badge/GitHub-Version%20Control-black?logo=github)

---

## 🌐 Web Application Modules

The Flask application contains multiple pages and features:

```text
🏠 Home
│
├── 🎯 Predict Placement
│
├── 📊 Prediction Result
│
├── 🕒 Prediction History
│
├── 🧠 Feature Importance
│
├── 📈 Model Performance
│
├── 📉 Visualizations
│
├── 📂 Batch CSV Upload
│
├── 📥 Download Report
│
├── ⚙️ How It Works
│
└── ℹ️ About
```

---

## 🗄️ Prediction History

Every prediction can be stored in a local SQLite database.

The stored information includes:

```text
ID
Date
Branch
College Tier
CGPA
Backlogs
Coding Skills
DSA Score
Aptitude Score
Communication Skills
Internships
Certifications
Placement Probability
Prediction
```

This allows previous predictions to be viewed through the **History** page.

---

## 📂 Batch Prediction

Need predictions for more than one student?

The system supports **CSV file upload** for batch processing.

```text
Upload CSV
     ↓
Validate Student Data
     ↓
ML Pipeline
     ↓
Predict Every Student
     ↓
Display Batch Results
```

This makes the system useful for analyzing larger groups of students.

---

## 🔍 Feature Importance

The Random Forest classifier provides feature importance values.

The application extracts the processed feature names using:

```python
preprocessor.get_feature_names_out()
```

and combines them with:

```python
classifier.feature_importances_
```

The most influential features can then be displayed through the **Feature Importance** page.

---

## 🏗️ Project Structure

```text
Student-Placement-Prediction/
│
├── app.py
│
├── predictions.db
├── requirements.txt
├── README.md
│
├── model/
│   ├── best_model.pkl
│   └── metrics.json
│
├── templates/
│   ├── home.html
│   ├── index.html
│   ├── result.html
│   ├── history.html
│   ├── how_it_works.html
│   ├── about.html
│   ├── model_performance.html
│   ├── visualizations.html
│   ├── upload.html
│   ├── batch_results.html
│   └── feature_importance.html
│
└── static/
    ├── css/
    ├── js/
    └── images/
```

> The exact structure may vary depending on the final version of the project.

---

## ⚡ Getting Started

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Enter the Project Directory

```bash
cd Student-Placement-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

**Windows**

```bash
venv\Scripts\activate
```

**macOS/Linux**

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python app.py
```

### 7. Open the Application

Open the local Flask address shown in your terminal, usually:

```text
http://127.0.0.1:5000
```

---

## 🎯 Example Prediction

Example of a strong student profile:

```text
CGPA                 : 8.5
Backlogs             : 0
Coding Skills        : 8/10
DSA Score            : 8/10
Aptitude Score       : 80/100
Communication Skills : 8/10
Internships           : 2
Certifications        : 3
```

The system processes the information through the trained ML pipeline and generates:

```text
🎯 Placement Probability
📊 Placement Prediction
⭐ Profile Level
💪 Strengths
📈 Areas for Improvement
💼 Recommended Career Roles
🧠 Important Contributing Features
```

---

## 🔮 Future Enhancements

Some potential improvements for future versions include:

- 🤖 Experimenting with advanced ML models
- 📊 More interactive analytics dashboards
- 🎓 College-specific placement analysis
- 💼 Company-specific eligibility prediction
- 🧠 More advanced explainable AI techniques such as SHAP
- 📄 Automatic PDF placement-readiness reports
- 🔐 Student login and authentication
- ☁️ Cloud database integration
- 🚀 Cloud deployment
- 📱 Mobile-responsive dashboard improvements
- 🔗 Integration with job and internship platforms

---

## 🎓 Academic Purpose

This project was developed as part of a **Machine Learning academic project** at:

**Chennai Institute of Technology**

### 👨‍💻 Project Team

**Syed Bilal Younus**  
**Mohamed Al Luqman**

### 👩‍🏫 Faculty Guide

**Mrs. Divya**

---

## 💭 Project Vision

> **“Placement prediction should not only tell students where they stand — it should help them understand what they can improve.”**

The project aims to combine **Machine Learning, Explainable AI, Web Development, and Career Guidance** into a single student-friendly platform.

---

## 🤝 Contributions

Suggestions and improvements are welcome!

You can:

- Fork the repository
- Create a new branch
- Implement improvements
- Submit a pull request

---

## ⭐ Support

If you find this project interesting or useful, consider giving the repository a **⭐ Star**.

It helps support the project and motivates us to continue improving it.

---

### 🚀 Built with Python, Machine Learning & Flask

**Developed by Syed Bilal Younus & Mohamed Al Luqman**  
**Chennai Institute of Technology**
