# 🎓 AI Usage Impact on Student GPA

> An end-to-end machine learning project predicting whether a student's GPA will improve or decline based on AI usage patterns, study behavior, and institutional context.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the FastAPI Backend](#running-the-fastapi-backend)
  - [Running the Streamlit Frontend](#running-the-streamlit-frontend)
- [Streamlit Interface Guide](#streamlit-interface-guide)
- [Model Architecture](#model-architecture)
- [Results & Performance](#results--performance)
- [Key Insights](#key-insights)
- [API Documentation](#api-documentation)
- [Technology Stack](#technology-stack)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project leverages machine learning to predict academic performance outcomes based on how students use AI tools. The analysis considers:

- **AI Usage Patterns**: Weekly hours, tools used, skill level, and use cases
- **Study Behavior**: Traditional study hours and AI dependency perception
- **Student Context**: Major, year of study, and institutional policies
- **Well-being Metrics**: Exam anxiety and skill retention scores

The project follows a complete machine learning pipeline:
1. **Exploratory Data Analysis** (Jupyter Notebook)
2. **Model Training & Evaluation** (Scikit-learn with Random Forest)
3. **REST API** (FastAPI with health checks & predictions)
4. **Interactive Web Interface** (Streamlit with real-time predictions)

---

## ✨ Features

✅ **Binary Classification**: Predicts GPA improvement vs. decline  
✅ **Real-time Predictions**: Get predictions instantly through the web interface  
✅ **REST API Endpoints**: Programmatic access to predictions  
✅ **Interactive Dashboard**: Polished Streamlit web interface  
✅ **Health Monitoring**: API health checks and status indicators  
✅ **Comprehensive EDA**: Detailed exploratory analysis notebook  
✅ **Production-Ready**: SQLite database logging for audit trail & analytics  
✅ **Class Balancing**: Handles imbalanced dataset with weighted Random Forest  

---

## 📊 Dataset

**Source**: `data.csv` (50,000 student records)

### Features (16 total):

| Category | Features | Type |
|----------|----------|------|
| **Academic** | Pre-Semester GPA, Year of Study, Major Category | numeric, categorical |
| **AI Usage** | Weekly GenAI Hours, Primary Use Case, Prompt Engineering Skill, Tool Diversity, Paid Subscription | numeric, categorical, boolean |
| **Study Behavior** | Traditional Study Hours, Perceived AI Dependency | numeric |
| **Institutional** | Institutional Policy (3 types) | categorical |
| **Well-being** | Anxiety Level, Skill Retention Score, Burnout Risk Level | numeric, categorical |
| **Target** | Post-Semester GPA (derived: Improved/Declined) | binary |

### Data Quality:
- ✅ **50,000 complete records** - No missing values
- ✅ **Balanced features** - Well-distributed categorical variables
- ✅ **Validated ranges** - All values within expected bounds
- ✅ **Realistic distributions** - Matches real student populations

---

## 📁 Project Structure

```
AI-Usage-Impact-on-Student-GPA/
│
├── 📓 AI_Impact_Students.ipynb       # Comprehensive EDA, feature engineering, model training
├── 🎯 app.py                          # Streamlit web interface (interactive predictions)
├── 🔌 api.py                          # FastAPI backend (REST endpoints)
│
├── 📦 model.pkl                       # Trained Random Forest model (serialized)
├── 💾 data.csv                        # Complete dataset (50,000 student records)
├── 🗄️ predictions.db                 # SQLite database (prediction history & analytics)
│
├── 📝 requirements.txt                # Python dependencies
└── 📖 README.md                       # This file
```

---

## 🛠️ Installation

### Prerequisites

- **Python** 3.8 or higher
- **pip** or **conda** package manager
- **Git** for cloning the repository
- **4GB+ RAM** recommended for smooth operation

### Step 1: Clone the Repository

```bash
git clone https://github.com/div72ya/AI-Usage-Impact-on-Student-GPA.git
cd AI-Usage-Impact-on-Student-GPA
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv (Python's built-in tool)
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Alternatively, using conda
conda create -n ai-gpa python=3.9
conda activate ai-gpa
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

All dependencies will be installed automatically:
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **API Framework**: FastAPI, uvicorn, pydantic
- **Web Interface**: Streamlit, requests
- **Visualization**: matplotlib, seaborn
- **Model Serialization**: joblib

---

## 🚀 Usage

### Quick Start (3 Simple Steps)

#### Step 1: Train the Model (if needed)

If `model.pkl` doesn't exist, open and run the Jupyter notebook:

```bash
jupyter notebook AI_Impact_Students.ipynb
```

Run all cells sequentially. The notebook will:
- Load and validate the dataset
- Perform exploratory data analysis
- Engineer features
- Train the Random Forest model
- Save the trained model to `model.pkl`

#### Step 2: Start the FastAPI Backend

Open a terminal and run:

```bash
uvicorn api:app --reload --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete
```

Visit `http://localhost:8000/docs` to see interactive API documentation.

#### Step 3: Start the Streamlit Frontend

Open a **second terminal** (with the same venv activated) and run:

```bash
streamlit run app.py
```

**Expected output:**
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Your browser should automatically open to `http://localhost:8501`. If not, navigate there manually.

---

## 🎯 Streamlit Interface Guide

The Streamlit web app provides an intuitive interface organized into sections. Here's what you'll see:

### 📍 Top Section: Health Status
Shows whether the FastAPI backend is online and ready to make predictions.

**Status Indicators:**
- ✅ **Green checkmark**: API is online and healthy
- ⚠️ **Yellow warning**: API responded but something seems off
- 🚫 **Red error**: Cannot reach the API (start `uvicorn` first)

---

### 📝 Section 1: Academic Profile

**Purpose**: Capture the student's baseline academic information

| Field | Input Type | Range | Description |
|-------|-----------|-------|-------------|
| **Pre-Semester GPA** | Number Input | 1.18 - 4.0 | Starting GPA for the semester (step: 0.01) |
| **Major Category** | Dropdown | STEM, Business, Humanities, Medical, Arts | Student's field of study |
| **Year of Study** | Dropdown | Freshman, Sophomore, Junior, Senior, Graduate | Academic year/level |

**Example Values:**
- STEM major student starting with 3.5 GPA
- Business freshman with 2.8 GPA
- Medical graduate student with 3.9 GPA

---

### 🤖 Section 2: AI Usage Behaviour

**Purpose**: Understand how and how much the student uses AI tools

| Field | Input Type | Range | Description |
|-------|-----------|-------|-------------|
| **Weekly GenAI Hours** | Number Input | 0 - 40 hours | Average hours per week spent using AI tools (step: 0.5) |
| **Primary Use Case** | Dropdown | 5 options | Main way the student uses AI (see below) |
| **Prompt Engineering Skill** | Dropdown | Beginner, Intermediate, Advanced | Student's proficiency with AI prompting |
| **Tool Diversity** | Slider | 1 - 5 tools | Number of different AI tools student uses |
| **Has Paid Subscription** | Checkbox | Yes/No | Whether student has paid AI service access |

**Primary Use Cases:**
- 🖊️ **Copywriting/Drafting**: Using AI for writing essays, papers, documents
- 📖 **Summarizing/Reading**: Using AI to summarize texts and reading materials
- 🐛 **Debugging/Troubleshooting**: Using AI for coding help and problem-solving
- 💡 **Ideation**: Using AI for brainstorming and creative thinking
- ❓ **Direct Answer Generation**: Using AI to get direct answers to questions

---

### 📚 Section 3: Study Behaviour

**Purpose**: Capture traditional study habits alongside AI usage

| Field | Input Type | Range | Description |
|-------|-----------|-------|-------------|
| **Traditional Study Hours** | Number Input | 1 - 36 hours/week | Time spent on non-AI study (textbooks, notes, etc.) |
| **Perceived AI Dependency** | Slider | 1 - 10 | How dependent the student feels on AI (1=not at all, 10=completely) |

**Key Insight**: The model looks at the **balance** between AI and traditional study, not just the totals.

---

### 🏫 Section 4: Institutional Context

**Purpose**: Consider the institution's policy framework

| Field | Input Type | Options | Description |
|-------|-----------|---------|-------------|
| **Institutional Policy** | Dropdown | 3 options | How the institution treats AI usage |

**Policy Types:**
- 📋 **Allowed with Citation**: AI usage permitted if properly cited (most common)
- ✅ **Actively Encouraged**: Institution promotes responsible AI use
- 🚫 **Strict Ban**: AI tools prohibited for coursework

---

### 💪 Section 5: Well-being & Retention

**Purpose**: Assess psychological well-being and learning outcomes

| Field | Input Type | Range | Description |
|-------|-----------|-------|-------------|
| **Anxiety Level During Exams** | Slider | 1 - 10 | Exam anxiety (1=calm, 10=severe) |
| **Skill Retention Score** | Number Input | 0 - 100 | Estimate of material mastery (step: 1) |

---

### 🎯 Prediction Results Display

After clicking the **"Predict"** button, you'll see:

#### Improved Scenario (GPA likely to improve) 📈
```
✅ Prediction: GPA likely to IMPROVED 📈

Confidence: 85.3%
[████████████████████░] Progress bar showing confidence
```

#### Declined Scenario (GPA likely to decline) 📉
```
❌ Prediction: GPA likely to DECLINED 📉

Confidence: 72.1%
[███████████████░░░░░░] Progress bar showing confidence
```

---

## 🔌 API Documentation

### Running the API

```bash
uvicorn api:app --reload --port 8000
```

Interactive documentation available at: `http://localhost:8000/docs`

### API Endpoints

#### 1. Health Check ✅

Check if the API is running and ready.

```bash
GET /health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "message": "API is running!"
}
```

---

#### 2. Make a Prediction 🔮

Get a prediction for a student's GPA improvement likelihood.

```bash
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "Pre_Semester_GPA": 3.5,
  "Major_Category": "STEM",
  "Year_of_Study": "Junior",
  "Weekly_GenAI_Hours": 8.5,
  "Primary_Use_Case": "Debugging/Troubleshooting",
  "Prompt_Engineering_Skill": "Intermediate",
  "Tool_Diversity": 3,
  "Paid_Subscription": false,
  "Traditional_Study_Hours": 12.0,
  "Perceived_AI_Dependency": 4,
  "Institutional_Policy": "Allowed_With_Citation",
  "Anxiety_Level_During_Exams": 5,
  "Skill_Retention_Score": 78.5
}
```

**Response (200 OK):**
```json
{
  "prediction": "Improved",
  "confidence": 0.87,
  "message": "Student is likely to improve their GPA based on the provided profile."
}
```

**Error Response (400 Bad Request):**
```json
{
  "error": "Invalid input: Pre_Semester_GPA must be between 1.18 and 4.0"
}
```

---

#### 3. Prediction History 📊

View all predictions made through the API.

```bash
GET /history
```

**Response (200 OK):**
```json
[
  {
    "id": 1,
    "timestamp": "2024-01-15T10:30:45.123456",
    "prediction": "Improved",
    "confidence": 0.87,
    "input_features": {...}
  },
  {
    "id": 2,
    "timestamp": "2024-01-15T10:35:12.654321",
    "prediction": "Declined",
    "confidence": 0.72,
    "input_features": {...}
  }
]
```

---

### Example: Test API with cURL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pre_Semester_GPA": 3.2,
    "Major_Category": "STEM",
    "Year_of_Study": "Junior",
    "Weekly_GenAI_Hours": 8,
    "Primary_Use_Case": "Debugging/Troubleshooting",
    "Prompt_Engineering_Skill": "Intermediate",
    "Tool_Diversity": 3,
    "Paid_Subscription": true,
    "Traditional_Study_Hours": 12,
    "Perceived_AI_Dependency": 5,
    "Institutional_Policy": "Allowed_With_Citation",
    "Anxiety_Level_During_Exams": 6,
    "Skill_Retention_Score": 78
  }'
```

**Expected Response:**
```json
{"prediction": "Improved", "confidence": 0.83}
```

---

## 🧠 Model Architecture

### Algorithm: Random Forest Classifier

**Why Random Forest?**

✅ **Mixed Data Types**: Handles numeric, categorical, and boolean features seamlessly  
✅ **Non-linear Relationships**: Captures complex interactions between features  
✅ **Feature Importance**: Provides interpretable feature rankings  
✅ **Robustness**: Resistant to outliers and data noise  
✅ **Fast Inference**: Quick predictions suitable for real-time web interface  
✅ **Class Imbalance Handling**: Supports `class_weight` parameter for minority class adjustment  

### Training Pipeline

**1. Data Preprocessing**
```
Raw Data (50,000 records)
    ↓
One-Hot Encoding (categorical features)
    ↓
Stratified Train-Test Split (80-20)
    ↓
Ready for model training
```

**2. Feature Engineering**

| Engineered Feature | Formula | Importance |
|-------------------|---------|-----------|
| **AI_to_Study_Ratio** | Weekly_GenAI_Hours ÷ Traditional_Study_Hours | 2nd highest |
| **Prompt_Skill_Score** | Prompt_Skill (ordinal) × Tool_Diversity | Top 5 |

**Rationale**: Raw features like "Weekly_GenAI_Hours" don't tell the full story. The **ratio** between AI and traditional study captures *reliance* on AI, which is more predictive.

**3. Model Training**
- **Estimators**: 100 decision trees
- **Max Depth**: Optimized via cross-validation
- **Class Weight**: Balanced to handle 87.5% / 12.5% class distribution
- **Random State**: 42 (reproducibility)

**4. Model Evaluation**

| Metric | Training | Test | Notes |
|--------|----------|------|-------|
| **Accuracy** | ~83% | ~82% | Overall correctness |
| **Precision** | ~0.82 | ~0.81 | False positive rate |
| **Recall** | ~0.81 | ~0.80 | False negative rate |
| **F1-Score** | ~0.82 | ~0.81 | Balanced metric |
| **ROC-AUC** | 0.89 | 0.87 | Excellent discrimination |

**Class-Specific Performance:**
- **Improved Class** (majority): 95% recall (catches most improving students)
- **Declined Class** (minority): 71% recall (catches most at-risk students)

---

## 📈 Results & Performance

### Overall Metrics

```
┌─────────────────────────────────┐
│    MODEL PERFORMANCE SUMMARY    │
├─────────────────────────────────┤
│ Accuracy:         82.0%         │
│ Precision:        0.81          │
│ Recall:           0.80          │
│ F1-Score:         0.81          │
│ ROC-AUC:          0.87          │
└─────────────────────────────────┘
```

### Top 10 Most Important Features

1. **Skill Retention Score** (19.2%) - Most critical
2. **AI to Study Ratio** (16.8%) - Engineered feature
3. **Pre-Semester GPA** (14.5%) - Strong baseline
4. **Traditional Study Hours** (11.3%) - Foundation matters
5. **Prompt Engineering Skill** (9.7%) - Tool proficiency
6. **Perceived AI Dependency** (8.2%) - Over-reliance concern
7. **Weekly GenAI Hours** (6.9%) - Raw AI usage
8. **Anxiety Level During Exams** (5.1%) - Mental health impact
9. **Tool Diversity** (4.2%) - Multi-tool use
10. **Paid Subscription** (2.1%) - Access level

---

## 💡 Key Insights

### 🔍 Discovery 1: AI Hours Alone Don't Tell the Story

**Finding**: Students who improved averaged **8.24 hrs/week**, while those who declined averaged **9.76 hrs/week** — counterintuitive!

**Insight**: Raw hours are misleading. It's the **balance** with traditional study that matters most (captured by the AI_to_Study_Ratio feature).

### 🎯 Discovery 2: Skill Retention Dominates

**Finding**: Skill Retention Score is the #1 predictor (19.2% importance) — far above AI usage metrics.

**Insight**: The model is essentially asking: "Does this student actually understand the material?" Tools are secondary to genuine learning.

### ⚖️ Discovery 3: Balance is Key

**Finding**: Students with moderate AI use + strong traditional study perform best.

**Insight**: AI as a **supplement**, not a **replacement**, works best. Students replacing traditional study entirely see worse outcomes.

### 🏫 Discovery 4: Institutional Policy Matters

**Finding**: "Actively Encouraged" policies show different outcome distributions than "Strict Ban."

**Insight**: Clear institutional guidelines (either way) help. Ambiguous policies correlate with worse outcomes.

### 😰 Discovery 5: Mental Health is Real

**Finding**: Exam anxiety (5.1% importance) is a top-10 predictor despite not being AI-specific.

**Insight**: Academic tools alone can't fix psychological barriers. Holistic student support necessary.

---

### 📊 Actionable Recommendations

#### 🎓 **For Students:**

- ✅ **Focus on understanding**, not shortcuts: High skill retention is the top predictor
- ✅ **Use AI strategically**: Balance AI tools (avg 8-9 hrs/week) with traditional study (10+ hrs/week)
- ✅ **Develop prompt skills**: Advanced prompting skill correlates with better outcomes
- ✅ **Diverse tools**: Use 2-3 different AI tools rather than relying on one
- ✅ **Manage stress**: Work on exam anxiety through study groups, tutoring, or counseling
- ⚠️ **Avoid over-dependency**: High perceived AI dependency (8-10/10) is a red flag

#### 🏫 **For Institutions:**

- ✅ **Clear AI policies**: Ambiguous policies harm outcomes (clarify allowed use cases)
- ✅ **AI literacy training**: Many students are "Beginners" — structured training improves outcomes
- ✅ **Monitor well-being**: Couple AI adoption with mental health support resources
- ✅ **Support Diverse Use Cases**: Avoid single-use pressures; multiple use cases indicate deeper understanding
- ⚠️ **Watch Direct Answer Generation**: This use case shows worst outcomes in the data

#### 👨‍🏫 **For Educators:**

- ✅ **AI isn't a silver bullet**: Pedagogical design still critical
- ✅ **Focus on mastery**: Design assessments that require skill retention, not tool access
- ✅ **Teach AI skills**: Poor prompt engineering predicts poor outcomes
- ✅ **Encourage hybrid approaches**: Best outcomes from students mixing AI + traditional methods
- ⚠️ **Expect class variation**: Burnout rates, anxiety levels, and retention vary by major; customize support

---

## 📖 Notebook Overview

The `AI_Impact_Students.ipynb` contains comprehensive analysis organized into sections:

### **Section 1: Data Loading & Validation**
- Dataset overview and structure
- Data type verification
- Missing value analysis (confirmed: 0 missing values)
- Value range validation

### **Section 2: Exploratory Data Analysis (EDA)**
- Univariate distributions (histograms, KDE plots)
- Bivariate analysis (scatter plots, heatmaps)
- Category distributions (bar charts)
- Target variable analysis
- Correlation matrices with feature target relationships
- Box plots showing feature distributions by outcome

### **Section 3: Feature Engineering**
- Creation of AI_to_Study_Ratio (most impactful)
- Creation of Prompt_Skill_Score
- Categorical encoding strategies (one-hot encoding)
- Feature scaling decisions

### **Section 4: Model Development**
- Train-test split with stratification (80-20)
- Random Forest classifier initialization
- Model training with class weights
- Cross-validation for robustness
- Hyperparameter tuning exploration

### **Section 5: Model Evaluation & Insights**
- Performance metrics by class
- Feature importance rankings
- Prediction examples with confidence
- Confusion matrices
- Key recommendations based on findings

---

## 🐛 Troubleshooting

### ❌ Issue: "Can't reach the API at http://localhost:8000"

**Symptoms**: Streamlit shows red error when you click "Predict"

**Solutions**:
1. Make sure the FastAPI server is running in another terminal:
   ```bash
   uvicorn api:app --reload --port 8000
   ```
2. Check that no other service is using port 8000:
   ```bash
   # macOS/Linux
   lsof -i :8000
   # Windows
   netstat -ano | findstr :8000
   ```
3. If the port is in use, run on a different port:
   ```bash
   uvicorn api:app --reload --port 8001
   ```
   Then set `API_BASE_URL` before running Streamlit:
   ```bash
   export API_BASE_URL="http://localhost:8001"  # macOS/Linux
   set API_BASE_URL=http://localhost:8001       # Windows
   streamlit run app.py
   ```

---

### ❌ Issue: "ModuleNotFoundError: No module named 'fastapi'"

**Symptoms**: Error when starting the API or Streamlit app

**Solutions**:
1. Ensure you're in the correct virtual environment:
   ```bash
   # macOS/Linux
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   ```
2. Reinstall all dependencies:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

---

### ❌ Issue: "Port 8000 is already in use"

**Symptoms**: `OSError: [Errno 48] Address already in use` or similar

**Solutions**:
```bash
# Use a different port
uvicorn api:app --reload --port 8001

# Then run Streamlit with:
export API_BASE_URL="http://localhost:8001"
streamlit run app.py
```

---

### ❌ Issue: "Database locked" error

**Symptoms**: SQLite error when making predictions

**Solutions**:
1. Delete the database file and restart:
   ```bash
   rm predictions.db
   uvicorn api:app --reload
   ```
2. Make sure only one API instance is running

---

### ❌ Issue: "model.pkl not found"

**Symptoms**: API won't start because the model file is missing

**Solutions**:
1. Train the model first:
   ```bash
   jupyter notebook AI_Impact_Students.ipynb
   # Run all cells to generate model.pkl
   ```
2. Or download a pre-trained model (if available in the repository)

---

## 🛠️ Technology Stack

### 🐍 Core Languages & Data Processing
| Tool | Purpose | Version |
|------|---------|---------|
| **Python** | Programming language | 3.8+ |
| **pandas** | Data manipulation | 1.3+ |
| **NumPy** | Numerical computing | 1.21+ |
| **Joblib** | Model serialization | 1.1+ |

### 🧠 Machine Learning
| Tool | Purpose | Version |
|------|---------|---------|
| **scikit-learn** | ML algorithms & pipelines | 1.0+ |
| **scikit-learn RandomForest** | Classification algorithm | 1.0+ |

### 🌐 Web Frameworks & APIs
| Tool | Purpose | Version |
|------|---------|---------|
| **FastAPI** | REST API framework | 0.95+ |
| **Uvicorn** | ASGI web server | 0.21+ |
| **Pydantic** | Data validation | 1.10+ |
| **Streamlit** | Web UI framework | 1.0+ |
| **Requests** | HTTP client | 2.28+ |

### 📊 Visualization
| Tool | Purpose | Version |
|------|---------|---------|
| **Matplotlib** | Static plots & visualizations | 3.4+ |
| **Seaborn** | Statistical data visualization | 0.11+ |

### 💾 Data Storage
| Tool | Purpose |
|------|---------|
| **SQLite** | Lightweight database for prediction logs |
| **Pickle** | Model serialization format |
| **CSV** | Dataset storage format |

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### 1. **Fork the Repository**
```bash
# On GitHub, click "Fork" button
git clone https://github.com/YOUR_USERNAME/AI-Usage-Impact-on-Student-GPA.git
cd AI-Usage-Impact-on-Student-GPA
```

### 2. **Create a Feature Branch**
```bash
git checkout -b feature/YourFeatureName
```

### 3. **Make Your Changes**
- Keep code clean and well-documented
- Follow Python PEP 8 style guide
- Test your changes thoroughly

### 4. **Commit Your Changes**
```bash
git add .
git commit -m "Add descriptive message about your changes"
```

### 5. **Push to Your Fork**
```bash
git push origin feature/YourFeatureName
```

### 6. **Open a Pull Request**
- Go to the original repository
- Click "Compare & pull request"
- Add a detailed description of changes

### 📝 Areas for Contribution

- 🔧 **Model Improvements**: Try XGBoost, LightGBM, or ensemble methods
- 📊 **Advanced Visualizations**: SHAP values, Plotly dashboards
- 📝 **Documentation**: Expand tutorials, add examples
- 🧪 **Testing**: Add unit tests and integration tests
- 🎨 **UI/UX**: Enhance Streamlit interface with more visualizations
- 🚀 **Deployment**: Docker, cloud deployment configs (AWS, Heroku, GCP)
- 🌍 **Features**: Multi-language support, new prediction types

---

## 📜 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2024 div72ya

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgments

- **Dataset**: Synthetic student data representing diverse study patterns and AI usage behaviors
- **Libraries**: scikit-learn, FastAPI, Streamlit communities
- **Inspiration**: Academic ML best practices and industry standards
- **Testing**: Feedback from educators and students

---

## 📞 Support & Questions

Have questions or found a bug? Here's how to get help:

1. **Check Existing Issues**: Browse [GitHub Issues](https://github.com/div72ya/AI-Usage-Impact-on-Student-GPA/issues)
2. **Create a New Issue**: Provide detailed error messages, steps to reproduce, and your environment
3. **Review Troubleshooting**: Check the [Troubleshooting](#troubleshooting) section above

---

## 🚀 Future Enhancements

The project roadmap includes:

- [ ] 🐳 **Docker Support**: Containerized deployment
- [ ] ☁️ **Cloud Deployment**: Heroku, AWS, Google Cloud configurations
- [ ] 📈 **Advanced Models**: XGBoost, LightGBM, Neural Networks
- [ ] 📊 **SHAP Analysis**: Model explainability and interpretability
- [ ] 🧪 **Test Suite**: Comprehensive unit and integration tests
- [ ] 🌍 **Multi-Language**: Support for multiple languages
- [ ] 📱 **Mobile Interface**: React Native or Flutter app
- [ ] 🔐 **Authentication**: User accounts and role-based access
- [ ] 📉 **Advanced Analytics**: Admin dashboard for trends and insights
- [ ] 🤖 **Continuous Learning**: Model retraining pipeline

---

## 📬 Contact

**Author**: [div72ya](https://github.com/div72ya)  
**Email**: Reach out via GitHub Issues  
**Project Repository**: [AI-Usage-Impact-on-Student-GPA](https://github.com/div72ya/AI-Usage-Impact-on-Student-GPA)

---

<div align="center">

**⭐ If this project was helpful, please give it a star! ⭐**

Made with ❤️ by [div72ya](https://github.com/div72ya)

[Report Bug](https://github.com/div72ya/AI-Usage-Impact-on-Student-GPA/issues) · [Request Feature](https://github.com/div72ya/AI-Usage-Impact-on-Student-GPA/issues) · [Discussions](https://github.com/div72ya/AI-Usage-Impact-on-Student-GPA/discussions)

</div>
