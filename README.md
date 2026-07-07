# AI Usage Impact on Student Academic Performance

Predicts whether a student's GPA is likely to **improve** or **decline** after
a semester, based on their AI usage behaviour, study habits, and
institutional context.

## Project Structure

```
.
├── api.py                     # FastAPI app (health & predict)
├── app.py                     # Streamlit UI
├── AI_Impact_Students.ipynb   # Training notebook (EDA, feature engineering, model)
├── model.pkl                  # Saved trained model (created by the notebook)
├── predictions.db             # SQLite store of saved predictions (created at runtime)
├── data.csv                   # Dataset
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Environment Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Train the model (if `model.pkl` isn't already present)

Open and run `AI_Impact_Students.ipynb` top to bottom. The last cell saves
the trained pipeline to `model.pkl`.

## 2. Start the API

```bash
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at
`http://localhost:8000/docs`.

## 3. Start the Streamlit app

In a **second terminal** (with the same venv activated):

```bash
streamlit run app.py
```

This opens a browser UI at `http://localhost:8501` that talks to the API
at `http://localhost:8000` by default. To point it at a different API URL,
set the `API_BASE_URL` environment variable before launching.

## Example: test the API directly with curl

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

Expected response:

```json
{"prediction": "Improved", "confidence": 0.83}
```

## Model & Feature Engineering Notes

- **Model:** RandomForestClassifier (`class_weight='balanced'` to handle the
  87.5% / 12.5% class imbalance in the target).
- **Engineered features:**
  - `AI_to_Study_Ratio` = `Weekly_GenAI_Hours / Traditional_Study_Hours` —
    captures reliance on AI relative to traditional studying. Turned out to
    be the 2nd most important feature overall, outperforming the raw
    `Weekly_GenAI_Hours` column it was derived from.
  - `Prompt_Skill_Score` = ordinal-encoded `Prompt_Engineering_Skill` ×
    `Tool_Diversity` — captures both skill level and breadth of AI tool use.
- **Evaluation:** precision/recall/F1 per class + ROC-AUC (0.81), since
  accuracy alone is misleading on an imbalanced target. On the minority
  ("Declined") class: 71% recall, 29% precision — a deliberate trade-off
  favouring catching at-risk students over avoiding false alarms.

Full EDA and reasoning are in `AI_Impact_Students.ipynb`.
