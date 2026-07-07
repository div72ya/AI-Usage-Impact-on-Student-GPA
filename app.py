import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
PREDICT_URL = f"{API_BASE_URL}/predict"
HEALTH_URL = f"{API_BASE_URL}/health"

st.set_page_config(page_title="AI Impact on Student GPA", page_icon="🎓")

st.title("🎓 AI Usage Impact on Student GPA")
st.caption(
    "Fill in a student's profile below to predict whether their GPA is "
    "likely to improve or decline based on their AI usage and study habits."
)

# ---------------------------------------------------------------------------
# Quick API health check — shown as a small badge, doesn't block the form
# ---------------------------------------------------------------------------
try:
    health_resp = requests.get(HEALTH_URL, timeout=3)
    if health_resp.status_code == 200:
        st.success("API is online", icon="✅")
    else:
        st.warning("API responded but health check failed.", icon="⚠️")
except requests.exceptions.RequestException:
    st.error(
        f"Can't reach the API at {API_BASE_URL}. "
        "Make sure it's running (`uvicorn api:app --reload`).",
        icon="🚫",
    )

st.divider()

# ---------------------------------------------------------------------------
# Input form — fields mirror PredictRequest in api.py exactly
# ---------------------------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("Academic Profile")
    col1, col2 = st.columns(2)
    with col1:
        pre_semester_gpa = st.number_input(
            "Pre-Semester GPA", min_value=1.18, max_value=4.0, value=3.0, step=0.01
        )
        major_category = st.selectbox(
            "Major Category", ["STEM", "Business", "Humanities", "Medical", "Arts"]
        )
    with col2:
        year_of_study = st.selectbox(
            "Year of Study",
            ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"],
        )

    st.subheader("AI Usage Behaviour")
    col3, col4 = st.columns(2)
    with col3:
        weekly_genai_hours = st.number_input(
            "Weekly GenAI Hours", min_value=0.0, max_value=40.0, value=5.0, step=0.5
        )
        primary_use_case = st.selectbox(
            "Primary Use Case",
            [
                "Copywriting/Drafting",
                "Summarizing_Reading",
                "Debugging/Troubleshooting",
                "Ideation",
                "Direct_Answer_Generation",
            ],
        )
        prompt_engineering_skill = st.selectbox(
            "Prompt Engineering Skill", ["Beginner", "Intermediate", "Advanced"]
        )
    with col4:
        tool_diversity = st.slider("Tool Diversity (number of AI tools used)", 1, 5, 2)
        paid_subscription = st.checkbox("Has a paid AI subscription?")

    st.subheader("Study Behaviour")
    col5, col6 = st.columns(2)
    with col5:
        traditional_study_hours = st.number_input(
            "Traditional Study Hours (weekly)",
            min_value=1.0,
            max_value=36.0,
            value=10.0,
            step=0.5,
        )
    with col6:
        perceived_ai_dependency = st.slider("Perceived AI Dependency (1-10)", 1, 10, 5)

    st.subheader("Institutional Context")
    institutional_policy = st.selectbox(
        "Institutional Policy",
        ["Allowed_With_Citation", "Strict_Ban", "Actively_Encouraged"],
    )

    st.subheader("Wellbeing")
    col7, col8 = st.columns(2)
    with col7:
        anxiety_level = st.slider("Anxiety Level During Exams (1-10)", 1, 10, 5)
    with col8:
        skill_retention_score = st.number_input(
            "Skill Retention Score (0-100)",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=1.0,
        )

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# On submit: call the API, display result or a friendly error
# ---------------------------------------------------------------------------
if submitted:
    payload = {
        "Pre_Semester_GPA": pre_semester_gpa,
        "Major_Category": major_category,
        "Year_of_Study": year_of_study,
        "Weekly_GenAI_Hours": weekly_genai_hours,
        "Primary_Use_Case": primary_use_case,
        "Prompt_Engineering_Skill": prompt_engineering_skill,
        "Tool_Diversity": tool_diversity,
        "Paid_Subscription": paid_subscription,
        "Traditional_Study_Hours": traditional_study_hours,
        "Perceived_AI_Dependency": perceived_ai_dependency,
        "Institutional_Policy": institutional_policy,
        "Anxiety_Level_During_Exams": anxiety_level,
        "Skill_Retention_Score": skill_retention_score,
    }

    try:
        with st.spinner("Getting prediction..."):
            resp = requests.post(PREDICT_URL, json=payload, timeout=10)

        if resp.status_code == 200:
            result = resp.json()
            prediction = result["prediction"]
            confidence = result["confidence"]

            st.divider()
            if prediction == "Improved":
                st.success(f"### Prediction: GPA likely to **{prediction}** 📈")
            else:
                st.error(f"### Prediction: GPA likely to **{prediction}** 📉")

            st.metric("Confidence", f"{confidence * 100:.1f}%")
            st.progress(confidence)

        elif resp.status_code == 400:
            error_detail = resp.json().get("error", "Invalid input.")
            st.error(f"Input error: {error_detail}", icon="⚠️")

        else:
            st.error(
                f"The API returned an unexpected error (status {resp.status_code}). "
                "Please try again.",
                icon="🚫",
            )

    except requests.exceptions.ConnectionError:
        st.error(
            f"Couldn't connect to the API at {API_BASE_URL}. "
            "Is it running? Start it with `uvicorn api:app --reload`.",
            icon="🚫",
        )
    except requests.exceptions.Timeout:
        st.error("The API took too long to respond. Please try again.", icon="⏱️")
    except requests.exceptions.RequestException as exc:
        st.error(f"Something went wrong while contacting the API: {exc}", icon="🚫")