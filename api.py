import json
import logging
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ai-impact-api")

MODEL_PATH = "model.pkl"
DB_PATH = "predictions.db"

model = None

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            input_features TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL
        )
        """
    )
    conn.commit()
    conn.close()


def save_prediction(input_features: dict, prediction: str, confidence: float):
    """
    Persist a prediction to SQLite. Deliberately never lets a DB failure
    break the API response — logging a prediction is a side effect, not
    the thing the caller is actually waiting for.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO predictions (timestamp, input_features, prediction, confidence) "
            "VALUES (?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                json.dumps(input_features),
                prediction,
                confidence,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to persist prediction to SQLite: %s", exc)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Starting up — loading model...")
    model = joblib.load(MODEL_PATH)
    init_db()
    logger.info("Model loaded successfully. Database ready at %s.", DB_PATH)
    yield
    logger.info("Shutting down.")


app = FastAPI(title="AI Impact on Student Performance API", lifespan=lifespan)

class PredictRequest(BaseModel):
    # Academic profile
    Pre_Semester_GPA: float = Field(..., ge=1.18, le=4.0)
    Major_Category: Literal["STEM", "Business", "Humanities", "Medical", "Arts"]
    Year_of_Study: Literal["Freshman", "Sophomore", "Junior", "Senior", "Graduate"]

    # AI usage behaviour
    Weekly_GenAI_Hours: float = Field(..., ge=0, le=40)
    Primary_Use_Case: Literal[
        "Copywriting/Drafting", "Summarizing_Reading",
        "Debugging/Troubleshooting", "Ideation", "Direct_Answer_Generation",
    ]
    Prompt_Engineering_Skill: Literal["Beginner", "Intermediate", "Advanced"]
    Tool_Diversity: int = Field(..., ge=1, le=5)
    Paid_Subscription: bool

    # Study behaviour
    Traditional_Study_Hours: float = Field(..., ge=1, le=36)
    Perceived_AI_Dependency: int = Field(..., ge=1, le=10)

    # Institutional context
    Institutional_Policy: Literal["Allowed_With_Citation", "Strict_Ban", "Actively_Encouraged"]

    # Mental health & wellbeing
    Anxiety_Level_During_Exams: int = Field(..., ge=1, le=10)
    Skill_Retention_Score: float = Field(..., ge=0, le=100)


class PredictResponse(BaseModel):
    prediction: str
    confidence: float


class HealthResponse(BaseModel):
    status: str


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error("Invalid input on %s: %s", request.url.path, exc.errors())
    messages = "; ".join(
        f"{'.'.join(str(loc) for loc in e['loc'] if loc != 'body')}: {e['msg']}"
        for e in exc.errors()
    )
    return JSONResponse(status_code=400, content={"error": f"Invalid input - {messages}"})


SKILL_MAP = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}


def engineer_features(payload: PredictRequest) -> pd.DataFrame:
    ai_to_study_ratio = payload.Weekly_GenAI_Hours / payload.Traditional_Study_Hours
    prompt_skill_ordinal = SKILL_MAP[payload.Prompt_Engineering_Skill]
    prompt_skill_score = prompt_skill_ordinal * payload.Tool_Diversity

    row = {
        "Pre_Semester_GPA": payload.Pre_Semester_GPA,
        "Weekly_GenAI_Hours": payload.Weekly_GenAI_Hours,
        "Tool_Diversity": payload.Tool_Diversity,
        "Traditional_Study_Hours": payload.Traditional_Study_Hours,
        "Perceived_AI_Dependency": payload.Perceived_AI_Dependency,
        "Anxiety_Level_During_Exams": payload.Anxiety_Level_During_Exams,
        "Skill_Retention_Score": payload.Skill_Retention_Score,
        "AI_to_Study_Ratio": ai_to_study_ratio,
        "Prompt_Skill_Score": prompt_skill_score,
        "Major_Category": payload.Major_Category,
        "Year_of_Study": payload.Year_of_Study,
        "Primary_Use_Case": payload.Primary_Use_Case,
        "Prompt_Engineering_Skill": payload.Prompt_Engineering_Skill,
        "Institutional_Policy": payload.Institutional_Policy,
        "Paid_Subscription": payload.Paid_Subscription,
    }
    return pd.DataFrame([row])


@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest):
    logger.info("Received /predict request: %s", payload.model_dump())

    try:
        X = engineer_features(payload)

        proba = model.predict_proba(X)[0]     
        pred_class = int(proba.argmax())  
        confidence = float(proba[pred_class])
        label = "Improved" if pred_class == 1 else "Declined"

        save_prediction(payload.model_dump(), label, confidence)

        return {"prediction": label, "confidence": round(confidence, 4)}

    except Exception as exc: 
        logger.exception("Error while generating prediction: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal error while generating prediction."},
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)