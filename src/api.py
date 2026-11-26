"""
api.py
======
שרת ה-API (מבוסס FastAPI) למודל חיזוי גניבות רכב (XGBoost) ולסוכן הביטוח האוטונומי (Bestie Agent).

הרצה מקומית:
    uvicorn src.api:app --reload
תיעוד אינטראקטיבי (Swagger):
    http://127.0.0.1:8000/docs

נקודות קצה עיקריות (Endpoints):
-------------------------------
GET  /health                      - בדיקת בריאות השרת, גרסת המודל הפעילה ומדדי הערכה.
GET  /predict?model=&city=&model_year= - חישוב תחזית גניבות פרטנית לשילוב (דגם, עיר, שנתון).
POST /agent/auto-compare          - הפעלת סוכן ההשוואה האוטונומי ברקע להפקת כרטיסי Bestie וחלופות.
GET  /agent/auto-compare          - גרסת GET נוחה להשוואה אוטונומית ישירות מהדפדפן.
POST /agent/chat                  - צ'אט סוכן מובנה עם 4 שכבות הגנה (Guardrails) ומניעת הזיות.
"""
import json
import os
from functools import lru_cache
from typing import Any

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agent import run_auto_compare_agent, run_chat_agent
from src.agent_schemas import GroundedAgentResponse

# ==============================================================================
# הגדרת נתיבי מערכת לקבצי נתונים, מודלים וקונפיגורציה
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_local_env_once() -> None:
    """
    טוען משתני סביבה מקובץ `.env` בהפעלת השרת החי (ומדלג בזמן pytest כדי שבדיקות
    היחידה ישלטו באופן מבוקר ב-monkeypatch על os.environ).
    """
    if "PYTEST_CURRENT_TEST" in os.environ:
        return
    env_path = os.path.join(BASE_DIR, ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v


_load_local_env_once()
WEB_DIR = os.path.join(BASE_DIR, "web")
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_theft_regressor.json")
CARD_PATH = os.path.join(BASE_DIR, "models", "model_card.json")
REGISTRY_PATH = os.path.join(BASE_DIR, "models", "model_registry.json")
DATA_PATH = os.path.join(BASE_DIR, "web", "assets", "data.json")
V9_SCORES_PATH = os.path.join(BASE_DIR, "web", "assets", "v9_scores.json")
FINAL_SCORES_PATH = os.path.join(BASE_DIR, "web", "assets", "final_scores.json")

# אתחול אפליקציית FastAPI
app = FastAPI(
    title="CarTheftRisk API & Autonomous Bestie Agent",
    version="final",
    description="Statistical estimate for research/education. Not insurance advice.",
)

# הגדרת מקורות מורשים (CORS) לתקשורת מול דפדפנים וממשקי ווב
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CAR_THEFT_ALLOWED_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:5500,http://localhost:5500",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# סכמות בקשה (Request Models)
# ==============================================================================

class AutoCompareRequest(BaseModel):
    """מודל בקשה להשוואה אוטונומית של דגם ועיר."""
    model: str = Field(..., description="שם דגם הרכב")
    city: str = Field(..., description="שם הרשות המקומית / העיר")


class AgentChatRequest(BaseModel):
    """מודל בקשה לצ'אט מול סוכן ה-AI."""
    message: str = Field(..., min_length=1, max_length=500, description="שאלת המשתמש בעברית (עד 500 תווים)")
    model: str = Field("טויוטה קורולה", max_length=120, description="דגם הרכב הנבחר כרגע בממשק")
    city: str = Field("תל אביב יפו", max_length=120, description="העיר הנבחרת כרגע בממשק")
    history: list[dict[str, Any]] | None = Field(default=None, max_length=20, description="היסטוריית שיחה אופציונלית להקשר המשכי")


# ==============================================================================
# פונקציות טעינה עם מטמון (Caching) למניעת קריאות דיסק חוזרות
# ==============================================================================

@lru_cache(maxsize=1)
def _load():
    """
    טוען לזיכרון פעם אחת בלבד את:
    1. כרטיס המודל (model_card.json) המכיל את הגדרת הפיצ'רים ומדדי הביצועים.
    2. מאגר הנתונים המלא של הווב (data.json).
    3. מודל ה-XGBoost הרגרסורי המאומן (xgb_theft_regressor.json).
    """
    with open(CARD_PATH, encoding="utf-8") as f:
        card = json.load(f)
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model, card, data


@lru_cache(maxsize=1)
def _load_registry():
    """טוען את רישום גרסאות המודל (Model Registry) לבדיקת סטטוס הגרסה הפעילה."""
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_scores():
    """טוען את קובץ הציונים המחושבים מראש (final_scores.json או v9_scores.json)."""
    path = FINAL_SCORES_PATH if os.path.exists(FINAL_SCORES_PATH) else V9_SCORES_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)["scores"]


# כינוי תואם לאחור לטעינת ציונים
_load_v9_scores = _load_scores


# ==============================================================================
# הגדרת נקודות הקצה (Endpoints)
# ==============================================================================

@app.get("/health")
def health():
    """
    נקודת קצה לבדיקת תקינות המערכת וזמינות המודל.
    מחזירה מטא-דאטה מקיף על הגרסה הרצה, רשימת הפיצ'רים, ביצועי בנצ'מרק וכמות הדגמים/הערים הנתמכים.
    """
    _, card, data = _load()
    registry = _load_registry()
    poisson_bm = card.get("evaluation", {}).get("xgboost_poisson", card.get("benchmark", {}))
    return {
        "status": "ok",
        "pipeline_version": registry["active_runtime_model"],
        "features": card["features"],
        "benchmark": poisson_bm,
        "models": len(data["models"]),
        "cities": len(data["locations"]),
        "agent_enabled": True,
        "model_version": registry["active_runtime_model"],
        "data_status": data["metadata"].get("dataset_role"),
        "model_status": registry["versions"][registry["active_runtime_model"]]["status"],
        "disclaimer": card.get("disclaimer", "Research dashboard only. Predictions are historical statistical estimates; insurance prices are simulations."),
    }


@app.get("/predict")
def predict(
    model: str = Query(..., description="שם הדגם כפי שמופיע במאגר הווב"),
    city: str = Query(..., description="שם העיר"),
    model_year: int = Query(2020, ge=1950, le=2026, description="שנת ייצור הרכב (1950-2026)"),
):
    """
    מבצע חיזוי כמות גניבות צפויה לשנת 2025 עבור שילוב פרטני של (דגם, עיר, שנתון).
    משתמש במודל XGBoost רגרסורי עם פונקציית הפסד Poisson, תוך הזנת פיצ'רים מעודכנים:
    שנתון, קצב גידול הדגם, ציון סיכון דגם, שיעור פענוח משטרתי, שיעור תיקים פתוחים וגניבות בפיגור (lag_thefts).
    """
    mdl, card, data = _load()
    scores = _load_v9_scores()
    m = data["models"].get(model)
    c = data["locations"].get(city)
    if m is None:
        raise HTTPException(404, f"Unknown model: {model}")
    if c is None:
        raise HTTPException(404, f"Unknown city: {city}")

    # בדיקה האם קיים אומדן מחושב מראש לשילוב דגם-עיר זה
    score = scores.get(f"{m['model_clean']}|{c['city_unified']}")
    if score is None:
        # טיפול במצב של 'התחלה קרה' (Cold-Start) כאשר אין תצפית היסטורית ישירה
        score = {
            "estimated_thefts_2025": None,
            "observed_thefts_2024": 0.0,
            "final_risk_score": None,
            "v9_risk_score": None,
            "coverage": "cold-start: no observed 2024 model/city count",
        }

    # בניית וקטור הפיצ'רים עבור המודל
    values = {
        "model_year": model_year,
        "growth_factor": float(m.get("growth_rate", 0.0)),
        "model_risk_score": float(m.get("risk_score", 50.0)),
        "detection_rate": float(c.get("detection_rate", 0.12)),
        "unsolved_unknown_rate": float(c.get("unsolved_unknown_rate", 0.85)),
        "lag_thefts": float(score.get("observed_thefts_2024", 0.0)),
        "city_detection_rate": float(c.get("detection_rate", 0.12)),
        "city_unsolved_rate": float(c.get("unsolved_unknown_rate", 0.85)),
        "city_cases_total": float(c.get("total_cases", 0)),
    }
    x = np.array([[values[f] for f in card["features"]]], dtype=float)
    # חיזוי המודל ומניעת תוצאות שליליות
    pred = float(max(0.0, mdl.predict(x)[0]))

    return {
        "model": model,
        "city": city,
        "model_year": model_year,
        "final_estimated_thefts_2025": round(pred, 2),
        "final_score": score,
        "v9_estimated_thefts_2025": round(pred, 2),
        "v9_score": score,
        "coverage_status": score.get("coverage", "observed 2024 model/city count available"),
        "model_version": "final",
        "data_cutoff": "2024 observed count; evaluated against 2025",
        "features_used": values,
        "benchmark": card["evaluation"]["xgboost_poisson"],
        "disclaimer": card["disclaimer"],
    }


@app.post("/agent/auto-compare")
def agent_auto_compare(req: AutoCompareRequest):
    """
    מפעיל את סוכן ההשוואה האוטונומי של Bestie בשיטת POST.
    שרשרת 3 כלים המפיקה כרטיסי מחיר וחלופות רכב זולות ובטוחות יותר.
    """
    return run_auto_compare_agent(req.model, req.city)


@app.get("/agent/auto-compare")
def agent_auto_compare_get(
    model: str = Query(..., description="Vehicle model name"),
    city: str = Query(..., description="City name"),
):
    """גרסת GET לסוכן ההשוואה האוטונומי של Bestie."""
    return run_auto_compare_agent(model, city)


@app.post("/agent/chat", response_model=GroundedAgentResponse)
def agent_chat(req: AgentChatRequest):
    """
    נקודת הקצה של צ'אט סוכן ה-AI.
    מפעילה ארכיטקטורת 4 שכבות הגנה (Input Guardrails -> Strict Grounding -> Deterministic Tools -> Numeric Verifier).
    מוודאת תשובות אמינות ללא הזיות ומחזירה מבנה נתונים מפורט ומאומת.
    """
    return run_chat_agent(req.message, req.model, req.city, req.history)


# הגשת קבצי הדשבורד של הווב (HTML/JS/CSS) במידה והתיקייה קיימת
if os.path.isdir(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
