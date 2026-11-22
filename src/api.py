"""
api.py
======
Research inference & AI Agent API for the CarTheftRisk v9 XGBoost model.

Run:   uvicorn src.api:app --reload      (from the project root)
Docs:  http://127.0.0.1:8000/docs

Endpoints
---------
GET  /health                      service + model status
GET  /predict?model=&city=&model_year=
     -> model prediction for ONE (model, model_year, city) leaf row, plus the
        pre-computed out-of-fold combo estimate used by the web app.
POST /agent/auto-compare          background autonomous Bestie insurance comparison agent
POST /agent/chat                  contextual inline agent chat grounded in tools
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(BASE_DIR, "web")
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_theft_regressor.json")
CARD_PATH = os.path.join(BASE_DIR, "models", "model_card.json")
REGISTRY_PATH = os.path.join(BASE_DIR, "models", "model_registry.json")
DATA_PATH = os.path.join(BASE_DIR, "web", "assets", "data.json")
V9_SCORES_PATH = os.path.join(BASE_DIR, "web", "assets", "v9_scores.json")
FINAL_SCORES_PATH = os.path.join(BASE_DIR, "web", "assets", "final_scores.json")

app = FastAPI(
    title="CarTheftRisk API & Autonomous Bestie Agent",
    version="final",
    description="Statistical estimate for research/education. Not insurance advice.",
)

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


class AutoCompareRequest(BaseModel):
    model: str = Field(..., description="Vehicle model name")
    city: str = Field(..., description="City / municipality name")


class AgentChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500, description="User question in Hebrew")
    model: str = Field("טויוטה קורולה", max_length=120, description="Currently selected vehicle model")
    city: str = Field("תל אביב יפו", max_length=120, description="Currently selected city")
    history: list[dict[str, Any]] | None = Field(default=None, max_length=20, description="Optional chat history")


@lru_cache(maxsize=1)
def _load():
    with open(CARD_PATH, encoding="utf-8") as f:
        card = json.load(f)
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model, card, data


@lru_cache(maxsize=1)
def _load_registry():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_scores():
    path = FINAL_SCORES_PATH if os.path.exists(FINAL_SCORES_PATH) else V9_SCORES_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)["scores"]


_load_v9_scores = _load_scores


@app.get("/health")
def health():
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
    model: str = Query(..., description="Model name as in the web app/data.json"),
    city: str = Query(..., description="City name"),
    model_year: int = Query(2020, ge=1950, le=2026),
):
    mdl, card, data = _load()
    scores = _load_v9_scores()
    m = data["models"].get(model)
    c = data["locations"].get(city)
    if m is None:
        raise HTTPException(404, f"Unknown model: {model}")
    if c is None:
        raise HTTPException(404, f"Unknown city: {city}")

    score = scores.get(f"{m['model_clean']}|{c['city_unified']}")
    if score is None:
        score = {
            "estimated_thefts_2025": None,
            "observed_thefts_2024": 0.0,
            "final_risk_score": None,
            "v9_risk_score": None,
            "coverage": "cold-start: no observed 2024 model/city count",
        }
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
    return run_auto_compare_agent(req.model, req.city)


@app.get("/agent/auto-compare")
def agent_auto_compare_get(
    model: str = Query(..., description="Vehicle model name"),
    city: str = Query(..., description="City name"),
):
    return run_auto_compare_agent(model, city)


@app.post("/agent/chat", response_model=GroundedAgentResponse)
def agent_chat(req: AgentChatRequest):
    return run_chat_agent(req.message, req.model, req.city, req.history)


if os.path.isdir(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
