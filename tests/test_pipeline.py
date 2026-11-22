"""Sanity tests for the final research pipeline, API, and insurance simulation.
Run from the project root:  pytest -q
"""
import json
import os
import sys

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
DATA = os.path.join(BASE, "web", "assets", "data.json")


@pytest.fixture(scope="module")
def data():
    with open(DATA, encoding="utf-8") as f:
        return json.load(f)


def test_web_data_declares_the_v9_runtime_contract(data):
    meta = data["metadata"]
    assert meta["pipeline_version"] == "CarTheftRisk_final"
    assert meta["runtime_model_version"] == "final"
    assert meta["runtime_model_status"] == "research_only"
    assert meta["runtime_scores_source"] == "web/assets/final_scores.json"


def test_model_location_has_predictions(data):
    ml = data["model_location"]
    assert len(ml) > 1000
    assert all(v["count"] >= 0 for v in ml.values())


def test_every_model_entry_links_to_predictions(data):
    keys = {k.split("|")[0] for k in data["model_location"]}
    linked = [m for m in data["models"].values() if m.get("model_clean") in keys]
    assert len(linked) > 100


def test_v9_score_artifact_is_covered_and_bounded(data):
    with open(os.path.join(BASE, "web", "assets", "final_scores.json"), encoding="utf-8") as f:
        scores = json.load(f)["scores"]
    embedded_final_rows = [
        row for row in data["model_location"].values()
        if row.get("model_version") == "final"
    ]
    assert len(embedded_final_rows) == len(scores)
    assert all(0 <= score["final_risk_score"] <= 100 for score in scores.values())
    assert all(score["coverage"] == "observed 2024 model/city count available" for score in scores.values())


def test_api_predict_roundtrip(data):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from src.api import app

    client = TestClient(app)
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["model_version"] == "final"
    assert health["pipeline_version"] == "final"
    model = next(k for k, m in data["models"].items() if m.get("model_clean") in {x.split("|")[0] for x in data["model_location"]})
    city = next(iter(data["locations"]))
    r = client.get("/predict", params={"model": model, "city": city})
    assert r.status_code == 200
    assert r.json()["final_estimated_thefts_2025"] >= 0
    assert r.json()["model_version"] == "final"
    assert "coverage_status" in r.json()
    assert client.get("/predict", params={"model": "nope", "city": city}).status_code == 404


def test_agent_tools_and_bestie_comparison():
    from src.agent_tools import (
        compare_alternatives,
        fetch_bestie_insurance_quotes,
        get_vehicle_risk,
        simulate_mitigation_discount,
    )

    risk = get_vehicle_risk("טויוטה קורולה", "תל אביב יפו")
    assert 0 <= risk["combined_risk_score"] <= 100
    assert risk["xgb_predicted_annual_thefts"] >= 0
    assert risk["model_version"] == "final"
    assert "model_risk_score" not in risk

    quote = fetch_bestie_insurance_quotes("טויוטה קורולה", "תל אביב יפו")
    assert quote["total_annual_nis"] == quote["comprehensive_nis"] + quote["mandatory_nis"]
    assert quote["total_annual_nis"] > 3000

    comp = compare_alternatives("טויוטה קורולה", "תל אביב יפו", top_k=3)
    assert len(comp["alternatives"]) == 3
    for alt in comp["alternatives"]:
        assert alt["annual_savings_nis"] > 0
        assert alt["total_annual_nis"] < comp["selected_vehicle"]["total_annual_nis"]

    mit = simulate_mitigation_discount("טויוטה קורולה", "תל אביב יפו")
    assert mit["mitigated_total_nis"] < mit["original_total_nis"]
    assert mit["annual_discount_nis"] > 0


def test_api_agent_endpoints():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from src.api import app

    client = TestClient(app)

    r_auto = client.post("/agent/auto-compare", json={"model": "יונדאי טוסון", "city": "פתח תקווה"})
    assert r_auto.status_code == 200
    payload = r_auto.json()
    assert payload["status"] == "ok"
    assert payload["engine"] == "research_bestie_simulation_final"
    assert len(payload["agent_steps"]) == 3
    assert len(payload["alternative_cards"]) == 3
    assert payload["selected_vehicle_card"]["total_annual_nis"] > 0

    r_chat = client.post(
        "/agent/chat",
        json={"message": "איך אפשר להוזיל את הביטוח עם מיגון?", "model": "יונדאי טוסון", "city": "פתח תקווה"},
    )
    assert r_chat.status_code == 200
    chat_data = r_chat.json()
    assert chat_data["status"] == "answered"
    assert "simulate_mitigation_discount" in chat_data["tools_used"]
    assert "₪" in chat_data["reply"]


def test_agent_report_and_links_exist():
    agent_report_path = os.path.join(BASE, "web", "agent_report.html")
    final_report_path = os.path.join(BASE, "web", "final.html")
    index_path = os.path.join(BASE, "web", "index.html")

    assert os.path.exists(agent_report_path)
    with open(agent_report_path, encoding="utf-8") as f:
        agent_html = f.read()
    assert "agent_screenshot_gallery.png" in agent_html
    assert "agent_screenshot_chat.png" in agent_html

    with open(final_report_path, encoding="utf-8") as f:
        final_html = f.read()
    assert "agent_report.html" in final_html

    with open(index_path, encoding="utf-8") as f:
        idx_html = f.read()
    assert 'id="ai-agent-section"' in idx_html
    assert 'id="bestie-cards-gallery"' in idx_html
    assert 'id="agent-inline-chat"' in idx_html
