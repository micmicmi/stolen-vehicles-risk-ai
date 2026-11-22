"""Integration tests for the FastAPI live server (API endpoints + static web frontend)."""
import os
import sys
import types

from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api import app  # noqa: E402

client = TestClient(app)


def test_root_serves_web_index():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "/agent/auto-compare" in resp.text
    assert "http://127.0.0.1:8000/agent/" not in resp.text

    final_page = client.get("/final.html")
    assert final_page.status_code == 200
    assert "text/html" in final_page.headers.get("content-type", "")

    data_asset = client.get("/assets/data.json")
    assert data_asset.status_code == 200


def test_health_and_predict_endpoints():
    health = client.get("/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["status"] == "ok"
    assert payload["agent_enabled"] is True

    pred = client.get("/predict", params={"model": "טויוטה קורולה", "city": "תל אביב יפו", "model_year": 2022})
    assert pred.status_code == 200
    pred_data = pred.json()
    assert pred_data["final_estimated_thefts_2025"] >= 0


def test_agent_endpoints_without_and_with_gemini(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    comp = client.post("/agent/auto-compare", json={"model": "TOYOTA COROLLA", "city": "תל אביב - יפו"})
    assert comp.status_code == 200
    comp_data = comp.json()
    assert comp_data["status"] == "ok"
    assert comp_data["alternative_cards"]

    chat = client.post(
        "/agent/chat",
        json={"message": "איך אפשר להוזיל את הביטוח עם מיגון?", "model": "TOYOTA COROLLA", "city": "תל אביב - יפו"},
    )
    assert chat.status_code == 200
    chat_data = chat.json()
    assert chat_data["status"] == "answered"
    assert chat_data["engine"] == "deterministic_grounded"
    assert chat_data["reply"]

    # Verify Gemini path when GEMINI_API_KEY is configured
    fake_models = types.SimpleNamespace(
        generate_content=lambda model, contents: types.SimpleNamespace(text="תשובה מבוססת ג'מיני")
    )
    fake_client = types.SimpleNamespace(models=fake_models)
    fake_genai = types.SimpleNamespace(Client=lambda api_key: fake_client)
    fake_google = types.SimpleNamespace(genai=fake_genai)

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    gemini_chat = client.post(
        "/agent/chat",
        json={"message": "מה החלופה הכי זולה?", "model": "TOYOTA COROLLA", "city": "תל אביב - יפו"},
    )
    assert gemini_chat.status_code == 200
    gemini_data = gemini_chat.json()
    assert gemini_data["engine"] == "gemini_verified"
    assert gemini_data["reply"] == "תשובה מבוססת ג'מיני"
