"""Regression tests for the single final runtime and simulation boundary."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agent_tools import compare_alternatives, get_vehicle_risk, load_artifacts


def test_v9_score_does_not_fallback_to_a_legacy_composite_formula():
    result = get_vehicle_risk("TOYOTA COROLLA", "תל אביב - יפו", 2022)
    assert 0 <= result["combined_risk_score"] <= 100
    assert result["model_version"] == "final"
    assert "model_risk_score" not in result
    assert "city_risk_score" not in result
    assert "proximity_to_corridors" not in result


def test_simulation_reports_non_negative_savings():
    result = compare_alternatives("TOYOTA COROLLA", "תל אביב - יפו", top_k=3)
    assert result["alternatives"]
    assert all(item["annual_savings_nis"] >= 0 for item in result["alternatives"])


def test_artifacts_load_with_v9_contract():
    model, card, data = load_artifacts()
    assert model is not None
    assert card["version"] == "final"
    assert data["metadata"]["runtime_model_version"] == "final"
