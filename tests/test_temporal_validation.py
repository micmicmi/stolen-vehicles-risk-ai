import json
from pathlib import Path

from src.temporal_validation import BANNED_FEATURES, FEATURES


ROOT = Path(__file__).resolve().parents[1]


def test_temporal_backtest_contract():
    path = ROOT / "models" / "temporal_backtest_final.json"
    if not path.exists():
        path = ROOT / "models" / "temporal_backtest_v9.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    protocol = result["protocol"]
    assert protocol["train_lag_year"] < protocol["train_target_year"]
    assert protocol["train_target_year"] == protocol["test_lag_year"]
    assert protocol["test_lag_year"] < protocol["test_target_year"]
    assert "proximity_to_corridors" in protocol["excluded_features"]
    assert "population_clean" in protocol["excluded_features"]
    assert result["n_train"] > 0 and result["n_test"] > 0
    assert result["protocol"]["rolling_backtest"]["status"] == "not_available"
    assert "mae_95_ci_low" in result["xgboost_poisson"]
    assert "holdout_diagnostics" in result


def test_v9_feature_contract_excludes_simulated_and_leakage_prone_fields():
    assert not BANNED_FEATURES.intersection(FEATURES)
    source = (ROOT / "src" / "temporal_validation.py").read_text(encoding="utf-8")
    # Deterministic bootstrap resampling is allowed; simulated input features are not.
    assert "np.random.beta" not in source
    assert "proximity_to_corridors" not in FEATURES
