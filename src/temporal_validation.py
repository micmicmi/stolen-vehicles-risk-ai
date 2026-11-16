"""Leakage-safe, one-step-ahead backtest for CarTheftRisk.

Train on 2023 observations to predict 2024 theft counts and evaluate only on
2025.  It deliberately excludes the composite risk score, population and
corridor-proximity fields: in the current notebook those fields are either
calculated over the full period or may be simulated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "merged_clean.csv"
OUTPUT = ROOT / "models" / "temporal_backtest_final.json"
LEGACY_OUTPUT = ROOT / "models" / "temporal_backtest_v9.json"
MODEL_OUTPUT = ROOT / "models" / "xgb_theft_regressor.json"
CARD_OUTPUT = ROOT / "models" / "model_card.json"
SCORES_OUTPUT = ROOT / "web" / "assets" / "final_scores.json"
LEGACY_SCORES_OUTPUT = ROOT / "web" / "assets" / "v9_scores.json"
DATA_PATH = ROOT / "web" / "assets" / "data.json"
EMBEDDED_DATA_PATH = ROOT / "web" / "assets" / "embedded_data.js"
STATUS_OUTPUT = ROOT / "web" / "assets" / "research_status.json"
CHART = ROOT / "web" / "assets" / "charts" / "temporal_backtest_final.png"
LEGACY_CHART = ROOT / "web" / "assets" / "charts" / "temporal_backtest_v9.png"
SCATTER_CHART = ROOT / "web" / "assets" / "charts" / "temporal_actual_vs_pred_final.png"
LEGACY_SCATTER_CHART = ROOT / "web" / "assets" / "charts" / "temporal_actual_vs_pred_v9.png"


FEATURES = [
    "model_year",
    "lag_thefts",
    "city_detection_rate",
    "city_unsolved_rate",
    "city_cases_total",
]

# Contract: every feature must have been observable by the prediction cutoff.
# Do not add derived full-period scores or simulated enrichments here.
BANNED_FEATURES = {
    "risk_score",
    "model_risk_score",
    "population_clean",
    "proximity_to_corridors",
    "growth_factor",
}


def _metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    predicted = np.maximum(0, predicted)
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 4),
        "rmse": round(float(mean_squared_error(actual, predicted) ** 0.5), 4),
        "r2": round(float(r2_score(actual, predicted)), 4),
    }


def _mae_confidence_interval(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Deterministic bootstrap interval for holdout MAE."""
    actual_values = actual.to_numpy(dtype=float)
    predicted_values = np.maximum(0, np.asarray(predicted, dtype=float))
    rng = np.random.default_rng(42)
    samples = np.empty(400, dtype=float)
    for index in range(len(samples)):
        rows = rng.integers(0, len(actual_values), size=len(actual_values))
        samples[index] = mean_absolute_error(actual_values[rows], predicted_values[rows])
    return {
        "mae_95_ci_low": round(float(np.quantile(samples, 0.025)), 4),
        "mae_95_ci_high": round(float(np.quantile(samples, 0.975)), 4),
    }


def _feature_drift(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, float]:
    """Descriptive feature drift; it is not a causal or production monitor."""
    drift = {}
    for feature in FEATURES:
        baseline_std = float(train[feature].std()) or 1.0
        drift[feature] = round(float((test[feature].mean() - train[feature].mean()) / baseline_std), 4)
    return drift


def run_backtest() -> dict:
    if BANNED_FEATURES.intersection(FEATURES):
        raise ValueError("Leakage-prone or simulated feature included in v9")
    raw = pd.read_csv(INPUT)
    required = set(
        ["model_year", "city_detection_rate", "city_unsolved_rate", "city_cases_total"]
        + ["thefts_2023", "thefts_2024", "thefts_2025"]
    )
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Values from police-case history are fixed at the pre-2024 cutoff.  All
    # nonnumeric values become zero only after explicit coercion.
    frame = raw.copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    train = frame.assign(lag_thefts=frame["thefts_2023"], target=frame["thefts_2024"])
    test = frame.assign(lag_thefts=frame["thefts_2024"], target=frame["thefts_2025"])

    # Poisson is trained on raw non-negative counts, not on log1p(counts).
    model = xgb.XGBRegressor(
        objective="count:poisson",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=1,
    )
    model.fit(train[FEATURES], train["target"])
    prediction = model.predict(test[FEATURES])
    model.save_model(MODEL_OUTPUT)

    xgb_metrics = _metrics(test["target"], prediction)
    xgb_metrics.update(_mae_confidence_interval(test["target"], prediction))
    persistence_metrics = _metrics(test["target"], test["lag_thefts"].to_numpy())
    median_metrics = _metrics(test["target"], np.full(len(test), train["target"].median()))
    result = {
        "protocol": {
            "name": "one-step-ahead chronological backtest",
            "train_target_year": 2024,
            "train_lag_year": 2023,
            "test_target_year": 2025,
            "test_lag_year": 2024,
            "features": FEATURES,
            "excluded_features": [
                "risk_score / model_risk_score",
                "population_clean",
                "proximity_to_corridors",
                "growth_factor",
            ],
            "caveat": "This validates one-year-ahead count prediction, not an individual vehicle theft probability.",
            "rolling_backtest": {
                "status": "not_available",
                "reason": "The processed source provides annual target columns only for 2023-2025; earlier annual counts are required for additional chronological folds.",
            },
        },
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "xgboost_poisson": xgb_metrics,
        "persistence_baseline": persistence_metrics,
        "median_baseline": median_metrics,
        "holdout_diagnostics": {
            "zero_theft_share": round(float((test["target"] == 0).mean()), 4),
            "feature_standardized_mean_shift": _feature_drift(train, test),
            "interpretation": "Descriptive only. Feature drift cannot be monitored prospectively without a newer observed year.",
        },
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    LEGACY_OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    CARD_OUTPUT.write_text(
        json.dumps(
            {
                "version": "final",
                "pipeline": "CarTheftRisk_final",
                "status": "research_only",
                "task": "one-step-ahead annual theft-count estimate",
                "features": FEATURES,
                "evaluation": result,
                "disclaimer": "Not an individual theft forecast, insurance quote, or production model.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    scale = float(np.quantile(np.maximum(0, prediction), 0.99)) or 1.0
    scores = {}
    for row, pred in zip(test.itertuples(index=False), np.maximum(0, prediction)):
        key = f"{row.model_clean}|{row.city}"
        risk_val = round(float(min(100, pred / scale * 100)), 1)
        scores[key] = {
            "estimated_thefts_2025": round(float(pred), 3),
            "observed_thefts_2024": round(float(row.thefts_2024), 3),
            "final_risk_score": risk_val,
            "v9_risk_score": risk_val,
            "coverage": "observed 2024 model/city count available",
        }
    SCORES_OUTPUT.write_text(
        json.dumps({"version": "final", "status": "research_only", "scores": scores}, ensure_ascii=False),
        encoding="utf-8",
    )
    LEGACY_SCORES_OUTPUT.write_text(
        json.dumps({"version": "v9", "status": "research_only", "scores": scores}, ensure_ascii=False),
        encoding="utf-8",
    )
    # `data.json` remains the single web lookup dataset. Its metadata is rebuilt
    # here, so standalone readers see the active final contract as well.
    web_data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    original_metadata = web_data.get("metadata", {})
    web_data["metadata"] = {
        "pipeline_version": "CarTheftRisk_final",
        "runtime_model_version": "final",
        "runtime_model_status": "research_only",
        "dataset_role": "historical vehicle/city lookup with final score overlay",
        "observed_years": [2023, 2024, 2025],
        "source": original_metadata.get("source", "Police open-data vehicle theft records"),
        "runtime_scores_source": "web/assets/final_scores.json",
        "data_cutoff_for_prediction": "2024 observed counts; evaluated against 2025",
        "disclaimer": "Research-only annual count estimate. Not an individual theft forecast or insurance quote.",
    }
    # Embed the same final-overlaid lookup for `file://` usage. This is a delivery
    # fallback, not a second source of truth: it is regenerated from web_data on
    # every backtest run.
    web_data["model_location"] = web_data.get("model_location", {})
    for key, score in scores.items():
        web_data["model_location"][key] = {
            "count": score["estimated_thefts_2025"],
            "actual": score["observed_thefts_2024"],
            "avg_risk": score["final_risk_score"],
            "model_version": "final",
        }
    DATA_PATH.write_text(json.dumps(web_data, ensure_ascii=False), encoding="utf-8")
    EMBEDDED_DATA_PATH.write_text(
        "// Generated by src.temporal_validation.py; do not edit manually.\n"
        "window.EMBEDDED_DATA = "
        + json.dumps(web_data, ensure_ascii=False)
        + ";\n",
        encoding="utf-8",
    )
    STATUS_OUTPUT.write_text(
        json.dumps(
            {
                "runtime_scores": {
                    "model_version": "final",
                    "source": "web/assets/final_scores.json",
                    "status": "research_only",
                },
                "research_backtest": result,
                "insurance": {
                    "status": "simulation_only",
                    "disclaimer": "Reference-driver pricing simulation; no insurer or Bestie quote is fetched.",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    labels = ["XGBoost\nPoisson", "Persistence", "Median\nbaseline"]
    values = [result["xgboost_poisson"]["mae"], result["persistence_baseline"]["mae"], result["median_baseline"]["mae"]]
    colors = ["#2563eb", "#94a3b8", "#f59e0b"]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_title("v9 chronological backtest — MAE on 2025", weight="bold")
    ax.set_ylabel("Mean absolute error (vehicles)")
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", weight="bold")
    fig.text(0.5, 0.01, "Lower is better. The median baseline remains the MAE reference.", ha="center", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    CHART.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART, bbox_inches="tight")
    fig.savefig(LEGACY_CHART, bbox_inches="tight")
    plt.close(fig)

    # A compact diagnostic: prediction quality across the 2025 holdout.
    sample = pd.DataFrame({"actual": test["target"], "predicted": np.maximum(0, prediction)}).sample(
        min(3_000, len(test)), random_state=42
    )
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.scatter(sample["actual"], sample["predicted"], s=9, alpha=0.22, color="#2563eb", edgecolors="none")
    limit = max(float(sample["actual"].max()), float(sample["predicted"].max()), 1.0)
    ax.plot([0, limit], [0, limit], "--", color="#f59e0b", lw=1.5, label="perfect prediction")
    ax.set_title("v9 holdout: actual vs predicted 2025 counts", weight="bold")
    ax.set_xlabel("Actual theft count")
    ax.set_ylabel("Predicted theft count")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(SCATTER_CHART, bbox_inches="tight")
    fig.savefig(LEGACY_SCATTER_CHART, bbox_inches="tight")
    plt.close(fig)
    return result


if __name__ == "__main__":
    print(json.dumps(run_backtest(), ensure_ascii=False, indent=2))
