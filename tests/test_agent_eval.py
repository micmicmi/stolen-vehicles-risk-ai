"""
test_agent_eval.py
==================
בדיקות אינטגרציה אוטומטיות עבור חבילת ה-Agent Evaluation (`src/agent_eval.py`).
מטרתן לוודא שמנגנון ההערכה עצמו (Golden Dataset, Verifier Stress, Auto-Compare Invariants)
עובד תקין, אינו שובר חוזים, וממשיך לדווח אמת בלי תלות במפתח API חיצוני.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent_eval import (  # noqa: E402
    GOLDEN_DATASET_PATH,
    AgentEvalReport,
    JudgeScore,
    evaluate_auto_compare_suite,
    load_golden_dataset,
    run_full_evaluation,
    run_live_gemini_and_judge_eval,
    run_verifier_stress_suite,
)


def test_golden_dataset_loads_cleanly():
    """מוודא שקובץ ה-Golden Dataset נטען במלואו ועומד בסכמת GoldenCase."""
    version, cases = load_golden_dataset(GOLDEN_DATASET_PATH)
    assert version == "1.0"
    assert len(cases) >= 30


def test_verifier_stress_has_perfect_recall_and_zero_false_alarms():
    """מוודא שמבחן העמידות של שכבה 4 מזהה 100% מההזיות המספריות עם 0% התראות שווא."""
    stress = run_verifier_stress_suite()
    assert stress["total_cases"] == 10
    assert stress["passed_cases"] == 10
    assert stress["hallucination_detection_recall"] == 1.0
    assert stress["false_alarm_rate"] == 0.0


def test_auto_compare_invariants_pass_all_pairs():
    """מוודא שסוכן ה-Auto-Compare עומד בכל אינווריאנטים (שלבים, חישוב, וללא חברות אסורות)."""
    summary = evaluate_auto_compare_suite()
    assert summary.total_pairs >= 5
    assert summary.passed_pairs == summary.total_pairs
    assert summary.pass_rate == 1.0


def test_full_agent_eval_produces_valid_report(tmp_path):
    """מריץ את חבילת ההערכה המלאה במצב Offline ומוודא שמבנה הדוח תקין ועקבי."""
    out_file = tmp_path / "eval_out.json"
    report = run_full_evaluation(GOLDEN_DATASET_PATH, out_file, run_live=False)
    validated = AgentEvalReport.model_validate(report.model_dump())
    assert out_file.exists()
    assert validated.total_cases == len(validated.all_case_results)
    assert validated.schema_valid_rate == 1.0
    assert validated.savings_sign_accuracy == 1.0


def test_openai_judge_mode_with_mocked_scorer(monkeypatch):
    """מוודא שמסלול OpenAI LLM-as-a-Judge מפענח תשובות ומחשב ממוצעים כראוי (ב-Mock ללא תלות ברשת)."""
    _, cases = load_golden_dataset(GOLDEN_DATASET_PATH)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        "src.agent_eval._call_openai_judge",
        lambda prompt, api_key, model_name="gpt-4o-mini": JudgeScore(
            faithfulness=5,
            numeric_consistency=5,
            clarity_he=4,
            explanation="בדיקת מוק תקינה",
        ),
    )

    live_summary = run_live_gemini_and_judge_eval(
        cases, sample_ids=("MC-01", "EG-01"), judge_provider="openai"
    )
    assert live_summary.status == "executed"
    assert live_summary.judge_provider == "openai"
    assert live_summary.judge_evaluated_count == 2
    assert live_summary.mean_judge_faithfulness == 5.0
    assert live_summary.mean_judge_clarity_he == 4.0
