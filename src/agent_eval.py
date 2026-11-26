"""
agent_eval.py
=============
חבילת הערכה מקצה לקצה (End-to-End Agent Evaluation Suite) לסוכן CarTheftRisk (Bestie Copilot).

מה קובץ זה עושה ולמה הוא נפרד מבדיקות התוכנה הרגילות (`tests/`)?
--------------------------------------------------------------------
בדיקות `pytest` בודקות פונקציונליות נקודתית (Unit & Contract Tests), ואילו הערכת סוכן (Agent Eval)
מודדת את ביצועי הסוכן בצורה כמותית ושיטתית על פני 4 צירים מקצועיים:

1. **Golden Dataset Evaluation (`run_golden_suite`):**
   הרצת 35 תרחישי זהב בעברית (השוואת דגמים, מעבר ערים, מיגון, שאלות המשך, דגמים לא קיימים,
   ו-Adversarial / Red Teaming) וחישוב מדדי דיוק:
   - Status Accuracy (האם הסוכן ענה / ביקש הבהרה / חסם בהתאם לנדרש)
   - Intent Classification Accuracy (דיוק זיהוי הכוונה מתוך 7 כוונות אפשריות)
   - Entity & Direction Accuracy (האם זוהו דגם/עיר היעד והסימן הנכון של החיסכון)
   - Latency (`mean`, `p50`, `p95` במילישניות) לכל שאילתה.

2. **Auto-Compare Invariants (`evaluate_auto_compare_suite`):**
   בדיקת המצב האוטונומי (`Zero-Click` ב-`run_auto_compare_agent`) על מטריצת שילובי דגם×עיר נפוצים:
   אימות שהשרשרת של 3 הכלים מחזירה תמיד כרטיסיות תקינות, חיסכון לא-שלילי, וללא חברות אסורות.

3. **Numeric Verifier Stress Test (`run_verifier_stress_suite`):**
   מבחן עמידות של שכבה 4 (`verify_numeric_grounding`) מול 10 תגובות מדומות של LLM —
   חלקן עם הזיות מספריות (סכום מומצא, עיגול סוטה ב-₪5, אחוז הנחה בדוי, ציון סיכון מומצא)
   וחלקן תשובות תקינות המשתמשות אך ורק במספרי הכלים וקבועיי הפרופיל — כדי למדוד
   Detection Rate (Recall) ו-False Alarm Rate (FAR).

4. **Live Gemini + LLM-as-a-Judge (`run_live_gemini_and_judge_eval`):**
   כאשר מוגדר מפתח `GEMINI_API_KEY` וחבילת `google-genai` מותקנת, פונקציה זו מריצה את המודל החי
   על מדגם מה-Golden Dataset, מודדת אחוז מעבר (`gemini_verified`) מול נפילה ל-Fallback,
   ומפעילה **שופט LLM אוטומטי (`LLM-as-a-Judge`)** לבדיקת נאמנות סמנטית ובהירות בעברית.
   כשאין מפתח בסביבה, ההרצה מתועדת כ-`skipped_no_api_key` מבלי לזייף תוצאות.

הרצה ישירה מהטרמינל (מתוך תיקיית הפרויקט):
    python -X utf8 -m src.agent_eval
"""
from __future__ import annotations

import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

# ייבוא רכיבי הסוכן האמיתיים מתוך הפרויקט (ללא חיקוי או שכפול לוגיקה)
from src.agent import (
    collect_allowed_numbers,
    plan_query,
    run_auto_compare_agent,
    run_chat_agent,
    verify_numeric_grounding,
)
from src.agent_schemas import GroundedAgentResponse, IntentEnum

# הגדרת נתיבי קבצים קנוניים ביחס לשורש הפרויקט
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE_PATH = PROJECT_ROOT / ".env"
GOLDEN_DATASET_PATH = PROJECT_ROOT / "evals" / "golden_dataset.json"
EVAL_RESULTS_PATH = PROJECT_ROOT / "evals" / "agent_eval_results.json"

# רשימת מותגים שהוסרו מהסימולציה לטובת שקיפות מול שותפי Bestie (חייבים להיות 0 מופעים בפלט)
FORBIDDEN_INSURERS = ("ביטוח ישיר", "ליברה", "AIG")


def load_env_file(env_path: Path = ENV_FILE_PATH) -> bool:
    """
    טוען משתני סביבה מקובץ `.env` לתוך `os.environ` בצורה בטוחה וללא תלות בחבילה חיצונית.
    - קורא שורה-אחר-שורה, מדלג על הערות (`#`) ועל שורות ריקות.
    - מסיר מרכאות עוטפות (`"..."` או `'...'`) אם הוזנו סביב המפתח.
    - אינו מדפיס לעולם את ערכי המפתחות כדי לשמור על אבטחת סודות (Secrets Hygiene).
    """
    if not env_path.exists():
        return False

    loaded_any = False
    # קריאת הקובץ בקידוד UTF-8 ופירוק לשורות בודדות
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        # דילוג על שורות ריקות או שורות הערה המתחילות ב-#
        if not line or line.startswith("#") or "=" not in line:
            continue
        # תמיכה בתחביר נפוץ של 'export KEY=VALUE' או 'KEY=VALUE'
        if line.startswith("export "):
            line = line[len("export "):].strip()
        # הפרדת שם המשתנה מערכו לפי הסימן '=' הראשון בלבד
        key, val = line.split("=", 1)
        key = key.strip()
        # ניקוי רווחים ומרכאות מיותרות מהערך
        val = val.strip().strip('"').strip("'")
        # טעינה ל-os.environ רק אם המשתנה לא הוגדר כבר ידנית במערכת
        if key and val and key not in os.environ:
            os.environ[key] = val
            loaded_any = True
    return loaded_any


# ==============================================================================
# שלב 1: הגדרת סכמות הפלט והדוח של ה-Eval באמצעות Pydantic v2
# ==============================================================================
class GoldenCase(BaseModel):
    """מייצג מקרה מבחן בודד מתוך קובץ ה-Golden Dataset."""

    id: str  # מזהה ייחודי לקריאה ומעקב (למשל MC-01)
    category: str  # קטגוריית ההערכה (model_comparison, city_comparison, adversarial_and_scope וכו')
    description: str  # הסבר קצר בעברית מה המקרה בודק
    message: str  # הטקסט החופשי שהמשתמש שולח לצ'אט
    base_model: str  # הדגם שבחור כרגע בממשק ה-Web
    base_city: str  # העיר שבחורה כרגע בממשק ה-Web
    history: list[dict[str, str]] = Field(default_factory=list)  # היסטוריית שיחה קודמת (Multi-turn)
    expected_status: Literal["answered", "needs_clarification", "blocked"]  # הסטטוס הצפוי של התשובה
    expected_intent: str  # הכוונה הצפויה מתוך IntentEnum
    expected_base_model_substr: str | None = None  # תת-מחרוזת שאמורה להופיע בשם דגם הבסיס (אופציונלי)
    expected_target_model_substr: str | None = None  # תת-מחרוזת שאמורה להופיע בשם דגם היעד (אופציונלי)
    expected_base_city: str | None = None  # שם עיר הבסיס הצפויה (אופציונלי)
    expected_target_city: str | None = None  # שם עיר היעד הצפויה (אופציונלי)
    expected_savings_sign: Literal["positive", "negative", "any", "none"] = "any"  # כיוון החיסכון הצפוי
    must_contain: list[str] = Field(default_factory=list)  # ביטויים שחייבים להופיע בתשובת הסוכן
    must_not_contain: list[str] = Field(default_factory=list)  # ביטויים שאסור שיופיעו בתשובה


class CaseEvalResult(BaseModel):
    """תוצאת ההרצה והבדיקה של מקרה זהב בודד."""

    id: str
    category: str
    description: str
    passed: bool  # True רק אם כל הבדיקות במקרה זה עברו בהצלחה
    latency_ms: float  # זמן ריצה במילישניות למדידת ביצועי המערכת
    schema_valid: bool  # האם הפלט עבר אימות קשיח מול סכמת GroundedAgentResponse
    status_match: bool  # האם הסטטוס בפועל תואם למצופה
    intent_match: bool  # האם הכוונה (Intent) שזוהתה תואמת למצופה
    entity_match: bool  # האם ישויות היעד/בסיס חולצו במדויק
    savings_sign_match: bool  # האם כיוון החיסכון (חיובי/שלילי) תואם למציאות האקטוארית
    content_match: bool  # האם must_contain ו-must_not_contain התקיימו
    actual_status: str
    actual_intent: str
    failure_reasons: list[str] = Field(default_factory=list)  # פירוט מדויק של כל כשל אם התרחש


class NumericVerifierStressResult(BaseModel):
    """תוצאה למקרה בודד במבחן העמידות של שכבה 4 (Post-Generation Numeric Verifier)."""

    case_id: str
    description: str
    expected_valid: bool  # True = תשובה תקינה; False = הזיה שצריכה להיחסם
    actual_valid: bool  # מה קבע verify_numeric_grounding בפועל
    passed: bool
    flagged_tokens: list[str] = Field(default_factory=list)  # המספרים החריגים שה-Verifier זיהה


class AutoCompareEvalSummary(BaseModel):
    """סיכום בדיקת המסלול האוטונומי (Zero-Click Auto-Compare) על זוגות דגם×עיר."""

    total_pairs: int
    passed_pairs: int
    pass_rate: float
    mean_latency_ms: float
    failures: list[str] = Field(default_factory=list)


class JudgeScore(BaseModel):
    """סכמת JSON מובנית עבור שופט אוטומטי (LLM-as-a-Judge) בהערכה החיה."""

    faithfulness: int = Field(..., ge=1, le=5)  # נאמנות למספרים ולעובדות מהכלים (1–5)
    numeric_consistency: int = Field(..., ge=1, le=5)  # התאמה בין הטקסט לכרטיס SavingsMetrics (1–5)
    clarity_he: int = Field(..., ge=1, le=5)  # בהירות הניסוח בעברית (1–5)
    explanation: str  # נימוק קצר של השופט


class LiveLLMEvalSummary(BaseModel):
    """סיכום הרצה חיה מול מודל הסוכן (Gemini) ושופט אוטומטי (OpenAI או Gemini LLM-as-a-Judge)."""

    status: Literal["executed", "skipped_no_api_key", "error"]
    reason: str
    model_used: str = ""  # תיאור מודל הסוכן שנבדק
    judge_provider: str = ""  # ספק השופט שביצע את ההערכה בפועל ('openai' / 'gemini' / '')
    judge_model: str = ""  # שם המודל המדויק של השופט (למשל gpt-4o-mini)
    sample_size: int = 0
    judge_evaluated_count: int = 0  # כמה תשובות סוכן דורגו בהצלחה על ידי השופט החי
    gemini_verified_count: int = 0
    verifier_fallback_count: int = 0
    gemini_verified_rate: float | None = None
    mean_live_latency_ms: float | None = None
    mean_judge_faithfulness: float | None = None
    mean_judge_numeric_consistency: float | None = None
    mean_judge_clarity_he: float | None = None
    details: list[dict[str, Any]] = Field(default_factory=list)


class AgentEvalReport(BaseModel):
    """דוח ההערכה המלא שנשמר כ-JSON ומוצג בסיכום."""

    generated_at_utc: str
    dataset_version: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    overall_pass_rate: float
    schema_valid_rate: float
    status_accuracy: float
    intent_accuracy: float
    entity_accuracy: float
    savings_sign_accuracy: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    by_category: dict[str, dict[str, Any]]
    verifier_stress: dict[str, Any]
    auto_compare_invariants: AutoCompareEvalSummary
    live_llm_eval: LiveLLMEvalSummary
    failures_detail: list[CaseEvalResult]
    all_case_results: list[CaseEvalResult]


# ==============================================================================
# שלב 2: טעינת ה-Golden Dataset מקובץ ה-JSON
# ==============================================================================
def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> tuple[str, list[GoldenCase]]:
    """
    טוען ומאמת את מקרי הבדיקה מתוך golden_dataset.json.
    השימוש ב-GoldenCase.model_validate מבטיח שאין שגיאות מבנה בדאטהסט עצמו.
    """
    # קריאת קובץ ה-JSON בקידוד UTF-8 לתמיכה תקינה בעברית
    raw = json.loads(path.read_text(encoding="utf-8"))
    # שליפת מספר הגרסה של הדאטהסט לצורך מעקב היסטורי
    version = str(raw.get("dataset_version", "1.0"))
    # המרת כל רשומת מילון לאובייקט GoldenCase מאומת
    cases = [GoldenCase.model_validate(item) for item in raw.get("cases", [])]
    return version, cases


# ==============================================================================
# שלב 3: הרצת מקרה זהב בודד מול הסוכן והערכת כל הצירי הבדיקה
# ==============================================================================
def evaluate_golden_case(case: GoldenCase, force_offline: bool = True) -> CaseEvalResult:
    """
    מריץ מקרה מבחן בודד מול run_chat_agent ובודק 6 תנאים במקביל:
    1. תקינות סכמה (GroundedAgentResponse)
    2. התאמת Status (answered / needs_clarification / blocked)
    3. התאמת Intent
    4. דיוק חילוץ ישויות (דגם/עיר בסיס ויעד)
    5. נכונות כיוון החיסכון (חיובי מול תוספת עלות שלילית)
    6. עמידה באילוצי תוכן (must_contain / must_not_contain)
    """
    # שמירת מפתחות ה-API אם קיימים, וניקוים הזמני בהרצה דטרמיניסטית כדי לבודד את הלוגיקה מה-LLM החיצוני
    prev_gemini_key = os.environ.get("GEMINI_API_KEY")
    prev_openai_key = os.environ.get("OPENAI_API_KEY")
    if force_offline:
        if prev_gemini_key is not None:
            os.environ.pop("GEMINI_API_KEY", None)
        if prev_openai_key is not None:
            os.environ.pop("OPENAI_API_KEY", None)

    try:
        # מדידת זמן התחלה ברזולוציית ננו-שניות למדידת Latency מדויקת
        t0 = time.perf_counter()
        # הרצה כפולה של תכנון השאילתה והסוכן המלא (run_chat_agent מחזיר dict)
        plan = plan_query(case.message, case.base_model, case.base_city, history=case.history or None)
        raw_response = run_chat_agent(
            message=case.message,
            model=case.base_model,
            city=case.base_city,
            history=case.history or None,
        )
        # חישוב משך הריצה במילישניות
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    finally:
        # שחזור משתני הסביבה תמיד (גם במקרה חריגה) כדי לא לפגוע בהרצות הבאות
        if force_offline:
            if prev_gemini_key is not None:
                os.environ["GEMINI_API_KEY"] = prev_gemini_key
            if prev_openai_key is not None:
                os.environ["OPENAI_API_KEY"] = prev_openai_key

    failures: list[str] = []

    # 1. אימות סכמת הפלט מול החוזה הרשמי של Pydantic
    schema_valid = True
    parsed: GroundedAgentResponse | None = None
    try:
        parsed = GroundedAgentResponse.model_validate(raw_response)
    except Exception as exc:
        schema_valid = False
        failures.append(f"Schema validation error: {exc}")

    actual_status = parsed.status if parsed else str(raw_response.get("status", "unknown"))
    actual_intent = parsed.intent.value if parsed else str(raw_response.get("intent", "unknown"))
    reply_text = parsed.reply if parsed else str(raw_response.get("reply", ""))
    metrics = parsed.metrics if parsed else None

    # 2. בדיקת התאמת הסטטוס (answered / blocked / needs_clarification)
    status_match = actual_status == case.expected_status
    if not status_match:
        failures.append(f"Status mismatch: expected='{case.expected_status}', actual='{actual_status}'")

    # 3. בדיקת התאמת הכוונה (Intent Classification)
    intent_match = actual_intent == case.expected_intent
    if not intent_match:
        failures.append(f"Intent mismatch: expected='{case.expected_intent}', actual='{actual_intent}'")

    # 4. בדיקת חילוץ ישויות (Base & Target Model/City) בהתאם לערכים שציפינו להם
    entity_match = True
    # בדיקת דגם הבסיס במידה והמקרה דורש זאת
    if case.expected_base_model_substr:
        actual_bm = (metrics.base_model_he if metrics else "") or ""
        if case.expected_base_model_substr not in actual_bm:
            entity_match = False
            failures.append(
                f"Base model mismatch: expected substr '{case.expected_base_model_substr}' in '{actual_bm}'"
            )
    # בדיקת דגם היעד (מתוך Metrics או מתוך QueryPlan אם נדחה ב-needs_clarification)
    if case.expected_target_model_substr:
        actual_tm = (metrics.target_model_he if metrics else None) or (plan.target_model or "")
        if case.expected_target_model_substr not in actual_tm:
            entity_match = False
            failures.append(
                f"Target model mismatch: expected substr '{case.expected_target_model_substr}' in '{actual_tm}'"
            )
    # בדיקת עיר הבסיס
    if case.expected_base_city:
        actual_bc = (metrics.base_city if metrics else "") or ""
        if actual_bc != case.expected_base_city:
            entity_match = False
            failures.append(f"Base city mismatch: expected='{case.expected_base_city}', actual='{actual_bc}'")
    # בדיקת עיר היעד
    if case.expected_target_city:
        actual_tc = (metrics.target_city if metrics else None) or (plan.target_city or "")
        if actual_tc != case.expected_target_city:
            entity_match = False
            failures.append(f"Target city mismatch: expected='{case.expected_target_city}', actual='{actual_tc}'")

    # 5. בדיקת סימן החיסכון (מוודא שרכב יקר יותר אכן מוצג כשלילי = תוספת עלות, וזול יותר כחיובי)
    savings_sign_match = True
    if case.expected_savings_sign == "positive":
        if metrics is None or metrics.annual_savings_nis <= 0:
            savings_sign_match = False
            val = metrics.annual_savings_nis if metrics else None
            failures.append(f"Savings sign mismatch: expected > 0, got {val}")
    elif case.expected_savings_sign == "negative":
        if metrics is None or metrics.annual_savings_nis >= 0:
            savings_sign_match = False
            val = metrics.annual_savings_nis if metrics else None
            failures.append(f"Savings sign mismatch: expected < 0 (extra cost), got {val}")
    elif case.expected_savings_sign == "none":
        # במקרי חסימה, הבהרה או הסבר סיכון גרידא, אין להחזיר אובייקט SavingsMetrics
        if metrics is not None:
            savings_sign_match = False
            failures.append("Unexpected SavingsMetrics returned for non-savings/blocked case")

    # 6. בדיקת אילוצי טקסט חובה (must_contain) וביטויים אסורים (must_not_contain)
    content_match = True
    for req in case.must_contain:
        if req not in reply_text:
            content_match = False
            failures.append(f"Missing required text '{req}' in reply")
    for banned in case.must_not_contain:
        if banned in reply_text:
            content_match = False
            failures.append(f"Forbidden text '{banned}' appeared in reply")

    # מקרה עובר רק אם כל 6 צירי הבדיקה עברו בהצלחה (אין הנחות)
    passed = all([schema_valid, status_match, intent_match, entity_match, savings_sign_match, content_match])

    return CaseEvalResult(
        id=case.id,
        category=case.category,
        description=case.description,
        passed=passed,
        latency_ms=latency_ms,
        schema_valid=schema_valid,
        status_match=status_match,
        intent_match=intent_match,
        entity_match=entity_match,
        savings_sign_match=savings_sign_match,
        content_match=content_match,
        actual_status=actual_status,
        actual_intent=actual_intent,
        failure_reasons=failures,
    )


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """מחשב אחוזון (כמו p50 או p95) מתוך רשימת זמנים ממוינת."""
    if not sorted_vals:
        return 0.0
    # חישוב אינדקס מדויק עם גבול עליון תקין למניעת IndexError
    idx = min(len(sorted_vals) - 1, max(0, int(round((pct / 100.0) * (len(sorted_vals) - 1)))))
    return round(sorted_vals[idx], 2)


# ==============================================================================
# שלב 4: בדיקת אינווריאנטים של מצב Zero-Click (`run_auto_compare_agent`)
# ==============================================================================
def evaluate_auto_compare_suite() -> AutoCompareEvalSummary:
    """
    מריץ את סוכן ההשוואה האוטונומי ברקע על מטריצת בדיקה של דגמים וערים נפוצות
    ובודק כללים מתמטיים ומוצריים קשיחים (Invariants) בכל תוצאה:
    - החזרת 3 שלבי ביצוע (agent_steps)
    - קיום דיסקליימר משפטי
    - כל חלופה זולה יותר (או שווה) ובעלת סיכון נמוך/שווה לרכב הנבחר
    - אפס אזכורים של חברות ביטוח שאינן שותפות ב-Bestie.
    """
    # מטריצת שילובי מבחן: דגמים פופולריים, SUV, חשמליים, ערים גדולות וערי שדה
    test_pairs = [
        ("טויוטה קורולה", "תל אביב-יפו"),
        ("יונדאי טוסון", "פתח תקווה"),
        ("קיה פיקנטו", "חיפה"),
        ("טויוטה ראב 4", "באר שבע"),
        ("יונדאי איוניק", "ראשון לציון"),
        ("קיה ספורטאז'", "נתניה"),
        ("סקודה אוקטביה", "ירושלים"),
        ("מאזדה 3", "אשדוד"),
    ]

    latencies: list[float] = []
    failures: list[str] = []
    passed_count = 0

    for model_name, city_name in test_pairs:
        t0 = time.perf_counter()
        res = run_auto_compare_agent(model_name, city_name)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        pair_label = f"{model_name} | {city_name}"
        pair_ok = True

        # בדיקת מבנה בסיסי: סטטוס תקין ושרשרת 3 כלים הושלמה
        if res.get("status") != "ok" or len(res.get("agent_steps", [])) != 3:
            failures.append(f"{pair_label}: missing status='ok' or 3 agent_steps")
            pair_ok = False

        # בדיקת נוכחות דיסקליימר והיעדר חברות אסורות בסיכום ובכרטיסים
        sel_card = res.get("selected_vehicle_card", {})
        alt_cards = res.get("alternative_cards", [])
        text_blob = json.dumps(res, ensure_ascii=False)
        for forbidden in FORBIDDEN_INSURERS:
            if forbidden in text_blob:
                failures.append(f"{pair_label}: forbidden insurer '{forbidden}' found")
                pair_ok = False

        # בדיקת עקביות חשבונאית של כרטיס הרכב הנבחר (מקיף + חובה = סה"כ)
        if sel_card.get("total_annual_nis") != sel_card.get("comprehensive_nis", 0) + sel_card.get("mandatory_nis", 0):
            failures.append(f"{pair_label}: selected card math mismatch (comp + mand != total)")
            pair_ok = False

        # בדיקה שהוצעו חלופות עם חיסכון אי-שלילי וסיכון שאינו גבוה מהרכב הנבחר
        if not alt_cards:
            failures.append(f"{pair_label}: returned 0 alternatives")
            pair_ok = False
        else:
            for alt in alt_cards:
                if alt.get("annual_savings_nis", -1) < 0:
                    failures.append(f"{pair_label}: negative savings on alternative {alt.get('model_hebrew')}")
                    pair_ok = False
                if alt.get("risk_score", 999) > sel_card.get("risk_score", 0):
                    failures.append(f"{pair_label}: alternative has higher risk than selected")
                    pair_ok = False

        if pair_ok:
            passed_count += 1

    total = len(test_pairs)
    return AutoCompareEvalSummary(
        total_pairs=total,
        passed_pairs=passed_count,
        pass_rate=round(passed_count / total, 4) if total else 0.0,
        mean_latency_ms=round(statistics.mean(latencies), 2) if latencies else 0.0,
        failures=failures,
    )


# ==============================================================================
# שלב 5: מבחן עמידות לשכבה 4 — Numeric Verifier Stress Test
# ==============================================================================
def run_verifier_stress_suite() -> dict[str, Any]:
    """
    בודק האם שכבה 4 (`verify_numeric_grounding`) יודעת להבחין בצורה מושלמת בין:
    - ניסוח LLM אמין שמשתמש רק במספרי הכלים (צריך לעבור כ-Valid)
    - ניסוח LLM הוזה שהמציא מחיר, שינה ספרה ב-5 ש"ח, המציא אחוז או הוסיף סכום משוער (צריך להיחסם!).
    """
    # בניית קבוצת מספרים מותרת מתוך תוצאת כלי אמיתית (מעבר מקורולה לקיה נירו)
    det_metrics = {
        "base_annual_nis": 7110,
        "target_annual_nis": 6400,
        "annual_savings_nis": 710,
        "monthly_savings_nis": 59,
        "base_risk_score": 84,
        "target_risk_score": 52,
        "risk_delta_pts": 32,
    }
    det_reply = (
        "מעבר מטויוטה קורולה לקיה נירו בתל אביב חוסך ₪710 בשנה (כ-₪59/חודש). "
        "העלות יורדת מ-₪7,110 ל-₪6,400 והסיכון יורד ב-32 נק׳ מ-84 ל-52/100."
    )
    allowed = collect_allowed_numbers(det_metrics, det_reply)

    # 10 מקרי מבחן: 4 תקינים (Negatives להזיה) + 6 הזיות (Positives שצריך לתפוס)
    stress_cases = [
        ("VS-01", "ניסוח אמין עם פסיקים וסימני שקלים מדויקים", True,
         "לפרופיל הייחוס (בן 35, 12 שנות ותק), קיה נירו בתל אביב תעלה ₪6,400 בשנה במקום ₪7,110 — חיסכון של ₪710 בשנה (₪59 בחודש)."),
        ("VS-02", "ניסוח אמין עם אזכור ציון סיכון והפרש נקודות", True,
         "ציון הסיכון צונח מ-84 ל-52 מתוך 100 (ירידה של 32 נקודות)."),
        ("VS-03", "ניסוח אמין ללא מספרים כלל (מלל איכותני בלבד)", True,
         "מעבר מקורולה לקיה נירו צפוי להוזיל את עלות הביטוח ולהפחית משמעותית את סיכון הגניבה."),
        ("VS-04", "ניסוח אמין המשתמש בקבוע שנתון מותר (2025)", True,
         "הסימולציה עבור שנתון 2025 מציגה עלות שנתית של ₪6,400 לעומת ₪7,110 בקורולה."),
        ("VS-05", "הזיה גסה: המצאת סכום ענק (₪99,999)", False,
         "תחסוך בדיוק ₪99,999 בשנה אם תחליף מחר לקיה נירו!"),
        ("VS-06", "הזיה עדינה (Off-by-5): עיגול אסור של החיסכון ל-₪715 במקום ₪710", False,
         "המעבר יחסוך לך כ-₪715 בשנה (₪59 לחודש)."),
        ("VS-07", "הזיית אחוזים: המצאת אחוז הנחה שלא הופק מהכלי (18%)", False,
         "קיה נירו מוזילה את הפוליסה ב-18% בשנה וחוסכת ₪710."),
        ("VS-08", "הזיית ציון סיכון משובש: 53 במקום 52", False,
         "ציון הסיכון החדש של קיה נירו הוא 53/100 (לעומת 84 בקורולה)."),
        ("VS-09", "הזיית עלות התקנת מיגון בדויה (₪1,250)", False,
         "בנוסף תוכל להתקין מערכת מיגון בעלות של ₪1,250 כדי לחסוך ₪710."),
        ("VS-10", "הזיית שנתון או גיל שגוי (בן 40 במקום 35)", False,
         "עבור נהג בן 40, המחיר לקיה נירו עומד על ₪6,400 לשנה."),
    ]

    results: list[NumericVerifierStressResult] = []
    hallucination_total = 0
    hallucination_caught = 0
    valid_total = 0
    false_alarms = 0

    for cid, desc, exp_valid, text in stress_cases:
        # קריאה ישירה לפונקציית השכבה הרביעית בקוד הסוכן
        actual_valid, bad_tokens = verify_numeric_grounding(text, allowed)
        passed = actual_valid == exp_valid

        if not exp_valid:
            hallucination_total += 1
            if not actual_valid:
                hallucination_caught += 1
        else:
            valid_total += 1
            if not actual_valid:
                false_alarms += 1

        results.append(
            NumericVerifierStressResult(
                case_id=cid,
                description=desc,
                expected_valid=exp_valid,
                actual_valid=actual_valid,
                passed=passed,
                flagged_tokens=bad_tokens,
            )
        )

    recall = round(hallucination_caught / hallucination_total, 4) if hallucination_total else 0.0
    far = round(false_alarms / valid_total, 4) if valid_total else 0.0
    total_passed = sum(1 for r in results if r.passed)

    return {
        "total_cases": len(results),
        "passed_cases": total_passed,
        "hallucination_detection_recall": recall,
        "false_alarm_rate": far,
        "results": [r.model_dump() for r in results],
    }


# ==============================================================================
# שלב 6: הערכה חיה מול Gemini + שופט אוטומטי (OpenAI / Gemini LLM-as-a-Judge)
# ==============================================================================
def _build_judge_prompt(case: GoldenCase, res: dict[str, Any]) -> str:
    """
    בונה פרומפט אחיד ומוגדר היטב עבור השופט האוטומטי (LLM-as-a-Judge).
    מזין לשופט את שאלת המשתמש, את נתוני האמת מהכלים ואת התשובה שהופקה בפועל,
    ומבקש דירוג בסולם 1 עד 5 בשלושה ממדים: נאמנות, עקביות מספרית ובהירות בעברית.
    """
    return (
        "אתה מעריך מקצועי (LLM-as-a-Judge) של סוכן ביטוח רכב בישראל.\n"
        "דרג כל ממד במספר שלם מ-1 (גרוע) עד 5 (מצוין) בהתאם להנחיות:\n"
        "1. faithfulness (1-5): האם התשובה נאמנה לעובדות, לא ממציאה נתונים ולא מבטיחה הצעת ביטוח מחייבת?\n"
        "2. numeric_consistency (1-5): האם כל המספרים בטקסט תואמים במדויק את אובייקט Metrics מהכלים (או אם אין Metrics — שאין המצאת מחיר)?\n"
        "3. clarity_he (1-5): האם הניסוח בעברית ברור, שוטף, מקצועי וישים למשתמש?\n"
        "החזר אובייקט JSON תקין בלבד עם השדות: faithfulness, numeric_consistency, clarity_he, explanation (הסבר קצר בעברית).\n\n"
        f"שאלת המשתמש: {case.message}\n"
        f"סטטוס תשובת הסוכן: {res.get('status')} | מנוע הפקה: {res.get('engine')}\n"
        f"נתוני אמת מובנים (Metrics): {json.dumps(res.get('metrics'), ensure_ascii=False)}\n"
        f"הערת כיסוי נתונים: {res.get('data_coverage_note', '')}\n"
        f"תשובת הסוכן שנבדקת:\n{res.get('reply')}"
    )


def _call_openai_judge(prompt: str, api_key: str, model_name: str = "gpt-4o-mini") -> JudgeScore:
    """
    שולח את תשובת הסוכן לשופט החיצוני של OpenAI (`chat/completions`) דרך `httpx`
    תוך אכיפת Structured Outputs (`json_schema`, `strict=True`) ללא תלות בספרייה חיצונית נוספת.
    """
    # ייבוא httpx מקומי (חבילה קיימת בדרישות הפרויקט עבור FastAPI ו-TestClient)
    import httpx

    # הגדרת סכמת JSON קשיחה התואמת לשדות של JudgeScore
    response_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "agent_judge_score",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "faithfulness": {"type": "integer"},
                    "numeric_consistency": {"type": "integer"},
                    "clarity_he": {"type": "integer"},
                    "explanation": {"type": "string"},
                },
                "required": ["faithfulness", "numeric_consistency", "clarity_he", "explanation"],
                "additionalProperties": False,
            },
        },
    }

    # בניית גוף בקשת HTTP ל-API של OpenAI עם temperature=0 להערכה עקבית
    payload = {
        "model": model_name,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "You are a strict bilingual AI Agent evaluator outputting valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "response_format": response_schema,
    }

    # שליחת הבקשה עם Timeout של 30 שניות והעלאת חריגה אם הסטטוס אינו 2xx
    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30.0,
    )
    resp.raise_for_status()

    # חילוץ תוכן ההודעה מתוך תשובת OpenAI ואימות סופי ב-Pydantic (מוודא טווח 1–5)
    raw_content = resp.json()["choices"][0]["message"]["content"]
    return JudgeScore.model_validate_json(raw_content)


def _call_gemini_judge(prompt: str, api_key: str, model_name: str = "gemini-2.5-flash") -> JudgeScore:
    """
    שולח את תשובת הסוכן לשופט של Gemini באמצעות `google-genai` (גיבוי כשאין מפתח OpenAI).
    """
    from google import genai  # type: ignore

    client = genai.Client(api_key=api_key)
    j_resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": JudgeScore,
            "temperature": 0.0,
        },
    )
    return JudgeScore.model_validate_json(j_resp.text)


def run_live_gemini_and_judge_eval(
    cases: list[GoldenCase],
    sample_ids: tuple[str, ...] = ("MC-01", "MC-02", "CC-01", "MT-01", "AR-01", "EG-01"),
    judge_provider: Literal["auto", "openai", "gemini"] = "auto",
) -> LiveLLMEvalSummary:
    """
    מריץ הערכה חיה דו-שלבית:
    1. שלב הסוכן: מריץ את `run_chat_agent` עם `GEMINI_API_KEY` (אם זמין) ובודק האם Gemini עבר
       את שכבה 4 (`gemini_verified`) או הפעיל Graceful Fallback לתשובה דטרמיניסטית.
    2. שלב השופט (`LLM-as-a-Judge`): מדרג כל תשובה שהוחזרה למשתמש בסולם 1–5
       (`faithfulness`, `numeric_consistency`, `clarity_he`) דרך:
       - OpenAI (`gpt-4o-mini` כברירת מחדל) כאשר מוגדר `OPENAI_API_KEY` (עדיף למניעת הטיה עצמית ועוקף בעיות קרדיט ב-Gemini),
       - או דרך Gemini (`gemini-2.5-flash`) כשאין מפתח OpenAI.
    """
    # טעינה אוטומטית של המפתחות מקובץ .env של הפרויקט (רק מחוץ ל-pytest כדי לשמור על בידוד בדיקות)
    if "PYTEST_CURRENT_TEST" not in os.environ:
        load_env_file()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()

    # אם אין אף מפתח API של ספק חיצוני, מחזירים סטטוס דילוג מפורש ללא כישלון
    if not gemini_key and not openai_key:
        return LiveLLMEvalSummary(
            status="skipped_no_api_key",
            reason="לא נמצאו GEMINI_API_KEY או OPENAI_API_KEY בסביבה או בקובץ .env.",
        )

    # בחירת ספק השופט: במצב 'auto' נעדיף OpenAI אם מוגדר מפתח (למניעת הטיה עצמית וכי OpenAI זמין), אחרת Gemini
    selected_judge_provider = ""
    if judge_provider == "openai" and openai_key:
        selected_judge_provider = "openai"
    elif judge_provider == "gemini" and gemini_key:
        selected_judge_provider = "gemini"
    elif openai_key:
        selected_judge_provider = "openai"
    elif gemini_key:
        selected_judge_provider = "gemini"

    # בחירת מודל הסוכן: ברירת המחדל היא OpenAI כשיש OPENAI_API_KEY (ללא פנייה ל-Gemini כשנגמרו שם הקרדיטים)
    openai_agent_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    gemini_agent_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    provider_pref = os.environ.get("LLM_PROVIDER", "auto").strip().lower()
    use_openai_agent = bool(openai_key) and provider_pref != "gemini"
    agent_provider_label = "openai" if use_openai_agent else "gemini"
    agent_model = openai_agent_model if use_openai_agent else gemini_agent_model

    openai_judge_model = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    judge_model_name = openai_judge_model if selected_judge_provider == "openai" else gemini_agent_model

    # סינון תרחישי המדגם לריצה החיה מתוך ה-Golden Dataset
    sample_cases = [c for c in cases if c.id in sample_ids]

    verified_cnt = 0
    fallback_cnt = 0
    latencies: list[float] = []
    faith_scores: list[int] = []
    num_consistency_scores: list[int] = []
    clarity_scores: list[int] = []
    details: list[dict[str, Any]] = []

    for case in sample_cases:
        # הרצת הסוכן האמיתי (ברירת מחדל OpenAI כש-OPENAI_API_KEY מוגדר, כולל Layer 4 Numeric Verifier)
        t0 = time.perf_counter()
        res = run_chat_agent(case.message, case.base_model, case.base_city, history=case.history or None)
        lat_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        latencies.append(lat_ms)

        engine = res.get("engine", "")
        guards = res.get("guardrails_passed", [])
        if engine in ("openai_verified", "gemini_verified"):
            verified_cnt += 1
        if "numeric_verifier_fallback_triggered" in guards:
            fallback_cnt += 1

        # בניית פרומפט השופט ושליחה לספק השופט שנבחר (עם גיבוי הדדי במקרה תקלה)
        judge_prompt = _build_judge_prompt(case, res)
        judge_out: dict[str, Any] = {}
        try:
            if selected_judge_provider == "openai":
                j_parsed = _call_openai_judge(judge_prompt, openai_key, openai_judge_model)
            else:
                try:
                    j_parsed = _call_gemini_judge(judge_prompt, gemini_key, gemini_agent_model)
                except Exception:
                    if openai_key:
                        selected_judge_provider = "openai"
                        judge_model_name = openai_judge_model
                        j_parsed = _call_openai_judge(judge_prompt, openai_key, openai_judge_model)
                    else:
                        raise

            faith_scores.append(j_parsed.faithfulness)
            num_consistency_scores.append(j_parsed.numeric_consistency)
            clarity_scores.append(j_parsed.clarity_he)
            judge_out = j_parsed.model_dump()
        except Exception as exc:
            judge_out = {"error": str(exc)}

        details.append(
            {
                "id": case.id,
                "engine": engine,
                "latency_ms": lat_ms,
                "guardrails_passed": guards,
                "reply_preview": str(res.get("reply", ""))[:220],
                "judge": judge_out,
            }
        )

    n = len(sample_cases)
    evaluated_cnt = len(faith_scores)
    judge_errors = [d["judge"]["error"] for d in details if isinstance(d.get("judge"), dict) and "error" in d["judge"]]

    # אם כל קריאות השופט נכשלו ואין אף ציון שופט אחד — נחזיר status='error' עם הודעה ברורה
    if n > 0 and evaluated_cnt == 0 and judge_errors:
        first_err = judge_errors[0][:220]
        return LiveLLMEvalSummary(
            status="error",
            reason=f"קריאות השופט החי נכשלו ({first_err}).",
            model_used=f"{agent_provider_label}/{agent_model}",
            judge_provider=selected_judge_provider,
            judge_model=judge_model_name,
            sample_size=n,
            judge_evaluated_count=0,
            gemini_verified_count=verified_cnt,
            verifier_fallback_count=fallback_cnt,
            gemini_verified_rate=round(verified_cnt / n, 4),
            mean_live_latency_ms=round(statistics.mean(latencies), 2) if latencies else None,
            details=details,
        )

    # ניסוח תמצית אמינה המתארת גם את התנהגות הסוכן החי וגם את תוצאות השופט
    if agent_provider_label == "openai":
        reason = (
            f"הערכה חיה מלאה הושלמה על OpenAI (Agent={agent_model}, Judge={judge_model_name}): "
            f"{verified_cnt}/{n} תשובות LLM אומתו ב-Layer 4 ({fallback_cnt} Fallbacks), "
            f"והשופט דירג {evaluated_cnt}/{n} מקרים."
        )
    elif gemini_key and verified_cnt == 0:
        reason = (
            f"הערכת LLM-as-a-Judge חיה הושלמה דרך {selected_judge_provider} ({judge_model_name}) על {evaluated_cnt}/{n} מקרים; "
            f"בצד הסוכן, Gemini ({agent_model}) לא אישר ניסוח והסוכן הפעיל Graceful Fallback דטרמיניסטי ב-100% מהמקרים."
        )
    else:
        reason = f"הערכה חיה מלאה הושלמה (Agent={agent_provider_label}/{agent_model}, Judge={selected_judge_provider}/{judge_model_name}) על {evaluated_cnt}/{n} מקרים."

    return LiveLLMEvalSummary(
        status="executed",
        reason=reason,
        model_used=f"{agent_provider_label}/{agent_model}",
        judge_provider=selected_judge_provider,
        judge_model=judge_model_name,
        sample_size=n,
        judge_evaluated_count=evaluated_cnt,
        gemini_verified_count=verified_cnt,
        verifier_fallback_count=fallback_cnt,
        gemini_verified_rate=round(verified_cnt / n, 4) if n else None,
        mean_live_latency_ms=round(statistics.mean(latencies), 2) if latencies else None,
        mean_judge_faithfulness=round(statistics.mean(faith_scores), 2) if faith_scores else None,
        mean_judge_numeric_consistency=round(statistics.mean(num_consistency_scores), 2) if num_consistency_scores else None,
        mean_judge_clarity_he=round(statistics.mean(clarity_scores), 2) if clarity_scores else None,
        details=details,
    )


# ==============================================================================
# שלב 7: פונקציית האורקסטרציה המרכזית (`run_full_evaluation`) וכניסה ראשית
# ==============================================================================
def run_full_evaluation(
    dataset_path: Path = GOLDEN_DATASET_PATH,
    output_path: Path = EVAL_RESULTS_PATH,
    run_live: bool = True,
) -> AgentEvalReport:
    """
    מריץ את כל 4 הרבדים של ה-Eval, מחשב סיכומים לפי קטגוריה ושומר דוח JSON מלא לדיסק.
    """
    version, cases = load_golden_dataset(dataset_path)

    # 1. הרצת כל מקרי ה-Golden Dataset על המנוע הדטרמיניסטי המאומת
    case_results = [evaluate_golden_case(case, force_offline=True) for case in cases]
    total = len(case_results)
    passed_cases = sum(1 for r in case_results if r.passed)
    failed_results = [r for r in case_results if not r.passed]

    # חישוב מדדי דיוק רוחביים לכל ציר בנפרד
    latencies_sorted = sorted(r.latency_ms for r in case_results)
    by_category: dict[str, dict[str, Any]] = {}
    for r in case_results:
        bucket = by_category.setdefault(r.category, {"total": 0, "passed": 0, "mean_latency_ms": 0.0, "_lats": []})
        bucket["total"] += 1
        if r.passed:
            bucket["passed"] += 1
        bucket["_lats"].append(r.latency_ms)

    # חישוב אחוז הצלחה וזמן ממוצע לכל קטגוריה (וניקוי רשימת העזר)
    for cat, stats in by_category.items():
        stats["pass_rate"] = round(stats["passed"] / stats["total"], 4) if stats["total"] else 0.0
        stats["mean_latency_ms"] = round(statistics.mean(stats["_lats"]), 2) if stats["_lats"] else 0.0
        del stats["_lats"]

    # 2. הרצת בדיקת מצב Auto-Compare
    auto_summary = evaluate_auto_compare_suite()

    # 3. הרצת מבחן העמידות לשכבה 4 (Numeric Verifier)
    verifier_stress = run_verifier_stress_suite()

    # 4. הרצת Live Gemini + LLM-as-a-Judge אם התבקש וזמין בסביבה
    live_summary = (
        run_live_gemini_and_judge_eval(cases)
        if run_live
        else LiveLLMEvalSummary(status="skipped_no_api_key", reason="הרצה חיה כובתה ידנית (run_live=False).")
    )

    report = AgentEvalReport(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        dataset_version=version,
        total_cases=total,
        passed_cases=passed_cases,
        failed_cases=len(failed_results),
        overall_pass_rate=round(passed_cases / total, 4) if total else 0.0,
        schema_valid_rate=round(sum(1 for r in case_results if r.schema_valid) / total, 4) if total else 0.0,
        status_accuracy=round(sum(1 for r in case_results if r.status_match) / total, 4) if total else 0.0,
        intent_accuracy=round(sum(1 for r in case_results if r.intent_match) / total, 4) if total else 0.0,
        entity_accuracy=round(sum(1 for r in case_results if r.entity_match) / total, 4) if total else 0.0,
        savings_sign_accuracy=round(sum(1 for r in case_results if r.savings_sign_match) / total, 4) if total else 0.0,
        latency_mean_ms=round(statistics.mean(latencies_sorted), 2) if latencies_sorted else 0.0,
        latency_p50_ms=_percentile(latencies_sorted, 50.0),
        latency_p95_ms=_percentile(latencies_sorted, 95.0),
        by_category=by_category,
        verifier_stress=verifier_stress,
        auto_compare_invariants=auto_summary,
        live_llm_eval=live_summary,
        failures_detail=failed_results,
        all_case_results=case_results,
    )

    # יצירת תיקיית הפלט אם אינה קיימת ושמירת דוח ההערכה המלא
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def _print_console_summary(report: AgentEvalReport) -> None:
    """מדפיס טבלת סיכום קריאה של מדדי ה-Eval למסוף."""
    print("=" * 72)
    print(f"CarTheftRisk AI Agent Evaluation Report (Dataset v{report.dataset_version})")
    print("=" * 72)
    print(
        f"Golden Dataset : {report.passed_cases}/{report.total_cases} passed "
        f"({report.overall_pass_rate * 100:.1f}%)"
    )
    print(
        f"Axis Accuracy  : Schema={report.schema_valid_rate * 100:.1f}% | "
        f"Status={report.status_accuracy * 100:.1f}% | "
        f"Intent={report.intent_accuracy * 100:.1f}% | "
        f"Entity={report.entity_accuracy * 100:.1f}% | "
        f"Sign={report.savings_sign_accuracy * 100:.1f}%"
    )
    print(
        f"Latency (ms)   : Mean={report.latency_mean_ms}ms | "
        f"p50={report.latency_p50_ms}ms | p95={report.latency_p95_ms}ms"
    )
    print("-" * 72)
    print("By Category:")
    for cat, st in report.by_category.items():
        print(f"  - {cat:28s}: {st['passed']:2d}/{st['total']:2d} ({st['pass_rate'] * 100:5.1f}%) | {st['mean_latency_ms']} ms")
    print("-" * 72)
    vs = report.verifier_stress
    print(
        f"Layer 4 Numeric Verifier Stress: {vs['passed_cases']}/{vs['total_cases']} passed | "
        f"Hallucination Recall={vs['hallucination_detection_recall'] * 100:.1f}% | "
        f"False Alarm Rate={vs['false_alarm_rate'] * 100:.1f}%"
    )
    ac = report.auto_compare_invariants
    print(
        f"Auto-Compare Invariants        : {ac.passed_pairs}/{ac.total_pairs} ({ac.pass_rate * 100:.1f}%) | "
        f"Mean={ac.mean_latency_ms}ms"
    )
    ll = report.live_llm_eval
    print(f"Live Agent + LLM-as-a-Judge    : {ll.status} ({ll.reason})")
    if ll.status == "executed":
        ver_pct = f"{ll.gemini_verified_rate * 100:.1f}%" if ll.gemini_verified_rate is not None else "N/A"
        print(
            f"  Agent={ll.model_used} | Judge={ll.judge_provider}/{ll.judge_model} | "
            f"Judged={ll.judge_evaluated_count}/{ll.sample_size} | "
            f"LLM Verified={ll.gemini_verified_count}/{ll.sample_size} ({ver_pct}) | "
            f"Fallbacks={ll.verifier_fallback_count} | Mean Latency={ll.mean_live_latency_ms}ms"
        )
        print(
            f"  Judge Scores (1-5): Faithfulness={ll.mean_judge_faithfulness} | "
            f"Numeric Consistency={ll.mean_judge_numeric_consistency} | "
            f"Clarity (HE)={ll.mean_judge_clarity_he}"
        )
    if report.failures_detail:
        print("-" * 72)
        print(f"Detected Failures ({len(report.failures_detail)}) - Real Agent Edge Cases to Address:")
        for fail in report.failures_detail:
            print(f"  [FAIL] {fail.id} ({fail.category}): {fail.description}")
            for reason in fail.failure_reasons:
                print(f"         -> {reason}")
    print("=" * 72)
    print(f"Saved JSON report to: {EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    # נקודת הכניסה בעת הרצה ישירה מהטרמינל (python -X utf8 -m src.agent_eval)
    final_report = run_full_evaluation()
    _print_console_summary(final_report)
