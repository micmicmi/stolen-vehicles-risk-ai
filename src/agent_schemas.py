"""
agent_schemas.py
================
Validated Pydantic v2 input/output contracts for the CarTheftRisk AI Agent.
Ensures structured intent planning, strict entity grounding, and schema-verified responses.

מודול סכמות הנתונים של סוכן ה-AI (מבוסס Pydantic v2).
תפקידו:
1. הגדרת חוזים קשיחים ומאומתים (Type Contracts) לקלטים ולפלטים של הסוכן.
2. תכנון מובנה של שאילתות (Query Planning) וסיווג כוונות המשתמש (Intent Classification).
3. מניעת הזיות (Hallucination Prevention) על ידי וידוא שכל נתון שמוחזר למשתמש עומד במבנה מוגדר ומאומת.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class IntentEnum(str, Enum):
    """
    סיווג כוונת המשתמש (User Intent Classification).
    מאפשר לסוכן להבין מה המשתמש שאל ולהפנות אותו לכלי החישוב הדטרמיניסטי המתאים:
    - compare_model_savings: השוואת עלויות וסיכון בין דגם נוכחי לדגם יעד מבוקש.
    - compare_city_savings: השוואת עלויות וסיכון במעבר מעיר נוכחית לעיר אחרת.
    - mitigation_savings: בדיקת השפעת התקנת אמצעי מיגון (קודן, איתוראן וכו') על עלות הביטוח.
    - risk_explanation: הסבר מפורט על גורמי הסיכון של הרכב/העיר (שיעור פענוח, היסטוריית גניבות).
    - alternatives_list: בקשה להמלצה על דגמים חלופיים בטוחים וזולים יותר באותה קטגוריה.
    - out_of_scope: שאלות שאינן קשורות לתחום (מתכונים, מזג אוויר, בדיחות).
    - prompt_injection: ניסיונות עקיפת הנחיות מערכת או הזרקת פקודות זדוניות.
    """
    compare_model_savings = "compare_model_savings"
    compare_city_savings = "compare_city_savings"
    mitigation_savings = "mitigation_savings"
    risk_explanation = "risk_explanation"
    alternatives_list = "alternatives_list"
    out_of_scope = "out_of_scope"
    prompt_injection = "prompt_injection"


# רשימת אמצעי המיגון המורשים והמוכרים בקטלוג המערכת (למניעת המצאת אמצעי מיגון פיקטיביים)
ALLOWED_MITIGATION_MEASURES: tuple[str, ...] = (
    "איתוראן פעיל כפול",
    "קודן מוגן + עלוקה",
    "מנעול הגה מכני",
    "חניה פרטית סגורה / מצלמות",
)


class QueryPlan(BaseModel):
    """
    תוכנית שאילתה מובנית (Structured Query Plan).
    מופקת בשכבה 1 (Layer 1) של הסוכן לפני הפעלת כלי תמחור או חישוב סיכונים.
    מפרקת את שאלת המשתמש החופשית לישויות מוגדרות:
    - intent: כוונת המשתמש המזוהה.
    - base_model_override / base_city_override: דגם/עיר בסיס חלופיים שהמשתמש ציין במפורש בשאילתה.
    - target_model / target_city: דגם היעד או עיר היעד להשוואה.
    - mitigation_measures: רשימת אמצעי מיגון שחולצו מטקסט המשתמש.
    - needs_clarification / clarification_question_he: דגלים לשאלת הבהרה במקרה של ישות לא מוכרת.
    """

    intent: IntentEnum
    base_model_override: str | None = None
    base_city_override: str | None = None
    target_model: str | None = None
    target_city: str | None = None
    mitigation_measures: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question_he: str | None = None

    @field_validator("mitigation_measures", mode="before")
    @classmethod
    def _filter_allowed_measures(cls, value: Any) -> list[str]:
        """
        ולידטור שמוודא שרק אמצעי מיגון תקניים מקטלוג המערכת ייכנסו לרשימה,
        מסיר כפילויות ומנקה רווחים מיותרים.
        """
        if not value:
            return []
        allowed = set(ALLOWED_MITIGATION_MEASURES)
        filtered: list[str] = []
        for item in value:
            s = str(item).strip()
            if s in allowed and s not in filtered:
                filtered.append(s)
        return filtered


class SavingsMetrics(BaseModel):
    """
    מדדי חיסכון והשוואת עלויות ביטוח Bestie (Typed Savings Metrics).
    מחושב בצורה דטרמיניסטית במלואה (ללא שימוש ב-LLM), ומבטיח שהמספרים מדויקים ועקביים:
    - base_model_he, base_city: דגם ועיר המוצא.
    - target_model_he, target_city: דגם ועיר היעד (אופציונלי).
    - base_annual_nis, target_annual_nis: עלות שנתית בסיסית ומיועדת (בש"ח).
    - annual_savings_nis, monthly_savings_nis: גובה החיסכון השנתי והחודשי (בש"ח).
    - base_risk_score, target_risk_score: ציון סיכון בסיס ויעד (סולם 0-100).
    - risk_delta_pts: הפרש נקודות הסיכון (חיובי = ירידה בסיכון, שלילי = עלייה).
    """

    base_model_he: str
    base_city: str
    target_model_he: str | None = None
    target_city: str | None = None
    base_annual_nis: int
    target_annual_nis: int
    annual_savings_nis: int
    monthly_savings_nis: int
    base_risk_score: int
    target_risk_score: int
    risk_delta_pts: int


class GroundedAgentResponse(BaseModel):
    """
    חוזה התשובה הרשמי והמאומת של סוכן ה-AI (מוחזר גם ב-API דרך `/agent/chat`).
    כולל שקיפות מלאה לגבי אופן הפקת התשובה:
    - status: 'answered' (נענה בהצלחה), 'needs_clarification' (דרושה הבהרה מהמשתמש), 'blocked' (נחסם ע"י Guardrail).
    - engine: 'openai_verified' (נוסח ע"י OpenAI ואומת מספרית), 'gemini_verified' (נוסח ע"י ג'מיני ואומת מספרית), 'deterministic_grounded' (תשובה מתבנית דטרמיניסטית), 'guardrail_filter' (חסימת אבטחה/תחום).
    - intent: הכוונה שזוהתה.
    - reply: נוסח התשובה בעברית שיוצג למשתמש.
    - metrics: אובייקט SavingsMetrics עם כל המספרים המאומתים (אם רלוונטי).
    - assumptions: הנחות היסוד ששימשו בחישוב (פרופיל נהג הייחוס).
    - data_coverage_note: הערת שקיפות לגבי מקור הנתונים (האם יש תצפית מאומתת מ-2024 או חישוב cold-start).
    - tools_used: שמות הכלים הדטרמיניסטיים שהופעלו בתהליך.
    - guardrails_passed: רשימת בדיקות האבטחה והאימות שעברו בהצלחה.
    - disclaimer: דיסקליימר משפטי חובה (אינו ייעוץ ביטוחי מחייב).
    """

    status: Literal["answered", "needs_clarification", "blocked"]
    engine: Literal["openai_verified", "gemini_verified", "deterministic_grounded", "guardrail_filter"]
    intent: IntentEnum
    reply: str
    metrics: SavingsMetrics | None = None
    assumptions: dict[str, Any] = Field(default_factory=dict)
    data_coverage_note: str = ""
    tools_used: list[str] = Field(default_factory=list)
    guardrails_passed: list[str] = Field(default_factory=list)
    disclaimer: str


class GeminiReplyPayload(BaseModel):
    """
    סכמת JSON מובנית שנאכפת על מודל ה-LLM (Gemini) בזמן ניסוח התשובה החופשית.
    מבטיח שה-LLM יחזיר אובייקט JSON תקין עם שדה reply בלבד, מבלי לחרוג מפורמט מובנה.
    """

    reply: str = Field(..., min_length=1, description="Grounded Hebrew answer using only verified numbers.")
