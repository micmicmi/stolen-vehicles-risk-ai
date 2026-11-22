"""
agent_schemas.py
================
Validated Pydantic v2 input/output contracts for the CarTheftRisk AI Agent.
Ensures structured intent planning, strict entity grounding, and schema-verified responses.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class IntentEnum(str, Enum):
    compare_model_savings = "compare_model_savings"
    compare_city_savings = "compare_city_savings"
    mitigation_savings = "mitigation_savings"
    risk_explanation = "risk_explanation"
    alternatives_list = "alternatives_list"
    out_of_scope = "out_of_scope"
    prompt_injection = "prompt_injection"


ALLOWED_MITIGATION_MEASURES: tuple[str, ...] = (
    "איתוראן פעיל כפול",
    "קודן מוגן + עלוקה",
    "מנעול הגה מכני",
    "חניה פרטית סגורה / מצלמות",
)


class QueryPlan(BaseModel):
    """Structured query plan extracted before running deterministic pricing/risk tools."""

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
    """Typed, deterministically computed risk and Bestie insurance cost comparison."""

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
    """Schema-verified response returned by the AI agent and `/agent/chat` endpoint."""

    status: Literal["answered", "needs_clarification", "blocked"]
    engine: Literal["gemini_verified", "deterministic_grounded", "guardrail_filter"]
    intent: IntentEnum
    reply: str
    metrics: SavingsMetrics | None = None
    assumptions: dict[str, Any] = Field(default_factory=dict)
    data_coverage_note: str = ""
    tools_used: list[str] = Field(default_factory=list)
    guardrails_passed: list[str] = Field(default_factory=list)
    disclaimer: str


class GeminiReplyPayload(BaseModel):
    """Structured JSON schema enforced when Gemini formats natural Hebrew phrasing."""

    reply: str = Field(..., min_length=1, description="Grounded Hebrew answer using only verified numbers.")
