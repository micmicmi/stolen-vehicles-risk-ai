"""
agent.py
========
Research AI Agent Orchestrator for CarTheftRisk (Final).
Supports two execution modes:
1. Background Auto-Compare (`run_auto_compare_agent`): Triggered automatically in parallel
   as soon as the user selects a vehicle model and city. Chains the 3 core tools to inspect
   Bestie insurance prices and compare against 3 lower-risk / cheaper-to-insure alternatives.
2. Contextual Inline Chat (`run_chat_agent`): Answers user follow-up questions below the
   Bestie comparison gallery using deterministic tool grounding (Gemini Function Calling
   when GEMINI_API_KEY is available, with deterministic semantic routing fallback).
"""
from __future__ import annotations

import os
from typing import Any

from src.agent_tools import (
    REFERENCE_DRIVER_PROFILE,
    compare_alternatives,
    fetch_bestie_insurance_quotes,
    get_vehicle_risk,
    simulate_mitigation_discount,
)

DISCLAIMER_HE = (
    "הערכה אקטוארית וסטטיסטית לצרכי מחקר ולימוד בלבד ואינה מהווה הצעת ביטוח מחייבת או ייעוץ פנסיוני/ביטוחי."
)

SYSTEM_PROMPT = """אתה CarTheftRisk Copilot, סוכן AI אקטוארי אוטונומי לניתוח סיכוני גניבת רכב והשוואת מחירי ביטוח מאתר Bestie בישראל.
חוקי ברזל (Guardrails):
1. אסור בתכלית האיסור להמציא ציוני סיכון, כמויות גניבה או מחירי ביטוח. עליך להסתמך אך ורק על פלט הכלים (Tools).
2. השוואת מחירי הביטוח ב-Bestie מבוססת על פרופיל נהג ייחוס קבוע (נהג בן 35, 12 שנות ותק, ללא תביעות ב-3 השנים האחרונות, שנתון 2025).
3. ענה תמיד בעברית ברורה, תמציתית ומקצועית, והדגש תובנות מעשיות שחוסכות למשתמש כסף או מפחיתות סיכון.
"""


def run_auto_compare_agent(model: str, city: str) -> dict[str, Any]:
    """
    Runs automatically in the background when the user enters (model, city).
    Executes the multi-step tool chain and returns structured Bestie comparison cards
    plus the live execution trace.
    """
    risk = get_vehicle_risk(model, city)
    comp = compare_alternatives(risk["model_key"], risk["city"], top_k=3)
    sel_card = comp["selected_vehicle"]
    alt_cards = comp["alternatives"]
    max_save = comp["max_annual_savings_nis"]

    best_alt = alt_cards[0] if alt_cards else None

    steps = [
        {
            "step": 1,
            "tool": "get_vehicle_risk",
            "title": "אומדן מחקרי לספירת גניבות (XGBoost Poisson Final)",
            "detail": (
                f"דגם {risk['model_hebrew']} ב{risk['city']}: ציון סיכון משולב {risk['combined_risk_score']}% "
                f"(רמה {risk['risk_level_he']}), חיזוי ~{risk['xgb_predicted_annual_thefts']} גניבות/שנה."
            ),
            "status": "completed",
        },
        {
            "step": 2,
            "tool": "fetch_bestie_insurance_quotes",
            "title": "סימולציית מחיר לפרופיל נהג ייחוס",
            "detail": (
                f"נבנתה סימולציית ייחוס עבור {risk['model_hebrew']}: סה״כ ₪{sel_card['total_annual_nis']:,}/שנה "
                f"(מקיף ₪{sel_card['comprehensive_nis']:,} + חובה ₪{sel_card['mandatory_nis']:,}, חברה מובילה: {sel_card['leading_insurer']})."
            ),
            "status": "completed",
        },
        {
            "step": 3,
            "tool": "compare_alternatives",
            "title": f"איתור חלופות עם ביטוח זול יותר בקטגוריית {risk['category']}",
            "detail": (
                f"נסרקו דגמים מקבילים ב-Bestie; אותרו {len(alt_cards)} רכבים בעלי סיכון גניבה נמוך יותר "
                f"וחיסכון של עד ₪{max_save:,} בשנה."
            ),
            "status": "completed",
        },
    ]

    if best_alt:
        summary = (
            f"בדיקת הסוכן ב-Bestie מראה שעלות הביטוח השנתית (מקיף + חובה) עבור {risk['model_hebrew']} ב{risk['city']} "
            f"עומדת על כ-₪{sel_card['total_annual_nis']:,} ({sel_card['leading_insurer']}, דרישת מיגון: {sel_card['required_protection']}). "
            f"מעבר לרכב מאותה קטגוריה ({risk['category']}) עם סיכון גניבה נמוך יותר כמו {best_alt['model_hebrew']} "
            f"(ציון סיכון {best_alt['risk_score']}%) מוזיל את הביטוח ל-₪{best_alt['total_annual_nis']:,} — חיסכון של ₪{best_alt['annual_savings_nis']:,} בכל שנה."
        )
    else:
        summary = (
            f"עלות הביטוח השנתית ב-Bestie עבור {risk['model_hebrew']} ב{risk['city']} מוערכת ב-₪{sel_card['total_annual_nis']:,}."
        )

    return {
        "status": "ok",
        "engine": "research_bestie_simulation_final",
        "model_hebrew": risk["model_hebrew"],
        "city": risk["city"],
        "category": risk["category"],
        "driver_profile": REFERENCE_DRIVER_PROFILE,
        "agent_steps": steps,
        "agent_summary": summary,
        "selected_vehicle_card": sel_card,
        "alternative_cards": alt_cards,
        "max_annual_savings_nis": max_save,
        "disclaimer": DISCLAIMER_HE,
    }


def _route_and_answer_deterministically(message: str, model: str, city: str) -> dict[str, Any]:
    msg = (message or "").strip()
    msg_lower = msg.lower()

    # Guardrail against prompt injection / off-topic requests
    blocked_terms = ["ignore previous", "system prompt", "התעלם מההוראות", "כתוב שיר", "מתכון", "בדיחה"]
    if any(b in msg_lower for b in blocked_terms):
        return {
            "reply": (
                "אני סוכן AI ייעודי לניתוח סיכוני גניבת רכב והשוואת מחירי ביטוח מ-Bestie בישראל. "
                "אשמח לענות על שאלות בנוגע לדגם שבחרת, דרכים להוזלת הביטוח באמצעות מיגון, או השוואה לרכבים חלופיים."
            ),
            "tools_used": ["guardrail_filter"],
            "data": {},
        }

    # 1. Intent: Mitigation / protection / lowering insurance for current car
    if any(w in msg for w in ["מיגון", "איתוראן", "קודן", "מנעול", "להוזיל", "הנחה", "עלוקה", "חניה"]):
        mit = simulate_mitigation_discount(model, city)
        measures_list = ", ".join(
            f"{m['measure']} (הפחתת סיכון {m['risk_reduction_pct']}%, הנחה {m['comp_discount_pct']}%)"
            for m in mit["applied_measures"]
        )
        reply = (
            f"🛡️ **סימולציית הוזלת ביטוח באמצעות מיגון עבור {mit['model_hebrew']} ב{mit['city']}:**\n\n"
            f"• **אמצעי מיגון שנבדקו:** {measures_list}.\n"
            f"• **השפעה על ציון הסיכון:** ירידה מ-**{mit['original_risk_score']}%** ל-**{mit['mitigated_risk_score']}%**.\n"
            f"• **השפעה על הצעת Bestie:** הוזלה של כ-**₪{mit['annual_discount_nis']:,} בשנה** "
            f"(מ-₪{mit['original_total_nis']:,} ל-**₪{mit['mitigated_total_nis']:,}** לשנה במקיף + חובה, כ-{mit['discount_pct']}% הנחה על המקיף)."
        )
        return {
            "reply": reply,
            "tools_used": ["simulate_mitigation_discount", "fetch_bestie_insurance_quotes"],
            "data": mit,
        }

    # 2. Intent: Alternative vehicles / cheaper insurance comparison
    if any(w in msg for w in ["חלופ", "זול יותר", "רכבים אחרים", "השווא", "להחליף", "דגמים דומים", "בסטי", "bestie"]):
        comp = compare_alternatives(model, city, top_k=3)
        sel = comp["selected_vehicle"]
        lines = [
            f"🔄 **השוואת Bestie לדגמים בטוחים וזולים יותר לביטוח בקטגוריית {comp['category']} ({comp['city']}):**\n",
            f"• **הרכב שבחרת ({sel['model_hebrew']}):** ₪{sel['total_annual_nis']:,}/שנה ב-Bestie ({sel['leading_insurer']}) | סיכון {sel['risk_score']}%.",
        ]
        for idx, alt in enumerate(comp["alternatives"], 1):
            lines.append(
                f"• **חלופה {idx} – {alt['model_hebrew']}:** ₪{alt['total_annual_nis']:,}/שנה ({alt['leading_insurer']}) | "
                f"סיכון {alt['risk_score']}% | **חיסכון של ₪{alt['annual_savings_nis']:,} בשנה**."
            )
        return {
            "reply": "\n".join(lines),
            "tools_used": ["compare_alternatives", "fetch_bestie_insurance_quotes"],
            "data": comp,
        }

    # 3. Intent: City impact / enforcement / why is it expensive here
    if any(w in msg for w in ["עיר", "יישוב", "אזור", "פענוח", "משטרה", "יקר", "למה", "סיכון"]):
        risk = get_vehicle_risk(model, city)
        quote = fetch_bestie_insurance_quotes(model, city, risk_score=risk["combined_risk_score"])
        reply = (
            f"📊 **ניתוח גורמי הסיכון והמחיר עבור {risk['model_hebrew']} ב{risk['city']}:**\n\n"
            f"• **אומדן סיכון (Final):** ציון {risk['combined_risk_score']}/100 ({risk['risk_level_he']}); "
            f"אומדן של כ-**{risk['xgb_predicted_annual_thefts']} גניבות בשנה** לשילוב דגם–עיר.\n"
            f"• **כיסוי נתונים:** {risk['coverage_status']}; מקור הציון: {risk['risk_score_source']}.\n"
            f"• **נתון הקשר:** {risk['total_city_thefts']:,} תיקים רשומים ב{risk['city']} ושיעור גילוי חשודים {risk['city_detection_rate_pct']}%.\n"
            f"• **סימולציית מחיר:** ₪{quote['total_annual_nis']:,}/שנה ודרישת מיגון {quote['required_protection']}; זו אינה הצעת מחיר חיה או הצעת ביטוח."
        )
        return {
            "reply": reply,
            "tools_used": ["get_vehicle_risk", "fetch_bestie_insurance_quotes"],
            "data": {"risk": risk, "quote": quote},
        }

    # Default comprehensive answer combining all insights
    risk = get_vehicle_risk(model, city)
    comp = compare_alternatives(model, city, top_k=3)
    mit = simulate_mitigation_discount(model, city)
    sel = comp["selected_vehicle"]
    best_alt = comp["alternatives"][0] if comp["alternatives"] else None
    alt_sentence = (
        f"• **חלופה מובילה ב-Bestie:** מעבר ל-**{best_alt['model_hebrew']}** (סיכון {best_alt['risk_score']}%) "
        f"יוריד את עלות הביטוח ל-**₪{best_alt['total_annual_nis']:,}/שנה** (חיסכון של **₪{best_alt['annual_savings_nis']:,}** בשנה).\n"
        if best_alt
        else ""
    )
    reply = (
        f"🤖 **סיכום יועץ ה-AI עבור {risk['model_hebrew']} ב{risk['city']}:**\n\n"
        f"• **ציון סיכון וחיזוי ML:** ציון משולב **{risk['combined_risk_score']}%** ({risk['risk_level_he']}), "
        f"צפי XGBoost של כ-**{risk['xgb_predicted_annual_thefts']} גניבות/שנה**.\n"
        f"• **עלות ביטוח ב-Bestie:** **₪{sel['total_annual_nis']:,}/שנה** (מקיף ₪{sel['comprehensive_nis']:,} + חובה ₪{sel['mandatory_nis']:,}, חברה מובילה: {sel['leading_insurer']}).\n"
        f"{alt_sentence}"
        f"• **חיסכון באמצעות מיגון ללא החלפת רכב:** שדרוג מיגון עשוי להוזיל את הפוליסה בכ-**₪{mit['annual_discount_nis']:,}/שנה**."
    )
    return {
        "reply": reply,
        "tools_used": ["get_vehicle_risk", "compare_alternatives", "simulate_mitigation_discount"],
        "data": {"risk": risk, "comparison": comp, "mitigation": mit},
    }


def run_chat_agent(
    message: str,
    model: str,
    city: str,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Executes the chat agent for user questions in the bottom inline chat section.
    Always grounds facts in our deterministic toolset; uses Gemini API for natural
    synthesis if GEMINI_API_KEY is set, otherwise uses the deterministic router.
    """
    det_result = _route_and_answer_deterministically(message, model, city)
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if api_key:
        model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"הקשר נוכחי ונתוני אמת מהכלים (אסור לשנות את המספרים):\n{det_result['reply']}\n\n"
            f"שאלת המשתמש: {message}\n"
            f"נסח תשובה קצרה, מועילה ומדויקת בעברית בהתבסס אך ורק על המספרים שלעיל:"
        )
        try:
            from google import genai  # type: ignore

            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(model=model_name, contents=prompt)
            text = getattr(resp, "text", None)
            if text:
                return {
                    "status": "ok",
                    "engine": "gemini_grounded",
                    "reply": text.strip(),
                    "tools_used": det_result["tools_used"],
                    "disclaimer": DISCLAIMER_HE,
                }
        except Exception:
            pass

    return {
        "status": "ok",
        "engine": "deterministic_grounded",
        "reply": det_result["reply"],
        "tools_used": det_result["tools_used"],
        "disclaimer": DISCLAIMER_HE,
    }
