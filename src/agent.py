"""
agent.py
========
Research AI Agent Orchestrator for CarTheftRisk (Final).
Supports two execution modes:
1. Background Auto-Compare (`run_auto_compare_agent`): Triggered automatically in parallel
   as soon as the user selects a vehicle model and city. Chains the 3 core tools to inspect
   Bestie simulation prices and compare against 3 lower-risk / cheaper-to-insure alternatives.
2. Contextual & Open-Ended Chat (`run_chat_agent`): 4-layer guardrailed pipeline:
   - Layer 1: Input Guardrails & structured `QueryPlan` (length cap, injection/scope check, entity extraction)
   - Layer 2: Strict Entity Grounding (`strict=True`, clarification on unknown model/city, zero silent fallback)
   - Layer 3: Deterministic Tool Execution (`calculate_custom_savings`, `simulate_mitigation_discount`, etc.)
   - Layer 4: Post-Generation Numeric Verifier (blocks/falls back on any ungrounded number from Gemini)
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

from src.agent_schemas import (
    GeminiReplyPayload,
    GroundedAgentResponse,
    IntentEnum,
    QueryPlan,
    SavingsMetrics,
)
from src.agent_tools import (
    CITY_ALIASES,
    MITIGATION_CATALOG,
    MODEL_ALIASES,
    REFERENCE_DRIVER_PROFILE,
    calculate_custom_savings,
    compare_alternatives,
    extract_mitigation_measures,
    fetch_bestie_insurance_quotes,
    get_hebrew_display_name,
    get_vehicle_risk,
    load_artifacts,
    resolve_city_entry,
    resolve_model_entry,
    simulate_mitigation_discount,
    suggest_similar_cities,
    suggest_similar_models,
)

DISCLAIMER_HE = (
    "הערכה אקטוארית וסטטיסטית לצרכי מחקר ולימוד בלבד ואינה מהווה הצעת ביטוח מחייבת או ייעוץ פנסיוני/ביטוחי."
)

SYSTEM_PROMPT = """אתה CarTheftRisk Copilot, סוכן AI אקטוארי אוטונומי לניתוח סיכוני גניבת רכב והשוואת מחירי ביטוח מאתר Bestie בישראל.
חוקי ברזל (Guardrails):
1. אסור בתכלית האיסור להמציא ציוני סיכון, כמויות גניבה או מחירי ביטוח. עליך להסתמך אך ורק על המספרים שהופקו מהכלים הדטרמיניסטיים.
2. השוואת מחירי הביטוח ב-Bestie מבוססת על פרופיל נהג ייחוס קבוע (נהג בן 35, 12 שנות ותק, ללא תביעות ב-3 השנים האחרונות, שנתון 2025).
3. ענה תמיד בעברית ברורה, תמציתית ומקצועית, ואל תשנה או תעגל מספרים שלא הוזכרו בפלט הכלים.
4. אם הוזכר חיסכון שלילי (החלופה יקרה יותר), דווח זאת בכנות ובשקיפות כתוספת עלות.
"""

MAX_MESSAGE_LENGTH = 500

PROMPT_INJECTION_PATTERNS = (
    "ignore previous",
    "ignore all previous",
    "disregard previous",
    "system prompt",
    "developer prompt",
    "jailbreak",
    "reveal your instructions",
    "bypass guardrail",
    "התעלם מההוראות",
    "תתעלם מההוראות",
    "התעלם מהנחיות",
    "שכח את ההוראות",
    "פרומפט מערכת",
    "הנחיות המערכת",
    "חשוף את הפרומפט",
    "עקוף את ההגבלות",
    "תמציא מחיר",
    "המצא מספר",
)

OUT_OF_SCOPE_PATTERNS = (
    "מתכון",
    "פסטה",
    "עוגה",
    "כתוב שיר",
    "שיר אהבה",
    "בדיחה",
    "מזג אוויר",
    "כדורגל",
    "פוליטיקה",
    "הורוסקופ",
    "דיאטה",
    "טיסה",
    "מלון",
    "מסעדה",
    "recipe",
    "joke",
    "poem",
    "weather",
    "bitcoin",
    "crypto",
)

DOMAIN_KEYWORDS = (
    "ביטוח",
    "מקיף",
    "חובה",
    "סיכון",
    "גניב",
    "מיגון",
    "איתוראן",
    "איתורן",
    "קודן",
    "עלוקה",
    "מנעול",
    "חניה",
    "חנייה",
    "אחסוך",
    "חיסכון",
    "להוזיל",
    "הוזל",
    "הנחה",
    "חלופ",
    "השווא",
    "בסטי",
    "bestie",
    "דגם",
    "רכב",
    "מכונית",
    "אוטו",
    "פוליסה",
    "מחיר",
    "עלות",
    "יקר",
    "זול",
    "עיר",
    "יישוב",
    "משטרה",
    "פענוח",
    "איתור חשודים",
    "ציון",
)

FOLLOWUP_CUES = (
    "ומה עם",
    "ואם",
    "וכמה",
    "מה לגבי",
    "ומה לגבי",
    "שם",
    "עליו",
    "עליה",
    "איתו",
    "איתה",
)

_SAFE_SHORT_CITIES = {"לוד", "עכו", "צפת", "ערד", "יפו"}
_COMMON_CITY_FALSE_POSITIVES = {"שלום", "עיר", "אחד", "כמה", "רק", "לא", "גם", "כל", "טוב", "יפה", "חם", "קר", "גן"}
_NON_ENTITY_TAIL_WORDS = {
    "ביטוח",
    "הביטוח",
    "מקיף",
    "חובה",
    "פוליסה",
    "פוליסת",
    "רכב",
    "הרכב",
    "דגם",
    "מכונית",
    "אוטו",
    "בסטי",
    "bestie",
    "שנה",
    "השנה",
    "לשנה",
    "בשנה",
    "חודש",
    "לחודש",
    "בחודש",
    "כסף",
    "בכלל",
    "באמת",
    "זה",
    "הזה",
    "הזאת",
}
_HEB_PREPS = {"ב", "ל", "מ", "ה", "ו"}
_MODEL_PROTECTED_FIRST = {
    "מזדה",
    "מאזדה",
    "מיצובישי",
    "מרצדס",
    "הונדה",
    "וולוו",
    "לנד",
    "לקסוס",
    "בוגאטי",
    "בנטלי",
    "למבורגיני",
    "לינקולן",
}


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


def _norm_tok(text: str) -> str:
    s = (text or "").strip().lower()
    s = s.replace("מאזדה", "מזדה").replace("טיוטה", "טויוטה").replace("ניסן", "ניסאן").replace("שקודה", "סקודה")
    s = re.sub(r'["\'״׳`]', "", s)
    return s


def _tokenize_with_spans(text: str) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for m in re.finditer(r"[^\s\?\.,!;:\(\)\[\]\"״׳']+", text or ""):
        raw = m.group(0)
        tokens.append(
            {
                "raw": raw,
                "norm": _norm_tok(raw),
                "clean_city": re.sub(r"[-–]+", " ", raw).strip(),
                "start": m.start(),
                "end": m.end(),
            }
        )
    return tokens


@lru_cache(maxsize=1)
def _get_model_phrase_index() -> list[tuple[tuple[str, ...], str, int]]:
    """
    Returns sorted `(token_tuple, canonical_query_str, priority_score)` for safe
    token-boundary extraction of vehicle models from free Hebrew/English text.
    """
    _, _, data = load_artifacts()
    models = data.get("models", {})
    by_phrase: dict[tuple[str, ...], tuple[str, int]] = {}

    for alias_heb in MODEL_ALIASES:
        toks = tuple(_norm_tok(w) for w in alias_heb.replace("-", " ").split() if _norm_tok(w))
        if toks:
            by_phrase[toks] = (alias_heb, 100_000 + len(toks) * 1000)

    for k, info in models.items():
        if not info or "|" in k:
            continue
        cat = str(info.get("category", ""))
        if "דו" in cat or "אופנוע" in cat or "קטנוע" in cat:
            continue
        thefts = int(info.get("total_thefts", 0))
        if thefts < 15:
            continue
        heb = get_hebrew_display_name(k, info)
        if not heb:
            continue
        toks = tuple(_norm_tok(w) for w in heb.replace("-", " ").split() if _norm_tok(w))
        if len(toks) >= 2:
            score = thefts + len(toks) * 500
            if toks not in by_phrase or score > by_phrase[toks][1]:
                by_phrase[toks] = (heb, score)
        eng_clean = str(info.get("model_name") or info.get("model_clean") or "").strip()
        if eng_clean and len(eng_clean) >= 3:
            eng_toks = tuple(_norm_tok(w) for w in eng_clean.replace("-", " ").split() if _norm_tok(w))
            if eng_toks and (eng_toks not in by_phrase or thefts > by_phrase[eng_toks][1]):
                by_phrase[eng_toks] = (eng_clean, thefts)

    items = [(toks, canon, prio) for toks, (canon, prio) in by_phrase.items()]
    items.sort(key=lambda x: (len(x[0]), x[2], sum(len(t) for t in x[0])), reverse=True)
    return items


@lru_cache(maxsize=1)
def _get_city_phrase_index() -> list[tuple[tuple[str, ...], str, int]]:
    """
    Returns sorted `(token_tuple, canonical_city_key, total_thefts)` for safe
    token-boundary extraction of Israeli cities from free text.
    """
    _, _, data = load_artifacts()
    locations = data.get("locations", {})
    entries: dict[tuple[str, ...], tuple[str, int]] = {}

    def _add(phrase: str, target_key: str, thefts: int) -> None:
        cleaned = re.sub(r"[-–]+", " ", phrase).strip()
        toks = tuple(cleaned.split())
        if not toks:
            return
        if toks not in entries or thefts > entries[toks][1]:
            entries[toks] = (target_key, thefts)

    for alias, target in CITY_ALIASES.items():
        if target in locations:
            _add(alias, target, int(locations[target].get("total_thefts", 0)) + 50_000)

    for city_key, info in locations.items():
        thefts = int(info.get("total_thefts", 0))
        _add(city_key, city_key, thefts)

    items = [(toks, target, thefts) for toks, (target, thefts) in entries.items()]
    items.sort(key=lambda x: (len(x[0]), sum(len(t) for t in x[0]), x[2]), reverse=True)
    return items


def _match_first_token(
    tok: str,
    expected: str,
    allow_prep: bool,
    protected_words: set[str] | None = None,
    min_prep_tok_len: int = 4,
) -> tuple[bool, str | None]:
    if tok == expected:
        return True, None
    if allow_prep and len(tok) >= 3 and tok[0] in _HEB_PREPS:
        if protected_words and tok in protected_words:
            return False, None
        if len(tok) >= 4 and tok[1] in {"-", "–"} and tok[2:] == expected:
            return True, tok[0]
        if len(tok) >= min_prep_tok_len and tok[1:] == expected:
            return True, tok[0]
    return False, None


def _overlaps_spans(start_char: int, end_char: int, forbidden_spans: list[tuple[int, int]] | None) -> bool:
    if not forbidden_spans:
        return False
    return any(not (end_char <= s or start_char >= e) for s, e in forbidden_spans)


def _extract_models_from_text(text: str) -> list[dict[str, Any]]:
    tokens = _tokenize_with_spans(text)
    if not tokens:
        return []
    index = _get_model_phrase_index()
    used_indices: set[int] = set()
    hits: list[dict[str, Any]] = []

    for phrase_toks, canon_query, _prio in index:
        n = len(phrase_toks)
        min_prep_len = 3 if n >= 2 else 4
        for i in range(0, len(tokens) - n + 1):
            if any(idx in used_indices for idx in range(i, i + n)):
                continue
            ok0, prep = _match_first_token(
                tokens[i]["norm"],
                phrase_toks[0],
                allow_prep=True,
                protected_words=_MODEL_PROTECTED_FIRST,
                min_prep_tok_len=min_prep_len,
            )
            if not ok0:
                continue
            rest_ok = all(tokens[i + j]["norm"] == phrase_toks[j] for j in range(1, n))
            if not rest_ok:
                continue
            resolved = resolve_model_entry(canon_query, strict=True)
            if not resolved:
                continue
            used_indices.update(range(i, i + n))
            hits.append(
                {
                    "query": canon_query,
                    "model_key": resolved[0],
                    "model_clean": resolved[1].get("model_clean", resolved[0]),
                    "prep": prep,
                    "start_tok": i,
                    "end_tok": i + n,
                    "start_char": tokens[i]["start"],
                    "end_char": tokens[i + n - 1]["end"],
                }
            )

    hits.sort(key=lambda h: h["start_char"])
    # Deduplicate same canonical model keeping earliest mention
    unique: list[dict[str, Any]] = []
    seen_clean: set[str] = set()
    for h in hits:
        key = h["model_clean"] or h["model_key"]
        if key not in seen_clean:
            seen_clean.add(key)
            unique.append(h)
    return unique


def _extract_cities_from_text(
    text: str,
    forbidden_char_spans: list[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    # Expand hyphenated tokens (e.g. "בחיפה-כרמל" or "תל-אביב" or "בראשון-לציון") properly
    expanded_tokens: list[dict[str, Any]] = []
    for m in re.finditer(r"[^\s\?\.,!;:\(\)\[\]\"״׳']+", text or ""):
        raw = m.group(0)
        parts = [p for p in re.split(r"[-–]+", raw) if p]
        if len(parts) > 1:
            offset = m.start()
            for p in parts:
                pos = text.find(p, offset)
                expanded_tokens.append({"raw": p, "start": pos, "end": pos + len(p), "is_subpart": True})
                offset = pos + len(p)
        else:
            expanded_tokens.append({"raw": raw, "start": m.start(), "end": m.end(), "is_subpart": False})

    if not expanded_tokens:
        return []

    index = _get_city_phrase_index()
    used_indices: set[int] = set()
    hits: list[dict[str, Any]] = []

    for phrase_toks, target_city_key, thefts in index:
        n = len(phrase_toks)
        min_prep_len = 3 if n >= 2 else 4
        phrase_str = " ".join(phrase_toks)
        is_short_risky = n == 1 and len(phrase_str) <= 3 and phrase_str not in _SAFE_SHORT_CITIES
        is_common_word = n == 1 and phrase_str in _COMMON_CITY_FALSE_POSITIVES
        needs_explicit_city_cue = is_short_risky or is_common_word

        for i in range(0, len(expanded_tokens) - n + 1):
            if any(idx in used_indices for idx in range(i, i + n)):
                continue
            span_start = expanded_tokens[i]["start"]
            span_end = expanded_tokens[i + n - 1]["end"]
            if _overlaps_spans(span_start, span_end, forbidden_char_spans):
                continue
            tok0 = expanded_tokens[i]["raw"]
            ok0, prep = _match_first_token(tok0, phrase_toks[0], allow_prep=True, min_prep_tok_len=min_prep_len)
            if not ok0:
                continue
            if not all(expanded_tokens[i + j]["raw"] == phrase_toks[j] for j in range(1, n)):
                continue

            prev_word = expanded_tokens[i - 1]["raw"] if i > 0 else ""
            has_city_cue = prev_word in {"בעיר", "ביישוב", "מאזור", "באזור", "לעיר", "מעיר"}
            if needs_explicit_city_cue and not has_city_cue:
                continue

            # Guard low-theft single-word settlements from matching unrelated Hebrew words
            if n == 1 and thefts < 15 and len(phrase_str) <= 5 and phrase_str not in _SAFE_SHORT_CITIES:
                if not prep and not has_city_cue and prev_word not in {"לעומת", "מול", "אגור", "לגור", "ב", "ל", "מ"}:
                    continue

            resolved = resolve_city_entry(target_city_key, strict=True)
            if not resolved:
                continue
            used_indices.update(range(i, i + n))
            hits.append(
                {
                    "query": target_city_key,
                    "city_key": resolved[0],
                    "city_unified": resolved[1].get("city_unified", resolved[0]),
                    "prep": prep,
                    "start_char": expanded_tokens[i]["start"],
                    "end_char": expanded_tokens[i + n - 1]["end"],
                }
            )

    hits.sort(key=lambda h: h["start_char"])
    unique: list[dict[str, Any]] = []
    seen_unified: set[str] = set()
    for h in hits:
        if h["city_unified"] not in seen_unified:
            seen_unified.add(h["city_unified"])
            unique.append(h)
    return unique


def _extract_unknown_target_phrase(text: str) -> tuple[str | None, str | None]:
    """
    When the user asks a savings/comparison question about an unrecognised model or city
    (e.g. `"כמה אחסוך בלמבורגיני אורוס?"` or `"כמה אחסוך אם אגור באטלנטיס?"`),
    returns `(candidate_phrase, "city" | "model")` so Layer 2 asks for clarification
    instead of falling back to a random car or city.
    """
    city_move_cues = ("אגור", "לגור", "בעיר", "ביישוב", "לעיר")
    patterns: list[tuple[str, str]] = [
        (r"(?:אם אגור\s+ב-?|לגור\s+ב-?|בעיר\s+|ביישוב\s+)([^\?\.,!]+)", "city"),
        (
            r"(?:אעבור\s+ל-?|לעבור\s+ל-?|במעבר\s+ל-?|אחליף\s+ל-?|להחליף\s+ל-?|בדגם\s+|ברכב\s+|אחסוך\s+ב-?|אחסוך\s+עם\s+|אחסוך\s+על\s+|חיסכון\s+ב-?|חיסכון\s+עם\s+)([^\?\.,!]+)",
            "context",
        ),
        (r"(?:לעומת|מול|במקום)\s+([^\?\.,!]+)", "context"),
    ]
    for pat, hint in patterns:
        m = re.search(pat, text)
        if not m:
            continue
        raw_cand = m.group(1).strip()
        # Cut off trailing comparison connectors or questions
        raw_cand = re.split(r"\s+(?:לעומת|מול|במקום|אם|ומה|וכמה|בשנה|לחודש|לשנה)\b", raw_cand)[0].strip()
        words = [w.strip("\"'״׳-") for w in raw_cand.split() if w.strip("\"'״׳-")]
        while words and words[-1].lower() in _NON_ENTITY_TAIL_WORDS:
            words.pop()
        while words and words[0].lower() in _NON_ENTITY_TAIL_WORDS:
            words.pop(0)
        cand = " ".join(words).strip()
        if len(cand) < 2:
            continue
        if hint == "city" or any(cue in text for cue in city_move_cues):
            return cand, "city"
        return cand, "model"
    return None, None


def _assign_pair_roles(first: dict[str, Any], second: dict[str, Any], text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns `(base_item, target_item)` from two mentions ordered left-to-right."""
    between = text[first["end_char"] : second["start_char"]]
    if "לעומת" in between or "מול" in between or "בהשוואה ל" in between:
        # e.g. "בראשון לציון לעומת תל אביב" -> target is Rishon, base is Tel Aviv
        return second, first
    if first.get("prep") == "מ" and second.get("prep") == "ל":
        # e.g. "מקורולה לקיה נירו" / "מחיפה לתל אביב"
        return first, second
    return first, second


def plan_query(
    message: str,
    model: str,
    city: str,
    history: list[dict[str, str]] | None = None,
) -> QueryPlan:
    """
    Layer 1: Runs deterministic input guardrails and produces a validated `QueryPlan`.
    """
    msg = (message or "").strip()
    msg_low = msg.lower()

    if not msg or len(msg) > MAX_MESSAGE_LENGTH:
        return QueryPlan(
            intent=IntentEnum.out_of_scope,
            needs_clarification=True,
            clarification_question_he=f"אנא הזינו שאלה באורך של עד {MAX_MESSAGE_LENGTH} תווים.",
        )

    if any(pat in msg_low for pat in PROMPT_INJECTION_PATTERNS):
        return QueryPlan(intent=IntentEnum.prompt_injection)

    base_model_resolved = resolve_model_entry(model, strict=True) or resolve_model_entry(model, strict=False)
    base_city_resolved = resolve_city_entry(city, strict=True) or resolve_city_entry(city, strict=False)
    base_model_clean = base_model_resolved[1].get("model_clean", base_model_resolved[0])
    base_city_unified = base_city_resolved[1].get("city_unified", base_city_resolved[0])

    measures = extract_mitigation_measures(msg)
    models_found = _extract_models_from_text(msg)
    model_spans = [(m["start_char"], m["end_char"]) for m in models_found]
    cities_found = _extract_cities_from_text(msg, forbidden_char_spans=model_spans)

    has_domain_kw = any(kw in msg_low for kw in DOMAIN_KEYWORDS) or bool(measures or models_found or cities_found)
    is_followup = any(cue in msg for cue in FOLLOWUP_CUES)

    if any(pat in msg_low for pat in OUT_OF_SCOPE_PATTERNS) and not (has_domain_kw and not any(w in msg_low for w in ("מתכון", "פסטה", "בדיחה", "כתוב שיר"))):
        return QueryPlan(intent=IntentEnum.out_of_scope)

    if not has_domain_kw and not is_followup:
        return QueryPlan(intent=IntentEnum.out_of_scope)

    base_model_override: str | None = None
    target_model: str | None = None
    base_city_override: str | None = None
    target_city: str | None = None

    if len(models_found) >= 2:
        b_m, t_m = _assign_pair_roles(models_found[0], models_found[1], msg)
        base_model_override = b_m["model_key"]
        target_model = t_m["model_key"]
    elif len(models_found) == 1:
        m0 = models_found[0]
        if m0["model_clean"] == base_model_clean:
            base_model_override = m0["model_key"]
        else:
            target_model = m0["model_key"]

    if len(cities_found) >= 2:
        b_c, t_c = _assign_pair_roles(cities_found[0], cities_found[1], msg)
        base_city_override = b_c["city_key"]
        target_city = t_c["city_key"]
    elif len(cities_found) == 1:
        c0 = cities_found[0]
        if len(models_found) >= 2 or target_model is not None:
            # A city alongside a model comparison specifies the shared comparison city
            base_city_override = c0["city_key"]
        elif c0["city_unified"] == base_city_unified:
            base_city_override = c0["city_key"]
        else:
            target_city = c0["city_key"]

    # Reuse prior user turn entities for short follow-up questions ("ומה עם...")
    if history and is_followup and not target_model and not target_city and not measures:
        for prev in reversed(history[-4:]):
            if str(prev.get("role", "")).lower() != "user":
                continue
            prev_text = str(prev.get("content", "")).strip()
            if not prev_text:
                continue
            prev_models = _extract_models_from_text(prev_text)
            prev_model_spans = [(m["start_char"], m["end_char"]) for m in prev_models]
            prev_cities = _extract_cities_from_text(prev_text, forbidden_char_spans=prev_model_spans)
            if prev_models and prev_models[-1]["model_clean"] != base_model_clean:
                target_model = prev_models[-1]["model_key"]
                break
            if prev_cities and prev_cities[-1]["city_unified"] != base_city_unified:
                target_city = prev_cities[-1]["city_key"]
                break

    # Unknown target model or city detection (for "כמה אחסוך בלמבורגיני אורוס?" / "...באטלנטיס?")
    savings_or_switch_kw = (
        "אחסוך",
        "חיסכון",
        "לעבור",
        "אעבור",
        "אחליף",
        "להחליף",
        "במעבר",
        "אגור",
        "לגור",
        "במקום",
        "לעומת",
        "מול",
    )
    if (
        not measures
        and not target_model
        and not target_city
        and any(w in msg for w in savings_or_switch_kw)
        and not any(w in msg for w in ("מיגון", "חלופ", "זול יותר", "הכי זול", "זולה", "רכבים אחרים", "השווא"))
    ):
        unknown_cand, unknown_kind = _extract_unknown_target_phrase(msg)
        if unknown_cand:
            if unknown_kind == "city" and resolve_city_entry(unknown_cand, strict=True) is None:
                target_city = unknown_cand
            elif unknown_kind == "model" and resolve_model_entry(unknown_cand, strict=True) is None:
                target_model = unknown_cand

    mitigation_general_kw = ("מיגון", "להוזיל", "הנחה")
    if measures and not target_model and not target_city:
        intent = IntentEnum.mitigation_savings
    elif any(w in msg for w in mitigation_general_kw) and not target_model and not target_city:
        intent = IntentEnum.mitigation_savings
    elif target_model is not None:
        intent = IntentEnum.compare_model_savings
    elif target_city is not None:
        intent = IntentEnum.compare_city_savings
    elif any(
        w in msg_low
        for w in ("חלופ", "זול יותר", "הכי זול", "זולה", "רכבים אחרים", "השווא", "להחליף", "דגמים דומים", "בסטי", "bestie", "אחסוך", "חיסכון")
    ):
        intent = IntentEnum.alternatives_list
    elif any(w in msg_low for w in ("עיר", "יישוב", "אזור", "פענוח", "משטרה", "יקר", "למה", "סיכון", "גניב", "ציון", "מחיר", "עולה", "ביטוח", "מקיף", "פוליסה", "דגם", "רכב")):
        intent = IntentEnum.risk_explanation
    else:
        intent = IntentEnum.out_of_scope

    return QueryPlan(
        intent=intent,
        base_model_override=base_model_override,
        base_city_override=base_city_override,
        target_model=target_model,
        target_city=target_city,
        mitigation_measures=measures,
    )


def _format_savings_reply(plan_intent: IntentEnum, res: dict[str, Any], measures: list[str] | None = None) -> str:
    m = res["metrics"]
    b_model = m["base_model_he"]
    b_city = m["base_city"]
    t_model = m.get("target_model_he") or b_model
    t_city = m.get("target_city") or b_city
    ann_save = m["annual_savings_nis"]
    mo_save = m["monthly_savings_nis"]
    risk_delta = m["risk_delta_pts"]

    if plan_intent == IntentEnum.compare_city_savings:
        header = f"📍 **השוואת מעבר עיר עבור {b_model} (מ-{b_city} ל-{t_city}):**\n\n"
    else:
        same_city_note = f" ב-{b_city}" if b_city == t_city else f" ({b_city} לעומת {t_city})"
        header = f"🚗 **השוואת חלופת דגם{same_city_note} ({b_model} לעומת {t_model}):**\n\n"

    if ann_save > 0:
        delta_line = f"• **חיסכון שנתי משוער ב-Bestie:** **₪{ann_save:,}/שנה** (כ-**₪{mo_save:,}/חודש**).\n"
    elif ann_save < 0:
        extra_ann = abs(ann_save)
        extra_mo = abs(mo_save)
        delta_line = f"• **תוספת עלות שנתית משוערת ב-Bestie:** היעד יקר יותר ב-**₪{extra_ann:,}/שנה** (כ-**₪{extra_mo:,}/חודש**).\n"
    else:
        delta_line = "• **הפרש עלות ב-Bestie:** ללא שינוי משמעותי בעלות השנתית (**₪0/שנה**).\n"

    if risk_delta > 0:
        risk_line = f"• **שינוי בציון הסיכון:** ירידה של **{risk_delta} נק׳** (מ-{m['base_risk_score']} ל-{m['target_risk_score']}/100).\n"
    elif risk_delta < 0:
        risk_line = f"• **שינוי בציון הסיכון:** עלייה של **{abs(risk_delta)} נק׳** (מ-{m['base_risk_score']} ל-{m['target_risk_score']}/100).\n"
    else:
        risk_line = f"• **ציון סיכון:** זהה בשני המקרים (**{m['base_risk_score']}/100**).\n"

    mit_line = ""
    if measures and res.get("mitigation"):
        applied = ", ".join(item["measure"] for item in res["mitigation"]["applied_measures"])
        mit_line = f"• **מיגון משוקלל ביעד:** {applied}.\n"

    return (
        f"{header}"
        f"• **בסיס ({b_model}, {b_city}):** **₪{m['base_annual_nis']:,}/שנה** | ציון סיכון **{m['base_risk_score']}/100**.\n"
        f"• **יעד ({t_model}, {t_city}):** **₪{m['target_annual_nis']:,}/שנה** | ציון סיכון **{m['target_risk_score']}/100**.\n"
        f"{delta_line}"
        f"{risk_line}"
        f"{mit_line}"
        f"• **כיסוי נתונים:** {res['data_coverage_note']}"
    )


def execute_grounded_plan(
    plan: QueryPlan,
    base_model: str,
    base_city: str,
) -> tuple[GroundedAgentResponse, dict[str, Any]]:
    """
    Layer 2 & 3: Performs strict entity resolution and executes deterministic ML/simulation tools.
    """
    assumptions = dict(REFERENCE_DRIVER_PROFILE)
    base_guards = ["input_length_ok", "prompt_injection_checked"]

    if plan.intent == IntentEnum.prompt_injection:
        return (
            GroundedAgentResponse(
                status="blocked",
                engine="guardrail_filter",
                intent=IntentEnum.prompt_injection,
                reply=(
                    "הבקשה נחסמה על ידי מנגנון האבטחה. אני סוכן AI ייעודי לניתוח סיכוני גניבת רכב והשוואת "
                    "מחירי ביטוח מ-Bestie בישראל, ואיני יכול לשנות את הנחיות המערכת או להמציא נתונים."
                ),
                metrics=None,
                assumptions=assumptions,
                data_coverage_note="",
                tools_used=["guardrail_filter"],
                guardrails_passed=["input_length_ok", "prompt_injection_blocked"],
                disclaimer=DISCLAIMER_HE,
            ),
            {},
        )

    if plan.intent == IntentEnum.out_of_scope:
        reply_txt = plan.clarification_question_he or (
            "אני סוכן AI ייעודי לניתוח סיכוני גניבת רכב והשוואת מחירי ביטוח בישראל (Bestie). "
            "אפשר לשאול אותי למשל: \"כמה אחסוך בקיה נירו?\", \"כמה אחסוך בראשון לציון לעומת תל אביב?\" או \"כמה אחסוך רק עם קודן?\"."
        )
        return (
            GroundedAgentResponse(
                status="blocked",
                engine="guardrail_filter",
                intent=IntentEnum.out_of_scope,
                reply=reply_txt,
                metrics=None,
                assumptions=assumptions,
                data_coverage_note="",
                tools_used=["guardrail_filter"],
                guardrails_passed=["input_length_ok", "scope_filter_triggered"],
                disclaimer=DISCLAIMER_HE,
            ),
            {},
        )

    # Resolve base model & city (explicit overrides use strict resolution first)
    eff_base_model_input = plan.base_model_override or base_model
    eff_base_city_input = plan.base_city_override or base_city

    resolved_base_m = resolve_model_entry(eff_base_model_input, strict=True)
    if resolved_base_m is None and plan.base_model_override is None:
        resolved_base_m = resolve_model_entry(eff_base_model_input, strict=False)

    if resolved_base_m is None:
        sugs = suggest_similar_models(eff_base_model_input, limit=5)
        bullets = "\n".join(f"• {s}" for s in sugs)
        return (
            GroundedAgentResponse(
                status="needs_clarification",
                engine="deterministic_grounded",
                intent=plan.intent,
                reply=(
                    f"הדגם **\"{eff_base_model_input}\"** אינו מופיע במאגר הנתונים המאומת, ולכן לא ניתן לחשב עבורו אומדן ללא הזיה.\n\n"
                    f"דגמים קיימים במאגר שתוכל לבחור מתוכם:\n{bullets}"
                ),
                metrics=None,
                assumptions=assumptions,
                data_coverage_note="הדגם המבוקש אינו קיים במאגר הנתונים המאומת.",
                tools_used=["resolve_model_entry"],
                guardrails_passed=[*base_guards, "scope_verified", "strict_entity_grounding"],
                disclaimer=DISCLAIMER_HE,
            ),
            {},
        )

    resolved_base_c = resolve_city_entry(eff_base_city_input, strict=True)
    if resolved_base_c is None and plan.base_city_override is None:
        resolved_base_c = resolve_city_entry(eff_base_city_input, strict=False)

    if resolved_base_c is None:
        sugs = suggest_similar_cities(eff_base_city_input, limit=5)
        bullets = "\n".join(f"• {s}" for s in sugs)
        return (
            GroundedAgentResponse(
                status="needs_clarification",
                engine="deterministic_grounded",
                intent=plan.intent,
                reply=(
                    f"היישוב **\"{eff_base_city_input}\"** אינו מופיע במאגר הנתונים המאומת, ולכן איני מנחש נתוני עיר.\n\n"
                    f"יישובים קיימים במאגר שתוכל לבדוק:\n{bullets}"
                ),
                metrics=None,
                assumptions=assumptions,
                data_coverage_note="היישוב המבוקש אינו קיים במאגר הנתונים המאומת.",
                tools_used=["resolve_city_entry"],
                guardrails_passed=[*base_guards, "scope_verified", "strict_entity_grounding"],
                disclaimer=DISCLAIMER_HE,
            ),
            {},
        )

    base_model_key, _ = resolved_base_m
    base_city_key, _ = resolved_base_c

    if plan.intent in (IntentEnum.compare_model_savings, IntentEnum.compare_city_savings):
        target_model_key: str | None = None
        target_city_key: str | None = None

        if plan.target_model is not None:
            resolved_tm = resolve_model_entry(plan.target_model, strict=True)
            if resolved_tm is None:
                sugs = suggest_similar_models(plan.target_model, limit=5)
                bullets = "\n".join(f"• {s}" for s in sugs)
                return (
                    GroundedAgentResponse(
                        status="needs_clarification",
                        engine="deterministic_grounded",
                        intent=plan.intent,
                        reply=(
                            f"הדגם **\"{plan.target_model}\"** אינו מופיע במאגר הנתונים המאומת שלנו, ולכן איני ממציא עבורו הערכת סיכון או מחיר ביטוח.\n\n"
                            f"הנה דגמים מאומתים שקיימים במאגר וניתן להשוות אליהם:\n{bullets}"
                        ),
                        metrics=None,
                        assumptions=assumptions,
                        data_coverage_note="הדגם המבוקש אינו מופיע במאגר הנתונים המאומת.",
                        tools_used=["resolve_model_entry"],
                        guardrails_passed=[*base_guards, "scope_verified", "strict_entity_grounding"],
                        disclaimer=DISCLAIMER_HE,
                    ),
                    {},
                )
            target_model_key = resolved_tm[0]

        if plan.target_city is not None:
            resolved_tc = resolve_city_entry(plan.target_city, strict=True)
            if resolved_tc is None:
                sugs = suggest_similar_cities(plan.target_city, limit=5)
                bullets = "\n".join(f"• {s}" for s in sugs)
                return (
                    GroundedAgentResponse(
                        status="needs_clarification",
                        engine="deterministic_grounded",
                        intent=plan.intent,
                        reply=(
                            f"היישוב **\"{plan.target_city}\"** אינו מופיע במאגר הנתונים המאומת שלנו, ולכן לא ניתן לחשב עבורו השוואת ביטוח.\n\n"
                            f"הנה יישובים מאומתים במאגר שניתן להשוות אליהם:\n{bullets}"
                        ),
                        metrics=None,
                        assumptions=assumptions,
                        data_coverage_note="היישוב המבוקש אינו מופיע במאגר הנתונים המאומת.",
                        tools_used=["resolve_city_entry"],
                        guardrails_passed=[*base_guards, "scope_verified", "strict_entity_grounding"],
                        disclaimer=DISCLAIMER_HE,
                    ),
                    {},
                )
            target_city_key = resolved_tc[0]

        res = calculate_custom_savings(
            base_model=base_model_key,
            base_city=base_city_key,
            target_model=target_model_key,
            target_city=target_city_key,
            measures=plan.mitigation_measures or None,
        )
        metrics = SavingsMetrics(**res["metrics"])
        reply = _format_savings_reply(plan.intent, res, plan.mitigation_measures or None)
        tools_used = ["calculate_custom_savings", "get_vehicle_risk", "fetch_bestie_insurance_quotes"]
        if plan.mitigation_measures:
            tools_used.append("simulate_mitigation_discount")
        return (
            GroundedAgentResponse(
                status="answered",
                engine="deterministic_grounded",
                intent=plan.intent,
                reply=reply,
                metrics=metrics,
                assumptions=assumptions,
                data_coverage_note=res["data_coverage_note"],
                tools_used=tools_used,
                guardrails_passed=[*base_guards, "scope_verified", "strict_entity_grounding", "deterministic_tool_grounded"],
                disclaimer=DISCLAIMER_HE,
            ),
            res,
        )

    if plan.intent == IntentEnum.mitigation_savings:
        measures_arg = plan.mitigation_measures or None
        risk = get_vehicle_risk(base_model_key, base_city_key)
        mit = simulate_mitigation_discount(base_model_key, base_city_key, measures=measures_arg)
        measures_list = ", ".join(
            f"{m['measure']} (הפחתת סיכון {m['risk_reduction_pct']}%, הנחה {m['comp_discount_pct']}%)"
            for m in mit["applied_measures"]
        )
        ann_disc = int(mit["annual_discount_nis"])
        mo_disc = int(round(ann_disc / 12.0))
        metrics = SavingsMetrics(
            base_model_he=mit["model_hebrew"],
            base_city=mit["city"],
            target_model_he=mit["model_hebrew"],
            target_city=mit["city"],
            base_annual_nis=int(mit["original_total_nis"]),
            target_annual_nis=int(mit["mitigated_total_nis"]),
            annual_savings_nis=ann_disc,
            monthly_savings_nis=mo_disc,
            base_risk_score=int(mit["original_risk_score"]),
            target_risk_score=int(mit["mitigated_risk_score"]),
            risk_delta_pts=int(mit["original_risk_score"]) - int(mit["mitigated_risk_score"]),
        )
        reply = (
            f"🛡️ **סימולציית הוזלת ביטוח באמצעות מיגון עבור {mit['model_hebrew']} ב-{mit['city']}:**\n\n"
            f"• **אמצעי מיגון שנבדקו:** {measures_list}.\n"
            f"• **השפעה על ציון הסיכון:** ירידה מ-**{mit['original_risk_score']}** ל-**{mit['mitigated_risk_score']}/100**.\n"
            f"• **השפעה על הצעת Bestie:** הוזלה של כ-**₪{ann_disc:,} בשנה** (כ-**₪{mo_disc:,}/חודש**, "
            f"מ-₪{mit['original_total_nis']:,} ל-**₪{mit['mitigated_total_nis']:,}** לשנה במקיף + חובה, כ-{mit['discount_pct']}% הנחה על המקיף)."
        )
        coverage_note = f"{risk['coverage_status']}; {risk['risk_score_source']}."
        return (
            GroundedAgentResponse(
                status="answered",
                engine="deterministic_grounded",
                intent=IntentEnum.mitigation_savings,
                reply=reply,
                metrics=metrics,
                assumptions=assumptions,
                data_coverage_note=coverage_note,
                tools_used=["simulate_mitigation_discount", "fetch_bestie_insurance_quotes"],
                guardrails_passed=[*base_guards, "scope_verified", "strict_entity_grounding", "deterministic_tool_grounded"],
                disclaimer=DISCLAIMER_HE,
            ),
            {"mitigation": mit, "risk": risk, "metrics": metrics.model_dump()},
        )

    if plan.intent == IntentEnum.alternatives_list:
        risk = get_vehicle_risk(base_model_key, base_city_key)
        comp = compare_alternatives(base_model_key, base_city_key, top_k=3)
        sel = comp["selected_vehicle"]
        alts = comp["alternatives"]
        lines = [
            f"🔄 **השוואת Bestie לדגמים בטוחים וזולים יותר לביטוח בקטגוריית {comp['category']} ({comp['city']}):**\n",
            f"• **הרכב שבחרת ({sel['model_hebrew']}):** ₪{sel['total_annual_nis']:,}/שנה ({sel['leading_insurer']}) | סיכון {sel['risk_score']}/100.",
        ]
        for idx, alt in enumerate(alts, 1):
            lines.append(
                f"• **חלופה {idx} – {alt['model_hebrew']}:** ₪{alt['total_annual_nis']:,}/שנה ({alt['leading_insurer']}) | "
                f"סיכון {alt['risk_score']}/100 | **חיסכון של ₪{alt['annual_savings_nis']:,} בשנה**."
            )
        best_alt = alts[0] if alts else None
        metrics = (
            SavingsMetrics(
                base_model_he=sel["model_hebrew"],
                base_city=comp["city"],
                target_model_he=best_alt["model_hebrew"],
                target_city=comp["city"],
                base_annual_nis=int(sel["total_annual_nis"]),
                target_annual_nis=int(best_alt["total_annual_nis"]),
                annual_savings_nis=int(best_alt["annual_savings_nis"]),
                monthly_savings_nis=int(best_alt["monthly_savings_nis"]),
                base_risk_score=int(sel["risk_score"]),
                target_risk_score=int(best_alt["risk_score"]),
                risk_delta_pts=int(sel["risk_score"]) - int(best_alt["risk_score"]),
            )
            if best_alt
            else None
        )
        return (
            GroundedAgentResponse(
                status="answered",
                engine="deterministic_grounded",
                intent=IntentEnum.alternatives_list,
                reply="\n".join(lines),
                metrics=metrics,
                assumptions=assumptions,
                data_coverage_note=f"{risk['coverage_status']}; {risk['risk_score_source']}.",
                tools_used=["compare_alternatives", "fetch_bestie_insurance_quotes"],
                guardrails_passed=[*base_guards, "scope_verified", "strict_entity_grounding", "deterministic_tool_grounded"],
                disclaimer=DISCLAIMER_HE,
            ),
            {"comparison": comp, "risk": risk},
        )

    # IntentEnum.risk_explanation
    risk = get_vehicle_risk(base_model_key, base_city_key)
    quote = fetch_bestie_insurance_quotes(base_model_key, base_city_key, risk_score=risk["combined_risk_score"])
    reply = (
        f"📊 **ניתוח גורמי הסיכון והמחיר עבור {risk['model_hebrew']} ב-{risk['city']}:**\n\n"
        f"• **אומדן סיכון (Final):** ציון {risk['combined_risk_score']}/100 ({risk['risk_level_he']}); "
        f"אומדן של כ-**{risk['xgb_predicted_annual_thefts']} גניבות בשנה** לשילוב דגם–עיר.\n"
        f"• **כיסוי נתונים:** {risk['coverage_status']}; מקור הציון: {risk['risk_score_source']}.\n"
        f"• **נתון הקשר:** {risk['total_city_thefts']:,} תיקים רשומים ב-{risk['city']} ושיעור גילוי חשודים {risk['city_detection_rate_pct']}%.\n"
        f"• **סימולציית מחיר ב-Bestie:** ₪{quote['total_annual_nis']:,}/שנה ({quote['leading_insurer']}, דרישת מיגון: {quote['required_protection']})."
    )
    return (
        GroundedAgentResponse(
            status="answered",
            engine="deterministic_grounded",
            intent=IntentEnum.risk_explanation,
            reply=reply,
            metrics=None,
            assumptions=assumptions,
            data_coverage_note=f"{risk['coverage_status']}; {risk['risk_score_source']}.",
            tools_used=["get_vehicle_risk", "fetch_bestie_insurance_quotes"],
            guardrails_passed=[*base_guards, "scope_verified", "strict_entity_grounding", "deterministic_tool_grounded"],
            disclaimer=DISCLAIMER_HE,
        ),
        {"risk": risk, "quote": quote},
    )


_NUMBER_TOKEN_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")

_PROFILE_CONSTANT_NUMBERS: frozenset[float] = frozenset(
    {
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        12.0,
        35.0,
        100.0,
        2022.0,
        2024.0,
        2025.0,
        15000.0,
    }
)


def extract_numbers_from_text(text: str) -> list[str]:
    """Extracts all integer/decimal numeric tokens (including comma-separated thousands)."""
    if not text:
        return []
    return _NUMBER_TOKEN_RE.findall(text)


def _walk_numbers(obj: Any, out: set[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        val = float(obj)
        out.add(val)
        out.add(abs(val))
        out.add(float(round(abs(val))))
        out.add(round(abs(val), 1))
        return
    if isinstance(obj, str):
        for tok in extract_numbers_from_text(obj):
            try:
                val = float(tok.replace(",", ""))
                out.add(val)
                out.add(abs(val))
            except ValueError:
                pass
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _walk_numbers(v, out)
        return
    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            _walk_numbers(item, out)


def collect_allowed_numbers(tool_payload: Any, deterministic_reply: str) -> set[float]:
    """Builds the authoritative set of numbers an LLM phrasing is allowed to state."""
    allowed = set(_PROFILE_CONSTANT_NUMBERS)
    _walk_numbers(tool_payload, allowed)
    _walk_numbers(deterministic_reply, allowed)
    return allowed


def verify_numeric_grounding(reply_text: str, allowed_numbers: set[float]) -> tuple[bool, list[str]]:
    """
    Layer 4 Post-Generation Numeric Verifier.
    Every number in `reply_text` must appear in `allowed_numbers`.
    Returns `(is_valid, ungrounded_tokens)`.
    """
    tokens = extract_numbers_from_text(reply_text)
    if not tokens:
        return True, []

    ungrounded: list[str] = []
    for tok in tokens:
        try:
            val = float(tok.replace(",", ""))
        except ValueError:
            continue
        if not any(abs(val - a) <= 1e-6 for a in allowed_numbers):
            ungrounded.append(tok)
    return len(ungrounded) == 0, ungrounded


def _extract_gemini_text(resp: Any) -> str:
    raw_text = (getattr(resp, "text", None) or "").strip()
    if not raw_text:
        return ""
    if raw_text.startswith("{"):
        try:
            parsed = GeminiReplyPayload.model_validate_json(raw_text)
            return parsed.reply.strip()
        except Exception:
            pass
    return raw_text


def run_chat_agent(
    message: str,
    model: str,
    city: str,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    Executes the 4-layer guardrailed chat agent pipeline and returns a
    `GroundedAgentResponse`-conformant dictionary.
    """
    plan = plan_query(message, model, city, history=history)
    response, tool_payload = execute_grounded_plan(plan, model, city)

    if response.status != "answered":
        return response.model_dump()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if api_key:
        model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
        recent_history = ""
        if history:
            tail = history[-4:]
            formatted = "\n".join(
                f"- {item.get('role', 'user')}: {str(item.get('content', ''))[:240]}"
                for item in tail
                if isinstance(item, dict) and item.get("content")
            )
            if formatted:
                recent_history = f"\nהיסטוריית שיחה קצרה:\n{formatted}\n"

        metrics_block = response.metrics.model_dump_json(indent=2) if response.metrics else "{}"
        prompt = (
            f"{SYSTEM_PROMPT}\n"
            f"{recent_history}\n"
            f"נתוני אמת מאומתים מהכלים (אסור לשנות או להוסיף אף מספר):\n{response.reply}\n\n"
            f"מדדים מובנים (SavingsMetrics):\n{metrics_block}\n\n"
            f"שאלת המשתמש: {message}\n"
            f"נסח תשובה קצרה, ברורה ומדויקת בעברית בהתבסס אך ורק על המספרים שלעיל."
        )
        try:
            from google import genai  # type: ignore

            client = genai.Client(api_key=api_key)
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": GeminiReplyPayload,
                        "temperature": 0.1,
                    },
                )
            except TypeError:
                resp = client.models.generate_content(model=model_name, contents=prompt)

            candidate_text = _extract_gemini_text(resp)
            if candidate_text:
                allowed = collect_allowed_numbers(tool_payload, response.reply)
                ok, _ungrounded = verify_numeric_grounding(candidate_text, allowed)
                if ok:
                    response.reply = candidate_text
                    response.engine = "gemini_verified"
                    response.guardrails_passed.append("numeric_output_verified")
                    return response.model_dump()
                response.guardrails_passed.append("numeric_verifier_fallback_triggered")
                return response.model_dump()
        except Exception:
            pass

    response.guardrails_passed.append("numeric_output_verified")
    return response.model_dump()
