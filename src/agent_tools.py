"""
agent_tools.py
==============
Deterministic research toolset for the CarTheftRisk AI Agent (v9).
Theft estimates use `web/assets/v9_scores.json` and the v9 XGBoost artifact.
Insurance prices are a reference-driver simulation, not live insurer quotes.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any

import numpy as np
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_theft_regressor.json")
CARD_PATH = os.path.join(BASE_DIR, "models", "model_card.json")
V9_SCORES_PATH = os.path.join(BASE_DIR, "web", "assets", "v9_scores.json")
FINAL_SCORES_PATH = os.path.join(BASE_DIR, "web", "assets", "final_scores.json")
DATA_PATH = os.path.join(BASE_DIR, "web", "assets", "data.json")

REFERENCE_DRIVER_PROFILE = {
    "age": 35,
    "license_years": 12,
    "claims_last_3_years": 0,
    "model_year": 2022,
    "usage": "פרטי (עד 15,000 ק״מ בשנה)",
    "label_he": "נהג בן 35 · 12 שנות ותק · היעדר תביעות 3 שנים · שנתון 2025",
}

TWO_WHEELER_KEYWORDS = {
    "ADV750", "CRUISYM", "JOYRIDES125", "JOYRIDES200", "LFA3", "N.DOWNTOWN",
    "TE 250", "TMAX560", "X-MAX 300", "TMAX", "XMAX", "DOWNTOWN", "FORZA",
    "PCX", "BEVERLY", "MEDLEY", "NMAX", "CYGNUS", "SUPER DINK", "AGILITY",
    "DUKE", "VESPA", "סאן יאנג", "קימקו", "ימהה", "קוואסאקי", "פיאג'ו",
    "דוקאטי", "הוסקוורנה", "קטנוע", "אופנוע", "דו-גלגלי", "דו/תלת גלגלי",
}

MODEL_ALIASES = {
    "קורולה": "COROLLA",
    "טוסון": "TUCSON",
    "ספורטאז": "SPORTAGE",
    "ספורטז": "SPORTAGE",
    "איוניק": "IONIQ",
    "פיקנטו": "PICANTO",
    "אוקטביה": "OCTAVIA",
    "ראב": "RAV 4",
    "ראב 4": "RAV 4",
    "נירו": "NIRO",
    "ברלינגו": "BERLINGO",
    "אאוטלנדר": "OUTLANDER",
    "לנד קרוזר": "LAND CRUISER",
    "יאריס": "YARIS",
    "סוויפט": "SWIFT",
    "סיוויק": "CIVIC",
    "גולף": "GOLF",
    "פולו": "POLO",
    "איביזה": "IBIZA",
    "טיגואן": "TIGUAN",
    "קונה": "KONA",
    "אלנטרה": "ELANTRA",
    "אקסנט": "ACCENT",
    "סטוניק": "STONIC",
    "סורנטו": "SORENTO",
    "מגאן": "MEGAN",
    "קשקאי": "QASHQAI",
    "מאזדה 3": "MAZDA 3",
    "מאזדה 2": "MAZDA 2",
    "CX5": "CX-5",
    "CX-5": "CX-5",
}

# Pre-fetched & calibrated Bestie reference base rates by model keyword (annual comprehensive + mandatory in NIS)
BESTIE_CALIBRATED_MODELS = {
    "LAND CRUISER": {"base_comp": 9800, "base_mand": 1580, "insurer": "הראל ביטוח", "alt_insurer": "הפניקס"},
    "RAV 4": {"base_comp": 6450, "base_mand": 1490, "insurer": "הפניקס Direct", "alt_insurer": "ביטוח ישיר"},
    "COROLLA": {"base_comp": 5650, "base_mand": 1460, "insurer": "ביטוח ישיר", "alt_insurer": "ליברה"},
    "TUCSON": {"base_comp": 5400, "base_mand": 1480, "insurer": "הפניקס Direct", "alt_insurer": "כלל ביטוח"},
    "SPORTAGE": {"base_comp": 5250, "base_mand": 1480, "insurer": "ביטוח ישיר", "alt_insurer": "מגדל"},
    "IONIQ": {"base_comp": 5100, "base_mand": 1450, "insurer": "ליברה", "alt_insurer": "ביטוח ישיר"},
    "NIRO": {"base_comp": 4950, "base_mand": 1450, "insurer": "ליברה", "alt_insurer": "הפניקס Direct"},
    "PICANTO": {"base_comp": 3850, "base_mand": 1420, "insurer": "ליברה", "alt_insurer": "שומרה"},
    "OCTAVIA": {"base_comp": 5150, "base_mand": 1470, "insurer": "שומרה", "alt_insurer": "מנורה מבטחים"},
    "OUTLANDER": {"base_comp": 5550, "base_mand": 1510, "insurer": "הראל ביטוח", "alt_insurer": "כלל ביטוח"},
    "YARIS": {"base_comp": 4250, "base_mand": 1430, "insurer": "ליברה", "alt_insurer": "ביטוח ישיר"},
    "SWIFT": {"base_comp": 3650, "base_mand": 1410, "insurer": "ליברה", "alt_insurer": "AIG"},
    "QASHQAI": {"base_comp": 4150, "base_mand": 1460, "insurer": "AIG", "alt_insurer": "ליברה"},
    "KONA": {"base_comp": 4300, "base_mand": 1450, "insurer": "ליברה", "alt_insurer": "הפניקס Direct"},
    "STONIC": {"base_comp": 3950, "base_mand": 1430, "insurer": "ליברה", "alt_insurer": "שומרה"},
    "ELANTRA": {"base_comp": 4450, "base_mand": 1450, "insurer": "שומרה", "alt_insurer": "ביטוח ישיר"},
    "CIVIC": {"base_comp": 4550, "base_mand": 1460, "insurer": "AIG", "alt_insurer": "מנורה מבטחים"},
}

INSURERS_BY_TIER = [
    ("ליברה", "AIG ישראל"),
    ("ליברה", "שומרה"),
    ("ביטוח ישיר", "ליברה"),
    ("הפניקס Direct", "ביטוח ישיר"),
    ("הראל ביטוח", "הפניקס"),
]

MITIGATION_CATALOG = {
    "איתוראן פעיל כפול": {"risk_reduction_pct": 18, "comp_discount_pct": 12, "cost_nis": 650},
    "קודן מוגן + עלוקה": {"risk_reduction_pct": 10, "comp_discount_pct": 7, "cost_nis": 450},
    "מנעול הגה מכני": {"risk_reduction_pct": 8, "comp_discount_pct": 4, "cost_nis": 220},
    "חניה פרטית סגורה / מצלמות": {"risk_reduction_pct": 12, "comp_discount_pct": 8, "cost_nis": 0},
}


@lru_cache(maxsize=1)
def load_artifacts():
    with open(CARD_PATH, encoding="utf-8") as f:
        card = json.load(f)
    with open(DATA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    return model, card, data


@lru_cache(maxsize=1)
def load_final_scores() -> dict[str, Any]:
    path = FINAL_SCORES_PATH if os.path.exists(FINAL_SCORES_PATH) else V9_SCORES_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)["scores"]


load_v9_scores = load_final_scores


def _norm_heb(text: str) -> str:
    if not text:
        return ""
    s = str(text)
    s = s.replace("מאזדה", "מזדה").replace("טיוטה", "טויוטה").replace("ניסן", "ניסאן").replace("שקודה", "סקודה")
    s = re.sub(r'["\'״׳\-()]', " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def _is_two_wheeler(info: dict[str, Any] | None, key: str = "") -> bool:
    cat = (info or {}).get("category", "")
    disp = f"{key} {(info or {}).get('display_name', '')} {(info or {}).get('model_name', '')} {(info or {}).get('maker', '')}".upper()
    if "דו" in cat or "אופנוע" in cat or "קטנוע" in cat:
        return True
    for kw in TWO_WHEELER_KEYWORDS:
        if kw.upper() in disp:
            return True
    return False


def get_hebrew_display_name(model_key: str, info: dict[str, Any] | None = None) -> str:
    if info and info.get("name_hebrew"):
        return str(info["name_hebrew"]).strip()
    raw = (info or {}).get("display_name") or model_key or ""
    m = re.search(r"[A-Za-z]", raw)
    if m and m.start() > 0:
        heb = raw[: m.start()].strip()
        if heb:
            return heb
    cleaned = re.sub(r"[A-Za-z()]", "", raw).strip()
    return re.sub(r"\s+", " ", cleaned) or raw


def resolve_model_entry(query: str) -> tuple[str, dict[str, Any]]:
    _, _, data = load_artifacts()
    models = data.get("models", {})
    raw = (query or "").strip()

    if raw in models and not _is_two_wheeler(models[raw], raw):
        return raw, models[raw]

    norm_q = _norm_heb(raw)
    alias_eng = ""
    for heb, eng in MODEL_ALIASES.items():
        if _norm_heb(heb) in norm_q or norm_q in _norm_heb(heb):
            alias_eng = eng.upper()
            break

    candidates = []
    for k, info in models.items():
        if not info or _is_two_wheeler(info, k):
            continue
        disp = info.get("display_name") or k
        heb_name = info.get("name_hebrew") or ""
        maker = info.get("maker") or ""
        model_clean = info.get("model_name") or info.get("model_clean") or ""

        if norm_q and (
            _norm_heb(disp) == norm_q
            or _norm_heb(heb_name) == norm_q
            or _norm_heb(k) == norm_q
        ):
            return k, info

        matched = False
        if norm_q and (
            norm_q in _norm_heb(disp)
            or norm_q in _norm_heb(heb_name)
            or norm_q in _norm_heb(maker)
            or norm_q in _norm_heb(model_clean)
        ):
            matched = True
        elif alias_eng and (alias_eng in disp.upper() or alias_eng == model_clean.upper()):
            matched = True

        if matched:
            candidates.append((k, info))

    if candidates:
        candidates.sort(key=lambda item: item[1].get("total_thefts", 0), reverse=True)
        return candidates[0]

    # Cold-start fallback based on global/maker average
    fallback_key = next(
        (k for k, v in models.items() if not _is_two_wheeler(v, k)),
        next(iter(models.keys())),
    )
    return raw or fallback_key, {
        "display_name": raw or fallback_key,
        "name_hebrew": raw or fallback_key,
        "maker": "כללי",
        "model_clean": raw or fallback_key,
        "category": "פרטי",
        "risk_score": 48.0,
        "growth_rate": 0.0,
        "total_thefts": 120,
        "trend": "stable",
        "rank": "משוער",
        "is_cold_start": True,
    }


def resolve_city_entry(query: str) -> tuple[str, dict[str, Any]]:
    _, _, data = load_artifacts()
    locations = data.get("locations", {})
    raw = (query or "").strip()
    if raw in locations:
        return raw, locations[raw]

    norm_q = raw.replace("-", " ").strip()
    exact_matches = []
    partial_matches = []
    for k, info in locations.items():
        norm_k = k.replace("-", " ").strip()
        if norm_k == norm_q:
            exact_matches.append((k, info))
        elif norm_q in norm_k or (len(norm_k) >= 3 and norm_k in norm_q):
            partial_matches.append((k, info))

    if exact_matches:
        exact_matches.sort(key=lambda item: item[1].get("total_thefts", 0), reverse=True)
        return exact_matches[0]

    if partial_matches:
        partial_matches.sort(
            key=lambda item: (
                1 if "תל אביב" in item[0] and "תל אביב" in norm_q else 0,
                item[1].get("total_thefts", 0),
                len(item[0]),
            ),
            reverse=True,
        )
        return partial_matches[0]

    default_city = "תל אביב-יפו" if "תל אביב-יפו" in locations else next(iter(locations.keys()))
    return default_city, locations[default_city]


def get_vehicle_risk(model: str, city: str, model_year: int = 2022) -> dict[str, Any]:
    """
    Tool 1: Returns the grounded ML theft prediction (XGBoost Regressor) and combined
    risk profile for a given (model, city, model_year).
    """
    xgb_model, card, data = load_artifacts()
    model_key, m_info = resolve_model_entry(model)
    city_key, c_info = resolve_city_entry(city)

    m_clean = m_info.get("model_clean", model_key)
    c_unified = c_info.get("city_unified", city_key)
    model_location = data.get("model_location", {})
    scores = load_v9_scores()
    combo = (
        model_location.get(f"{m_clean}|{c_unified}")
        or model_location.get(f"{m_clean}|{city_key}")
        or scores.get(f"{m_clean}|{c_unified}")
        or scores.get(f"{m_clean}|{city_key}")
    )
    m_risk = float(m_info.get("risk_score", 50.0))
    raw_city_risk = c_info.get("city_risk_score")
    if raw_city_risk is None:
        raw_city_risk = c_info.get("risk_score", 50.0)
    c_risk = float(raw_city_risk)
    combined_score = int(round(min(100.0, max(10.0, m_risk * 0.50 + c_risk * 0.45 + 2.5))))
    score_source = (
        "Option B composite risk (50% model, 45% city with corridors, 5% macro)"
        if combo
        else "Option B composite risk (cold-start calculated)"
    )

    features = {
        "model_year": int(model_year),
        "growth_factor": float(m_info.get("growth_rate", 0.0)),
        "model_risk_score": float(m_info.get("risk_score", 50.0)),
        "detection_rate": float(c_info.get("detection_rate", 0.12)),
        "unsolved_unknown_rate": float(c_info.get("unsolved_unknown_rate", 0.85)),
        "lag_thefts": float(combo.get("actual", combo.get("observed_thefts_2024", 0.0))) if combo else 0.0,
        "city_detection_rate": float(c_info.get("detection_rate", 0.12)),
        "city_unsolved_rate": float(c_info.get("unsolved_unknown_rate", 0.85)),
        "city_cases_total": float(c_info.get("total_cases", 0)),
    }
    x = np.array([[features[f] for f in card["features"]]], dtype=float)
    live_xgb_pred = float(max(0.0, xgb_model.predict(x)[0]))
    oof_annual_pred = float(combo.get("count", combo.get("estimated_thefts_2025", live_xgb_pred))) if combo else round(live_xgb_pred, 2)

    if combined_score >= 75:
        risk_level_he = "גבוהה מאוד"
    elif combined_score >= 50:
        risk_level_he = "גבוהה"
    elif combined_score >= 30:
        risk_level_he = "בינונית"
    else:
        risk_level_he = "נמוכה"

    return {
        "model_key": model_key,
        "model_display": m_info.get("display_name", model_key),
        "model_hebrew": get_hebrew_display_name(model_key, m_info),
        "maker": m_info.get("maker", ""),
        "category": m_info.get("category", "פרטי"),
        "city": city_key,
        "model_year": model_year,
        "combined_risk_score": combined_score,
        "risk_level_he": risk_level_he,
        "risk_score_source": score_source,
        "total_model_thefts": int(m_info.get("total_thefts", 0)),
        "total_city_thefts": int(c_info.get("total_thefts", c_info.get("total_cases", 0))),
        "city_detection_rate_pct": round(float(c_info.get("detection_rate", 0.12)) * 100, 1),
        "xgb_predicted_annual_thefts": round(oof_annual_pred, 1),
        "coverage_status": combo.get("coverage", "observed 2024 model/city count available") if combo else "cold-start: no observed 2024 model/city count",
        "model_version": "final",
        "trend": m_info.get("trend", "stable"),
        "rank": m_info.get("rank", "-"),
    }


def fetch_bestie_insurance_quotes(
    model: str,
    city: str,
    risk_score: float | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    """
    Tool 2: Hybrid Bestie Insurance Aggregator Quote Engine.
    Computes realistic Bestie market quotes (comprehensive + mandatory + insurer +
    protection requirements) for the standard reference driver profile.
    """
    risk_data = get_vehicle_risk(model, city)
    eff_risk = float(risk_score) if risk_score is not None else float(risk_data["combined_risk_score"])
    eff_cat = category or risk_data["category"]
    disp_up = risk_data["model_display"].upper()

    # Match calibrated Bestie base or derive from category + risk
    matched_cal = None
    for kw, cal in BESTIE_CALIBRATED_MODELS.items():
        if kw in disp_up:
            matched_cal = cal
            break

    cat_base_map = {
        "ג'יפונים (SUV)": 4400,
        "פנאי / שטח": 4600,
        "משפחתיות": 3900,
        "חשמליות": 4500,
        "מיני / סופר-מיני": 3350,
        "מנהלים / יוקרה": 5600,
        "מסחרי": 4900,
        "פרטי": 3850,
    }
    base_comp = matched_cal["base_comp"] if matched_cal else cat_base_map.get(eff_cat, 3900)
    base_mand = matched_cal["base_mand"] if matched_cal else 1450

    # Reference-driver simulation. No insurer, Bestie, or municipal pricing feed is queried.
    risk_factor = 0.72 + (eff_risk / 100.0) * 0.68
    if matched_cal:
        # Calibrated models already encode model popularity; adjust smoothly for city & risk delta
        comp_nis = int(round((base_comp * (0.82 + (eff_risk / 100.0) * 0.36)) / 50.0) * 50)
    else:
        comp_nis = int(round((base_comp * risk_factor) / 50.0) * 50)

    mand_nis = int(round((base_mand + (eff_risk * 1.8)) / 10.0) * 10)
    total_nis = comp_nis + mand_nis

    tier_idx = min(4, max(0, int(eff_risk // 20)))
    default_insurer, default_alt = INSURERS_BY_TIER[tier_idx]
    leading_insurer = matched_cal["insurer"] if matched_cal else default_insurer
    second_insurer = matched_cal["alt_insurer"] if matched_cal else default_alt

    if eff_risk >= 75:
        protection = "איתוראן פעיל כפול + קודן + עלוקה"
    elif eff_risk >= 55:
        protection = "איתוראן פעיל (או מערכת איתור) + קודן"
    elif eff_risk >= 35:
        protection = "קודן תקני + אימובילייזר מקורי"
    else:
        protection = "אימובילייזר מקורי בלבד (ללא דרישת איתוראן)"

    return {
        "model_key": risk_data["model_key"],
        "model_display": risk_data["model_display"],
        "model_hebrew": risk_data["model_hebrew"],
        "category": eff_cat,
        "city": risk_data["city"],
        "risk_score": int(round(eff_risk)),
        "risk_level_he": risk_data["risk_level_he"],
        "total_model_thefts": risk_data["total_model_thefts"],
        "xgb_predicted_annual_thefts": risk_data["xgb_predicted_annual_thefts"],
        "comprehensive_nis": comp_nis,
        "mandatory_nis": mand_nis,
        "total_annual_nis": total_nis,
        "leading_insurer": leading_insurer,
        "second_insurer": second_insurer,
        "required_protection": protection,
        "driver_profile": REFERENCE_DRIVER_PROFILE["label_he"],
        "bestie_source": "Research pricing simulation (not live Bestie quotes)",
        "bestie_url": "https://www.bestie.co.il/car-insurance",
    }


def compare_alternatives(model: str, city: str, top_k: int = 3) -> dict[str, Any]:
    """
    Tool 3: Finds top_k alternative 4-wheel vehicles in the same category with lower
    theft risk and lower Bestie insurance quotes in the selected city.
    """
    _, _, data = load_artifacts()
    base_risk = get_vehicle_risk(model, city)
    base_quote = fetch_bestie_insurance_quotes(
        base_risk["model_key"],
        base_risk["city"],
        risk_score=base_risk["combined_risk_score"],
        category=base_risk["category"],
    )

    target_cat = base_risk["category"]
    base_score = base_risk["combined_risk_score"]
    base_total_nis = base_quote["total_annual_nis"]
    base_heb = base_risk["model_hebrew"]

    # Collect canonical 4-wheel candidates in the same category
    seen_heb = {base_heb}
    pool = []
    for k, info in data.get("models", {}).items():
        if not info or _is_two_wheeler(info, k):
            continue
        disp = info.get("display_name") or k
        if not disp or "- (-)" in disp or "כללי" in disp:
            continue
        heb = get_hebrew_display_name(k, info)
        if heb in seen_heb:
            continue
        cat = info.get("category", "פרטי")
        thefts = int(info.get("total_thefts", 0))
        # Ensure we pick recognizable mainstream vehicles (with real road presence)
        if thefts < 25:
            continue
        pool.append((k, info, cat))

    same_cat = [item for item in pool if item[2] == target_cat]
    if len(same_cat) < top_k + 2:
        same_cat.extend([item for item in pool if item[2] != target_cat])

    scored_candidates = []
    for k, info, cat in same_cat:
        heb = get_hebrew_display_name(k, info)
        if heb in seen_heb:
            continue
        cand_risk = get_vehicle_risk(k, base_risk["city"])
        cand_score = cand_risk["combined_risk_score"]
        cand_quote = fetch_bestie_insurance_quotes(
            k,
            base_risk["city"],
            risk_score=cand_score,
            category=cat,
        )
        # Guarantee that alternatives shown to the user have lower risk & lower insurance cost
        # when the selected vehicle has medium/high risk
        savings_nis = base_total_nis - cand_quote["total_annual_nis"]
        risk_diff = base_score - cand_score
        if risk_diff > 0 and savings_nis > 0:
            seen_heb.add(heb)
            scored_candidates.append((savings_nis, risk_diff, cand_risk, cand_quote))

    # Sort by meaningful savings and mainstream relevance (prefer 20-45 risk score sweet spot)
    scored_candidates.sort(
        key=lambda x: (
            1 if 18 <= x[2]["combined_risk_score"] <= 48 else 0,
            x[0],
            x[1],
        ),
        reverse=True,
    )

    # If selected vehicle is already low-risk or few candidates met both criteria,
    # find peers with lower or equal theft risk (never suggest higher-risk vehicles)
    if len(scored_candidates) < top_k:
        # Sort remaining pool by lowest risk first
        remaining = []
        seen_cand_heb = set()
        for k, info, cat in same_cat:
            heb = get_hebrew_display_name(k, info)
            if heb in seen_heb or heb in seen_cand_heb:
                continue
            cand_risk = get_vehicle_risk(k, base_risk["city"])
            cand_score = cand_risk["combined_risk_score"]
            if cand_score <= base_score:
                remaining.append((cand_score, k, info, cat, cand_risk))
                seen_cand_heb.add(heb)
        remaining.sort(key=lambda item: item[0])

        for cand_score, k, info, cat, cand_risk in remaining:
            heb = get_hebrew_display_name(k, info)
            if heb in seen_heb:
                continue
            seen_heb.add(heb)
            cand_quote = fetch_bestie_insurance_quotes(
                k,
                base_risk["city"],
                risk_score=cand_score,
                category=cat,
            )
            savings_nis = base_total_nis - cand_quote["total_annual_nis"]
            risk_diff = max(0, base_score - cand_score)
            if savings_nis <= 0:
                # Honest reporting: Lower risk does not guarantee cheaper insurance across different segments
                savings_nis = 0
            scored_candidates.append((savings_nis, risk_diff, cand_risk, cand_quote))
            if len(scored_candidates) >= top_k:
                break

    alternatives = []
    for savings_nis, risk_diff, c_risk, c_quote in scored_candidates[:top_k]:
        reason = (
            f"ציון סיכון נמוך ב-{risk_diff} נק׳ ({c_risk['combined_risk_score']}% לעומת {base_score}%), "
            f"דרישת מיגון מופחתת ({c_quote['required_protection']}) ב-{base_risk['city']}."
        )
        alternatives.append({
            **c_quote,
            "annual_savings_nis": int(savings_nis),
            "monthly_savings_nis": int(round(savings_nis / 12.0)),
            "risk_reduction_pts": int(risk_diff),
            "agent_reason": reason,
        })

    max_savings = max((alt["annual_savings_nis"] for alt in alternatives), default=0)
    return {
        "selected_vehicle": {
            **base_quote,
            "is_selected": True,
            "annual_savings_nis": 0,
            "agent_reason": (
                f"הרכב שנבחר לבדיקה ב-{base_risk['city']} · ציון סיכון {base_score}% "
                f"({base_risk['total_model_thefts']:,} גניבות רשומות לדגם)."
            ),
        },
        "alternatives": alternatives,
        "max_annual_savings_nis": max_savings,
        "category": target_cat,
        "city": base_risk["city"],
        "driver_profile": REFERENCE_DRIVER_PROFILE["label_he"],
    }


def simulate_mitigation_discount(
    model: str,
    city: str,
    measures: list[str] | None = None,
) -> dict[str, Any]:
    """
    Tool 4: Calculates actuarial risk reduction and estimated Bestie comprehensive
    insurance discount when installing anti-theft protection measures.
    """
    quote = fetch_bestie_insurance_quotes(model, city)
    chosen = measures or list(MITIGATION_CATALOG.keys())[:3]

    total_risk_red_pct = 0
    total_comp_disc_pct = 0
    applied_details = []

    for m in chosen:
        spec = MITIGATION_CATALOG.get(m)
        if not spec:
            # Fuzzy match
            for k, v in MITIGATION_CATALOG.items():
                if m in k or k in m:
                    spec = v
                    m = k
                    break
        if spec:
            total_risk_red_pct += spec["risk_reduction_pct"]
            total_comp_disc_pct += spec["comp_discount_pct"]
            applied_details.append({"measure": m, **spec})

    total_risk_red_pct = min(40, total_risk_red_pct)
    total_comp_disc_pct = min(25, total_comp_disc_pct)

    new_risk = max(10, int(round(quote["risk_score"] * (1.0 - total_risk_red_pct / 100.0))))
    annual_discount_nis = int(round((quote["comprehensive_nis"] * (total_comp_disc_pct / 100.0)) / 20.0) * 20)
    new_total_nis = quote["total_annual_nis"] - annual_discount_nis

    return {
        "model_hebrew": quote["model_hebrew"],
        "city": quote["city"],
        "original_risk_score": quote["risk_score"],
        "mitigated_risk_score": new_risk,
        "original_total_nis": quote["total_annual_nis"],
        "mitigated_total_nis": new_total_nis,
        "annual_discount_nis": annual_discount_nis,
        "discount_pct": total_comp_disc_pct,
        "applied_measures": applied_details,
    }
