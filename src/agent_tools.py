"""
agent_tools.py
==============
Deterministic research toolset for the CarTheftRisk AI Agent (v9).
Theft estimates use `web/assets/v9_scores.json` and the v9 XGBoost artifact.
Insurance prices are a reference-driver simulation, not live insurer quotes.
"""
from __future__ import annotations

import difflib
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
    "RAV 4": {"base_comp": 6450, "base_mand": 1490, "insurer": "הפניקס", "alt_insurer": "כלל ביטוח"},
    "COROLLA": {"base_comp": 5650, "base_mand": 1460, "insurer": "הפניקס", "alt_insurer": "weSure"},
    "TUCSON": {"base_comp": 5400, "base_mand": 1480, "insurer": "איילון ביטוח", "alt_insurer": "כלל ביטוח"},
    "SPORTAGE": {"base_comp": 5250, "base_mand": 1480, "insurer": "מגדל", "alt_insurer": "הכשרה ביטוח"},
    "IONIQ": {"base_comp": 5100, "base_mand": 1450, "insurer": "weSure", "alt_insurer": "מנורה מבטחים"},
    "NIRO": {"base_comp": 4950, "base_mand": 1450, "insurer": "weSure", "alt_insurer": "שלמה ביטוח"},
    "PICANTO": {"base_comp": 3850, "base_mand": 1420, "insurer": "שלמה ביטוח", "alt_insurer": "שומרה"},
    "OCTAVIA": {"base_comp": 5150, "base_mand": 1470, "insurer": "שומרה", "alt_insurer": "מנורה מבטחים"},
    "OUTLANDER": {"base_comp": 5550, "base_mand": 1510, "insurer": "הראל ביטוח", "alt_insurer": "איילון ביטוח"},
    "YARIS": {"base_comp": 4250, "base_mand": 1430, "insurer": "הכשרה ביטוח", "alt_insurer": "שומרה"},
    "SWIFT": {"base_comp": 3650, "base_mand": 1410, "insurer": "weSure", "alt_insurer": "איילון ביטוח"},
    "QASHQAI": {"base_comp": 4150, "base_mand": 1460, "insurer": "שלמה ביטוח", "alt_insurer": "הכשרה ביטוח"},
    "KONA": {"base_comp": 4300, "base_mand": 1450, "insurer": "weSure", "alt_insurer": "איילון ביטוח"},
    "STONIC": {"base_comp": 3950, "base_mand": 1430, "insurer": "הכשרה ביטוח", "alt_insurer": "שומרה"},
    "ELANTRA": {"base_comp": 4450, "base_mand": 1450, "insurer": "שומרה", "alt_insurer": "שלמה ביטוח"},
    "CIVIC": {"base_comp": 4550, "base_mand": 1460, "insurer": "איילון ביטוח", "alt_insurer": "מנורה מבטחים"},
}

INSURERS_BY_TIER = [
    ("weSure", "שלמה ביטוח"),
    ("איילון ביטוח", "שומרה"),
    ("הכשרה ביטוח", "מגדל"),
    ("הפניקס", "כלל ביטוח"),
    ("הראל ביטוח", "מנורה מבטחים"),
]

CITY_ALIASES = {
    "תל אביב": "תל אביב-יפו",
    "תל אביב יפו": "תל אביב-יפו",
    'ת"א': "תל אביב-יפו",
    "תא": "תל אביב-יפו",
    'ראשל"צ': "ראשון לציון",
    "ראשלצ": "ראשון לציון",
    'פ"ת': "פתח תקווה",
    "פתח תקוה": "פתח תקווה",
    'ב"ש': "באר שבע",
    'כפ"ס': "כפר סבא",
}

MITIGATION_CATALOG = {
    "איתוראן פעיל כפול": {"risk_reduction_pct": 18, "comp_discount_pct": 12, "cost_nis": 650},
    "קודן מוגן + עלוקה": {"risk_reduction_pct": 10, "comp_discount_pct": 7, "cost_nis": 450},
    "מנעול הגה מכני": {"risk_reduction_pct": 8, "comp_discount_pct": 4, "cost_nis": 220},
    "חניה פרטית סגורה / מצלמות": {"risk_reduction_pct": 12, "comp_discount_pct": 8, "cost_nis": 0},
}

MITIGATION_SYNONYMS: dict[str, tuple[str, ...]] = {
    "איתוראן פעיל כפול": ("איתוראן", "איתורן", "מערכת איתור", "פוינטר", "ituran", "pointer", "gps"),
    "קודן מוגן + עלוקה": ("קודן", "עלוקה", "משבת", "אימובילייזר", "kodan"),
    "מנעול הגה מכני": ("מנעול הגה", "מנעול", "סנדל"),
    "חניה פרטית סגורה / מצלמות": ("חניה", "חנייה", "מצלמ", "חניון סגור", "שער חשמלי"),
}

_HEB_PREFIX_WHITELIST_PROTECTED = {
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
_FILLER_MODEL_WORDS = {"רכב", "הרכב", "דגם", "מכונית", "אוטו"}


def extract_mitigation_measures(text: str) -> list[str]:
    """Deterministically extracts canonical MITIGATION_CATALOG measure keys mentioned in free text."""
    if not text:
        return []
    low = re.sub(r"\s+", " ", text.lower()).strip()
    found: list[str] = []
    for canonical_name in MITIGATION_CATALOG:
        synonyms = MITIGATION_SYNONYMS.get(canonical_name, ())
        if canonical_name.lower() in low or any(syn in low for syn in synonyms):
            if canonical_name not in found:
                found.append(canonical_name)
    return found


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


def _prefer_canonical_model(key: str, models: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    entry = models[key]
    disp = entry.get("display_name")
    if disp and disp in models and not _is_two_wheeler(models[disp], disp):
        if int(models[disp].get("total_thefts", 0)) > int(entry.get("total_thefts", 0)):
            return disp, models[disp]
    return key, entry


def _model_query_variants(norm_q: str) -> list[str]:
    if not norm_q:
        return []
    variants = [norm_q]
    tokens = norm_q.split()
    first = tokens[0]
    if len(first) >= 4 and first[0] in "בלמהו" and first not in _HEB_PREFIX_WHITELIST_PROTECTED:
        stripped = " ".join([first[1:], *tokens[1:]]).strip()
        if stripped and stripped not in variants:
            variants.append(stripped)
    return variants


def resolve_model_entry(query: str, strict: bool = False) -> tuple[str, dict[str, Any]] | None:
    """
    Resolves a vehicle model name to a `(model_key, info)` entry from `data.json`.
    When `strict=True`, returns `None` instead of silently falling back to a default/cold-start model.
    """
    _, _, data = load_artifacts()
    models = data.get("models", {})
    raw = (query or "").strip()

    if strict and not raw:
        return None

    if raw in models and not _is_two_wheeler(models[raw], raw) and (not strict or "|" not in raw):
        return _prefer_canonical_model(raw, models)

    norm_q = _norm_heb(raw)
    if strict:
        if not norm_q:
            return None

        sorted_aliases = sorted(MODEL_ALIASES.items(), key=lambda kv: len(_norm_heb(kv[0])), reverse=True)
        for q_cand in _model_query_variants(norm_q):
            q_tokens_all = q_cand.split()
            # 1. Exact normalized match across canonical non-pipe 4-wheel entries
            exact_hits: list[tuple[str, dict[str, Any]]] = []
            for k, info in models.items():
                if not info or "|" in k or _is_two_wheeler(info, k):
                    continue
                disp = info.get("display_name") or k
                if not disp or "- (-)" in disp or "כללי" in disp:
                    continue
                heb_name = info.get("name_hebrew") or ""
                model_clean = info.get("model_name") or info.get("model_clean") or ""
                if (
                    _norm_heb(disp) == q_cand
                    or _norm_heb(heb_name) == q_cand
                    or _norm_heb(k) == q_cand
                    or (len(q_cand) >= 2 and _norm_heb(model_clean) == q_cand)
                ):
                    exact_hits.append((k, info))
            if exact_hits:
                exact_hits.sort(key=lambda item: int(item[1].get("total_thefts", 0)), reverse=True)
                return _prefer_canonical_model(exact_hits[0][0], models)

            # 2. Alias token-boundary match (all remaining non-filler tokens must also match candidate)
            for alias_heb, alias_eng in sorted_aliases:
                norm_alias = _norm_heb(alias_heb)
                if not norm_alias:
                    continue
                if re.search(rf"(?:^|\s){re.escape(norm_alias)}(?:$|\s)", q_cand):
                    alias_tok_set = set(norm_alias.split())
                    remaining = [t for t in q_tokens_all if t not in alias_tok_set and t not in _FILLER_MODEL_WORDS]
                    alias_hits: list[tuple[str, dict[str, Any]]] = []
                    for k, info in models.items():
                        if not info or "|" in k or _is_two_wheeler(info, k):
                            continue
                        disp = info.get("display_name") or k
                        if not disp or "- (-)" in disp or "כללי" in disp:
                            continue
                        heb_name = info.get("name_hebrew") or ""
                        maker = info.get("maker") or ""
                        model_clean = info.get("model_name") or info.get("model_clean") or ""
                        if not (alias_eng.upper() in disp.upper() or alias_eng.upper() == model_clean.upper()):
                            continue
                        cand_tokens = set(_norm_heb(f"{disp} {heb_name} {maker} {model_clean}").split())
                        if all(t in cand_tokens for t in remaining):
                            alias_hits.append((k, info))
                    if alias_hits:
                        alias_hits.sort(key=lambda item: int(item[1].get("total_thefts", 0)), reverse=True)
                        return _prefer_canonical_model(alias_hits[0][0], models)

            # 3. Full-token coverage match (must cover all significant tokens + at least one non-maker token)
            sig_tokens = [t for t in q_tokens_all if (len(t) >= 2 or t.isdigit()) and t not in _FILLER_MODEL_WORDS]
            if sig_tokens:
                token_hits: list[tuple[str, dict[str, Any]]] = []
                for k, info in models.items():
                    if not info or "|" in k or _is_two_wheeler(info, k):
                        continue
                    disp = info.get("display_name") or k
                    if not disp or "- (-)" in disp or "כללי" in disp:
                        continue
                    heb_name = info.get("name_hebrew") or ""
                    maker = info.get("maker") or ""
                    model_clean = info.get("model_name") or info.get("model_clean") or ""
                    cand_tokens = set(_norm_heb(f"{disp} {heb_name} {maker} {model_clean}").split())
                    maker_tokens = set(_norm_heb(maker).split())
                    model_only_tokens = (cand_tokens - maker_tokens) | set(_norm_heb(model_clean).split())
                    if all(t in cand_tokens for t in sig_tokens) and any(t in model_only_tokens for t in sig_tokens):
                        token_hits.append((k, info))
                if token_hits:
                    token_hits.sort(key=lambda item: int(item[1].get("total_thefts", 0)), reverse=True)
                    return _prefer_canonical_model(token_hits[0][0], models)

        return None

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
            return _prefer_canonical_model(k, models)

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
        return _prefer_canonical_model(candidates[0][0], models)

    # Cold-start fallback based on global/maker average (non-strict mode only)
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


def _norm_city_name(text: str) -> str:
    s = re.sub(r"[-–]+", " ", text or "")
    return re.sub(r"\s+", " ", s).strip()


def _city_query_variants(raw: str) -> list[str]:
    clean = re.sub(r"\s+", " ", (raw or "").strip())
    if not clean:
        return []
    variants = [clean]
    tokens = clean.split()
    first = tokens[0]
    if len(first) >= 4 and first[0] in "בלמהו":
        stripped = " ".join([first[1:], *tokens[1:]]).strip()
        if stripped and stripped not in variants:
            variants.append(stripped)
    return variants


def resolve_city_entry(query: str, strict: bool = False) -> tuple[str, dict[str, Any]] | None:
    """
    Resolves a municipality/city name to a `(city_key, info)` entry from `data.json`.
    When `strict=True`, returns `None` instead of falling back to Tel Aviv-Yafo.
    """
    _, _, data = load_artifacts()
    locations = data.get("locations", {})
    raw = (query or "").strip()

    if strict and not raw:
        return None

    if raw in locations:
        return raw, locations[raw]

    for cand in _city_query_variants(raw):
        norm_c = _norm_city_name(cand)
        alias_target = CITY_ALIASES.get(cand) or CITY_ALIASES.get(norm_c)
        if alias_target and alias_target in locations:
            return alias_target, locations[alias_target]

        exact_matches = [
            (k, info)
            for k, info in locations.items()
            if _norm_city_name(k) == norm_c
        ]
        if exact_matches:
            exact_matches.sort(key=lambda item: int(item[1].get("total_thefts", 0)), reverse=True)
            return exact_matches[0]

        if strict and len(norm_c) >= 4:
            prefix_matches = [
                (k, info)
                for k, info in locations.items()
                if _norm_city_name(k).startswith(norm_c + " ")
            ]
            if prefix_matches:
                prefix_matches.sort(key=lambda item: int(item[1].get("total_thefts", 0)), reverse=True)
                return prefix_matches[0]

    if strict:
        return None

    norm_q = _norm_city_name(raw)
    partial_matches = []
    for k, info in locations.items():
        norm_k = _norm_city_name(k)
        if norm_q and (norm_q in norm_k or (len(norm_k) >= 3 and norm_k in norm_q)):
            partial_matches.append((k, info))

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


def suggest_similar_models(query: str, limit: int = 5) -> list[str]:
    """Returns canonical Hebrew vehicle model names closest to `query` or top mainstream models."""
    _, _, data = load_artifacts()
    models = data.get("models", {})
    by_heb: dict[str, int] = {}
    for k, info in models.items():
        if not info or "|" in k or _is_two_wheeler(info, k):
            continue
        disp = info.get("display_name") or k
        if not disp or "- (-)" in disp or "כללי" in disp:
            continue
        heb = get_hebrew_display_name(k, info)
        if not heb:
            continue
        thefts = int(info.get("total_thefts", 0))
        if thefts > by_heb.get(heb, -1):
            by_heb[heb] = thefts

    ranked_names = [heb for heb, _ in sorted(by_heb.items(), key=lambda item: item[1], reverse=True)]
    norm_q = _norm_heb(query)
    suggestions: list[str] = []

    if norm_q:
        norm_to_heb = {_norm_heb(name): name for name in ranked_names}
        for tok in norm_q.split():
            if len(tok) < 2 or tok in _FILLER_MODEL_WORDS:
                continue
            for name in ranked_names:
                if tok in _norm_heb(name) and name not in suggestions:
                    suggestions.append(name)
                    if len(suggestions) >= limit:
                        return suggestions
        for close_norm in difflib.get_close_matches(norm_q, list(norm_to_heb.keys()), n=limit, cutoff=0.45):
            name = norm_to_heb[close_norm]
            if name not in suggestions:
                suggestions.append(name)
                if len(suggestions) >= limit:
                    return suggestions

    for name in ranked_names:
        if name not in suggestions:
            suggestions.append(name)
            if len(suggestions) >= limit:
                break
    return suggestions


def suggest_similar_cities(query: str, limit: int = 5) -> list[str]:
    """Returns verified Israeli municipality names closest to `query` or top cities by theft count."""
    _, _, data = load_artifacts()
    locations = data.get("locations", {})
    ranked_cities = [
        city
        for city, _ in sorted(
            locations.items(),
            key=lambda item: int(item[1].get("total_thefts", 0)),
            reverse=True,
        )
    ]
    suggestions: list[str] = []
    clean_q = _norm_city_name(query)
    if clean_q:
        for close in difflib.get_close_matches(clean_q, ranked_cities, n=limit, cutoff=0.45):
            if close not in suggestions:
                suggestions.append(close)
                if len(suggestions) >= limit:
                    return suggestions

    for city in ranked_cities:
        if city not in suggestions:
            suggestions.append(city)
            if len(suggestions) >= limit:
                break
    return suggestions


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
        "SUV / פנאי": 4400,
        "פנאי / שטח": 4600,
        "SUV / שטח": 4700,
        "היברידי / פנאי": 4300,
        "משפחתיות": 3900,
        "משפחתי": 3900,
        "חשמליות": 4500,
        "מיני / סופר-מיני": 3350,
        "סופרמיני / עירוני": 3350,
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
        "bestie_url": "https://bestie.co.il/car_insurance/",
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
    seen_measures: set[str] = set()

    for m in chosen:
        canonical = m
        spec = MITIGATION_CATALOG.get(canonical)
        if not spec:
            # Fuzzy match
            for k, v in MITIGATION_CATALOG.items():
                if m in k or k in m:
                    spec = v
                    canonical = k
                    break
        if spec and canonical not in seen_measures:
            seen_measures.add(canonical)
            total_risk_red_pct += spec["risk_reduction_pct"]
            total_comp_disc_pct += spec["comp_discount_pct"]
            applied_details.append({"measure": canonical, **spec})

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


def calculate_custom_savings(
    base_model: str,
    base_city: str,
    target_model: str | None = None,
    target_city: str | None = None,
    measures: list[str] | None = None,
) -> dict[str, Any]:
    """
    Tool 5: Deterministically calculates exact risk and annual/monthly Bestie simulation deltas
    between a base (model, city) and a custom target (model, city, and/or mitigation measures).
    Never clamps negative savings to zero so cost increases are reported transparently.
    """
    resolved_base_model = resolve_model_entry(base_model, strict=True)
    if resolved_base_model is None:
        raise ValueError(f"Unknown base model: {base_model}")
    resolved_base_city = resolve_city_entry(base_city, strict=True)
    if resolved_base_city is None:
        raise ValueError(f"Unknown base city: {base_city}")

    base_model_key, _ = resolved_base_model
    base_city_key, _ = resolved_base_city

    if target_model is not None:
        resolved_target_model = resolve_model_entry(target_model, strict=True)
        if resolved_target_model is None:
            raise ValueError(f"Unknown target model: {target_model}")
        target_model_key = resolved_target_model[0]
    else:
        target_model_key = base_model_key

    if target_city is not None:
        resolved_target_city = resolve_city_entry(target_city, strict=True)
        if resolved_target_city is None:
            raise ValueError(f"Unknown target city: {target_city}")
        target_city_key = resolved_target_city[0]
    else:
        target_city_key = base_city_key

    base_risk = get_vehicle_risk(base_model_key, base_city_key)
    base_quote = fetch_bestie_insurance_quotes(
        base_model_key,
        base_city_key,
        risk_score=base_risk["combined_risk_score"],
        category=base_risk["category"],
    )

    target_risk = get_vehicle_risk(target_model_key, target_city_key)
    target_quote = fetch_bestie_insurance_quotes(
        target_model_key,
        target_city_key,
        risk_score=target_risk["combined_risk_score"],
        category=target_risk["category"],
    )

    mitigation_data: dict[str, Any] | None = None
    if measures:
        mitigation_data = simulate_mitigation_discount(target_model_key, target_city_key, measures=measures)
        final_target_annual = int(mitigation_data["mitigated_total_nis"])
        final_target_risk = int(mitigation_data["mitigated_risk_score"])
    else:
        final_target_annual = int(target_quote["total_annual_nis"])
        final_target_risk = int(target_risk["combined_risk_score"])

    annual_savings = int(base_quote["total_annual_nis"]) - final_target_annual
    monthly_savings = int(round(annual_savings / 12.0))
    risk_delta_pts = int(base_risk["combined_risk_score"]) - final_target_risk

    base_obs = "observed" in str(base_risk.get("coverage_status", "")).lower()
    target_obs = "observed" in str(target_risk.get("coverage_status", "")).lower()
    if base_obs and target_obs:
        coverage_note = (
            f"קיימת תצפית 2024 מאומתת לשני השילובים ({base_risk['model_hebrew']} ב-{base_risk['city']}, "
            f"{target_risk['model_hebrew']} ב-{target_risk['city']})."
        )
    elif not base_obs and not target_obs:
        coverage_note = "אין תצפית 2024 ישירה לדגם–עיר לשני השילובים; האומדן מבוסס על שקלול cold-start ומודל XGBoost."
    else:
        coverage_note = (
            f"תצפית 2024 זמינה עבור {'הבסיס' if base_obs else 'היעד'} בלבד; "
            f"השילוב השני מבוסס על אומדן cold-start ומודל XGBoost."
        )

    metrics = {
        "base_model_he": base_risk["model_hebrew"],
        "base_city": base_risk["city"],
        "target_model_he": target_risk["model_hebrew"] if target_model is not None else None,
        "target_city": target_risk["city"] if target_city is not None else None,
        "base_annual_nis": int(base_quote["total_annual_nis"]),
        "target_annual_nis": final_target_annual,
        "annual_savings_nis": annual_savings,
        "monthly_savings_nis": monthly_savings,
        "base_risk_score": int(base_risk["combined_risk_score"]),
        "target_risk_score": final_target_risk,
        "risk_delta_pts": risk_delta_pts,
    }

    return {
        "metrics": metrics,
        "base_risk": base_risk,
        "base_quote": base_quote,
        "target_risk": target_risk,
        "target_quote": target_quote,
        "mitigation": mitigation_data,
        "data_coverage_note": coverage_note,
        "assumptions": dict(REFERENCE_DRIVER_PROFILE),
    }
