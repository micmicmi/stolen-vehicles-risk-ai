# מילון נתונים וייחוס (Data Dictionary & Lineage)

| נכס נתונים | רזולוציה (Grain) | כיסוי זמני | שימוש בפרויקט | תיאור |
| :--- | :--- | :--- | :--- | :--- |
| `data/raw/stolen_cars_models_2017_2023.csv` | דגם מצטבר | 2017–2023 | הקשר היסטורי ודעיכה מעריכית | 74 דגמים מובילים |
| `data/raw/stolen_cars_models_2023_2025_FULL.csv` | דגם × שנתון × עיר | 2023–2025 | אימון ובחינת רגרסיה | 39,331 שורות Pivot מקוריות (54,007 גניבות) |
| `data/processed/merged_clean.csv` | דגם-שנתון × עיר | 2023–2025 | מטריצת הלמידה המפוקחת | 24,519 רשומות ממופתחות ומסונכרנות |
| `web/assets/data.json` | מאגר Lookup מלא | 2017–2025 | שירות ממשק המשתמש (Client) ב-$O(1)$ | 8 מפתחות כולל `cities`, `models`, `locations`, `model_location` |

## מאפייני המודל המפוקח (Model Feature Contract)
* `model_year`: שנת ייצור הרכב.
* `growth_factor`: קצב צמיחה שנתי יחסי בביקוש לגניבות.
* `model_risk_score`: מדד סיכון אמפירי מנורמל לדגם (0–100).
* `detection_rate`: שיעור גילוי חשודים בתחנת המשטרה המרחבית.
* `unsolved_unknown_rate`: שיעור תיקים סגורים בעילת "עבריין לא נודע" (על"ן).

## נוסחת שקלול סיכון משולב (Option B - Ground Truth)
$$\text{Risk\_Score} = 0.50 \times \text{Model\_Risk} + 0.45 \times \text{City\_Risk\_with\_Corridors} + 0.05 \times \text{Macro\_Adjustment}$$
