# Project: Stolen Vehicles Risk Analyzer (CarTheftRisk)
> פרויקט גמר בקורס AI Engineering — מערכת לחיזוי ואפיון סיכוני גניבות רכב בישראל (2017–2025)

## Context & Ground Truth
- **מקור האמת הרשמי:** כל המדדים, המתודולוגיה והגבולות המתודולוגיים מנוהלים ב-`SINGLE_SOURCE_OF_TRUTH.md`.
- **מחברת המחקר הקנונית:** `notebooks/CarTheftRisk_final.ipynb` (58 תאים: 33 קוד + 25 מרקדאון, `0` שגיאות, ייצוא מבודד ל-`notebooks/data_final.json`).
  - **Track 1 (Count Champion):** `XGBoost Poisson (max_depth=3, Offset log(0.75))` על מבחן עיוור 2025 (`n=16,853`, 9 חודשים): `MAE = 0.4776`, `RMSE = 1.2898`, `R² = +0.1984`, `Poisson Deviance = 0.9690`, `Tail MAE = 8.6687`, `Spearman ρ = 0.4089`, נתח עשירון עליון `D10 = 58.18%`.
  - **Track 2 (Continuous 0–100 Champion):** `Random Forest Regressor` (תקרת אימון 2024 בלבד): `R² = +0.3157`, `MAE = 5.5951`.
  - **5 מאפייני ML נקיים מזליגה:** `['model_year', 'lag_1_thefts', 'city_lag_1_thefts', 'city_detection_rate', 'city_unsolved_rate']`.
- **חוזה הריצה המאוחד (Final Runtime Contract):** `CarTheftRisk_final.ipynb` + `src/temporal_validation.py` מזינים את `web/assets/data.json` ו-`web/assets/final_scores.json` עבור `web/index.html`, `src/api.py` וסוכן ה-AI (`src/agent.py`, `src/agent_schemas.py`, `src/agent_tools.py`), ומוגנים ב-32 בדיקות `pytest`. גרסה 9 הועברה רשמית לארכיון (`archive/notebooks/`).

## Standard Commands
```powershell
# 1. הרצת 32 בדיקות המערכת
python -X utf8 -m pytest -v

# 2. הפעלת שרת FastAPI
python -X utf8 -m uvicorn src.api:app --reload

# 3. הגשת אפליקציית ה-Web והדוחות
python -X utf8 -m http.server 8080 --directory web
```

## Hard Rules
1. אין לדרוס את `web/assets/data.json` או `web/assets/final_scores.json` מתוך ניסויי מחברת — `CarTheftRisk_final.ipynb` מייצאת אך ורק ל-`notebooks/data_final.json`.
2. בסביבת Windows יש להריץ פייתון עם `python -X utf8`.
3. חובה לשמור על הדיסקליימרים ב-`docs/RESEARCH_LIMITATIONS.md` ובממשק: ציון ה-UI (`50/45/5` ללא צירי מילוט) וסימולציית הביטוח אינם תעריף אקטוארי מסחרי.
