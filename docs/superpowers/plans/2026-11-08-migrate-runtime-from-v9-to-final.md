# Migrate Runtime Architecture from Legacy v9 to Canonical Final Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Completely eliminate the legacy `v9` runtime contract, unifying the entire project around `CarTheftRisk_final` as the sole canonical source of truth for both research and runtime (Web dashboard, FastAPI server, AI agent, model registry, and tests), and safely archiving `CarTheftRisk_v9.ipynb`.

**Architecture:** Transition the data generation script (`src/temporal_validation.py`), FastAPI API (`src/api.py`), AI agent tools (`src/agent_tools.py`), and web frontend (`web/index.html`) to declare and consume `final` runtime artifacts (`final_scores.json`, `temporal_backtest_final.json`, `model_version: "final"`). Update all 21 `pytest` tests to assert against the `final` contract, then archive `CarTheftRisk_v9.ipynb` to `archive/notebooks/`.

**Tech Stack:** Python 3.14, FastAPI, XGBoost, Scikit-Learn, PyTest, Vanilla JS / HTML5.

**Spec:** User request: "אני רוצה שתעביר את הכל, שינותבו מהקובץ הסופי final. ולא מגרסא 9 . תכין לי תוכנית לשינוי".

## Global Constraints
- Preserve zero-leakage temporal design: 2024 train -> 2025 blind test.
- Maintain 21/21 passing `pytest` tests throughout and after migration.
- No synthetic data generation; maintain exact counts and verified methodology.
- Ensure offline `file://` compatibility via `web/assets/embedded_data.js`.
- Windows environment: all python executions must use `python -X utf8`.

## Review Focus
1. Backward compatibility: Ensure `web/index.html` seamlessly resolves both `final_risk_score` and fallback fields without console errors.
2. API contract integrity: Ensure `/predict`, `/health`, and `/simulate_bestie` return valid `final` metadata and numeric estimates.
3. Test suite alignment: Ensure all 21 tests in `tests/` pass with 0 errors and 0 warnings against the `final` contract.
4. Model registry validation: Ensure `models/model_registry.json` declares `"active_runtime_model": "final"` and links to `CarTheftRisk_final.ipynb`.
5. Clean archive state: Ensure `CarTheftRisk_v9.ipynb` is moved cleanly to `archive/notebooks/` and documented in `ARCHIVE_STATUS.md`.

---

### Task 1: Unify Runtime Data Pipeline & Asset Generation (`src/temporal_validation.py`)

**Files:**
- Modify: `stolen-vehicles-project/src/temporal_validation.py`
- Test: `stolen-vehicles-project/tests/test_temporal_validation.py`

**Interfaces:**
- Produces: `models/temporal_backtest_final.json`, `web/assets/final_scores.json`, `web/assets/charts/temporal_backtest_final.png`, `web/assets/charts/temporal_actual_vs_pred_final.png`, updated `web/assets/data.json`, `web/assets/embedded_data.js`, and `web/assets/research_status.json`.
- Metadata contract: `pipeline_version: "CarTheftRisk_final"`, `runtime_model_version: "final"`, `runtime_scores_source: "web/assets/final_scores.json"`.

- [ ] **Step 1: Update artifact paths and metadata in `src/temporal_validation.py`**
  Update constants:
  - `OUTPUT = ROOT / "models" / "temporal_backtest_final.json"`
  - `SCORES_OUTPUT = ROOT / "web" / "assets" / "final_scores.json"`
  - `CHART = ROOT / "web" / "assets" / "charts" / "temporal_backtest_final.png"`
  - `SCATTER_CHART = ROOT / "web" / "assets" / "charts" / "temporal_actual_vs_pred_final.png"`
  Update model version strings from `"v9"` to `"final"`, add `"final_risk_score"` (and keep `"v9_risk_score"` alias in `scores` for graceful transition).
  Ensure `web_data["metadata"]` sets:
  - `"pipeline_version": "CarTheftRisk_final"`
  - `"runtime_model_version": "final"`
  - `"runtime_scores_source": "web/assets/final_scores.json"`
  - `"dataset_role": "historical vehicle/city lookup with final score overlay"`

- [ ] **Step 2: Run `temporal_validation.py` to generate final runtime assets**
  Run: `python -X utf8 stolen-vehicles-project/src/temporal_validation.py`
  Verify: `models/temporal_backtest_final.json` and `web/assets/final_scores.json` are created successfully.

- [ ] **Step 3: Update `tests/test_temporal_validation.py` to assert the final contract**
  Update test assertions to verify `temporal_backtest_final.json` and feature contract.

- [ ] **Step 4: Run pytest on `test_temporal_validation.py`**
  Run: `python -X utf8 -m pytest stolen-vehicles-project/tests/test_temporal_validation.py -v`
  Expected: PASS

---

### Task 2: Migrate FastAPI Endpoints & AI Agent Tools (`src/api.py`, `src/agent_tools.py`, `src/agent.py`)

**Files:**
- Modify: `stolen-vehicles-project/src/api.py`
- Modify: `stolen-vehicles-project/src/agent_tools.py`
- Modify: `stolen-vehicles-project/src/agent.py`
- Test: `stolen-vehicles-project/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `web/assets/final_scores.json`
- Produces: API response with `"model_version": "final"`, `"final_estimated_thefts_2025"`, `"final_score"`, and `"engine": "research_bestie_simulation_final"`.

- [ ] **Step 1: Update `src/api.py`**
  - Set `FINAL_SCORES_PATH = os.path.join(BASE_DIR, "web", "assets", "final_scores.json")` (fallback to `v9_scores.json`).
  - Update `_load_final_scores()`.
  - In `/health`: return `{"status": "ok", "model_version": "final", "pipeline_version": "final"}`.
  - In `/predict`: return `"final_estimated_thefts_2025": round(pred, 2)`, `"final_score": score`, `"model_version": "final"`, while preserving `"v9_estimated_thefts_2025"` and `"v9_score"` for backward compatibility.
  - In `/simulate_bestie`: set `engine: "research_bestie_simulation_final"`.

- [ ] **Step 2: Update `src/agent_tools.py` and `src/agent.py`**
  - Set `FINAL_SCORES_PATH` and `load_final_scores()`.
  - Support score retrieval via `final_risk_score` or `avg_risk`.
  - In `simulate_insurance_quote`: return `"model_version": "final"`, `"engine": "research_bestie_simulation_final"`.
  - In `src/agent.py`: update agent system prompt to reference `CarTheftRisk_final`.

- [ ] **Step 3: Update `tests/test_pipeline.py`**
  - Update `test_web_data_declares_the_final_runtime_contract` to check `pipeline_version == "CarTheftRisk_final"` and `runtime_model_version == "final"`.
  - Update `test_final_score_artifact_is_covered_and_bounded` to test `final_scores.json`.
  - Update API test assertions to verify `"model_version" == "final"` and engine `"research_bestie_simulation_final"`.

- [ ] **Step 4: Run pytest on `test_pipeline.py`**
  Run: `python -X utf8 -m pytest stolen-vehicles-project/tests/test_pipeline.py -v`
  Expected: PASS

---

### Task 3: Update Frontend Application (`web/index.html` & `embedded_data.js`)

**Files:**
- Modify: `stolen-vehicles-project/web/index.html`
- Modify: `stolen-vehicles-project/web/assets/embedded_data.js` (regenerated via `temporal_validation.py`)
- Test: `stolen-vehicles-project/tests/test_model_registry.py::test_frontend_applies_v9_scores`

- [ ] **Step 1: Update `web/index.html` score loader**
  - Rename `applyV9Scores()` to `applyFinalScores()`:
    - Fetches `assets/final_scores.json`.
    - Maps `score.final_risk_score || score.v9_risk_score`.
    - Sets `model_version: 'final'`.
    - Sets `DATA.metadata.runtime_model_version = 'final'`.
  - Update UI labels and tooltips:
    - Replace references from `ציון v9` to `ציון סיכון (Final)`.
    - Replace `research_bestie_simulation_v9` with `research_bestie_simulation_final`.

- [ ] **Step 2: Verify frontend score loading**
  Check that `web/assets/embedded_data.js` contains `metadata.runtime_model_version: "final"` and `model_version: "final"` on records.

---

### Task 4: Update Model Registry & Model Card (`models/`)

**Files:**
- Modify: `stolen-vehicles-project/models/model_registry.json`
- Modify: `stolen-vehicles-project/models/model_card.json`
- Test: `stolen-vehicles-project/tests/test_model_registry.py`

- [ ] **Step 1: Update `models/model_registry.json`**
  - Set `"active_runtime_model": "final"`.
  - Set `"temporal_validation_fix": "notebooks/CarTheftRisk_final.ipynb"`.
  - Add version entry for `"final"`:
    ```json
    "final": {
      "status": "validated_research_backtest",
      "artifact": "models/temporal_backtest_final.json",
      "pipeline": "CarTheftRisk_final",
      "evaluation": { ... }
    }
    ```
  - Mark `"v9"` in `"versions"` as `"historical_archived"`.

- [ ] **Step 2: Update `models/model_card.json`**
  - Set `"version": "final"`, `"pipeline": "CarTheftRisk_final"`.

- [ ] **Step 3: Update `tests/test_model_registry.py`**
  - Update tests to check `active_runtime_model == "final"`.
  - Update chart assertion to `temporal_backtest_final.png`.
  - Update `test_frontend_applies_final_scores` to check `applyFinalScores` and `assets/final_scores.json`.
  - Update `test_research_status_separates_runtime_and_backtest` to verify `"model_version" == "final"`.

- [ ] **Step 4: Run pytest on `test_model_registry.py`**
  Run: `python -X utf8 -m pytest stolen-vehicles-project/tests/test_model_registry.py -v`
  Expected: PASS

---

### Task 5: Refactor Runtime Contract Tests (`tests/test_v9_runtime_contract.py`)

**Files:**
- Rename / Modify: `stolen-vehicles-project/tests/test_v9_runtime_contract.py` -> `stolen-vehicles-project/tests/test_final_runtime_contract.py`
- Test: Full pytest run

- [ ] **Step 1: Update test file to validate final runtime contract**
  - Assert that `final_scores.json` and `data.json` declare `model_version == "final"`.
  - Verify that simulation reports non-negative savings under final engine.
  - Verify that artifacts load cleanly under `final` contract.

- [ ] **Step 2: Run all test files**
  Run: `python -X utf8 -m pytest stolen-vehicles-project/tests/ -v`
  Expected: All 21 tests PASS.

---

### Task 6: Archive `CarTheftRisk_v9.ipynb` & Update Archive Policy

**Files:**
- Move: `stolen-vehicles-project/notebooks/CarTheftRisk_v9.ipynb` -> `stolen-vehicles-project/archive/notebooks/CarTheftRisk_v9.ipynb`
- Modify: `stolen-vehicles-project/ARCHIVE_STATUS.md`
- Modify: `stolen-vehicles-project/notebooks/ARCHIVE_STATUS.md`

- [ ] **Step 1: Move notebook to archive**
  Move `stolen-vehicles-project/notebooks/CarTheftRisk_v9.ipynb` to `stolen-vehicles-project/archive/notebooks/CarTheftRisk_v9.ipynb`.
  Ensure `notebooks/` contains exclusively:
  - `CarTheftRisk_final.ipynb`
  - `data_final.json`
  - `ARCHIVE_STATUS.md`

- [ ] **Step 2: Update `ARCHIVE_STATUS.md` files**
  Document the retirement of `CarTheftRisk_v9.ipynb` to `archive/notebooks/`, citing the migration to the unified `final` contract.
  Ensure tests asserting archive status (`test_archive_policy_points_to_active_research`) continue to pass.

---

### Task 7: Update Project SSOT & Documentation

**Files:**
- Modify: `stolen-vehicles-project/SINGLE_SOURCE_OF_TRUTH.md`
- Modify: `stolen-vehicles-project/GEMINI.md`
- Modify: `stolen-vehicles-project/README.md`
- Modify: `stolen-vehicles-project/web/agent_report.html`
- Modify: `stolen-vehicles-project/stolen_vehicles_web_engineering_guide_v4.html`

- [ ] **Step 1: Update `SINGLE_SOURCE_OF_TRUTH.md`**
  - Reframe Section 1: The architecture is now **Unified Canonical Architecture**: `CarTheftRisk_final.ipynb` powers both the research and the runtime applications.
  - Document `final_scores.json`, `data_final.json`, and `temporal_backtest_final.json`.

- [ ] **Step 2: Update `GEMINI.md`**
  - Update Hard Rules and context to state that `final` is the sole active runtime contract.

- [ ] **Step 3: Update `README.md` and report pages**
  - Update status banners and references from v9 to Final.

- [ ] **Step 4: Final verification**
  - Run full test suite: `python -X utf8 -m pytest stolen-vehicles-project/tests/ -v`.
  - Confirm all 21 tests pass with 0 errors.

