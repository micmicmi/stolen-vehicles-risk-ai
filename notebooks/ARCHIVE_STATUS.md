# Status of Project Notebooks, Reports & Archives

## Active Notebooks (`notebooks/`)
- **`CarTheftRisk_final.ipynb`**: The sole canonical end-to-end Capstone research and runtime notebook containing the strict-child-reset state-machine parser, explicit quarantine of incomplete leaves, symmetric pre-cutoff cohort construction (`2024 -> 2025`), spatial `GroupKFold` tuning, `XGBoost Poisson` with exposure offset (`log(0.75)`), tie-preserving decile lift & city-clustered bootstrap CIs, `PyTorch CountMLP` with early stopping, `SHAP` explainability, two-tier CBS 2025 motor vehicle fleet exposure normalization (Appendix D), and unified runtime artifact export (`notebooks/data_final.json`, `web/assets/final_scores.json`).

## Active Documentation & Web Reports
- **Single Source of Truth**: `SINGLE_SOURCE_OF_TRUTH.md` and `README.md`
- **Canonical Research Process Report**: `web/final.html` (with 13 rendered charts in `web/assets/charts/final_*.png`)
- **Engineering & Interview Guide (v4)**: `stolen_vehicles_web_engineering_guide_v4.html` (mirrored in `web/stolen_vehicles_web_engineering_guide_v4.html`)
- **Live Web Application & Runtime Reports**: `web/index.html`, `web/agent_report.html`, `web/process_report.html`, `docs/CAPSTONE_PROJECT_PLAN.md`, `docs/RESEARCH_LIMITATIONS.md`

## Archived Assets (`archive/`)
All superseded drafts, historical notebook iterations, intermediate caches, audit working files, and unreferenced assets are organized under `archive/`:
- **`archive/notebooks/`**:
  - `CarTheftRisk_v9.ipynb` (Archived after unifying the runtime and test contract under `CarTheftRisk_final`)
  - `CarTheftRisk.ipynb`, `CarTheftRisk_v1.ipynb` through `CarTheftRisk_v7.ipynb`
  - `CarTheftRisk_v8.ipynb`, `CarTheftRisk_v8_old.ipynb`, `CarTheftRisk_v8_updated.ipynb`, `CarTheftRisk_v8_updated_2.ipynb`
  - `CarTheftRisk_v10.ipynb`, `temporal_4models_official.json`
  - `legacy_cache/` (`data.json`, `production_assets.json`, `df_areas_tab1.csv`, `df_areas_tab2.csv`, `df_models_full.csv`, `df_models_hist.csv`)
- **`archive/reports_and_guides/`**:
  - Legacy engineering guides (`v2`, `v3`) and layout drafts
  - `v8_audits/` (`CarTheftRisk_v8_Notebook_Audit*.md/pdf/html/css` — remediated in `CarTheftRisk_final.ipynb`)
  - `legacy_root_docs/` (duplicate root `CAPSTONE_PROJECT_PLAN.md`, legacy `MODEL_STATUS.md`, unreferenced root diagram PNGs)
- **`archive/src_legacy/`**: Deprecated v1/v5/v7 helper scripts (`generate_notebook.py`, `sync_v7_to_web.py`)
- **`archive/scripts_oct30/`**: One-off historical UI inspection and patch scripts
- **`archive/web_legacy/`**: `old_layout_archive/`, `.hallmark/` visual crop review cache, and unreferenced image/chart files (`assets_unused/`, `images_unused/`, `charts_unused/`)
- **`archive/scratch/`**: Temporary scratch scripts and browser rendering profile
