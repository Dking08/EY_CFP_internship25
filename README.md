# AlcoBev Cash Flow & Universal Dynamic Forecasting Engine

A full-stack forecasting & analytics platform delivering: data generation → model training → domain FastAPI with KPIs → Streamlit dashboard → accuracy reporting → PLUS a universal dynamic engine that can ingest ANY time‑series CSV and auto-train multi‑algorithm models.

---
## High-Level Modules

| Module | Purpose | Tech | Key File(s) |
|--------|---------|------|-------------|
| AlcoBev Cashflow API | Domain forecasts (Sales, COGS, Volume) + KPIs + promo & holiday analytics | FastAPI, XGBoost | `Cashflow Forecasting/app.py` |
| Model Training | Train & persist XGBoost pipelines | scikit-learn, XGBoost | `Cashflow Forecasting/train_models.py` |
| Accuracy & Reporting | Holdout / rolling / segment metrics + Excel | openpyxl | `Cashflow Forecasting/testcase.py` |
| Streamlit Dashboard | Enhanced KPI visualization (basic vs comprehensive) | Streamlit, Plotly | `frontend/streamlit_app.py` |
| Dynamic Forecasting Engine | Universal feature detection & model selection | Custom | `dynamic_engine/dynamic_forecasting_engine.py` |
| Dynamic API | Upload → Train → Predict → Forecast | FastAPI | `dynamic_engine/dynamic_api.py` |
| Dynamic Dashboard (optional) | Generic dataset UI | Streamlit | `frontend/dynamic_dashboard.py` |

---
 
## Repository Structure

```text
Cashflow Forecasting/
    app.py                   # AlcoBev FastAPI (KPIs, promo/holiday logic)
    train_models.py          # Trains sales/COGS/volume models
    generate_data.py         # Synthetic data generator
    testcase.py              # Accuracy tests & Excel report
    requirements.txt         # Domain dependencies
    alcobev_europe_sales_data.csv
    models/                  # Saved pipelines (.pkl)

dynamic_engine/
    dynamic_forecasting_engine.py  # Universal engine core
    dynamic_api.py                 # Universal FastAPI service
    demo_dynamic_engine.py         # Demo runner
    demo_*.csv                     # Sample datasets (ecommerce, restaurant, stocks)
    DYNAMIC_README.md              # Deep documentation

frontend/
    streamlit_app.py         # Domain dashboard
    dynamic_dashboard.py     # Universal dashboard
```

Note: Path contains a space (`Cashflow Forecasting/`)—quote in Windows commands.

---
 
## Architecture Overview

```text
┌─────────────────────┐      /forecast (/comprehensive)      ┌──────────────────────┐
│  Streamlit (Domain) │ ◄──────────────────────────────────► │  FastAPI Domain App  │
└──────────┬──────────┘                                      └──────────┬───────────┘
           │    uses models                                             │ loads models
           ▼                                                            ▼
        models/*.pkl     ◄──────────── train_models.py ◄────────   data (CSV)

┌─────────────────────┐      /upload /train /predict /forecast ┌──────────────────────┐
│ Dynamic Dashboard   │ ◄────────────────────────────────────► │ Dynamic API Engine   │
└──────────┬──────────┘                                        └──────────┬───────────┘
           │                                                              │
           ▼                                                              ▼
     dynamic_forecasting_engine.py (auto feature detection + model selection)
```

---
 
## Feature Summary

### Domain (AlcoBev) API

* Predicts Sales Revenue, COGS, Volume
* KPIs: Gross Profit, Margins, ASP, COGS/Litre, Marketing efficiency, Cash Flow
* Promotional impact + Holiday lift
* Segment breakdown (Country / Channel / Product)
* Comprehensive vs Basic modes

### Dynamic Universal Engine

* Accept ANY CSV with at least one date/time column & numeric targets
* Auto-detect numerical / categorical / date / potential targets
* Time & lag feature enrichment
* Multi-algorithm training (LightGBM, XGBoost, Random Forest) + best selection
* Multi-target forecasting & future horizon generation
* Session-based workflow (in-memory; pluggable to Redis/DB)

---
 
## Business KPIs (Domain)

| KPI | Definition |
|-----|------------|
| Gross Profit (€) | Revenue − COGS |
| Gross Profit Margin % | (Revenue − COGS)/Revenue |
| ASP per Litre (€) | Revenue / Volume |
| COGS per Litre (€) | COGS / Volume |
| Operating Cash Flow (€) | Revenue − COGS − Marketing Spend |
| Marketing Spend Ratio % | Marketing / Revenue |
| Revenue per € Marketing | Revenue / Marketing |
| Litres per € Marketing | Volume / Marketing |
| Promotional Impact % | Promo vs non‑promo uplift |
| Holiday Sales Lift % | Holiday vs baseline uplift |

---
 
## Environment Setup (Windows / PowerShell)

```pwsh
python -m venv cflow_ven
./cflow_ven/Scripts/Activate.ps1
pip install -r "Cashflow Forecasting/requirements.txt"
pip install fastapi uvicorn lightgbm xgboost python-multipart streamlit plotly requests openpyxl
```
Python 3.11 used; 3.10+ expected compatible.

---
 
## Train Domain Models

`train_models.py`:

1. Loads dataset
2. Builds time + lag features
3. Trains XGBoost models (Sales, COGS, Volume)
4. Saves pipelines to `models/`

```pwsh
python "Cashflow Forecasting/train_models.py"
```

---
 
## FastAPI Services

### 1. Domain API (`app.py`)

Run:

```pwsh
cd "Cashflow Forecasting"
uvicorn app:app --reload --port 8000
```
Endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health & model load status |
| `/forecast` | POST | Basic list of daily forecasts |
| `/forecast/comprehensive` | POST | Forecast + KPI + promo/holiday/segment summaries |
| `/kpi/calculate` | GET | KPI calc for ad‑hoc inputs |
| `/analytics/summary` | GET | Aggregated KPIs & impacts |

Sample payload:

```json
{
    "country": "Germany",
    "channel": "Off-Trade",
    "product_category": "Beer",
    "start_date": "2025-09-19",
    "end_date": "2025-10-18"
}
```
PowerShell call:

```pwsh
Invoke-RestMethod -Uri http://127.0.0.1:8000/forecast/comprehensive -Method Post -ContentType 'application/json' -Body '{
    "country":"Germany","channel":"Off-Trade","product_category":"Beer",
    "start_date":"2025-09-19","end_date":"2025-10-18"
}' | ConvertTo-Json -Depth 4
```

### 2. Dynamic API (`dynamic_api.py`)

Run on separate port:

```pwsh
cd dynamic_engine
uvicorn dynamic_api:app --reload --port 8001
```
Workflow:

| Step | Endpoint | Method | Purpose |
|------|----------|--------|---------|
| 1 | `/upload` | POST (multipart) | Upload CSV & auto-detect schema |
| 2 | `/train` | POST | Train selected target columns |
| 3 | `/predict` | POST | Predictions for supplied rows |
| 4 | `/forecast` | POST | Future horizon synthesis & forecast |
| - | `/session/{id}/info` | GET | Session & model performance |
| - | `/sessions` | GET | List active sessions |
| - | `/analytics/summary` | GET | Global usage/profile stats |
| - | `/health` | GET | API health |

Examples:

```pwsh
# Upload
Invoke-RestMethod -Uri http://127.0.0.1:8001/upload -Method Post -Form @{ file = Get-Item ..\demo_ecommerce_data.csv }

# Train
Invoke-RestMethod -Uri http://127.0.0.1:8001/train -Method Post -ContentType 'application/json' -Body '{
    "session_id":"<SESSION>","target_columns":["daily_sales"],"test_size":0.2
}' | ConvertTo-Json -Depth 6

# Forecast
Invoke-RestMethod -Uri http://127.0.0.1:8001/forecast -Method Post -ContentType 'application/json' -Body '{
    "session_id":"<SESSION>","forecast_days":30,
    "base_values":{"region":"North","marketing_spend":1000}
}' | ConvertTo-Json -Depth 6
```
See `dynamic_engine/DYNAMIC_README.md` for in-depth docs.

---
 
## Streamlit Dashboards

### Domain Dashboard

```pwsh
streamlit run frontend/streamlit_app.py
```
Features: parameter sidebar, KPI cards, multi-panel Plotly (Revenue, Gross Profit, Volume, Cash Flow, Margin %, ASP), promo & holiday analysis, segment summaries.

### Dynamic Dashboard (optional)

```pwsh
streamlit run frontend/dynamic_dashboard.py
```
Upload → train → forecast for arbitrary datasets.

---
 
## Accuracy & Reporting (`testcase.py`)

Provides:

* Holdout metrics (MAE, RMSE, R², MAPE, bias, % within 10% / 20%)
* Rolling window evaluation
* Segment-wise accuracy (Country / Channel / Product)
* Styled multi-sheet Excel report

Run:

```pwsh
python "Cashflow Forecasting/testcase.py"
```
Outputs: `AlcoBev_Model_Accuracy_Report_YYYYMMDD_HHMMSS.xlsx`.

---
 
## Basic vs Comprehensive (Domain UI)

| Mode | Output |
|------|--------|
| Basic | Daily forecast rows only |
| Comprehensive | Daily rows + aggregated totals, averages, promo/holiday lift, segment breakdown, performance highlights |

---
 
## Extensibility

| Goal | Change Location |
|------|-----------------|
| Add KPI | KPI aggregation logic in `app.py` (summary creation) |
| New algorithm (dynamic engine) | Extend training loop in `dynamic_forecasting_engine.py` |
| Feature engineering | `create_time_features` + engine prep methods |
| New segment dimension | Adjust Pydantic models + summary builders |
| Persist sessions | Replace in-memory `active_engines` with DB/Redis |

---
 
## Quick Test

1. `python Cashflow Forecasting/train_models.py`
2. Start domain API → `/health`
3. POST `/forecast/comprehensive` → verify KPI fields
4. Launch Streamlit domain dashboard
5. Start dynamic API (port 8001)
6. Upload demo CSV → train → forecast
7. Run `testcase.py` → confirm Excel report

---