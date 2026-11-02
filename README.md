# Walmart Demand Forecasting & Inventory Optimization

Forecasts product demand across Walmart stores using the M5 dataset,
then recommends optimal order quantities to minimise stockouts and holding costs.

---

## 🗂️ Project Structure

```
.
├── data/                    # M5 Forecasting dataset (raw, do not modify)
│   ├── calendar.csv
│   ├── sales_train_validation.csv
│   ├── sales_train_evaluation.csv
│   └── sell_prices.csv
│
├── notebooks/
│   └── ml_workflow.ipynb    # ← Start here. Full ML pipeline.
│
├── backend/                 # FastAPI backend
│   ├── main.py              # App entry point
│   ├── config.py            # All constants
│   ├── schemas.py           # Pydantic types
│   ├── models/              # Forecasting model implementations
│   ├── services/            # Business logic (data, forecast, inventory)
│   ├── routers/             # API route handlers
│   └── requirements.txt
│
├── frontend/                # Plain HTML/CSS/JS dashboard
│   ├── index.html
│   ├── css/style.css
│   └── js/
│       ├── api.js           # Backend client
│       ├── charts.js        # All charting
│       └── main.js          # Page routing + app logic
│
└── docs/                    # Presentation, architecture, reflection
    ├── presentation/presentation.md
    ├── architecture/architecture.md
    ├── api_documentation.md
    └── reflection.md
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

> **Note:** `torch` and `pytorch-forecasting` can be heavy. If you don't need TFT, you can skip them — the API falls back to exponential smoothing automatically.

### 2. Start the Backend

```bash
# From the project root:
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API will pre-load the dataset on startup (~30 seconds first time).
Subsequent restarts use the cached LightGBM model from `backend/saved_models/`.

### 3. Open the Dashboard

Open `frontend/index.html` in your browser, **or** navigate to:
```
http://localhost:8000/
```
(The FastAPI server also serves the frontend as static files.)

### 4. Interactive API Docs

```
http://localhost:8000/docs
```

---

## 📓 Jupyter Notebook

The main ML workflow lives in `notebooks/ml_workflow.ipynb`.
Run it with:

```bash
jupyter notebook notebooks/ml_workflow.ipynb
```

**Sections:**
1. Setup & imports
2. Data loading & exploration
3. Feature engineering
4. Exploratory data analysis
5. Train/test split
6. Naive baseline
7. Prophet model
8. LightGBM model
9. TFT (Temporal Fusion Transformer)
10. Hierarchical reconciliation (Darts)
11. Prediction intervals
12. Model comparison
13. Inventory optimization (OR-Tools)
14. Summary & conclusions

---

## 🔧 Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python | Main language |
| FastAPI + Uvicorn | REST API backend |
| Prophet | Time-series forecasting (trend + seasonality) |
| LightGBM | ML forecasting (gradient-boosted trees) |
| Darts | Hierarchical reconciliation |
| PyTorch Forecasting | TFT deep learning model |
| OR-Tools | Inventory optimization (GLOP LP solver) |
| Plotly / Chart.js | Interactive charts in dashboard |
| Pydantic | Request/response validation |

---

## 📊 Dataset

**M5 Forecasting — Accuracy and Uncertainty (Walmart)**

- 30,490 time series (item × store combinations)
- 3,049 unique products across 3 categories (FOODS, HOUSEHOLD, HOBBIES)
- 10 stores in 3 US states: California (CA), Texas (TX), Wisconsin (WI)
- ~5.5 years of daily sales data
- Calendar includes SNAP days, holidays, and sporting events

We work on a **3-store sample** (CA_1, TX_1, WI_1) for speed.
To use all stores, set `SAMPLE_STORES = None` in `backend/config.py`.

---

## 🚀 API Quick Reference

```bash
# Get a 28-day LightGBM forecast
curl "http://localhost:8000/api/forecast/quick/HOBBIES_1_001/CA_1?horizon=28&model=lightgbm"

# Compare all models
curl "http://localhost:8000/api/metrics/compare?item_id=HOBBIES_1_001&store_id=CA_1"

# Optimize inventory
curl "http://localhost:8000/api/inventory/quick/HOBBIES_1_001/CA_1?current_stock=30&forecast_demand=120&demand_std=18"

# Dashboard summary
curl "http://localhost:8000/api/metrics/summary"
```

---

## 📁 Docs Folder

The `docs/` folder contains all deliverables that are separate from the code:

- `presentation/presentation.md` — 14-slide project presentation
- `architecture/architecture.md` — System architecture + file structure
- `api_documentation.md` — Full API reference
- `reflection.md` — Personal reflection on learnings and challenges

---

## ⚠️ Notes

- The LightGBM model trains on first run and saves to `backend/saved_models/lgbm_model.pkl`. Subsequent API starts reuse this file.
- Prophet and TFT calls may be slow on the first request (model fitting is lazy). LightGBM and Naive are fast.
- All OR-Tools inventory calls use the GLOP LP solver by default, with a pure-Python analytic fallback if OR-Tools isn't installed.

