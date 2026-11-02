"""
backend/config.py
=================
Central Configuration Module for the Walmart Demand Forecasting & Inventory
Optimization System.

WHY THIS EXISTS:
Keeping all data paths, model hyperparameters, sampling configurations, and
financial penalty constants centralized in one file eliminates magic numbers,
ensures uniform configuration across both API endpoints and background workers,
and makes adjusting experimental parameters straightforward without touching
operational business logic. Supports automated loading from .env configuration files.
"""

import os
from pathlib import Path

# ── Base Directory Paths ───────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "backend" / "saved_models"

# ── Load Environment Variables from .env if present ───────────────────────────
try:
    from dotenv import load_dotenv
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass

# ── Dataset File Locations ────────────────────────────────────────────────────
SALES_TRAIN_VALIDATION  = DATA_DIR / "sales_train_validation.csv"
SALES_TRAIN_EVALUATION  = DATA_DIR / "sales_train_evaluation.csv"
CALENDAR_FILE           = DATA_DIR / "calendar.csv"
SELL_PRICES_FILE        = DATA_DIR / "sell_prices.csv"

# ── Multi-Store & Series Sampling Configuration ──────────────────────────────
# In deployment, SAMPLE_STORES can be overridden via environment variables (e.g. "CA_1,TX_1,WI_1")
_env_stores = os.getenv("SAMPLE_STORES")
if _env_stores:
    SAMPLE_STORES = [s.strip() for s in _env_stores.split(",") if s.strip()]
else:
    SAMPLE_STORES = ["CA_1", "TX_1", "WI_1"]

SAMPLE_ITEMS  = None

# Evaluation holdout horizon in days (default: 28 days for M5 competition standard)
N_EVAL_DAYS   = int(os.getenv("N_EVAL_DAYS", 28))

# Trailing operational history days retained in memory (default: 180 days = ~350MB RAM)
HISTORY_DAYS  = int(os.getenv("HISTORY_DAYS", 180))

# ── Time-Series Feature Engineering Parameters ────────────────────────────────
LAG_DAYS         = [7, 14, 21, 28]
ROLLING_WINDOWS  = [7, 14, 28]

# ── LightGBM Hyperparameters ──────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective":               "tweedie",
    "tweedie_variance_power":  1.1,
    "metric":                  "rmse",
    "learning_rate":           0.05,
    "num_leaves":              128,
    "min_child_weight":        32,
    "feature_fraction":        0.8,
    "bagging_fraction":        0.8,
    "bagging_freq":            1,
    "n_estimators":            500,
    "verbosity":               -1,
    "n_jobs":                  -1,
    "seed":                    42,
}

# ── Facebook Prophet Forecasting Defaults ─────────────────────────────────────
PROPHET_PARAMS = {
    "yearly_seasonality":      True,
    "weekly_seasonality":      True,
    "daily_seasonality":       False,
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "interval_width":          0.95,
}

# ── Newsvendor & Inventory Optimization Cost Constants ───────────────────────
DEFAULT_HOLDING_COST_PER_UNIT  = float(os.getenv("DEFAULT_HOLDING_COST", 0.20))
DEFAULT_STOCKOUT_COST_PER_UNIT = float(os.getenv("DEFAULT_STOCKOUT_COST", 2.00))
DEFAULT_SAFETY_STOCK_FACTOR    = float(os.getenv("DEFAULT_SAFETY_STOCK_FACTOR", 1.65))
DEFAULT_ORDER_CYCLE_DAYS       = int(os.getenv("DEFAULT_ORDER_CYCLE_DAYS", 7))

# ── FastAPI Server Network Settings ──────────────────────────────────────────
API_HOST    = os.getenv("HOST", os.getenv("API_HOST", "127.0.0.1"))
API_PORT    = int(os.getenv("PORT", os.getenv("API_PORT", 8000)))
API_TITLE   = "Walmart Demand Forecasting & Inventory Optimization API"
API_VERSION = "1.0.0"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
