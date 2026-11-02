"""
backend/models/lgbm_model.py
============================
Gradient-Boosted Tree Forecasting Engine using LightGBM.

WHY THIS EXISTS:
LightGBM is consistently the top-performing tabular machine learning model
in competitive time-series competitions (such as the M5 Walmart competition).
Its strengths for this task include:
1. Handling complex non-linear feature interactions (e.g., price discounts × weekend surges).
2. Native support for categorical features (departments, categories, stores).
3. The Tweedie regression loss, which accounts for the zero-inflated, right-skewed
   distribution of retail SKU sales.
4. Recursive multi-step forecasting that projects dynamic lag updates into the future.
"""

import numpy as np
import pandas as pd
import logging
import pickle
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Categorical identifiers provided to LightGBM
CAT_FEATURES = [
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "weekday", "event_name"
]

# Engineered numerical inputs
NUM_FEATURES = [
    "wday", "month", "year", "day_of_week", "day_of_month",
    "week_of_year", "is_weekend", "quarter",
    "has_event", "snap",
    "sell_price", "price_change", "price_rel_mean",
    "lag_7", "lag_14", "lag_21", "lag_28",
    "roll_mean_7",  "roll_mean_14",  "roll_mean_28",
    "roll_std_7",   "roll_std_14",   "roll_std_28",
]

ALL_FEATURES = CAT_FEATURES + NUM_FEATURES


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Filters the feature list to include only columns present in the input DataFrame.
    Prevents KeyError exceptions if certain experimental features are omitted.
    """
    return [c for c in ALL_FEATURES if c in df.columns]


def fit_lgbm(train_df: pd.DataFrame, lgbm_params: dict,
             model_path: Optional[Path] = None):
    """
    Trains a LightGBM regressor on engineered tabular features.
    Target transformation: log1p(sales) = log(sales + 1) to stabilize high variance.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM is not installed. Install via: pip install lightgbm")

    feat_cols = get_feature_cols(train_df)
    
    # Drop warm-up rows containing NaN in the maximum lag window (lag_14 / lag_28)
    ready = train_df.dropna(subset=["lag_7", "lag_14"]).copy()
    
    X = ready[feat_cols]
    # log1p transformation compresses large outliers and stabilizes error gradients
    y = np.log1p(ready["sales"].clip(lower=0).values)

    cat_cols_present = [c for c in CAT_FEATURES if c in feat_cols]

    logger.info(f"Training LightGBM on {len(X):,} records with {len(feat_cols)} features …")
    
    model = lgb.LGBMRegressor(**lgbm_params)
    model.fit(
        X, y,
        categorical_feature=cat_cols_present,
    )

    # Persist model binary to disk if path is provided
    if model_path:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"LightGBM model checkpoint saved to: {model_path}")

    return model


def load_lgbm(model_path: Path):
    """Loads a serialized LightGBM model from disk."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


def lgbm_forecast(arg1, arg2, horizon: int = 28) -> pd.DataFrame:
    """
    Executes recursive multi-step forecasting. Accepts (model, series, horizon) or (series, model, horizon).

    HOW RECURSIVE FORECASTING WORKS:
    1. Step 1: Predict day T+1 using true historical lags.
    2. Step 2: Append the predicted T+1 value into the historical buffer.
    3. Step 3: Recompute lag_7, lag_14, and rolling averages dynamically using
       the predicted values.
    4. Repeat up to day T+horizon.
    This simulates realistic forward deployment where future sales are unobserved.
    """
    if isinstance(arg1, pd.DataFrame):
        series = arg1
        model = arg2
    else:
        model = arg1
        series = arg2
    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("LightGBM unavailable for inference.")
        return pd.DataFrame(columns=["date", "yhat", "yhat_lower", "yhat_upper"])

    feat_cols = get_feature_cols(series)
    last_date = series["date"].max()

    # Working buffer that expands with each forward step
    history = series.copy()
    results = []

    # Historical standard deviation of recent sales for prediction intervals
    recent_std = float(np.std(history["sales"].values[-28:])) if len(history) >= 28 else float(np.std(history["sales"].values))
    if recent_std == 0:
        recent_std = 1.0

    for step in range(horizon):
        next_date = last_date + pd.Timedelta(days=step + 1)
        
        # Clone the last row structure and update calendar time dimensions
        future_row = history.iloc[[-1]].copy()
        future_row["date"]         = next_date
        future_row["day_of_week"]  = next_date.dayofweek
        future_row["day_of_month"] = next_date.day
        future_row["week_of_year"] = int(next_date.isocalendar()[1])
        future_row["is_weekend"]   = int(next_date.dayofweek >= 5)
        future_row["month"]        = next_date.month
        future_row["year"]         = next_date.year
        future_row["quarter"]      = (next_date.month - 1) // 3 + 1
        future_row["has_event"]    = 0
        if "wday" in future_row.columns:
            # M5 convention: 1 = Saturday, 2 = Sunday, ..., 7 = Friday
            future_row["wday"] = ((next_date.weekday() + 2) % 7) + 1
        if "weekday" in future_row.columns and hasattr(future_row["weekday"], "cat"):
            day_name = next_date.strftime("%A")
            if day_name in future_row["weekday"].cat.categories:
                future_row["weekday"] = pd.Categorical([day_name], categories=future_row["weekday"].cat.categories)

        # Dynamically sample lag features from the expanding simulated sales history
        sales_hist = history["sales"].values
        for lag in [7, 14, 21, 28]:
            col = f"lag_{lag}"
            if col in future_row.columns:
                future_row[col] = float(sales_hist[-lag]) if len(sales_hist) >= lag else 0.0

        # Dynamically update rolling statistics over the shifted window
        shifted = sales_hist[:-7] if len(sales_hist) > 7 else sales_hist
        shifted_len = len(shifted)
        shifted_mean = float(np.mean(shifted)) if shifted_len > 0 else 0.0
        shifted_std  = float(np.std(shifted)) if shifted_len > 0 else 0.0

        for win in [7, 14, 28]:
            if f"roll_mean_{win}" in future_row.columns:
                future_row[f"roll_mean_{win}"] = float(np.mean(shifted[-win:])) if shifted_len >= win else shifted_mean
            if f"roll_std_{win}" in future_row.columns:
                future_row[f"roll_std_{win}"]  = float(np.std(shifted[-win:])) if shifted_len >= win else shifted_std

        # Model inference: X -> predict log1p(y) -> expm1(y) -> physical units
        X_pred = future_row[feat_cols]
        yhat_log = model.predict(X_pred)[0]
        yhat = float(np.expm1(yhat_log))
        yhat = max(0.0, yhat)

        # 95% Confidence Interval (± 1.96 * sigma)
        yhat_lower = max(0.0, yhat - 1.96 * recent_std)
        yhat_upper = max(0.0, yhat + 1.96 * recent_std)

        results.append({
            "date":       next_date.strftime("%Y-%m-%d"),
            "yhat":       round(yhat, 2),
            "yhat_lower": round(yhat_lower, 2),
            "yhat_upper": round(yhat_upper, 2),
        })

        # Append prediction into working history buffer to feed subsequent lags
        next_row = future_row.copy()
        next_row["sales"] = yhat
        history = pd.concat([history, next_row], ignore_index=True)

    return pd.DataFrame(results)


def compute_lgbm_metrics(actual: np.ndarray, predicted: np.ndarray,
                         train_sales: np.ndarray) -> dict:
    """Computes RMSE, MAE, WAPE, and MASE for LightGBM predictions."""
    if len(actual) == 0 or len(predicted) == 0:
        return {"model": "lightgbm", "rmse": 0.0, "mae": 0.0, "wape": 0.0, "mase": None}

    errors = actual - predicted
    mae    = float(np.mean(np.abs(errors)))
    rmse   = float(np.sqrt(np.mean(errors ** 2)))
    sum_act = float(np.sum(np.abs(actual)))
    wape   = float(np.sum(np.abs(errors)) / (sum_act + 1e-8))
    
    naive_diff = np.diff(train_sales) if len(train_sales) > 1 else np.array([1.0])
    naive_mae  = float(np.mean(np.abs(naive_diff))) if len(naive_diff) > 0 else 1.0
    mase       = mae / (naive_mae + 1e-8)

    return {
        "model": "lightgbm",
        "rmse":  round(rmse, 3),
        "mae":   round(mae, 3),
        "wape":  round(wape, 4),
        "mase":  round(mase, 3)
    }
