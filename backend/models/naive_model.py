"""
backend/models/naive_model.py
=============================
Baseline Forecasting Algorithms: Simple Naive & Seasonal Naive Benchmark.

WHY THIS EXISTS:
In time-series machine learning, establishing a robust statistical baseline
is non-negotiable. Before deploying complex machine learning (LightGBM) or
deep learning architectures (TFT), we must quantify the accuracy of simple
heuristics. If a sophisticated neural network cannot comfortably outperform
a Seasonal Naive model (repeating sales from 7 days ago), the operational
overhead and compute cost of the neural network cannot be justified.
"""

import numpy as np
import pandas as pd


def naive_forecast(series: pd.DataFrame, horizon: int,
                   strategy: str = "seasonal_naive") -> pd.DataFrame:
    """
    Generates a baseline forecast for `horizon` forward steps.

    METHODOLOGY:
    - 'seasonal_naive': For weekly retail data, human shopping habits repeat
      strongly on a 7-day cycle (e.g., Saturday sales resemble previous Saturdays).
      We extract the most recent 7-day vector and tile it over the horizon.
    - 'last_value': Flat persistence model — projects the last observed value forward.
    - Uncertainty Bounds: An empirical 95% confidence interval is estimated as
      yhat ± 1.96 * std_dev of the recent 28-day historical residuals.

    Parameters
    ----------
    series   : DataFrame with at least ['date', 'sales'] ordered chronologically
    horizon  : Number of daily steps to project into the future
    strategy : 'seasonal_naive' (default) or 'last_value'

    Returns
    -------
    pd.DataFrame with columns ['date', 'yhat', 'yhat_lower', 'yhat_upper']
    """
    last_date  = series["date"].max()
    sales_vals = series["sales"].values

    if strategy == "last_value":
        # Project the single latest observation forward
        point = float(sales_vals[-1]) if len(sales_vals) > 0 else 0.0
        std   = float(series["sales"].rolling(7, min_periods=1).std().iloc[-1]) if len(sales_vals) > 1 else 1.0

    elif strategy == "seasonal_naive":
        # Extract the last 7 daily observations (Monday through Sunday pattern)
        pattern = sales_vals[-7:]
        if len(pattern) < 7:
            # If the history is shorter than 7 days, pad with the edge value
            pattern = np.pad(pattern, (7 - len(pattern), 0), mode="edge")
        
        # Historical standard deviation over the last 28 days
        std = float(np.std(sales_vals[-28:])) if len(sales_vals) >= 28 else float(np.std(sales_vals)) if len(sales_vals) > 0 else 1.0
    else:
        raise ValueError(f"Unknown naive strategy: {strategy}. Expected 'seasonal_naive' or 'last_value'.")

    # Construct the chronological date sequence for the forecast horizon
    future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(horizon)]
    rows = []
    
    for i, d in enumerate(future_dates):
        if strategy == "last_value":
            yhat = point
        else:
            # Cycle through the 7-day pattern vector
            yhat = float(pattern[i % 7])

        # Enforce non-negativity constraint (retail sales cannot be negative)
        yhat = max(0.0, yhat)
        yhat_lower = max(0.0, yhat - 1.96 * std)
        yhat_upper = max(0.0, yhat + 1.96 * std)

        rows.append({
            "date":       d.strftime("%Y-%m-%d"),
            "yhat":       round(yhat, 2),
            "yhat_lower": round(yhat_lower, 2),
            "yhat_upper": round(yhat_upper, 2),
        })

    return pd.DataFrame(rows)


def compute_naive_metrics(actual: np.ndarray, predicted: np.ndarray,
                          train_sales: np.ndarray) -> dict:
    """
    Evaluates out-of-sample forecast accuracy against true historical values.

    Metrics computed:
    - RMSE: Sensitivity to large outlier prediction errors.
    - MAE: Average error in physical units.
    - WAPE: Weighted Absolute Percentage Error (sum(|actual - pred|) / sum(actual)).
            Unlike MAPE, WAPE does not divide by zero when daily sales are 0.
    - MASE: Scaled against in-sample one-step naive benchmark. MASE = 1.0 by definition
            for the naive benchmark.
    """
    if len(actual) == 0 or len(predicted) == 0:
        return {"model": "naive", "rmse": 0.0, "mae": 0.0, "wape": 0.0, "mase": 1.0}

    errors = actual - predicted
    mae    = float(np.mean(np.abs(errors)))
    rmse   = float(np.sqrt(np.mean(errors ** 2)))
    sum_actual = float(np.sum(np.abs(actual)))
    wape   = float(np.sum(np.abs(errors)) / (sum_actual + 1e-8))

    # Denominator: Mean absolute error of 1-step naive forecast on training data
    naive_diff = np.diff(train_sales) if len(train_sales) > 1 else np.array([1.0])
    naive_mae  = float(np.mean(np.abs(naive_diff))) if len(naive_diff) > 0 else 1.0
    mase       = mae / (naive_mae + 1e-8)

    return {
        "model": "naive",
        "rmse":  round(rmse, 3),
        "mae":   round(mae, 3),
        "wape":  round(wape, 4),
        "mase":  round(mase, 3)
    }
