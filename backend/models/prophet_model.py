"""
backend/models/prophet_model.py — Meta Prophet Additive Forecaster with SNAP Regressors.

========================================================================================
THEORETICAL FOUNDATION:
Prophet (developed by Meta Core Data Science) frames time series forecasting as a curve-
fitting exercise using a Generalized Additive Model (GAM) with three primary components:
    y(t) = g(t) + s(t) + h(t) + ε_t

Where:
  1. g(t): Piecewise linear or logistic growth curve representing non-periodic trend.
     Changepoints are automatically discovered using a sparse Laplace prior on rate changes:
     `changepoint_prior_scale` governs model flexibility (0.05 prevents overfitting).
  2. s(t): Periodic seasonal changes modeled via standard Fourier series harmonics:
     s(t) = Σ [a_n * cos(2π n t / P) + b_n * sin(2π n t / P)]
     For weekly retail cycles (P = 7 days), low-order Fourier terms flexibly model the
     Saturday/Sunday sales surge compared to midweek troughs.
  3. h(t): Holiday and event effects. In this M5 implementation, we inject two vital
     exogenous regressors:
       - `snap`: Supplemental Nutrition Assistance Program benefit disbursement days.
         Low-income households receive food stamps during specific calendar windows,
         generating significant, predictable surges in grocery demand.
       - `has_event`: Major cultural and sporting events (Super Bowl, Easter, etc.).
  4. ε_t: Normally distributed error term.
========================================================================================
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)


def build_prophet_df(series: pd.DataFrame, state_id: str) -> pd.DataFrame:
    """
    Format internal feature DataFrame into Prophet's strict canonical schema.

    Prophet expects two required columns:
      - 'ds': Datetime timestamp representing the observation date.
      - 'y': Continuous target variable (sales volume clipped to non-negative values).

    Additionally, this function attaches external regressors:
      - 'snap': State-specific SNAP food stamp subsidy activation indicator.
      - 'has_event': Indicator of major calendar, cultural, or religious events.

    Args:
        series: Master time series for a single SKU/Store.
        state_id: State abbreviation ('CA', 'TX', 'WI') to resolve correct SNAP benefits.

    Returns:
        pd.DataFrame: Clean DataFrame conforming to Prophet input specifications.
    """
    snap_col = f"snap_{state_id}" if state_id in ("CA", "TX", "WI") else None
    
    prop_df = pd.DataFrame({
        "ds": pd.to_datetime(series["date"]),
        "y":  series["sales"].clip(lower=0).astype(float),
    })
    
    # Extract state-specific SNAP subsidy flag if available in the source series
    if snap_col and snap_col in series.columns:
        prop_df["snap"] = series[snap_col].values
    elif "snap" in series.columns:
        prop_df["snap"] = series["snap"].values
    else:
        prop_df["snap"] = 0
    
    # Extract event occurrence indicator
    prop_df["has_event"] = series["has_event"].values if "has_event" in series.columns else 0
    return prop_df


def fit_prophet(
    series: pd.DataFrame,
    state_id: str,
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0
) -> Tuple[Any, pd.DataFrame]:
    """
    Instantiate and train a Prophet Generalized Additive Model.

    Hyperparameter Rationale:
      - `changepoint_prior_scale=0.05`: Restricts rate of trend changes at inflection points.
        Higher values risk fitting historical noise; lower values make trend too rigid.
      - `seasonality_prior_scale=10.0`: Allows moderate flexibility for weekly Fourier cycles.
      - `interval_width=0.95`: Simulates posterior draws to construct empirical 95%
        prediction intervals for downstream Newsvendor safety stock determination.

    Args:
        series: Historical daily sales series.
        state_id: Store state identifier.
        changepoint_prior_scale: Laplace prior scale on trend changepoints.
        seasonality_prior_scale: Normal prior scale on Fourier seasonality terms.

    Returns:
        Tuple[Prophet, pd.DataFrame]: (fitted_model, training_dataframe)
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("Prophet package is not installed. Run: pip install prophet")

    prop_df = build_prophet_df(series, state_id)

    # Configure additive decomposition structure
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        interval_width=0.95,
    )
    
    # Register exogenous covariates
    m.add_regressor("snap")
    m.add_regressor("has_event")

    # Fit GAM via Stan optimization
    m.fit(prop_df)
    return m, prop_df


def prophet_forecast(
    series: pd.DataFrame,
    state_id: str,
    horizon: int = 28
) -> pd.DataFrame:
    """
    Generate out-of-sample demand forecasts with 95% uncertainty intervals.

    Workflow:
      1. Fits the Prophet additive model on historical sales.
      2. Constructs a future calendar grid spanning `horizon` consecutive days.
      3. Populates future exogenous regressors (SNAP and events).
      4. Evaluates posterior prediction intervals (yhat, yhat_lower, yhat_upper).
      5. Post-processes outputs: clips lower bounds at 0 (retail demand cannot be negative).

    Args:
        series: Daily sales time series DataFrame.
        state_id: State jurisdiction ('CA', 'TX', 'WI').
        horizon: Planning horizon in days (default: 28).

    Returns:
        pd.DataFrame: Contains columns ['date', 'yhat', 'yhat_lower', 'yhat_upper'].
    """
    try:
        from prophet import Prophet
    except ImportError:
        logger.warning("Prophet not available in environment — returning empty fallback.")
        return pd.DataFrame(columns=["date", "yhat", "yhat_lower", "yhat_upper"])

    m, prop_df = fit_prophet(series, state_id)

    # Create future date spine
    future = m.make_future_dataframe(periods=horizon)
    future["snap"]      = 0
    future["has_event"] = 0
    
    # Merge known historical regressor values back in
    future = future.merge(
        prop_df[["ds", "snap", "has_event"]],
        on="ds",
        how="left",
        suffixes=("_x", "")
    )
    if "snap_x" in future.columns:
        future["snap"] = future["snap"].combine_first(future["snap_x"]).fillna(0)
        future.drop(columns=["snap_x"], inplace=True)
    if "has_event_x" in future.columns:
        future["has_event"] = future["has_event"].combine_first(future["has_event_x"]).fillna(0)
        future.drop(columns=["has_event_x"], inplace=True)

    # Perform inference and extract posterior distribution
    forecast = m.predict(future)

    # Isolate out-of-sample horizon (strictly after historical cutoff)
    last_train_date = prop_df["ds"].max()
    future_fc = forecast[forecast["ds"] > last_train_date].copy()

    # Retail sales non-negativity constraint
    future_fc["yhat"]       = future_fc["yhat"].clip(lower=0)
    future_fc["yhat_lower"] = future_fc["yhat_lower"].clip(lower=0)
    future_fc["yhat_upper"] = future_fc["yhat_upper"].clip(lower=0)

    # Standardize column naming for API consumer consistency
    result = future_fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "date"})
    result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
    result.reset_index(drop=True, inplace=True)
    return result
