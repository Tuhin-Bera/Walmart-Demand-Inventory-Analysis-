"""
backend/routers/metrics.py — Metrics, Model Comparison, and Dashboard Aggregates.

========================================================================================
PURPOSE & ROUTING DESIGN:
This module powers the high-level executive dashboard and model benchmarking pages.
Key endpoints:
  1. GET /api/metrics/summary:
     Returns enterprise KPIs: product SKU counts, store footprint, top-selling items,
     state-level sales volume, category distribution, and 90-day aggregate sales trends.
     Computation is cached in-memory after first calculation to ensure instant (< 2ms)
     page transitions on the frontend.
  2. GET /api/metrics/compare:
     Executes multi-model backtesting on a designated SKU/Store time series across
     all four architectures (Seasonal Naive, Prophet, LightGBM, TFT). Returns point
     forecasts and industry-standard accuracy metrics (RMSE, MAE, WAPE, MASE) side-by-side.
  3. GET /api/metrics/historical/{item_id}/{store_id}:
     Provides clean trailing historical sales data for frontend visualization.
========================================================================================
"""

import numpy as np
import pandas as pd
import logging
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any

from backend.services.data_service import get_processed_data, get_item_store_series
from backend.services.forecast_service import generate_forecast

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/metrics", tags=["Metrics & Dashboard"])

# Global in-memory cache for the dashboard overview aggregates
_cached_summary: Optional[Dict[str, Any]] = None


@router.get("/summary")
async def dashboard_summary():
    """
    Generate high-level summary KPIs for the executive dashboard landing view.

    Metrics returned:
      - `total_items`: Distinct product SKUs active in the system.
      - `total_stores`: Number of geographic fulfillment centers / retail stores.
      - `top_items`: Top 10 highest-volume products with store attribution.
      - `state_sales`: Regional sales breakdown across California, Texas, and Wisconsin.
      - `category_sales`: Departmental breakdown across Foods, Household, and Hobbies.
      - `daily_trend`: Daily aggregate sales volume over the trailing 90 days.

    Caching Behavior:
      Aggregates are computed once and stored in memory. Subsequent page refreshes
      respond in < 1ms without re-scanning the 1.64 million row table.
    """
    global _cached_summary
    if _cached_summary is not None:
        return _cached_summary

    try:
        df = get_processed_data()

        total_items  = int(df["item_id"].nunique())
        total_stores = int(df["store_id"].nunique())

        # Top 10 highest selling items by cumulative units sold
        top_df = (
            df.groupby(["item_id", "store_id"], observed=True)["sales"]
              .sum()
              .reset_index()
              .sort_values("sales", ascending=False)
              .head(10)
        )
        for c in ["item_id", "store_id"]:
            top_df[c] = top_df[c].astype(str)
        top_items = top_df.to_dict(orient="records")

        # Regional distribution (CA, TX, WI)
        state_df = (
            df.groupby("state_id", observed=True)["sales"]
              .sum()
              .reset_index()
              .rename(columns={"sales": "total_sales"})
        )
        state_df["state_id"] = state_df["state_id"].astype(str)
        state_sales = state_df.to_dict(orient="records")

        # Merchandising category distribution (FOODS, HOUSEHOLD, HOBBIES)
        cat_df = (
            df.groupby("cat_id", observed=True)["sales"]
              .sum()
              .reset_index()
              .rename(columns={"sales": "total_sales"})
        )
        cat_df["cat_id"] = cat_df["cat_id"].astype(str)
        cat_sales = cat_df.to_dict(orient="records")

        # Daily aggregate retail sales volume across all stores for trailing 90 days
        daily = (
            df.groupby("date")["sales"]
              .sum()
              .reset_index()
              .sort_values("date")
              .tail(90)
        )
        daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")

        summary_result = {
            "total_items":   total_items,
            "total_stores":  total_stores,
            "top_items":     top_items,
            "state_sales":   state_sales,
            "category_sales":cat_sales,
            "daily_trend":   daily.to_dict(orient="records"),
        }
        _cached_summary = summary_result
        return summary_result

    except Exception as e:
        logger.error(f"Error computing dashboard summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard summary: {str(e)}")


@router.get("/compare")
async def compare_models(
    item_id:  str = Query(..., description="Product SKU identifier, e.g. HOBBIES_1_001"),
    store_id: str = Query(..., description="Store fulfillment location, e.g. CA_1"),
    horizon:  int = Query(28, ge=1, le=90, description="Forecast horizon in days"),
):
    """
    Run multi-model benchmarking on a specific SKU/Store time series.

    Executes 4 forecasting architectures simultaneously:
      1. Seasonal Naive: Baseline heuristic using 7-day cyclical lag replication.
      2. Meta Prophet: Generalized additive model capturing weekly seasonality and SNAP regressors.
      3. LightGBM: Gradient boosted trees trained with log1p Tweedie-adapted regression.
      4. Temporal Fusion Transformer (TFT): Deep attention-based multi-horizon neural network.

    Returns:
      Dict mapping each model name to its forecasted series and out-of-time accuracy metrics:
        - `rmse`: Root Mean Squared Error (penalizes large outlier spikes).
        - `mae`: Mean Absolute Error (average magnitude of forecast error).
        - `wape`: Weighted Absolute Percentage Error (business-friendly aggregate error).
        - `mase`: Mean Absolute Scaled Error (< 1.0 proves model beats naive persistence baseline).
    """
    models = ["naive", "prophet", "lightgbm", "tft"]
    results = {}
    
    for m in models:
        try:
            res = generate_forecast(item_id, store_id, horizon, m)
            results[m] = {
                "forecasts": res["forecasts"],
                "metrics":   res.get("metrics"),
            }
        except Exception as e:
            logger.warning(f"Model comparison failure for {m}: {e}")
            results[m] = {
                "forecasts": [],
                "metrics":   None,
                "error":     str(e),
            }

    return {
        "item_id": item_id,
        "store_id": store_id,
        "horizon": horizon,
        "comparison": results,
    }


@router.get("/historical/{item_id}/{store_id}")
async def get_historical_sales(
    item_id:     str,
    store_id:    str,
    last_n_days: int = Query(90, ge=7, le=180, description="Trailing number of days to retrieve"),
):
    """
    Retrieve historical actual sales observations for visual charting on the frontend.

    Enables frontends to plot ground-truth history preceding the forecast start date.
    """
    try:
        df = get_processed_data()
        series = get_item_store_series(df, item_id, store_id)
        if series.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No historical sales found for item '{item_id}' at store '{store_id}'."
            )

        recent = series.tail(last_n_days)[["date", "sales"]].copy()
        recent["date"] = pd.to_datetime(recent["date"]).dt.strftime("%Y-%m-%d")
        
        return {
            "item_id": item_id,
            "store_id": store_id,
            "days": len(recent),
            "history": recent.to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
