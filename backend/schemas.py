"""
backend/schemas.py
==================
Pydantic Data Models and Validation Schemas for the FastAPI REST Service.

WHY THIS EXISTS:
Strict schemas serve multiple critical functions:
1. Automated request body validation (rejects invalid types before hitting models).
2. Type serialization that guarantees clean JSON response outputs for the UI.
3. Auto-generation of OpenAPI documentation (/docs and /redoc Swagger UI).
4. Explicit definition of data contracts across forecasting and inventory domains.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# 1. Forecasting Primitives & Metric Schemas
# ─────────────────────────────────────────────────────────────────────────────

class PredictionInterval(BaseModel):
    """
    Represents a single forecasted timestamp with confidence uncertainty bounds.
    
    Business relevance:
    Rather than displaying an unrealistic single point forecast, providing lower
    and upper percentiles (typically 10th and 90th or 2.5th and 97.5th percentiles)
    empowers store managers to prepare for both conservative and surge demand scenarios.
    """
    date: str = Field(..., description="ISO-8601 calendar date string (YYYY-MM-DD)")
    yhat: float = Field(..., description="Expected mean point forecast in units sold")
    yhat_lower: float = Field(..., description="Lower confidence bound (e.g., 2.5% quantile)")
    yhat_upper: float = Field(..., description="Upper confidence bound (e.g., 97.5% quantile)")


class ModelMetrics(BaseModel):
    """
    Standardized model evaluation accuracy metrics computed against held-out actuals.
    
    - RMSE: Root Mean Squared Error (penalizes large outlier mistakes heavily).
    - MAE:  Mean Absolute Error (average magnitude of errors).
    - WAPE: Weighted Absolute Percentage Error (sum of errors / sum of actual sales).
            Far superior to MAPE on zero-demand intermittent series.
    - MASE: Mean Absolute Scaled Error (< 1.0 indicates model beats naive seasonal baseline).
    """
    model: str = Field(..., description="Name of the model algorithm (naive, prophet, lightgbm, tft)")
    rmse:  float = Field(..., description="Root Mean Squared Error")
    mae:   float = Field(..., description="Mean Absolute Error")
    wape:  float = Field(..., description="Weighted Absolute Percentage Error (ratio 0.0 to 1.0+)")
    mase:  Optional[float] = Field(None, description="Mean Absolute Scaled Error relative to naive")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Demand Forecast Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    """
    Payload for initiating a multi-step demand forecast.
    """
    item_id: str = Field(..., example="HOBBIES_1_001", description="Unique Walmart SKU product code")
    store_id: str = Field(..., example="CA_1", description="Store identifier (e.g., CA_1, TX_1, WI_1)")
    horizon: int = Field(28, ge=1, le=90, description="Forward forecast horizon in days (default: 28 days)")
    model: str = Field("lightgbm", description="Algorithm engine: naive | prophet | lightgbm | tft | all")


class ForecastResponse(BaseModel):
    """
    Complete forecast payload delivered to the client.
    """
    item_id:   str
    store_id:  str
    model:     str
    horizon:   int
    forecasts: List[PredictionInterval] = Field(..., description="Chronological sequence of predictions")
    metrics:   Optional[ModelMetrics] = Field(None, description="Accuracy scorecard on hold-out data")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hierarchical Forecasting Schema
# ─────────────────────────────────────────────────────────────────────────────

class HierarchyForecastResponse(BaseModel):
    """
    Aggregated forecasts reconciled across the product / store / state hierarchy.
    """
    level: str = Field(..., description="Aggregation level: item | store | state")
    group: str = Field(..., description="Grouping identifier (e.g., CA_1 for store, CA for state)")
    forecasts: List[PredictionInterval]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Inventory Optimization Schemas
# ─────────────────────────────────────────────────────────────────────────────

class InventoryRequest(BaseModel):
    """
    Input parameters for the OR-Tools Newsvendor linear program.
    Balances holding surplus units against the penalty of lost sales.
    """
    item_id:         str = Field(..., description="Product SKU code")
    store_id:        str = Field(..., description="Store facility code")
    current_stock:   float = Field(..., ge=0, description="Current on-hand inventory level in units")
    forecast_demand: float = Field(..., ge=0, description="Predicted mean demand over the review period")
    demand_std:      float = Field(0.0, ge=0, description="Standard deviation of demand forecast for safety stock")
    holding_cost:    float = Field(0.20, ge=0, description="Holding/carrying cost per unit stored ($)")
    stockout_cost:   float = Field(2.00, ge=0, description="Stockout shortage cost per lost sale ($)")
    max_order_qty:   Optional[float] = Field(None, description="Optional physical warehouse/truck capacity limit")
    budget:          Optional[float] = Field(None, description="Optional capital budget limit ($)")
    unit_price:      Optional[float] = Field(None, description="Unit cost ($) required for budget calculations")


class InventoryResponse(BaseModel):
    """
    Recommended order quantity, safety buffer, and cost impact output.
    """
    item_id:              str
    store_id:             str
    current_stock:        float
    forecast_demand:      float
    safety_stock:         float = Field(..., description="Buffer units required to achieve target service level")
    reorder_point:        float = Field(..., description="Threshold inventory level that triggers an order")
    recommended_order:    float = Field(..., description="Optimal order units determined by the LP solver")
    estimated_total_cost: float = Field(..., description="Expected combined holding and stockout penalty ($)")
    solver_status:        str   = Field(..., description="Status returned by OR-Tools: OPTIMAL, FEASIBLE, or FALLBACK")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Dashboard Summary KPI Schema
# ─────────────────────────────────────────────────────────────────────────────

class DashboardSummary(BaseModel):
    """
    Executive summary payload driving the dashboard home view.
    """
    total_items:              int
    total_stores:             int
    top_items:                List[Dict[str, Any]]
    state_sales:              List[Dict[str, Any]]
    category_sales:           List[Dict[str, Any]]
    daily_trend:              List[Dict[str, Any]]
