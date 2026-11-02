"""
backend/routers/forecast.py — Demand Forecasting Endpoints & SKU Metadata Discovery.

========================================================================================
ROUTER PURPOSE & API DESIGN:
Provides REST endpoints for querying SKU items, store locations, and executing demand
forecasts across multiple models (Seasonal Naive, Prophet, LightGBM, TFT).

Endpoints:
  1. POST /api/forecast/
     Main body-driven forecast generation endpoint accepting a structured ForecastRequest.
  2. GET /api/forecast/quick/{item_id}/{store_id}
     Lightweight URL-parameter endpoint for fast interactive frontend querying.
  3. GET /api/forecast/items
     Instant metadata lookup returning unique item SKUs and category tags.
  4. GET /api/forecast/stores
     Returns the active fulfillment stores (CA_1, TX_1, WI_1).
========================================================================================
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any

from backend.schemas import ForecastRequest, ForecastResponse, PredictionInterval, ModelMetrics
from backend.services.forecast_service import generate_forecast, get_available_items
from backend.config import SAMPLE_STORES

router = APIRouter(prefix="/api/forecast", tags=["Forecasting"])


@router.post("/", response_model=Dict[str, Any])
async def create_forecast(request: ForecastRequest):
    """
    Generate out-of-sample demand forecast for a designated SKU item and store fulfillment node.

    Payload parameters:
      - `item_id`: Target SKU identifier (e.g., 'HOBBIES_1_001').
      - `store_id`: Fulfillment center location (e.g., 'CA_1').
      - `horizon`: Multi-step forward forecast horizon in days (1 to 90).
      - `model`: Forecasting engine: 'naive' | 'prophet' | 'lightgbm' | 'tft' | 'all'.

    Returns:
      Forecast series with point predictions and 95% uncertainty confidence intervals,
      plus holdout evaluation metrics (RMSE, MAE, WAPE, MASE).
    """
    try:
        result = generate_forecast(
            item_id    = request.item_id,
            store_id   = request.store_id,
            horizon    = request.horizon,
            model_name = request.model,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demand forecasting failed: {str(e)}")


@router.get("/quick/{item_id}/{store_id}", response_model=Dict[str, Any])
async def quick_forecast(
    item_id:  str,
    store_id: str,
    horizon:  int = Query(28, ge=1, le=90, description="Planning horizon in days"),
    model:    str = Query("lightgbm", description="Model architecture: lightgbm | prophet | naive | tft"),
):
    """
    Fast GET endpoint for interactive dashboard query execution without POST payload serialization.
    """
    try:
        result = generate_forecast(item_id, store_id, horizon, model)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demand forecasting failed: {str(e)}")


@router.get("/items", response_model=List[Dict[str, Any]])
async def list_items(store_id: Optional[str] = Query(None, description="Optional store filter, e.g. CA_1")):
    """
    Retrieve available product SKUs with category and department taxonomy metadata.
    Responds in < 1ms using cached metadata.
    """
    try:
        return get_available_items(store_id=store_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve SKU inventory: {str(e)}")


@router.get("/stores", response_model=List[str])
async def list_stores():
    """
    Retrieve active fulfillment store identifiers in the M5 benchmark dataset.
    """
    return SAMPLE_STORES
