"""
backend/routers/inventory.py — Inventory Optimization Endpoints via Google OR-Tools.

========================================================================================
ROUTER PURPOSE & ARCHITECTURE:
Exposes mathematical optimization endpoints that convert demand predictions into
actionable, cost-optimal inventory replenishment recommendations.

Endpoints:
  1. POST /api/inventory/optimize
     Solves the single-SKU constrained Newsvendor LP problem via OR-Tools GLOP.
  2. POST /api/inventory/optimize/batch
     Solves multi-SKU portfolio allocation subject to an aggregate working capital budget.
  3. GET /api/inventory/quick/{item_id}/{store_id}
     Enables instant parametric simulation for dashboard sliders and widgets.
========================================================================================
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any

from backend.schemas import InventoryRequest, InventoryResponse
from backend.services.inventory_service import solve_order_quantity, batch_optimize

router = APIRouter(prefix="/api/inventory", tags=["Inventory Optimization"])


@router.post("/optimize", response_model=InventoryResponse)
async def optimize_single_item(req: InventoryRequest):
    """
    Calculate the mathematically optimal order quantity for a single product SKU.

    Minimizes aggregate financial regret:
      Cost = (holding_cost * excess_units) + (stockout_cost * shortage_units)
    Subject to on-hand inventory, forecast variance, supplier order capacity, and budget.
    """
    try:
        result = solve_order_quantity(
            forecast_demand = req.forecast_demand,
            current_stock   = req.current_stock,
            demand_std      = req.demand_std,
            holding_cost    = req.holding_cost,
            stockout_cost   = req.stockout_cost,
            max_order_qty   = req.max_order_qty,
            budget          = req.budget,
            unit_price      = req.unit_price,
        )
        return InventoryResponse(
            item_id              = req.item_id,
            store_id             = req.store_id,
            current_stock        = req.current_stock,
            forecast_demand      = req.forecast_demand,
            safety_stock         = result["safety_stock"],
            reorder_point        = result["reorder_point"],
            recommended_order    = result["recommended_order"],
            estimated_total_cost = result["estimated_total_cost"],
            solver_status        = result["solver_status"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inventory optimization failed: {str(e)}")


@router.post("/optimize/batch", response_model=List[Dict[str, Any]])
async def optimize_batch(items: List[InventoryRequest], budget: Optional[float] = None):
    """
    Simultaneously optimize inventory orders across a portfolio of multiple items.
    
    If `budget` is specified, procurement spend is apportioned proportionally across
    products based on their demand gap to prevent capital exhaustion.
    """
    try:
        item_dicts = [
            {
                "item_id":         i.item_id,
                "store_id":        i.store_id,
                "current_stock":   i.current_stock,
                "forecast_demand": i.forecast_demand,
                "demand_std":      i.demand_std,
                "holding_cost":    i.holding_cost,
                "stockout_cost":   i.stockout_cost,
                "max_order_qty":   i.max_order_qty,
                "unit_price":      i.unit_price,
            }
            for i in items
        ]
        results = batch_optimize(item_dicts, total_budget=budget)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inventory optimization failed: {str(e)}")


@router.get("/quick/{item_id}/{store_id}", response_model=Dict[str, Any])
async def quick_inventory(
    item_id:         str,
    store_id:        str,
    current_stock:   float = Query(50.0, ge=0, description="Current on-hand stock"),
    forecast_demand: float = Query(100.0, ge=0, description="Expected demand over horizon"),
    demand_std:      float = Query(15.0, ge=0, description="Demand standard deviation"),
    holding_cost:    float = Query(0.20, ge=0, description="Carrying cost per unit"),
    stockout_cost:   float = Query(2.00, ge=0, description="Stockout penalty per unit"),
):
    """
    Fast parametric query for frontend scenario modeling and live slider manipulation.
    """
    result = solve_order_quantity(
        forecast_demand = forecast_demand,
        current_stock   = current_stock,
        demand_std      = demand_std,
        holding_cost    = holding_cost,
        stockout_cost   = stockout_cost,
    )
    result["item_id"]  = item_id
    result["store_id"] = store_id
    return result
