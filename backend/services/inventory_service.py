"""
backend/services/inventory_service.py — Mathematical Inventory Optimization via OR-Tools & Newsvendor Theory.

========================================================================================
THEORETICAL FOUNDATION:
In enterprise retail supply chains, converting demand forecasts into purchasing orders
presents a fundamental trade-off governed by Newsvendor Economics:
  1. Over-ordering leads to excess holding/carrying costs $c_h$ (tied-up working capital,
     warehousing, spoilage, depreciation).
  2. Under-ordering leads to stockout/shortage penalties $c_s$ (lost margin, broken customer trust).

This module solves for the optimal replenishment order quantity $q^*$ through two complementary methods:

1. OR-Tools GLOP Linear Programming Formulation:
   Variables:
     - $q \ge 0$: Replenishment order quantity to procure.
     - $h \ge 0$: Slack variable representing excess inventory carried past period end.
     - $s \ge 0$: Slack variable representing unsatisfied shortage / lost sales units.

   Constraints:
     - Inventory Balance: $\text{current\_stock} + q - (\text{forecast\_demand} + \text{safety\_stock}) = h - s$
     - Procurement Budget: $q \cdot \text{unit\_price} \le \text{budget} - (\text{current\_stock} \cdot \text{unit\_price})$
     - Storage Capacity: $q \le \text{max\_order\_qty}$

   Objective:
     $\min_{\{q, h, s\}} \quad c_h \cdot h + c_s \cdot s$

2. Analytic Newsvendor Solution (Fallback):
   When unconstrained by multi-item budgets, the classical stochastic Newsvendor theorem
   dictates the optimal in-stock service level via the Critical Ratio:
     $CR = \frac{c_s}{c_s + c_h}$
   The optimal target inventory level is:
     $S^* = \mu + z^* \cdot \sigma$
   Where $z^* = \Phi^{-1}(CR)$ is evaluated via high-precision Beasley-Springer-Moro rational approximation.
========================================================================================
"""

import numpy as np
import logging
from math import sqrt
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


# ── Portable High-Precision Normal PPF (Inverse Gaussian CDF) ─────────────────

def _normal_ppf(p: float) -> float:
    """
    Evaluate the Standard Normal Percent-Point Function $z = \\Phi^{-1}(p)$ with $|error| < 4.5 \\times 10^{-4}$.

    Why standard library math does not suffice:
      Python's built-in `math` module does not contain `erfinv` or normal quantile functions.
      Rather than introducing a heavy compiled `scipy` dependency, this implements the
      Beasley-Springer-Moro rational minimax approximation algorithm.
      It splits the probability domain $(0, 1)$ into a central region $|p - 0.5| < 0.42$
      and extreme tail regions, computing rational polynomial quotients.

    Args:
        p: Target probability (service level or critical ratio, $0 < p < 1$).

    Returns:
        float: Corresponding standard normal z-score.
    """
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")

    # Prefer scipy if already present in environment
    try:
        from scipy.stats import norm
        return float(norm.ppf(p))
    except ImportError:
        pass

    # Beasley-Springer-Moro minimax coefficients
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
         0.0276438810333863, 0.0038405729373609, 0.0003951896511349,
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187]

    y = p - 0.5
    if abs(y) < 0.42:
        # Central region rational polynomial
        r = y * y
        x = y * (((a[3]*r + a[2])*r + a[1])*r + a[0]) / \
               ((((b[3]*r + b[2])*r + b[1])*r + b[0])*r + 1.0)
    else:
        # Asymptotic tail expansion
        r = p if y < 0 else 1.0 - p
        r = sqrt(-2.0 * __import__("math").log(r))
        x = c[0] + r*(c[1] + r*(c[2] + r*(c[3] + r*(c[4] + r*(
            c[5] + r*(c[6] + r*(c[7] + r*c[8])))))))
        if y < 0:
            x = -x

    return float(x)


# ─────────────────────────────────────────────────────────────────────────────
# Core Constrained Optimization Solver
# ─────────────────────────────────────────────────────────────────────────────

def solve_order_quantity(
    forecast_demand:    float,
    current_stock:      float,
    demand_std:         float  = 0.0,
    holding_cost:       float  = 0.20,
    stockout_cost:      float  = 2.00,
    max_order_qty:      Optional[float] = None,
    budget:             Optional[float] = None,
    unit_price:         Optional[float] = None,
    safety_stock_z:     float  = 1.65,   # Default z = 1.65 for 95% service level
) -> Dict[str, Any]:
    """
    Solve for optimal replenishment order quantity using Google OR-Tools GLOP solver.

    Args:
        forecast_demand: Expected cumulative demand over the planning horizon ($\\mu$).
        current_stock: Physical on-hand inventory already present in the warehouse.
        demand_std: Forecast error volatility ($\\sigma$) over the planning horizon.
        holding_cost: Unit inventory holding cost per period ($c_h$).
        stockout_cost: Unit stockout penalty / lost customer margin ($c_s$).
        max_order_qty: Optional upper bound on supplier order capacity.
        budget: Optional financial budget ceiling for replenishment spend.
        unit_price: Purchase price per SKU unit.
        safety_stock_z: Normal z-score corresponding to desired cycle-service level.

    Returns:
        Dict[str, Any]: Optimization plan with recommended order, safety stock,
                        reorder point, expected holding/stockout costs, and solver status.
    """
    # Defensive parameter sanitization
    forecast_demand = max(0.0, float(forecast_demand))
    current_stock   = max(0.0, float(current_stock))
    demand_std      = max(0.0, float(demand_std))
    holding_cost    = max(0.0, float(holding_cost))
    stockout_cost   = max(0.0, float(stockout_cost))

    # Classical safety stock formulation: SS = z * sigma
    safety_stock = safety_stock_z * demand_std

    # Attempt instantiation of Google OR-Tools GLOP linear programming solver
    use_ortools = False
    try:
        from ortools.linear_solver import pywraplp
        solver = pywraplp.Solver.CreateSolver("GLOP")
        if solver:
            use_ortools = True
        else:
            logger.warning("OR-Tools GLOP solver failed to instantiate — falling back to analytical engine.")
    except ImportError:
        logger.warning("OR-Tools not available in runtime — falling back to analytical engine.")

    if use_ortools:
        INF  = solver.infinity()
        q_ub = float(max_order_qty) if max_order_qty is not None else INF
        q    = solver.NumVar(0.0, q_ub, "order_quantity")
        h    = solver.NumVar(0.0, INF,  "excess_units_held")
        s    = solver.NumVar(0.0, INF,  "shortage_units_lost")

        # Inventory Balance Equation: Current + Order - TargetDemand == Excess - Shortage
        target = forecast_demand + safety_stock
        solver.Add(current_stock + q - target == h - s)

        # Budget Ceiling Constraint: q * unit_price <= budget
        if budget is not None and unit_price is not None and float(unit_price) > 0:
            max_affordable = float(budget) / float(unit_price)
            solver.Add(q <= max(0.0, max_affordable))

        # Objective Function: Minimize weighted economic regret
        obj = solver.Objective()
        obj.SetCoefficient(h, holding_cost)
        obj.SetCoefficient(s, stockout_cost)
        obj.SetMinimization()

        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            order_qty     = max(0.0, round(q.solution_value(), 2))
            excess_units  = max(0.0, h.solution_value())
            short_units   = max(0.0, s.solution_value())
            solver_status = "OPTIMAL"
        else:
            # Solver infeasible or unbounded fallback
            order_qty     = max(0.0, forecast_demand + safety_stock - current_stock)
            excess_units  = max(0.0, current_stock + order_qty - forecast_demand)
            short_units   = max(0.0, forecast_demand - current_stock - order_qty)
            solver_status = "FALLBACK"

    else:
        # Analytical Newsvendor formulation: CR = c_s / (c_s + c_h)
        denom          = stockout_cost + holding_cost
        critical_ratio = stockout_cost / denom if denom > 0 else 0.5
        z_star         = _normal_ppf(critical_ratio)
        target_stock   = forecast_demand + z_star * demand_std
        order_qty      = max(0.0, target_stock - current_stock)

        # Apply hard procurement constraints
        if max_order_qty is not None:
            order_qty = min(order_qty, float(max_order_qty))
        if budget is not None and unit_price is not None and float(unit_price) > 0:
            affordable = float(budget) / float(unit_price)
            order_qty  = min(order_qty, max(0.0, affordable))

        order_qty     = round(order_qty, 2)
        excess_units  = max(0.0, current_stock + order_qty - forecast_demand)
        short_units   = max(0.0, forecast_demand - current_stock - order_qty)
        solver_status = "ANALYTIC"

    # Compute expected financial outcomes
    estimated_holding  = round(holding_cost  * excess_units, 4)
    estimated_stockout = round(stockout_cost * short_units,  4)
    total_cost         = round(estimated_holding + estimated_stockout, 4)
    reorder_point      = round(forecast_demand + safety_stock, 2)

    return {
        "recommended_order":      order_qty,
        "safety_stock":           round(safety_stock, 2),
        "reorder_point":          reorder_point,
        "estimated_total_cost":   total_cost,
        "estimated_holding_cost": estimated_holding,
        "estimated_stockout_cost":estimated_stockout,
        "solver_status":          solver_status,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Item Portfolio Batch Allocation
# ─────────────────────────────────────────────────────────────────────────────

def batch_optimize(items: List[Dict[str, Any]], total_budget: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Perform multi-item portfolio replenishment optimization under shared budget limits.

    When aggregate capital is constrained ($Budget < \\sum Q_i \\cdot Price_i$),
    capital is allocated proportionally to each SKU's unconstrained demand shortfall,
    prioritizing products with severe stockout risk.

    Args:
        items: List of item dictionaries with keys (item_id, store_id, current_stock, forecast_demand).
        total_budget: Aggregate procurement capital ceiling across all items.

    Returns:
        List[Dict[str, Any]]: Replenishment recommendations for each SKU.
    """
    if not items:
        return []

    # Calculate aggregate unconstrained procurement cost
    total_unconstrained = sum(
        float(i.get("unit_price") or 1.0) * max(0.0, float(i["forecast_demand"]) - float(i["current_stock"]))
        for i in items
    )

    results = []
    for item in items:
        # Proportionate budget share based on demand gap
        item_budget = None
        if total_budget is not None and total_unconstrained > 0:
            gap = float(item.get("unit_price") or 1.0) * max(
                0.0, float(item["forecast_demand"]) - float(item["current_stock"])
            )
            share       = gap / (total_unconstrained + 1e-9)
            item_budget = share * total_budget

        result = solve_order_quantity(
            forecast_demand = float(item["forecast_demand"]),
            current_stock   = float(item["current_stock"]),
            demand_std      = float(item.get("demand_std") or 0.0),
            holding_cost    = float(item.get("holding_cost") or 0.20),
            stockout_cost   = float(item.get("stockout_cost") or 2.00),
            max_order_qty   = item.get("max_order_qty"),
            budget          = item_budget,
            unit_price      = item.get("unit_price"),
        )
        result["item_id"]  = item["item_id"]
        result["store_id"] = item["store_id"]
        results.append(result)

    return results
