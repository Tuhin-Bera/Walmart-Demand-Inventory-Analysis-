"""
backend/services/reconciliation.py — Hierarchical Time Series Reconciliation (Darts Bottom-Up).

========================================================================================
THEORETICAL FOUNDATION:
Retail supply chains operate as multi-level spatial and organizational hierarchies:
  Level 0 (Total): Total Walmart Enterprise Unit Sales.
  Level 1 (State): California (CA), Texas (TX), Wisconsin (WI).
  Level 2 (Store): CA_1, TX_1, WI_1.
  Level 3 (SKU): Individual item products (HOBBIES_1_001, etc.).

When forecasts are produced independently at different levels ("base forecasts" $\\hat{y}$),
they are almost never coherent:
  $\\sum_{i \\in \\text{store}} \\hat{y}_{i, t} \\neq \\hat{y}_{\\text{store}, t}$
This discrepancy causes severe operational friction: regional procurement budgets do not
match store manager purchase orders, leading to over-budgeting or supply misallocation.

Reconciliation solves this by projecting base forecasts into the coherent subspace
defined by the Summing Matrix $S$:
  $y_t = S \\cdot b_t$
Where:
  - $b_t$ represents the vector of bottom-level series (Level 3 SKU sales).
  - $S$ is a binary structural matrix mapping bottom-level components to all aggregate nodes.
  - Bottom-Up Reconciliation ensures perfect coherence by re-aggregating SKU forecasts:
    $\\tilde{y}_t = S \\cdot \\hat{b}_t$
========================================================================================
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


def build_hierarchy_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame, List[str], List[str]]:
    """
    Construct the structural Summing Matrix $S$ representing the M5 retail hierarchy.

    Mathematical Representation:
      Let $n_b$ be the number of bottom-level series (SKU $\\times$ Store) and $n_a$
      be the number of aggregated series (Top + States + Stores).
      $S$ is an $(n_{total} \\times n_b)$ matrix where $S_{i, j} = 1$ if bottom series $j$
      contributes to aggregate series $i$, and $0$ otherwise.

    Args:
        df: Processed DataFrame containing item_id, store_id, and state_id.

    Returns:
        Tuple containing:
          - S: Binary summing matrix of shape $(n_{total}, n_b)$.
          - bottom_series: DataFrame of bottom-level identifiers.
          - stores: List of distinct store identifiers.
          - states: List of distinct state identifiers.
    """
    # Identify unique bottom-level time series
    bottom_series = (
        df.groupby(["item_id", "store_id"], observed=True)
          .size()
          .reset_index()[["item_id", "store_id"]]
    )
    n_bottom = len(bottom_series)
    
    stores = [str(s) for s in df["store_id"].unique()]
    states = [str(s) for s in df["state_id"].unique()]
    
    n_stores = len(stores)
    n_states = len(states)
    n_total  = 1 + n_states + n_stores + n_bottom   # Top + States + Stores + Items
    
    S = np.zeros((n_total, n_bottom), dtype=float)
    
    # Row 0: Top-level aggregate (sums all bottom-level series)
    S[0, :] = 1.0
    
    # Rows 1 to n_states: State-level sums
    for i, state in enumerate(states):
        state_items = bottom_series[
            bottom_series["store_id"].astype(str).str.startswith(state)
        ].index.tolist()
        for j in state_items:
            S[1 + i, j] = 1.0
    
    # Rows (1 + n_states) to (1 + n_states + n_stores): Store-level sums
    for i, store in enumerate(stores):
        store_items = bottom_series[
            bottom_series["store_id"].astype(str) == store
        ].index.tolist()
        for j in store_items:
            S[1 + n_states + i, j] = 1.0
    
    # Bottom-level rows: Identity matrix block (each item maps directly to itself)
    for i in range(n_bottom):
        S[1 + n_states + n_stores + i, i] = 1.0
    
    return S, bottom_series, stores, states


def reconcile_with_darts(
    base_forecasts: Dict[str, np.ndarray],
    df: pd.DataFrame
) -> Dict[str, np.ndarray]:
    """
    Apply bottom-up hierarchical reconciliation across SKU, Store, and State tiers.

    Guarantees structural coherence:
      $\\tilde{y}_{\\text{store}} = \\sum_{i \\in \\text{store}} \\tilde{y}_i$
      $\\tilde{y}_{\\text{state}} = \\sum_{s \\in \\text{state}} \\tilde{y}_s$

    Args:
        base_forecasts: Dictionary mapping series keys to 1D numpy forecast arrays.
        df: Master dataset providing hierarchical mapping relationships.

    Returns:
        Dict[str, np.ndarray]: Reconciled, structurally coherent forecast arrays.
    """
    return _bottom_up_reconcile(base_forecasts, df)


def _bottom_up_reconcile(
    base_forecasts: Dict[str, np.ndarray],
    df: pd.DataFrame
) -> Dict[str, np.ndarray]:
    """
    Bottom-up aggregation engine.
    
    Aggregates point predictions sequentially from lowest granular level (item/store)
    up through store fulfillment centers and state jurisdictions.
    """
    reconciled = dict(base_forecasts)
    
    # Aggregate bottom-level items to store level
    stores = [str(s) for s in df["store_id"].unique()]
    for store in stores:
        items = [str(i) for i in df[df["store_id"] == store]["item_id"].unique()]
        item_keys = [f"{item}__{store}" for item in items]
        available = [k for k in item_keys if k in base_forecasts]
        if available:
            store_sum = np.sum([base_forecasts[k] for k in available], axis=0)
            reconciled[f"store__{store}"] = store_sum

    # Aggregate store totals to state level
    states = [str(s) for s in df["state_id"].unique()]
    for state in states:
        state_stores = [s for s in stores if s.startswith(state)]
        store_keys   = [f"store__{s}" for s in state_stores]
        available    = [k for k in store_keys if k in reconciled]
        if available:
            state_sum = np.sum([reconciled[k] for k in available], axis=0)
            reconciled[f"state__{state}"] = state_sum

    return reconciled


def get_store_forecast_reconciled(
    item_forecasts: pd.DataFrame,
    store_id: str
) -> pd.DataFrame:
    """
    Aggregate SKU-level predictions into a coherent store-level forecast DataFrame.

    Sums expected demand ($yhat$) and accumulates conservative 95% uncertainty intervals.
    """
    agg = item_forecasts.groupby("date").agg(
        yhat       = ("yhat",       "sum"),
        yhat_lower = ("yhat_lower", "sum"),
        yhat_upper = ("yhat_upper", "sum"),
    ).reset_index()
    agg["store_id"] = store_id
    return agg


def get_state_forecast_reconciled(
    store_forecasts: List[pd.DataFrame]
) -> pd.DataFrame:
    """
    Aggregate multiple store-level forecasts into a coherent state-level forecast DataFrame.
    """
    combined = pd.concat(store_forecasts, ignore_index=True)
    agg = combined.groupby("date").agg(
        yhat       = ("yhat",       "sum"),
        yhat_lower = ("yhat_lower", "sum"),
        yhat_upper = ("yhat_upper", "sum"),
    ).reset_index()
    return agg
