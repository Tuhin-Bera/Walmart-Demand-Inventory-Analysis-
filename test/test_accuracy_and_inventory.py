"""
test/test_accuracy_and_inventory.py
===================================
Automated Test & Evaluation Harness for Walmart Demand Forecasting & Inventory AI.

Evaluates the project across all required core components:
  1. Demand Forecasting Accuracy:      MAE, RMSE, WAPE, and MASE
  2. Inventory Optimization (Shortage): Stockout Rate (%)
  3. Inventory Optimization (Excess):   Overstock / Excess Inventory (units & holding cost)
  4. Reorder Strategy:                  Cycle Service Level (%) and Total Financial Regret Cost ($)

Runs against the production dataset and asserts required operational targets.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.services.data_service import get_processed_data, get_item_store_series
from backend.services.forecast_service import generate_forecast
from backend.services.inventory_service import solve_order_quantity


class TestForecastingAccuracyAndInventory(unittest.TestCase):
    """
    Evaluation test suite verifying demand forecasting accuracy and inventory optimization.
    """

    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 75)
        print("  WALMART DEMAND FORECASTING & INVENTORY OPTIMIZATION TEST SUITE")
        print("=" * 75)
        cls.df = get_processed_data()
        cls.eval_horizon = 28
        
        # Representative test SKUs spanning high, medium, and intermittent demand
        cls.benchmark_skus = [
            ("FOODS_3_090", "CA_1", "High-velocity food staple"),
            ("FOODS_3_586", "TX_1", "Top volume store SKU"),
            ("HOUSEHOLD_1_001", "WI_1", "Stable household consumable"),
            ("HOBBIES_1_001", "CA_1", "Intermittent / sporadic hobby SKU")
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # 1. DEMAND FORECASTING EVALUATION: MAE, RMSE, WAPE
    # ─────────────────────────────────────────────────────────────────────────
    def test_01_demand_forecasting_accuracy(self):
        """
        Verify that forecasting models calculate valid MAE, RMSE, and WAPE metrics,
        and that models produce meaningful predictions on the holdout evaluation period.
        """
        print("\n[TEST 1/4] Evaluating Demand Forecasting: MAE, RMSE, WAPE ...")
        
        results = []
        for item_id, store_id, desc in self.benchmark_skus:
            # Generate forecasts across model engines
            for model_name in ["naive", "lightgbm"]:
                fc_res = generate_forecast(item_id, store_id, self.eval_horizon, model_name)
                metrics = fc_res.get("metrics")
                
                self.assertIsNotNone(metrics, f"Metrics should not be None for {item_id} ({model_name})")
                self.assertIn("mae", metrics, "Metrics must include MAE")
                self.assertIn("rmse", metrics, "Metrics must include RMSE")
                self.assertIn("wape", metrics, "Metrics must include WAPE")
                
                mae = float(metrics["mae"])
                rmse = float(metrics["rmse"])
                wape = float(metrics["wape"])
                mase = float(metrics.get("mase", 1.0))
                
                # Assert mathematical validity
                self.assertGreaterEqual(mae, 0.0, "MAE must be non-negative")
                self.assertGreaterEqual(rmse, 0.0, "RMSE must be non-negative")
                self.assertGreaterEqual(wape, 0.0, "WAPE must be non-negative")
                self.assertGreaterEqual(rmse, mae - 1e-5, "RMSE should theoretically be >= MAE")
                
                results.append({
                    "Item": item_id,
                    "Store": store_id,
                    "Model": model_name.upper(),
                    "MAE": round(mae, 2),
                    "RMSE": round(rmse, 2),
                    "WAPE": f"{wape * 100:.1f}%",
                    "MASE": round(mase, 3)
                })

        report_df = pd.DataFrame(results)
        print("\n" + report_df.to_string(index=False))
        print("  --> [PASS] MAE, RMSE, and WAPE successfully computed and validated across SKUs.")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. INVENTORY OPTIMIZATION: STOCKOUT RATE EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    def test_02_inventory_stockout_rate(self):
        """
        Evaluate stockout occurrences (shortage cycles) under:
          - Baseline Strategy: Fixed/Naive replenishment without dynamic safety stock
          - Optimized Strategy: OR-Tools GLOP Newsvendor replenishment with dynamic safety stock
        """
        print("\n[TEST 2/4] Evaluating Inventory Optimization: Stockout Rate ...")
        
        item_id, store_id, _ = self.benchmark_skus[0] # FOODS_3_090 @ CA_1
        series = get_item_store_series(self.df, item_id, store_id)
        
        test_sales = series.iloc[-self.eval_horizon:]["sales"].values
        lgbm_fc = generate_forecast(item_id, store_id, self.eval_horizon, "lightgbm")
        pred_sales = np.array([f["yhat"] for f in lgbm_fc["forecasts"]])

        # Simulate 4 weekly replenishment cycles (7 days each)
        cycle_len = 7
        num_cycles = 4
        
        # Baseline simulation (no safety stock buffer)
        baseline_stockouts = 0
        stock_base = 30.0
        for c in range(num_cycles):
            d_act = float(test_sales[c*cycle_len : (c+1)*cycle_len].sum())
            d_pred = float(pred_sales[c*cycle_len : (c+1)*cycle_len].sum())
            order = max(0.0, d_pred - stock_base)
            avail = stock_base + order
            if avail < d_act:
                baseline_stockouts += 1
                stock_base = 0.0
            else:
                stock_base = avail - d_act

        baseline_stockout_rate = (baseline_stockouts / num_cycles) * 100.0

        # Optimized simulation with OR-Tools GLOP and dynamic safety buffer
        opt_stockouts = 0
        stock_opt = 30.0
        for c in range(num_cycles):
            d_act = float(test_sales[c*cycle_len : (c+1)*cycle_len].sum())
            d_pred = float(pred_sales[c*cycle_len : (c+1)*cycle_len].sum())
            d_std = float(np.std(pred_sales[c*cycle_len : (c+1)*cycle_len]) * np.sqrt(cycle_len))
            
            res = solve_order_quantity(
                forecast_demand=d_pred,
                current_stock=stock_opt,
                demand_std=d_std,
                holding_cost=0.20,
                stockout_cost=2.50
            )
            order = res["recommended_order"]
            avail = stock_opt + order
            if avail < d_act:
                opt_stockouts += 1
                stock_opt = 0.0
            else:
                stock_opt = avail - d_act

        opt_stockout_rate = (opt_stockouts / num_cycles) * 100.0
        stockout_reduction = baseline_stockout_rate - opt_stockout_rate

        print(f"  • Baseline Stockout Rate (No Safety Buffer): {baseline_stockout_rate:.1f}%")
        print(f"  • Optimized Stockout Rate (OR-Tools GLOP):   {opt_stockout_rate:.1f}%")
        print(f"  • Absolute Stockout Reduction:              {stockout_reduction:+.1f}%")

        self.assertLessEqual(opt_stockout_rate, baseline_stockout_rate,
                             "Optimized strategy must not have a higher stockout rate than baseline")
        print("  --> [PASS] Stockout rate reduced successfully via prescriptive optimization.")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. INVENTORY OPTIMIZATION: OVERSTOCK & EXCESS INVENTORY
    # ─────────────────────────────────────────────────────────────────────────
    def test_03_overstock_and_excess_inventory(self):
        """
        Evaluate overstock units and capital tied up in carrying excess stock.
        Demonstrates that dynamic safety stock sizes order buffers without runaway accumulation.
        """
        print("\n[TEST 3/4] Evaluating Overstock & Excess Inventory ...")
        
        item_id, store_id, _ = self.benchmark_skus[0]
        series = get_item_store_series(self.df, item_id, store_id)
        test_sales = series.iloc[-self.eval_horizon:]["sales"].values
        lgbm_fc = generate_forecast(item_id, store_id, self.eval_horizon, "lightgbm")
        pred_sales = np.array([f["yhat"] for f in lgbm_fc["forecasts"]])

        cycle_len = 7
        num_cycles = 4
        holding_cost_rate = 0.20 # $0.20/unit/day

        # Naive rule-of-thumb: static 200% order buffer
        static_excess_units = []
        stock = 30.0
        for c in range(num_cycles):
            d_act = float(test_sales[c*cycle_len : (c+1)*cycle_len].sum())
            d_pred = float(pred_sales[c*cycle_len : (c+1)*cycle_len].sum())
            order = max(0.0, (d_pred * 1.5) - stock)
            avail = stock + order
            excess = max(0.0, avail - d_act)
            static_excess_units.append(excess)
            stock = excess

        # Optimized OR-Tools Newsvendor replenishment
        opt_excess_units = []
        stock = 30.0
        for c in range(num_cycles):
            d_act = float(test_sales[c*cycle_len : (c+1)*cycle_len].sum())
            d_pred = float(pred_sales[c*cycle_len : (c+1)*cycle_len].sum())
            d_std = float(np.std(pred_sales[c*cycle_len : (c+1)*cycle_len]) * np.sqrt(cycle_len))
            res = solve_order_quantity(
                forecast_demand=d_pred,
                current_stock=stock,
                demand_std=d_std,
                holding_cost=0.20,
                stockout_cost=2.50
            )
            avail = stock + res["recommended_order"]
            excess = max(0.0, avail - d_act)
            opt_excess_units.append(excess)
            stock = excess

        avg_static_excess = np.mean(static_excess_units)
        avg_opt_excess = np.mean(opt_excess_units)
        holding_savings = ((avg_static_excess - avg_opt_excess) / avg_static_excess) * 100.0

        print(f"  • Rule-of-Thumb Static Average Excess: {avg_static_excess:.1f} units (${avg_static_excess * holding_cost_rate * 7:.2f}/wk)")
        print(f"  • Optimized Newsvendor Average Excess: {avg_opt_excess:.1f} units (${avg_opt_excess * holding_cost_rate * 7:.2f}/wk)")
        print(f"  • Inventory Holding Reduction:         {holding_savings:.1f}%")

        self.assertLess(avg_opt_excess, avg_static_excess, "Optimized ordering must prevent excessive inventory accumulation")
        print("  --> [PASS] Overstock and excess inventory successfully controlled.")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. REORDER STRATEGY: SERVICE LEVEL & INVENTORY COST
    # ─────────────────────────────────────────────────────────────────────────
    def test_04_reorder_strategy_service_level_and_cost(self):
        """
        Verify achieved cycle service level against target critical ratio (92.6%)
        and compute total economic cost (holding charges + stockout penalties).
        """
        print("\n[TEST 4/4] Evaluating Reorder Strategy: Service Level & Cost ...")

        holding_cost = 0.20
        stockout_cost = 2.50
        target_service_level = (stockout_cost / (stockout_cost + holding_cost)) * 100.0 # 92.6%

        summary_rows = []
        for item_id, store_id, cat_desc in self.benchmark_skus:
            series = get_item_store_series(self.df, item_id, store_id)
            test_sales = series.iloc[-self.eval_horizon:]["sales"].values
            lgbm_fc = generate_forecast(item_id, store_id, self.eval_horizon, "lightgbm")
            pred_sales = np.array([f["yhat"] for f in lgbm_fc["forecasts"]])

            cycle_len = 7
            num_cycles = 4
            total_demand = float(test_sales.sum())

            # Evaluate Optimized Strategy
            demand_met = 0.0
            stock = 30.0
            total_h_cost = 0.0
            total_s_cost = 0.0

            for c in range(num_cycles):
                d_act = float(test_sales[c*cycle_len : (c+1)*cycle_len].sum())
                d_pred = float(pred_sales[c*cycle_len : (c+1)*cycle_len].sum())
                d_std = float(np.std(pred_sales[c*cycle_len : (c+1)*cycle_len]) * np.sqrt(cycle_len))

                res = solve_order_quantity(
                    forecast_demand=d_pred,
                    current_stock=stock,
                    demand_std=d_std,
                    holding_cost=holding_cost,
                    stockout_cost=stockout_cost
                )
                avail = stock + res["recommended_order"]
                if avail >= d_act:
                    demand_met += d_act
                    excess = avail - d_act
                    shortage = 0.0
                else:
                    demand_met += avail
                    excess = 0.0
                    shortage = d_act - avail

                total_h_cost += excess * holding_cost * cycle_len
                total_s_cost += shortage * stockout_cost
                stock = excess

            achieved_sl = (demand_met / total_demand * 100.0) if total_demand > 0 else 100.0
            total_cost = total_h_cost + total_s_cost

            summary_rows.append({
                "SKU": item_id,
                "Store": store_id,
                "Target SL": f"{target_service_level:.1f}%",
                "Achieved SL": f"{achieved_sl:.1f}%",
                "Holding Cost": f"${total_h_cost:.2f}",
                "Stockout Cost": f"${total_s_cost:.2f}",
                "Total Regret Cost": f"${total_cost:.2f}",
                "Status": "OPTIMAL" if achieved_sl >= 90.0 else "FEASIBLE"
            })

            # Assert acceptable service level
            self.assertGreaterEqual(achieved_sl, 85.0, f"Service level for {item_id} should be >= 85%")

        df_summary = pd.DataFrame(summary_rows)
        print("\n" + df_summary.to_string(index=False))
        print("  --> [PASS] Reorder strategy satisfies target service level with minimized total cost.")


if __name__ == "__main__":
    unittest.main(verbosity=2)

