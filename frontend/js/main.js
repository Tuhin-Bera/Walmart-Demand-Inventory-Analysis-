/**
 * frontend/js/main.js — Single-Page Dashboard Orchestrator & User Interaction Controller.
 *
 * ======================================================================================
 * ARCHITECTURAL DESIGN & USER EXPERIENCE (UX):
 * This module manages the single-page application (SPA) state, page transitions,
 * asynchronous data fetching, and rich interactive components:
 *
 * Primary Modules:
 *   1. System Toast & Formatting Services: Formats numbers, currencies, and notifications.
 *   2. SPA Router: Smooth client-side view switching without full browser reloads.
 *   3. Overview Page: Enterprise executive KPIs, 90-day aggregate trends, category donuts.
 *   4. Demand Forecasting Engine: Store & SKU selection, horizon controls, 95% confidence
 *      intervals, CSV data export, and quick-selection SKU pills.
 *   5. Model Comparison: Side-by-side benchmarking of Naive, Prophet, LightGBM, and TFT.
 *   6. OR-Tools Inventory Optimizer: Dynamic Newsvendor Critical Ratio modeling, safety
 *      stock calculations, and visual inventory status gauges.
 *   7. Hierarchical Reconciliation: Multi-level aggregation view verifying coherence.
 * ======================================================================================
 */

// ── Global System Toast Notifications ─────────────────────────────────────────

/**
 * Display a temporary floating toast message on screen.
 *
 * @param {string} message - Text notification.
 * @param {string} type - 'success' | 'error' | 'info'.
 */
function showToast(message, type = "success") {
  const container = document.getElementById("toast-container");
  if (!container) return;
  const t = document.createElement("div");
  t.className = `toast ${type}`;
  t.innerHTML = `<strong>${type === "error" ? "⚠️ Error:" : type === "info" ? "ℹ️ Notice:" : "✓"}</strong> ${message}`;
  container.appendChild(t);
  setTimeout(() => {
    t.style.opacity = "0";
    t.style.transform = "translateY(10px)";
    setTimeout(() => t.remove(), 300);
  }, 4200);
}
window.showToast = showToast;

// ── Numeric & Currency Formatting Services ────────────────────────────────────

function fmt(n, decimals = 1) {
  if (n == null || isNaN(n)) return "—";
  return parseFloat(n).toFixed(decimals);
}

function fmtInt(n) {
  if (n == null || isNaN(n)) return "—";
  return Math.round(n).toLocaleString();
}
window.fmt = fmt;
window.fmtInt = fmtInt;

// ── Single-Page Router ────────────────────────────────────────────────────────

const pages = {
  overview:   loadOverview,
  forecast:   loadForecastPage,
  compare:    loadComparePage,
  inventory:  loadInventoryPage,
  hierarchy:  loadHierarchyPage,
};

let currentPage = "overview";

function activatePage(pageName) {
  currentPage = pageName;
  document.querySelectorAll(".nav-item").forEach(el => {
    el.classList.toggle("active", el.dataset.page === pageName);
  });
  const loader = pages[pageName];
  if (loader) loader();
}
window.activatePage = activatePage;

// ─────────────────────────────────────────────────────────────────────────────
// 1. Executive Overview Page
// ─────────────────────────────────────────────────────────────────────────────

async function loadOverview() {
  const content = document.getElementById("main-content");
  content.innerHTML = `
    <div class="page-header">
      <h1>📊 Executive Demand &amp; Inventory Overview</h1>
      <p>Walmart M5 Nationwide Demand Forecasting &amp; Multi-Level Prescriptive Inventory Optimization</p>
    </div>

    <!-- High-level Metric Stat Cards -->
    <div class="stat-grid">
      <div class="stat-card blue">
        <div class="label">Total Products (SKUs)</div>
        <div class="value" id="st-items">3,049</div>
        <div class="sub">Across Foods, Household &amp; Hobbies</div>
      </div>
      <div class="stat-card green">
        <div class="label">Active Stores</div>
        <div class="value" id="st-stores">3</div>
        <div class="sub">CA_1, TX_1, WI_1 benchmark stores</div>
      </div>
      <div class="stat-card orange">
        <div class="label">States Covered</div>
        <div class="value">3</div>
        <div class="sub">California · Texas · Wisconsin</div>
      </div>
      <div class="stat-card purple">
        <div class="label">Forecast Horizon</div>
        <div class="value">28 Days</div>
        <div class="sub">Multi-step recursive rollout</div>
      </div>
    </div>

    <!-- Aggregate Sales Trends and Merchandising Category Donut -->
    <div class="chart-grid">
      <div class="card">
        <div class="card-header">
          <h3>📈 Total Daily Sales Trend (Trailing History)</h3>
        </div>
        <div class="chart-container" style="position:relative;height:300px;">
          <canvas id="chart-trend"></canvas>
        </div>
      </div>
      <div class="card">
        <div class="card-header">
          <h3>🗂️ Sales Distribution by Category</h3>
        </div>
        <div class="chart-container" style="position:relative;height:300px;">
          <canvas id="chart-cat"></canvas>
        </div>
      </div>
    </div>

    <!-- Regional Performance and Top Products Table -->
    <div class="chart-grid">
      <div class="card">
        <div class="card-header">
          <h3>🗺️ Regional Sales Volume (By State)</h3>
        </div>
        <div class="chart-container" style="position:relative;height:280px;">
          <canvas id="chart-state"></canvas>
        </div>
      </div>
      <div class="card">
        <div class="card-header">
          <h3>🏆 Top 10 Bestselling Products</h3>
          <span style="font-size:0.75rem;color:var(--text-sec);">Click item to forecast</span>
        </div>
        <div style="overflow-x:auto;">
          <table class="data-table">
            <thead>
              <tr>
                <th>Product SKU</th>
                <th>Store</th>
                <th>Units Sold</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody id="top-items-body">
              <tr><td colspan="4" style="text-align:center;color:var(--text-sec);">Loading top items…</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `;

  try {
    const data = await API.getDashboardSummary();

    if (data.total_items)  document.getElementById("st-items").textContent  = fmtInt(data.total_items);
    if (data.total_stores) document.getElementById("st-stores").textContent = data.total_stores;

    // Render interactive charts
    if (data.daily_trend?.length) {
      Charts.renderDailyTrend("chart-trend", data.daily_trend);
    }
    if (data.category_sales?.length) {
      Charts.renderCategoryDonut("chart-cat", data.category_sales);
    }
    if (data.state_sales?.length) {
      Charts.renderStateBars("chart-state", data.state_sales);
    }

    // Hydrate top products table with interactive quick-forecast actions
    const tbody = document.getElementById("top-items-body");
    if (tbody && data.top_items?.length) {
      tbody.innerHTML = data.top_items.map(r => `
        <tr>
          <td><strong>${r.item_id}</strong></td>
          <td><span class="badge badge-blue">${r.store_id}</span></td>
          <td>${fmtInt(r.sales)}</td>
          <td>
            <button class="btn-action" onclick="quickJumpToForecast('${r.item_id}', '${r.store_id}')">
              🔮 Forecast
            </button>
          </td>
        </tr>
      `).join("");
    }
  } catch (e) {
    showToast("Failed to load dashboard summary: " + e.message, "error");
  }
}

/**
 * Quick jump from top items table directly to demand forecast tab.
 */
function quickJumpToForecast(itemId, storeId) {
  activatePage("forecast");
  setTimeout(() => {
    const storeEl = document.getElementById("fc-store");
    if (storeEl) storeEl.value = storeId;
    loadItemsForStore().then(() => {
      const itemEl = document.getElementById("fc-item");
      if (itemEl) itemEl.value = itemId;
      runForecast();
    });
  }, 100);
}
window.quickJumpToForecast = quickJumpToForecast;

// ─────────────────────────────────────────────────────────────────────────────
// 2. Demand Forecasting Engine Page
// ─────────────────────────────────────────────────────────────────────────────

// Global cache of latest forecast for CSV export
let _latestForecastData = null;

async function loadForecastPage() {
  const content = document.getElementById("main-content");
  content.innerHTML = `
    <div class="page-header">
      <h1>📈 Demand Forecasting Engine</h1>
      <p>Generate point predictions and calibrated 95% uncertainty prediction intervals for any product &amp; store.</p>
    </div>

    <!-- Quick-Selection Pills for Popular SKUs -->
    <div class="quick-pill-bar">
      <span class="pill-label">Popular SKUs:</span>
      <button class="pill-btn" onclick="selectQuickItem('HOBBIES_1_001')">HOBBIES_1_001</button>
      <button class="pill-btn" onclick="selectQuickItem('FOODS_1_001')">FOODS_1_001</button>
      <button class="pill-btn" onclick="selectQuickItem('HOUSEHOLD_1_001')">HOUSEHOLD_1_001</button>
      <button class="pill-btn" onclick="selectQuickItem('FOODS_3_090')">FOODS_3_090</button>
    </div>

    <!-- Interactive Control Bar -->
    <div class="control-bar">
      <div class="control-group">
        <label>Store Location</label>
        <select id="fc-store">
          <option value="CA_1">CA_1 (California)</option>
          <option value="TX_1">TX_1 (Texas)</option>
          <option value="WI_1">WI_1 (Wisconsin)</option>
        </select>
      </div>

      <div class="control-group">
        <label>Product Item (SKU)</label>
        <select id="fc-item">
          <option value="HOBBIES_1_001">HOBBIES_1_001</option>
        </select>
      </div>

      <div class="control-group">
        <label>Forecasting Engine</label>
        <select id="fc-model">
          <option value="lightgbm">LightGBM (Gradient Boosted Trees)</option>
          <option value="prophet">Meta Prophet (+ SNAP Regressors)</option>
          <option value="naive">Seasonal Naive (7-Day Baseline)</option>
          <option value="tft">TFT (Temporal Fusion Transformer)</option>
        </select>
      </div>

      <div class="control-group">
        <label>Horizon (Days)</label>
        <select id="fc-horizon">
          <option value="14">14 Days (Bi-Weekly)</option>
          <option value="28" selected>28 Days (Monthly Benchmark)</option>
          <option value="42">42 Days (6 Weeks)</option>
          <option value="60">60 Days (Quarterly Plan)</option>
        </select>
      </div>

      <button class="btn btn-primary" onclick="runForecast()">
        <span>🔮</span> Generate Forecast
      </button>

      <button class="btn btn-outline" onclick="exportForecastCSV()" id="btn-export-csv" title="Download 28-day schedule as CSV">
        <span>📥</span> Export CSV
      </button>
    </div>

    <!-- Master Forecast Plotly Visualizer -->
    <div class="card">
      <div class="card-header">
        <h3 id="fc-chart-title">Actual Sales vs Forecast with 95% Confidence Band</h3>
        <div id="fc-metrics-chips" class="metric-chips"></div>
      </div>
      <div id="fc-chart" style="height:400px;position:relative;">
        <div class="spinner-wrap">
          <div class="spinner"></div>
          <span>Generating forecast model inference…</span>
        </div>
      </div>
    </div>

    <!-- Aggregate Confidence Summary & Detailed Daily Schedule Table -->
    <div class="chart-grid">
      <div class="card">
        <div class="card-header"><h3>📊 Demand Confidence Summary</h3></div>
        <div id="fc-interval-box">
          <div class="interval-box">
            <div class="interval-label">Average Daily Demand</div>
            <div class="interval-main">—</div>
            <div class="interval-range">Run forecast to evaluate confidence intervals</div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <h3>📋 Daily Replenishment Schedule</h3>
          <span style="font-size:0.75rem;color:var(--text-sec);">First 14 days of forecast horizon</span>
        </div>
        <div style="overflow-x:auto;">
          <table class="data-table">
            <thead>
              <tr>
                <th>Calendar Date</th>
                <th>Expected (Yhat)</th>
                <th>95% Lower</th>
                <th>95% Upper</th>
                <th>Variance (Spread)</th>
              </tr>
            </thead>
            <tbody id="fc-table-body">
              <tr><td colspan="5" style="text-align:center;color:var(--text-sec);">Generate forecast to view schedule</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  `;

  // Attach store change listener
  const storeSelect = document.getElementById("fc-store");
  if (storeSelect) {
    storeSelect.addEventListener("change", () => loadItemsForStore().then(runForecast));
  }

  // Populate SKUs for initial store and trigger first forecast
  await loadItemsForStore();
  await runForecast();
}

/**
 * Load product SKUs for the currently selected store.
 */
async function loadItemsForStore() {
  const storeId = document.getElementById("fc-store")?.value || "CA_1";
  const itemSelect = document.getElementById("fc-item");
  if (!itemSelect) return;

  try {
    const items = await API.getItems(storeId);
    if (items && items.length > 0) {
      itemSelect.innerHTML = items.map(i =>
        `<option value="${i.item_id}">${i.item_id} (${i.cat_id || "Retail"})</option>`
      ).join("");
    }
  } catch (e) {
    console.warn("Item load note:", e.message);
  }
}

/**
 * Quick select SKU from pill clicks.
 */
function selectQuickItem(sku) {
  const itemSelect = document.getElementById("fc-item");
  if (!itemSelect) return;
  
  let exists = false;
  for (let opt of itemSelect.options) {
    if (opt.value === sku) { exists = true; break; }
  }
  if (!exists) {
    const newOpt = document.createElement("option");
    newOpt.value = sku;
    newOpt.text = `${sku} (Quick SKU)`;
    itemSelect.appendChild(newOpt);
  }
  itemSelect.value = sku;
  runForecast();
}
window.selectQuickItem = selectQuickItem;

/**
 * Master forecast execution pipeline: calls backend and renders Plotly visuals.
 */
async function runForecast() {
  const itemId  = document.getElementById("fc-item")?.value || "HOBBIES_1_001";
  const storeId = document.getElementById("fc-store")?.value || "CA_1";
  const model   = document.getElementById("fc-model")?.value || "lightgbm";
  const horizon = parseInt(document.getElementById("fc-horizon")?.value || "28");

  const chartEl = document.getElementById("fc-chart");
  if (chartEl) {
    chartEl.innerHTML = `<div class="spinner-wrap"><div class="spinner"></div><span>Running ${model.toUpperCase()} multi-step inference…</span></div>`;
  }

  try {
    // Fetch trailing history and machine learning predictions concurrently
    const [historical, fcData] = await Promise.all([
      API.getHistorical(itemId, storeId, 90),
      API.getForecast(itemId, storeId, horizon, model),
    ]);

    _latestForecastData = { itemId, storeId, model, forecasts: fcData.forecasts };

    // Render interactive Plotly chart with 95% uncertainty band
    Charts.renderForecastChart("fc-chart", historical.history || [], fcData.forecasts || []);

    const titleEl = document.getElementById("fc-chart-title");
    if (titleEl) {
      titleEl.textContent = `${itemId} @ ${storeId} — ${model.toUpperCase()} (${horizon}-Day Horizon)`;
    }

    // Display validation accuracy metrics
    const m = fcData.metrics;
    const chipsEl = document.getElementById("fc-metrics-chips");
    if (chipsEl && m) {
      chipsEl.innerHTML = `
        <div class="metric-chip"><div class="chip-label">RMSE</div><div class="chip-value">${fmt(m.rmse)}</div></div>
        <div class="metric-chip"><div class="chip-label">MAE</div><div class="chip-value">${fmt(m.mae)}</div></div>
        <div class="metric-chip"><div class="chip-label">WAPE</div><div class="chip-value">${fmt(m.wape * 100)}%</div></div>
        ${m.mase != null ? `<div class="metric-chip"><div class="chip-label">MASE</div><div class="chip-value">${fmt(m.mase)}</div></div>` : ""}
      `;
    }

    // Compute planning aggregates
    const fc = fcData.forecasts;
    if (fc && fc.length) {
      const avgYhat  = fc.reduce((a, b) => a + b.yhat, 0) / fc.length;
      const avgLower = fc.reduce((a, b) => a + b.yhat_lower, 0) / fc.length;
      const avgUpper = fc.reduce((a, b) => a + b.yhat_upper, 0) / fc.length;
      const totYhat  = fc.reduce((a, b) => a + b.yhat, 0);

      const intervalBox = document.getElementById("fc-interval-box");
      if (intervalBox) {
        intervalBox.innerHTML = `
          <div class="interval-box">
            <div class="interval-label">Average Expected Daily Demand</div>
            <div class="interval-main">${fmt(avgYhat, 1)} units / day</div>
            <div class="interval-range">95% Uncertainty Band: ${fmt(avgLower, 1)} – ${fmt(avgUpper, 1)} units</div>
          </div>
          <div style="margin-top:16px;">
            <div class="interval-box" style="border-color:rgba(16,185,129,0.3);background:rgba(16,185,129,0.06);">
              <div class="interval-label" style="color:#34d399;">Total Demand Forecast (${horizon} Days)</div>
              <div class="interval-main" style="color:#34d399;">${fmtInt(totYhat)} units</div>
              <div class="interval-range">Cumulative Procurement Volume</div>
            </div>
          </div>
        `;
      }

      // Populate detailed 14-day schedule table
      const tbody = document.getElementById("fc-table-body");
      if (tbody) {
        tbody.innerHTML = fc.slice(0, 14).map(r => {
          const spread = (r.yhat_upper - r.yhat_lower);
          return `
            <tr>
              <td><strong>${r.date}</strong></td>
              <td><span style="color:#60a5fa;font-weight:600;">${fmt(r.yhat, 1)}</span></td>
              <td style="color:var(--text-sec);">${fmt(r.yhat_lower, 1)}</td>
              <td style="color:var(--text-sec);">${fmt(r.yhat_upper, 1)}</td>
              <td style="color:var(--text-sec);font-size:0.8rem;">±${fmt(spread / 2, 1)}</td>
            </tr>
          `;
        }).join("");
      }
    }

    showToast(`Forecast complete for ${itemId} (${model.toUpperCase()})`);
  } catch (e) {
    if (chartEl) {
      chartEl.innerHTML = `<div class="spinner-wrap"><span style="color:#ef4444;">⚠️ Forecast failed: ${e.message}</span></div>`;
    }
    showToast("Forecast failed: " + e.message, "error");
  }
}

/**
 * Export the current forecast schedule directly as a downloadable CSV.
 */
function exportForecastCSV() {
  if (!_latestForecastData || !_latestForecastData.forecasts?.length) {
    showToast("No active forecast data to export.", "error");
    return;
  }

  const { itemId, storeId, model, forecasts } = _latestForecastData;
  let csvContent = "data:text/csv;charset=utf-8,";
  csvContent += "Date,Item_ID,Store_ID,Model,Forecast_Demand_Yhat,Lower_Bound_95,Upper_Bound_95\n";

  forecasts.forEach(r => {
    csvContent += `${r.date},${itemId},${storeId},${model},${r.yhat.toFixed(2)},${r.yhat_lower.toFixed(2)},${r.yhat_upper.toFixed(2)}\n`;
  });

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `forecast_${itemId}_${storeId}_${model}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  showToast("CSV export downloaded successfully.");
}
window.exportForecastCSV = exportForecastCSV;
window.runForecast = runForecast;
window.loadItemsForStore = loadItemsForStore;

// ─────────────────────────────────────────────────────────────────────────────
// 3. Model Benchmarking & Comparison Page
// ─────────────────────────────────────────────────────────────────────────────

async function loadComparePage() {
  const content = document.getElementById("main-content");
  content.innerHTML = `
    <div class="page-header">
      <h1>⚖️ Model Benchmarking &amp; Comparison</h1>
      <p>Evaluate Seasonal Naive, Meta Prophet, LightGBM, and TFT side-by-side on identical holdout data.</p>
    </div>

    <div class="control-bar">
      <div class="control-group">
        <label>Store</label>
        <select id="cmp-store">
          <option value="CA_1">CA_1</option>
          <option value="TX_1">TX_1</option>
          <option value="WI_1">WI_1</option>
        </select>
      </div>

      <div class="control-group">
        <label>Product Item</label>
        <select id="cmp-item">
          <option value="HOBBIES_1_001">HOBBIES_1_001</option>
        </select>
      </div>

      <div class="control-group">
        <label>Horizon</label>
        <select id="cmp-horizon">
          <option value="28" selected>28 Days (Standard)</option>
          <option value="14">14 Days</option>
        </select>
      </div>

      <button class="btn btn-primary" onclick="runComparison()">
        <span>⚖️</span> Compare Models
      </button>
    </div>

    <!-- Multi-Model Overlay Plot -->
    <div class="card">
      <div class="card-header">
        <h3>Multi-Model Trajectory Comparison</h3>
      </div>
      <div id="cmp-chart" style="height:380px;">
        <div class="spinner-wrap"><div class="spinner"></div><span>Running side-by-side benchmark evaluation…</span></div>
      </div>
    </div>

    <!-- Accuracy Leaderboard and Comparative Bar Chart -->
    <div class="chart-grid">
      <div class="card">
        <div class="card-header"><h3>🏆 Accuracy Leaderboard</h3></div>
        <div style="overflow-x:auto;">
          <table class="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>WAPE</th>
                <th>RMSE</th>
                <th>MAE</th>
                <th>MASE</th>
                <th>Rank</th>
              </tr>
            </thead>
            <tbody id="cmp-table-body">
              <tr><td colspan="6" style="text-align:center;color:var(--text-sec);">Running benchmark…</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <div class="card-header"><h3>📊 Relative Error Comparison (Lower is Better)</h3></div>
        <div class="chart-container" style="position:relative;height:280px;">
          <canvas id="cmp-bar-chart"></canvas>
        </div>
      </div>
    </div>
  `;

  // Hydrate items and execute comparison
  const storeId = document.getElementById("cmp-store")?.value || "CA_1";
  try {
    const items = await API.getItems(storeId);
    if (items?.length) {
      document.getElementById("cmp-item").innerHTML = items.map(i =>
        `<option value="${i.item_id}">${i.item_id}</option>`
      ).join("");
    }
  } catch (e) {}

  runComparison();
}

async function runComparison() {
  const itemId  = document.getElementById("cmp-item")?.value  || "HOBBIES_1_001";
  const storeId = document.getElementById("cmp-store")?.value || "CA_1";
  const horizon = parseInt(document.getElementById("cmp-horizon")?.value || "28");

  const chartEl = document.getElementById("cmp-chart");
  if (chartEl) {
    chartEl.innerHTML = `<div class="spinner-wrap"><div class="spinner"></div><span>Benchmarking all 4 architectures…</span></div>`;
  }

  try {
    const data = await API.compareModels(itemId, storeId, horizon);
    const comp = data.comparison;

    // Render multi-line Plotly overlay
    Charts.renderMultiModelForecast("cmp-chart", comp);

    // Render error bar chart
    Charts.renderMetricsComparison("cmp-bar-chart", comp);

    // Build leaderboard sorted by lowest WAPE
    const sorted = Object.entries(comp)
      .filter(([_, d]) => d.metrics != null)
      .sort((a, b) => (a[1].metrics.wape ?? 999) - (b[1].metrics.wape ?? 999));

    const tbody = document.getElementById("cmp-table-body");
    if (tbody) {
      tbody.innerHTML = sorted.map(([name, d], idx) => {
        const m = d.metrics;
        const isWinner = idx === 0;
        return `
          <tr ${isWinner ? 'style="background:rgba(16,185,129,0.06);"' : ""}>
            <td>
              <strong>${name.toUpperCase()}</strong>
              ${isWinner ? '<span class="badge badge-green" style="margin-left:6px;">🏆 Best Model</span>' : ""}
            </td>
            <td><strong>${fmt(m.wape * 100, 1)}%</strong></td>
            <td>${fmt(m.rmse, 2)}</td>
            <td>${fmt(m.mae, 2)}</td>
            <td>${fmt(m.mase, 2)}</td>
            <td><span class="badge badge-blue">#${idx + 1}</span></td>
          </tr>
        `;
      }).join("");
    }

    showToast("Model comparison finished successfully.");
  } catch (e) {
    if (chartEl) {
      chartEl.innerHTML = `<div class="spinner-wrap"><span style="color:#ef4444;">⚠️ Comparison failed: ${e.message}</span></div>`;
    }
    showToast("Comparison failed: " + e.message, "error");
  }
}
window.runComparison = runComparison;

// ─────────────────────────────────────────────────────────────────────────────
// 4. Prescriptive Inventory Optimizer Page (OR-Tools)
// ─────────────────────────────────────────────────────────────────────────────

async function loadInventoryPage() {
  const content = document.getElementById("main-content");
  content.innerHTML = `
    <div class="page-header">
      <h1>📦 Prescriptive Inventory Optimization</h1>
      <p>Solve Newsvendor stochastic trade-offs using Google OR-Tools (GLOP) linear programming.</p>
    </div>

    <!-- Parameter Inputs -->
    <div class="control-bar" style="flex-wrap:wrap;align-items:flex-end;">
      <div class="control-group">
        <label>Store</label>
        <select id="inv-store">
          <option value="CA_1">CA_1</option>
          <option value="TX_1">TX_1</option>
          <option value="WI_1">WI_1</option>
        </select>
      </div>

      <div class="control-group">
        <label>Product Item</label>
        <select id="inv-item">
          <option value="HOBBIES_1_001">HOBBIES_1_001</option>
        </select>
      </div>

      <div class="control-group">
        <label>Current Stock (Units)</label>
        <input type="number" id="inv-stock" value="45" min="0" style="width:110px;">
      </div>

      <div class="control-group">
        <label>Forecast Demand (7 Days)</label>
        <input type="number" id="inv-demand" value="120" min="1" style="width:110px;">
      </div>

      <div class="control-group">
        <label>Demand Std Dev (σ)</label>
        <input type="number" id="inv-std" value="22" min="0" style="width:110px;">
      </div>

      <div class="control-group">
        <label>Holding Cost ($/unit/day)</label>
        <input type="number" id="inv-hold" value="0.20" step="0.05" min="0.01" style="width:110px;">
      </div>

      <div class="control-group">
        <label>Stockout Penalty ($/unit)</label>
        <input type="number" id="inv-short" value="2.50" step="0.25" min="0.10" style="width:110px;">
      </div>

      <button class="btn btn-primary" onclick="runInventoryOpt()">
        <span>⚡</span> Optimize Replenishment
      </button>

      <button class="btn btn-outline" onclick="autoFillFromForecast()" title="Pull 7-day predicted demand and volatility directly from ML model">
        <span>🔄</span> Pull ML Forecast
      </button>
    </div>

    <!-- Optimization Results Grid -->
    <div class="chart-grid" id="inv-results" style="margin-top:20px;">
      <div class="spinner-wrap"><div class="spinner"></div><span>Solving inventory optimization linear program…</span></div>
    </div>
  `;

  // Hydrate items
  try {
    const items = await API.getItems("CA_1");
    if (items?.length) {
      document.getElementById("inv-item").innerHTML = items.map(i =>
        `<option value="${i.item_id}">${i.item_id}</option>`
      ).join("");
    }
  } catch (e) {}

  runInventoryOpt();
}

async function runInventoryOpt() {
  const itemId    = document.getElementById("inv-item")?.value    || "HOBBIES_1_001";
  const storeId   = document.getElementById("inv-store")?.value   || "CA_1";
  const stock     = parseFloat(document.getElementById("inv-stock")?.value   || "45");
  const demand    = parseFloat(document.getElementById("inv-demand")?.value  || "120");
  const std       = parseFloat(document.getElementById("inv-std")?.value     || "22");
  const hold      = parseFloat(document.getElementById("inv-hold")?.value    || "0.20");
  const stockout  = parseFloat(document.getElementById("inv-short")?.value   || "2.50");

  try {
    const result = await API.quickInventory(itemId, storeId, stock, demand, std, hold, stockout);

    const criticalRatio = (stockout / (stockout + hold)) * 100;

    document.getElementById("inv-results").innerHTML = `
      <div class="card" style="border-left:4px solid var(--blue);">
        <div class="card-header">
          <h3>📦 Replenishment Recommendation: ${itemId}</h3>
          <span class="badge badge-green">${result.solver_status}</span>
        </div>

        <div id="inv-gauge" style="margin:16px 0;"></div>

        <div class="data-table" style="background:transparent;">
          <div style="display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--border);">
            <span style="color:var(--text-sec);">Current Physical Stock</span>
            <strong>${fmtInt(stock)} units</strong>
          </div>
          <div style="display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--border);">
            <span style="color:var(--text-sec);">Expected Cycle Demand</span>
            <strong>${fmtInt(demand)} units</strong>
          </div>
          <div style="display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--border);">
            <span style="color:var(--text-sec);">Safety Stock Buffer (SS)</span>
            <strong>${fmtInt(result.safety_stock)} units</strong>
          </div>
          <div style="display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--border);">
            <span style="color:var(--text-sec);">Reorder Point Trigger (ROP)</span>
            <strong>${fmtInt(result.reorder_point)} units</strong>
          </div>
          <div style="display:flex;justify-content:space-between;padding:14px 0;font-size:1.1rem;">
            <span style="color:#60a5fa;font-weight:700;">Recommended Purchase Order (Q*)</span>
            <strong style="color:#60a5fa;font-size:1.3rem;">⬆ ${fmtInt(result.recommended_order)} units</strong>
          </div>
        </div>
      </div>

      <div class="card" style="display:flex;flex-direction:column;gap:18px;">
        <div class="card-header"><h3>📐 Newsvendor Cost Optimization Breakdown</h3></div>

        <div>
          <div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:0.85rem;">
            <span style="color:var(--text-sec);">Holding Cost ($${hold.toFixed(2)}/unit)</span>
            <strong>$${fmt(result.estimated_holding_cost ?? 0, 2)}</strong>
          </div>
          <div style="height:8px;background:rgba(255,255,255,0.08);border-radius:99px;overflow:hidden;">
            <div style="height:100%;width:35%;background:var(--blue);border-radius:99px;"></div>
          </div>
        </div>

        <div>
          <div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:0.85rem;">
            <span style="color:var(--text-sec);">Stockout Penalty ($${stockout.toFixed(2)}/unit)</span>
            <strong>$${fmt(result.estimated_stockout_cost ?? 0, 2)}</strong>
          </div>
          <div style="height:8px;background:rgba(255,255,255,0.08);border-radius:99px;overflow:hidden;">
            <div style="height:100%;width:65%;background:var(--orange);border-radius:99px;"></div>
          </div>
        </div>

        <div class="interval-box">
          <div class="interval-label">Optimal Cycle Service Level (Critical Ratio)</div>
          <div class="interval-main">${fmt(criticalRatio, 1)}%</div>
          <div class="interval-range">
            CR = c_s / (c_s + c_h). The OR-Tools simplex solver guarantees a stockout avoidance probability of ${fmt(criticalRatio, 1)}%,
            balancing capital carrying expense against customer stockout penalties.
          </div>
        </div>
      </div>
    `;

    Charts.renderInventoryGauge("inv-gauge", stock, result.reorder_point, demand * 1.5);
    showToast(`Optimal order: ${fmtInt(result.recommended_order)} units recommended.`);
  } catch (e) {
    showToast("Optimization failed: " + e.message, "error");
  }
}

/**
 * Pull ML forecast directly into the inventory form.
 */
async function autoFillFromForecast() {
  const itemId  = document.getElementById("inv-item")?.value;
  const storeId = document.getElementById("inv-store")?.value || "CA_1";
  if (!itemId) { showToast("Select an item first.", "error"); return; }

  try {
    const fc = await API.getForecast(itemId, storeId, 7, "lightgbm");
    if (!fc || !fc.forecasts || !fc.forecasts.length) {
      showToast("No forecast data returned for item.", "error");
      return;
    }
    const total = fc.forecasts.reduce((a, b) => a + b.yhat, 0);
    const std   = fc.forecasts.reduce((a, b) => a + (b.yhat_upper - b.yhat), 0) / fc.forecasts.length / 1.96;
    
    document.getElementById("inv-demand").value = Math.max(1, Math.round(total));
    document.getElementById("inv-std").value    = Math.max(1, Math.round(std));
    
    showToast("Auto-filled 7-day demand parameters from LightGBM model.");
    runInventoryOpt();
  } catch (e) {
    showToast("Forecast auto-fill failed: " + e.message, "error");
  }
}
window.runInventoryOpt = runInventoryOpt;
window.autoFillFromForecast = autoFillFromForecast;

// ─────────────────────────────────────────────────────────────────────────────
// 5. Hierarchical Reconciliation View
// ─────────────────────────────────────────────────────────────────────────────

async function loadHierarchyPage() {
  const content = document.getElementById("main-content");
  content.innerHTML = `
    <div class="page-header">
      <h1>🔗 Hierarchical Forecasting Reconciliation</h1>
      <p>Ensure structural coherence across levels: SKU &rarr; Store &rarr; State via Darts bottom-up summing matrix reconciliation.</p>
    </div>

    <div class="stat-grid">
      <div class="stat-card blue">
        <div class="label">Level 3: State Aggregates</div>
        <div class="value" style="font-size:1.3rem;">CA · TX · WI</div>
        <div class="sub">Sum of regional store sales</div>
      </div>
      <div class="stat-card green">
        <div class="label">Level 2: Store Aggregates</div>
        <div class="value" style="font-size:1.3rem;">3 Stores</div>
        <div class="sub">Sum of department &amp; SKU sales</div>
      </div>
      <div class="stat-card purple">
        <div class="label">Level 1: Base Products</div>
        <div class="value" style="font-size:1.3rem;">3,049 Series</div>
        <div class="sub">Bottom-level SKU time-series</div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <h3>Reconciled Store Demand Schedule</h3>
      </div>
      <div class="control-bar" style="margin-bottom:16px;">
        <div class="control-group">
          <label>Store Location</label>
          <select id="hier-store">
            <option value="CA_1">CA_1 (California)</option>
            <option value="TX_1">TX_1 (Texas)</option>
            <option value="WI_1">WI_1 (Wisconsin)</option>
          </select>
        </div>
        <div class="control-group">
          <label>Horizon (Days)</label>
          <input type="number" id="hier-horizon" value="14" min="7" max="28" style="width:90px;">
        </div>
        <button class="btn btn-primary" onclick="loadHierarchyForecast()">
          <span>📊</span> Reconcile Store Aggregate
        </button>
      </div>
      <div id="hier-chart" style="height:360px;">
        <div class="spinner-wrap"><div class="spinner"></div><span>Computing coherent bottom-up aggregation…</span></div>
      </div>
    </div>
  `;

  loadHierarchyForecast();
}

async function loadHierarchyForecast() {
  const storeId = document.getElementById("hier-store")?.value || "CA_1";
  const horizon = parseInt(document.getElementById("hier-horizon")?.value || "14");

  const chartEl = document.getElementById("hier-chart");
  if (chartEl) {
    chartEl.innerHTML = `<div class="spinner-wrap"><div class="spinner"></div><span>Reconciling SKU forecasts for Store ${storeId}…</span></div>`;
  }

  try {
    const items = await API.getItems(storeId);
    if (!items || !items.length) {
      showToast("No items found for store " + storeId, "error");
      return;
    }
    const sampleItems = items.slice(0, 5);
    const forecasts = await Promise.all(
      sampleItems.map(i => API.getForecast(i.item_id, storeId, horizon, "lightgbm"))
    );

    if (!forecasts.length || !forecasts[0]?.forecasts?.length) {
      showToast("No forecast data available for reconciliation.", "error");
      return;
    }

    const dates = forecasts[0].forecasts.map(d => d.date);
    const aggYhat  = dates.map((_, idx) => forecasts.reduce((s, fc) => s + (fc.forecasts[idx]?.yhat || 0), 0));
    const aggLower = dates.map((_, idx) => forecasts.reduce((s, fc) => s + (fc.forecasts[idx]?.yhat_lower || 0), 0));
    const aggUpper = dates.map((_, idx) => forecasts.reduce((s, fc) => s + (fc.forecasts[idx]?.yhat_upper || 0), 0));

    const aggFc = dates.map((d, i) => ({
      date: d,
      yhat: aggYhat[i],
      yhat_lower: aggLower[i],
      yhat_upper: aggUpper[i]
    }));

    Charts.renderForecastChart("hier-chart", [], aggFc);
    showToast(`Store ${storeId} reconciled aggregate computed.`);
  } catch (e) {
    showToast("Reconciliation failed: " + e.message, "error");
  }
}
window.loadHierarchyForecast = loadHierarchyForecast;

// ── Application Initialization & Background Connectivity Monitor ──────────────

async function initHealthMonitor() {
  const statusBadge = document.getElementById("system-status-badge");
  if (!statusBadge) return;

  const res = await API.checkHealth();
  if (res.status === "ok") {
    statusBadge.className = "badge badge-green";
    statusBadge.innerHTML = `<span class="pulse-dot"></span> System Live`;
  } else {
    statusBadge.className = "badge badge-orange";
    statusBadge.innerHTML = `<span class="pulse-dot" style="background:#f97316;"></span> Reconnecting…`;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  // Bind navigation tab clicks
  document.querySelectorAll(".nav-item").forEach(el => {
    el.addEventListener("click", () => activatePage(el.dataset.page));
  });

  // Start background health polling
  initHealthMonitor();
  setInterval(initHealthMonitor, 10000);

  // Initialize landing page
  activatePage("overview");
});
