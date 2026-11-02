/**
 * frontend/js/charts.js — High-Performance Charting Engine (Chart.js & Plotly).
 *
 * ======================================================================================
 * ARCHITECTURAL DESIGN:
 * This module encapsulates all data visualization logic for the demand forecasting and
 * inventory platform. It manages two rendering backends:
 *   1. Chart.js (HTML5 Canvas): Used for lightweight, reactive categorical aggregations,
 *      department donut distributions, state volume bars, and daily trend sparklines.
 *   2. Plotly.js (WebGL / SVG): Used for high-dimensional scientific time-series plots,
 *      specifically multi-horizon forecasts with shaded 95% confidence bands and
 *      multi-model overlay comparisons.
 *
 * All chart instances track container lifecycles to prevent memory leaks by destroying
 * obsolete canvas instances prior to re-rendering.
 * ======================================================================================
 */

// ── Enterprise Color Palette ──────────────────────────────────────────────────
const COLORS = {
  blue:       "#2563eb",
  blueLight:  "#3b82f6",
  blueArea:   "rgba(37, 99, 235, 0.15)",
  green:      "#10b981",
  greenLight: "#34d399",
  orange:     "#f97316",
  yellow:     "#f59e0b",
  purple:     "#8b5cf6",
  grey:       "#64748b",
  border:     "#232d4b",
  bgCard:     "#131a2e",
  textPri:    "#f0f4fc",
  textSec:    "#94a3b8",
};

const MODEL_COLORS = {
  naive:     COLORS.grey,
  prophet:   COLORS.yellow,
  lightgbm:  COLORS.blue,
  tft:       COLORS.purple,
  actual:    COLORS.green,
};

// ── Global Chart.js Theming Configuration ─────────────────────────────────────
if (typeof Chart !== "undefined") {
  Chart.defaults.color              = COLORS.textSec;
  Chart.defaults.borderColor        = COLORS.border;
  Chart.defaults.backgroundColor    = COLORS.bgCard;
  Chart.defaults.font.family        = "'Inter', system-ui, -apple-system, sans-serif";
  Chart.defaults.font.size          = 12;
  if (Chart.defaults.plugins?.legend?.labels) {
    Chart.defaults.plugins.legend.labels.color = COLORS.textPri;
  }
}

/**
 * Standard baseline canvas options ensuring consistent responsiveness and theme styles.
 */
function baseChartOptions(extraOptions = {}) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: {
        position: "top",
        labels: {
          usePointStyle: true,
          padding: 16,
          color: COLORS.textPri,
        },
      },
      tooltip: {
        backgroundColor: "#0f172a",
        borderColor: COLORS.border,
        borderWidth: 1,
        padding: 12,
        titleColor: COLORS.textPri,
        bodyColor: COLORS.textSec,
        cornerRadius: 8,
      },
    },
    scales: {
      x: {
        ticks: { color: COLORS.textSec, maxTicksLimit: 10 },
        grid:  { color: "rgba(35, 45, 75, 0.6)" },
      },
      y: {
        ticks: { color: COLORS.textSec },
        grid:  { color: "rgba(35, 45, 75, 0.6)" },
        beginAtZero: true,
      },
    },
    ...extraOptions,
  };
}

// ── Internal Canvas Lifecycle Registry ────────────────────────────────────────
const _charts = {};

function destroyChart(id) {
  if (_charts[id]) {
    _charts[id].destroy();
    delete _charts[id];
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Daily Sales Trend (Overview Sparkline / Area Chart)
// ─────────────────────────────────────────────────────────────────────────────

function renderDailyTrend(canvasId, trendData) {
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId);
  if (!ctx || !trendData?.length) return;

  const labels = trendData.map(d => d.date);
  const values = trendData.map(d => d.sales);

  _charts[canvasId] = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label:           "Total Daily Retail Sales",
        data:            values,
        borderColor:     COLORS.blueLight,
        backgroundColor: COLORS.blueArea,
        fill:            true,
        tension:         0.35,
        pointRadius:     0,
        borderWidth:     2.5,
      }],
    },
    options: baseChartOptions(),
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Actual vs Forecast with 95% Confidence Band (Plotly)
// ─────────────────────────────────────────────────────────────────────────────

function renderForecastChart(divId, historical, forecast) {
  const container = document.getElementById(divId);
  if (!container || typeof Plotly === "undefined") return;

  const histDates = historical.map(d => d.date);
  const histSales = historical.map(d => d.sales);
  const fcDates   = forecast.map(d => d.date);
  const fcYhat    = forecast.map(d => d.yhat);
  const fcLower   = forecast.map(d => d.yhat_lower);
  const fcUpper   = forecast.map(d => d.yhat_upper);

  // 95% Prediction Interval Ribbon (Constructed via polygon coordinates)
  const intervalTrace = {
    x:          [...fcDates, ...fcDates.slice().reverse()],
    y:          [...fcUpper, ...fcLower.slice().reverse()],
    fill:       "toself",
    fillcolor:  "rgba(37, 99, 235, 0.18)",
    line:       { color: "transparent" },
    name:       "95% Prediction Band",
    showlegend: true,
    type:       "scatter",
    hoverinfo:  "skip",
  };

  // Trailing Ground-Truth Sales Line
  const actualTrace = {
    x:    histDates,
    y:    histSales,
    mode: "lines",
    name: "Historical Sales (Actual)",
    line: { color: COLORS.greenLight, width: 2.2 },
    type: "scatter",
  };

  // Machine Learning Forward Forecast Line
  const forecastTrace = {
    x:    fcDates,
    y:    fcYhat,
    mode: "lines+markers",
    name: "Forecasted Demand (Yhat)",
    line: { color: "#60a5fa", width: 2.5, dash: "dot" },
    marker: { size: 5, color: "#93c5fd" },
    type: "scatter",
  };

  const layout = {
    paper_bgcolor: "transparent",
    plot_bgcolor:  "transparent",
    font:    { color: COLORS.textPri, family: "'Inter', sans-serif", size: 12 },
    xaxis:   {
      gridcolor: "rgba(35, 45, 75, 0.7)",
      title: { text: "Calendar Date", font: { color: COLORS.textSec } },
      tickcolor: COLORS.border,
    },
    yaxis:   {
      gridcolor: "rgba(35, 45, 75, 0.7)",
      title: { text: "Daily Units Sold", font: { color: COLORS.textSec } },
      tickcolor: COLORS.border,
      rangemode: "tozero",
    },
    legend:  {
      orientation: "h",
      yanchor: "bottom",
      y: 1.04,
      font: { color: COLORS.textPri },
    },
    margin:  { l: 55, r: 25, t: 25, b: 50 },
    hovermode: "x unified",
    shapes: [{
      // Vertical dashed line marking historical vs forecast cutoff boundary
      type: "line",
      x0: histDates.at(-1),
      x1: histDates.at(-1),
      y0: 0,
      y1: 1,
      yref: "paper",
      line: { color: "#ef4444", dash: "dash", width: 1.5 },
    }],
  };

  Plotly.newPlot(divId, [intervalTrace, actualTrace, forecastTrace], layout, {
    responsive: true,
    displayModeBar: false,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Multi-Model Forecast Overlay (Plotly)
// ─────────────────────────────────────────────────────────────────────────────

function renderMultiModelForecast(divId, compareData) {
  const container = document.getElementById(divId);
  if (!container || typeof Plotly === "undefined") return;

  const traces = Object.entries(compareData).map(([modelName, data]) => {
    const displayName = {
      naive:    "Seasonal Naive",
      prophet:  "Meta Prophet",
      lightgbm: "LightGBM Regressor",
      tft:      "Temporal Fusion Transformer",
    }[modelName] || modelName;

    return {
      x:    data.forecasts.map(d => d.date),
      y:    data.forecasts.map(d => d.yhat),
      mode: "lines+markers",
      name: displayName,
      line: { color: MODEL_COLORS[modelName] || "#ffffff", width: 2 },
      marker: { size: 4 },
      type: "scatter",
    };
  });

  const layout = {
    paper_bgcolor: "transparent",
    plot_bgcolor:  "transparent",
    font:    { color: COLORS.textPri, family: "'Inter', sans-serif", size: 12 },
    xaxis:   { gridcolor: "rgba(35, 45, 75, 0.7)", title: "Date" },
    yaxis:   { gridcolor: "rgba(35, 45, 75, 0.7)", title: "Forecasted Units" },
    legend:  { orientation: "h", yanchor: "bottom", y: 1.04 },
    margin:  { l: 55, r: 25, t: 25, b: 50 },
    hovermode: "x unified",
  };

  Plotly.newPlot(divId, traces, layout, { responsive: true, displayModeBar: false });
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Model Accuracy Comparison Bar Chart (Chart.js)
// ─────────────────────────────────────────────────────────────────────────────

function renderMetricsComparison(canvasId, metricsData) {
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const labels = Object.keys(metricsData);
  const wape   = labels.map(m => metricsData[m]?.metrics?.wape ?? 0);
  const rmse   = labels.map(m => metricsData[m]?.metrics?.rmse ?? 0);
  const mae    = labels.map(m => metricsData[m]?.metrics?.mae  ?? 0);

  _charts[canvasId] = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels.map(l => l.toUpperCase()),
      datasets: [
        { label: "WAPE (Weighted Error)", data: wape, backgroundColor: COLORS.blueLight },
        { label: "RMSE (Root Mean Sq)",   data: rmse, backgroundColor: COLORS.yellow    },
        { label: "MAE (Mean Absolute)",   data: mae,  backgroundColor: COLORS.purple    },
      ],
    },
    options: baseChartOptions({
      plugins: {
        title: {
          display: true,
          text: "Model Accuracy Benchmark (Lower is Better)",
          color: COLORS.textPri,
          font: { size: 14, weight: 600 },
        },
      },
    }),
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Category Distribution Donut (Chart.js)
// ─────────────────────────────────────────────────────────────────────────────

function renderCategoryDonut(canvasId, catData) {
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const labels = catData.map(d => d.cat_id);
  const values = catData.map(d => d.total_sales);

  _charts[canvasId] = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: [COLORS.blue, COLORS.green, COLORS.orange],
        borderWidth: 2,
        borderColor: COLORS.bgCard,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "right", labels: { color: COLORS.textPri } },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.label}: ${Math.round(ctx.parsed).toLocaleString()} units`,
          },
        },
      },
    },
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Regional Sales Volume Bars (Chart.js)
// ─────────────────────────────────────────────────────────────────────────────

function renderStateBars(canvasId, stateData) {
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  _charts[canvasId] = new Chart(ctx, {
    type: "bar",
    data: {
      labels:   stateData.map(d => d.state_id),
      datasets: [{
        label:           "Total Regional Sales",
        data:            stateData.map(d => d.total_sales),
        backgroundColor: [COLORS.blue, COLORS.green, COLORS.orange],
        borderRadius:    6,
      }],
    },
    options: baseChartOptions({ plugins: { legend: { display: false } } }),
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Interactive Inventory Fulfillment Gauge (SVG)
// ─────────────────────────────────────────────────────────────────────────────

function renderInventoryGauge(containerId, currentStock, reorderPoint, maxCapacity) {
  const el = document.getElementById(containerId);
  if (!el) return;

  const capacity = maxCapacity || (reorderPoint * 2.2);
  const pct = Math.min(100, Math.max(0, (currentStock / (capacity || 1)) * 100));
  
  const color = pct < 25 ? COLORS.orange : pct < 60 ? COLORS.yellow : COLORS.green;
  const statusText = pct < 25 ? "⚠️ Stockout Risk" : pct < 60 ? "✓ Normal Operating Stock" : "🛡️ High Safety Buffer";

  el.innerHTML = `
    <div style="display:flex;flex-direction:column;align-items:center;padding:12px;">
      <svg width="140" height="80" viewBox="0 0 120 70">
        <path d="M10 60 A50 50 0 0 1 110 60" stroke="#1f293d" stroke-width="12" fill="none" stroke-linecap="round"/>
        <path d="M10 60 A50 50 0 0 1 110 60" stroke="${color}" stroke-width="12" fill="none"
              stroke-linecap="round"
              stroke-dasharray="${pct * 1.57} 157"
              transform="rotate(0,60,60)"/>
        <text x="60" y="58" text-anchor="middle" fill="#ffffff" font-size="16" font-weight="700">${Math.round(pct)}%</text>
      </svg>
      <div style="font-size:0.85rem;color:${color};font-weight:600;margin-top:4px;">${statusText}</div>
      <div style="font-size:0.75rem;color:${COLORS.textSec};margin-top:2px;">Target Capacity: ${Math.round(capacity).toLocaleString()} units</div>
    </div>
  `;
}

// Global namespace exposure
window.Charts = {
  renderDailyTrend,
  renderForecastChart,
  renderMetricsComparison,
  renderMultiModelForecast,
  renderCategoryDonut,
  renderStateBars,
  renderInventoryGauge,
};
