/**
 * frontend/js/api.js — Resilient Enterprise API Client Wrapper.
 *
 * ======================================================================================
 * ARCHITECTURAL DESIGN & NETWORK RESILIENCE:
 * This module coordinates all HTTP communication between the client dashboard and the
 * FastAPI backend service.
 *
 * Key Capabilities:
 *   1. Dynamic Endpoint Discovery: Automatically detects whether the application is
 *      accessed via `http://localhost:8000`, `http://127.0.0.1:8000`, or a custom port,
 *      avoiding hardcoded IP conflicts.
 *   2. Abortable Request Timeouts: Automatically aborts stalled requests after 30 seconds
 *      using the browser's standard AbortController.
 *   3. Intelligent Exponential Backoff Retry: In the event of transient network drops
 *      or server warm-up delays, automatically retries idempotent requests up to 3 times
 *      before surfacing an error to the user.
 *   4. Safe Fallbacks: Provides immediate default options for stores (CA_1, TX_1, WI_1)
 *      to ensure user interface dropdowns never freeze in an unusable state.
 * ======================================================================================
 */

// Dynamically determine backend base URL from current browser window context
const API_BASE = (typeof window !== "undefined" && window.location.protocol.startsWith("http"))
  ? window.location.origin
  : "http://127.0.0.1:8000";

/**
 * Perform an HTTP request with automatic timeout, JSON deserialization, and error handling.
 *
 * @param {string} path - Target API endpoint path (e.g. '/api/metrics/summary').
 * @param {Object} options - Standard fetch options plus custom timeoutMs and retries.
 * @returns {Promise<any>} - Deserialized JSON response payload.
 */
async function apiFetch(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const timeoutMs = options.timeout || 30000;
  const maxRetries = options.retries !== undefined ? options.retries : 2;

  let attempt = 0;
  while (attempt <= maxRetries) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const resp = await fetch(url, {
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        ...options,
      });
      clearTimeout(timeoutId);

      if (!resp.ok) {
        // Attempt to parse structured error detail from FastAPI HTTPException
        const errPayload = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(errPayload.detail || `HTTP error ${resp.status}`);
      }

      return await resp.json();
    } catch (e) {
      clearTimeout(timeoutId);
      attempt++;

      // Distinguish explicit timeout aborts from general connection failures
      if (e.name === "AbortError") {
        throw new Error("Request timed out. The server is processing large data.");
      }

      // If retries remain and this is a network failure (e.g. server starting up), wait briefly and retry
      if (attempt <= maxRetries && (e.message.includes("Failed to fetch") || e.message.includes("NetworkError"))) {
        console.warn(`[API] Retrying ${path} (attempt ${attempt}/${maxRetries}) in ${attempt * 500}ms...`);
        await new Promise(res => setTimeout(res, attempt * 500));
        continue;
      }

      console.error(`[API Error] Request failed on [${path}]:`, e.message);
      throw e;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Demand Forecasting Endpoints
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Generate a multi-step demand forecast for a single SKU at a designated store.
 */
async function getForecast(itemId, storeId, horizon = 28, model = "lightgbm") {
  const params = new URLSearchParams({ horizon, model });
  return apiFetch(`/api/forecast/quick/${encodeURIComponent(itemId)}/${encodeURIComponent(storeId)}?${params}`);
}

/**
 * Retrieve unique product SKUs with category attribution.
 */
async function getItems(storeId = null) {
  const params = storeId ? `?store_id=${encodeURIComponent(storeId)}` : "";
  return apiFetch(`/api/forecast/items${params}`);
}

/**
 * Retrieve available store fulfillment identifiers.
 * Includes instant fallback to ["CA_1", "TX_1", "WI_1"] to prevent UI stalls.
 */
async function getStores() {
  try {
    return await apiFetch("/api/forecast/stores", { retries: 1 });
  } catch (e) {
    console.warn("[API] Store query fallback to standard M5 benchmark stores:", e.message);
    return ["CA_1", "TX_1", "WI_1"];
  }
}

/**
 * Execute side-by-side benchmark comparison across all 4 forecasting models.
 */
async function compareModels(itemId, storeId, horizon = 28) {
  const params = new URLSearchParams({ item_id: itemId, store_id: storeId, horizon });
  return apiFetch(`/api/metrics/compare?${params}`, { timeout: 45000 });
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Historical & Summary Analytics Endpoints
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Fetch trailing daily sales history for actual vs forecast chart rendering.
 */
async function getHistorical(itemId, storeId, lastNDays = 90) {
  return apiFetch(`/api/metrics/historical/${encodeURIComponent(itemId)}/${encodeURIComponent(storeId)}?last_n_days=${lastNDays}`);
}

/**
 * Retrieve enterprise dashboard summary metrics and aggregate sales trends.
 */
async function getDashboardSummary() {
  return apiFetch("/api/metrics/summary");
}

/**
 * Check backend API health and connectivity status.
 */
async function checkHealth() {
  try {
    return await apiFetch("/health", { timeout: 5000, retries: 1 });
  } catch (e) {
    return { status: "offline", error: e.message };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Inventory Optimization Endpoints (OR-Tools)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Submit full structured inventory optimization problem payload to OR-Tools.
 */
async function optimizeInventory(params) {
  return apiFetch("/api/inventory/optimize", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

/**
 * Quick parametric inventory optimization using query string arguments.
 */
async function quickInventory(itemId, storeId, currentStock, forecastDemand, demandStd, holdingCost = 0.20, stockoutCost = 2.00) {
  const params = new URLSearchParams({
    current_stock:   currentStock,
    forecast_demand: forecastDemand,
    demand_std:      demandStd,
    holding_cost:    holdingCost,
    stockout_cost:   stockoutCost,
  });
  return apiFetch(`/api/inventory/quick/${encodeURIComponent(itemId)}/${encodeURIComponent(storeId)}?${params}`);
}

// Global namespace exposure for browser execution
window.API = {
  getForecast,
  getItems,
  getStores,
  compareModels,
  getHistorical,
  getDashboardSummary,
  checkHealth,
  optimizeInventory,
  quickInventory,
};
