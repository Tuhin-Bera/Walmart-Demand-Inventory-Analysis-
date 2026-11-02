"""
backend/main.py — FastAPI Application Server & Enterprise Gateway.

========================================================================================
SYSTEM ARCHITECTURE:
This module initializes the FastAPI ASGI application, configures cross-origin resource
sharing (CORS), mounts modular domain routers, and manages the application lifecycle
(e.g., in-memory pre-warming of the M5 feature dataset).

Key Components:
  1. Lifespan Context Manager: Runs at application startup to ingest raw CSVs and
     engineer features into memory. This eliminates the 8-10 second cold-start latency
     for end users.
  2. Sub-Application Routers:
     - `/api/forecast`: Time-series generation across Naive, Prophet, LightGBM, TFT.
     - `/api/inventory`: OR-Tools linear programming order quantity solver.
     - `/api/metrics`: High-level KPIs, 90-day aggregate trends, and model comparison.
  3. Static Asset Mounting: Mounts `/static`, `/css`, and `/js` directories to serve
     the native single-page application dashboard directly from root `/`.
  4. Health Checks: Exposes `/health` for container orchestration and uptime probes.
========================================================================================
"""

import logging
import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

# Ensure repository root is on Python module search path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import API_TITLE, API_VERSION, API_HOST, API_PORT, ENVIRONMENT
from backend.routers import forecast, inventory, metrics

# Standard structured logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan: Pre-warm Data and Models on Startup ─────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous lifecycle hook executed during server startup and shutdown.

    At startup:
      Pre-loads and feature-engineers the M5 dataset into memory. This eliminates
      cold-start latency on the first API request and pre-computes dashboard KPIs.
    """
    logger.info("🚀 Initializing Walmart Demand Forecasting & Inventory Platform …")
    try:
        from backend.config import DATA_DIR
        from backend.services.data_downloader import ensure_data_files
        ensure_data_files(DATA_DIR)

        logger.info("Pre-warming processed dataset in memory …")
        from backend.services.data_service import get_processed_data
        get_processed_data()
        logger.info("✅ Dataset pre-warmed successfully. Server ready for traffic.")
    except Exception as e:
        logger.error(f"⚠️ Startup initialization encountered an error: {e}", exc_info=True)
        logger.warning("Application will fall back to lazy-loading upon first API request.")
    
    yield
    
    logger.info("🛑 Shutting down Walmart Demand Forecasting API server.")


# ── FastAPI Application Instance ─────────────────────────────────────────────

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="""
## Walmart Demand Forecasting & Inventory Optimization Platform

High-performance AI retail platform built for the Walmart M5 dataset.

### Core Capabilities:
- 📈 **Multi-Engine Forecasting**: Seasonal Naive, Meta Prophet, LightGBM, and Temporal Fusion Transformer.
- 🔗 **Hierarchical Reconciliation**: Bottom-up summing matrix reconciliation from SKU to Store to State.
- 📊 **Calibrated Uncertainty**: 95% empirical and analytical prediction intervals.
- 📦 **Prescriptive Inventory Optimization**: Google OR-Tools GLOP linear programming solver.
    """,
    lifespan=lifespan,
)

# ── Cross-Origin Resource Sharing (CORS) ─────────────────────────────────────
# Permits local frontend dashboards and remote client applications to query the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Domain Router Registration ───────────────────────────────────────────────
app.include_router(forecast.router)
app.include_router(inventory.router)
app.include_router(metrics.router)

# ── Static Frontend Asset Hosting ────────────────────────────────────────────
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
    
    # Mount css and js subdirectories directly so relative paths in index.html resolve cleanly
    css_dir = os.path.join(frontend_dir, "css")
    js_dir = os.path.join(frontend_dir, "js")
    if os.path.isdir(css_dir):
        app.mount("/css", StaticFiles(directory=css_dir), name="css")
    if os.path.isdir(js_dir):
        app.mount("/js", StaticFiles(directory=js_dir), name="js")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        """Serve the single-page application dashboard entrypoint."""
        return FileResponse(os.path.join(frontend_dir, "index.html"))


# ── Enterprise Health Probes ──────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health_check():
    """Liveness probe returning operational status and semantic version."""
    return {"status": "ok", "version": API_VERSION}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Suppress 404 logs for browser favicon requests."""
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=API_HOST, port=API_PORT, reload=(ENVIRONMENT == "development"))
