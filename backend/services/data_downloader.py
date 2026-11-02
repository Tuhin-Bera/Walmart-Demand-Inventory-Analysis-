"""
backend/services/data_downloader.py
=====================================
Automatic Dataset Downloader for Cloud Deployment (Render, Railway, etc.)

Downloads required CSV dataset files (>100MB) from cloud storage (Google Drive)
if they are not already present on disk in data/.
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Mapping: env var name → local file path ───────────────────────────────────
_FILE_MAP = {
    "DATA_CALENDAR_URL":          "calendar.csv",
    "DATA_SALES_VALIDATION_URL":  "sales_train_validation.csv",
    "DATA_SALES_EVALUATION_URL":  "sales_train_evaluation.csv",
    "DATA_SELL_PRICES_URL":       "sell_prices.csv",
}

# ── Built-in Default Fallback URLs (Google Drive) ──────────────────────────────
# Used if environment variables are not explicitly set in the hosting dashboard
_DEFAULT_URLS = {
    "DATA_SALES_VALIDATION_URL": "https://drive.google.com/file/d/1BMf1VgeAkkwmlEsdliqQ3cIr1oxvC6fU/view?usp=sharing",
    "DATA_SELL_PRICES_URL":      "https://drive.google.com/file/d/10LCz2fDeDwkDMwLe_Eqc9nd0b3_T5B_h/view?usp=sharing",
}

# Files that are mandatory for the app to start
_REQUIRED_FILES = {
    "calendar.csv",
    "sales_train_validation.csv",
    "sell_prices.csv",
}


def _extract_gdrive_id(url: str) -> Optional[str]:
    """Extract the Google Drive file ID from any share or direct link."""
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None


def _is_google_drive_url(url: str) -> bool:
    return "drive.google.com" in url or bool(_extract_gdrive_id(url))


def _is_dropbox_url(url: str) -> bool:
    return "dropbox.com" in url


def _make_dropbox_direct(url: str) -> str:
    """Convert Dropbox share link to direct download URL."""
    return url.replace("?dl=0", "?dl=1").replace("&dl=0", "&dl=1")


def _download_gdrive(url: str, dest: Path) -> None:
    """Download a Google Drive file using gdown (handles >100 MB confirm bypass)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    file_id = _extract_gdrive_id(url)
    logger.info(f"  📥 Downloading Google Drive file for {dest.name} (File ID: {file_id}) …")

    # Attempt 1: gdown via ID
    try:
        import gdown
        if file_id:
            gdown.download(id=file_id, output=str(dest), quiet=False)
        else:
            gdown.download(url, output=str(dest), quiet=False, fuzzy=True)

        if dest.exists() and dest.stat().st_size > 1024 * 1024:
            logger.info(f"  ✅ {dest.name} downloaded successfully via gdown ({dest.stat().st_size // 1_048_576} MB)")
            return
    except Exception as e:
        logger.warning(f"  ⚠️ gdown encountered an error: {e}. Trying direct stream download …")

    # Attempt 2: Fallback to direct requests streaming
    if file_id:
        direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        _download_direct(direct_url, dest)
    else:
        _download_direct(url, dest)


def _download_direct(url: str, dest: Path) -> None:
    """Download any direct HTTP/HTTPS URL with streaming to handle large files."""
    import requests

    logger.info(f"  📥 Streaming {dest.name} from {url[:80]} …")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):  # 8 MB chunks
                if chunk:
                    f.write(chunk)
    logger.info(f"  ✅ {dest.name} saved successfully ({dest.stat().st_size // 1_048_576} MB)")


def _download_file(url: str, dest: Path) -> None:
    """Route download to the correct handler based on URL type."""
    if _is_google_drive_url(url):
        _download_gdrive(url, dest)
    elif _is_dropbox_url(url):
        _download_direct(_make_dropbox_direct(url), dest)
    else:
        _download_direct(url, dest)


def ensure_data_files(data_dir: Path) -> None:
    """
    Check that all required CSV files exist in data_dir.
    Downloads any missing files from their configured URLs or built-in default URLs.

    Args:
        data_dir: Path to the project's data/ directory.
    """
    logger.info("🔍 Checking dataset files in data directory …")
    missing_required = []

    for env_var, filename in _FILE_MAP.items():
        dest = data_dir / filename
        # Priority: Environment variable -> Built-in default URL
        url = os.getenv(env_var, "").strip() or _DEFAULT_URLS.get(env_var, "").strip()

        if dest.exists() and dest.stat().st_size > 1024:
            size_mb = dest.stat().st_size / 1_048_576
            logger.info(f"  ✅ {filename} already present ({size_mb:.1f} MB) — skipping download")
            continue

        # File is missing
        if url:
            logger.info(f"  📥 {filename} not found locally — initiating download …")
            try:
                _download_file(url, dest)
            except Exception as e:
                logger.error(f"  ❌ Failed to download {filename} from {url}: {e}")
                if filename in _REQUIRED_FILES:
                    missing_required.append(filename)
        else:
            if filename in _REQUIRED_FILES:
                missing_required.append(filename)
                logger.error(
                    f"  ❌ MISSING: {filename}\n"
                    f"     Set env var {env_var} with a download URL, "
                    f"or place the file at: {dest}"
                )

    if missing_required:
        raise FileNotFoundError(
            f"Required data files are missing and could not be downloaded:\n"
            f"  {missing_required}\n"
            f"Please verify network access or configure DATA_SALES_VALIDATION_URL and DATA_SELL_PRICES_URL."
        )

    logger.info("✅ All required dataset files are verified.")
