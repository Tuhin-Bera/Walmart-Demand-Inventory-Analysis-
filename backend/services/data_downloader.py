"""
backend/services/data_downloader.py
=====================================
Automatic Dataset Downloader for Cloud Deployment (Render, Railway, etc.)

Downloads required CSV dataset files (>100MB) from cloud storage (Google Drive)
if they are not already present on disk in data/.

Handles Google Drive's >100MB 'Virus scan warning' page automatically by
extracting the dynamic session token (uuid) and streaming the raw CSV.
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional
import requests

logger = logging.getLogger(__name__)

# ── Mapping: env var name → local file path ───────────────────────────────────
_FILE_MAP = {
    "DATA_CALENDAR_URL":          "calendar.csv",
    "DATA_SALES_VALIDATION_URL":  "sales_train_validation.csv",
    "DATA_SALES_EVALUATION_URL":  "sales_train_evaluation.csv",
    "DATA_SELL_PRICES_URL":       "sell_prices.csv",
}

# ── Built-in Default Fallback URLs (Google Drive) ──────────────────────────────
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


def _download_gdrive_with_token(file_id: str, dest: Path) -> bool:
    """
    Download a large Google Drive file by bypassing the >100MB virus scan warning.
    Extracts the dynamic form action and uuid token from drive.usercontent.google.com.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(".tmp")
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })

    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    logger.info(f"  📥 Initiating Google Drive handshake for {dest.name} (File ID: {file_id}) …")
    
    resp = session.get(download_url, stream=True, timeout=30)
    
    # Check if we got the actual file immediately
    content_type = resp.headers.get("Content-Type", "")
    if "application/octet-stream" in content_type or "text/csv" in content_type:
        logger.info(f"  📥 Direct stream available for {dest.name} …")
        _save_response_stream(resp, temp_dest, dest)
        return True

    # If HTML returned, Google presented the >100MB virus scan confirmation page
    html_text = resp.text
    
    # Extract form action and input parameters (confirm, uuid, id, etc.)
    action_match = re.search(r'<form[^>]*action="([^"]*)"', html_text)
    action_url = action_match.group(1) if action_match else "https://drive.usercontent.google.com/download"
    if action_url.startswith("/"):
        action_url = "https://drive.google.com" + action_url

    # Extract all hidden inputs
    params = {}
    for input_match in re.finditer(r'<input[^>]*name="([^"]*)"[^>]*value="([^"]*)"', html_text):
        params[input_match.group(1)] = input_match.group(2)

    # Ensure required parameters exist
    if "id" not in params:
        params["id"] = file_id
    if "export" not in params:
        params["export"] = "download"
    if "confirm" not in params:
        params["confirm"] = "t"

    logger.info(f"  📥 Confirming large-file token bypass for {dest.name} …")
    file_resp = session.get(action_url, params=params, stream=True, timeout=300)
    
    if file_resp.status_code == 200:
        _save_response_stream(file_resp, temp_dest, dest)
        return True
    
    logger.error(f"  ❌ Confirmation request failed with HTTP {file_resp.status_code}")
    return False


def _save_response_stream(response: requests.Response, temp_dest: Path, final_dest: Path) -> None:
    """Save an HTTP response stream to disk in chunks with atomic rename."""
    total_bytes = int(response.headers.get("Content-Length", 0))
    downloaded = 0
    chunk_size = 4 * 1024 * 1024  # 4 MB chunks

    with open(temp_dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_bytes > 0:
                    pct = (downloaded / total_bytes) * 100
                    if downloaded % (16 * 1024 * 1024) < chunk_size:
                        logger.info(f"    {final_dest.name}: {pct:.1f}% ({downloaded // 1_048_576} MB / {total_bytes // 1_048_576} MB)")

    # Atomic rename upon completion
    if temp_dest.exists():
        if final_dest.exists():
            final_dest.unlink()
        temp_dest.rename(final_dest)
        size_mb = final_dest.stat().st_size / 1_048_576
        logger.info(f"  ✅ {final_dest.name} successfully saved to disk ({size_mb:.1f} MB)")


def _download_direct(url: str, dest: Path) -> None:
    """Download any direct HTTP/HTTPS URL with streaming."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(".tmp")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        _save_response_stream(r, temp_dest, dest)


def _download_file(url: str, dest: Path) -> None:
    """Route download to the correct handler based on URL type."""
    if _is_google_drive_url(url):
        file_id = _extract_gdrive_id(url)
        if file_id:
            success = _download_gdrive_with_token(file_id, dest)
            if success and dest.exists() and dest.stat().st_size > 1024 * 1024:
                return
        # Secondary fallback using gdown if available
        try:
            import gdown
            logger.info(f"  Trying gdown fallback for {dest.name} …")
            gdown.download(url, output=str(dest), quiet=False, fuzzy=True)
        except Exception as e:
            logger.error(f"  gdown fallback also failed: {e}")
    elif _is_dropbox_url(url):
        _download_direct(_make_dropbox_direct(url), dest)
    else:
        _download_direct(url, dest)


def ensure_data_files(data_dir: Path) -> None:
    """
    Check that all required CSV files exist in data_dir.
    Downloads any missing files from configured or built-in Google Drive URLs.

    Args:
        data_dir: Path to the project's data/ directory.
    """
    logger.info("🔍 Checking dataset files in data directory …")
    missing_required = []

    for env_var, filename in _FILE_MAP.items():
        dest = data_dir / filename
        # Priority: Environment variable -> Built-in default URL
        url = os.getenv(env_var, "").strip() or _DEFAULT_URLS.get(env_var, "").strip()

        # Check if file exists and has valid content (calendar.csv is ~103KB, others are >100MB)
        min_size = 1000 if filename == "calendar.csv" else 1024 * 1024
        if dest.exists() and dest.stat().st_size > min_size:
            size_mb = dest.stat().st_size / 1_048_576
            logger.info(f"  ✅ {filename} already present ({size_mb:.1f} MB) — skipping download")
            continue

        # File is missing or incomplete
        if url:
            logger.info(f"  📥 {filename} missing on disk — initiating download …")
            try:
                _download_file(url, dest)
            except Exception as e:
                logger.error(f"  ❌ Failed to download {filename}: {e}")
                if filename in _REQUIRED_FILES:
                    missing_required.append(filename)
        else:
            if filename in _REQUIRED_FILES:
                missing_required.append(filename)
                logger.error(f"  ❌ MISSING: {filename} and no URL provided.")

    if missing_required:
        raise FileNotFoundError(
            f"Required data files are missing and could not be downloaded:\n"
            f"  {missing_required}\n"
            f"Please verify network access to Google Drive."
        )

    logger.info("✅ All required dataset files are verified.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from backend.config import DATA_DIR
    ensure_data_files(DATA_DIR)
