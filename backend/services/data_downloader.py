"""
backend/services/data_downloader.py
=====================================
Automatic Dataset Downloader for Cloud Deployment (Render, Railway, etc.)

WHY THIS EXISTS:
The M5 Walmart dataset CSV files are 114–194 MB each — too large to commit
to a git repository (GitHub blocks files > 100 MB). On Render, the app's
filesystem is ephemeral, so this module downloads the required CSVs from
cloud storage (Google Drive or any direct URL) at server startup, only if
the files are not already present on disk.

HOW TO USE:
1. Upload your CSV files to Google Drive (make them "Anyone with link can view").
2. Set the following environment variables in Render's dashboard:
      DATA_SALES_VALIDATION_URL   = <Google Drive / Dropbox share URL>
      DATA_SALES_EVALUATION_URL   = <Google Drive / Dropbox share URL>
      DATA_SELL_PRICES_URL        = <Google Drive / Dropbox share URL>
      DATA_CALENDAR_URL           = <optional — file is tiny (0.1 MB)>
3. On startup, this module checks if each file exists. If missing, it
   downloads from the configured URL automatically.

SUPPORTED URL FORMATS:
- Google Drive share link:  https://drive.google.com/file/d/FILE_ID/view?usp=sharing
  → Converted internally to direct download via gdown (handles >100 MB confirm page)
- Dropbox share link:       https://www.dropbox.com/s/.../file.csv?dl=0
  → dl=0 is rewritten to dl=1 for direct download
- Any direct HTTP/HTTPS URL (S3, GCS public, etc.)
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Mapping: env var name → local file path ───────────────────────────────────
_FILE_MAP = {
    "DATA_CALENDAR_URL":          "calendar.csv",
    "DATA_SALES_VALIDATION_URL":  "sales_train_validation.csv",
    "DATA_SALES_EVALUATION_URL":  "sales_train_evaluation.csv",
    "DATA_SELL_PRICES_URL":       "sell_prices.csv",
}

# Files that are mandatory for the app to start
_REQUIRED_FILES = {
    "calendar.csv",
    "sales_train_validation.csv",
    "sell_prices.csv",
}


def _is_google_drive_url(url: str) -> bool:
    return "drive.google.com" in url


def _is_dropbox_url(url: str) -> bool:
    return "dropbox.com" in url


def _make_dropbox_direct(url: str) -> str:
    """Convert Dropbox share link to direct download URL."""
    return url.replace("?dl=0", "?dl=1").replace("&dl=0", "&dl=1")


def _download_gdrive(url: str, dest: Path) -> None:
    """Download a Google Drive file using gdown (handles >100 MB confirm bypass)."""
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required for Google Drive downloads. "
            "Add 'gdown>=5.1.0' to requirements.txt and rebuild."
        )

    # Accept both share links and direct /uc?id= links
    gdown.download(url, str(dest), quiet=False, fuzzy=True)


def _download_direct(url: str, dest: Path) -> None:
    """Download any direct HTTP/HTTPS URL with streaming to handle large files."""
    import requests

    logger.info(f"  Downloading {dest.name} from {url[:80]}…")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):  # 8 MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        logger.info(f"  {dest.name}: {pct:.1f}%  ({downloaded // 1_048_576} MB / {total // 1_048_576} MB)")
    logger.info(f"  ✅ {dest.name} saved ({dest.stat().st_size // 1_048_576} MB)")


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
    Downloads any missing files from their configured URLs.

    Args:
        data_dir: Path to the project's data/ directory.

    Raises:
        FileNotFoundError: If a required file is missing and no download URL is set.
        Exception: If a download fails.
    """
    logger.info("🔍 Checking dataset files …")
    missing_required = []

    for env_var, filename in _FILE_MAP.items():
        dest = data_dir / filename
        url = os.getenv(env_var, "").strip()

        if dest.exists():
            size_mb = dest.stat().st_size / 1_048_576
            logger.info(f"  ✅ {filename} already present ({size_mb:.1f} MB) — skipping download")
            continue

        # File is missing
        if url:
            logger.info(f"  📥 {filename} not found — downloading from {env_var} …")
            try:
                _download_file(url, dest)
            except Exception as e:
                logger.error(f"  ❌ Failed to download {filename}: {e}")
                raise
        else:
            if filename in _REQUIRED_FILES:
                missing_required.append(filename)
                logger.error(
                    f"  ❌ MISSING: {filename}\n"
                    f"     Set env var {env_var} with a download URL, "
                    f"or place the file at: {dest}"
                )
            else:
                logger.warning(
                    f"  ⚠️  Optional file {filename} not found and {env_var} not set — skipping."
                )

    if missing_required:
        raise FileNotFoundError(
            f"Required data files are missing and no download URL was configured:\n"
            f"  {missing_required}\n\n"
            f"Set the following environment variables in Render → Environment:\n"
            + "\n".join(
                f"  {v}=<your URL>"
                for v, f in _FILE_MAP.items() if f in missing_required
            )
        )

    logger.info("✅ All required dataset files are present.")

