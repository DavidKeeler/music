"""Semi-manual download functions for access-gated datasets."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from .datasets import DatasetConfig
from .downloader import download_file

logger = logging.getLogger(__name__)

MOSA_MIN_FREE_BYTES = 2_500_000_000_000  # 2.5 TB


def download_mosa(config: DatasetConfig, dest: Path, zenodo_token: str | None) -> None:
    """Download MOSA from Zenodo using zenodo_get.

    Requires zenodo_get installed and a valid Zenodo access token.
    """
    if not zenodo_token:
        print(f"\n{config.manual_instructions}")
        raise RuntimeError("MOSA requires --zenodo-token. See instructions above.")

    # Check zenodo_get is available
    if shutil.which("zenodo_get") is None:
        raise RuntimeError(
            "zenodo_get not found. Install with: pip install zenodo-get"
        )

    # Check disk space
    free = shutil.disk_usage(dest.parent if dest.parent.exists() else Path.home()).free
    if free < MOSA_MIN_FREE_BYTES:
        free_tb = free / 1e12
        logger.warning(
            "Low disk space: %.1f TB free, MOSA needs ~2.3 TB. Proceeding anyway.",
            free_tb,
        )
        print(f"WARNING: Only {free_tb:.1f} TB free. MOSA is ~2.3 TB.")

    dest.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading MOSA via zenodo_get (record %s)", config.zenodo_record)
    env = {**os.environ, "ZENODO_TOKEN": zenodo_token}
    result = subprocess.run(
        ["zenodo_get", "-r", config.zenodo_record, "-o", str(dest), "-k"],
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"zenodo_get failed with exit code {result.returncode}")

    logger.info("MOSA downloaded to %s", dest)


def download_urmp(config: DatasetConfig, dest: Path, url: str | None) -> None:
    """Download URMP from user-provided URL.

    User must first fill out the Google Form to receive a download link.
    """
    if not url:
        print(f"\n{config.manual_instructions}")
        raise RuntimeError("URMP requires --url. See instructions above.")

    dest.mkdir(parents=True, exist_ok=True)
    filename = url.rsplit("/", 1)[-1] or "urmp_download"
    dl_path = dest / filename

    logger.info("Downloading URMP from %s", url)
    download_file(url, dl_path)

    # Extract if it's a zip
    if zipfile.is_zipfile(dl_path):
        logger.info("Extracting %s", dl_path.name)
        with zipfile.ZipFile(dl_path, "r") as zf:
            zf.extractall(dest)
        dl_path.unlink()

    logger.info("URMP downloaded to %s", dest)
