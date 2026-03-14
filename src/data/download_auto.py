"""Auto-download functions for freely available datasets."""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

from .datasets import DatasetConfig
from .downloader import download_file, download_files

logger = logging.getLogger(__name__)


def download_phenicx(config: DatasetConfig, dest: Path) -> None:
    """Download PHENICX Conduct dataset from UPF/RepoVizz.

    The RepoVizz platform hosting the original datapacks is no longer available.
    This function attempts to scrape download links from the UPF page. If that
    fails, it raises with instructions for manual download.
    """
    import re

    import requests

    page_url = "https://www.upf.edu/web/mtg/phenicx-conduct-dataset"
    logger.info("Fetching PHENICX datapack links from %s", page_url)

    try:
        resp = requests.get(page_url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Cannot reach PHENICX page ({e}).\n{config.manual_instructions}"
        ) from e

    # Extract RepoVizz datapack URLs from the page
    urls = list(set(re.findall(r'https?://repovizz\.upf\.edu/repo/Vizz/\d+', resp.text)))
    if not urls:
        raise RuntimeError(
            "No datapack links found on PHENICX page.\n" + config.manual_instructions
        )

    # Test first URL to see if RepoVizz is alive
    try:
        test = requests.head(urls[0], timeout=15, allow_redirects=True)
        if test.status_code == 404:
            raise RuntimeError(
                "RepoVizz platform is no longer available.\n"
                + config.manual_instructions
            )
    except requests.RequestException:
        raise RuntimeError(
            "RepoVizz platform is unreachable.\n" + config.manual_instructions
        )

    logger.info("Found %d datapacks, downloading to %s", len(urls), dest)
    download_files(urls, dest)


def download_edinburgh(config: DatasetConfig, dest: Path) -> None:
    """Download Edinburgh conducting MoCap from DataShare.

    Downloads a single zip (~498 MB) containing 54 C3D files and extracts it.
    """
    url = config.urls[0]
    zip_path = dest / "DS_10283_2913.zip"

    logger.info("Downloading Edinburgh MoCap zip from %s", url)
    download_file(url, zip_path)

    logger.info("Extracting %s", zip_path.name)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    zip_path.unlink()
    logger.info("Edinburgh MoCap extracted to %s", dest)


def download_aist_plusplus(config: DatasetConfig, dest: Path) -> None:
    """Download AIST++ motion data from Google Cloud Storage.

    Downloads motions.zip (~306 MB), keypoints3d.zip (~834 MB), and
    cameras.zip (~19 KB), then extracts each.
    """
    for url in config.urls:
        filename = url.rsplit("/", 1)[-1]
        zip_path = dest / filename

        logger.info("Downloading AIST++ %s", filename)
        download_file(url, zip_path)

        logger.info("Extracting %s", filename)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)

        zip_path.unlink()

    logger.info("AIST++ data extracted to %s", dest)
