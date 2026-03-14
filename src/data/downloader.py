"""HTTP file downloader with resume, retry, and checksum support."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BACKOFF_BASE = 2  # seconds
CHUNK_SIZE = 8192


def download_file(
    url: str,
    dest: Path,
    checksum: str | None = None,
    hash_algo: str = "md5",
) -> bool:
    """Download a file with resume support and optional checksum verification.

    Returns True if file is present and verified after call.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Check existing file
    if dest.exists() and checksum:
        if _verify_checksum(dest, checksum, hash_algo):
            logger.info("Skipping %s — checksum matches", dest.name)
            return True
        logger.warning("Checksum mismatch for %s, re-downloading", dest.name)
        dest.unlink()
    elif dest.exists() and not checksum:
        logger.info("Skipping %s — already exists", dest.name)
        return True

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return _download_with_resume(url, dest, checksum, hash_algo)
        except (requests.RequestException, IOError) as e:
            if attempt == MAX_RETRIES:
                logger.error("Failed after %d attempts: %s", MAX_RETRIES, e)
                raise
            wait = BACKOFF_BASE**attempt
            logger.warning("Attempt %d failed (%s), retrying in %ds", attempt, e, wait)
            time.sleep(wait)
    return False


def _download_with_resume(
    url: str, dest: Path, checksum: str | None, hash_algo: str
) -> bool:
    partial = dest.with_suffix(dest.suffix + ".part")
    headers = {}
    mode = "wb"
    existing_size = 0

    if partial.exists():
        existing_size = partial.stat().st_size
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"

    resp = requests.get(url, headers=headers, stream=True, timeout=60)

    if resp.status_code == 416:  # Range not satisfiable — file complete
        if partial.exists():
            partial.rename(dest)
        return True

    resp.raise_for_status()

    with open(partial, mode) as f:
        for chunk in resp.iter_content(CHUNK_SIZE):
            f.write(chunk)

    partial.rename(dest)

    if checksum and not _verify_checksum(dest, checksum, hash_algo):
        logger.error("Checksum mismatch after download: %s", dest.name)
        dest.unlink()
        raise IOError(f"Checksum mismatch for {dest.name}")

    return True


def download_files(
    urls: list[str],
    dest_dir: Path,
    checksums: dict[str, str] | None = None,
) -> list[Path]:
    """Download multiple files to a directory. Returns list of downloaded paths."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for url in urls:
        filename = url.rsplit("/", 1)[-1]
        dest = dest_dir / filename
        cs = checksums.get(filename) if checksums else None
        if download_file(url, dest, checksum=cs):
            results.append(dest)
    return results


def _verify_checksum(path: Path, expected: str, algo: str = "md5") -> bool:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest() == expected
