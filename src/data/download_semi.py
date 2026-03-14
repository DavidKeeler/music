"""Semi-manual download functions for access-gated datasets."""

from __future__ import annotations

from pathlib import Path

from .datasets import DatasetConfig


def download_mosa(config: DatasetConfig, dest: Path, zenodo_token: str | None) -> None:
    """Download MOSA from Zenodo using zenodo_get."""
    raise NotImplementedError("MOSA download not yet implemented")


def download_urmp(config: DatasetConfig, dest: Path, url: str | None) -> None:
    """Download URMP from user-provided URL."""
    raise NotImplementedError("URMP download not yet implemented")
