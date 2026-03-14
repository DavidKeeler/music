"""Auto-download functions for freely available datasets."""

from __future__ import annotations

from pathlib import Path

from .datasets import DatasetConfig
from .downloader import download_file


def download_phenicx(config: DatasetConfig, dest: Path) -> None:
    """Download PHENICX Conduct dataset from UPF/RepoVizz."""
    raise NotImplementedError("PHENICX download not yet implemented")


def download_edinburgh(config: DatasetConfig, dest: Path) -> None:
    """Download Edinburgh conducting MoCap from DataShare."""
    raise NotImplementedError("Edinburgh download not yet implemented")


def download_aist_plusplus(config: DatasetConfig, dest: Path) -> None:
    """Download AIST++ from Google."""
    raise NotImplementedError("AIST++ download not yet implemented")
