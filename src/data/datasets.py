"""Dataset configurations and registry."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    name: str
    display_name: str
    target_dir: str
    access_tier: str  # "auto", "semi-manual", "manual"
    description: str
    urls: list[str] = field(default_factory=list)
    zenodo_record: str | None = None
    manual_instructions: str = ""


DATASETS: dict[str, DatasetConfig] = {
    "phenicx": DatasetConfig(
        name="phenicx",
        display_name="PHENICX Conduct Dataset",
        target_dir="phenicx",
        access_tier="auto",
        description="Conducting motion capture from UPF (CC BY-NC-SA 4.0). 3 sub-datasets: performance, spontaneous, articulation.",
        urls=[
            "https://repovizz.upf.edu/phenicx/datasets/performance/",
            "https://repovizz.upf.edu/phenicx/datasets/spontaneous/",
            "https://repovizz.upf.edu/phenicx/datasets/articulation/",
        ],
        manual_instructions="Download from https://www.upf.edu/web/mtg/phenicx-conduct-dataset",
    ),
    "edinburgh": DatasetConfig(
        name="edinburgh",
        display_name="Edinburgh Orchestral Conducting MoCap",
        target_dir="edinburgh",
        access_tier="auto",
        description="54 C3D motion capture recordings of 6 professional conductors. DOI: 10.7488/ds/2223.",
        urls=["https://datashare.ed.ac.uk/download/DS_10283_2223.zip"],
        manual_instructions="Download from https://datashare.ed.ac.uk/handle/10283/2223",
    ),
    "aist-plusplus": DatasetConfig(
        name="aist-plusplus",
        display_name="AIST++ Dance Motion Dataset",
        target_dir="aist-plusplus",
        access_tier="auto",
        description="3D dance motion + music from Google. 5.2 hrs, 1408 sequences, 10 genres (CC BY 4.0).",
        urls=["https://google.github.io/aistplusplus_dataset/download.html"],
        manual_instructions="Download from https://google.github.io/aistplusplus_dataset/download.html",
    ),
    "mosa": DatasetConfig(
        name="mosa",
        display_name="MOSA Dataset",
        target_dir="mosa",
        access_tier="semi-manual",
        description="742 piano/violin performances, 23 musicians, 30+ hours. Zenodo restricted, ~2.3 TB.",
        zenodo_record="11393449",
        manual_instructions=(
            "1. Create a Zenodo account at https://zenodo.org\n"
            "2. Request access to record https://zenodo.org/records/11393449\n"
            "3. Once approved, create an API token at https://zenodo.org/account/settings/applications/\n"
            "4. Re-run with: --dataset mosa --zenodo-token YOUR_TOKEN"
        ),
    ),
    "urmp": DatasetConfig(
        name="urmp",
        display_name="URMP Dataset",
        target_dir="urmp",
        access_tier="semi-manual",
        description="44 multi-instrument pieces, ~12.5 GB. Requires Google Form submission.",
        manual_instructions=(
            "1. Fill out the Google Form: https://goo.gl/forms/xSvMzlwl3IWijvcp2\n"
            "2. You will receive a download URL via email\n"
            "3. Re-run with: --dataset urmp --url DOWNLOAD_URL"
        ),
    ),
    "human36m": DatasetConfig(
        name="human36m",
        display_name="Human3.6M",
        target_dir="human36m",
        access_tier="manual",
        description="3.6M frames, 11 subjects, 4 camera views, 50 Hz. Academic license required.",
        manual_instructions=(
            "1. Register at http://vision.imar.ro/human3.6m/\n"
            "2. Accept the license agreement (academic email required)\n"
            "3. Download 'D3 Positions' for subjects 1, 5, 6, 7, 8, 9, 11\n"
            "4. Extract to ~/data/conducting/human36m/"
        ),
    ),
}

AUTO_DATASETS = [n for n, c in DATASETS.items() if c.access_tier == "auto"]
VALID_NAMES = list(DATASETS.keys())
