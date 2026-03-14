# PROMPT.md — Dataset Download Tool

## Objective

Build a Python CLI tool at `src/data/download_datasets.py` that downloads conducting-related datasets to `~/data/conducting/`. Idempotent, resumable, with per-dataset flags.

## Key Requirements

- CLI: `python -m src.data.download_datasets --data_dir ~/data/conducting --dataset <names>` or `--all`
- Flags: `--zenodo-token`, `--url`, `--dry-run`
- HTTP downloads with resume (Range headers), retry (3x backoff), checksum verification
- `.download_state.json` tracks per-dataset progress
- Generates `~/data/conducting/README.md` with manual instructions

## Datasets

| Name | Tier | Source | Notes |
|---|---|---|---|
| phenicx | auto | UPF RepoVizz links (CC BY-NC-SA 4.0) | 3 sub-datasets: performance, spontaneous, articulation |
| edinburgh | auto | Edinburgh DataShare DOI:10.7488/ds/2223 | 54 C3D files, zip download |
| aist-plusplus | auto | Google AIST++ download page | 3D dance motion + music |
| mosa | semi-manual | Zenodo record 11393449 (restricted, 2.3TB) | Needs --zenodo-token; check disk space |
| urmp | semi-manual | User-provided URL (Google Form gated, 12.5GB) | Needs --url |
| human36m | manual | Instructions only (vision.imar.ro) | Academic license required |

## Acceptance Criteria

- Given `--all`, when auto datasets are available, then PHENICX, Edinburgh, AIST++ download to subdirectories
- Given `--dataset mosa --zenodo-token TOKEN`, when Zenodo access granted, then MOSA downloads via zenodo_get
- Given `--dataset urmp --url URL`, then URMP downloads to `urmp/`
- Given `--dataset human36m`, then instructions printed and written to README
- Given interrupted download re-run, then resumes without re-downloading completed files
- Given `--dry-run`, then prints what would download with no side effects

## Reference

See `specs/dataset-download/` for design, requirements, and research.
