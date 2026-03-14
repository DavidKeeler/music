# Summary: Dataset Download Tool

## Artifacts

| File | Description |
|---|---|
| `specs/dataset-download/rough-idea.md` | Original dataset list and priorities |
| `specs/dataset-download/requirements.md` | 9 Q&A pairs refining scope and decisions |
| `specs/dataset-download/research/auto-download-datasets.md` | Research on MOSA, DeepDance→AIST++, URMP |
| `specs/dataset-download/research/manual-access-datasets.md` | Research on PHENICX, Edinburgh, Human3.6M |
| `specs/dataset-download/research/python-tooling.md` | zenodo_get, requests, directory structure |
| `specs/dataset-download/design.md` | Full design: architecture, components, data models, acceptance criteria |
| `specs/dataset-download/plan.md` | 6-step implementation plan with checklist |
| `specs/dataset-download/summary.md` | This file |

## Overview

Python CLI tool (`src/data/download_datasets.py`) that downloads 6 conducting-related datasets to `~/data/conducting/`:

- **Auto:** PHENICX Conduct, Edinburgh MoCap, AIST++ (replaces DeepDance)
- **Semi-manual:** MOSA (Zenodo restricted, 2.3 TB), URMP (Google Form, 12.5 GB)
- **Manual:** Human3.6M (academic license)

Idempotent, resumable, with per-dataset CLI flags and state tracking.

## Key Decisions

- DeepDance dropped (not found publicly), replaced with AIST++
- MPII dropped (still images, not useful for motion)
- Synthetic data and YouTube scraping deferred to separate specs
- No post-download format conversion — raw data only
- PHENICX and Edinburgh reclassified as auto-downloadable after research

## Next Steps

Implement via Ralph using `specs/dataset-download/plan.md`.
