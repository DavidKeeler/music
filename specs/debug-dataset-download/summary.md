# Summary: Debug Dataset Download Tool

## Artifacts

| File | Description |
|---|---|
| `specs/debug-dataset-download/rough-idea.md` | Initial bug report from first run output |
| `specs/debug-dataset-download/requirements.md` | Q&A clarifying fix approach for each bug |
| `specs/debug-dataset-download/research/phenicx-availability.md` | Research confirming RepoVizz is dead, no alternative source |
| `specs/debug-dataset-download/design.md` | Detailed design for all 4 fixes |
| `specs/debug-dataset-download/plan.md` | 4-step implementation plan |

## Overview

Fixes 4 bugs in `src/data/` discovered during first run of the dataset download tool:

1. **PHENICX 403 / RepoVizz dead** → Reclassify to manual tier, remove scraping code, add contact instructions
2. **MOSA/URMP false failures** → Warn and skip when `--all` without credentials (no `[FAIL]`, no state change)
3. **Hardcoded `~/data/conducting/` path** → Use `{data_dir}` placeholder in Human3.6M instructions
4. **README path** → Format instructions with actual `data_dir` in readme.py

## Files Modified

- `src/data/datasets.py` — PHENICX reclassified, Human3.6M path templatized
- `src/data/download_datasets.py` — Skip logic for semi-manual, remove phenicx dispatch
- `src/data/download_auto.py` — Remove `download_phenicx()`
- `src/data/readme.py` — Format instructions with `data_dir`

## Next Steps

Implement via Ralph or manually. Verify with `--all --data_dir ~/data/music`.
