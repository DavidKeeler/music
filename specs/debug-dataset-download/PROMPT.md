# PROMPT.md — Debug Dataset Download Tool

## Objective

Fix 4 bugs in `src/data/` discovered when running `python -m src.data.download_datasets --all --data_dir ~/data/music`.

## Key Requirements

1. **PHENICX**: RepoVizz hosting is dead (404). Reclassify from "auto" to "manual" in `datasets.py`. Remove `download_phenicx()` from `download_auto.py`. Remove phenicx dispatch branch from `download_datasets.py`. Update `manual_instructions` to explain RepoVizz is down, suggest contacting MTG at UPF.

2. **MOSA/URMP false failures**: In `download_datasets.py`, when `--all` is used without credentials, print `[SKIP]` with instructions and `continue` — no `[FAIL]`, no `mark_failed()`, no state change. Remove `validate_args()` and its warning block (redundant).

3. **Hardcoded path**: In `datasets.py`, change Human3.6M `manual_instructions` step 4 to `"4. Extract to {data_dir}/human36m/"`. In `readme.py`, format `manual_instructions` with `data_dir=data_dir` in both `generate_instructions()` and `generate_readme()`.

4. **Manual testing only** — no unit tests.

## Acceptance Criteria

- Given `--all --data_dir ~/data/music` without credentials, then MOSA/URMP show `[SKIP]` (not `[FAIL]`), PHENICX prints manual instructions, Edinburgh/AIST++ download normally
- Given `--dataset phenicx`, then instructions explain RepoVizz is down and how to contact UPF
- Given `--dataset human36m --data_dir ~/data/music`, then output says "Extract to ~/data/music/human36m/"
- Given `--all --data_dir ~/data/music`, then `~/data/music/README.md` contains correct paths (no `~/data/conducting/`)
- Given `--dry-run --all`, then PHENICX shows as MANUAL tier

## Reference

See `specs/debug-dataset-download/` for design, requirements, and research.
