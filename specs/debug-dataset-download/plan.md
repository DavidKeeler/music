# Implementation Plan: Debug Dataset Download Tool

## Checklist

- [ ] Step 1: Reclassify PHENICX and remove scraping code
- [ ] Step 2: Graceful skip for semi-manual datasets without credentials
- [ ] Step 3: Fix hardcoded path in Human3.6M instructions
- [ ] Step 4: Manual verification

## Step 1: Reclassify PHENICX and Remove Scraping Code

**Objective:** PHENICX is no longer auto-downloadable (RepoVizz is dead). Reclassify to manual and remove dead code.

**Implementation:**
- `src/data/datasets.py`: Change PHENICX `access_tier` to `"manual"`, clear `urls`, update `manual_instructions` to explain RepoVizz is down and suggest contacting MTG at UPF
- `src/data/download_auto.py`: Delete `download_phenicx()` function entirely
- `src/data/download_datasets.py`: Remove the `if name == "phenicx"` branch from the download dispatch — PHENICX will now fall through to the manual handler alongside `human36m`

**Integration:** Running `--dataset phenicx` prints contact instructions instead of attempting a download. `--dry-run` shows PHENICX as MANUAL tier.

**Demo:** `python -m src.data.download_datasets --dry-run --dataset phenicx` shows `[MANUAL]` status.

## Step 2: Graceful Skip for Semi-Manual Datasets Without Credentials

**Objective:** `--all` without MOSA token or URMP URL should warn and skip, not fail.

**Implementation:**
- `src/data/download_datasets.py`:
  - In the download loop, before dispatching to `download_mosa`/`download_urmp`, check if credentials are missing
  - If missing: print `[SKIP]` with the dataset's `manual_instructions`, then `continue` (no state change, no exception)
  - Remove the `validate_args()` call and the warnings block at the top — the per-dataset skip logic replaces it

**Integration:** `--all` completes without any `[FAIL]` entries for credential-gated datasets. State file is not polluted with false failures.

**Demo:** `python -m src.data.download_datasets --all --data_dir ~/data/music` shows `[SKIP]` for MOSA and URMP.

## Step 3: Fix Hardcoded Path in Human3.6M Instructions

**Objective:** Manual instructions should reflect the actual `--data_dir`, not hardcoded `~/data/conducting/`.

**Implementation:**
- `src/data/datasets.py`: Change Human3.6M `manual_instructions` step 4 to use `{data_dir}` placeholder: `"4. Extract to {data_dir}/human36m/"`
- `src/data/readme.py`:
  - `generate_instructions(config, data_dir)`: format `config.manual_instructions` with `data_dir=data_dir` before printing
  - `generate_readme(data_dir, state)`: format `cfg.manual_instructions` with `data_dir=data_dir` before writing to README
- `src/data/download_datasets.py`: Update `generate_instructions()` call to pass `data_dir` (already does — just verify the signature matches)

**Integration:** Instructions in both stdout and README.md show the correct user-specified path.

**Demo:** `python -m src.data.download_datasets --dataset human36m --data_dir ~/data/music` prints "Extract to ~/data/music/human36m/".

## Step 4: Manual Verification

**Objective:** Verify all fixes work together.

**Verification steps:**
1. `python -m src.data.download_datasets --dry-run --all --data_dir ~/data/music` — PHENICX shows MANUAL, no errors
2. `python -m src.data.download_datasets --all --data_dir ~/data/music` — MOSA/URMP show `[SKIP]`, PHENICX prints contact instructions, Human3.6M shows correct path
3. Check `~/data/music/README.md` — all paths reference `~/data/music/`, not `~/data/conducting/`
4. Edinburgh and AIST++ still download successfully (regression check)
