# Implementation Plan: Dataset Download Tool

## Checklist

- [ ] Step 1: Project scaffolding and CLI
- [ ] Step 2: HTTP downloader with resume
- [ ] Step 3: Dataset registry and state tracking
- [ ] Step 4: Auto-download datasets (PHENICX, Edinburgh, AIST++)
- [ ] Step 5: Semi-manual datasets (MOSA, URMP)
- [ ] Step 6: Manual instructions and README generation

## Step 1: Project Scaffolding and CLI

**Objective:** Create the module structure and CLI entry point with argument parsing.

**Implementation:**
- Create `src/data/__init__.py`, `src/data/download_datasets.py`
- Implement argparse CLI: `--data_dir`, `--dataset`, `--all`, `--zenodo-token`, `--url`, `--dry-run`
- Validate arguments (e.g., `--url` only valid with `--dataset urmp`)
- Create target `data_dir` if it doesn't exist

**Tests:** Verify CLI parses all flag combinations correctly. Verify `--dataset` accepts valid names and rejects unknown ones.

**Integration:** Running `python -m src.data.download_datasets --dry-run --all` prints the list of datasets and their status.

**Demo:** CLI prints help text and dry-run output showing all 6 datasets.

## Step 2: HTTP Downloader with Resume

**Objective:** Implement `download_file()` and `download_files()` with resume and checksum support.

**Implementation:**
- Create `src/data/downloader.py`
- `download_file(url, dest, checksum=None)`: HEAD request for size, Range header for resume, stream to disk, optional checksum verify
- Skip if file exists and size matches (or checksum matches)
- Retry up to 3 times with exponential backoff
- `download_files(urls, dest_dir, checksums=None)`: batch wrapper

**Tests:** Mock HTTP server returning partial content with Range support. Test skip-existing, resume, checksum pass/fail, retry on failure.

**Integration:** Importable utility used by all dataset download functions.

**Demo:** Download a small test file, interrupt, re-run to demonstrate resume.

## Step 3: Dataset Registry and State Tracking

**Objective:** Define all 6 dataset configs and implement `.download_state.json` persistence.

**Implementation:**
- Create `src/data/datasets.py` with `DatasetConfig` dataclass and `DATASETS` dict
- Register all 6 datasets with their metadata (name, tier, URLs, descriptions)
- Implement `load_state(data_dir)` and `save_state(data_dir, state)` for `.download_state.json`
- State tracks per-dataset: status (pending, in_progress, complete, failed, manual), timestamp

**Tests:** Verify all 6 datasets registered. Test state load/save round-trip. Test state updates.

**Integration:** CLI reads state on startup, skips completed datasets, updates state after each download.

**Demo:** `--dry-run` reads state and shows which datasets need downloading vs already complete.

## Step 4: Auto-Download Datasets (PHENICX, Edinburgh, AIST++)

**Objective:** Implement download functions for the 3 freely available datasets.

**Implementation:**
- `download_phenicx()`: Scrape RepoVizz datapack links from UPF page, download to `phenicx/{performance,spontaneous,articulation}/`
- `download_edinburgh()`: Download zip from Edinburgh DataShare (`https://datashare.ed.ac.uk/download/DS_10283_{id}.zip`), extract C3D files to `edinburgh/`
- `download_aist_plusplus()`: Download motion data from Google AIST++ page to `aist-plusplus/`
- Each function updates download state on completion

**Tests:** Mock HTTP responses for each dataset's URL patterns. Verify correct directory structure created. Verify state updated to "complete".

**Integration:** `python -m src.data.download_datasets --dataset phenicx edinburgh aist-plusplus` downloads all three end-to-end.

**Demo:** Run against real URLs, show downloaded files in correct directory structure.

## Step 5: Semi-Manual Datasets (MOSA, URMP)

**Objective:** Implement assisted downloads for access-gated datasets.

**Implementation:**
- `download_mosa()`: Check for `zenodo_get` installation. If `--zenodo-token` provided, invoke `zenodo_get -r 11393449 -o {data_dir}/mosa/`. Check disk space first (warn if < 2.5 TB free). Without token, print access request instructions.
- `download_urmp()`: If `--url` provided, download 12.5 GB zip to `urmp/`, extract. Without URL, print Google Form link and instructions.
- Both update state appropriately (pending_access, pending_url, or complete)

**Tests:** Mock zenodo_get invocation. Test disk space check. Test behavior with and without credentials/URL.

**Integration:** `python -m src.data.download_datasets --dataset mosa --zenodo-token TOKEN` triggers MOSA download.

**Demo:** Run without credentials to show instruction output. Run with credentials to show download initiation.

## Step 6: Manual Instructions and README Generation

**Objective:** Generate `~/data/conducting/README.md` with instructions for all datasets.

**Implementation:**
- `generate_readme(data_dir, state)`: Write README.md with:
  - Overview of all datasets and their status (from state)
  - Manual instructions for Human3.6M (registration URL, license steps, which files to download)
  - Semi-manual instructions for MOSA (Zenodo access request) and URMP (Google Form)
  - Directory structure reference
- `download_human36m()`: Just generates instructions, marks state as "manual"
- README is regenerated on every run to reflect current state

**Tests:** Verify README contains instructions for all manual/semi-manual datasets. Verify state-dependent status rendering.

**Integration:** Every run of the tool updates README.md. `--dataset human36m` generates instructions without attempting download.

**Demo:** Run `--all`, show generated README.md with status of each dataset and clear manual steps.
