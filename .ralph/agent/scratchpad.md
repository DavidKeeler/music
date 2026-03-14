# Dataset Download Tool — Scratchpad

## 2026-03-13: Initial Planning

Objective: Build `src/data/download_datasets.py` CLI tool for downloading conducting datasets.

Specs are thorough in `specs/dataset-download/`. Key files:
- design.md: Full architecture, components, interfaces
- plan.md: 6-step implementation plan
- requirements.md: Q&A clarifying scope
- research/: URL details for each dataset

Plan (following spec's plan.md closely):
1. Scaffolding: `src/data/__init__.py`, CLI entry point with argparse, dataset registry dataclass, state tracking
2. HTTP downloader with resume (Range headers), retry, checksum
3. Auto-download functions (phenicx, edinburgh, aist-plusplus)
4. Semi-manual datasets (mosa via zenodo_get, urmp via user URL)
5. Manual instructions + README generation (human36m)
6. Wire everything together in CLI main(), dry-run support

Starting with task 1: scaffolding + CLI + registry + state tracking. This is the foundation everything else builds on.

## Completed: Scaffolding + Downloader (tasks c9d2, e7a6)

Implemented all foundational modules. CLI works with --dry-run, --all, --dataset, argument validation.
Downloader has full resume/retry/checksum logic. State tracking round-trips correctly.
.gitignore needed `!src/data/` to override the `data/` pattern.

Remaining tasks:
- task-1773465388-fd1a: Auto-download datasets (PHENICX, Edinburgh, AIST++)
- task-1773465388-1256: Semi-manual datasets (MOSA, URMP)
- task-1773465388-281e: README generation and manual instructions
- task-1773465388-3cab: Wire CLI main() and dry-run support

Note: README generation and CLI wiring are already mostly done in the scaffolding.
The readme.py and main() dispatch are functional. Tasks 281e and 3cab may just need
verification/polish rather than full implementation.

## 2026-03-13: Implementing Auto-Download Datasets (task fd1a)

Picking task-1773465388-fd1a: Auto-download datasets (PHENICX, Edinburgh, AIST++).

Analysis of each dataset's download approach:

1. **PHENICX**: URLs in config point to RepoVizz pages, not direct downloads. The spec says
   "scrape RepoVizz datapack links from UPF page". However, RepoVizz may be unreliable/down.
   Pragmatic approach: try downloading from the configured URLs. If they're HTML pages rather
   than direct files, fall back to printing manual instructions. The config already has
   manual_instructions as fallback.

2. **Edinburgh**: Single zip from DataShare: `https://datashare.ed.ac.uk/download/DS_10283_2223.zip`.
   Download zip, extract C3D files. Straightforward.

3. **AIST++**: Google's download page. The actual data files are hosted on Google Cloud Storage.
   Need to download motion annotations (SMPL, keypoints) from known URLs. The page lists
   specific download links for different data types.

For all three: use the existing `download_file`/`download_files` from downloader.py.
Edinburgh needs zip extraction. PHENICX and AIST++ need specific URL lists.

Key decision: Since these are real URLs that may change/break, the download functions should
be robust — catch errors, log clearly, and not crash the whole tool. The existing downloader
already has retry logic.

For AIST++, the actual download URLs from Google are:
- Annotations: `https://storage.googleapis.com/aist_plusplus_public/...`
- The download page lists specific files. I'll use known URLs from the research.

Implementation plan:
- PHENICX: download from configured URLs, create subdirectories
- Edinburgh: download zip, extract with zipfile module
- AIST++: download annotation files from Google Cloud Storage URLs

## Completed: Auto-download datasets (task fd1a)

Implemented all 3 auto-download functions. Key findings:
- PHENICX RepoVizz platform is dead (404). UPF page returns 403. Function handles both gracefully.
- Edinburgh DOI resolves to handle 2913 (not 2223 as originally in config). Fixed URL.
- AIST++ files are on public GCS bucket, direct download works.
- End-to-end tested with AIST++ cameras.zip (21KB).

Remaining tasks:
- task-1773465388-1256: Semi-manual datasets (MOSA, URMP)
- task-1773465388-281e: README generation and manual instructions
- task-1773465388-3cab: Wire CLI main() and dry-run support

## 2026-03-13: Implementing Semi-Manual Datasets (task 1256)

Picking task-1773465388-1256: Semi-manual datasets (MOSA, URMP).

Analysis:

1. **MOSA**: Zenodo record 11393449, restricted, ~2.3 TB. Use `zenodo_get` as subprocess
   (spec says to use it). Need to:
   - Check `zenodo_get` is installed (ImportError/subprocess check)
   - Check disk space (shutil.disk_usage, warn if < 2.5 TB free)
   - Invoke: `zenodo_get -r 11393449 -o dest/ -k` with ZENODO_TOKEN env var
   - Without token: print manual_instructions from config
   - zenodo_get uses ZENODO_TOKEN env var for auth on restricted records

2. **URMP**: User provides URL after Google Form. ~12.5 GB.
   - If URL provided: download with our downloader, extract if zip
   - Without URL: print manual_instructions from config

Both follow the pattern from download_auto.py — use existing downloader infrastructure.
The CLI already dispatches to download_mosa/download_urmp with the right args.

## Completed: Semi-manual datasets (task 1256)

Implemented download_mosa() and download_urmp() in download_semi.py.
- MOSA: zenodo_get subprocess with ZENODO_TOKEN env, disk space check, graceful errors
- URMP: download_file + zip extraction, graceful errors without URL
- Both print manual_instructions when credentials missing
- End-to-end tested: URMP with real zip download, MOSA error paths verified
- CLI dry-run and dispatch both work correctly

Remaining tasks:
- task-1773465388-281e: README generation and manual instructions
- task-1773465388-3cab: Wire CLI main() and dry-run support

## 2026-03-13: Final verification and completion

Verified remaining tasks (281e: README generation, 3cab: CLI wiring) were already fully
implemented during scaffolding (commit 15fff56). Comprehensive verification:

- `--dry-run --all`: Shows all 6 datasets with correct status labels
- `--dry-run --dataset human36m`: Shows manual dataset correctly
- `--dataset human36m`: Prints instructions to stdout, writes README
- README includes per-dataset status, descriptions, and manual instructions
- CLI dispatches to all download functions (auto, semi-manual, manual)
- State tracking works (skips completed datasets, marks failures)
- All modules import correctly, registry has all 6 datasets

All 6 tasks closed. Objective complete.
