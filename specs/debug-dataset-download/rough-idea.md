# Debug Dataset Download Tool

The dataset download tool (`src/data/download_datasets.py`) was implemented from `specs/dataset-download/` but has several bugs visible on first run with `--all --data_dir ~/data/music`:

## Observed Issues

1. **PHENICX 403 Forbidden** — `requests.get()` to the UPF page gets blocked, likely missing a browser-like User-Agent header. The scraping approach for RepoVizz datapack links may also be fragile if the page structure changed.

2. **MOSA/URMP false failures** — Running `--all` marks MOSA and URMP as `[FAIL]` in output and state when no credentials are provided. These are expected "pending" states, not errors. The tool should skip them gracefully with `--all` and only fail when the user explicitly requests them without credentials.

3. **Hardcoded `~/data/conducting/` path in Human3.6M instructions** — The manual instructions say "Extract to ~/data/conducting/human36m/" regardless of the `--data_dir` flag. Should use the actual data_dir.

4. **README instructions also hardcoded** — Same hardcoded path issue likely appears in the generated README.md for manual/semi-manual datasets.

## Source Files

- `src/data/download_datasets.py` — CLI entry point
- `src/data/downloader.py` — HTTP downloader
- `src/data/datasets.py` — dataset registry with hardcoded instructions
- `src/data/download_auto.py` — auto-download functions (PHENICX scraping logic)
- `src/data/download_semi.py` — semi-manual downloads (MOSA/URMP)
- `src/data/readme.py` — README generation
- `src/data/state.py` — download state tracking
