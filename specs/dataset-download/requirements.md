# Requirements — Dataset Download

## Questions & Answers

### Q1: Which datasets should the Ralph job actually attempt to download automatically vs. which ones need manual/academic access that we should just document instructions for?

Several of these require license agreements or request forms (PHENICX, Human3.6M, Edinburgh MoCap). Should the job handle only the freely downloadable ones (e.g., MOSA on Zenodo, DeepDance on Zenodo, URMP) and generate a README with manual steps for the gated ones?

**A1:** Yes. Auto-download freely available datasets, generate manual instructions for gated ones.

### Q2: Where should downloaded datasets be stored? The music training data currently lives at `~/data/music/musicnet/train_data/`. Should conducting datasets go under a parallel path like `~/data/conducting/` with subdirectories per dataset?

**A2:** Yes. `~/data/conducting/` with subdirectories per dataset (e.g. `~/data/conducting/phenicx/`, `~/data/conducting/deepdance/`, `~/data/conducting/mosa/`, etc.).

### Q3: Should the Tier 6 synthetic beat trajectory generator and Tier 7 YouTube scraping pipeline be part of this Ralph job, or should those be separate jobs? They're fundamentally different tasks (generation/scraping vs. downloading existing datasets).

**A3:** Synthetic data generation → separate spec. YouTube scraping → out of scope entirely. This job covers Tiers 1–5 only.

### Q4: Should the job include any post-download processing — like converting everything to the unified format (17–33 joint skeleton, hand velocity/acceleration, beat positions, etc.) — or just download raw data and leave conversion for a later step?

**A4:** Just download raw data. Format conversion is a separate step.

### Q5: Should the download script be idempotent — i.e., skip datasets that are already downloaded, verify checksums, and support resuming interrupted downloads?

**A5:** Yes, where possible. Idempotent, skip existing, resume if supported.

### Q6: What language/tooling for the download script? A Python script (fits the existing project) using `requests`/`wget`/`zenodo_get`, or a shell script, or something else?

**A6:** Python. Fits the existing project.

### Q7: For the MPII Human Pose dataset — it's still images only (no motion/video). Is it still worth including given this job is focused on motion datasets for conducting? Or drop it from scope?

**A7:** Drop MPII. Final dataset list:
- Tier 1: PHENICX Conduct, Edinburgh MoCap (manual access — document instructions)
- Tier 2: MOSA (auto-download)
- Tier 3: DeepDance (auto-download)
- Tier 4: URMP (auto-download)
- Tier 5: Human3.6M (manual access — document instructions)

### Q8: Any preference on how the script is invoked? A single CLI entry point like `python -m src.data.download_datasets --data_dir ~/data/conducting` with flags to select specific datasets, or just a simple run-all script?

**A8:** CLI with flags to select specific datasets.

### Q9: Research findings updated the access picture. Revised scope:

**A9:** Confirmed. Updated dataset plan:

**Auto-download (script handles fully):**
- PHENICX Conduct (CC BY-NC-SA 4.0, direct links from UPF/RepoVizz)
- Edinburgh MoCap (open access, Edinburgh DataShare, DOI: 10.7488/ds/2223)
- AIST++ (replaces DeepDance — free from Google, 5.2 hrs, 1408 sequences)

**Semi-manual (script downloads once user obtains access):**
- MOSA (Zenodo restricted, 2.3 TB — user must request access first, then zenodo_get)
- URMP (Google Form required, 12.5 GB — user provides download URL)

**Manual (instructions only):**
- Human3.6M (academic registration + license agreement at vision.imar.ro)

**Tooling:** zenodo_get for Zenodo, requests with resume for direct HTTP.
**Storage:** ~/data/conducting/{dataset_name}/
**Language:** Python, CLI with per-dataset flags, idempotent where possible.
