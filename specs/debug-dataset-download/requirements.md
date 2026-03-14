# Requirements: Debug Dataset Download Tool

## Q&A

### Q1: For the PHENICX 403 — should we fix the scraping approach (add User-Agent, handle the current page structure), or switch to hardcoded direct download URLs for the dataset files? Scraping is fragile if the page changes again.

**A1:** Use hardcoded direct download URLs instead of scraping.

### Q2: When running `--all` without MOSA/URMP credentials, what should the behavior be? Options:
- (a) Skip them silently and print instructions (no `[FAIL]`, no state change)
- (b) Print instructions and mark state as "pending_access"/"pending_url" (not "failed")
- (c) Something else?

**A2:** Print a warning with the instructions and move on. No `[FAIL]`, no failed state.

### Q3: For the hardcoded `~/data/conducting/` path in Human3.6M instructions — should we make all `manual_instructions` strings in the dataset registry into templates that accept the actual `data_dir`, or just fix the Human3.6M one since it's the only one with a hardcoded path?

**A3:** Just fix the Human3.6M one.

### Q4: The PHENICX download function currently scrapes the UPF page for RepoVizz URLs. Since we're switching to hardcoded URLs, do you know the actual direct download URLs for the PHENICX dataset files? Or should we research what's currently available?

**A4:** Research what's currently available.

### Q5: Any other bugs or issues you've noticed beyond these four (PHENICX 403, MOSA/URMP false failures, hardcoded paths, README paths)?

**A5:** No.

### Q6: For testing the fixes, what's the approach? Options:
- (a) Just manual testing — run the tool and verify output
- (b) Unit tests for the specific fixes (mock HTTP, check skip logic, check path substitution)
- (c) Both

**A6:** Manual testing only.

