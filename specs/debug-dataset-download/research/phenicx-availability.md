# Research: PHENICX Dataset Availability

## Findings

### UPF Page Status
- The UPF dataset page at `https://www.upf.edu/web/mtg/phenicx-conduct-dataset` **is accessible** via browser
- The 403 error from the download tool is caused by `requests.get()` without a User-Agent header — UPF blocks bot-like requests
- The page contains links to all 3 sub-datasets with individual datapack links

### RepoVizz Platform Status
- All datapack links point to `http://repovizz.upf.edu/repo/Vizz/{id}`
- RepoVizz returns **404** — the platform is no longer available
- The performance dataset link (`https://repovizz.upf.edu/phenicx/datasets/`) also returns 404
- No alternative download mirror found on Zenodo, GitHub, or elsewhere

### Datapack Link Structure
- **Spontaneous**: 25 participants × 3 fragments = 75 datapacks (IDs 232-345)
- **Articulation**: 24 participants × 4 conditions × 2 parts = 192 datapacks (IDs 1804-1995)
- **Performance**: linked via `repovizz.upf.edu/phenicx/datasets/` (also dead)

### Conclusion
PHENICX should be **reclassified from "auto" to "manual"** since the RepoVizz hosting platform is down with no alternative download source. The tool should generate instructions directing users to contact the MTG group at UPF for access.

## Sources
- UPF dataset page: https://www.upf.edu/web/mtg/phenicx-conduct-dataset
- RepoVizz (dead): https://repovizz.upf.edu/
