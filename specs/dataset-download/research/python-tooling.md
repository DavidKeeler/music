# Research: Python Download Tooling

## zenodo_get

- **PyPI:** https://pypi.org/project/zenodo-get/
- **GitHub:** https://github.com/dvolgyes/zenodo_get
- **Purpose:** Download complete Zenodo records by record ID or DOI
- **Features:** Handles multi-file records, checksum verification (MD5), can generate wget/curl scripts
- **Resume:** Supports `-k` flag to skip already downloaded files
- **Usage:** `zenodo_get 10.5281/zenodo.11393449` or `zenodo_get -r 11393449`
- **Note:** For restricted records (like MOSA), user must have Zenodo access token

## zenodo-downloader

- **PyPI:** https://pypi.org/project/zenodo-downloader/
- **Features:** Parallel downloading support
- **Newer alternative to zenodo_get**

## requests (stdlib-adjacent)

- Standard Python HTTP library
- Resume support via `Range` headers
- Good for direct URL downloads (Edinburgh DataShare, URMP, PHENICX)

## Recommended Approach

1. **Zenodo datasets (MOSA):** Use `zenodo_get` with access token for restricted records
2. **Direct HTTP downloads (Edinburgh, PHENICX, AIST++):** Use `requests` with:
   - `Range` header for resume
   - File existence check for idempotency
   - MD5/SHA256 verification where checksums are available
3. **Google Form-gated (URMP):** Document manual steps; optionally accept a pre-obtained download URL as CLI argument

## Directory Structure

```
~/data/conducting/
├── phenicx/           # PHENICX Conduct (3 sub-datasets)
│   ├── performance/
│   ├── spontaneous/
│   └── articulation/
├── edinburgh-mocap/   # Edinburgh C3D files
├── mosa/              # MOSA Zenodo download
├── aist-plusplus/     # AIST++ (replaces DeepDance)
├── urmp/              # URMP (manual download target)
├── human36m/          # Human3.6M (manual download target)
└── README.md          # Manual download instructions
```
