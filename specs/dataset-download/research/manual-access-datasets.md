# Research: Manual-Access Datasets

## PHENICX Conduct Dataset (Tier 1)

- **URL:** https://www.upf.edu/web/mtg/phenicx-conduct-dataset
- **License:** CC BY-NC-SA 4.0
- **Access:** Publicly available! Data is hosted on RepoVizz platform (https://repovizz.upf.edu/phenicx/datasets/)
- **Contents (3 sub-datasets):**
  1. **Conductor during performance:** Beethoven 9th Symphony, Kinect V1 mocap (9 joints at 30fps), RGB video (640x480), RGBD video, 32-mic audio, aligned score
  2. **Spontaneous conducting:** 25 participants × 3 fragments (Beethoven 3rd), Kinect V1 (15 joints), beat annotations (ground truth), beat predictions, loudness predictions, mocap descriptors
  3. **Articulation study:** 24 participants × legato/staccato × 3/4 and 4/4, Kinect V1 (15 joints), beat annotations

**⚠️ Reclassification:** This is actually downloadable! Individual datapacks are linked directly from the UPF page as RepoVizz URLs. The script could potentially scrape these links and download them. However, the RepoVizz platform may require authentication or have rate limits.

## Edinburgh Orchestral Conducting MoCap (Tier 1)

- **DOI:** 10.7488/ds/2223
- **URL:** https://datashare.ed.ac.uk/ (search for handle based on DOI)
- **Direct download pattern:** `https://datashare.ed.ac.uk/download/DS_10283_{handle_id}.zip`
- **Size:** 54 C3D motion capture recordings
- **Access:** Edinburgh DataShare — appears to be open access (no login required for download)
- **Contents:** C3D format mocap files, 6 professional conductors, Mozart/Dvořák/Bartók
- **License:** Edinburgh DataShare end-user licence

**⚠️ Reclassification:** This may also be auto-downloadable! The DataShare platform provides direct download links. Need to confirm the exact handle ID to construct the download URL.

## Human3.6M (Tier 5)

- **URL:** http://vision.imar.ro/human3.6m/
- **License page:** http://vision.imar.ro/human3.6m/eula.php
- **Access:** Requires registration + license agreement (academic only). Must send request from academic email address.
- **Size:** Very large (~36 GB for poses only, much more with video)
- **Contents:** 3.6M frames, 11 subjects, 4 camera views, 50 Hz
- **Key files needed:** D3 Positions for subjects 1, 5, 6, 7, 8, 9, 11
- **Helper tool:** https://github.com/kotaro-inoue/human3.6m_downloader (Python downloader)

**Status:** Truly manual — requires academic registration and license acceptance through web form. Document instructions only.
