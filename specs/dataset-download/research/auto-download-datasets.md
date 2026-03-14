# Research: Auto-Download Datasets

## MOSA Dataset (Tier 2)

- **Zenodo record:** https://zenodo.org/records/11393449
- **DOI:** 10.5281/zenodo.11393449
- **Size:** ~2.3 TB
- **License:** CC BY 4.0
- **Access:** Restricted — requires Zenodo sign-in and access request (academic use)
- **GitHub:** https://github.com/yufenhuang/MOSA-Music-mOtion-and-Semantic-Annotation-dataset
- **Contents:** 742 piano/violin performances, 23 musicians, 30+ hours, 570K+ notes. 3D mocap (Qualisys/Vicon, 34 body/instrument markers), audio recordings, semantic annotations (notes, beats, harmony, dynamics, cadences, structure).
- **Format:** Motion data in proprietary mocap format, audio files, annotation text files.

**⚠️ Issue:** 2.3 TB is very large. Access is restricted (requires Zenodo request approval). This may need to be classified as "semi-manual" — the script can handle the download once access is granted, but the user must request access first.

## DeepDance Dataset (Tier 3)

- **Status: NOT FOUND as described.** No public dataset matching "DeepDance" with 30 dancers, 540 sequences, 240 Hz on Zenodo or elsewhere.
- **Recommended replacement: AIST++**
  - **URL:** https://google.github.io/aistplusplus_dataset/download.html
  - **Size:** 5.2 hours of 3D dance motion, 1408 sequences, 10 dance genres, 30 subjects
  - **Contents:** 3D keypoints (SMPL), multi-view video, paired music
  - **Access:** Free download from Google (videos from AIST Dance DB at https://aistdancedb.ongaaccel.jp/)
  - **License:** Creative Commons Attribution 4.0
  - **Advantage:** Larger, better documented, music-paired, 3D motion

## URMP Dataset (Tier 4)

- **URL:** http://labsites.rochester.edu/air/projects/URMP.html
- **Size:** ~12.5 GB (whole dataset package)
- **Access:** Requires Google Form submission: https://goo.gl/forms/xSvMzlwl3IWijvcp2
- **Contents:** 44 multi-instrument pieces, MIDI scores, individual audio tracks, assembled videos, pitch/note annotations
- **Format:** WAV audio, MP4 video, MIDI, text annotations

**⚠️ Issue:** Download requires filling out a Google Form. The actual download link is provided after form submission. This is semi-manual — document the form, but the script can't auto-download without the resulting link.
