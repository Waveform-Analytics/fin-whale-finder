# Fin Whale Finder — Project Context

## What this project is

A learning-first project to build a fin whale call detector from OOI (Ocean Observatories Initiative) hydrophone data, with a longer-term goal of broadband soundscape analysis using Perch 2.0.

Michelle has a PhD in oceanography (fin whale acoustics) and is building ML fluency through this project.

## Current state (Feb 2026)

### Completed
- **Phase 0**: 7 days of pilot data from Axial_Base (Jan 1-7, 2026, 200 Hz low-frequency hydrophone)
- **304 labeled clips** (15-min each): 167 present, 76 maybe, 61 not_present — in `labels/labels.json`
- **Spectrogram CNN classifier** (`notebooks/classify_fins.py`): ResNet18 fine-tuned on spectrograms, 98% accuracy on held-out test (46 clips). Proof of concept, small test set.
- **WAV export**: 168 one-hour WAV files in `data/wav/` (gitignored)

### Key finding: Perch 2.0 cannot see fin whale calls
Perch 2.0's mel spectrogram has a **60 Hz floor**. Fin whale 20 Hz calls are invisible to it. This rules out Perch for fin whale detection. It IS viable for broadband soundscape analysis (everything >60 Hz).

### Immediate next step: Broadband soundscape exploration on JupyterHub

Run `notebooks/soundscape_explore.py` on JupyterHub. This notebook:

1. Fetches 6 hours of **broadband** (64 kHz) data from `Axial_Base_Seafloor` via the existing data pipeline
2. Exports to WAV files (Perch's expected input format)
3. Embeds each 5-second window with **Perch v2** (produces 1280-dim vectors per window)
4. Runs UMAP to project embeddings to 2D
5. Clusters with k-means and shows example spectrograms from each cluster

**The goal**: see if Perch naturally separates different sound types (ship noise, whale calls, rain, seismics, etc.) without any labeling. If clusters look meaningful, take them to Liz Ferguson to help identify the sounds → that becomes the label taxonomy for soundscape classification.

This is a proof-of-concept / exploration step. No training, no labels needed yet.

**To run on JupyterHub:**
```bash
git pull
uv sync
uv run python notebooks/soundscape_explore.py
# Or with Marimo UI: uv run marimo edit notebooks/soundscape_explore.py
```

## Architecture

- **Data pipeline**: `src/fin_whale_finder/data_access.py` — OOI data fetching with caching (supports both low-freq 200 Hz and broadband 64 kHz)
- **Labeling**: `scripts/spectrogram_viewer.py` — Streamlit app with keyboard-driven labeling
- **Classification**: `notebooks/classify_fins.py` — Marimo notebook, PyTorch/ResNet18
- **Soundscape**: `notebooks/soundscape_explore.py` — Marimo notebook, Perch 2.0 embeddings + UMAP
- **WAV export**: `scripts/export_wav.py` — converts cached pickles to WAV for Perch

## OOI instruments and data access

### Target instruments (Cabled Axial Seamount Array)

Low frequency (200 Hz, channel HDH):
- `Axial_Base` — RS03AXBS-MJ03A-05-HYDLFA301, 2642 m, HTI-90-U hydrophone
- `Central_Caldera` — RS03CCAL-MJ03F-06-HYDLFA305
- `Eastern_Caldera` — RS03ECAL-MJ03E-09-HYDLFA304

Broadband (64 kHz):
- `Axial_Base_Seafloor` — RS03AXBS-LJ03A-09-HYDBBA302

Coverage: 2015-12-14 to present (continuous). All timestamps UTC.

### Data access

- API: OOI M2M via `ooipy`, wrapped by `src/fin_whale_finder/data_access.py`
- `get_hydrophone_data(node, starttime, endtime, low_freq=True)` — fetches with caching
- `low_freq=True` → 200 Hz HYDLFA data; `low_freq=False` → 64 kHz HYDBBA broadband
- Credentials: `~/.netrc` for `ooinet.oceanobservatories.org`
- Cache: `data/cache/` (pickle files, gitignored)
- M2M downloads are slow from laptops, fast on JupyterHub (same network)

### JupyterHub environment

- URL: `https://jupyter.oceanobservatories.org`
- Data mounted under `/home/jovyan/ooi/`
- Marimo can't be proxied through JupyterHub without `jupyter-marimo-proxy` at the hub level. Run via `uv run marimo edit <file>` from a terminal with SSH port-forwarding, or run as plain script with `uv run python <file>`.

## Key parameters

- **Fin whale band**: 15-30 Hz
- **Spectrogram settings** (for classifier): FFT window 5s, overlap 50%, 10-50 Hz range
- **Clip duration**: 15 minutes (labeling), 1 hour (WAV export)

## Dependencies and environment

- Python 3.12, managed with `uv` (never pip)
- PyTorch + torchvision — spectrogram image classification
- TensorFlow 2.20 + perch-hoplite — Perch 2.0 embeddings (broadband soundscape work)
- Both coexist without conflicts
- umap-learn — dimensionality reduction for embedding visualization

## Roadmap and future directions

See `docs/roadmap.md` for full details. Three end goals:
1. **Fin whale detection** — individual call timing via object detection or classical methods
2. **Broadband soundscape analysis** — Perch 2.0 embeddings on OOI broadband data (novel/cutting-edge)
3. **Gamified labeling tool** — web-based citizen science for kids, with active learning and calibration rounds

## Key references

- [Perch 2.0 paper (Bittern)](https://arxiv.org/abs/2508.04665) — architecture, training, mel spectrogram params (60 Hz floor)
- [Burns et al. NeurIPS 2025](https://arxiv.org/abs/2512.03219) — Perch 2.0 transfer to marine tasks (PIPAN, DCLDE)
- [Google Research blog: AI trained on birds surfacing underwater mysteries](https://research.google/blog/how-ai-trained-on-birds-is-surfacing-underwater-mysteries/) (Feb 2026)
- [NOAA whale demo Colab](https://github.com/google-research/perch/blob/main/chirp/projects/whale_demo/agile_modeling_noaa_demo.ipynb) — end-to-end agile modeling tutorial
- [`perch-hoplite` repo](https://github.com/google-research/perch-hoplite) — installable package

## Conventions

- Always use `uv` (never pip)
- Notebooks in Marimo (not Jupyter)
- Marimo quirk: use separate markdown cells, not `mo.md()` inline with code
- Marimo quirk: prefix loop variables with `_` to avoid cross-cell conflicts
- Data files are gitignored; labels are tracked
