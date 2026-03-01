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

### Active work
- **Soundscape exploration** (`notebooks/soundscape_explore.py`): Fetch broadband (64 kHz) data, embed with Perch v2, UMAP cluster, visualize. Designed to run on JupyterHub (big data downloads). This is the immediate next step.

## Architecture

- **Data pipeline**: `src/fin_whale_finder/data_access.py` — OOI data fetching with caching (supports both low-freq 200 Hz and broadband 64 kHz)
- **Labeling**: `scripts/spectrogram_viewer.py` — Streamlit app with keyboard-driven labeling
- **Classification**: `notebooks/classify_fins.py` — Marimo notebook, PyTorch/ResNet18
- **Soundscape**: `notebooks/soundscape_explore.py` — Marimo notebook, Perch 2.0 embeddings + UMAP
- **WAV export**: `scripts/export_wav.py` — converts cached pickles to WAV for Perch

## Key parameters

- **Node**: Axial_Base (RS03AXBS-MJ03A-05-HYDLFA301) for low-freq, Axial_Base_Seafloor for broadband
- **Sampling rate**: 200 Hz (low-freq) or 64 kHz (broadband)
- **Fin whale band**: 15-30 Hz
- **Spectrogram settings** (for classifier): FFT window 5s, overlap 50%, 10-50 Hz range
- **Clip duration**: 15 minutes (labeling), 1 hour (WAV export)

## Dependencies and environment

- Python 3.13, managed with `uv`
- PyTorch + torchvision — spectrogram image classification
- TensorFlow 2.20 + perch-hoplite — Perch 2.0 embeddings (broadband soundscape work)
- Both coexist without conflicts
- OOI credentials in `~/.netrc` for `ooinet.oceanobservatories.org`

## Roadmap and future directions

See `docs/roadmap.md` for full details. Three end goals:
1. **Fin whale detection** — individual call timing via object detection or classical methods
2. **Broadband soundscape analysis** — Perch 2.0 embeddings on OOI broadband data (novel/cutting-edge)
3. **Gamified labeling tool** — web-based citizen science for kids, with active learning and calibration rounds

## Conventions

- Always use `uv` (never pip)
- Notebooks in Marimo (not Jupyter)
- Marimo quirk: use separate markdown cells, not `mo.md()` inline with code
- Marimo quirk: prefix loop variables with `_` to avoid cross-cell conflicts
- Data files are gitignored; labels are tracked
