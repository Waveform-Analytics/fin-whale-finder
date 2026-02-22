# fin-whale-finder

Detecting and classifying fin whale calls in OOI hydrophone data — and learning modern detection/classification approaches along the way.

## What this is

A learning-first project exploring how to find fin whale calls in continuous ocean acoustic data from the [Ocean Observatories Initiative (OOI)](https://oceanobservatories.org/). The goal is both practical (build a working detector) and educational (understand the landscape of bioacoustic detection methods).

## Learning arc

1. **Embedding + similarity search (Perch)** — encode audio into vector space, find sounds similar to known fin whale calls. Good for rapid candidate discovery with minimal labeled data.
2. **Object detection on spectrograms** — treat calls as objects in spectrogram images, use CNNs to detect and classify them. The approach used by [DeepAcoustics](https://github.com/Ocean-Science-Analytics/DeepAcoustics) (YOLO-family networks on spectrograms).
3. **Sequence models / transformers** — treat audio as a time series, learn temporal patterns directly. Potentially interesting for fin whales given their distinctive rhythmic pulse patterns (regular inter-pulse intervals).

Each phase builds on the previous: discover candidates, then detect precisely, then model temporal structure.

## Mentorship & connections

- **Liz Ferguson (OSA)** — mentoring on detection/classification theory
- **Dax, George Voulgaris** — OOI relationships
- **Andy Reed** — OOI data access expertise

## Data

OOI broadband hydrophone data, accessed via OOI JupyterHub. Data access utilities will live in this repo initially and may be extracted into a separate package later.

## Related projects

- [ooi-discovery](https://github.com/michellejw/ooi-discovery) — OOI metadata search tool (Phase 0 for data access)
- [science-spec-kit](https://github.com/Waveform-Analytics/science-spec-kit) — this project is a natural demo case

## Getting started

```bash
# Clone and set up environment
git clone https://github.com/Waveform-Analytics/fin-whale-finder.git
cd fin-whale-finder
uv sync

# Download a week of data from OOI
uv run python scripts/fetch_week_data.py

# Launch spectrogram viewer (local only - won't work on OOI JupyterHub)
uv run streamlit run scripts/spectrogram_viewer.py
```

## Current Status

- **Data**: Jan 1-7, 2026 from Axial_Base hydrophone (168 hours)
- **Verified**: Fin whale 20 Hz calls are visible in spectrograms
- **Next**: Integrate PERCH for automated detection

See [docs/HANDOVER.md](docs/HANDOVER.md) for detailed handover notes.
