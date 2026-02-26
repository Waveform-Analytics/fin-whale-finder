# Roadmap

## Quick start

```bash
git clone https://github.com/Waveform-Analytics/fin-whale-finder.git
cd fin-whale-finder
uv sync

# Fetch data (7 days from OOI via M2M)
uv run python scripts/fetch_week_data.py

# Launch spectrogram viewer + labeling tool
uv run streamlit run scripts/spectrogram_viewer.py
# Open http://localhost:8501
```

## Current status

**Phase 0: complete.** Pilot data acquired, labeling complete. Pivoting to spectrogram image classification.

- **Data**: Jan 1-7, 2026 from Axial_Base hydrophone (168 hours, 200 Hz)
- **Verified**: Fin whale 20 Hz calls visible in spectrograms (hour 72 / Jan 3 shows clear calls)
- **Labels**: 228 usable clips (167 present, 61 not present). "Maybe" labels dropped for initial binary classification.
- **Perch finding**: Perch 2.0's mel spectrogram has a 60 Hz floor — it fundamentally cannot see fin whale 20 Hz calls. Perch remains viable for broadband soundscape analysis (see Future Directions) but is not on the path for fin whale detection.
- **Framework**: PyTorch for classification work. TensorFlow installed for Perch but kept separate.
- **Next**: Generate spectrogram images tuned for 15-30 Hz band, train CNN classifier (fine-tuned ResNet/EfficientNet) on existing labels

### Key parameters

- **Node**: Axial_Base (RS03AXBS-MJ03A-05-HYDLFA301)
- **Sampling rate**: 200 Hz (channel HDH)
- **Fin whale band**: 15-30 Hz
- **Clip duration for labeling**: 15 minutes (672 clips total)

### Project structure

```
fin-whale-finder/
├── .specify/memory/constitution.md   # Project rules and data sources
├── src/fin_whale_finder/
│   ├── data_access.py                # OOI data fetching with caching
│   ├── config.py                     # Configuration
│   ├── manifest.py                   # Data manifests
│   └── cli.py                        # CLI entrypoint (fwf)
├── scripts/
│   ├── fetch_week_data.py            # Download week of data
│   └── spectrogram_viewer.py         # Streamlit browser + labeling tool
├── labels/
│   └── labels.json                   # Human labels (tracked in git)
├── data/
│   └── cache/                        # Cached hydrophone data (gitignored)
├── docs/
│   └── roadmap.md                    # This file
├── configs/                          # Run parameters
├── models/                           # Trained model artifacts
├── notebooks/                        # Marimo notebooks
└── results/                          # Detection outputs
```

### Notes

- Streamlit runs locally only (not on OOI JupyterHub — no proxy support)
- OOI credentials stored in `~/.netrc` for `ooinet.oceanobservatories.org`
- M2M data downloads are slow from laptop (fast on JupyterHub — same network)
- Data is ~150MB per week at 200 Hz

---

## Learning arc

This project is learning-first. The goal is to build a working fin whale detector *and* develop real fluency in modern detection/classification approaches.

### Approach 1: Spectrogram image classification (PyTorch) [ACTIVE]

Generate spectrogram images tuned for the 15-30 Hz fin whale band, then train a CNN classifier (fine-tuned ResNet or EfficientNet) on the existing labeled dataset. This is the current active approach.

- **Dataset**: 228 usable clips (167 present, 61 not present) — "maybe" labels dropped for clean binary classification
- **Pipeline**: audio clip → spectrogram image (15-30 Hz band) → CNN → present/not present
- **Why this approach**: Perch 2.0 can't see below 60 Hz (see below), so we need a custom pipeline for the 20 Hz fin whale band. Spectrogram images are intuitive, debuggable, and a natural stepping stone to object detection.
- **Framework**: PyTorch (keeping TensorFlow separate for Perch-related work)
- Key concepts: transfer learning, fine-tuning, spectrogram generation, data augmentation, binary classification

### Approach 1b: Embedding + few-shot classification (Perch 2.0) [ON HOLD for fin whales]

**Critical finding**: Perch 2.0's mel spectrogram has a 60 Hz frequency floor. Fin whale 20 Hz calls are completely invisible to the model. This rules out Perch for fin whale detection specifically.

Burns et al. PIPAN results were mostly on species with energy above 60 Hz (humpback, orca), which explains why Perch performed well there. `perch-hoplite` and `tensorflow` are installed and working locally — they'll be used for broadband soundscape analysis (see Future Directions) where Perch's strengths actually apply.

Original context (still valid for non-fin-whale work):

[Perch 2.0](https://arxiv.org/abs/2508.04665) (Google DeepMind, released Aug 2025) was trained on 14,597 species — primarily birds — but transfers well to marine mammal tasks with no underwater audio in training. [Burns et al. (NeurIPS 2025)](https://arxiv.org/abs/2512.03219) showed Perch 2.0 is consistently top-performing for few-shot marine classification, including fin whale detection on the NOAA PIPAN dataset.

The recommended workflow is "agile modeling":
1. Embed audio windows using Perch 2.0
2. Search for candidates similar to known fin whale calls
3. Label candidates (we already have 235 labels)
4. Train a linear classifier on the embeddings (as few as 4-32 examples per class)
5. Iterate: classify at scale, review errors, refine

- **Package**: [`perch-hoplite`](https://github.com/google-research/perch-hoplite) (installable via pip)
- **Available models**: `perch_v2`, `perch_8`, `humpback`, `multispecies_whale`, `surfperch`, `birdnet_V2.3`
- Key concepts: embeddings, few-shot transfer learning, agile modeling, linear probing

### Approach 2: Object detection on spectrograms [FUTURE — builds on Approach 1]

Treat calls as objects in spectrogram images. Use CNN-based detectors (YOLO-family, Faster R-CNN, etc.) to find and classify them with bounding boxes. Same concept as DeepAcoustics/DeepSqueak but in Python.

This is the natural next step after binary classification (Approach 1). Instead of "does this clip contain a fin whale call?", the model draws bounding boxes around individual calls, giving precise timing and frequency information — which is needed for inter-pulse interval (IPI) analysis. Builds directly on the same spectrogram image pipeline.

- [DeepAcoustics](https://github.com/Ocean-Science-Analytics/DeepAcoustics) (MATLAB, forked from DeepSqueak) — Liz's group's tool. Study the underlying algorithm, not the GUI.
- [Sugarman et al., 2025](https://pubs.aip.org/asa/jasa/article/157/6/4613/3350873) — network selection and acoustic environment effects on object detection
- Key concepts: spectrograms as images, anchor boxes, IoU, NMS, transfer learning

### Approach 3: Sequence models / transformers (future)

Treat audio as a time series and learn temporal patterns directly. Potentially interesting for fin whales because of their distinctive rhythmic inter-pulse intervals.

- Key concepts: RNNs, attention mechanisms, transformers on audio features
- Explore when: after Approaches 1-2 give a working baseline

---

## Execution plan

### Phase 0 — Scope and success criteria [COMPLETE]

- Pilot dataset: 7 days of OOI hydrophone data (Jan 1-7, 2026, Axial_Base)
- Label taxonomy: `present`, `maybe`, `not_present`
- Labeling tool built (Streamlit, keyboard-driven: f/d/s keys)
- 235 clips labeled, fin whale calls confirmed visible

### Phase 1 — Spectrogram image classification (PyTorch) [ACTIVE]

- Generate spectrogram images tuned for 15-30 Hz band from existing labeled clips
- Split 228 labeled clips (167 present, 61 not present) into train/val/test sets
- Fine-tune ResNet or EfficientNet for binary classification (present / not present)
- Held-out evaluation from day one
- Output: v0 binary classifier, evaluation metrics, error analysis

### Phase 2 — Expand classification and iterate

- Address class imbalance (167:61 ratio) — augmentation, oversampling, or additional labeling
- Revisit "maybe" labels with classifier assistance
- Expand to more data (beyond the pilot week)
- Output: improved classifier, larger labeled dataset, error library

### Phase 3 — Object detection on spectrograms

- Move from clip-level classification to call-level bounding boxes
- Same spectrogram pipeline, but train YOLO or Faster R-CNN
- Output: per-call detections with timing and frequency, enabling IPI analysis

### Phase 4 — Production pilot and science outputs (~1-2 weeks)

- Process month-scale slice
- Generate detections for IPI/frequency analysis
- Comparison-ready summary vs prior studies

---

## First features to build

1. **Data slice selector** — choose station/time range, export file manifest
2. **Candidate table + review loop** — timestamp, score, spectrogram preview, one-click labels
3. **Query set manager** — save positive exemplars and hard negatives
4. **Run logging** — parameters, dataset slice, model, counts
5. **Evaluation snapshot** — precision/recall per cycle, confusion categories

---

## References & tools

### Perch 2.0 / agile modeling
- [Perch 2.0 transfers 'whale' to underwater tasks](https://arxiv.org/abs/2512.03219) — Burns et al., NeurIPS 2025 workshop paper
- [How AI trained on birds is surfacing underwater mysteries](https://research.google/blog/how-ai-trained-on-birds-is-surfacing-underwater-mysteries/) — Google Research blog (Feb 2026)
- [NOAA whale demo Colab notebook](https://github.com/google-research/perch/blob/main/chirp/projects/whale_demo/agile_modeling_noaa_demo.ipynb) — end-to-end agile modeling tutorial
- [`perch-hoplite`](https://github.com/google-research/perch-hoplite) — installable package for embedding + agile modeling
- [Google Perch repo](https://github.com/google-research/perch) — main research repo

### Object detection
- [DeepAcoustics](https://github.com/Ocean-Science-Analytics/DeepAcoustics) — object detection approach (MATLAB). Study the algorithm, not the GUI.
- [Sugarman et al., 2025](https://pubs.aip.org/asa/jasa/article/157/6/4613/3350873) — network selection and acoustic environment effects

### Data access
- [ooipy](https://github.com/Ocean-Data-Lab/ooipy) — OOI hydrophone data access (Python)
- [OOINet (Andy Reed)](https://github.com/reedan88/OOINet) — M2M API wrapper

### Mentorship & connections
- Liz Ferguson / OSA — detection/classification theory mentorship
- Dax, George Voulgaris — OOI relationships
- Andy Reed — OOI data access expertise

---

## Dependencies

- `ooipy` — OOI data access
- `streamlit` — spectrogram viewer + labeling UI
- `matplotlib` — plotting
- `scipy` — signal processing
- `numpy` — array operations
- `yooink` — OOI metadata
- `xarray`, `netcdf4` — data formats
- `torch`, `torchvision` — PyTorch for spectrogram image classification (Phase 1)
- `marimo` — notebooks (planned for Perch integration)
- `perch-hoplite` — Perch 2.0 embeddings + agile modeling (broadband soundscape work)
- `tensorflow` >= 2.20 — required by perch-hoplite (installed, kept separate from PyTorch work)

---

## Future directions

### Broadband soundscape analysis with Perch 2.0

This is still the big goal. Perch 2.0's 60 Hz mel floor rules it out for fin whale 20 Hz calls, but it's well-suited to everything above 60 Hz — which is where most of the interesting broadband soundscape lives. Use Perch embeddings on broadband OOI data (64 kHz instruments) for multi-class soundscape decomposition. This is genuinely novel/cutting-edge work. Complementary to (not competing with) the fin whale classifier.

This connects to prior work on MBON acoustic indices, but with richer event-level decomposition instead of bulk indices (ACI, NDSI, etc.).

**Why this works (and why Perch is the right tool here):**
- Perch 2.0 demonstrated strong performance on diverse marine sound tasks (NOAA PIPAN, ReefSet, DCLDE) — Burns et al. results were mostly on species with energy above 60 Hz (humpback, orca, etc.), right in Perch's sweet spot
- `perch-hoplite` and `tensorflow` are installed and working locally, ready when we are
- The existing labeling tool and workflow generalize to multi-class labeling
- OOI broadband instruments (64 kHz, e.g. Axial_Base_Seafloor via `low_freq=False`) open up a much wider frequency range

**Potential sound classes to explore:**
- Humpback whale
- Orca
- Other baleen whales (blue, sei)
- Odontocetes (if present in range)
- Fish sounds
- Ship noise / anthropogenic noise
- Seismic / volcanic activity (Axial Seamount is active)
- Rain, wind, other ambient

**Next steps:**
- Discuss with Liz Ferguson — identify sounds of interest, prioritize labeling targets
- Fetch a sample of broadband data and survey what's audible/visible
- Expand label taxonomy beyond fin whale presence/absence
- Evaluate Perch 2.0 embedding quality on higher-frequency marine sounds

### Gamified labeling tool

A web-based labeling app (beyond Streamlit) where people — including kids, in an educational context — could label spectrograms in a gamified way. A working classifier could pre-screen clips to serve up interesting ones for human labelers, making the task more engaging and less tedious.

- **Tech**: Next.js with D3/Observable Plot for spectrogram rendering
- **Prerequisite**: Working binary classifier (Phase 1) to pre-filter and rank clips
- **Connection**: More labels feed back into better classifiers — a virtuous cycle

**Calibration rounds**: New labelers start by classifying clips where we already have ground truth. This builds trust in both directions — we learn whether they're reliable, and they learn what fin whale calls actually look like. Once they demonstrate accuracy, they graduate to real unlabeled data. This is how Zooniverse-style citizen science projects handle quality control, and it doubles as onboarding.

**Active learning queue**: Use model confidence to decide which clips to serve labelers. Clips the model is 98% confident on don't need human review — the valuable ones are where the model is uncertain (40-70% confidence range). Every human label on an uncertain clip is maximally informative for improving the model. This keeps labeling efficient and keeps labelers working on genuinely interesting, ambiguous cases instead of obvious ones.

**Competitive model training**: Labelers' contributions directly train models — "your labels trained a model that got 94% accuracy, can someone beat that?" This gamifies the feedback loop and gives labelers a tangible sense of impact. Could also surface which labelers are most reliable over time, enabling label weighting (higher-accuracy labelers' labels count more in training).

**Feedback loop**: The overall system is: humans label → model trains → model pre-filters uncertain clips → humans label those → model improves → repeat. Each cycle makes the model better AND makes labeling more efficient, because the model handles more of the easy cases and only surfaces the hard ones. This is the virtuous cycle that makes the whole thing scale.
