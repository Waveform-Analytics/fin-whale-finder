# Roadmap

## Learning arc

This project is learning-first. The goal is to build a working fin whale detector *and* develop real fluency in modern detection/classification approaches.

### Approach 1: Embedding + similarity search (Perch)

Encode audio windows into a vector space using a pretrained model, then find sounds similar to known fin whale calls. Low barrier to entry, minimal labeled data needed, great for candidate discovery.

- [Google Perch](https://github.com/google-research/perch) — pretrained bird audio embeddings, transferable to marine
- Key concepts: embeddings, cosine similarity, nearest-neighbor search

### Approach 2: Object detection on spectrograms

Treat calls as objects in spectrogram images. Use CNN-based detectors (YOLO-family, Faster R-CNN, etc.) to find and classify them with bounding boxes.

- [DeepAcoustics](https://github.com/Ocean-Science-Analytics/DeepAcoustics) (MATLAB, forked from DeepSqueak) — Liz's group's tool. Study the underlying algorithm, not the GUI.
- [Sugarman et al., 2025](https://pubs.aip.org/asa/jasa/article/157/6/4613/3350873) — network selection and acoustic environment effects on object detection
- Key concepts: spectrograms as images, anchor boxes, IoU, NMS, transfer learning

### Approach 3: Sequence models / transformers (future)

Treat audio as a time series and learn temporal patterns directly. Potentially interesting for fin whales because of their distinctive rhythmic inter-pulse intervals.

- Key concepts: RNNs, attention mechanisms, transformers on audio features
- Explore when: after Approaches 1-2 give a working baseline

## Execution plan

### Phase 0 — Scope and success criteria (1-2 days)

- Choose pilot dataset slice (3-7 days of OOI hydrophone data, then expand to 1 month)
- Define "good enough" for pilot (recall-focused triage quality)
- Label taxonomy: `fin`, `not_fin`, `maybe`

### Phase 1 — Candidate discovery with Perch (~1 week)

- Load pilot audio, embed windows, query with known fin examples
- Retrieve and review top candidates
- Output: ranked candidate table, 150-300 reviewed labels, error notes

### Phase 2 — Iterative label refinement (~1 week)

- Add confirmed hits as positive queries, add hard negatives
- Re-run retrieval, continue labeling
- Output: 500-1500 labeled windows, hard-negative library

### Phase 3 — Train first detector/classifier (~1-2 weeks)

- Train from curated labels, prioritize recall
- Held-out evaluation from day one
- Output: v0 model, evaluation metrics, error analysis

### Phase 4 — Production pilot and science outputs (~1-2 weeks)

- Process month-scale slice
- Generate detections for IPI/frequency analysis
- Comparison-ready summary vs prior studies

## First features to build

1. **Data slice selector** — choose station/time range, export file manifest
2. **Candidate table + review loop** — timestamp, score, spectrogram preview, one-click labels
3. **Query set manager** — save positive exemplars and hard negatives
4. **Run logging** — parameters, dataset slice, model, counts
5. **Evaluation snapshot** — precision/recall per cycle, confusion categories

## References & tools to study

- [DeepAcoustics](https://github.com/Ocean-Science-Analytics/DeepAcoustics) — object detection approach (MATLAB). Study the algorithm, not the GUI.
- [Google Perch](https://github.com/google-research/perch) — pretrained audio embeddings
- [ooipy](https://github.com/Ocean-Data-Lab/ooipy) — OOI hydrophone data access (Python)
- [OOINet (Andy Reed)](https://github.com/reedan88/OOINet) — M2M API wrapper
- Liz Ferguson / OSA — detection/classification theory mentorship
