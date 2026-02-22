# Fin Whale Finder - Project Handover

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd fin-whale-finder
uv sync

# Fetch data (7 days from OOI)
uv run python scripts/fetch_week_data.py

# Launch spectrogram viewer
uv run streamlit run scripts/spectrogram_viewer.py
# Open http://localhost:8501
```

## What We've Done

1. **Data pipeline**: Built `src/fin_whale_finder/data_access.py` - caching wrapper around `ooipy`
2. **Fetch script**: `scripts/fetch_week_data.py` - downloads week of hydrophone data with progress bar
3. **Spectrogram viewer**: `scripts/spectrogram_viewer.py` - Streamlit app to browse data
4. **Data**: Downloaded Jan 1-7, 2026 from Axial_Base (168 hours, 200 Hz)

## Verified: Fin Whales Visible!

- Hour 72 (Jan 3) shows clear fin whale 20 Hz calls - vertical blobs between 15-25 Hz
- Calls are ~1 sec pulses spaced ~20-30 sec apart

## Project Structure

```
fin-whale-finder/
├── .specify/memory/constitution.md   # Project rules and data sources
├── src/fin_whale_finder/
│   └── data_access.py                # OOI data fetching with caching
├── scripts/
│   ├── fetch_week_data.py            # Download week of data
│   └── spectrogram_viewer.py         # Streamlit spectrogram browser
├── data/
│   └── cache/                        # Cached hydrophone data (gitignored)
└── docs/
    └── roadmap.md                   # Original detection roadmap
```

## Key Parameters

- **Node**: Axial_Base (RS03AXBS-MJ03A-05-HYDLFA301)
- **Sampling rate**: 200 Hz (channel HDH)
- **Time range**: Jan 1-7, 2026
- **Fin whale band**: 15-30 Hz
- **Detection approach**: PERCH (not yet integrated)

## Next Steps

1. **Browse more data** - Use spectrogram viewer to find more fin whale activity
2. **PERCH integration** - Need to figure out:
   - How PERCH expects input format
   - Training vs inference mode
   - Labeling workflow for ground truth
3. **Labeling UI** - Simple y/n interface for marking detections

## Dependencies

- `ooipy` - OOI data access
- `streamlit` - Web UI
- `matplotlib` - Plotting
- `scipy` - Signal processing
- `numpy` - Array operations
- `tqdm` - Progress bars

## Notes

- Streamlit works locally but NOT on OOI JupyterHub (no proxy)
- OOI credentials stored in `~/.netrc` for `ooinet.oceanobservatories.org`
- Data is ~150MB for a week at 200 Hz
