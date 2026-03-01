import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Soundscape exploration with Perch 2.0

    **Goal**: Fetch a few hours of broadband OOI hydrophone data, embed it with
    Perch 2.0, and see if the embeddings naturally cluster into distinct sound
    types — without any labeling.

    **Why this works**: Perch 2.0 was trained to produce embeddings where similar
    sounds land near each other. If the underwater soundscape contains distinct
    sound types (ship noise, whale calls, rain, seismic activity), they should
    form separate clusters in embedding space. We can visualize those clusters
    with UMAP (a dimensionality reduction technique that squashes 1280 dimensions
    down to 2 for plotting).

    **What we're hoping to see**: Distinct blobs in the UMAP plot. We'll then pull
    example spectrograms from each blob to figure out what each sound type is.
    That gives us a starting point for building a label taxonomy — and something
    concrete to show Liz.

    **This notebook is designed to run on JupyterHub** where OOI data downloads
    are fast (same network).
    """)
    return


@app.cell
def _():
    """Imports."""
    import pickle
    import tomllib
    from datetime import datetime
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for headless environments
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import signal
    from scipy.io import wavfile

    return (
        Path,
        datetime,
        np,
        pickle,
        plt,
        signal,
        tomllib,
        wavfile,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 1. Configuration

    We're pulling broadband data (64 kHz) from Axial_Base_Seafloor. This is
    a different instrument than the low-frequency hydrophone (200 Hz) we used
    for fin whale labeling.

    **Why broadband?** The low-frequency stream maxes out at 100 Hz. Perch 2.0's
    mel spectrogram covers 60 Hz – 16 kHz, so we need data with content in that
    range. The broadband instrument captures up to 32 kHz — plenty of overlap
    with Perch's window.

    **How much data?** 6 hours is enough to capture different sound conditions
    (day/night, shipping patterns, etc.) while keeping download and embedding
    times reasonable. That gives us ~4,300 five-second windows for Perch to embed.
    """)
    return


@app.cell
def _(Path, datetime, tomllib):
    # --- Load config from TOML (single source of truth) ---
    _config_path = Path(__file__).parent.parent / "configs" / "soundscape.toml"
    with open(_config_path, "rb") as _f:
        _cfg = tomllib.load(_f)

    NODE = _cfg["node"]
    START = _cfg["start"]
    END = _cfg["end"]
    FETCH_CHUNK_HOURS = _cfg["chunk_hours"]
    PERCH_MODEL_NAME = _cfg["perch_model"]
    N_CLUSTERS = _cfg["n_clusters"]

    # Derive a slug for namespacing outputs — changing the date range
    # automatically writes to a new directory, never clobbering old results.
    _start_dt = datetime.fromisoformat(START)
    _end_dt = datetime.fromisoformat(END)
    _slug = f"{_start_dt.strftime('%Y-%m-%d')}_{_start_dt.strftime('%Hh')}-{_end_dt.strftime('%Hh')}"

    PROJECT_ROOT = Path(__file__).parent.parent
    CACHE_DIR = PROJECT_ROOT / "data" / "cache_broadband"
    WAV_DIR = PROJECT_ROOT / "data" / "wav_broadband" / _slug
    EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings" / _slug
    FIGURES_DIR = PROJECT_ROOT / "results" / "soundscape_figures" / _slug
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Config: {_config_path}")
    print(f"Node:   {NODE}")
    print(f"Range:  {START} → {END}")
    print(f"Run:    {_slug}")
    print(f"Figs:   {FIGURES_DIR}")

    return (
        CACHE_DIR,
        EMBEDDING_DIR,
        END,
        FETCH_CHUNK_HOURS,
        FIGURES_DIR,
        N_CLUSTERS,
        NODE,
        PERCH_MODEL_NAME,
        PROJECT_ROOT,
        START,
        WAV_DIR,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 2. Load broadband data from cache

    Loads pre-fetched broadband data from the local cache. If files are missing,
    run the fetch script first:

    ```bash
    uv run python scripts/fetch_broadband.py
    ```
    """)
    return


@app.cell
def _(CACHE_DIR, END, FETCH_CHUNK_HOURS, NODE, START, datetime, pickle):
    from datetime import timedelta

    _start = datetime.fromisoformat(START)
    _end = datetime.fromisoformat(END)

    # Build expected cache filenames using the same key format as data_access.py
    _current = _start
    _expected = []
    while _current < _end:
        _chunk_end = min(_current + timedelta(hours=FETCH_CHUNK_HOURS), _end)
        _node_clean = NODE.replace(" ", "_").replace("/", "_")
        _key = f"{_node_clean}_{_current.strftime('%Y-%m-%d_%H-%M-%S')}_{_chunk_end.strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
        _expected.append((CACHE_DIR / _key, _current, _chunk_end))
        _current = _chunk_end

    # Fail fast if any files are missing (neither .pkl nor .empty marker exists)
    _missing = [
        str(f) for f, _, _ in _expected
        if not f.exists() and not (f.parent / (f.name + ".empty")).exists()
    ]
    if _missing:
        raise FileNotFoundError(
            f"Missing {len(_missing)} cache file(s):\n" +
            "\n".join(f"  {f}" for f in _missing) +
            "\n\nRun the fetch script first:\n" +
            "  uv run python scripts/fetch_broadband.py"
        )

    # Load from cache, skipping OOI gaps (marked with .empty)
    broadband_chunks = []
    for _cache_file, _t0, _t1 in _expected:
        _marker = _cache_file.parent / (_cache_file.name + ".empty")
        if _marker.exists():
            print(f"Skipping {_t0.isoformat()} to {_t1.isoformat()} (OOI gap)")
            continue
        print(f"Loading {_t0.isoformat()} to {_t1.isoformat()}...")
        with open(_cache_file, "rb") as _f:
            _chunk = pickle.load(_f)
        broadband_chunks.append(_chunk)
        _fs = round(_chunk.stats.sampling_rate)
        _n = len(_chunk.data)
        print(f"  {_n:,} samples ({_n/_fs:.0f}s) at {_fs} Hz")

    print(f"\nLoaded {len(broadband_chunks)} chunks")

    return (broadband_chunks,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Export to WAV

    Perch expects WAV files as input. We'll export each hour as a separate file.
    The audio is normalized to [-1, 1] float32 (standard WAV format).
    """)
    return


@app.cell
def _(WAV_DIR, broadband_chunks, np, wavfile):
    WAV_DIR.mkdir(parents=True, exist_ok=True)

    wav_files = []

    for _chunk in broadband_chunks:
        _fs = round(_chunk.stats.sampling_rate)
        _starttime = _chunk.stats.starttime
        _timestamp = _starttime.strftime("%Y-%m-%dT%H-%M")
        _filename = f"broadband_{_timestamp}.wav"
        _filepath = WAV_DIR / _filename

        if _filepath.exists():
            wav_files.append(_filepath)
            print(f"Cached  {_filename} ({_fs} Hz)")
            continue

        _data = np.array(_chunk.data, dtype=np.float32)

        # Replace any NaN/Inf from instrument glitches or data gaps with zeros
        _data = np.nan_to_num(_data, nan=0.0, posinf=0.0, neginf=0.0)

        # Despike: OOI transmits data in 1-second packets; seams between packets
        # create broadband impulse artifacts. Find samples far from the median
        # (using MAD, which is robust to spikes themselves) and interpolate over them.
        _med = np.median(_data)
        _mad = np.median(np.abs(_data - _med))
        _spike_mask = np.abs(_data - _med) > 8 * _mad
        if _spike_mask.any():
            _idx = np.arange(len(_data))
            _data[_spike_mask] = np.interp(
                _idx[_spike_mask], _idx[~_spike_mask], _data[~_spike_mask]
            )
            print(f"  Despiked {_spike_mask.sum()} samples ({100*_spike_mask.mean():.2f}%)")

        # Remove DC offset and normalize
        _data = _data - _data.mean()
        _peak = np.abs(_data).max()
        if _peak > 0:
            _data = _data / _peak

        wavfile.write(_filepath, _fs, _data)
        wav_files.append(_filepath)
        print(f"Wrote   {_filename} ({_fs} Hz, {len(_data)/_fs:.0f}s)")

    print(f"\n{len(wav_files)} WAV files in {WAV_DIR}")

    return (wav_files,)


@app.cell
def _(mo):
    mo.md("""
    ## 4. Quick look at the broadband spectrograms

    Before embedding, let's see what the data looks like. This shows a
    1-minute spectrogram from the first chunk across the full frequency range.
    You should see a much richer picture than the low-frequency data —
    energy across a wide band, possibly with visible ship noise (narrowband
    lines), whale calls, or other features.
    """)
    return


@app.cell
def _(FIGURES_DIR, broadband_chunks, np, plt, signal):
    # Grab first 60 seconds of first chunk for a quick look
    _chunk = broadband_chunks[0]
    _data = np.array(_chunk.data, dtype=np.float32)
    _fs = round(_chunk.stats.sampling_rate)
    _n_samples = min(60 * _fs, len(_data))  # 60 seconds
    _snippet = _data[:_n_samples]

    _nperseg = int(0.05 * _fs)  # 50ms FFT window for broadband
    _f, _t, _Sxx = signal.spectrogram(
        _snippet, fs=_fs, nperseg=_nperseg, noverlap=_nperseg // 2
    )
    _Sxx_db = 10 * np.log10(_Sxx + 1e-10)

    # Clip colorscale to 2nd–98th percentile so spike artifacts don't
    # dominate and wash out the actual signal variation.
    _vmin = np.percentile(_Sxx_db, 2)
    _vmax = np.percentile(_Sxx_db, 98)

    fig_spec, (ax_full, ax_low) = plt.subplots(2, 1, figsize=(14, 8))

    # Full range
    ax_full.pcolormesh(_t, _f, _Sxx_db, shading="gouraud", cmap="viridis",
                       vmin=_vmin, vmax=_vmax)
    ax_full.set_ylabel("Frequency (Hz)")
    ax_full.set_title(f"Broadband spectrogram — full range (0–{_fs//2} Hz)")
    ax_full.set_ylim(0, _fs // 2)

    # Zoomed to Perch's range
    ax_low.pcolormesh(_t, _f, _Sxx_db, shading="gouraud", cmap="viridis",
                      vmin=_vmin, vmax=_vmax)
    ax_low.set_ylabel("Frequency (Hz)")
    ax_low.set_xlabel("Time (seconds)")
    ax_low.set_title("Zoomed to Perch's range (60–16,000 Hz)")
    ax_low.set_ylim(60, 16000)

    plt.tight_layout()
    fig_spec.savefig(FIGURES_DIR / "01_broadband_spectrogram.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {FIGURES_DIR / '01_broadband_spectrogram.png'}")
    fig_spec

    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Embed with Perch 2.0

    Now the interesting part. We load the Perch v2 model and feed it our WAV
    files. For each file, Perch slides a 5-second window across the audio and
    produces a **1280-dimensional embedding vector** for each window.

    These vectors are the "coordinates" of each sound in Perch's learned space.
    Similar sounds → similar vectors → nearby points when we plot them.

    This step may take a few minutes depending on how much data you have.
    """)
    return


@app.cell
def _(EMBEDDING_DIR, PERCH_MODEL_NAME, Path, np, wav_files):
    from perch_hoplite.zoo.model_configs import load_model_by_name
    from perch_hoplite import audio_io

    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    # Check if we already computed embeddings
    _emb_file = EMBEDDING_DIR / "broadband_embeddings.npz"

    if _emb_file.exists():
        print("Loading cached embeddings...")
        _cached = np.load(_emb_file)
        all_embeddings = _cached["embeddings"]
        all_offsets = _cached["offsets"]
        all_file_indices = _cached["file_indices"]
        print(f"Loaded {len(all_embeddings)} embeddings")
    else:
        print(f"Loading {PERCH_MODEL_NAME} model...")
        perch_model = load_model_by_name(PERCH_MODEL_NAME)
        print(f"Model loaded. Sample rate: {perch_model.sample_rate}, "
              f"window: {perch_model.window_size_s}s, hop: {perch_model.hop_size_s}s")

        all_embeddings = []
        all_offsets = []
        all_file_indices = []

        for _file_idx, _wav_path in enumerate(wav_files):
            print(f"\nEmbedding {_wav_path.name}...")

            # Load audio at Perch's expected sample rate
            _audio = audio_io.load_audio_window(
                filepath=str(_wav_path),
                offset_s=0,
                sample_rate=perch_model.sample_rate,
                window_size_s=-1,  # load entire file
            )
            _audio = np.array(_audio)
            print(f"  Audio: {len(_audio)} samples "
                  f"({len(_audio)/perch_model.sample_rate:.0f}s at {perch_model.sample_rate} Hz)")

            # Embed — this returns an object with .embeddings attribute
            _outputs = perch_model.embed(_audio)
            _embs = _outputs.embeddings

            if _embs is not None and len(_embs) > 0:
                # embeddings shape: (n_windows, 1, embedding_dim) — squeeze channel axis
                if _embs.ndim == 3:
                    _embs = _embs[:, 0, :]

                # Track which file and offset each embedding came from
                for _w in range(len(_embs)):
                    _offset = _w * perch_model.hop_size_s
                    all_embeddings.append(_embs[_w])
                    all_offsets.append(_offset)
                    all_file_indices.append(_file_idx)

                print(f"  {len(_embs)} windows embedded (dim={_embs.shape[-1]})")

        all_embeddings = np.array(all_embeddings)
        all_offsets = np.array(all_offsets)
        all_file_indices = np.array(all_file_indices)

        # Cache embeddings
        np.savez(
            _emb_file,
            embeddings=all_embeddings,
            offsets=all_offsets,
            file_indices=all_file_indices,
        )
        print(f"\nSaved {len(all_embeddings)} embeddings to {_emb_file}")

    print(f"\nTotal: {len(all_embeddings)} embeddings, shape: {all_embeddings.shape}")

    return all_embeddings, all_file_indices, all_offsets


@app.cell
def _(mo):
    mo.md("""
    ## 6. UMAP visualization

    UMAP (Uniform Manifold Approximation and Projection) takes our
    1280-dimensional embeddings and projects them down to 2D so we can
    actually see them on a plot. It tries to preserve the neighborhood
    structure — points that are close in 1280D should stay close in 2D.

    **What to look for**: distinct clusters or blobs of points. Each cluster
    likely corresponds to a different type of sound. Spread-out, uniform
    distributions would mean Perch isn't finding meaningful structure
    (unlikely, but possible).

    Points are colored by time so you can see if certain sound types
    are concentrated at particular times of day.
    """)
    return


@app.cell
def _(FIGURES_DIR, all_embeddings, all_offsets, all_file_indices, np, plt):
    from umap import UMAP

    print("Running UMAP (this may take a moment)...")
    _reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    umap_coords = _reducer.fit_transform(all_embeddings)
    print(f"UMAP done: {umap_coords.shape}")

    # Color by time (offset within the full recording)
    # Combine file index and offset for a continuous time variable
    _hours_into_recording = all_file_indices + all_offsets / 3600

    fig_umap, ax_umap = plt.subplots(figsize=(10, 8))
    _scatter = ax_umap.scatter(
        umap_coords[:, 0],
        umap_coords[:, 1],
        c=_hours_into_recording,
        cmap="twilight",
        s=3,
        alpha=0.5,
    )
    plt.colorbar(_scatter, ax=ax_umap, label="Hours into recording")
    ax_umap.set_xlabel("UMAP 1")
    ax_umap.set_ylabel("UMAP 2")
    ax_umap.set_title("Perch 2.0 embeddings — UMAP projection\n(each point = one 5-second window)")
    fig_umap.savefig(FIGURES_DIR / "02_umap_by_time.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {FIGURES_DIR / '02_umap_by_time.png'}")
    fig_umap

    return (umap_coords,)


@app.cell
def _(mo):
    mo.md("""
    ## 7. Explore clusters

    Now let's look at what's *in* the clusters. We pick a few representative
    points from different regions of the UMAP plot and show their spectrograms.

    This is the "aha" moment — if the clusters correspond to recognizable
    sounds, Perch is doing its job and this approach has legs.

    We use a simple k-means clustering to automatically identify groups,
    then show one example spectrogram from each cluster.
    """)
    return


@app.cell
def _(
    FIGURES_DIR,
    N_CLUSTERS,
    all_file_indices,
    all_offsets,
    np,
    plt,
    signal,
    umap_coords,
    wav_files,
    wavfile,
):
    import math as _math
    from sklearn.cluster import KMeans

    # Find clusters in UMAP space
    _n_clusters = N_CLUSTERS
    _kmeans = KMeans(n_clusters=_n_clusters, random_state=42, n_init=10)
    cluster_labels = _kmeans.fit_predict(umap_coords)

    # For each cluster, pick the point closest to the centroid
    _ncols = min(_n_clusters, 4)
    _nrows = _math.ceil(_n_clusters / _ncols)
    fig_clusters, axes_cl = plt.subplots(_nrows, _ncols, figsize=(18, _nrows * 4 + 1))
    axes_cl = np.array(axes_cl).flatten()

    for _c in range(_n_clusters):
        _mask = cluster_labels == _c
        _cluster_points = umap_coords[_mask]
        _centroid = _cluster_points.mean(axis=0)

        # Find closest point to centroid
        _dists = np.linalg.norm(_cluster_points - _centroid, axis=1)
        _closest_in_cluster = np.where(_mask)[0][_dists.argmin()]

        # Load the audio for this point
        _file_idx = int(all_file_indices[_closest_in_cluster])
        _offset_s = float(all_offsets[_closest_in_cluster])
        _wav_path = wav_files[_file_idx]

        _fs_wav, _audio_full = wavfile.read(_wav_path)
        _start_sample = int(_offset_s * _fs_wav)
        _end_sample = _start_sample + int(5.0 * _fs_wav)  # 5-second window
        _end_sample = min(_end_sample, len(_audio_full))
        _snippet = _audio_full[_start_sample:_end_sample].astype(np.float32)

        # Generate spectrogram
        _nperseg = int(0.05 * _fs_wav)  # 50ms window
        _f, _t, _Sxx = signal.spectrogram(
            _snippet, fs=_fs_wav, nperseg=_nperseg, noverlap=_nperseg // 2
        )
        _Sxx_db = 10 * np.log10(_Sxx + 1e-10)

        axes_cl[_c].pcolormesh(_t, _f, _Sxx_db, shading="gouraud", cmap="viridis")
        axes_cl[_c].set_ylim(60, 16000)
        axes_cl[_c].set_title(
            f"Cluster {_c} (n={_mask.sum()})\n"
            f"File {_file_idx}, offset {_offset_s:.0f}s",
            fontsize=9,
        )
        if _c % _ncols == 0:
            axes_cl[_c].set_ylabel("Freq (Hz)")
        if _c >= _ncols * (_nrows - 1):
            axes_cl[_c].set_xlabel("Time (s)")

    plt.suptitle("Example spectrograms from each cluster", fontsize=13)
    plt.tight_layout()
    fig_clusters.savefig(FIGURES_DIR / "03_cluster_spectrograms.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {FIGURES_DIR / '03_cluster_spectrograms.png'}")
    fig_clusters

    return (cluster_labels,)


@app.cell
def _(FIGURES_DIR, cluster_labels, np, plt, umap_coords):
    # Show the UMAP plot again, this time colored by cluster
    fig_labeled, ax_labeled = plt.subplots(figsize=(10, 8))

    _n_clusters = len(np.unique(cluster_labels))
    _colors = plt.cm.tab10(np.linspace(0, 1, _n_clusters))

    for _c in range(_n_clusters):
        _mask = cluster_labels == _c
        ax_labeled.scatter(
            umap_coords[_mask, 0],
            umap_coords[_mask, 1],
            c=[_colors[_c]],
            s=3,
            alpha=0.5,
            label=f"Cluster {_c} (n={_mask.sum()})",
        )

    ax_labeled.legend(markerscale=5, fontsize=8)
    ax_labeled.set_xlabel("UMAP 1")
    ax_labeled.set_ylabel("UMAP 2")
    ax_labeled.set_title("UMAP colored by cluster assignment")
    fig_labeled.savefig(FIGURES_DIR / "04_umap_by_cluster.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {FIGURES_DIR / '04_umap_by_cluster.png'}")
    fig_labeled

    return


@app.cell
def _(mo):
    mo.md("""
    ## What's next?

    If you see distinct clusters with recognizably different spectrograms:

    1. **Identify the sounds** — show the cluster spectrograms to Liz,
       listen to the audio, compare to known marine sound catalogs
    2. **Name the clusters** — start building a label taxonomy
       (e.g., "ship_noise", "humpback", "ambient_low", "rain", etc.)
    3. **Label a sample** — use the existing Streamlit tool (or build a new one)
       to confirm/correct the cluster assignments
    4. **Train a classifier** — use the labeled embeddings to train a
       lightweight classifier (logistic regression on Perch embeddings)

    If the clusters are messy or hard to interpret, try:
    - Different UMAP parameters (n_neighbors, min_dist)
    - More or fewer clusters in k-means
    - A different time range (night vs day, different season)
    - Filtering out very quiet windows (ambient noise floor)
    """)
    return


if __name__ == "__main__":
    app.run()
