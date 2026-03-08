#!/usr/bin/env python
"""
Soundscape exploration with Perch 2.0 embeddings + HDBSCAN anomaly detection.

Run after fetching data with scripts/fetch_broadband.py.

Usage
-----
    uv run python scripts/soundscape_explore.py

Config lives in configs/soundscape.toml. Output goes to
results/soundscape_figures/<node>/<slug>/ where slug is derived from the
date range (e.g. 2025-06-04_12h-18h).

Output figures
--------------
    01_umap_by_time.png           UMAP colored by time
    02_umap_clusters_outliers.png UMAP colored by HDBSCAN label + outlier score
    03_outlier_timeline.png       Outlier score over time — shows WHEN events occurred
    04_top_anomalies.png          Spectrograms of the N most unusual 5-second windows
    05_cluster_examples.png       One representative spectrogram per HDBSCAN cluster
"""

import math
import os
import pickle
import time
import tomllib
from datetime import datetime, timedelta
from pathlib import Path

# Must be set before TensorFlow / Perch is imported.
# Tell TF to grow GPU memory on demand rather than pre-allocating the whole GPU.
# This is essential when running multiple analysis processes in parallel.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(date_override=None, duty_cycle_override=None):
    """Load config from soundscape.toml.

    Args:
        date_override:       'YYYY-MM-DD' — overrides start/end to one calendar day.
        duty_cycle_override: int minutes — overrides duty_cycle_minutes in the TOML.
    """
    config_path = PROJECT_ROOT / "configs" / "soundscape.toml"
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    if date_override:
        dt = datetime.strptime(date_override, "%Y-%m-%d")
        cfg["start"] = dt.strftime("%Y-%m-%dT00:00:00")
        cfg["end"] = (dt + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")

    if duty_cycle_override is not None:
        cfg["duty_cycle_minutes"] = duty_cycle_override

    start_dt = datetime.fromisoformat(cfg["start"])
    end_dt = datetime.fromisoformat(cfg["end"])
    slug = (
        f"{start_dt.strftime('%Y-%m-%d')}"
        f"_{start_dt.strftime('%Hh')}-{end_dt.strftime('%Hh')}"
    )
    node = cfg["node"]
    dirs = {
        "cache": PROJECT_ROOT / "data" / "cache_broadband",
        "wav": PROJECT_ROOT / "data" / "wav_broadband" / node / slug,
        "embeddings": PROJECT_ROOT / "data" / "embeddings" / node / slug,
        "figures": PROJECT_ROOT / "results" / "soundscape_figures" / node / slug,
    }
    dirs["figures"].mkdir(parents=True, exist_ok=True)

    dc = cfg.get("duty_cycle_minutes", 0)
    chunk_min = cfg.get("chunk_hours", 1) * 60
    dc_label = (
        f"{dc} min/{chunk_min} min per chunk  ({dc / chunk_min * 100:.0f}% of data)"
        if dc > 0
        else "full (no duty cycle)"
    )

    print(f"Config:      {config_path}")
    print(f"Node:        {cfg['node']}")
    print(f"Range:       {cfg['start']} → {cfg['end']}")
    print(f"Run:         {slug}")
    print(f"Duty cycle:  {dc_label}")
    print(f"Figures:     {dirs['figures']}")
    print()

    return cfg, dirs, start_dt


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_broadband_chunks(cfg, dirs):
    from fin_whale_finder.data_access import _make_cache_key

    start_dt = datetime.fromisoformat(cfg["start"])
    end_dt = datetime.fromisoformat(cfg["end"])
    node = cfg["node"]
    cache_dir = dirs["cache"]

    expected = []
    current = start_dt
    while current < end_dt:
        chunk_end = min(current + timedelta(hours=cfg["chunk_hours"]), end_dt)
        key = _make_cache_key(node, current, chunk_end)
        expected.append((cache_dir / key, current, chunk_end))
        current = chunk_end

    missing = [
        str(f)
        for f, _, _ in expected
        if not f.exists() and not (f.parent / (f.name + ".empty")).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} cache file(s):\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\n\nRun first:\n  uv run python scripts/fetch_broadband.py"
        )

    chunks = []
    for cache_file, t0, t1 in expected:
        marker = cache_file.parent / (cache_file.name + ".empty")
        if marker.exists():
            print(f"Skipping {t0.isoformat()} → {t1.isoformat()} (OOI gap)")
            continue
        print(f"Loading  {t0.isoformat()} → {t1.isoformat()}...")
        with open(cache_file, "rb") as f:
            chunk = pickle.load(f)
        chunks.append(chunk)
        fs = round(chunk.stats.sampling_rate)
        n = len(chunk.data)
        print(f"  {n:,} samples ({n / fs:.0f}s) at {fs} Hz")

    print(f"\nLoaded {len(chunks)} chunks")
    return chunks


# ---------------------------------------------------------------------------
# WAV export with despiking
# ---------------------------------------------------------------------------


def _despike_rolling(data, fs, window_s=5.0, threshold=8.0):
    """Rolling 1-second window MAD despiking.

    Instead of computing a single global MAD (which is inflated when spike
    rate is high), we split the signal into 1-second windows matching OOI's
    packet boundaries and compute median/MAD independently per window.
    This keeps the threshold local, so a heavily-contaminated second doesn't
    raise the bar for neighbouring seconds.
    """
    window = int(window_s * fs)
    n = len(data)
    pad = (-n) % window  # pad to exact multiple of window
    padded = np.pad(data.astype(np.float64), (0, pad), mode="reflect")
    chunks = padded.reshape(-1, window)
    meds = np.median(chunks, axis=1, keepdims=True)
    mads = np.median(np.abs(chunks - meds), axis=1, keepdims=True)
    mads = np.maximum(mads, 1e-10)
    spike_mask = (np.abs(chunks - meds) > threshold * mads).reshape(-1)[:n]
    if spike_mask.any():
        idx = np.arange(n)
        data = data.astype(np.float32).copy()
        data[spike_mask] = np.interp(
            idx[spike_mask], idx[~spike_mask], data[~spike_mask]
        )
    return data, spike_mask


def export_wavs(chunks, wav_dir):
    wav_dir.mkdir(parents=True, exist_ok=True)
    wav_files = []

    for chunk in chunks:
        fs = round(chunk.stats.sampling_rate)
        timestamp = chunk.stats.starttime.strftime("%Y-%m-%dT%H-%M")
        filename = f"broadband_{timestamp}.wav"
        filepath = wav_dir / filename

        if filepath.exists():
            wav_files.append(filepath)
            print(f"Cached   {filename}")
            continue

        data = np.array(chunk.data, dtype=np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # OOI transmits in 1-second packets. Each packet has its own DC baseline;
        # the step between adjacent packets appears as a broadband spectral spike
        # (leakage in any FFT window that straddles the seam). MAD cannot catch
        # this because no individual sample is an outlier — the whole packet is
        # shifted. Fix: subtract each packet's mean before concatenating.
        packet_size = fs  # exactly 1 second at 64 kHz
        data = data.astype(np.float64)
        for i in range(len(data) // packet_size):
            s, e = i * packet_size, (i + 1) * packet_size
            data[s:e] -= data[s:e].mean()
        data = data.astype(np.float32)

        # Then remove any remaining narrow amplitude transients.
        data, spike_mask = _despike_rolling(data, fs)

        data = data - data.mean()
        peak = np.abs(data).max()
        if peak > 0:
            data = data / peak

        wavfile.write(filepath, fs, data)
        wav_files.append(filepath)
        print(f"Wrote    {filename} ({fs} Hz, {len(data) / fs:.0f}s)")

    print(f"\n{len(wav_files)} WAV files in {wav_dir}")
    return wav_files


# ---------------------------------------------------------------------------
# Perch embeddings
# ---------------------------------------------------------------------------


def embed_with_perch(wav_files, embedding_dir, model_name, duty_cycle_minutes=0):
    embedding_dir.mkdir(parents=True, exist_ok=True)
    dc_tag = f"_dc{duty_cycle_minutes}m" if duty_cycle_minutes > 0 else ""
    emb_file = embedding_dir / f"embeddings{dc_tag}.npz"

    if emb_file.exists():
        print("Loading cached embeddings...")
        cached = np.load(emb_file)
        embeddings = cached["embeddings"]
        offsets = cached["offsets"]
        file_indices = cached["file_indices"]
        print(f"Loaded {len(embeddings)} embeddings from cache")
        return embeddings, offsets, file_indices

    from perch_hoplite import audio_io
    from perch_hoplite.zoo.model_configs import load_model_by_name

    # Checkpoint file: updated after every WAV file so a crash mid-run doesn't
    # lose completed work. Deleted once the final embeddings.npz is written.
    partial_file = embedding_dir / f"embeddings{dc_tag}_partial.npz"

    all_embeddings, all_offsets, all_file_indices = [], [], []
    n_files_done = 0

    if partial_file.exists():
        print("Resuming from checkpoint...")
        ckpt = np.load(partial_file)
        all_embeddings = list(ckpt["embeddings"])
        all_offsets = list(ckpt["offsets"])
        all_file_indices = list(ckpt["file_indices"])
        n_files_done = int(ckpt["n_files_done"])
        print(
            f"  {len(all_embeddings)} embeddings, {n_files_done}/{len(wav_files)} files done"
        )

    print(f"Loading {model_name} model...")
    model = load_model_by_name(model_name)
    print(
        f"Model loaded. Sample rate: {model.sample_rate} Hz, "
        f"window: {model.window_size_s}s, hop: {model.hop_size_s}s"
    )

    for file_idx, wav_path in enumerate(wav_files):
        if file_idx < n_files_done:
            print(f"  [skip] {wav_path.name}  (checkpoint)")
            continue

        t0_file = time.time()
        print(f"\n[{file_idx + 1}/{len(wav_files)}] Embedding {wav_path.name}...")
        audio = audio_io.load_audio_window(
            filepath=str(wav_path),
            offset_s=0,
            sample_rate=model.sample_rate,
            window_size_s=-1,
        )
        audio = np.array(audio)
        if duty_cycle_minutes > 0:
            max_samples = int(duty_cycle_minutes * 60 * model.sample_rate)
            audio = audio[:max_samples]
        print(f"  {len(audio):,} samples ({len(audio) / model.sample_rate:.0f}s)")

        outputs = model.embed(audio)
        embs = outputs.embeddings

        if embs is not None and len(embs) > 0:
            if embs.ndim == 3:
                embs = embs[:, 0, :]
            for w in range(len(embs)):
                all_embeddings.append(embs[w])
                all_offsets.append(w * model.hop_size_s)
                all_file_indices.append(file_idx)
            print(f"  {len(embs)} windows  [{_fmt(time.time() - t0_file)}]")

        # Save checkpoint after every file
        np.savez(
            partial_file,
            embeddings=np.array(all_embeddings),
            offsets=np.array(all_offsets),
            file_indices=np.array(all_file_indices),
            n_files_done=np.array(file_idx + 1),
        )

    embeddings = np.array(all_embeddings)
    offsets = np.array(all_offsets)
    file_indices = np.array(all_file_indices)

    np.savez(
        emb_file, embeddings=embeddings, offsets=offsets, file_indices=file_indices
    )
    partial_file.unlink(missing_ok=True)
    print(f"\nSaved {len(embeddings)} embeddings → {emb_file}")
    return embeddings, offsets, file_indices


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------


def run_umap(embeddings, embedding_dir=None):
    from umap import UMAP

    umap_file = embedding_dir / "umap_coords.npz" if embedding_dir else None

    if umap_file and umap_file.exists():
        print("Loading cached UMAP coords...")
        cached = np.load(umap_file)
        coords = cached["coords"]
        print(f"Loaded UMAP coords: {coords.shape}")
        return coords

    print("Running UMAP...")
    reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    coords = reducer.fit_transform(embeddings)
    print(f"UMAP done: {coords.shape}")

    if umap_file:
        np.savez(umap_file, coords=coords)
        print(f"Saved UMAP coords → {umap_file}")

    return coords


# ---------------------------------------------------------------------------
# HDBSCAN
# ---------------------------------------------------------------------------


def run_hdbscan(umap_coords, min_cluster_size):
    from sklearn.cluster import HDBSCAN

    print(f"Running HDBSCAN (min_cluster_size={min_cluster_size})...")
    hdb = HDBSCAN(min_cluster_size=min_cluster_size)
    labels = hdb.fit_predict(umap_coords)
    # sklearn's HDBSCAN exposes probabilities_ (cluster membership strength)
    # rather than GLOSH outlier scores. 1 - probability gives a useful proxy:
    # noise points (prob=0) get score 1, confident cluster members get score ~0.
    scores = 1.0 - hdb.probabilities_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(
        f"Found {n_clusters} cluster(s), {n_noise} noise points ({100 * n_noise / len(labels):.1f}%)"
    )
    for c in sorted(set(labels)):
        tag = f"Cluster {c}" if c >= 0 else "Noise   "
        print(f"  {tag}: {(labels == c).sum()} windows")

    return labels, scores


# ---------------------------------------------------------------------------
# Spectrogram helper
# ---------------------------------------------------------------------------


def _spec_on_ax(ax, wav_path, offset_s, duration_s=5.0, freq_lo=60, freq_hi=16000):
    """Plot a WAV snippet on ax. Frequency range clipped to Perch's window.

    The display window is centered on the 5-second Perch clip at offset_s.
    For long durations (>30s) the FFT window auto-scales to keep ~500 time
    columns, trading frequency resolution for a readable time axis.

    Uses soundfile for seeking so only the needed slice is read into memory —
    avoids loading the full 1-hour WAV (~922 MB at 64 kHz).
    """
    import soundfile as sf

    with sf.SoundFile(wav_path) as f:
        fs = f.samplerate
        file_duration_s = f.frames / fs

        # Center on the middle of the 5-second Perch window
        center_s = offset_s + 2.5
        disp_start = max(0.0, center_s - duration_s / 2)
        disp_end = min(file_duration_s, disp_start + duration_s)
        disp_start = max(0.0, disp_end - duration_s)  # re-adjust if clipped at end

        start_frame = int(disp_start * fs)
        n_frames = int((disp_end - disp_start) * fs)
        f.seek(start_frame)
        snippet = f.read(n_frames, dtype="float64", always_2d=False)

    if len(snippet) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    # Per-packet DC removal: each OOI 1-second packet can have a different
    # baseline; subtract each packet's mean before the FFT.
    packet_size = fs
    for i in range(len(snippet) // packet_size):
        s, e = i * packet_size, (i + 1) * packet_size
        snippet[s:e] -= snippet[s:e].mean()

    # Auto-scale FFT window: 50ms for short clips, larger for long ones so
    # we get ~500 time columns regardless of total duration.
    nperseg = max(int(0.05 * fs), int(len(snippet) / 500))
    f, t, Sxx = signal.spectrogram(
        snippet, fs=fs, nperseg=nperseg, noverlap=nperseg // 2
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # Spectrogram-domain broadband spike removal.
    # Any transient that fires energy across ALL frequencies simultaneously
    # (ADCP pings, echosounders, packet seam leakage, clipping) appears as a
    # column with anomalously high mean energy. Detect via MAD on per-column
    # mean, then interpolate each frequency row over the flagged columns.
    col_mean = Sxx_db.mean(axis=0)
    col_med = np.median(col_mean)
    col_mad = np.median(np.abs(col_mean - col_med))
    spike_cols = col_mean > col_med + 5 * col_mad
    if spike_cols.any() and (~spike_cols).any():
        idx = np.arange(len(t))
        good = idx[~spike_cols]
        for row in range(Sxx_db.shape[0]):
            Sxx_db[row, spike_cols] = np.interp(
                idx[spike_cols], good, Sxx_db[row, ~spike_cols]
            )

    mask = (f >= freq_lo) & (f <= freq_hi)
    vmin, vmax = np.percentile(Sxx_db[mask], [2, 98])
    # Shift time axis so it shows minutes from start of display window
    t_min = (t + disp_start) / 60
    ax.pcolormesh(
        t_min,
        f[mask],
        Sxx_db[mask],
        shading="gouraud",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_ylim(freq_lo, freq_hi)


def _cluster_colors(cluster_ids):
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(cluster_ids), 1)))
    return {c: colors[i] for i, c in enumerate(cluster_ids)}


# ---------------------------------------------------------------------------
# Figure 1: UMAP colored by time
# ---------------------------------------------------------------------------


def fig_umap_by_time(umap_coords, file_indices, offsets, chunk_hours, figures_dir):
    hours = file_indices * chunk_hours + offsets / 3600

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        umap_coords[:, 0], umap_coords[:, 1], c=hours, cmap="twilight", s=3, alpha=0.5
    )
    plt.colorbar(sc, ax=ax, label="Hours into recording")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(
        "Perch 2.0 embeddings — UMAP\n(each point = 5-second window, colored by time)"
    )

    path = figures_dir / "01_umap_by_time.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 2: UMAP colored by HDBSCAN label + outlier score side by side
# ---------------------------------------------------------------------------


def fig_umap_clusters_and_outliers(
    umap_coords, cluster_labels, outlier_scores, figures_dir
):
    cluster_ids = sorted(c for c in set(cluster_labels) if c >= 0)
    cmap = _cluster_colors(cluster_ids)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), squeeze=False)
    ax_left, ax_right = axes[0]

    # Left: HDBSCAN labels (noise in faint gray)
    for label in sorted(set(cluster_labels)):
        mask = cluster_labels == label
        if label >= 0:
            color = cmap[label]
            tag = f"Cluster {label} (n={mask.sum()})"
            alpha = 0.7
        else:
            color = (0.75, 0.75, 0.75)
            tag = f"Noise (n={mask.sum()})"
            alpha = 0.3
        ax_left.scatter(
            umap_coords[mask, 0],
            umap_coords[mask, 1],
            color=color,
            s=3,
            alpha=alpha,
            label=tag,
        )

    ax_left.legend(markerscale=5, fontsize=8)
    ax_left.set_xlabel("UMAP 1")
    ax_left.set_ylabel("UMAP 2")
    ax_left.set_title("HDBSCAN clusters\n(gray = noise / unassigned)")

    # Right: outlier score (0 = typical, 1 = most anomalous)
    sc = ax_right.scatter(
        umap_coords[:, 0],
        umap_coords[:, 1],
        c=outlier_scores,
        cmap="hot",
        s=3,
        alpha=0.6,
        vmin=0,
        vmax=1,
    )
    plt.colorbar(sc, ax=ax_right, label="Outlier score (GLOSH)")
    ax_right.set_xlabel("UMAP 1")
    ax_right.set_ylabel("UMAP 2")
    ax_right.set_title("Outlier score\n(bright = most anomalous)")

    plt.tight_layout()
    path = figures_dir / "02_umap_clusters_outliers.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 3: Outlier score timeline
# ---------------------------------------------------------------------------


def fig_outlier_timeline(
    file_indices,
    offsets,
    chunk_hours,
    outlier_scores,
    cluster_labels,
    start_dt,
    figures_dir,
):
    hours = file_indices * chunk_hours + offsets / 3600
    times = np.array([start_dt + timedelta(hours=float(h)) for h in hours])

    cluster_ids = sorted(c for c in set(cluster_labels) if c >= 0)
    cmap = _cluster_colors(cluster_ids)

    fig, ax = plt.subplots(figsize=(16, 4))

    # Noise points in light gray behind everything
    noise_mask = cluster_labels == -1
    if noise_mask.any():
        ax.scatter(
            times[noise_mask],
            outlier_scores[noise_mask],
            color=(0.75, 0.75, 0.75),
            s=4,
            alpha=0.4,
            label="Noise",
            zorder=1,
        )

    # Cluster points on top in color
    for c in cluster_ids:
        mask = cluster_labels == c
        ax.scatter(
            times[mask],
            outlier_scores[mask],
            color=cmap[c],
            s=6,
            alpha=0.8,
            label=f"Cluster {c}",
            zorder=2,
        )

    ax.set_xlabel("UTC time")
    ax.set_ylabel("Outlier score (GLOSH)")
    ax.set_title(
        "Outlier score over time\n"
        "score → 1 means this 5-second window sounds unlike anything else in the recording"
    )
    ax.legend(markerscale=3, fontsize=8, loc="upper left")
    ax.set_ylim(-0.05, 1.05)
    fig.autofmt_xdate()
    plt.tight_layout()

    path = figures_dir / "03_outlier_timeline.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 4: Top-N most anomalous spectrograms
# ---------------------------------------------------------------------------


def fig_top_anomalies(
    outlier_scores,
    file_indices,
    offsets,
    wav_files,
    start_dt,
    chunk_hours,
    n_top,
    figures_dir,
    duration_s=5.0,
):
    n_top = min(n_top, len(outlier_scores))
    if n_top == 0:
        print("n_top_anomalies = 0 — skipping anomaly figure.")
        return
    top_idx = np.argsort(outlier_scores)[::-1][:n_top]

    ncols = min(n_top, 4)
    nrows = math.ceil(n_top / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 4, nrows * 3 + 1), squeeze=False
    )
    axes_flat = axes.flatten()

    for i, idx in enumerate(top_idx):
        file_idx = int(file_indices[idx])
        offset_s = float(offsets[idx])
        score = outlier_scores[idx]
        abs_time = start_dt + timedelta(hours=file_idx * chunk_hours + offset_s / 3600)

        _spec_on_ax(axes_flat[i], wav_files[file_idx], offset_s, duration_s=duration_s)
        axes_flat[i].set_title(
            f"#{i + 1}  score={score:.3f}\n{abs_time.strftime('%H:%M:%S UTC')}",
            fontsize=8,
        )
        if i % ncols == 0:
            axes_flat[i].set_ylabel("Freq (Hz)")
        if i >= ncols * (nrows - 1):
            axes_flat[i].set_xlabel("Time (min)")

    for j in range(n_top, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle(f"Top {n_top} most anomalous 5-second windows", fontsize=13)
    plt.tight_layout()
    path = figures_dir / "04_top_anomalies.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 5: One representative spectrogram per HDBSCAN cluster
# ---------------------------------------------------------------------------


def fig_cluster_examples(
    cluster_labels,
    umap_coords,
    file_indices,
    offsets,
    wav_files,
    start_dt,
    chunk_hours,
    figures_dir,
    duration_s=5.0,
):
    cluster_ids = sorted(c for c in set(cluster_labels) if c >= 0)
    if not cluster_ids:
        print("No HDBSCAN clusters found — skipping cluster examples figure.")
        return

    ncols = min(len(cluster_ids), 4)
    nrows = math.ceil(len(cluster_ids) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 4, nrows * 3 + 1), squeeze=False
    )
    axes_flat = axes.flatten()

    for i, c in enumerate(cluster_ids):
        mask = cluster_labels == c
        cluster_pts = umap_coords[mask]
        centroid = cluster_pts.mean(axis=0)
        dists = np.linalg.norm(cluster_pts - centroid, axis=1)
        best = np.where(mask)[0][dists.argmin()]

        file_idx = int(file_indices[best])
        offset_s = float(offsets[best])
        abs_time = start_dt + timedelta(hours=file_idx * chunk_hours + offset_s / 3600)

        _spec_on_ax(axes_flat[i], wav_files[file_idx], offset_s, duration_s=duration_s)
        axes_flat[i].set_title(
            f"Cluster {c}  (n={mask.sum()})\n{abs_time.strftime('%H:%M:%S UTC')}",
            fontsize=8,
        )
        if i % ncols == 0:
            axes_flat[i].set_ylabel("Freq (Hz)")
        if i >= ncols * (nrows - 1):
            axes_flat[i].set_xlabel("Time (min)")

    for j in range(len(cluster_ids), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Representative spectrogram per HDBSCAN cluster", fontsize=13)
    plt.tight_layout()
    path = figures_dir / "05_cluster_examples.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 6: random time sample
# ---------------------------------------------------------------------------


def fig_random_sample(
    file_indices,
    offsets,
    wav_files,
    start_dt,
    chunk_hours,
    figures_dir,
    n=16,
    seed=42,
    duration_s=5.0,
):
    """Grid of windows sampled uniformly across the recording — unbiased view."""
    rng = np.random.default_rng(seed)
    n_windows = len(file_indices)
    chosen = np.sort(rng.choice(n_windows, size=min(n, n_windows), replace=False))

    ncols = 4
    nrows = math.ceil(len(chosen) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes_flat = axes.flatten()

    for ax_i, win_i in enumerate(chosen):
        file_idx = int(file_indices[win_i])
        offset_s = float(offsets[win_i])
        abs_time = start_dt + timedelta(hours=file_idx * chunk_hours + offset_s / 3600)

        _spec_on_ax(
            axes_flat[ax_i], wav_files[file_idx], offset_s, duration_s=duration_s
        )
        axes_flat[ax_i].set_title(abs_time.strftime("%H:%M UTC"), fontsize=8)
        if ax_i % ncols == 0:
            axes_flat[ax_i].set_ylabel("Freq (Hz)")
        if ax_i >= ncols * (nrows - 1):
            axes_flat[ax_i].set_xlabel("Time (min)")

    for j in range(len(chosen), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle(
        "Random sample of 5-second windows (uniform across recording)", fontsize=13
    )
    plt.tight_layout()
    path = figures_dir / "06_random_sample.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _fmt(seconds):
    """Format elapsed seconds as 'X min Y s' or just 'X.Xs'."""
    if seconds >= 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f}s"


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Soundscape exploration with Perch + HDBSCAN"
    )
    parser.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        help="Process a single calendar day, overriding start/end in soundscape.toml",
    )
    parser.add_argument(
        "--duty-cycle",
        type=int,
        metavar="MINUTES",
        dest="duty_cycle",
        help="Minutes of each hour to embed (0 = full hour). Overrides soundscape.toml.",
    )
    parser.add_argument(
        "--figures",
        metavar="LIST",
        help=(
            "Comma-separated subset of figures to generate. "
            "Choices: umap,clusters,timeline,anomalies,cluster_examples,random. "
            "Default: all. Example: --figures cluster_examples,anomalies"
        ),
    )
    args = parser.parse_args()
    wanted = set(args.figures.split(",")) if args.figures else None

    t_wall = time.time()
    cfg, dirs, start_dt = load_config(
        date_override=args.date,
        duty_cycle_override=args.duty_cycle,
    )

    print("=== Loading broadband data ===")
    t0 = time.time()
    chunks = load_broadband_chunks(cfg, dirs)
    print(f"  [{_fmt(time.time() - t0)}]")

    print("\n=== Exporting WAV files ===")
    t0 = time.time()
    wav_files = export_wavs(chunks, dirs["wav"])
    print(f"  [{_fmt(time.time() - t0)}]")

    print("\n=== Perch embeddings ===")
    t0 = time.time()
    embeddings, offsets, file_indices = embed_with_perch(
        wav_files,
        dirs["embeddings"],
        cfg["perch_model"],
        duty_cycle_minutes=cfg.get("duty_cycle_minutes", 0),
    )
    print(f"  [{_fmt(time.time() - t0)}]")

    # Release TensorFlow GPU memory before figure generation.
    # The spectrogram figures read full WAV files and can OOM if the model
    # is still resident on the GPU.
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
        print("TensorFlow session cleared (GPU memory released).")
    except Exception:
        pass

    print("\n=== UMAP ===")
    t0 = time.time()
    umap_coords = run_umap(embeddings, embedding_dir=dirs["embeddings"])
    print(f"  [{_fmt(time.time() - t0)}]")

    print("\n=== HDBSCAN ===")
    t0 = time.time()
    cluster_labels, outlier_scores = run_hdbscan(umap_coords, cfg["min_cluster_size"])
    print(f"  [{_fmt(time.time() - t0)}]")

    print("\n=== Saving figures ===")
    t0 = time.time()
    chunk_hours = cfg["chunk_hours"]
    figs = dirs["figures"]
    spec_dur = cfg.get("spectrogram_duration_s", 5.0)

    def want(name):
        return wanted is None or name in wanted

    if want("umap"):
        fig_umap_by_time(umap_coords, file_indices, offsets, chunk_hours, figs)
    if want("clusters"):
        fig_umap_clusters_and_outliers(
            umap_coords, cluster_labels, outlier_scores, figs
        )
    if want("timeline"):
        fig_outlier_timeline(
            file_indices,
            offsets,
            chunk_hours,
            outlier_scores,
            cluster_labels,
            start_dt,
            figs,
        )
    if want("anomalies"):
        fig_top_anomalies(
            outlier_scores,
            file_indices,
            offsets,
            wav_files,
            start_dt,
            chunk_hours,
            cfg["n_top_anomalies"],
            figs,
            duration_s=spec_dur,
        )
    if want("cluster_examples"):
        fig_cluster_examples(
            cluster_labels,
            umap_coords,
            file_indices,
            offsets,
            wav_files,
            start_dt,
            chunk_hours,
            figs,
            duration_s=spec_dur,
        )
    if want("random"):
        fig_random_sample(
            file_indices,
            offsets,
            wav_files,
            start_dt,
            chunk_hours,
            figs,
            duration_s=spec_dur,
        )
    print(f"  [{_fmt(time.time() - t0)}]")

    print(f"\nAll done in {_fmt(time.time() - t_wall)}. Figures in {figs}")


if __name__ == "__main__":
    main()
