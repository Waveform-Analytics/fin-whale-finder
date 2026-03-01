#!/usr/bin/env python
"""
Soundscape exploration with Perch 2.0 embeddings + HDBSCAN anomaly detection.

Run after fetching data with scripts/fetch_broadband.py.

Usage
-----
    uv run python scripts/soundscape_explore.py

Config lives in configs/soundscape.toml. Output goes to
results/soundscape_figures/<slug>/ where slug is derived from the
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
import tomllib
from datetime import datetime, timedelta
from pathlib import Path

# Must be set before TensorFlow / Perch is imported.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_cuda_data_dir=/opt/conda")

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

def load_config():
    config_path = PROJECT_ROOT / "configs" / "soundscape.toml"
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    start_dt = datetime.fromisoformat(cfg["start"])
    end_dt = datetime.fromisoformat(cfg["end"])
    slug = (
        f"{start_dt.strftime('%Y-%m-%d')}"
        f"_{start_dt.strftime('%Hh')}-{end_dt.strftime('%Hh')}"
    )
    dirs = {
        "cache":      PROJECT_ROOT / "data" / "cache_broadband",
        "wav":        PROJECT_ROOT / "data" / "wav_broadband" / slug,
        "embeddings": PROJECT_ROOT / "data" / "embeddings" / slug,
        "figures":    PROJECT_ROOT / "results" / "soundscape_figures" / slug,
    }
    dirs["figures"].mkdir(parents=True, exist_ok=True)

    print(f"Config:  {config_path}")
    print(f"Node:    {cfg['node']}")
    print(f"Range:   {cfg['start']} → {cfg['end']}")
    print(f"Run:     {slug}")
    print(f"Figures: {dirs['figures']}")
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
        str(f) for f, _, _ in expected
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

        # OOI transmits 64 kHz data in 1-second packets; seams between packets
        # create broadband impulse artifacts. Detect via MAD, interpolate over.
        med = np.median(data)
        mad = np.median(np.abs(data - med))
        spike_mask = np.abs(data - med) > 8 * mad
        if spike_mask.any():
            idx = np.arange(len(data))
            data[spike_mask] = np.interp(
                idx[spike_mask], idx[~spike_mask], data[~spike_mask]
            )
            print(f"  Despiked {spike_mask.sum()} samples ({100 * spike_mask.mean():.2f}%)")

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

def embed_with_perch(wav_files, embedding_dir, model_name):
    embedding_dir.mkdir(parents=True, exist_ok=True)
    emb_file = embedding_dir / "embeddings.npz"

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

    print(f"Loading {model_name} model...")
    model = load_model_by_name(model_name)
    print(
        f"Model loaded. Sample rate: {model.sample_rate} Hz, "
        f"window: {model.window_size_s}s, hop: {model.hop_size_s}s"
    )

    all_embeddings, all_offsets, all_file_indices = [], [], []

    for file_idx, wav_path in enumerate(wav_files):
        print(f"\nEmbedding {wav_path.name}...")
        audio = audio_io.load_audio_window(
            filepath=str(wav_path),
            offset_s=0,
            sample_rate=model.sample_rate,
            window_size_s=-1,
        )
        audio = np.array(audio)
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
            print(f"  {len(embs)} windows (dim={embs.shape[-1]})")

    embeddings = np.array(all_embeddings)
    offsets = np.array(all_offsets)
    file_indices = np.array(all_file_indices)

    np.savez(emb_file, embeddings=embeddings, offsets=offsets, file_indices=file_indices)
    print(f"\nSaved {len(embeddings)} embeddings → {emb_file}")
    return embeddings, offsets, file_indices


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def run_umap(embeddings):
    from umap import UMAP

    print("Running UMAP...")
    reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    coords = reducer.fit_transform(embeddings)
    print(f"UMAP done: {coords.shape}")
    return coords


# ---------------------------------------------------------------------------
# HDBSCAN
# ---------------------------------------------------------------------------

def run_hdbscan(umap_coords, min_cluster_size):
    from sklearn.cluster import HDBSCAN

    print(f"Running HDBSCAN (min_cluster_size={min_cluster_size})...")
    hdb = HDBSCAN(min_cluster_size=min_cluster_size)
    labels = hdb.fit_predict(umap_coords)
    scores = hdb.outlier_scores_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"Found {n_clusters} cluster(s), {n_noise} noise points ({100*n_noise/len(labels):.1f}%)")
    for c in sorted(set(labels)):
        tag = f"Cluster {c}" if c >= 0 else "Noise   "
        print(f"  {tag}: {(labels == c).sum()} windows")

    return labels, scores


# ---------------------------------------------------------------------------
# Spectrogram helper
# ---------------------------------------------------------------------------

def _spec_on_ax(ax, wav_path, offset_s, duration_s=5.0, freq_lo=60, freq_hi=16000):
    """Plot a despiked WAV snippet on ax. Frequency range clipped to Perch's window."""
    fs, audio = wavfile.read(wav_path)
    start = int(offset_s * fs)
    end = min(start + int(duration_s * fs), len(audio))
    snippet = audio[start:end].astype(np.float32)

    if len(snippet) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    nperseg = int(0.05 * fs)  # 50ms FFT window
    f, t, Sxx = signal.spectrogram(snippet, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    mask = (f >= freq_lo) & (f <= freq_hi)
    vmin, vmax = np.percentile(Sxx_db[mask], [2, 98])
    ax.pcolormesh(t, f[mask], Sxx_db[mask], shading="gouraud",
                  cmap="viridis", vmin=vmin, vmax=vmax)
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
    sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1],
                    c=hours, cmap="twilight", s=3, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="Hours into recording")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Perch 2.0 embeddings — UMAP\n(each point = 5-second window, colored by time)")

    path = figures_dir / "01_umap_by_time.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 2: UMAP colored by HDBSCAN label + outlier score side by side
# ---------------------------------------------------------------------------

def fig_umap_clusters_and_outliers(umap_coords, cluster_labels, outlier_scores, figures_dir):
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
        ax_left.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                        color=color, s=3, alpha=alpha, label=tag)

    ax_left.legend(markerscale=5, fontsize=8)
    ax_left.set_xlabel("UMAP 1")
    ax_left.set_ylabel("UMAP 2")
    ax_left.set_title("HDBSCAN clusters\n(gray = noise / unassigned)")

    # Right: outlier score (0 = typical, 1 = most anomalous)
    sc = ax_right.scatter(umap_coords[:, 0], umap_coords[:, 1],
                          c=outlier_scores, cmap="hot", s=3, alpha=0.6,
                          vmin=0, vmax=1)
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

def fig_outlier_timeline(file_indices, offsets, chunk_hours, outlier_scores,
                         cluster_labels, start_dt, figures_dir):
    hours = file_indices * chunk_hours + offsets / 3600
    times = np.array([start_dt + timedelta(hours=float(h)) for h in hours])

    cluster_ids = sorted(c for c in set(cluster_labels) if c >= 0)
    cmap = _cluster_colors(cluster_ids)

    fig, ax = plt.subplots(figsize=(16, 4))

    # Noise points in light gray behind everything
    noise_mask = cluster_labels == -1
    if noise_mask.any():
        ax.scatter(times[noise_mask], outlier_scores[noise_mask],
                   color=(0.75, 0.75, 0.75), s=4, alpha=0.4, label="Noise", zorder=1)

    # Cluster points on top in color
    for c in cluster_ids:
        mask = cluster_labels == c
        ax.scatter(times[mask], outlier_scores[mask],
                   color=cmap[c], s=6, alpha=0.8, label=f"Cluster {c}", zorder=2)

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

def fig_top_anomalies(outlier_scores, file_indices, offsets, wav_files,
                      start_dt, chunk_hours, n_top, figures_dir):
    n_top = min(n_top, len(outlier_scores))
    top_idx = np.argsort(outlier_scores)[::-1][:n_top]

    ncols = min(n_top, 4)
    nrows = math.ceil(n_top / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3 + 1), squeeze=False)
    axes_flat = axes.flatten()

    for i, idx in enumerate(top_idx):
        file_idx = int(file_indices[idx])
        offset_s = float(offsets[idx])
        score = outlier_scores[idx]
        abs_time = start_dt + timedelta(hours=file_idx * chunk_hours + offset_s / 3600)

        _spec_on_ax(axes_flat[i], wav_files[file_idx], offset_s)
        axes_flat[i].set_title(
            f"#{i + 1}  score={score:.3f}\n{abs_time.strftime('%H:%M:%S UTC')}",
            fontsize=8,
        )
        if i % ncols == 0:
            axes_flat[i].set_ylabel("Freq (Hz)")
        if i >= ncols * (nrows - 1):
            axes_flat[i].set_xlabel("Time (s)")

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

def fig_cluster_examples(cluster_labels, umap_coords, file_indices, offsets,
                         wav_files, start_dt, chunk_hours, figures_dir):
    cluster_ids = sorted(c for c in set(cluster_labels) if c >= 0)
    if not cluster_ids:
        print("No HDBSCAN clusters found — skipping cluster examples figure.")
        return

    ncols = min(len(cluster_ids), 4)
    nrows = math.ceil(len(cluster_ids) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3 + 1), squeeze=False)
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

        _spec_on_ax(axes_flat[i], wav_files[file_idx], offset_s)
        axes_flat[i].set_title(
            f"Cluster {c}  (n={mask.sum()})\n{abs_time.strftime('%H:%M:%S UTC')}",
            fontsize=8,
        )
        if i % ncols == 0:
            axes_flat[i].set_ylabel("Freq (Hz)")
        if i >= ncols * (nrows - 1):
            axes_flat[i].set_xlabel("Time (s)")

    for j in range(len(cluster_ids), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Representative spectrogram per HDBSCAN cluster", fontsize=13)
    plt.tight_layout()
    path = figures_dir / "05_cluster_examples.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg, dirs, start_dt = load_config()

    print("=== Loading broadband data ===")
    chunks = load_broadband_chunks(cfg, dirs)

    print("\n=== Exporting WAV files ===")
    wav_files = export_wavs(chunks, dirs["wav"])

    print("\n=== Perch embeddings ===")
    embeddings, offsets, file_indices = embed_with_perch(
        wav_files, dirs["embeddings"], cfg["perch_model"]
    )

    print("\n=== UMAP ===")
    umap_coords = run_umap(embeddings)

    print("\n=== HDBSCAN ===")
    cluster_labels, outlier_scores = run_hdbscan(umap_coords, cfg["min_cluster_size"])

    print("\n=== Saving figures ===")
    chunk_hours = cfg["chunk_hours"]
    figs = dirs["figures"]

    fig_umap_by_time(umap_coords, file_indices, offsets, chunk_hours, figs)
    fig_umap_clusters_and_outliers(umap_coords, cluster_labels, outlier_scores, figs)
    fig_outlier_timeline(file_indices, offsets, chunk_hours, outlier_scores,
                         cluster_labels, start_dt, figs)
    fig_top_anomalies(outlier_scores, file_indices, offsets, wav_files,
                      start_dt, chunk_hours, cfg["n_top_anomalies"], figs)
    fig_cluster_examples(cluster_labels, umap_coords, file_indices, offsets,
                         wav_files, start_dt, chunk_hours, figs)

    print(f"\nAll done. Figures in {figs}")


if __name__ == "__main__":
    main()
