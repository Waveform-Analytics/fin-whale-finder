#!/usr/bin/env python
"""
Plot spectrograms for windows in a specific UMAP region.

Loads saved embeddings, re-runs UMAP (or loads cached coords), then plots
spectrograms for all windows whose UMAP position falls inside the given bounds.

Usage
-----
    # Plot lower-left outliers from June 10
    uv run python scripts/explore_region.py --date 2025-06-10 --x-max -4

    # Custom bounding box
    uv run python scripts/explore_region.py --date 2025-06-10 --x-min -10 --x-max -4 --y-min 3 --y-max 7

    # Limit to 12 spectrograms (default 16)
    uv run python scripts/explore_region.py --date 2025-06-10 --x-max -4 --n 12
"""

import argparse
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_cuda_data_dir=/opt/conda")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_june10_paths(date_str):
    """Return (embedding_dir, wav_dir, figures_dir) for a given date."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    slug = f"{dt.strftime('%Y-%m-%d')}_00h-00h"
    embedding_dir = PROJECT_ROOT / "data" / "embeddings" / slug
    wav_dir       = PROJECT_ROOT / "data" / "wav_broadband" / slug
    figures_dir   = PROJECT_ROOT / "results" / "soundscape_figures" / slug
    return embedding_dir, wav_dir, figures_dir


def load_embeddings(embedding_dir):
    emb_file = embedding_dir / "embeddings.npz"
    if not emb_file.exists():
        raise FileNotFoundError(f"No embeddings found at {emb_file}")
    d = np.load(emb_file)
    return d["embeddings"], d["offsets"], d["file_indices"]


def get_umap_coords(embeddings, embedding_dir):
    umap_file = embedding_dir / "umap_coords.npz"
    if umap_file.exists():
        print("Loading cached UMAP coords...")
        return np.load(umap_file)["coords"]

    print("Running UMAP (no cached coords found)...")
    from umap import UMAP
    coords = UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(embeddings)
    np.savez(umap_file, coords=coords)
    print(f"Saved UMAP coords → {umap_file}")
    return coords


def _spec_on_ax(ax, wav_path, offset_s, duration_s=5.0, freq_lo=60, freq_hi=16000):
    fs, audio = wavfile.read(wav_path)
    file_duration_s = len(audio) / fs

    center_s   = offset_s + 2.5
    disp_start = max(0.0, center_s - duration_s / 2)
    disp_end   = min(file_duration_s, disp_start + duration_s)
    disp_start = max(0.0, disp_end - duration_s)

    start = int(disp_start * fs)
    end   = int(disp_end   * fs)
    snippet = audio[start:end].astype(np.float64)

    if len(snippet) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    packet_size = fs
    for i in range(len(snippet) // packet_size):
        s, e = i * packet_size, (i + 1) * packet_size
        snippet[s:e] -= snippet[s:e].mean()

    nperseg = max(int(0.05 * fs), int(len(snippet) / 500))
    f, t, Sxx = signal.spectrogram(snippet, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    col_mean = Sxx_db.mean(axis=0)
    col_med  = np.median(col_mean)
    col_mad  = np.median(np.abs(col_mean - col_med))
    spike_cols = col_mean > col_med + 5 * col_mad
    if spike_cols.any() and (~spike_cols).any():
        idx = np.arange(len(t))
        good = idx[~spike_cols]
        for row in range(Sxx_db.shape[0]):
            Sxx_db[row, spike_cols] = np.interp(idx[spike_cols], good, Sxx_db[row, ~spike_cols])

    mask = (f >= freq_lo) & (f <= freq_hi)
    vmin, vmax = np.percentile(Sxx_db[mask], [2, 98])
    t_min = (t + disp_start) / 60
    ax.pcolormesh(t_min, f[mask], Sxx_db[mask], shading="gouraud",
                  cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_ylim(freq_lo, freq_hi)


def main():
    parser = argparse.ArgumentParser(description="Plot spectrograms for a UMAP region")
    parser.add_argument("--date", required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--x-min", type=float, default=-np.inf, metavar="X")
    parser.add_argument("--x-max", type=float, default=np.inf,  metavar="X")
    parser.add_argument("--y-min", type=float, default=-np.inf, metavar="Y")
    parser.add_argument("--y-max", type=float, default=np.inf,  metavar="Y")
    parser.add_argument("--n", type=int, default=16, help="Max spectrograms to plot")
    parser.add_argument("--duration", type=float, default=900.0, metavar="SECONDS",
                        help="Spectrogram display duration in seconds (default 900 = 15 min)")
    parser.add_argument("--out", metavar="PATH", help="Output PNG path (default: figures dir)")
    args = parser.parse_args()

    embedding_dir, wav_dir, figures_dir = load_june10_paths(args.date)

    print(f"Loading embeddings from {embedding_dir}...")
    embeddings, offsets, file_indices = load_embeddings(embedding_dir)
    print(f"  {len(embeddings)} windows")

    coords = get_umap_coords(embeddings, embedding_dir)

    # Select windows in the bounding box
    in_region = (
        (coords[:, 0] >= args.x_min) & (coords[:, 0] <= args.x_max) &
        (coords[:, 1] >= args.y_min) & (coords[:, 1] <= args.y_max)
    )
    region_idx = np.where(in_region)[0]
    print(f"\nRegion x=[{args.x_min}, {args.x_max}] y=[{args.y_min}, {args.y_max}]")
    print(f"  {len(region_idx)} windows in region")

    if len(region_idx) == 0:
        print("No windows found in that region — try wider bounds.")
        return

    # Sample evenly if more than n
    if len(region_idx) > args.n:
        step = len(region_idx) / args.n
        chosen = region_idx[np.round(np.arange(args.n) * step).astype(int)]
    else:
        chosen = region_idx

    # Load WAV file list
    wav_files = sorted(wav_dir.glob("broadband_*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {wav_dir}")

    dt = datetime.strptime(args.date, "%Y-%m-%d")

    ncols = min(4, len(chosen))
    nrows = math.ceil(len(chosen) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3 + 1), squeeze=False)
    axes_flat = axes.flatten()

    for ax_i, win_i in enumerate(chosen):
        file_idx = int(file_indices[win_i])
        offset_s = float(offsets[win_i])
        abs_time = dt + timedelta(hours=file_idx + offset_s / 3600)
        x, y = coords[win_i]

        _spec_on_ax(axes_flat[ax_i], wav_files[file_idx], offset_s,
                    duration_s=args.duration)
        axes_flat[ax_i].set_title(
            f"{abs_time.strftime('%H:%M:%S UTC')}\nUMAP ({x:.1f}, {y:.1f})",
            fontsize=8,
        )
        if ax_i % ncols == 0:
            axes_flat[ax_i].set_ylabel("Freq (Hz)")
        if ax_i >= ncols * (nrows - 1):
            axes_flat[ax_i].set_xlabel("Time (min)")

    for j in range(len(chosen), len(axes_flat)):
        axes_flat[j].set_visible(False)

    bounds_str = (
        f"x=[{args.x_min:.1f}, {args.x_max:.1f}]  "
        f"y=[{args.y_min:.1f}, {args.y_max:.1f}]"
        if not (np.isinf(args.x_min) and np.isinf(args.x_max) and
                np.isinf(args.y_min) and np.isinf(args.y_max))
        else "all"
    )
    plt.suptitle(
        f"UMAP region spectrograms — {args.date}\n"
        f"{len(region_idx)} windows in region  ({bounds_str})",
        fontsize=11,
    )
    plt.tight_layout()

    if args.out:
        out_path = Path(args.out)
    else:
        tag = f"region_x{args.x_max:.0f}" if not np.isinf(args.x_max) else "region"
        out_path = figures_dir / f"07_{tag}.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
