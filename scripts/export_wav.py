#!/usr/bin/env python
"""
Export cached OOI hydrophone data to WAV files for Perch 2.0.

Reads per-day pickle files from data/cache/ and writes 1-hour WAV files
to data/wav/. These WAV files are the input format Perch expects.

Usage
-----
    uv run python scripts/export_wav.py [C: can't we just use uv run scripts/export_wav.py?]

Output
------
    data/wav/
        Axial_Base_2026-01-01T00.wav
        Axial_Base_2026-01-01T01.wav
        ...
        Axial_Base_2026-01-07T23.wav
"""

import pickle
from pathlib import Path

import numpy as np
from scipy.io import wavfile

# --- Configuration ---

CACHE_DIR = Path("data/cache")
OUTPUT_DIR = Path("data/wav")
CHUNK_DURATION_S = 3600  # 1 hour


def load_day(pkl_path: Path):
    """Load a per-day HydrophoneData pickle file.

    Returns (data, sampling_rate, starttime) or None if loading fails.
    """
    with open(pkl_path, "rb") as f:
        hdata = pickle.load(f)

    # Expect an ooipy HydrophoneData object with .data, .statsf
    if not hasattr(hdata, "stats"):
        print(f"  Skipping {pkl_path.name} — not a HydrophoneData object")
        return None

    data = np.array(hdata.data, dtype=np.float32)
    fs = round(hdata.stats.sampling_rate)
    starttime = hdata.stats.starttime

    return data, fs, starttime


def export_chunks(data, fs, starttime, node_name: str, output_dir: Path):
    """Split data into 1-hour chunks and write as WAV files."""
    samples_per_chunk = fs * CHUNK_DURATION_S
    n_chunks = len(data) // samples_per_chunk
    remainder = len(data) % samples_per_chunk

    written = 0
    for i in range(n_chunks):
        chunk = data[i * samples_per_chunk : (i + 1) * samples_per_chunk]

        # Build timestamp for this chunk
        chunk_time = starttime + (i * CHUNK_DURATION_S)
        timestamp = chunk_time.strftime("%Y-%m-%dT%H")
        filename = f"{node_name}_{timestamp}.wav"
        filepath = output_dir / filename

        # Remove DC offset and normalize to [-1, 1] float32 for WAV
        chunk = chunk - chunk.mean()
        peak = np.abs(chunk).max()
        if peak > 0:
            chunk = chunk / peak

        wavfile.write(filepath, fs, chunk.astype(np.float32))
        written += 1

    # Write remainder if it's substantial (> 1 minute)
    if remainder > fs * 60:
        chunk = data[n_chunks * samples_per_chunk :]
        chunk_time = starttime + (n_chunks * CHUNK_DURATION_S)
        timestamp = chunk_time.strftime("%Y-%m-%dT%H-%M")
        filename = f"{node_name}_{timestamp}.wav"
        filepath = output_dir / filename

        chunk = chunk - chunk.mean()
        peak = np.abs(chunk).max()
        if peak > 0:
            chunk = chunk / peak

        wavfile.write(filepath, fs, chunk.astype(np.float32))
        written += 1

    return written


def find_day_files(cache_dir: Path) -> list[Path]:
    """Find per-day cache files (exclude combined files)."""
    all_pkls = sorted(cache_dir.glob("*.pkl"))
    # Per-day files have the pattern: Node_YYYY-MM-DD_HH-MM-SS_YYYY-MM-DD_HH-MM-SS.pkl
    # Combined files have: Node_YYYY-MM-DD_to_YYYY-MM-DD.pkl
    return [p for p in all_pkls if "_to_" not in p.name]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    day_files = find_day_files(CACHE_DIR)
    if not day_files:
        print(f"No per-day cache files found in {CACHE_DIR}")
        print("Run 'uv run python scripts/fetch_week_data.py' first.")
        return

    print(f"Found {len(day_files)} day files in {CACHE_DIR}")
    print(f"Output: {OUTPUT_DIR}/")
    print()

    total_written = 0
    for pkl_path in day_files:
        print(f"Processing {pkl_path.name}...")
        result = load_day(pkl_path)
        if result is None:
            continue

        data, fs, starttime = result

        # Extract node name from filename (e.g., "Axial_Base")
        # Filename: Axial_Base_2026-01-01_00-00-00_2026-01-02_00-00-00.pkl
        parts = pkl_path.stem.split("_")
        # Find where the date starts (YYYY-MM-DD pattern)
        node_parts = []
        for part in parts:
            if len(part) == 10 and part[4] == "-":
                break
            node_parts.append(part)
        node_name = "_".join(node_parts)

        n = export_chunks(data, fs, starttime, node_name, OUTPUT_DIR)
        total_written += n
        print(f"  {n} WAV files written ({fs} Hz, {CHUNK_DURATION_S}s chunks)")

    print()
    print(f"Done. {total_written} WAV files in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
