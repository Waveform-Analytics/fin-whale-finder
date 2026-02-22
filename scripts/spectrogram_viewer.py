#!/usr/bin/env python
"""
Spectrogram viewer app.

Usage
-----
    uv run streamlit run scripts/spectrogram_viewer.py

"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy import signal


@st.cache_data
def load_data():
    cache_file = Path("data/cache/Axial_Base_2026-01-01_to_2026-01-07.pkl")
    if not cache_file.exists():
        st.error("Run scripts/fetch_week_data.py first!")
        return None
    with open(cache_file, "rb") as f:
        return pickle.load(f)


def plot_spectrogram(data, times, fs, window_sec, overlap, fmin, fmax):
    nperseg = int(window_sec * fs)
    noverlap = int(nperseg * overlap)

    f, t, Sxx = signal.spectrogram(
        data,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
    )

    if fmin is not None or fmax is not None:
        mask = np.ones_like(f, dtype=bool)
        if fmin is not None:
            mask &= f >= fmin
        if fmax is not None:
            mask &= f <= fmax
        f = f[mask]
        Sxx = Sxx[mask, :]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.pcolormesh(t / 3600, f, 10 * np.log10(Sxx), shading="gouraud", cmap="viridis")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (hours from start)")
    ax.set_ylim(fmin or 0, fmax or f[-1])
    return fig


def main():
    st.title("Fin Whale Spectrogram Viewer")

    data_dict = load_data()
    if data_dict is None:
        return

    data = data_dict["data"]
    times = data_dict["times"]
    fs = data_dict.get("sampling_rate", 200)

    st.sidebar.header("Parameters")

    total_hours = (times[-1] - times[0]) / 3600
    st.sidebar.write(f"Total data: {total_hours:.1f} hours")

    window_min = st.sidebar.slider("Window size (minutes)", 5, 60, 15)
    overlap = st.sidebar.slider("Overlap", 0.5, 0.95, 0.75)
    fmin = st.sidebar.number_input("Min freq (Hz)", 0, 100, 10)
    fmax = st.sidebar.number_input("Max freq (Hz)", 0, 100, 50)

    start_hour = st.slider("Start hour", 0.0, total_hours - 0.25, 0.0, 0.25)

    start_idx = int(start_hour * 3600 * fs)
    end_idx = int((start_hour + window_min / 60) * 3600 * fs)

    window_data = data[start_idx:end_idx]
    window_times = times[start_idx:end_idx] - times[start_idx]

    st.write(f"Showing: {start_hour:.2f} - {start_hour + window_min / 60:.2f} hours")

    fig = plot_spectrogram(
        window_data,
        window_times,
        fs,
        window_sec=window_min * 60,
        overlap=overlap,
        fmin=fmin,
        fmax=fmax,
    )

    st.pyplot(fig)

    st.write(f"Window size: {len(window_data) / fs:.1f} seconds")


if __name__ == "__main__":
    main()
