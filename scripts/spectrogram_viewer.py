#!/usr/bin/env python
"""Spectrogram viewer + labeling app."""

import json
import pickle
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from scipy import signal

DATASET_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
CLIP_DURATION_SEC = 900
SAMPLING_RATE = 200
SAMPLES_PER_CLIP = CLIP_DURATION_SEC * SAMPLING_RATE
LABELS_FILE = Path("labels/labels.json")

LABEL_FFT_WINDOW_SEC = 5
LABEL_OVERLAP = 0.5
LABEL_FMIN = 10
LABEL_FMAX = 50


@st.cache_data
def load_data():
    cache_file = Path("data/cache/Axial_Base_2026-01-01_to_2026-01-07.pkl")
    if not cache_file.exists():
        st.error("Run scripts/fetch_week_data.py first!")
        return None
    with open(cache_file, "rb") as f:
        return pickle.load(f)


def load_labels() -> list[dict]:
    if LABELS_FILE.exists():
        return json.loads(LABELS_FILE.read_text())
    return []


def save_labels(labels: list[dict]):
    LABELS_FILE.parent.mkdir(exist_ok=True)
    LABELS_FILE.write_text(json.dumps(labels, indent=2))


def plot_spectrogram(data, fs, window_sec, overlap, fmin, fmax, label=None):
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
    ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading="gouraud", cmap="viridis")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylim(fmin or 0, fmax or f[-1])

    if label:
        label_colors = {"present": "green", "maybe": "orange", "not_present": "red"}
        color = label_colors.get(label, "white")
        ax.text(
            0.02, 0.95, label.upper().replace("_", " "),
            transform=ax.transAxes, fontsize=14, fontweight="bold",
            color=color, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    return fig


def get_total_clips(data) -> int:
    return len(data) // SAMPLES_PER_CLIP


def get_clip_data(data, clip_index):
    start = clip_index * SAMPLES_PER_CLIP
    end = start + SAMPLES_PER_CLIP
    return data[start:end]


def clip_utc_range(clip_index) -> tuple[str, str]:
    start = DATASET_EPOCH + timedelta(seconds=clip_index * CLIP_DURATION_SEC)
    end = start + timedelta(seconds=CLIP_DURATION_SEC)
    return start.isoformat(), end.isoformat()


def build_clip_queue(total_clips, labels, review_maybes=False):
    labeled_starts = {entry["start_utc"] for entry in labels}
    if review_maybes:
        maybe_starts = {entry["start_utc"] for entry in labels if entry["label"] == "maybe"}
        queue = [i for i in range(total_clips) if clip_utc_range(i)[0] in maybe_starts]
    else:
        queue = [i for i in range(total_clips) if clip_utc_range(i)[0] not in labeled_starts]
    random.seed(42)
    random.shuffle(queue)
    return queue


def inject_keyboard_listener():
    components.html(
        """
    <script>
    const doc = window.parent.document;
    const pw = window.parent.window;
    if (!pw.__fwfKeyListener) {
        doc.addEventListener('keydown', function(e) {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            const keyMap = {
                'ArrowLeft':  '.st-key-prev_btn button',
                'ArrowRight': '.st-key-next_btn button',
                'f':          '.st-key-label_present button',
                'd':          '.st-key-label_maybe button',
                's':          '.st-key-label_not_present button',
            };
            const selector = keyMap[e.key];
            if (selector) {
                e.preventDefault();
                e.stopPropagation();
                e.stopImmediatePropagation();
                const btn = doc.querySelector(selector);
                if (btn) btn.click();
            }
        }, true);
        pw.__fwfKeyListener = true;
    }
    </script>
    """,
        height=0,
        width=0,
    )


def label_tab(data_dict):
    data = data_dict["data"]
    total_clips = get_total_clips(data)

    if "labels" not in st.session_state:
        st.session_state.labels = load_labels()
    if "clip_queue" not in st.session_state:
        st.session_state.clip_queue = build_clip_queue(total_clips, st.session_state.labels)
        st.session_state.clip_idx = 0
        st.session_state.session_labeled = 0
        st.session_state.review_maybes = False

    queue = st.session_state.clip_queue
    idx = st.session_state.clip_idx
    labels = st.session_state.labels

    st.sidebar.header("Labeling Progress")
    st.sidebar.metric("This session", st.session_state.session_labeled)
    st.sidebar.metric("Total labeled", f"{len(labels)} / {total_clips}")
    st.sidebar.progress(len(labels) / total_clips)
    if st.session_state.review_maybes:
        st.sidebar.info("Reviewing 'maybe' labels")

    if idx >= len(queue):
        maybe_count = sum(1 for l in labels if l["label"] == "maybe")
        if maybe_count > 0 and not st.session_state.review_maybes:
            st.success(f"All done! {maybe_count} 'maybe' labels to review.")
            if st.button("Review maybes"):
                st.session_state.review_maybes = True
                st.session_state.clip_queue = build_clip_queue(
                    total_clips, labels, review_maybes=True
                )
                st.session_state.clip_idx = 0
                st.rerun()
        else:
            st.success("All clips labeled!")
        return

    clip_index = queue[idx]
    start_utc, end_utc = clip_utc_range(clip_index)
    clip_data = get_clip_data(data, clip_index)

    current_label = None
    for lbl in labels:
        if lbl["start_utc"] == start_utc:
            current_label = lbl["label"]
            break

    st.write(f"**Clip {idx + 1} of {len(queue)} remaining**")
    st.write(f"`{start_utc}` to `{end_utc}`")

    fig = plot_spectrogram(
        clip_data,
        SAMPLING_RATE,
        LABEL_FFT_WINDOW_SEC,
        LABEL_OVERLAP,
        LABEL_FMIN,
        LABEL_FMAX,
        label=current_label,
    )
    st.pyplot(fig)
    plt.close(fig)

    st.caption("**f** = present | **d** = maybe | **s** = not present | **left/right** = navigate")

    st.markdown(
        """<style>
    .st-key-prev_btn, .st-key-next_btn,
    .st-key-label_present, .st-key-label_maybe,
    .st-key-label_not_present { position: absolute; left: -9999px; }
    </style>""",
        unsafe_allow_html=True,
    )

    def do_label(label_value):
        start_utc, end_utc = clip_utc_range(queue[st.session_state.clip_idx])
        st.session_state.labels = [l for l in st.session_state.labels if l["start_utc"] != start_utc]
        st.session_state.labels.append({"start_utc": start_utc, "end_utc": end_utc, "label": label_value})
        save_labels(st.session_state.labels)
        st.session_state.session_labeled += 1
        st.session_state.clip_idx += 1

    st.button("prev", key="prev_btn", on_click=lambda: setattr(st.session_state, "clip_idx", max(0, st.session_state.clip_idx - 1)))
    st.button("next", key="next_btn", on_click=lambda: setattr(st.session_state, "clip_idx", min(len(queue) - 1, st.session_state.clip_idx + 1)))
    st.button("f", key="label_present", on_click=lambda: do_label("present"))
    st.button("d", key="label_maybe", on_click=lambda: do_label("maybe"))
    st.button("s", key="label_not_present", on_click=lambda: do_label("not_present"))

    inject_keyboard_listener()


def browse_tab(data_dict):
    data = data_dict["data"]
    times = data_dict["times"]
    fs = data_dict.get("sampling_rate", 200)

    st.sidebar.header("Parameters")

    total_hours = (times[-1] - times[0]) / 3600
    st.sidebar.write(f"Total data: {total_hours:.1f} hours")

    display_window_min = st.sidebar.slider("Display window (minutes)", 5, 60, 15)
    fft_window_sec = st.sidebar.slider("FFT window (seconds)", 1, 30, 5)
    overlap = st.sidebar.slider("Overlap", 0.25, 0.95, 0.5)
    fmin = st.sidebar.number_input("Min freq (Hz)", 0, 100, 10)
    fmax = st.sidebar.number_input("Max freq (Hz)", 0, 100, 50)

    start_hour = st.slider("Start hour", 0.0, total_hours - 0.25, 0.0, 0.25)

    start_idx = int(start_hour * 3600 * fs)
    end_idx = int((start_hour + display_window_min / 60) * 3600 * fs)

    window_data = data[start_idx:end_idx]
    window_times = times[start_idx:end_idx] - times[start_idx]

    st.write(
        f"Showing: {start_hour:.2f} - {start_hour + display_window_min / 60:.2f} hours"
    )

    fig = plot_spectrogram(
        window_data,
        fs,
        fft_window_sec,
        overlap,
        fmin,
        fmax,
    )

    st.pyplot(fig)

    st.write(f"Window size: {len(window_data) / fs:.1f} seconds")


def main():
    st.title("Fin Whale Spectrogram Viewer")

    data_dict = load_data()
    if data_dict is None:
        return

    tab_browse, tab_label = st.tabs(["Browse", "Label"])
    with tab_browse:
        browse_tab(data_dict)
    with tab_label:
        label_tab(data_dict)


if __name__ == "__main__":
    main()
