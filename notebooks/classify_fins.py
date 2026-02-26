import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Fin whale spectrogram classifier

    This notebook trains a CNN (convolutional neural network) to classify
    15-minute hydrophone clips as "fin whale present" or "not present",
    using spectrograms as input images.

    **What's happening at a high level:**

    1. Load our labeled audio clips (from the Streamlit labeling tool)
    2. Convert each clip into a spectrogram image focused on the 15–30 Hz band
    3. Split into training and test sets
    4. Fine-tune a pretrained image classifier (ResNet18) on our spectrograms
    5. Evaluate how well it learned to tell the difference

    **Why spectrograms?** Fin whale 20 Hz pulses are visible as bright vertical
    bands in a spectrogram. By converting audio to images, we can use powerful
    image classification models that already know how to detect visual patterns —
    we just teach them what *our* patterns look like.
    """)
    return


@app.cell
def _():
    """Imports and configuration."""
    import json
    import pickle
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import signal

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision import models

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        ConfusionMatrixDisplay,
    )

    # --- Configuration ---
    # These match the spectrogram_viewer.py settings so the images look
    # the same as what we labeled against.
    DATASET_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
    CLIP_DURATION_SEC = 900  # 15 minutes per clip
    SAMPLING_RATE = 200      # Hz — the OOI low-frequency hydrophone rate
    SAMPLES_PER_CLIP = CLIP_DURATION_SEC * SAMPLING_RATE

    # Spectrogram parameters
    FFT_WINDOW_SEC = 5   # each FFT looks at 5 seconds of audio
    OVERLAP = 0.5        # consecutive FFT windows overlap by 50%
    FMIN = 10            # Hz — low end of our frequency range
    FMAX = 50            # Hz — high end (fin whale calls are ~15-30 Hz)

    # Paths — resolve relative to this notebook's location
    _NB_DIR = Path(__file__).parent
    _PROJECT_ROOT = _NB_DIR.parent
    CACHE_FILE = _PROJECT_ROOT / "data/cache/Axial_Base_2026-01-01_to_2026-01-07.pkl"
    LABELS_FILE = _PROJECT_ROOT / "labels/labels.json"
    return (
        CACHE_FILE,
        CLIP_DURATION_SEC,
        ConfusionMatrixDisplay,
        DATASET_EPOCH,
        DataLoader,
        FFT_WINDOW_SEC,
        FMAX,
        FMIN,
        LABELS_FILE,
        OVERLAP,
        Path,
        SAMPLES_PER_CLIP,
        SAMPLING_RATE,
        TensorDataset,
        classification_report,
        confusion_matrix,
        datetime,
        json,
        models,
        nn,
        np,
        pickle,
        plt,
        signal,
        timezone,
        torch,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Load data and labels

    We load the full week of hydrophone audio (168 hours) and our hand-labeled
    clips from the Streamlit labeling tool.

    We drop the "maybe" labels and keep only "present" and "not_present" —
    a clean binary signal for the classifier to learn from.
    """)
    return


@app.cell
def _(CACHE_FILE, LABELS_FILE, json, pickle):
    with open(CACHE_FILE, "rb") as _f:
        _data_dict = pickle.load(_f)

    audio_data = _data_dict["data"]
    print(f"Audio: {len(audio_data):,} samples ({len(audio_data)/200/3600:.1f} hours)")

    labels_raw = json.loads(LABELS_FILE.read_text())
    print(f"Labels: {len(labels_raw)} total")

    # Keep only clean binary labels — "maybe" is too ambiguous for training
    labels_binary = [l for l in labels_raw if l["label"] in ("present", "not_present")]
    n_present = sum(1 for l in labels_binary if l["label"] == "present")
    n_absent = sum(1 for l in labels_binary if l["label"] == "not_present")
    print(f"Binary labels: {len(labels_binary)} ({n_present} present, {n_absent} not_present)")
    return audio_data, labels_binary


@app.cell
def _(mo):
    mo.md("""
    ## 2. Generate spectrograms

    For each labeled 15-minute clip, we compute a spectrogram — a 2D image where:

    - **X axis** = time (across the 15-minute clip)
    - **Y axis** = frequency (10–50 Hz, focused on the fin whale band)
    - **Color** = intensity (how loud that frequency is at that moment)

    Fin whale 20 Hz pulses show up as bright spots or horizontal bands
    at ~20 Hz, repeating at regular intervals.

    We use the same spectrogram settings as the Streamlit labeling tool
    so the images match what you were looking at when you labeled them.
    The values are converted to decibels (log scale) because acoustic
    energy varies over huge ranges — log scale compresses that into
    something more visually useful.
    """)
    return


@app.cell
def _(
    CLIP_DURATION_SEC,
    DATASET_EPOCH,
    FFT_WINDOW_SEC,
    FMAX,
    FMIN,
    OVERLAP,
    SAMPLES_PER_CLIP,
    SAMPLING_RATE,
    audio_data,
    datetime,
    labels_binary,
    np,
    signal,
    timezone,
):
    def make_spectrogram(clip_data, fs, fft_window_sec, overlap, fmin, fmax):
        """Generate a spectrogram array for one clip.

        Returns a 2D numpy array (freq_bins x time_bins) in decibels.
        """
        nperseg = int(fft_window_sec * fs)
        noverlap = int(nperseg * overlap)

        # scipy.signal.spectrogram computes the Short-Time Fourier Transform:
        # it slides a window across the audio and computes the frequency content
        # at each position. The result is a 2D array: frequency x time.
        f, t, Sxx = signal.spectrogram(
            clip_data,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
        )

        # Keep only the frequency range we care about (10-50 Hz)
        mask = (f >= fmin) & (f <= fmax)
        Sxx_crop = Sxx[mask, :]

        # Convert to decibels (log scale). The 1e-10 prevents log(0).
        Sxx_db = 10 * np.log10(Sxx_crop + 1e-10)

        return Sxx_db, f[mask], t

    # Walk through each label, find the corresponding audio clip,
    # and generate its spectrogram
    spectrograms = []
    spec_labels = []
    spec_times = []  # keep timestamps so we can trace back to specific clips

    for _entry in labels_binary:
        _start_utc = datetime.fromisoformat(_entry["start_utc"])
        # Figure out which clip this is by calculating seconds from the start
        _offset_sec = (_start_utc - DATASET_EPOCH.replace(tzinfo=timezone.utc)).total_seconds()
        _clip_index = int(_offset_sec / CLIP_DURATION_SEC)

        # Extract the audio samples for this clip
        _start_sample = _clip_index * SAMPLES_PER_CLIP
        _end_sample = _start_sample + SAMPLES_PER_CLIP
        if _end_sample > len(audio_data):
            continue

        _clip = audio_data[_start_sample:_end_sample]
        _Sxx_db, _, _ = make_spectrogram(
            _clip, SAMPLING_RATE, FFT_WINDOW_SEC, OVERLAP, FMIN, FMAX
        )

        spectrograms.append(_Sxx_db)
        spec_labels.append(1 if _entry["label"] == "present" else 0)
        spec_times.append(_entry["start_utc"])

    spectrograms = np.array(spectrograms, dtype=np.float32)
    spec_labels = np.array(spec_labels)

    print(f"Generated {len(spectrograms)} spectrograms")
    print(f"Each spectrogram shape: {spectrograms.shape[1:]} (freq_bins x time_bins)")
    return spec_labels, spec_times, spectrograms


@app.cell
def _(mo):
    mo.md("""
    ## 3. Visual sanity check

    Before training anything, let's look at example spectrograms from each class
    side by side. **If you can't see a difference, the model probably can't either.**

    Look for: bright horizontal features at ~20 Hz repeating at regular intervals
    in the "present" row. The "not present" row should look quieter or have
    different patterns (e.g., ship noise, ambient rumble).
    """)
    return


@app.cell
def _(np, plt, spec_labels, spec_times, spectrograms):
    _present_idxs = np.where(spec_labels == 1)[0][:3]
    _absent_idxs = np.where(spec_labels == 0)[0][:3]

    fig_examples, axes_ex = plt.subplots(2, 3, figsize=(15, 6))

    for _i, _idx in enumerate(_present_idxs):
        axes_ex[0, _i].imshow(
            spectrograms[_idx], aspect="auto", origin="lower", cmap="viridis"
        )
        axes_ex[0, _i].set_title(f"PRESENT\n{spec_times[_idx][:16]}", fontsize=9)
        axes_ex[0, _i].set_ylabel("Freq bin" if _i == 0 else "")

    for _i, _idx in enumerate(_absent_idxs):
        axes_ex[1, _i].imshow(
            spectrograms[_idx], aspect="auto", origin="lower", cmap="viridis"
        )
        axes_ex[1, _i].set_title(f"NOT PRESENT\n{spec_times[_idx][:16]}", fontsize=9)
        axes_ex[1, _i].set_ylabel("Freq bin" if _i == 0 else "")

    plt.tight_layout()
    fig_examples
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Train/test split and normalization

    We split the data into two groups:

    - **Training set (80%)**: the model learns from these
    - **Test set (20%)**: held out — the model never sees these during training,
      so they tell us how well it generalizes to new data

    We use a **stratified** split, meaning both sets keep the same ratio of
    present/not_present clips. Without this, the small test set might end up
    with too few of one class to evaluate properly.

    **Normalization**: we subtract the mean and divide by standard deviation
    so all pixel values are centered around zero. Neural networks train better
    when inputs are on a similar scale.

    **3-channel trick**: ResNet was trained on color photos (RGB = 3 channels).
    Our spectrograms are single-channel (grayscale). We just copy the same
    image into all 3 channels — the model doesn't care that it's not "real" color.
    """)
    return


@app.cell
def _(np, spec_labels, spectrograms, torch, train_test_split):
    X_train, X_test, y_train, y_test = train_test_split(
        spectrograms, spec_labels, test_size=0.2, random_state=42, stratify=spec_labels
    )

    print(f"Train: {len(X_train)} ({sum(y_train)} present, {len(y_train) - sum(y_train)} not present)")
    print(f"Test:  {len(X_test)} ({sum(y_test)} present, {len(y_test) - sum(y_test)} not present)")

    # Normalize using training set statistics only.
    # Important: we compute mean/std from training data and apply it to both
    # sets. If we used test data for normalization, we'd be "peeking" at the
    # test set, which would make our evaluation less trustworthy.
    train_mean = X_train.mean()
    train_std = X_train.std()
    X_train_norm = (X_train - train_mean) / train_std
    X_test_norm = (X_test - train_mean) / train_std

    # Copy single channel into 3 channels for ResNet compatibility
    _X_train_3ch = np.stack([X_train_norm] * 3, axis=1)  # shape: (N, 3, freq, time)
    _X_test_3ch = np.stack([X_test_norm] * 3, axis=1)

    # Convert numpy arrays to PyTorch tensors
    train_tensors = torch.tensor(_X_train_3ch, dtype=torch.float32)
    train_labels_t = torch.tensor(y_train, dtype=torch.long)
    test_tensors = torch.tensor(_X_test_3ch, dtype=torch.float32)
    test_labels_t = torch.tensor(y_test, dtype=torch.long)

    print(f"\nTensor shapes: train={train_tensors.shape}, test={test_tensors.shape}")
    return (
        X_test,
        test_tensors,
        train_labels_t,
        train_mean,
        train_std,
        train_tensors,
        y_test,
        y_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Train the classifier

    We're using **ResNet18** — a convolutional neural network originally trained
    on ImageNet (millions of photos of everyday objects). Even though it's never
    seen a spectrogram, its early layers learned to detect edges, textures, and
    patterns that transfer well to any kind of image.

    **Fine-tuning** means we take this pretrained network, chop off its last layer
    (which originally predicted 1000 object categories), and replace it with our own
    layer that just predicts 2 classes: present vs not_present. Then we retrain
    the whole network on our spectrograms with a small learning rate, so it gently
    adapts its existing knowledge to our specific task.

    **Class weights** compensate for the imbalanced dataset (more "present" than
    "not_present" clips). Without weights, the model could get 73% accuracy by
    just always predicting "present" — not useful. The weights make errors on
    the smaller class count for more, forcing the model to actually learn both classes.

    **What to watch for in the training output:**
    - Loss should decrease over epochs (the model is learning)
    - Accuracy should increase
    - If loss stops decreasing early, 20 epochs might be more than enough
    - If accuracy hits 100% on training data very quickly, the model might be
      memorizing rather than generalizing — the test set evaluation will reveal this
    """)
    return


@app.cell
def _(
    DataLoader,
    TensorDataset,
    models,
    nn,
    torch,
    train_labels_t,
    train_tensors,
    y_train,
):
    # Class weights: inversely proportional to class frequency
    _n_not_present = int((y_train == 0).sum())
    _n_present_train = int((y_train == 1).sum())
    _weight_not_present = len(y_train) / (2 * _n_not_present)
    _weight_present = len(y_train) / (2 * _n_present_train)
    class_weights = torch.tensor([_weight_not_present, _weight_present], dtype=torch.float32)
    print(f"Class weights: not_present={_weight_not_present:.2f}, present={_weight_present:.2f}")

    # Load pretrained ResNet18 and swap the final classification layer
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 output classes

    # CrossEntropyLoss combines softmax + negative log likelihood — the standard
    # loss function for classification. Adam is an optimizer that adapts the
    # learning rate per-parameter (generally works well out of the box).
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # DataLoader handles batching and shuffling — instead of feeding all 182
    # training examples at once, it serves them in batches of 16, in random
    # order each epoch. This helps the model learn more robustly.
    _train_dataset = TensorDataset(train_tensors, train_labels_t)
    _train_loader = DataLoader(_train_dataset, batch_size=16, shuffle=True)

    n_epochs = 20
    model.train()  # puts model in training mode (enables dropout, batch norm, etc.)
    losses = []

    for _epoch in range(n_epochs):
        _epoch_loss = 0.0
        _correct = 0
        _total = 0

        for _batch_x, _batch_y in _train_loader:
            optimizer.zero_grad()       # reset gradients from previous batch
            _outputs = model(_batch_x)  # forward pass: images → predictions
            _loss = criterion(_outputs, _batch_y)  # how wrong were we?
            _loss.backward()            # compute gradients (backpropagation)
            optimizer.step()            # update weights using gradients

            _epoch_loss += _loss.item() * _batch_x.size(0)
            _, _predicted = _outputs.max(1)
            _correct += (_predicted == _batch_y).sum().item()
            _total += _batch_y.size(0)

        _avg_loss = _epoch_loss / _total
        _acc = _correct / _total
        losses.append(_avg_loss)
        if (_epoch + 1) % 5 == 0:
            print(f"Epoch {_epoch+1:3d}/{n_epochs}  loss={_avg_loss:.4f}  acc={_acc:.3f}")
    return losses, model


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Training loss curve

    The loss measures how wrong the model's predictions are — lower is better.
    A healthy training curve drops steeply at first (the model is learning fast)
    then levels off. If it bounces around a lot at the end, that's normal for
    a small dataset.
    """)
    return


@app.cell
def _(losses, plt):
    fig_loss, ax_loss = plt.subplots(figsize=(8, 3))
    ax_loss.plot(losses, linewidth=2)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Loss")
    ax_loss.grid(True, alpha=0.3)
    fig_loss
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Evaluation on held-out test set

    This is the moment of truth. The model has never seen these 46 clips.

    **Key metrics:**
    - **Precision** = of the clips the model *called* present, how many actually were?
      (High precision = few false alarms)
    - **Recall** = of the clips that *actually were* present, how many did the model catch?
      (High recall = few missed detections)
    - **F1-score** = harmonic mean of precision and recall (a single summary number)

    **The confusion matrix** shows the 2x2 breakdown: correct predictions on the
    diagonal, errors off the diagonal. Read it as: rows are what the clip *actually*
    is, columns are what the model *predicted*.
    """)
    return


@app.cell
def _(
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    model,
    plt,
    test_tensors,
    torch,
    y_test,
):
    model.eval()  # puts model in evaluation mode (disables dropout, etc.)
    with torch.no_grad():  # don't compute gradients — we're just predicting
        _test_outputs = model(test_tensors)
        _, _test_preds = _test_outputs.max(1)  # pick the class with highest score

    test_preds_np = _test_preds.numpy()

    print(classification_report(
        y_test, test_preds_np, target_names=["not_present", "present"]
    ))

    _cm = confusion_matrix(y_test, test_preds_np)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(_cm, display_labels=["not_present", "present"]).plot(ax=ax_cm)
    ax_cm.set_title("Confusion Matrix (Test Set)")
    fig_cm
    return (test_preds_np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. Error analysis

    The most useful part of any ML experiment: **look at what the model got wrong.**
    Each spectrogram below was misclassified. Ask yourself:

    - Does the true label look correct? (Maybe *you* mislabeled it?)
    - Is there something ambiguous about this clip? (Faint calls, partial calls?)
    - Is there a pattern to the errors? (All from the same time period? All borderline?)

    Understanding the errors tells you where to focus next — more labeling,
    different spectrogram parameters, or different model architecture.
    """)
    return


@app.cell
def _(X_test, mo, np, plt, test_preds_np, y_test):
    _wrong = np.where(test_preds_np != y_test)[0]
    _label_names = {0: "not_present", 1: "present"}

    if len(_wrong) == 0:
        _result = mo.md(
            """**No misclassifications!** Every test clip was predicted correctly.

            That sounds great, but be skeptical — with only 46 test clips,
            it's possible the model got lucky. More data would give us a more
            trustworthy evaluation."""
        )
    else:
        _n_show = min(len(_wrong), 6)
        fig_err, axes_err = plt.subplots(1, _n_show, figsize=(4 * _n_show, 4))
        if _n_show == 1:
            axes_err = [axes_err]

        for _i in range(_n_show):
            _idx = _wrong[_i]
            axes_err[_i].imshow(
                X_test[_idx], aspect="auto", origin="lower", cmap="viridis"
            )
            axes_err[_i].set_title(
                f"True: {_label_names[y_test[_idx]]}\n"
                f"Pred: {_label_names[test_preds_np[_idx]]}",
                fontsize=9,
            )
        plt.tight_layout()
        _result = fig_err

    _result
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8. Save model + config

    We save the trained model weights AND the spectrogram settings together
    in one file. This is important because the model only works with spectrograms
    generated using these exact parameters. If you later change the FFT window
    or frequency range, you'd need to retrain.

    The saved file contains:
    - `model_state_dict`: the learned weights
    - `spectrogram_config`: FFT window, overlap, frequency range, etc.
    - `normalization`: mean and std used to normalize the spectrograms
    - `class_names`: what 0 and 1 mean
    """)
    return


@app.cell
def _(Path, model, torch, train_mean, train_std):
    _save_dir = Path(__file__).parent.parent / "models"
    _save_dir.mkdir(exist_ok=True)

    _checkpoint = {
        "model_state_dict": model.state_dict(),
        "spectrogram_config": {
            "fft_window_sec": 5,
            "overlap": 0.5,
            "fmin": 10,
            "fmax": 50,
            "sampling_rate": 200,
            "clip_duration_sec": 900,
        },
        "normalization": {
            "mean": float(train_mean),
            "std": float(train_std),
        },
        "class_names": ["not_present", "present"],
    }

    _save_path = _save_dir / "resnet18_fin_classifier_v0.pt"
    torch.save(_checkpoint, _save_path)
    print(f"Saved to {_save_path}")
    return


if __name__ == "__main__":
    app.run()
