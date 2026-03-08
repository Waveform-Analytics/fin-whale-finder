import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("# Soundscape Results Browser")
    return


@app.cell
def _(mo):
    from pathlib import Path
    FIGURES_ROOT = Path(__file__).parent.parent / "results" / "soundscape_figures"

    if FIGURES_ROOT.exists():
        nodes = sorted(d.name for d in FIGURES_ROOT.iterdir() if d.is_dir())
    else:
        nodes = []

    node_picker = mo.ui.dropdown(
        options=nodes,
        label="Node",
        value=nodes[0] if nodes else None,
    )
    node_picker
    return FIGURES_ROOT, Path, node_picker, nodes


@app.cell
def _(FIGURES_ROOT, mo, node_picker):
    _node = node_picker.value
    if _node:
        _node_dir = FIGURES_ROOT / _node
        dates = sorted(d.name for d in _node_dir.iterdir() if d.is_dir())
    else:
        dates = []

    date_picker = mo.ui.dropdown(
        options=dates,
        label="Date / slug",
        value=dates[-1] if dates else None,
    )
    date_picker
    return date_picker, dates


@app.cell
def _(FIGURES_ROOT, date_picker, mo, node_picker):
    _node = node_picker.value
    _slug = date_picker.value

    # Figures worth showing in the browser (skip 04_top_anomalies — slow to gen)
    WANTED = ["01_umap_by_time", "02_umap_clusters_outliers", "03_outlier_timeline", "05_cluster_examples"]

    if _node and _slug:
        _fig_dir = FIGURES_ROOT / _node / _slug
        _images = []
        for _prefix in WANTED:
            _matches = sorted(_fig_dir.glob(f"{_prefix}*.png"))
            for _p in _matches:
                _images.append(
                    mo.vstack([
                        mo.md(f"**{_p.name}**"),
                        mo.image(str(_p)),
                    ])
                )
        if _images:
            result = mo.vstack(_images)
        else:
            result = mo.callout(mo.md(f"No figures found in `{_fig_dir}`"), kind="warn")
    else:
        result = mo.callout(mo.md("Select a node and date above."), kind="info")

    result
    return WANTED, result
