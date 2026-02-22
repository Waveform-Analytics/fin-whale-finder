# Fin Whale Finder Research Constitution

## Research Context

This project detects and characterizes fin whale calls in OOI hydrophone data.
It supports reproducible bioacoustic analysis and helps future scientists
quickly understand and extend the workflow. Primary users are project
scientists and collaborators who need traceable data products for analysis,
validation, and publication.
<!-- What scientific questions does this project address? How does it fit
     into the broader research program? Who are the intended users of
     the outputs? -->

## Core Principles

### I. Reproducibility

Analysis should be fully reproducible from raw data to final outputs.
Scripts run without manual intervention. Random seeds are fixed and
documented. Environment dependencies are explicit (requirements.txt,
environment.yml, or equivalent).

### II. Data Integrity

Raw data is immutable - all transformations produce new files, never
overwrite sources. Data lineage is traceable through the analysis chain.
Missing or suspect values are flagged, not silently dropped or filled.

### III. Provenance

Every output links back to: the code that produced it, the input data,
and key parameter choices. Figures and tables can be regenerated from
tracked artifacts. If you can't trace how a number was made, it doesn't
belong in the paper.

## Data Sources

- **OOI hydrophone data**: Primary acoustic source for fin whale call detection.
  - Access method: OOI M2M API via `ooipy` library, wrapped with local caching.
    - The project provides `src/fin_whale_finder/data_access.py` with:
      - `get_hydrophone_data(node, starttime, endtime, ...)` — fetches with
        automatic caching to avoid redundant downloads.
      - `cache_info()` — see what's in the cache.
      - `clear_cache()` — wipe the cache.
    - Credentials stored in `~/.netrc` for `ooipy` authentication.
    - Cache location: `data/cache/` (one `.pkl` file per request).
    - Use `low_freq=True` (default) for 200 Hz HYDLFA data.
    - Use `low_freq=False` for broadband 64 kHz HYDBBA data.
  - Local filesystem context: The OOI JupyterHub at
    `https://jupyter.oceanobservatories.org` mounts data under `/home/jovyan/ooi/`.
    `kdata/` contains metadata-only NetCDF index files (no waveforms locally).
    `rsn_cabled/rsn_data/rsn-tier1/` contains `.genc.bz2` waveforms up to ~2019
    in a proprietary format — not currently used.
  - Target instruments (Cabled Axial Seamount Array):
    - `RS03AXBS-MJ03A-05-HYDLFA301` — Axial Base Seafloor, 2642 m, HTI-90-U
      hydrophone. Station: `OO.AXBA1..HDH` (200 Hz), `OO.AXBA1..LDH` (1 Hz).
    - `RS03CCAL-MJ03F-06-HYDLFA305` — Central Caldera, 1526 m.
    - `RS03ECAL-MJ03E-09-HYDLFA304` — Eastern Caldera, 1516 m.
    - `RS03AXBS-LJ03A-09-HYDBBA302` — Axial Base broadband hydrophone (HYDBBA).
    - `RS03AXBS-MJ03A-05-OBSBBA303` — Axial Base broadband seismometer, co-located
      with HYDLFA301. Useful for cross-validation of T-phase detections.
  - Coverage: 2015-12-14 to present (continuous single deployment).
  - Channel of interest: `HDH` at 200 Hz — captures fin whale 20 Hz calls.
    Avoid `LDH` (1 Hz decimated — too coarse for fin whale work).
  - Update frequency: Near-real-time streaming via cabled array.
  - Known issues: Data gaps exist; check metadata NetCDF for coverage before
    requesting waveforms. Clock drift unknown — assume OOI NTP sync is reliable.
  - Metadata index: `/home/jovyan/ooi/kdata/RS03AXBS-MJ03A-05-HYDLFA301-streamed-antelope_metadata/`
  - Documentation: https://oceanobservatories.org/array/cabled-axial-seamount-array/
<!-- For each major data source:
     - Name and brief description
     - Access method (URL, API, local path)
     - Spatial/temporal coverage
     - Update frequency (if applicable)
     - Known quality issues or limitations
     - Contact or documentation link -->

## Technical Environment

- Language: Python >=3.13 (from `pyproject.toml`).
- Environment/package management: Use `uv` for all package, run, and tool tasks
  (`uv add`, `uv run`, `uv tool`, etc.).
- Linting/formatting/type checking preference: Astral tooling (`ruff`, `ty`) when
  lint/type systems are introduced.
- Notebook policy: Use `marimo` for notebooks; do not use Jupyter notebooks.
  Note: marimo cannot be proxied through the OOI JupyterHub without
  `jupyter-marimo-proxy` installed at the hub level (a sysadmin task).
  Run notebooks via `uv run marimo edit <file>` from a terminal with
  SSH port-forwarding, or run as plain scripts with `uv run python <file>`.
- Scripting preference: Put reusable workflows in `scripts/` as named Python
  scripts rather than ad-hoc bash commands. Use `uv run python <script>` to
  execute. One-off exploratory commands are fine in REPL/debugging; scripts
  are for reproducible, documented workflows.
- Current project structure: `src/` layout with CLI entrypoint `fwf`.
- Version control: Git repository; analysis and pipeline changes are tracked and
  reviewable.
- Compute environment: OOI JupyterHub (shared cloud instance); data is
  co-located, no external transfer needed during analysis.
- Data storage locations:
  - Raw data: fetched on demand via M2M API; not stored locally except as cache.
  - Intermediate: `data/` directory in project root (gitignored for large files).
  - Outputs: `results/` for detections, `outputs/` for figures.
  - Instrument inventory: `data/axial_instrument_inventory.csv` (committed).
<!-- - Language and version (e.g., Python 3.11)
     - Key packages and versions
     - Compute environment (laptop, cluster, cloud)
     - Data storage locations
     - Version control practices -->

## Coordinate Systems & Units

- Spatial reference system: Not applicable for primary analysis — instruments are
  fixed seafloor nodes. Station coordinates (WGS84) stored as metadata only:
  HYDLFA301 at 45.820189°N, -129.736708°E, 2642 m depth.
- Time conventions: All timestamps in UTC. OOI data uses NTP-synced timing.
  Internal representation: `datetime64[ns]` (xarray/pandas default).
- Standard units:
  - Acoustic pressure: counts (raw) → Pa via instrument sensitivity calibration.
  - Frequency: Hz. Fin whale 20 Hz pulse band of interest: 15–30 Hz.
  - Sample rate: 200 Hz (HDH channel).
  - Time window size for detection: TBD — likely 60–300 s segments.
- Missing data conventions: NaN for float data gaps. Gaps in time series flagged
  explicitly; do not interpolate across outages.
<!-- - Spatial reference system(s) with EPSG codes
     - Time zone and calendar conventions
     - Standard units for key variables
     - Missing data conventions (NaN, -9999, etc.) -->

## Figure Standards

- Palette: colorblind-safe defaults required (e.g., Okabe-Ito or similar).
- Output targets: [TODO: define journal, report, or presentation targets]
- Standard dimensions: [TODO: define figure sizes for target venues]
- Required elements: clear axis labels/units, station metadata, time zone note
  (UTC), uncertainty or confidence indicators where applicable.
- Formats/resolution: vector (`.pdf`/`.svg`) for line art, raster (`.png`) at
  >=300 dpi for publication.
<!-- - Color palette (prefer colorblind-safe)
     - Standard dimensions for publication
     - Required elements (scale bars, colorbars, uncertainty)
     - File formats and resolution (e.g., PDF for vectors, 300dpi PNG) -->

## Quality Checks

- Input validation: verify station IDs, time bounds, and positive window sizes.
- Temporal consistency: enforce `end > start`; ensure contiguous/expected window
  slicing in manifests.
- Data completeness: [TODO: thresholds for acceptable data gaps per station/day]
- Detection validation: [TODO: define reference labels, precision/recall targets,
  and review protocol]
- Suspect data handling: flag and log suspect segments; do not silently drop.
<!-- - Range and sanity checks for key variables
     - Spatial/temporal consistency checks
     - Comparison against reference or validation data
     - How suspect data is flagged and handled -->

## Project Notes

- Reproducible science and high-quality code are explicit project priorities.
- Documentation should be learner-friendly so new scientists can onboard quickly.
- Tooling constraints: prefer Astral ecosystem and `uv`-managed workflows.
- Collaboration/publication constraints: [TODO: add embargoes, agreements,
  deadlines, and sharing rules]
<!-- - Collaborator agreements or data sharing restrictions
     - Publication timelines or embargo periods
     - Any other project-specific constraints -->
