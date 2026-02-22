"""
inventory_axial_instruments.py
===============================
Scan /ooi/kdata for acoustic and seismic instruments near Axial Seamount
that are relevant to fin whale call detection (~20 Hz).

Usage
-----
    uv run python scripts/inventory_axial_instruments.py

Output
------
- A summary table printed to the terminal.
- A CSV saved to data/axial_instrument_inventory.csv (safe to commit to git).

Background: OOI reference designators
--------------------------------------
OOI organises data files using reference designators that encode the array,
site, node, and instrument. Example:

    RS03AXBS-MJ03A-05-HYDLFA301

    RS03AXBS  site: Axial Base Seafloor (Regional Cabled Array)
    MJ03A     node: Medium-Power Junction Box A
    05        port on the junction box
    HYDLFA301 instrument class/series/unit

Instruments we search for
--------------------------
HYDLFA  Low Frequency Acoustic Receiver (HTI 90-U hydrophone)
        Bandwidth ~0-1000 Hz — directly captures fin whale 20 Hz calls.
        Three units at Axial Seamount:
          RS03AXBS-MJ03A-05-HYDLFA301  Axial Base Seafloor, 2642 m
          RS03CCAL-MJ03F-06-HYDLFA305  Central Caldera, 1526 m
          RS03ECAL-MJ03E-09-HYDLFA304  Eastern Caldera, 1516 m

OBSBBA  Broadband Ocean Bottom Seismometer (Guralp CMG-1T)
        Bandwidth 360 s – 50 Hz — fin whale calls appear as hydroacoustic
        T-phases on the vertical channel. Co-located with a hydrophone.
        One unit at Axial:
          RS03AXBS-LJ03A-12-OBSBBA302  Axial Base Seafloor, buried in caisson
"""

import csv
import re
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Root of the OOI data filesystem on this JupyterHub instance.
KDATA_ROOT = Path("/home/jovyan/ooi/kdata")

# OOI site codes for Axial Seamount. All reference designators at these sites
# start with one of these strings.
AXIAL_SITE_CODES = [
    "RS03AXBS",  # Axial Base Seafloor
    "RS03CCAL",  # Central Caldera
    "RS03ECAL",  # Eastern Caldera
]

# Instrument class codes to search for within those sites.
TARGET_INSTRUMENT_CLASSES = [
    "HYDLFA",  # Low Frequency Hydrophone
    "OBSBBA",  # Broadband Seismometer
]

# Where to write the output CSV.
OUTPUT_CSV = Path("data/axial_instrument_inventory.csv")


# ---------------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------------


def extract_date(filename: str) -> datetime | None:
    """
    Parse a date from an OOI data filename.

    OOI filenames typically embed timestamps in one of these formats:
      20150424T134500   (compact ISO-8601)
      2015-04-24T13-45-00
      20150424_134500

    Returns a datetime if a date is found, otherwise None. We only need
    the date portion for the inventory range summary.
    """
    patterns = [
        r"(\d{4})(\d{2})(\d{2})T\d{6}",  # 20150424T134500
        r"(\d{4})-(\d{2})-(\d{2})T[\d\-]+",  # 2015-04-24T13-45-00
        r"(\d{4})(\d{2})(\d{2})_\d{6}",  # 20150424_134500
        r"(\d{4})(\d{2})(\d{2})",  # bare YYYYMMDD fallback
    ]
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            try:
                year = int(match.group(1))
                month = int(match.group(2))
                day = int(match.group(3))
                return datetime(year, month, day)
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Filesystem search
# ---------------------------------------------------------------------------


def find_axial_site_dirs(root: Path) -> list[Path]:
    """
    Find directories under root whose names start with an Axial site code.

    We check two layouts because OOI kdata trees vary:
      Layout A: site dirs are direct children of root
                root/RS03AXBS-MJ03A/...
      Layout B: site dirs are one level deeper, grouped by array
                root/RS/RS03AXBS-MJ03A/...

    Returns a flat list of matching directories.
    """
    # Layout A: direct children
    candidates = [d for d in root.iterdir() if d.is_dir()]
    matches = [
        d
        for d in candidates
        if any(d.name.startswith(code) for code in AXIAL_SITE_CODES)
    ]

    if matches:
        return matches

    # Layout B: one level deeper
    matches = [
        child
        for d in candidates
        if d.is_dir()
        for child in d.iterdir()
        if child.is_dir()
        and any(child.name.startswith(code) for code in AXIAL_SITE_CODES)
    ]
    return matches


def find_instrument_dirs(site_dirs: list[Path]) -> list[Path]:
    """
    Filter site_dirs for those whose names contain a target instrument code.

    In /home/jovyan/ooi/kdata the layout is flat — each directory IS the
    full reference designator plus stream name, e.g.:
        RS03CCAL-MJ03F-06-HYDLFA305-streamed-antelope_metadata

    So we don't recurse; we just check whether each site directory name
    contains one of our target instrument class codes.
    """
    return [
        d
        for d in site_dirs
        if any(code in d.name for code in TARGET_INSTRUMENT_CLASSES)
    ]


# ---------------------------------------------------------------------------
# Inventory builder
# ---------------------------------------------------------------------------


def build_inventory(instrument_dirs: list[Path]) -> list[dict]:
    """
    For each instrument directory, count data files and determine the date
    range from filenames.

    Returns a list of dicts, one per instrument, with keys:
      instrument_code   - the directory name (= reference designator fragment)
      instrument_path   - full path on the filesystem
      file_count        - number of files found (recursively)
      earliest_date     - ISO date string of oldest file, or 'unknown'
      latest_date       - ISO date string of newest file, or 'unknown'
      file_extensions   - comma-separated set of file suffixes found
    """
    rows = []
    for inst_dir in instrument_dirs:
        data_files = [f for f in inst_dir.rglob("*") if f.is_file()]

        dates = [extract_date(f.name) for f in data_files]
        dates = [d for d in dates if d is not None]

        rows.append(
            {
                "instrument_code": inst_dir.name,
                "instrument_path": str(inst_dir),
                "file_count": len(data_files),
                "earliest_date": min(dates).date().isoformat() if dates else "unknown",
                "latest_date": max(dates).date().isoformat() if dates else "unknown",
                "file_extensions": ", ".join(
                    sorted({f.suffix for f in data_files if f.suffix})
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_summary(rows: list[dict]) -> None:
    """Print a readable summary table to stdout."""
    if not rows:
        print("No matching instruments found.")
        return

    # Column widths based on content
    col_code = max(len(r["instrument_code"]) for r in rows)
    col_files = max(len(str(r["file_count"])) for r in rows)

    header = (
        f"{'INSTRUMENT CODE':<{col_code}}  "
        f"{'FILES':>{col_files}}  "
        f"{'EARLIEST':10}  {'LATEST':10}  EXTENSIONS"
    )
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['instrument_code']:<{col_code}}  "
            f"{r['file_count']:>{col_files}}  "
            f"{r['earliest_date']:10}  "
            f"{r['latest_date']:10}  "
            f"{r['file_extensions']}"
        )
    print()
    print(f"Total instruments found: {len(rows)}")
    print(f"Total files:             {sum(r['file_count'] for r in rows)}")


def save_csv(rows: list[dict], path: Path) -> None:
    """Write the inventory rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "instrument_code",
        "instrument_path",
        "file_count",
        "earliest_date",
        "latest_date",
        "file_extensions",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Inventory saved to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # 1. Check the data root exists before doing anything else.
    if not KDATA_ROOT.exists():
        print(f"ERROR: {KDATA_ROOT} not found.")
        print("The OOI data filesystem may not be mounted in this session.")
        print(
            "Check available mounts with: python -c \"import os; print(os.listdir('/'))\""
        )
        sys.exit(1)

    print(f"Scanning {KDATA_ROOT} for Axial Seamount instruments...")

    # 2. Find site-level directories for our three Axial sites.
    site_dirs = find_axial_site_dirs(KDATA_ROOT)
    if not site_dirs:
        print(f"ERROR: No Axial Seamount site directories found under {KDATA_ROOT}.")
        print(f"Expected directories starting with: {AXIAL_SITE_CODES}")
        print("The directory layout may differ — check the top level manually:")
        print(
            f"  python -c \"from pathlib import Path; print(list(Path('{KDATA_ROOT}').iterdir())[:20])\""
        )
        sys.exit(1)

    print(f"Found {len(site_dirs)} Axial site director(ies):")
    for d in site_dirs:
        print(f"  {d}")

    # 3. Find instrument directories within those sites.
    print(f"\nSearching for {TARGET_INSTRUMENT_CLASSES} instruments...")
    instrument_dirs = find_instrument_dirs(site_dirs)
    if not instrument_dirs:
        print("ERROR: No target instrument directories found.")
        print("Check the site directory structure manually.")
        sys.exit(1)

    # 4. Build the inventory.
    print(f"Found {len(instrument_dirs)} instrument director(ies). Counting files...")
    rows = build_inventory(instrument_dirs)

    # 5. Print and save.
    print_summary(rows)
    save_csv(rows, OUTPUT_CSV)


if __name__ == "__main__":
    main()
