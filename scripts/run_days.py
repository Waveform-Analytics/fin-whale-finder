#!/usr/bin/env python
"""
Run the full soundscape pipeline (fetch + analyze) for one or more calendar days.

Fetch is always parallelized across days (network-bound).
Analysis is sequential by default; use --parallel-analyze N to run N days at once.
Each day's results are independent: if analysis fails for one day,
the others are unaffected and the fetch cache is preserved.

Usage
-----
    # Fetch all days in parallel, then analyze one at a time
    uv run python scripts/run_days.py 2025-06-04 2025-06-11 2025-06-18

    # Process a date range
    uv run python scripts/run_days.py --from 2025-06-04 --to 2025-06-10

    # Only fetch (skip analysis)
    uv run python scripts/run_days.py --fetch-only 2025-06-04 2025-06-11

    # Only analyze (data already fetched)
    uv run python scripts/run_days.py --analyze-only 2025-06-04 2025-06-11

    # Analyze multiple days in parallel (safe if GPU has enough VRAM)
    uv run python scripts/run_days.py --analyze-only --parallel-analyze 4 --from 2025-07-01 --to 2025-07-31

    # Limit parallel fetch workers per day (default 4; lower if OOI throttles)
    uv run python scripts/run_days.py --workers 2 2025-06-04 2025-06-11

    # Embed only 5 minutes per hour (duty cycle)
    uv run python scripts/run_days.py --duty-cycle 5 2025-06-04 2025-06-11
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
PYTHON = sys.executable


def _run(cmd, label, log_path):
    """Run a subprocess, writing all output to log_path. Prints one clean line to stdout."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"  [{label}] starting  →  {log_path.relative_to(PROJECT_ROOT)}", flush=True)
    with open(log_path, "w") as lf:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=lf, stderr=lf)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (see log)"
    print(f"  [{label}] {status} in {elapsed / 60:.1f} min", flush=True)
    return result.returncode, elapsed


def fetch_day(date, workers):
    cmd = [
        PYTHON,
        "scripts/fetch_broadband.py",
        "--date",
        date,
        "--workers",
        str(workers),
    ]
    return _run(cmd, f"FETCH {date}", LOG_DIR / f"fetch_{date}.log")


def analyze_day(date, duty_cycle=None, figures=None):
    cmd = [PYTHON, "scripts/soundscape_explore.py", "--date", date]
    if duty_cycle is not None:
        cmd += ["--duty-cycle", str(duty_cycle)]
    if figures is not None:
        cmd += ["--figures", figures]
    return _run(cmd, f"ANALYZE {date}", LOG_DIR / f"analyze_{date}.log")


def main():
    parser = argparse.ArgumentParser(
        description="Run soundscape pipeline for multiple days",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dates",
        nargs="*",
        metavar="YYYY-MM-DD",
        help="Individual calendar days to process",
    )
    parser.add_argument(
        "--from",
        dest="date_from",
        metavar="YYYY-MM-DD",
        help="Start of a date span (inclusive)",
    )
    parser.add_argument(
        "--to",
        dest="date_to",
        metavar="YYYY-MM-DD",
        help="End of a date span (inclusive). Requires --from.",
    )
    parser.add_argument(
        "--duty-cycle",
        type=int,
        metavar="MINUTES",
        dest="duty_cycle",
        help="Minutes of each hour to embed (0 = full hour, 5 = 5 min/hr)",
    )
    parser.add_argument(
        "--fetch-only", action="store_true", help="Only download data, skip analysis"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only run analysis (data must already be fetched)",
    )
    parser.add_argument(
        "--fetch-first",
        action="store_true",
        help="(No-op: parallel fetch is now always the default)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Download workers per day (default 4; lower if OOI throttles)",
    )
    parser.add_argument(
        "--parallel-days",
        type=int,
        default=3,
        dest="parallel_days",
        help="Max days to fetch simultaneously (default 3; raise with caution)",
    )
    parser.add_argument(
        "--parallel-analyze",
        type=int,
        default=1,
        dest="parallel_analyze",
        help="Max days to analyze simultaneously (default 1 = sequential). "
        "Safe to raise on high-VRAM systems (L40S has 46GB; Perch uses ~1GB each).",
    )
    parser.add_argument(
        "--figures",
        metavar="LIST",
        help="Comma-separated figures to generate per day (e.g. cluster_examples,anomalies)",
    )
    args = parser.parse_args()

    if args.fetch_only and args.analyze_only:
        parser.error("--fetch-only and --analyze-only are mutually exclusive")
    if args.date_to and not args.date_from:
        parser.error("--to requires --from")

    # Build date list: span + individual dates, deduped and sorted
    dates = list(args.dates)
    if args.date_from:
        d = datetime.strptime(args.date_from, "%Y-%m-%d")
        end = datetime.strptime(args.date_to, "%Y-%m-%d") if args.date_to else d
        while d <= end:
            dates.append(d.strftime("%Y-%m-%d"))
            d += timedelta(days=1)
    dates = sorted(set(dates))
    if not dates:
        parser.error("Provide at least one date (positional or via --from/--to)")
    t_wall = time.time()
    results = {}  # date → {"fetch": (rc, s), "analyze": (rc, s)}

    # -----------------------------------------------------------------------
    # Fetch phase — always parallel (network-bound)
    # -----------------------------------------------------------------------
    if not args.analyze_only:
        print(
            f"\nFetching {len(dates)} day(s), up to {args.parallel_days} at a time (--workers {args.workers} each)..."
        )
        with ThreadPoolExecutor(max_workers=args.parallel_days) as pool:
            futures = {pool.submit(fetch_day, d, args.workers): d for d in dates}
            for fut in as_completed(futures):
                d = futures[fut]
                rc, elapsed = fut.result()
                results.setdefault(d, {})["fetch"] = (rc, elapsed)

    # -----------------------------------------------------------------------
    # Analyze phase — sequential by default, parallel if --parallel-analyze > 1
    # -----------------------------------------------------------------------
    if not args.fetch_only:
        days_to_analyze = [
            d for d in dates if results.get(d, {}).get("fetch", (0,))[0] == 0
        ]
        skipped = len(dates) - len(days_to_analyze)
        if skipped:
            print(f"  Skipping {skipped} day(s) with failed fetch.")

        n_parallel = args.parallel_analyze
        mode = f"up to {n_parallel} at a time" if n_parallel > 1 else "sequentially"
        print(f"\nAnalyzing {len(days_to_analyze)} day(s) ({mode})...")

        if n_parallel == 1:
            for d in days_to_analyze:
                rc, elapsed = analyze_day(
                    d, duty_cycle=args.duty_cycle, figures=args.figures
                )
                results.setdefault(d, {})["analyze"] = (rc, elapsed)
        else:
            with ThreadPoolExecutor(max_workers=n_parallel) as pool:
                futures = {
                    pool.submit(analyze_day, d, args.duty_cycle, args.figures): d
                    for d in days_to_analyze
                }
                for fut in as_completed(futures):
                    d = futures[fut]
                    rc, elapsed = fut.result()
                    results.setdefault(d, {})["analyze"] = (rc, elapsed)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total = time.time() - t_wall
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY  ({total / 60:.1f} min total)")
    print(f"{'=' * 60}")
    print(f"  {'Date':<14} {'Fetch':>10} {'Analyze':>10}  Status")
    print(f"  {'-' * 50}")
    for d in dates:
        r = results.get(d, {})
        fetch_str = f"{r['fetch'][1] / 60:.1f} min" if "fetch" in r else "—"
        analyze_str = f"{r['analyze'][1] / 60:.1f} min" if "analyze" in r else "—"
        fetch_ok = r.get("fetch", (0,))[0] == 0
        analyze_ok = r.get("analyze", (0,))[0] == 0
        status = "OK" if (fetch_ok and analyze_ok) else "FAILED"
        print(f"  {d:<14} {fetch_str:>10} {analyze_str:>10}  {status}")
    print()


if __name__ == "__main__":
    main()
