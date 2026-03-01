#!/usr/bin/env python
"""
Fetch broadband hydrophone data from OOI and cache locally.

Downloads data in 1-hour chunks, skipping any that are already cached.
Chunks are downloaded in parallel for speed — OOI is the bottleneck,
not local CPU.

Usage
-----
    uv run python scripts/fetch_broadband.py

    # Different time range
    uv run python scripts/fetch_broadband.py --start 2026-01-04T00:00:00 --end 2026-01-04T06:00:00

    # Limit parallelism if OOI throttles
    uv run python scripts/fetch_broadband.py --workers 2

Output
------
    data/cache_broadband/<node>_<start>_<end>.pkl  (one file per chunk)

These files are read by notebooks/soundscape_explore.py.
"""

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

from fin_whale_finder.data_access import get_hydrophone_data, _make_cache_key

# --- Defaults (match notebooks/soundscape_explore.py config) ---
DEFAULT_NODE = "Axial_Base_Seafloor"
DEFAULT_START = "2026-01-03T12:00:00"
DEFAULT_END = "2026-01-03T18:00:00"
DEFAULT_CHUNK_HOURS = 1
DEFAULT_WORKERS = 6
CACHE_DIR = Path("data/cache_broadband")

print_lock = threading.Lock()


def log(msg):
    with print_lock:
        print(msg, flush=True)


def fetch_chunk(node, start, end, cache_dir, chunk_num, n_total):
    """Download one chunk, printing start/done with timing and file size."""
    label = f"{start.strftime('%H:%M')}–{end.strftime('%H:%M')}"
    cache_file = cache_dir / _make_cache_key(node, start, end)

    if cache_file.exists():
        size_gb = cache_file.stat().st_size / 1e9
        log(f"  [CACHE ] {label}  ({size_gb:.2f} GB already cached)")
        return start, end, "cached"

    log(f"  [START ] {label}  (chunk {chunk_num}/{n_total})")
    t0 = time.time()

    data = get_hydrophone_data(
        node=node,
        starttime=start,
        endtime=end,
        low_freq=False,
        cache_dir=cache_dir,
        verbose=False,
    )

    elapsed = (time.time() - t0) / 60
    if data is None:
        log(f"  [EMPTY ] {label}  ({elapsed:.1f} min — no data returned)")
        return start, end, "empty"

    size_gb = cache_file.stat().st_size / 1e9 if cache_file.exists() else 0
    log(f"  [DONE  ] {label}  ({elapsed:.1f} min, {size_gb:.2f} GB)")
    return start, end, "downloaded"


def size_monitor(cache_dir, chunks, node, stop_event, interval=30):
    """Background thread: print total cached size every `interval` seconds."""
    while not stop_event.wait(interval):
        files = [cache_dir / _make_cache_key(node, s, e) for s, e in chunks]
        present = [f for f in files if f.exists()]
        total_gb = sum(f.stat().st_size for f in present) / 1e9
        log(f"  [STATUS] {len(present)}/{len(chunks)} files on disk, {total_gb:.2f} GB total")


def main():
    parser = argparse.ArgumentParser(description="Fetch broadband OOI hydrophone data")
    parser.add_argument("--node", default=DEFAULT_NODE)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--chunk-hours", type=int, default=DEFAULT_CHUNK_HOURS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Parallel download workers (default 6, lower if OOI throttles)")
    parser.add_argument("--status-interval", type=int, default=30,
                        help="Seconds between status updates (default 30)")
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Build list of chunks
    chunks = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(hours=args.chunk_hours), end)
        chunks.append((current, chunk_end))
        current = chunk_end

    n_cached = sum(
        1 for s, e in chunks
        if (CACHE_DIR / _make_cache_key(args.node, s, e)).exists()
    )
    n_to_fetch = len(chunks) - n_cached

    print(f"Node:    {args.node}")
    print(f"Range:   {args.start} → {args.end}")
    print(f"Chunks:  {len(chunks)} × {args.chunk_hours}h  ({n_cached} cached, {n_to_fetch} to download)")
    print(f"Workers: {min(args.workers, n_to_fetch) if n_to_fetch else 0}")
    print()

    if n_to_fetch == 0:
        print("All chunks already cached. Nothing to do.")
        return

    # Start background status thread
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=size_monitor,
        args=(CACHE_DIR, chunks, args.node, stop_event, args.status_interval),
        daemon=True,
    )
    monitor.start()

    results = {}
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(fetch_chunk, args.node, s, e, CACHE_DIR, i + 1, len(chunks)): (s, e)
                for i, (s, e) in enumerate(chunks)
            }
            n_done = 0
            for future in as_completed(futures):
                s, e, status = future.result()
                results[(s, e)] = status
                if status != "cached":
                    n_done += 1
                    n_remaining = n_to_fetch - n_done
                    log(f"  [PROGRESS] {n_done}/{n_to_fetch} downloads complete"
                        + (f", {n_remaining} remaining" if n_remaining else " — all done!"))
    finally:
        stop_event.set()

    n_ok = sum(1 for v in results.values() if v in ("cached", "downloaded"))
    n_empty = sum(1 for v in results.values() if v == "empty")
    print()
    print(f"Done. {n_ok}/{len(chunks)} chunks available, {n_empty} returned no data.")
    if n_empty:
        print("Empty chunks are normal for gaps in OOI coverage.")


if __name__ == "__main__":
    main()
