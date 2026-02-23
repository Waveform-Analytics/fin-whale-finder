#!/usr/bin/env python
"""
Fetch a week of hydrophone data from OOI and save as a combined dataset.

Usage
-----
    uv run python scripts/fetch_week_data.py

"""

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

from fin_whale_finder.data_access import get_hydrophone_data, _make_cache_key


def _is_cached(node: str, starttime: datetime, endtime: datetime) -> bool:
    cache_key = _make_cache_key(node, starttime, endtime)
    cache_path = Path("data/cache") / cache_key
    return cache_path.exists()


def main():
    node = "Axial_Base"
    year = 2026
    month = 1
    start_day = 1
    end_day = 8  # exclusive (fetches Jan 1-7)

    print(f"Fetching {node} data: Jan {start_day} - {end_day - 1}, {year}")
    print()

    days = list(range(start_day, end_day))
    cached_count = 0
    downloaded_count = 0

    all_data = []
    all_times = []

    for day in days:
        start = datetime(year, month, day)
        end = (
            datetime(year, month, day + 1)
            if day < end_day - 1
            else datetime(year, month, end_day)
        )

        if _is_cached(node, start, end):
            print(f"[CACHE] Jan {day}")
            cached_count += 1
        else:
            print(f"[M2M  ] Jan {day}")
            downloaded_count += 1

        data = get_hydrophone_data(
            node=node,
            starttime=start,
            endtime=end,
            verbose=False,
        )

        if data:
            offset = (day - start_day) * 86400
            all_data.append(data.data)
            all_times.append(data.times() + offset)
        else:
            print(f"[EMPTY] Warning: No data for Jan {day}")

    print()
    print(f"Summary: {cached_count} from cache, {downloaded_count} downloaded")

    if not all_data:
        print("No data fetched!")
        return

    full_data = np.concatenate(all_data)
    full_times = np.concatenate(all_times)

    print(f"\nTotal samples: {len(full_data):,}")
    print(f"Total hours: {len(full_data) / 200 / 3600:.1f}")
    print(f"Time span: {(full_times[-1] - full_times[0]) / 3600:.1f} hours")

    output_file = f"data/cache/{node}_{year}-{month:02d}-{start_day:02d}_to_{year}-{month:02d}-{end_day - 1:02d}.pkl"
    with open(output_file, "wb") as f:
        pickle.dump({"data": full_data, "times": full_times, "sampling_rate": 200}, f)

    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
