"""
Data Access Module for OOI Hydrophone Data

This module provides a simple caching layer around ooipy to avoid
redundant downloads. It checks a local cache directory before
fetching from the OOI API.

Usage
-----
    from fin_whale_finder.data_access import get_hydrophone_data

    # First call - downloads from OOI
    data = get_hydrophone_data(
        node='Axial_Base',
        starttime=datetime(2024, 1, 1),
        endtime=datetime(2024, 1, 2)
    )

    # Second call - loads from cache (instant)
    data = get_hydrophone_data(
        node='Axial_Base',
        starttime=datetime(2024, 1, 1),
        endtime=datetime(2024, 1, 2)
    )

Supported Nodes (from ooipy)
---------------------------
Low frequency (200 Hz):
  - 'Axial_Base' or 'AXBA1' — RS03AXBS-MJ03A-05-HYDLFA301
  - 'Central_Caldera' or 'AXCC1' — RS03CCAL-MJ03F-06-HYDLFA305
  - 'Eastern_Caldera' or 'AXEC2' — RS03ECAL-MJ03E-09-HYDLFA304
  - 'Southern_Hydrate' or 'HYS14' — RS01SUM1-LJ01B-05-HYDLFA104
  - 'Slope_Base' or 'HYSB1' — RS01SLBS-MJ01A-05-HYDLFA101

Broadband (64 kHz):
  - 'Axial_Base_Seafloor' or 'LJ03A' — RS03AXBS-LJ03A-09-HYDBBA302
  - 'Oregon_Shelf_Base_Seafloor' or 'LJ01D' — CE02SHBP-LJ01D-11-HYDBBA106
  - 'Oregon_Slope_Base_Seafloor' or 'LJ01A' — RS01SLBS-LJ01A-09-HYDBBA102
  - 'Oregon_Slope_Base_Shallow' or 'PC01A' — RS01SBPS-PC01A-08-HYDBBA103
  - 'Axial_Base_Shallow' or 'PC03A' — RS03AXPS-PC03A-08-HYDBBA303
  - 'Oregon_Offshore_Base_Seafloor' or 'LJ01C' — CE04OSBP-LJ01C-11-HYDBBA105

Cache Location
--------------
Data is cached in the `data/cache/` directory (relative to project root).
Each file is named: `{node}_{start}_{end}.pkl`
Cache can be cleared by deleting files in `data/cache/`.

"""

import logging
import pickle
from datetime import datetime
from pathlib import Path

import ooipy  # [C: this is not used... do we need it?]
from ooipy import get_acoustic_data, get_acoustic_data_LF
from ooipy.hydrophone.basic import HydrophoneData

logger = logging.getLogger(__name__)

# Default cache directory - can be overridden
DEFAULT_CACHE_DIR = Path("data/cache")


# ============================================================================
# Caching Functions
# ============================================================================


def _make_cache_key(node: str, starttime: datetime, endtime: datetime) -> str:
    """
    Create a filename-safe cache key from request parameters. [C: What does this mean? what problem does this solve?]

    Example: 'Axial_Base_2024-01-01_2024-01-02.pkl'
    """
    start_str = starttime.strftime("%Y-%m-%d_%H-%M-%S")
    end_str = endtime.strftime("%Y-%m-%d_%H-%M-%S")
    # Clean node name for filesystem
    node_clean = node.replace(" ", "_").replace("/", "_")
    return f"{node_clean}_{start_str}_{end_str}.pkl"


def get_hydrophone_data(
    node: str,
    starttime: datetime,
    endtime: datetime,
    low_freq: bool = True,
    cache_dir: Path | str | None = None,
    verbose: bool = False,
    **ooipy_kwargs,
) -> HydrophoneData | None:
    """
    Fetch hydrophone data with local caching.

    If the requested data exists in the cache, it is loaded instantly.
    Otherwise, it is downloaded from OOI and saved to the cache.

    Parameters
    ----------
    node : str
        Hydrophone node name (e.g., 'Axial_Base', 'Central_Caldera').
        See module docstring for supported nodes.
    starttime : datetime
        Start of the time window.
    endtime : datetime
        End of the time window.
    low_freq : bool
        If True (default), use low-frequency (200 Hz) data.
        If False, use broadband (64 kHz) data.
    cache_dir : Path, str, or None
        Directory to store cached data. Defaults to `data/cache/`.
        Set to False to disable caching entirely.
    verbose : bool
        Print progress messages.
    **ooipy_kwargs
        Additional arguments passed to ooipy:
        - fmin, fmax: bandpass filter corners
        - zero_mean: remove mean from data
        - channel: for LF data ('HDH', 'HNE', 'HNN', 'HNZ')
        - correct: apply IRIS calibration
        - merge_traces: merge multiple traces

    Returns
    -------
    HydrophoneData or None
        The hydrophone data, or None if no data is available
        for the specified time window.

    Examples
    --------
    >>> from datetime import datetime
    >>> from fin_whale_finder.data_access import get_hydrophone_data
    >>>
    >>> # Low-frequency (200 Hz) - default
    >>> data = get_hydrophone_data(
    ...     node='Axial_Base',
    ...     starttime=datetime(2024, 1, 1),
    ...     endtime=datetime(2024, 1, 2),
    ...     verbose=True
    ... )
    >>>
    >>> # Broadband (64 kHz)
    >>> data = get_hydrophone_data(
    ...     node='Axial_Base_Seafloor',
    ...     starttime=datetime(2024, 1, 1),
    ...     endtime=datetime(2024, 1, 2),
    ...     low_freq=False,
    ...     verbose=True
    ... )
    """
    # Handle cache directory
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    use_cache = cache_dir is not False
    cache_file = None

    if use_cache:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_file = cache_path / _make_cache_key(node, starttime, endtime)

        if cache_file.exists():
            if verbose:
                print(f"Loading from cache: {cache_file}")
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                if verbose:
                    print(f"Cache read failed: {e}, re-downloading...")

    # Fetch from OOI
    if verbose:
        print(f"Downloading from OOI: {node} {starttime} -> {endtime}")

    if low_freq:
        data = get_acoustic_data_LF(
            starttime,
            endtime,
            node,
            verbose=verbose,
            merge_traces=True,
            **ooipy_kwargs,
        )
    else:
        data = get_acoustic_data(
            starttime,
            endtime,
            node,
            verbose=verbose,
            **ooipy_kwargs,
        )

    # Cache the result if we got data
    if data is not None and use_cache and cache_file is not None:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f)
            if verbose:
                print(f"Saved to cache: {cache_file}")
        except Exception as e:
            if verbose:
                print(f"Cache write failed: {e}")

    return data


def clear_cache(cache_dir: Path | str | None = None) -> int:
    """
    Clear all cached data files.

    Parameters
    ----------
    cache_dir : Path, str, or None
        Directory to clear. Defaults to `data/cache/`.

    Returns
    -------
    int
        Number of files deleted.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        return 0

    files = list(cache_path.glob("*.pkl"))
    for f in files:
        f.unlink()

    return len(files)


def cache_info(cache_dir: Path | str | None = None) -> dict:
    """
    Get information about the cache.

    Parameters
    ----------
    cache_dir : Path, str, or None
        Directory to inspect. Defaults to `data/cache/`.

    Returns
    -------
    dict
        Dictionary with 'file_count', 'total_size_mb', and 'files' keys.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        return {"file_count": 0, "total_size_mb": 0.0, "files": []}

    files = sorted(cache_path.glob("*.pkl"))
    total_size = sum(f.stat().st_size for f in files)

    return {
        "file_count": len(files),
        "total_size_mb": total_size / (1024 * 1024),
        "files": [f.name for f in files],
    }


__all__ = [
    "get_hydrophone_data",
    "clear_cache",
    "cache_info",
    "HydrophoneData",
]
