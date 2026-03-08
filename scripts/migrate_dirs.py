#!/usr/bin/env python
"""
One-time migration: move slug-only result dirs under Axial_Base_Seafloor/.

Before (old layout):
  data/wav_broadband/2025-06-04_00h-00h/
  data/embeddings/2025-06-04_00h-00h/
  results/soundscape_figures/2025-06-04_00h-00h/

After (new layout):
  data/wav_broadband/Axial_Base_Seafloor/2025-06-04_00h-00h/
  data/embeddings/Axial_Base_Seafloor/2025-06-04_00h-00h/
  results/soundscape_figures/Axial_Base_Seafloor/2025-06-04_00h-00h/

Stray WAV files at data/wav_broadband/*.wav are moved into
  data/wav_broadband/Axial_Base_Seafloor/<slug>/ based on their filename date.

Usage:
  uv run python scripts/migrate_dirs.py          # dry run (prints actions only)
  uv run python scripts/migrate_dirs.py --apply  # execute moves
"""

import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
NODE = "Axial_Base_Seafloor"

# Slug pattern: YYYY-MM-DD_HHh-HHh
SLUG_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}h-\d{2}h$")

# WAV filename pattern: YYYYMMDDTHHMMSS.wav  (OOI naming)
WAV_DATE_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})T")


def slug_from_wav_name(name: str) -> str | None:
    """Derive a slug like 2025-06-04_00h-00h from an OOI WAV filename."""
    m = WAV_DATE_RE.match(name)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}_00h-00h"


def migrate_dir(src: Path, dst: Path, apply: bool) -> None:
    if not src.exists():
        return
    if dst.exists():
        print(f"  SKIP (dst exists): {src} -> {dst}")
        return
    print(f"  MOVE: {src.relative_to(PROJECT_ROOT)}  ->  {dst.relative_to(PROJECT_ROOT)}")
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def migrate_stray_wavs(wav_root: Path, node_dir: Path, apply: bool) -> None:
    stray = list(wav_root.glob("*.wav"))
    if not stray:
        return
    print(f"\nStray WAVs in {wav_root.relative_to(PROJECT_ROOT)}:")
    for wav in sorted(stray):
        slug = slug_from_wav_name(wav.name)
        if slug is None:
            print(f"  SKIP (unrecognized name): {wav.name}")
            continue
        dst = node_dir / slug / wav.name
        print(f"  MOVE: {wav.relative_to(PROJECT_ROOT)}  ->  {dst.relative_to(PROJECT_ROOT)}")
        if apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(wav), str(dst))


def main() -> None:
    apply = "--apply" in sys.argv
    mode = "APPLY" if apply else "DRY RUN"
    print(f"=== migrate_dirs.py ({mode}) ===")
    if not apply:
        print("Pass --apply to execute moves.\n")

    roots = [
        PROJECT_ROOT / "data" / "wav_broadband",
        PROJECT_ROOT / "data" / "embeddings",
        PROJECT_ROOT / "results" / "soundscape_figures",
    ]

    for root in roots:
        if not root.exists():
            continue
        node_dir = root / NODE
        slug_dirs = [d for d in root.iterdir() if d.is_dir() and SLUG_RE.match(d.name)]
        if slug_dirs:
            print(f"\n{root.relative_to(PROJECT_ROOT)}:")
            for src in sorted(slug_dirs):
                migrate_dir(src, node_dir / src.name, apply)

    # Stray WAVs sitting directly in wav_broadband/
    wav_root = PROJECT_ROOT / "data" / "wav_broadband"
    if wav_root.exists():
        migrate_stray_wavs(wav_root, wav_root / NODE, apply)

    print("\nDone." if apply else "\nDry run complete. Re-run with --apply to execute.")


if __name__ == "__main__":
    main()
