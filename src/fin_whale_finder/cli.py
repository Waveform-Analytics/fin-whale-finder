from __future__ import annotations

import argparse
from pathlib import Path

from fin_whale_finder.config import load_config
from fin_whale_finder.manifest import ManifestParams, parse_iso8601, write_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fwf",
        description="Fin Whale Finder command line tools",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    manifest_parser = sub.add_parser("manifest", help="Manifest operations")
    manifest_sub = manifest_parser.add_subparsers(
        dest="manifest_command", required=True
    )

    create = manifest_sub.add_parser("create", help="Create a time-slice manifest CSV")
    create.add_argument("--station", help="OOI station id")
    create.add_argument("--start", help="ISO-8601 start datetime")
    create.add_argument("--end", help="ISO-8601 end datetime")
    create.add_argument("--window-minutes", type=int, help="Window size in minutes")
    create.add_argument("--out", type=Path, help="Output CSV path")
    create.add_argument(
        "--config",
        type=Path,
        help="Optional TOML config path (fields can be overridden by flags)",
    )

    return parser


def _params_from_args(args: argparse.Namespace) -> ManifestParams:
    config_manifest = None
    if args.config:
        config_manifest = load_config(args.config).manifest

    station = args.station or (config_manifest.station if config_manifest else None)
    start_raw = args.start or (
        config_manifest.start.isoformat().replace("+00:00", "Z")
        if config_manifest
        else None
    )
    end_raw = args.end or (
        config_manifest.end.isoformat().replace("+00:00", "Z")
        if config_manifest
        else None
    )
    window_minutes = args.window_minutes or (
        config_manifest.window_minutes if config_manifest else None
    )
    out = args.out or (config_manifest.output_path if config_manifest else None)

    missing = [
        key
        for key, value in {
            "station": station,
            "start": start_raw,
            "end": end_raw,
            "window_minutes": window_minutes,
            "out": out,
        }.items()
        if value is None
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"Missing required inputs: {missing_text}. Supply flags or --config."
        )

    return ManifestParams(
        station=station,
        start=parse_iso8601(start_raw),
        end=parse_iso8601(end_raw),
        window_minutes=window_minutes,
        output_path=Path(out),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "manifest" and args.manifest_command == "create":
        params = _params_from_args(args)
        row_count = write_manifest(params)
        print(f"Wrote {row_count} rows to {params.output_path}")
        return

    parser.error("Unsupported command")


if __name__ == "__main__":
    main()
