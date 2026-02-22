from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from fin_whale_finder.manifest import ManifestParams, parse_iso8601


@dataclass(frozen=True)
class AppConfig:
    manifest: ManifestParams


def _required_string(mapping: dict, key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required string key: {key}")
    return value.strip()


def _required_int(mapping: dict, key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Missing required integer key: {key}")
    return value


def load_config(path: Path) -> AppConfig:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    manifest_block = payload.get("manifest")
    if not isinstance(manifest_block, dict):
        raise ValueError("Missing [manifest] section")

    manifest = ManifestParams(
        station=_required_string(manifest_block, "station"),
        start=parse_iso8601(_required_string(manifest_block, "start")),
        end=parse_iso8601(_required_string(manifest_block, "end")),
        window_minutes=_required_int(manifest_block, "window_minutes"),
        output_path=Path(_required_string(manifest_block, "output_path")),
    )
    return AppConfig(manifest=manifest)
