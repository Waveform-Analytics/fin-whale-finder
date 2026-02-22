from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path


@dataclass(frozen=True)
class ManifestParams:
    station: str
    start: datetime
    end: datetime
    window_minutes: int
    output_path: Path


def parse_iso8601(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def build_rows(params: ManifestParams) -> list[dict[str, str]]:
    if params.window_minutes < 1:
        raise ValueError("window_minutes must be >= 1")
    if params.end <= params.start:
        raise ValueError("end must be after start")

    rows: list[dict[str, str]] = []
    current = params.start
    step = timedelta(minutes=params.window_minutes)

    while current < params.end:
        window_end = min(current + step, params.end)
        start_s = current.isoformat().replace("+00:00", "Z")
        end_s = window_end.isoformat().replace("+00:00", "Z")
        rows.append(
            {
                "station": params.station,
                "window_start": start_s,
                "window_end": end_s,
                "asset_hint": (
                    f"ooi://{params.station}/{current.strftime('%Y/%m/%d/%H%M%S')}.mseed"
                ),
            }
        )
        current = window_end

    return rows


def write_manifest(params: ManifestParams) -> int:
    rows = build_rows(params)
    params.output_path.parent.mkdir(parents=True, exist_ok=True)

    header = "station,window_start,window_end,asset_hint\n"
    lines = [
        f"{row['station']},{row['window_start']},{row['window_end']},{row['asset_hint']}\n"
        for row in rows
    ]
    params.output_path.write_text(header + "".join(lines), encoding="utf-8")
    return len(rows)
