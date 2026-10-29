"""Helpers for the physical test: photo metadata and per-condition summaries.

A photo shoot is described by a CSV with a ``photo`` column (file name, or path
relative to the photo folder) plus any condition columns, e.g.::

    photo,size_in,distance_cm,angle_deg,lighting
    photo-101500-123.jpg,3,50,0,indoor
    photo-101512-456.jpg,3,100,30,indoor

``adai-sticker eval-photos --meta shoot.csv`` then reports success per condition.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "photo" not in reader.fieldnames:
            raise ValueError(f"{path} needs a 'photo' column.")
        return {row["photo"]: {k: v for k, v in row.items() if k != "photo"} for row in reader}


def summarize_by_condition(rows: list[dict], columns: list[str]) -> list[str]:
    """Plain-text lines: success rate for each value of each condition column."""
    lines = []
    for col in columns:
        groups: dict[str, list[bool]] = defaultdict(list)
        for r in rows:
            if col in r and r[col] != "":
                groups[str(r[col])].append(bool(r["target_top1"]))
        if not groups:
            continue
        lines.append(f"  by {col}:")
        for value, hits in sorted(groups.items(), key=lambda kv: _sort_key(kv[0])):
            lines.append(f"    {value:>12}  {sum(hits) / len(hits):5.0%}  (n={len(hits)})")
    return lines


def _sort_key(value: str):
    try:
        return (0, float(value), "")
    except ValueError:
        return (1, 0.0, value)
