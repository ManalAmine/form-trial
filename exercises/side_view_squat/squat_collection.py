"""CSV persistence helpers for manually labelled squat repetitions."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Mapping

try:
    from .config import CSV_COLUMNS, ERROR_COLUMNS, METADATA_COLUMNS, TARGET_COLUMN
except ImportError:
    from config import CSV_COLUMNS, ERROR_COLUMNS, METADATA_COLUMNS, TARGET_COLUMN


def ensure_dataset(path: str | os.PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=CSV_COLUMNS).writeheader()
        return path
    with path.open("r", newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), [])
    if header != CSV_COLUMNS:
        raise ValueError(
            f"Dataset schema mismatch in {path}. Use a new file or migrate it; "
            "the collector will not silently reorder existing data."
        )
    return path


def sample_counts(path: str | os.PathLike) -> tuple[int, int]:
    good = bad = 0
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get(TARGET_COLUMN) == "1":
                good += 1
            elif row.get(TARGET_COLUMN) == "0":
                bad += 1
    return good, bad


def append_sample(
    path: str | os.PathLike,
    metadata: Mapping,
    features: Mapping[str, float],
    is_good: int,
    selected_errors: set[str] | None = None,
) -> None:
    selected_errors = selected_errors or set()
    row = {name: metadata.get(name, "") for name in METADATA_COLUMNS}
    row.update(features)
    row.update({name: int(name in selected_errors) for name in ERROR_COLUMNS})
    row[TARGET_COLUMN] = int(bool(is_good))
    missing = [name for name in CSV_COLUMNS if name not in row]
    if missing:
        raise ValueError(f"Cannot save squat sample; missing columns: {missing}")
    with Path(path).open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=CSV_COLUMNS).writerow(row)


def undo_last_sample(path: str | os.PathLike) -> bool:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) <= 1:
        return False
    rows.pop()
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    return True
