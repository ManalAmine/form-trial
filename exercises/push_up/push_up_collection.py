"""Schema-safe CSV persistence for manually labelled push-up repetitions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

try:
    from .config import CSV_COLUMNS, ERROR_COLUMNS, METADATA_COLUMNS, TARGET_COLUMN
except ImportError:
    from config import CSV_COLUMNS, ERROR_COLUMNS, METADATA_COLUMNS, TARGET_COLUMN


def ensure_dataset(path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=CSV_COLUMNS).writeheader()
        return path
    with path.open("r", newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), [])
    if header != CSV_COLUMNS:
        raise ValueError("Push-up dataset schema/order mismatch; use a new file or explicitly migrate it.")
    return path


def sample_counts(path) -> tuple[int, int, dict[str, int]]:
    good = bad = 0
    errors = {name: 0 for name in ERROR_COLUMNS}
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            good += row.get(TARGET_COLUMN) == "1"
            bad += row.get(TARGET_COLUMN) == "0"
            for name in ERROR_COLUMNS:
                errors[name] += row.get(name) == "1"
    return int(good), int(bad), errors


def append_sample(path, metadata: Mapping, features: Mapping[str, float], is_good: int, selected_errors=None) -> None:
    selected_errors = set(selected_errors or set())
    unknown = selected_errors.difference(ERROR_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown push-up error labels: {sorted(unknown)}")
    if int(is_good) == 1 and selected_errors:
        raise ValueError("A GOOD repetition cannot contain error labels.")
    row = {name: metadata.get(name, "") for name in METADATA_COLUMNS}
    row.update(features)
    row.update({name: int(name in selected_errors) for name in ERROR_COLUMNS})
    row[TARGET_COLUMN] = int(is_good)
    missing = [name for name in CSV_COLUMNS if name not in row]
    if missing:
        raise ValueError(f"Cannot save push-up sample; missing columns: {missing}")
    with Path(path).open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writerow(row)
        handle.flush()


def undo_last_sample(path) -> bool:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) <= 1:
        return False
    rows.pop()
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
        handle.flush()
    return True
