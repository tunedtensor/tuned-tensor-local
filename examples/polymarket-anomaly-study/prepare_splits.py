#!/usr/bin/env python3
"""Prepare deterministic, purged TT Local Study splits from collector CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path


OUTPUT_COLUMNS = [
    "id",
    "observed_at",
    "future_observed_at",
    "spread_bps",
    "liquidity",
    "volume_24h",
    "bid_depth_3c",
    "ask_depth_3c",
    "imbalance_3c",
    "big_move",
]
NUMERIC_COLUMNS = OUTPUT_COLUMNS[3:-1]


def utc(value: str) -> datetime:
    if not value.endswith("Z"):
        raise ValueError(f"Expected an RFC 3339 UTC timestamp ending in Z: {value}")
    return datetime.fromisoformat(value[:-1] + "+00:00")


def sample_rows(
    rows: list[dict[str, str]],
    *,
    split: str,
    maximum: int,
    seed: int,
) -> list[dict[str, str]]:
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}:{split}:{row['id']}".encode()
        ).digest(),
    )
    selected = ranked[:maximum]
    selected.sort(key=lambda row: (row["observed_at"], row["id"]))
    labels = Counter(row["big_move"] for row in selected)
    if labels.keys() != {"0", "1"}:
        raise ValueError(
            f"{split} must contain both labels after sampling; found {dict(labels)}"
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--training-end", required=True)
    parser.add_argument("--validation-start", required=True)
    parser.add_argument("--validation-end", required=True)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--max-rows-per-split", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_rows_per_split < 2:
        raise ValueError("--max-rows-per-split must be at least 2")

    training_end = utc(args.training_end)
    validation_start = utc(args.validation_start)
    validation_end = utc(args.validation_end)
    test_start = utc(args.test_start)
    if not training_end < validation_start <= validation_end < test_start:
        raise ValueError("Split boundaries must be strictly ordered")

    buckets: dict[str, list[dict[str, str]]] = {
        "training": [],
        "validation": [],
        "test": [],
    }
    seen_ids: set[str] = set()
    with args.input_csv.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = set(OUTPUT_COLUMNS + ["has_two_sided_book", "mid"]) - set(
            reader.fieldnames or []
        )
        if missing:
            raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
        for row in reader:
            if row["big_move"] not in {"0", "1"}:
                continue
            if row["has_two_sided_book"] != "1":
                continue
            midpoint = float(row["mid"])
            if not math.isfinite(midpoint) or not 0.02 <= midpoint <= 0.98:
                continue
            if not row["id"] or row["id"] in seen_ids:
                raise ValueError(f"Duplicate or blank row ID: {row['id']!r}")
            seen_ids.add(row["id"])
            for column in NUMERIC_COLUMNS:
                if not row[column].strip():
                    continue
                value = float(row[column])
                if not math.isfinite(value):
                    raise ValueError(
                        f"Row {row['id']!r} has non-finite {column}: {row[column]!r}"
                    )
            observed_at = utc(row["observed_at"])
            utc(row["future_observed_at"])
            projected = {column: row[column] for column in OUTPUT_COLUMNS}
            if observed_at <= training_end:
                buckets["training"].append(projected)
            elif validation_start <= observed_at <= validation_end:
                buckets["validation"].append(projected)
            elif observed_at >= test_start:
                buckets["test"].append(projected)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "source": str(args.input_csv.resolve()),
        "seed": args.seed,
        "max_rows_per_split": args.max_rows_per_split,
        "splits": {},
    }
    for split, candidates in buckets.items():
        selected = sample_rows(
            candidates,
            split=split,
            maximum=args.max_rows_per_split,
            seed=args.seed,
        )
        output_path = args.output_directory / f"{split}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(selected)
        labels = Counter(row["big_move"] for row in selected)
        summary["splits"][split] = {
            "path": str(output_path.resolve()),
            "rows": len(selected),
            "positive_rows": labels["1"],
            "positive_rate": labels["1"] / len(selected),
            "first_observed_at": selected[0]["observed_at"],
            "last_observed_at": selected[-1]["observed_at"],
        }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
