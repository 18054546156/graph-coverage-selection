#!/usr/bin/env python3
"""Validate that one race method produced exactly the requested seed rows."""

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--expected-seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()

    path = args.output / "results.csv"
    if not path.exists():
        raise SystemExit(f"missing results file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    method_rows = [row for row in rows if row.get("base_method") == args.method]
    seeds = sorted(int(row["seed"]) for row in method_rows)
    expected = sorted(args.expected_seeds)
    if seeds != expected:
        raise SystemExit(
            f"{args.method}: expected seeds {expected}, found {seeds} in {path}"
        )
    for row in method_rows:
        value = float(row["balanced_accuracy"])
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{args.method}: invalid balanced_accuracy={value}")
    print(f"validated {args.method}: seeds={seeds}, rows={len(method_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
