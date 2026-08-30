#!/usr/bin/env python3
"""Compare raw race results against Graph-A2 by matching seed."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_rows(root: Path):
    rows = []
    for path in sorted(root.glob("*/results.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["source"] = str(path)
                row["seed"] = int(row["seed"])
                row["balanced_accuracy"] = 100.0 * float(row["balanced_accuracy"])
                rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()

    by_method = defaultdict(dict)
    for row in load_rows(args.results_dir):
        by_method[row["base_method"]][row["seed"]] = row["balanced_accuracy"]

    baseline = by_method.get("graph_a2", {})
    if not baseline:
        raise SystemExit("graph_a2 baseline results are missing")

    print("method,seed42,seed43,mean_ba,paired_delta_pp,status")
    baseline_mean = sum(baseline.values()) / len(baseline)
    for method in sorted(by_method):
        values = by_method[method]
        common = sorted(set(values) & set(baseline))
        mean_ba = sum(values.values()) / len(values)
        delta = sum(values[s] - baseline[s] for s in common) / len(common)
        status = "complete" if sorted(values) == [42, 43] else "incomplete"
        print(
            f"{method},{values.get(42, float('nan')):.4f},"
            f"{values.get(43, float('nan')):.4f},{mean_ba:.4f},"
            f"{delta:.4f},{status}"
        )
    print(f"baseline_mean={baseline_mean:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
