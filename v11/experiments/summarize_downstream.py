#!/usr/bin/env python3
"""Aggregate completed v11 Job-2 result files into CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("accuracy", "balanced_accuracy", "worst_class_recall", "class_cvar20")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_root
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(args.input_root.rglob("*_result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["result_path"] = str(path)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no result JSON files below {args.input_root}")

    raw_fields = [
        "dataset",
        "ratio",
        "variant",
        "training_seed",
        "evaluation_split",
        *METRICS,
        "best_balanced_accuracy",
        "elapsed_seconds",
        "result_path",
    ]
    with (output_dir / "raw_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    groups: dict[tuple[str, float, str, str], list[dict]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            float(row["ratio"]),
            str(row["variant"]),
            str(row["evaluation_split"]),
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, values in sorted(groups.items()):
        summary = {
            "dataset": key[0],
            "ratio": key[1],
            "variant": key[2],
            "evaluation_split": key[3],
            "n_seeds": len(values),
            "seeds": ",".join(str(value["training_seed"]) for value in values),
        }
        for metric in METRICS:
            observations = np.asarray([float(value[metric]) for value in values])
            summary[f"{metric}_mean"] = float(np.mean(observations))
            summary[f"{metric}_std"] = float(np.std(observations, ddof=1)) if len(values) > 1 else 0.0
        summaries.append(summary)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        "# v11 downstream summary",
        "",
        "All values are final-split percentages. Best-epoch BA is intentionally not summarized.",
        "",
        "| Dataset | Ratio | Variant | Seeds | BA | Worst recall | CVaR20 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['dataset']} | {100 * row['ratio']:.0f}% | {row['variant']} | "
            f"{row['n_seeds']} | {100 * row['balanced_accuracy_mean']:.2f} +/- "
            f"{100 * row['balanced_accuracy_std']:.2f} | "
            f"{100 * row['worst_class_recall_mean']:.2f} +/- "
            f"{100 * row['worst_class_recall_std']:.2f} | "
            f"{100 * row['class_cvar20_mean']:.2f} +/- "
            f"{100 * row['class_cvar20_std']:.2f} |"
        )
    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_dir / "SUMMARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
