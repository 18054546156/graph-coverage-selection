#!/usr/bin/env python3
"""Aggregate frozen-index downstream results into a Table 1-shaped report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any


METHODS = [
    ("random", "Random"),
    ("el2n", "EL2N"),
    ("forgetting", "Forgetting"),
    ("eva", "EVA"),
    ("facility", "Facility"),
    ("fps", "FPS"),
    ("herding", "Herding"),
    ("graph_a2", "Graph-A2/Ours"),
]


def ratio_key(ratio: float) -> str:
    return f"r{ratio:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def fmt(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "--"
    return f"{value * 100:.{digits}f}"


def load_results(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.glob("**/*_result.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["dataset"]), float(row["ratio"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)
    output = []
    for (dataset, ratio, method), values in sorted(grouped.items()):
        ba = [float(value["balanced_accuracy"]) for value in values]
        acc = [float(value["accuracy"]) for value in values]
        worst = [float(value["worst_class_recall"]) for value in values]
        cvar = [float(value["class_cvar20"]) for value in values]
        output.append(
            {
                "dataset": dataset,
                "ratio": ratio,
                "method": method,
                "n_seeds": len(values),
                "training_seeds": ",".join(str(value["training_seed"]) for value in values),
                "balanced_accuracy_mean": statistics.mean(ba),
                "balanced_accuracy_std": statistics.stdev(ba) if len(ba) > 1 else 0.0,
                "accuracy_mean": statistics.mean(acc),
                "accuracy_std": statistics.stdev(acc) if len(acc) > 1 else 0.0,
                "worst_class_recall_mean": statistics.mean(worst),
                "worst_class_recall_std": statistics.stdev(worst) if len(worst) > 1 else 0.0,
                "class_cvar20_mean": statistics.mean(cvar),
                "class_cvar20_std": statistics.stdev(cvar) if len(cvar) > 1 else 0.0,
            }
        )
    return output


def table_markdown(rows: list[dict[str, Any]]) -> str:
    lookup = {
        (row["dataset"], float(row["ratio"]), row["method"]): row for row in rows
    }
    datasets = sorted({row["dataset"] for row in rows})
    ratios = sorted({float(row["ratio"]) for row in rows})
    lines = [
        "# Table 1 Reproduction",
        "",
        "Values are test balanced accuracy (%) mean +/- sample std over the available training seeds.",
        "A missing cell means that Job 2 has not produced all or any result for that condition.",
        "",
        "| Dataset | Ratio | Random | EL2N | Forgetting | EVA | Facility | FPS | Herding | Graph-A2/Ours |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in datasets:
        for ratio in ratios:
            cells = [dataset, f"{ratio * 100:.0f}%"]
            for method, _display in METHODS:
                row = lookup.get((dataset, ratio, method))
                if row is None:
                    cells.append("--")
                else:
                    cells.append(
                        f"{fmt(row['balanced_accuracy_mean'])} +/- {fmt(row['balanced_accuracy_std'])}"
                    )
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "ratio", "method", "n_seeds", "training_seeds",
        "balanced_accuracy_mean", "balanced_accuracy_std",
        "accuracy_mean", "accuracy_std",
        "worst_class_recall_mean", "worst_class_recall_std",
        "class_cvar20_mean", "class_cvar20_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output_dir = (args.output_dir or input_root / "summary").resolve()
    raw = load_results(input_root)
    rows = aggregate(raw)
    write_csv(output_dir / "per_condition_mean_std.csv", rows)
    (output_dir / "table1_reproduction.md").write_text(
        table_markdown(rows), encoding="utf-8"
    )
    (output_dir / "raw_result_count.json").write_text(
        json.dumps({"raw_result_files": len(raw), "conditions": len(rows)}, indent=2),
        encoding="utf-8",
    )
    print(f"raw result files: {len(raw)}")
    print(f"aggregated conditions: {len(rows)}")
    print(f"table: {output_dir / 'table1_reproduction.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
