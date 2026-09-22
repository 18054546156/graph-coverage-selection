#!/usr/bin/env python3
"""Aggregate the five fixed-training-seed Random selection arms."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import statistics


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    shutil.move(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifest = json.loads((args.artifacts / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for seed in protocol["selection"]["selection_seeds"]:
        arm = f"random_seed{int(seed)}"
        path = args.results / arm / "metrics.json"
        metric = json.loads(path.read_text(encoding="utf-8"))
        if metric.get("complete") is not True or int(metric["selection_seed"]) != int(seed):
            raise ValueError(f"invalid metrics: {path}")
        rows.append({
            "selection_seed": int(seed),
            "final_ba_pct": 100.0 * float(metric["balanced_accuracy"]),
            "best_ba_audit_pct": 100.0 * float(metric["best_balanced_accuracy_audit_only"]),
            "best_epoch_audit_only": int(metric["best_epoch_audit_only"]),
            "canonical_index_sha256": metric["canonical_index_sha256"],
            "jaccard_to_seed42": float(manifest["pairwise_jaccard"]["42"][str(int(seed))]),
        })
    values = [row["final_ba_pct"] for row in rows]
    ref = float(protocol["graph_a2_reference"]["final_ba_pct"])
    mean = statistics.mean(values)
    summary = {
        "schema": "graphcov-table1/path5-random-selection-seed-sweep-summary-v1",
        "complete": True,
        "question": protocol["question"],
        "endpoint": protocol["training"]["endpoint"],
        "selection_seeds": [int(seed) for seed in protocol["selection"]["selection_seeds"]],
        "fixed_training_seed": int(protocol["training"]["training_seed"]),
        "rows": rows,
        "aggregate": {
            "random_final_ba_mean_pct": mean,
            "random_final_ba_sample_std_pct": statistics.stdev(values),
            "random_final_ba_min_pct": min(values),
            "random_final_ba_max_pct": max(values),
            "random_final_ba_range_pp": max(values) - min(values),
            "graph_a2_reference_final_ba_pct": ref,
            "random_minus_graph_a2_reference_mean_pp": mean - ref,
            "random_minus_graph_a2_reference_min_pp": min(values) - ref,
            "random_minus_graph_a2_reference_max_pp": max(values) - ref,
            "jaccard_to_seed42_min": min(row["jaccard_to_seed42"] for row in rows),
            "jaccard_to_seed42_max": max(row["jaccard_to_seed42"] for row in rows),
        },
        "interpretation": "This is a selection-seed sensitivity screen: training seed is fixed, so the observed BA spread is attributable to different Random subsets plus the fixed training realization, not to training-seed variation across arms.",
        "best_metric_policy": "best BA is audit-only and not used for the aggregate",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_text(args.output / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = [
        "# PathMNIST 5% Random selection-seed sensitivity",
        "",
        "Five Random subsets were generated with the public `random` selector using selection seeds 42-46. The downstream training seed was fixed at 42 for every arm.",
        "",
        "| Selection seed | Final BA (%) | Best BA audit (%) | Best epoch | Jaccard vs seed 42 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['selection_seed']} | {row['final_ba_pct']:.4f} | {row['best_ba_audit_pct']:.4f} | {row['best_epoch_audit_only']} | {row['jaccard_to_seed42']:.4f} |")
    agg = summary["aggregate"]
    lines.extend([
        "",
        "## Aggregate",
        "",
        f"- Random final BA: {agg['random_final_ba_mean_pct']:.4f}+/-{agg['random_final_ba_sample_std_pct']:.4f}%",
        f"- Selection-seed range: {agg['random_final_ba_range_pp']:.4f} pp ({agg['random_final_ba_min_pct']:.4f}-{agg['random_final_ba_max_pct']:.4f}%)",
        f"- Graph-A2 reference (training seed 42): {ref:.4f}%",
        f"- Random mean minus Graph-A2 reference: {agg['random_minus_graph_a2_reference_mean_pp']:+.4f} pp",
        f"- Jaccard vs seed 42: {agg['jaccard_to_seed42_min']:.4f}-{agg['jaccard_to_seed42_max']:.4f}",
        "",
        "Best BA is retained only as an audit field; the primary result uses final BA.",
        "",
    ])
    atomic_text(args.output / "RESULT.md", "\n".join(lines))
    with (args.output / "selection_seed_results.csv.tmp").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shutil.move(args.output / "selection_seed_results.csv.tmp", args.output / "selection_seed_results.csv")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
