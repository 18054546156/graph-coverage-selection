#!/usr/bin/env python3
"""Aggregate final BA into a preregistered paired comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import statistics


T_CRITICAL_DF4_95 = 2.7764451051977987


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    shutil.move(temporary, path)


def summarize(protocol: dict, result_root: Path) -> dict:
    rows = []
    for seed in protocol["training_seeds"]:
        values = {}
        hashes = {}
        for arm in protocol["arm_order"]:
            path = result_root / f"seed{seed}" / arm / "metrics.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            metric = json.loads(path.read_text(encoding="utf-8"))
            if metric.get("complete") is not True or int(metric["training_seed"]) != int(seed):
                raise ValueError(f"invalid metrics: {path}")
            values[arm] = 100.0 * float(metric["balanced_accuracy"])
            hashes[arm] = metric["canonical_index_sha256"]
        delta = values["graph_a2"] - values["random_seed42"]
        rows.append(
            {
                "seed": int(seed),
                "random_seed42_ba_pct": values["random_seed42"],
                "graph_a2_ba_pct": values["graph_a2"],
                "graph_a2_minus_random_pp": delta,
                "winner": "graph_a2" if delta > 0 else ("random_seed42" if delta < 0 else "tie"),
                "random_index_sha256": hashes["random_seed42"],
                "graph_a2_index_sha256": hashes["graph_a2"],
            }
        )

    random_values = [row["random_seed42_ba_pct"] for row in rows]
    graph_values = [row["graph_a2_ba_pct"] for row in rows]
    deltas = [row["graph_a2_minus_random_pp"] for row in rows]
    n = len(rows)
    mean_delta = statistics.mean(deltas)
    sd_delta = statistics.stdev(deltas)
    margin = T_CRITICAL_DF4_95 * sd_delta / math.sqrt(n)
    ranking_reproduced = mean_delta > 0
    return {
        "schema": "graphcov-table1/path5-fixed-selection-summary-v1",
        "complete": True,
        "endpoint": protocol["training"]["endpoint"],
        "n_pairs": n,
        "rows": rows,
        "aggregate": {
            "random_seed42_mean_ba_pct": statistics.mean(random_values),
            "random_seed42_sample_std_ba_pct": statistics.stdev(random_values),
            "graph_a2_mean_ba_pct": statistics.mean(graph_values),
            "graph_a2_sample_std_ba_pct": statistics.stdev(graph_values),
            "mean_paired_delta_graph_a2_minus_random_pp": mean_delta,
            "sample_std_paired_delta_pp": sd_delta,
            "paired_delta_95pct_t_interval_pp": [mean_delta - margin, mean_delta + margin],
            "graph_a2_wins": sum(delta > 0 for delta in deltas),
            "random_wins": sum(delta < 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
        },
        "decision": {
            "paper_ranking_reproduced": ranking_reproduced,
            "label": "paper ranking reproduced" if ranking_reproduced else "paper ranking not reproduced",
            "interpretation": (
                protocol["decision"]["interpretation_if_reproduced"]
                if ranking_reproduced
                else protocol["decision"]["interpretation_if_not_reproduced"]
            ),
        },
        "paper_path5_pct": protocol["paper_path5_pct"],
        "note": "The 95% interval is descriptive for five paired seeds; the frozen ranking decision uses only the sign of the mean paired delta.",
    }


def render_markdown(summary: dict) -> str:
    aggregate = summary["aggregate"]
    lines = [
        "# PathMNIST 5% fixed-selection paired result",
        "",
        f"Decision: **{summary['decision']['label']}**.",
        "",
        "Primary endpoint: final official-test balanced accuracy. Both arms use one frozen, canonically sorted subset; only downstream seed changes.",
        "",
        "| Seed | Random seed-42 BA (%) | Graph-A2 BA (%) | Graph-A2 - Random (pp) | Winner |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['seed']} | {row['random_seed42_ba_pct']:.4f} | "
            f"{row['graph_a2_ba_pct']:.4f} | {row['graph_a2_minus_random_pp']:+.4f} | {row['winner']} |"
        )
    ci_low, ci_high = aggregate["paired_delta_95pct_t_interval_pp"]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Random: {aggregate['random_seed42_mean_ba_pct']:.4f}+/-{aggregate['random_seed42_sample_std_ba_pct']:.4f}%",
            f"- Graph-A2: {aggregate['graph_a2_mean_ba_pct']:.4f}+/-{aggregate['graph_a2_sample_std_ba_pct']:.4f}%",
            f"- Mean paired delta: {aggregate['mean_paired_delta_graph_a2_minus_random_pp']:+.4f} pp",
            f"- Descriptive paired 95% t interval: [{ci_low:+.4f}, {ci_high:+.4f}] pp",
            f"- Wins: Graph-A2 {aggregate['graph_a2_wins']}, Random {aggregate['random_wins']}, ties {aggregate['ties']}",
            "",
            "## Interpretation",
            "",
            summary["decision"]["interpretation"],
            "",
            "This controlled result diagnoses the selection-seed confound. It is not a new method result or a SOTA claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    summary = summarize(protocol, args.results)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_text(args.output / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    atomic_text(args.output / "RESULT.md", render_markdown(summary))
    csv_path = args.output / "paired_results.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["rows"][0]))
        writer.writeheader()
        writer.writerows(summary["rows"])
    shutil.move(temporary, csv_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
