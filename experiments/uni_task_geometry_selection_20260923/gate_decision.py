"""Pre-registered probe -> GPU go/no-go rule (decision G).

This file exists so the gate cannot be re-derived after the results are on
screen. The thresholds below were frozen before any audit number was read;
changing them is a protocol amendment and must be recorded as one, not edited
in silently.

Usage:
    python gate_decision.py --results results/run_XXXX/linear_probe_results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TREATMENT = "task_geometry_plus_pilot"
CONTROL = "permuted_task_geometry_plus_pilot"
ENDPOINT = "balanced_accuracy"
WEIGHTING = "equal"

# Frozen thresholds, in balanced-accuracy points.
GO_THRESHOLD_PP = 1.0
NO_GO_THRESHOLD_PP = 0.5
MIN_DATASETS_AT_THRESHOLD = 3
MAX_NEGATIVE_DATASETS = 3


def load_primary(path: Path) -> dict[str, dict[int, dict[str, float]]]:
    """Return {dataset: {block: {method: endpoint}}} for the primary cells only."""
    table: dict[str, dict[int, dict[str, float]]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("training_weighting") != WEIGHTING:
                continue
            if row["method"] not in (TREATMENT, CONTROL):
                continue
            block = table.setdefault(row["dataset"], {}).setdefault(int(row["block"]), {})
            block[row["method"]] = float(row[ENDPOINT])
    return table


def per_dataset_effects(table) -> dict[str, dict]:
    out = {}
    for dataset in sorted(table):
        deltas = []
        for block in sorted(table[dataset]):
            cell = table[dataset][block]
            if TREATMENT not in cell or CONTROL not in cell:
                raise RuntimeError(f"{dataset} block {block}: incomplete primary pair {sorted(cell)}")
            deltas.append(100.0 * (cell[TREATMENT] - cell[CONTROL]))
        mean = sum(deltas) / len(deltas)
        if len(deltas) > 1:
            var = sum((d - mean) ** 2 for d in deltas) / (len(deltas) - 1)
            se = (var / len(deltas)) ** 0.5
        else:
            se = float("nan")
        out[dataset] = {
            "n_blocks": len(deltas),
            "delta_pp": [round(d, 4) for d in deltas],
            "mean_delta_pp": mean,
            "se_pp": se,
            "n_blocks_positive": sum(1 for d in deltas if d > 0),
        }
    return out


def decide(effects: dict[str, dict]) -> dict:
    means = [e["mean_delta_pp"] for e in effects.values()]
    overall = sum(means) / len(means)
    at_threshold = sum(1 for m in means if m >= GO_THRESHOLD_PP)
    negative = sum(1 for m in means if m < 0)

    if overall >= GO_THRESHOLD_PP and at_threshold >= MIN_DATASETS_AT_THRESHOLD:
        verdict = "GO"
        action = "proceed to the 120 GPU runs with the same frozen tau and the same primary contrast"
    elif overall < NO_GO_THRESHOLD_PP or negative >= MAX_NEGATIVE_DATASETS:
        verdict = "NO_GO"
        action = "no GPU spend; report the probe result as a negative and close the task-geometry branch"
    else:
        verdict = "AMBIGUOUS"
        action = "extend blocks 6 -> 12 (CPU only) and re-apply this rule exactly once; a second AMBIGUOUS resolves to NO_GO"

    return {
        "verdict": verdict,
        "action": action,
        "overall_mean_delta_pp": overall,
        "n_datasets": len(means),
        "n_datasets_at_or_above_threshold": at_threshold,
        "n_datasets_negative": negative,
        "thresholds": {
            "go_pp": GO_THRESHOLD_PP,
            "no_go_pp": NO_GO_THRESHOLD_PP,
            "min_datasets_at_threshold": MIN_DATASETS_AT_THRESHOLD,
            "max_negative_datasets": MAX_NEGATIVE_DATASETS,
        },
        "contrast": f"{TREATMENT} - {CONTROL}",
        "endpoint": ENDPOINT,
        "weighting": WEIGHTING,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    table = load_primary(args.results)
    if not table:
        raise SystemExit(f"no primary rows in {args.results} (need training_weighting={WEIGHTING})")
    effects = per_dataset_effects(table)
    report = {"per_dataset": effects, "decision": decide(effects), "results_csv": str(args.results)}

    print(f"{'dataset':<16}{'n':>4}{'mean delta (pp)':>18}{'se':>8}{'blocks>0':>10}")
    for dataset, e in effects.items():
        print(f"{dataset:<16}{e['n_blocks']:>4}{e['mean_delta_pp']:>18.3f}{e['se_pp']:>8.3f}"
              f"{e['n_blocks_positive']:>7}/{e['n_blocks']}")
    d = report["decision"]
    print(f"\noverall mean {d['overall_mean_delta_pp']:.3f} pp | "
          f"{d['n_datasets_at_or_above_threshold']}/{d['n_datasets']} at >= {GO_THRESHOLD_PP} pp | "
          f"{d['n_datasets_negative']}/{d['n_datasets']} negative")
    print(f"VERDICT: {d['verdict']}\n  -> {d['action']}")

    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
