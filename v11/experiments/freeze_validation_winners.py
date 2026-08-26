#!/usr/bin/env python3
"""Freeze one validation-selected variant per dataset and write the test config.

The calibration ratio is deliberately shared across budgets. This avoids tuning
one method independently on every dataset-ratio test condition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--validation-config", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--test-output-root", type=Path, required=True)
    parser.add_argument("--reference-variant", default="a0_original")
    parser.add_argument("--calibration-ratio", type=float, default=0.05)
    parser.add_argument("--confirmation-ratios", type=float, nargs="+", default=[0.02, 0.05])
    parser.add_argument("--confirmation-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--required-calibration-seeds", type=int, default=3)
    parser.add_argument("--minimum-ba-gain", type=float, default=0.0)
    parser.add_argument("--worst-recall-tolerance", type=float, default=0.02)
    return parser.parse_args()


def load_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*_result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("evaluation_split") != "val" or row.get("test_read", False):
            raise ValueError(f"calibration input is not validation-only: {path}")
        row["result_path"] = str(path)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no validation result JSON below {root}")
    return rows


def summarize(rows: list[dict], ratio: float) -> dict[tuple[str, str], dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if not np.isclose(float(row["ratio"]), ratio):
            continue
        key = (str(row["dataset"]), str(row["variant"]))
        grouped.setdefault(key, []).append(row)
    output = {}
    for key, values in grouped.items():
        values = sorted(values, key=lambda item: int(item["training_seed"]))
        seeds = [int(item["training_seed"]) for item in values]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seed in calibration results: {key}")
        output[key] = {
            "n_seeds": len(values),
            "seeds": seeds,
            "balanced_accuracy_mean": float(np.mean([item["balanced_accuracy"] for item in values])),
            "balanced_accuracy_std": float(np.std([item["balanced_accuracy"] for item in values], ddof=1)) if len(values) > 1 else 0.0,
            "worst_class_recall_mean": float(np.mean([item["worst_class_recall"] for item in values])),
            "class_cvar20_mean": float(np.mean([item["class_cvar20"] for item in values])),
        }
    return output


def main() -> int:
    args = parse_args()
    source_config = json.loads(args.validation_config.read_text(encoding="utf-8"))
    if source_config.get("schema") != "graphcov-v11/job2-config-v1":
        raise ValueError("unexpected validation config schema")
    stats = summarize(load_rows(args.validation_root), args.calibration_ratio)
    datasets = sorted({dataset for dataset, _ in stats})
    decisions = {}
    jobs = []
    for dataset in datasets:
        reference_key = (dataset, args.reference_variant)
        if reference_key not in stats:
            raise KeyError(f"missing reference results for {dataset}")
        reference = stats[reference_key]
        candidates = []
        for (candidate_dataset, variant), values in stats.items():
            if candidate_dataset != dataset:
                continue
            if values["n_seeds"] < args.required_calibration_seeds:
                raise ValueError(
                    f"{dataset}/{variant} has {values['n_seeds']} seeds; "
                    f"need {args.required_calibration_seeds}"
                )
            safe = (
                values["worst_class_recall_mean"]
                >= reference["worst_class_recall_mean"] - args.worst_recall_tolerance
            )
            gain = values["balanced_accuracy_mean"] - reference["balanced_accuracy_mean"]
            candidates.append((safe and gain >= args.minimum_ba_gain, gain, variant, values))
        eligible = [item for item in candidates if item[0]]
        winner = max(eligible, key=lambda item: (item[1], item[2])) if eligible else None
        winner_id = winner[2] if winner is not None else args.reference_variant
        variants = [args.reference_variant]
        if winner_id != args.reference_variant:
            variants.append(winner_id)
        decisions[dataset] = {
            "reference": args.reference_variant,
            "winner": winner_id,
            "winner_ba_gain": float(winner[1]) if winner is not None else 0.0,
            "reference_metrics": reference,
            "all_candidates": {variant: values for _, _, variant, values in candidates},
        }
        jobs.append(
            {
                "dataset": dataset,
                "ratios": list(args.confirmation_ratios),
                "variants": variants,
                "seeds": list(args.confirmation_seeds),
            }
        )

    output = {
        "schema": "graphcov-v11/job2-config-v1",
        "purpose": "Frozen test confirmation generated from validation-only calibration",
        "selection_root": str(args.selection_root.resolve()),
        "output_root": str(args.test_output_root.resolve()),
        "evaluation_split": "test",
        "training": source_config["training"],
        "jobs": jobs,
    }
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(json.dumps(output, indent=2), encoding="utf-8")
    decision_path = args.output_config.with_suffix(".freeze_manifest.json")
    decision_path.write_text(
        json.dumps(
            {
                "schema": "graphcov-v11/frozen-winners-v1",
                "calibration_ratio": args.calibration_ratio,
                "required_calibration_seeds": args.required_calibration_seeds,
                "minimum_ba_gain": args.minimum_ba_gain,
                "worst_recall_tolerance": args.worst_recall_tolerance,
                "decisions": decisions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(args.output_config)
    print(decision_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
