#!/usr/bin/env python3
"""Run isolated Graph-A2 method sweeps from configs/sweep.json."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PARAM_FLAGS = {
    "quota_floor_ratio": "--quota-floor-ratio",
    "quota_normalization": "--quota-normalization",
    "relation_mode": "--relation-mode",
    "cross_credit": "--cross-credit",
    "relation_strength": "--relation-strength",
    "pair_min_credit": "--pair-min-credit",
    "pair_smoothing": "--pair-smoothing",
    "purity_exponent": "--purity-exponent",
    "trust_mode": "--trust-mode",
    "trust_weight_floor": "--trust-weight-floor",
}


def csv_values(value: str | None, cast):
    return None if value is None else [cast(item) for item in value.split(",")]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "configs" / "sweep.json")
    parser.add_argument("--datasets", help="Comma-separated override")
    parser.add_argument("--ratios", help="Comma-separated override")
    parser.add_argument("--seeds", help="Comma-separated override")
    parser.add_argument("--k-values", help="Comma-separated override")
    parser.add_argument("--case", action="append", help="Run only named case; repeatable")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--output-root", type=Path, default=root / "outputs")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    datasets = csv_values(args.datasets, str) or config["datasets"]
    ratios = csv_values(args.ratios, float) or config["ratios"]
    seeds = csv_values(args.seeds, int) or config["seeds"]
    k_values = csv_values(args.k_values, int) or config["k_values"]
    cases = [case for case in config["cases"] if not args.case or case["name"] in args.case]
    if not cases:
        raise ValueError("no sweep cases matched --case")

    commands = []
    for dataset in datasets:
        for ratio in ratios:
            for seed in seeds:
                for k_value in k_values:
                    for case_index, case in enumerate(cases):
                        methods = [config["method"]]
                        if config["method"] != "graph_a2" and case_index == 0:
                            methods.insert(0, "graph_a2")
                        ratio_tag = str(ratio).replace(".", "p")
                        output = (
                            args.output_root / dataset / f"r{ratio_tag}" /
                            f"k{k_value}_h{config['hops']}" / case["name"] / f"seed{seed}"
                        )
                        command = [
                            sys.executable, "-m", "graphcov.run",
                            "--datasets", dataset,
                            "--methods", *methods,
                            "--embeddings", "uni",
                            "--ratios", str(ratio),
                            "--trials", "1",
                            "--seed", str(seed),
                            "--training-paradigm", "epoch",
                            "--epochs", str(args.epochs),
                            "--batch-size", "256",
                            "--size", "224",
                            "--k-neighbors", str(k_value),
                            "--k-hops", str(config["hops"]),
                            "--global", "--sparse-cpu",
                            "--output", str(output),
                        ]
                        for key, value in case["params"].items():
                            command.extend([PARAM_FLAGS[key], str(value)])
                        commands.append((output, command))

    if args.max_runs is not None:
        commands = commands[: args.max_runs]
    print(f"Prepared {len(commands)} isolated runs")
    for index, (output, command) in enumerate(commands, start=1):
        if args.resume and (output / "results.csv").exists():
            print(f"[{index}/{len(commands)}] skip completed {output}")
            continue
        print(f"[{index}/{len(commands)}] {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
