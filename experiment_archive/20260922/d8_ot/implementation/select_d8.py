"""CLI: select from a source-train embeddings/labels NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from d8_ot import select_equal_mass_ot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="NPZ containing embeddings and labels")
    parser.add_argument("--output", type=Path, help="Output NPZ path")
    parser.add_argument("--budget-per-class", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--outer-iterations", type=int, default=8)
    parser.add_argument("--sinkhorn-iterations", type=int, default=60)
    args = parser.parse_args()
    if args.input is None:
        parser.print_help()
        return
    if args.output is None or args.budget_per_class is None:
        parser.error("--output and --budget-per-class are required with --input")

    with np.load(args.input, allow_pickle=False) as data:
        embeddings = data["embeddings"]
        labels = data["labels"]
    selected, diagnostics = select_equal_mass_ot(
        embeddings,
        labels,
        args.budget_per_class,
        args.seed,
        epsilon=args.epsilon,
        outer_iterations=args.outer_iterations,
        sinkhorn_iterations=args.sinkhorn_iterations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, selected_indices=selected)
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "method": "d8_equal_mass_class_conditional_entropic_ot_heuristic",
        "seed": args.seed,
        "budget_per_class": args.budget_per_class,
        "epsilon": args.epsilon,
        "outer_iterations": args.outer_iterations,
        "sinkhorn_iterations": args.sinkhorn_iterations,
        "selected_count": int(len(selected)),
        "class_diagnostics": diagnostics,
    }, indent=2), encoding="utf-8")
    print(f"selected={len(selected)} indices={args.output} diagnostics={sidecar}")


if __name__ == "__main__":
    main()
