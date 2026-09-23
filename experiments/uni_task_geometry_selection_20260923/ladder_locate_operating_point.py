"""Locate the budget/scale regime where Voronoi coverage-repair is largest.

Purpose: the probe effect at budget 50 with a 1000/class cap is +0.85pp, which
sits under the measured real-training paired noise (2-3pp). Before spending GPU
we need the operating point where the effect is largest. The prior says gain is
6-13% of headroom, and headroom grows as the budget shrinks, so the effect
should grow at small budgets.

Only `random` and `herding` are run: they are the two arms with opposite
predicted weighting signs, so their difference-in-differences is the estimand,
and neither needs an O(n^2) step, which is what makes full-scale feasible.

Both are **nested** in the budget -- greedy herding's prefix is the k-budget
solution, and a fixed per-class shuffle's prefix is the random k-budget solution
-- so the whole ladder costs one selection per (dataset, cap, block).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import run_linear_probe as R

DATASETS = ("bloodmnist", "organsmnist", "tissuemnist")  # 3 independent sources
BUDGETS = (10, 25, 50, 100, 250)
ARMS = ("random", "herding")


def herding_order(z: np.ndarray, y: np.ndarray, k_max: int) -> dict[int, np.ndarray]:
    """Greedy herding order per class. Prefix of length k IS the k-budget set."""
    order = {}
    for label in np.unique(y):
        ids = np.flatnonzero(y == label)
        zc = z[ids]
        k = min(k_max, len(ids))
        target = zc.mean(axis=0)
        running = np.zeros_like(target)
        chosen: list[int] = []
        taken = np.zeros(len(ids), dtype=bool)
        for step in range(k):
            scores = zc @ ((step + 1) * target - running)
            scores[taken] = -np.inf
            pick = int(np.argmax(scores))
            taken[pick] = True
            chosen.append(pick)
            running += zc[pick]
        order[int(label)] = ids[np.asarray(chosen, dtype=np.int64)]
    return order


def random_order(y: np.ndarray, k_max: int, seed: int) -> dict[int, np.ndarray]:
    order = {}
    for label in np.unique(y):
        ids = np.flatnonzero(y == label)
        rng = np.random.default_rng(seed + 17 * int(label))
        order[int(label)] = rng.permutation(ids)[:min(k_max, len(ids))]
    return order


def take_prefix(order: dict[int, np.ndarray], budget: int) -> np.ndarray:
    return np.concatenate([v[:budget] for v in order.values()]).astype(np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--audit-fraction", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--caps", nargs="+", type=int, default=[1000, 0],
                   help="per-class cap; 0 means no cap (full train split)")
    args = p.parse_args()

    rows = []
    for dataset in DATASETS:
        emb = args.embedding_root / f"{dataset}_train_uni_{args.size}.npz"
        x = np.asarray(np.load(emb, allow_pickle=False)["embeddings"], dtype=np.float32)
        y = R.read_train_labels(emb, args.data_root / f"{dataset}_{args.size}.npz")
        z = R.normalize(x)
        del x
        di = R.DATASETS.index(dataset)

        for cap in args.caps:
            label = "full" if cap == 0 else str(cap)
            effective = len(y) if cap == 0 else cap
            candidate = R.cap_per_class(y, effective, args.seed + 100003 * di)
            for block in range(args.blocks):
                bseed = args.seed + 100003 * di + 1009 * block
                sel, aud = R.split_block(candidate, y, args.audit_fraction, bseed)
                pool_z, pool_y = z[sel], y[sel]
                aud_z, aud_y = z[aud], y[aud]
                k_max = max(BUDGETS)

                t0 = time.time()
                orders = {"random": random_order(pool_y, k_max, bseed),
                          "herding": herding_order(pool_z, pool_y, k_max)}
                sel_seconds = time.time() - t0

                # Ceiling for this (dataset, cap, block): the whole pool.
                full_ba = R.metrics(pool_z, pool_y, aud_z, aud_y, bseed)["balanced_accuracy"]

                for budget in BUDGETS:
                    for arm in ARMS:
                        local = take_prefix(orders[arm], budget)
                        _, w_within, distortion = R.voronoi_weights(pool_z, pool_y, local)
                        out = {}
                        for weighting, sw in (("equal", None), ("voronoi_within", w_within)):
                            out[weighting] = R.metrics(
                                pool_z[local], pool_y[local], aud_z, aud_y, bseed,
                                sample_weight=sw)["balanced_accuracy"]
                        rows.append({
                            "dataset": dataset, "cap": label, "pool_n": int(len(sel)),
                            "block": block, "budget_per_class": budget, "arm": arm,
                            "ba_equal": out["equal"],
                            "ba_voronoi_within": out["voronoi_within"],
                            "delta_pp": 100.0 * (out["voronoi_within"] - out["equal"]),
                            "ba_full_pool": full_ba,
                            "headroom_pp": 100.0 * (full_ba - out["equal"]),
                            "covering_distortion": distortion,
                            "weight_cv": float(w_within.std() / w_within.mean()),
                        })
                print(f"[{dataset} cap={label} block={block}] pool={len(sel)} "
                      f"select={sel_seconds:.1f}s full_ba={full_ba:.4f}", flush=True)

    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    import collections
    import statistics as st
    agg = collections.defaultdict(dict)
    for r in rows:
        agg[(r["dataset"], r["cap"], r["budget_per_class"])].setdefault(r["arm"], []).append(r)
    print("\n%-12s %5s %7s %8s %9s %9s %11s" % (
        "dataset", "cap", "budget", "headroom", "d_random", "d_herding", "interaction"))
    for (dataset, cap, budget), arms in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        hr = st.mean(r["headroom_pp"] for r in arms["random"])
        dr = st.mean(r["delta_pp"] for r in arms["random"])
        dh = st.mean(r["delta_pp"] for r in arms["herding"])
        print("%-12s %5s %7d %8.2f %9.2f %9.2f %11.2f" % (
            dataset, cap, budget, hr, dr, dh, dr - dh))


if __name__ == "__main__":
    main()
