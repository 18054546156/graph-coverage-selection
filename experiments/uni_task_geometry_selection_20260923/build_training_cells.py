"""Emit the 360-cell jsonl for the real-training factorial.

One cell = one ResNet run = (dataset, replicate, arm, weighting, train_seed).
Selection and weights are computed here, on CPU, from the frozen UNI embeddings,
so the GPU shards do nothing but train.

Both arms are nested in the budget, so each replicate runs one selection pass.
Both arms run on the SAME replicate pool, which is what keeps the selection
variance symmetric between them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import ladder_locate_operating_point as L
import run_linear_probe as R

DATASETS = ("bloodmnist", "organsmnist", "tissuemnist")
ARMS = ("random", "herding")
WEIGHTINGS = ("equal", "voronoi_within", "permuted_within")


def replicate_pool(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Stratified subsample of the full train split, one per replicate."""
    parts = []
    for label in np.unique(y):
        ids = np.flatnonzero(y == label)
        rng = np.random.default_rng(seed + 7919 * int(label))
        take = max(1, int(round(len(ids) * fraction)))
        parts.append(rng.choice(ids, size=take, replace=False))
    return np.sort(np.concatenate(parts)).astype(np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--budget-per-class", type=int, default=25)
    p.add_argument("--replicates", type=int, default=10)
    p.add_argument("--train-seeds", type=int, default=2)
    p.add_argument("--pool-fraction", type=float, default=0.80)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--size", type=int, default=224)
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_cells = 0
    with args.output.open("w", encoding="utf-8") as out:
        for dataset in DATASETS:
            emb = args.embedding_root / f"{dataset}_train_uni_{args.size}.npz"
            x = np.asarray(np.load(emb, allow_pickle=False)["embeddings"], dtype=np.float32)
            y = R.read_train_labels(emb, args.data_root / f"{dataset}_{args.size}.npz")
            if len(x) != len(y):
                raise RuntimeError(f"{dataset}: embeddings={len(x)} labels={len(y)}")
            di = R.DATASETS.index(dataset)
            align = R.assert_labels_aligned(x, y, dataset, args.seed + 500009 * di, 0.30)
            z = R.normalize(x)
            del x
            print(f"[{dataset}] n={len(y)} classes={len(np.unique(y))} align_ba={align:.4f}",
                  flush=True)

            for replicate in range(args.replicates):
                rseed = args.seed + 100003 * di + 1009 * replicate
                pool = replicate_pool(y, args.pool_fraction, rseed)
                pool_z, pool_y = z[pool], y[pool]
                orders = {
                    "random": L.random_order(pool_y, args.budget_per_class, rseed),
                    "herding": L.herding_order(pool_z, pool_y, args.budget_per_class),
                }
                for arm in ARMS:
                    local = L.take_prefix(orders[arm], args.budget_per_class)
                    _, w_within, distortion = R.voronoi_weights(pool_z, pool_y, local)

                    # Permuted control: same weight multiset, correspondence
                    # destroyed, within class so the class totals are untouched.
                    perm = np.array(w_within, dtype=np.float64)
                    prng = np.random.default_rng(rseed + 800003)
                    for label in np.unique(pool_y):
                        at = np.flatnonzero(pool_y[local] == label)
                        perm[at] = perm[prng.permutation(at)]

                    weights = {
                        "equal": np.ones(len(local), dtype=np.float64),
                        "voronoi_within": w_within,
                        "permuted_within": perm,
                    }
                    global_ids = pool[local]
                    for weighting in WEIGHTINGS:
                        for s in range(args.train_seeds):
                            out.write(json.dumps({
                                "dataset": dataset,
                                "replicate": replicate,
                                "arm": arm,
                                "weighting": weighting,
                                "train_seed": args.seed + 10007 * s,
                                "budget_per_class": args.budget_per_class,
                                "pool_n": int(len(pool)),
                                "covering_distortion": distortion,
                                "pool_sha256": hashlib.sha256(pool.tobytes()).hexdigest()[:16],
                                "selected_indices": global_ids.tolist(),
                                "weights": weights[weighting].tolist(),
                            }) + "\n")
                            n_cells += 1
                print(f"  replicate {replicate}: pool={len(pool)} "
                      f"budget={args.budget_per_class}/class", flush=True)
    print(json.dumps({"cells": n_cells, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
