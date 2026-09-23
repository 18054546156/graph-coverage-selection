"""Test PREDICTION_moment_matching.md against results/run_33659.

Zero new selection and zero new training: the manifest already stores the
selected ids and the Voronoi masses, and the pool is reconstructible from the
recorded seed schedule. The reconstruction is verified against the manifest's
audit ids before any statistic is computed, so a seed-schedule drift fails loudly
instead of producing a plausible wrong answer.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import run_linear_probe as R

METHODS = list(R.ROUND1) + list(R.ROUND2)


def first_moment_error(pool_z, pool_y, sel_local, weights):
    """Mean over classes of || weighted mean of selected - pool class mean ||."""
    errors = []
    for label in np.unique(pool_y):
        mask = pool_y[sel_local] == label
        z = pool_z[sel_local[mask]]
        w = np.asarray(weights, dtype=np.float64)[mask]
        mu = pool_z[pool_y == label].mean(axis=0)
        errors.append(float(np.linalg.norm((z * w[:, None]).sum(0) / w.sum() - mu)))
    return st.mean(errors)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--embedding-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--max-per-class", type=int, default=1000)
    p.add_argument("--audit-fraction", type=float, default=0.20)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--size", type=int, default=224)
    args = p.parse_args()

    manifest = collections.defaultdict(dict)
    for line in (args.run_dir / "selection_manifest.jsonl").open(encoding="utf-8"):
        rec = json.loads(line)
        manifest[(rec["dataset"], rec["block"])][rec["method"]] = rec

    ba = collections.defaultdict(dict)
    for line in (args.run_dir / "linear_probe_results.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        ba[(row["dataset"], row["block"], row["method"])][row["training_weighting"]] = \
            row["balanced_accuracy"]

    eps = collections.defaultdict(lambda: collections.defaultdict(list))
    for di, dataset in enumerate(R.DATASETS):
        emb = args.embedding_root / f"{dataset}_train_uni_{args.size}.npz"
        x = np.asarray(np.load(emb, allow_pickle=False)["embeddings"], dtype=np.float32)
        y = R.read_train_labels(emb, args.data_root / f"{dataset}_{args.size}.npz")
        z = R.normalize(x)
        candidate = R.cap_per_class(y, args.max_per_class, args.seed + 100003 * di)
        for block in range(args.blocks):
            block_seed = args.seed + 100003 * di + 1009 * block
            select_ids, audit_ids = R.split_block(candidate, y, args.audit_fraction, block_seed)
            pool_z, pool_y = z[select_ids], y[select_ids]
            order = {int(g): i for i, g in enumerate(select_ids)}
            for method in METHODS:
                rec = manifest[(dataset, block)][method]
                if rec["audit_ids"] != audit_ids.tolist():
                    raise RuntimeError(
                        f"{dataset} block {block}: reconstructed pool does not match the "
                        "manifest audit ids -- the seed schedule drifted"
                    )
                sel_local = np.asarray([order[i] for i in rec["selected_ids"]], dtype=np.int64)
                w = np.asarray(rec["voronoi_weights"], dtype=np.float64)
                eps[(dataset, method)]["equal"].append(
                    first_moment_error(pool_z, pool_y, sel_local, np.ones(len(sel_local))))
                eps[(dataset, method)]["voronoi"].append(
                    first_moment_error(pool_z, pool_y, sel_local, w))
        print(f"[{dataset}] reconstructed and verified", flush=True)

    mean_eps = {k: {w: st.mean(v) for w, v in d.items()} for k, d in eps.items()}
    delta_eps = {k: d["voronoi"] - d["equal"] for k, d in mean_eps.items()}
    delta_ba = {
        (dataset, method): st.mean(
            100.0 * (ba[(dataset, b, method)]["voronoi"] - ba[(dataset, b, method)]["equal"])
            for b in range(args.blocks))
        for dataset in R.DATASETS for method in METHODS
    }

    head = "%-36s" % "arm" + "".join("%12s" % d[:9] for d in R.DATASETS)
    print("\n=== eps_mean(equal): herding should be SMALLEST (P3) ===")
    print(head)
    for m in METHODS:
        print("%-36s" % m + "".join("%12.4f" % mean_eps[(d, m)]["equal"] for d in R.DATASETS))
    p3 = sum(1 for d in R.DATASETS
             if min(METHODS, key=lambda m: mean_eps[(d, m)]["equal"]) == "herding")
    print(f"P3: herding has the smallest eps_mean(equal) on {p3}/5 datasets "
          f"-> {'PASS' if p3 >= 4 else 'FAIL'}")

    print("\n=== delta_eps = eps(voronoi) - eps(equal)  (P1) ===")
    print(head)
    for m in METHODS:
        print("%-36s" % m + "".join("%12.4f" % delta_eps[(d, m)] for d in R.DATASETS))
    for method, want in (("herding", +1), ("random", -1), ("random_plus_pilot", -1)):
        n = sum(1 for d in R.DATASETS if np.sign(delta_eps[(d, method)]) == want)
        print(f"P1: {method:<26} sign {'+' if want > 0 else '-'} on {n}/5 "
              f"-> {'PASS' if n >= 4 else 'FAIL'}")

    print("\n=== Spearman(delta_eps, delta_BA) over the 8 arms (P2) ===")
    negative = 0
    for d in R.DATASETS:
        rho, pv = spearmanr([delta_eps[(d, m)] for m in METHODS],
                            [delta_ba[(d, m)] for m in METHODS])
        negative += rho < 0
        print(f"  {d:<14} rho={rho:+.3f} p={pv:.3f}")
    print(f"P2: negative on {negative}/5 -> {'PASS' if negative >= 3 else 'FAIL'}")

    (args.run_dir / "moment_matching.json").write_text(json.dumps({
        "eps_mean": {f"{d}|{m}": mean_eps[(d, m)] for d in R.DATASETS for m in METHODS},
        "delta_eps": {f"{d}|{m}": delta_eps[(d, m)] for d in R.DATASETS for m in METHODS},
        "delta_ba_pp": {f"{d}|{m}": delta_ba[(d, m)] for d in R.DATASETS for m in METHODS},
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
