"""Full-scale dataset priors. Replaces measure_dataset_priors.py, which was wrong.

What was wrong with the old one, and why it mattered:

* It capped the pool at `--max-per-class` (800 after the audit split) and then named
  the resulting number `probe_ba_full_pool`. For pathmnist that is 7,200 of 89,996
  images -- 8% -- so the reported "headroom" of 0.61pp was the headroom of an 8%
  subsample, and pathmnist was excluded from the real-training factorial on it.
* The near-duplicate rate sampled 3,000 points out of the already-capped pool. A
  point's true nearest neighbour lands in a 3.3% subsample 3.3% of the time, so the
  measurement had almost no power. Nearest-neighbour distance under subsampling
  inflates as (N/m)^(1/d); at d ~ 10 and 30x dilution that is a factor of ~1.4.
  The reported "pathmnist has zero near-duplicates and the largest NN distance" is
  an artefact of that, and the prediction it supposedly refuted was never tested.

Everything here is exact on the full split. Nothing is subsampled except TwoNN,
which is a local scaling estimator where subsampling is the standard procedure and
is reported with its sample size.

The important addition is a **selection-free** measure of the conditional channel.
Cell label entropy H(q) depends on (dataset, k, selector, pool size), so it is not a
dataset property and cannot support a cross-dataset ordering prediction on its own.
The K-NN label entropy of every pool point, computed against the whole pool, is a
property of the dataset and the embedding alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import run_linear_probe as R

DATASETS = ("bloodmnist", "organamnist", "organsmnist", "pathmnist", "tissuemnist")
KNN_LEVELS = (5, 10, 20, 50)
CELL_BUDGETS = (10, 25, 50, 100, 250)


def blockwise_knn(z: torch.Tensor, y: np.ndarray, kmax: int):
    """Exact K-NN over the WHOLE pool. Returns (nn1_distance, knn_labels)."""
    n = len(z)
    yt = torch.as_tensor(y, device=z.device)
    nn1 = torch.empty(n, device=z.device)
    lab = torch.empty((n, kmax), dtype=torch.long, device=z.device)
    step = max(1, int(1.5e8 // n))
    for lo in range(0, n, step):
        hi = min(n, lo + step)
        d = torch.cdist(z[lo:hi], z)
        d[torch.arange(hi - lo, device=z.device),
          torch.arange(lo, hi, device=z.device)] = float("inf")
        nd, ni = torch.topk(d, kmax, largest=False)
        nn1[lo:hi] = nd[:, 0]
        lab[lo:hi] = yt[ni]
        del d, nd, ni
    return nn1, lab


def label_entropy(lab: torch.Tensor, y: np.ndarray, k: int, n_classes: int):
    """Mean normalised entropy of the K-NN label distribution, and agreement rate.

    Averaged with equal weight per class, because the endpoint is balanced accuracy.
    """
    sub = lab[:, :k]
    oh = torch.zeros((len(sub), n_classes), device=sub.device)
    oh.scatter_add_(1, sub, torch.ones_like(sub, dtype=torch.float))
    q = oh / k
    h = -(q * torch.log(q.clamp_min(1e-12))).sum(1) / np.log(n_classes)
    agree = (sub == torch.as_tensor(y, device=sub.device)[:, None]).float().mean(1)
    yt = torch.as_tensor(y, device=sub.device)
    per_class_h = [float(h[yt == c].mean()) for c in range(n_classes)]
    per_class_a = [float(agree[yt == c].mean()) for c in range(n_classes)]
    return float(np.mean(per_class_h)), float(np.mean(per_class_a)), per_class_h, per_class_a


def cell_stats(z: torch.Tensor, y: np.ndarray, sel: np.ndarray, n_classes: int):
    """Cross-class Voronoi cell purity / entropy on the FULL pool."""
    S = z[torch.as_tensor(sel, device=z.device)]
    n = len(z)
    nearest = torch.empty(n, dtype=torch.long, device=z.device)
    step = max(1, int(2e8 // max(len(sel), 1)))
    for lo in range(0, n, step):
        hi = min(n, lo + step)
        nearest[lo:hi] = torch.cdist(z[lo:hi], S).argmin(1)
    q = torch.zeros((len(sel), n_classes), device=z.device)
    q.scatter_add_(0, nearest[:, None].expand(-1, n_classes),
                   torch.nn.functional.one_hot(
                       torch.as_tensor(y, device=z.device), n_classes).float())
    mass = q.sum(1)
    live = mass > 0
    qn = q[live] / mass[live][:, None]
    h = -(qn * torch.log(qn.clamp_min(1e-12))).sum(1) / np.log(n_classes)
    w = mass[live] / mass[live].sum()
    return {
        "cell_purity_massweighted": float((qn.max(1).values * w).sum()),
        "cell_entropy_massweighted": float((h * w).sum()),
        "cell_purity_unweighted": float(qn.max(1).values.mean()),
        "frac_mixed_cells": float((qn.max(1).values < 0.99).float().mean()),
        "n_live_cells": int(live.sum()),
    }


def twonn(z: np.ndarray, rng, take: int = 4000) -> float:
    take = min(len(z), take)
    sub = z[rng.choice(len(z), take, replace=False)]
    d = R.pairwise_l2(sub, sub)
    np.fill_diagonal(d, np.inf)
    part = np.sort(np.partition(d, 1, axis=1)[:, :2], axis=1)
    r1, r2 = part.T
    keep = (r1 > 1e-9) & (r2 > r1)
    return float(keep.sum() / np.log(r2[keep] / r1[keep]).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--audit-fraction", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--size", type=int, default=224)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for dataset in args.datasets:
        emb = args.embedding_root / f"{dataset}_train_uni_{args.size}.npz"
        x = np.asarray(np.load(emb, allow_pickle=False)["embeddings"], dtype=np.float32)
        y = R.read_train_labels(emb, args.data_root / f"{dataset}_{args.size}.npz")
        di = DATASETS.index(dataset)
        R.assert_labels_aligned(x, y, dataset, args.seed + 500009 * di, 0.30)
        z = R.normalize(x)
        del x
        rng = np.random.default_rng(args.seed + di)
        n_classes = int(y.max()) + 1

        # NO per-class cap. The whole train split, split once into pool and audit.
        pool_ids, audit_ids = R.split_block(np.arange(len(y)), y, args.audit_fraction,
                                            args.seed + 100003 * di)
        zp = torch.as_tensor(z[pool_ids], device=dev)
        yp = y[pool_ids]
        print(f"[{dataset}] train={len(y)} pool={len(pool_ids)} audit={len(audit_ids)} "
              f"C={n_classes}", flush=True)

        rec = {
            "n_train": int(len(y)),
            "n_pool": int(len(pool_ids)),
            "n_audit": int(len(audit_ids)),
            "n_classes": n_classes,
            "full_imbalance_ratio": float(np.bincount(y).max() / np.bincount(y).min()),
            "pool_imbalance_ratio": float(np.bincount(yp).max() / np.bincount(yp).min()),
        }

        # --- headroom on the ACTUAL full pool, not an 800/class cap -------------
        full_ba = R.metrics(z[pool_ids], yp, z[audit_ids], y[audit_ids],
                            args.seed)["balanced_accuracy"]
        rec["probe_ba_full_pool_TRUE"] = full_ba
        for b in (10, 25, 50):
            ids = R.random_balanced(None, yp, b, args.seed)
            ba = R.metrics(z[pool_ids][ids], yp[ids], z[audit_ids], y[audit_ids],
                           args.seed)["balanced_accuracy"]
            rec[f"probe_ba_budget{b}"] = ba
            rec[f"headroom_pp_budget{b}"] = 100.0 * (full_ba - ba)
        print(f"    TRUE headroom b=25: {rec['headroom_pp_budget25']:.2f} pp "
              f"(full-pool BA {full_ba:.4f})", flush=True)

        # --- exact nearest neighbour over the whole pool -----------------------
        nn1, lab = blockwise_knn(zp, yp, max(KNN_LEVELS))
        rec["nn_distance_median_EXACT"] = float(nn1.median())
        for thr in (0.05, 0.10, 0.25):
            rec[f"near_duplicate_rate_d_lt_{thr:.2f}_EXACT"] = float((nn1 < thr).float().mean())

        # --- selection-free conditional channel --------------------------------
        for k in KNN_LEVELS:
            h, a, hc, ac = label_entropy(lab, yp, k, n_classes)
            rec[f"knn{k}_label_entropy_norm"] = h
            rec[f"knn{k}_label_agreement"] = a
            if k == 20:
                rec["knn20_entropy_per_class"] = hc
                rec["knn20_agreement_per_class"] = ac

        # --- selection-dependent cell statistics, across budgets ---------------
        rec["cells"] = {}
        for b in CELL_BUDGETS:
            sel = R.random_balanced(None, yp, b, args.seed)
            rec["cells"][f"random_b{b}"] = cell_stats(zp, yp, sel, n_classes)
        sel_h = R.herding(z[pool_ids], yp, 25)
        rec["cells"]["herding_b25"] = cell_stats(zp, yp, sel_h, n_classes)

        dims = [twonn(z[pool_ids][yp == c], rng) for c in range(n_classes)]
        rec["intrinsic_dimension_median"] = float(np.median(dims))
        rec["intrinsic_dimension_per_class"] = [float(v) for v in dims]
        rec["twonn_sample_size"] = min(4000, int(np.bincount(yp).min()))

        out[dataset] = rec
        del zp, lab, nn1, z
        torch.cuda.empty_cache()
        args.output.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print("\n%-13s %8s %10s %10s %10s %10s %10s" % (
        "dataset", "n_pool", "head_b25", "knn20_H", "knn20_agr", "cellH_b25", "nndist"))
    for d, r in out.items():
        print("%-13s %8d %10.2f %10.3f %10.3f %10.3f %10.3f" % (
            d, r["n_pool"], r["headroom_pp_budget25"], r["knn20_label_entropy_norm"],
            r["knn20_label_agreement"], r["cells"]["random_b25"]["cell_entropy_massweighted"],
            r["nn_distance_median_EXACT"]))


if __name__ == "__main__":
    main()
