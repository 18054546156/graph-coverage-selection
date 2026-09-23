"""Dataset-level covariates for the five MedMNIST subsets, on UNI embeddings.

Descriptive only -- no method comparison, no protocol decision. The point is to
know which datasets can carry signal at all before reading any method contrast
off them.

Motivated by two literature facts:
  * OrganA/OrganS are the same 201 LiTS CT volumes, different slice planes, so
    they are not independent replication units.
  * PathMNIST's 100k patches come from 86 WSIs and MedMNIST splits them at the
    patch level, so a train-only audit split puts same-slide tiles on both
    sides. near_duplicate_rate is the direct measurement of that.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_linear_probe as R


def twonn_dimension(z: np.ndarray, rng) -> float:
    """TwoNN intrinsic dimension (Facco et al.): d = N / sum(log(r2/r1))."""
    take = min(len(z), 1500)
    sub = z[rng.choice(len(z), take, replace=False)]
    d = R.pairwise_l2(sub, sub)
    np.fill_diagonal(d, np.inf)
    part = np.partition(d, 1, axis=1)[:, :2]
    r1, r2 = np.sort(part, axis=1).T
    keep = (r1 > 1e-9) & (r2 > r1)
    mu = r2[keep] / r1[keep]
    return float(keep.sum() / np.log(mu).sum())


def probe_ba(train_z, train_y, test_z, test_y, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)
    clf.fit(train_z, train_y)
    return float(balanced_accuracy_score(test_y, clf.predict(test_z)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--max-per-class", type=int, default=1000)
    p.add_argument("--budget-per-class", type=int, default=50)
    p.add_argument("--audit-fraction", type=float, default=0.20)
    p.add_argument("--size", type=int, default=224)
    args = p.parse_args()

    out = {}
    for di, dataset in enumerate(R.DATASETS):
        emb = args.embedding_root / f"{dataset}_train_uni_{args.size}.npz"
        x = np.asarray(np.load(emb, allow_pickle=False)["embeddings"], dtype=np.float32)
        y = R.read_train_labels(emb, args.data_root / f"{dataset}_{args.size}.npz")
        z = R.normalize(x)
        rng = np.random.default_rng(args.seed + di)

        full_counts = np.bincount(y)
        candidate = R.cap_per_class(y, args.max_per_class, args.seed + 100003 * di)
        select_ids, audit_ids = R.split_block(candidate, y, args.audit_fraction,
                                              args.seed + 100003 * di)
        pool_y, pool_z = y[select_ids], z[select_ids]
        pool_counts = np.bincount(pool_y)

        # Headroom: what the probe gets from the whole pool vs from the budget.
        # If this is ~0 no selection method can differ by more than noise.
        full_ba = probe_ba(pool_z, pool_y, z[audit_ids], y[audit_ids], args.seed)
        budget_ids = R.random_balanced(pool_z, pool_y, args.budget_per_class, args.seed)
        budget_ba = probe_ba(pool_z[budget_ids], pool_y[budget_ids],
                             z[audit_ids], y[audit_ids], args.seed)

        # Near-duplicate rate: nearest-neighbour distance within the pool,
        # sampled. On L2-normalised features d = sqrt(2-2cos), so d<0.10
        # corresponds to cosine > 0.995 -- effectively the same image.
        take = min(len(pool_z), 3000)
        sub = pool_z[rng.choice(len(pool_z), take, replace=False)]
        dmat = R.pairwise_l2(sub, sub)
        np.fill_diagonal(dmat, np.inf)
        nn = dmat.min(axis=1)

        dims = [twonn_dimension(pool_z[pool_y == c], rng) for c in np.unique(pool_y)]

        # Voronoi mass concentration under a random balanced selection.
        w, _, distortion = R.voronoi_weights(pool_z, pool_y, budget_ids)
        s = np.sort(w)
        n = len(s)
        gini = float((2 * np.arange(1, n + 1) - n - 1).dot(s) / (n * s.sum()))

        out[dataset] = {
            "n_train": int(len(y)),
            "n_classes": int(len(full_counts)),
            "full_imbalance_ratio": float(full_counts.max() / full_counts.min()),
            "pool_per_class_min": int(pool_counts.min()),
            "pool_per_class_max": int(pool_counts.max()),
            "pool_imbalance_ratio": float(pool_counts.max() / pool_counts.min()),
            "probe_ba_full_pool": full_ba,
            "probe_ba_budget50": budget_ba,
            "headroom_pp": 100.0 * (full_ba - budget_ba),
            "nn_distance_median": float(np.median(nn)),
            "near_duplicate_rate_d_lt_0.10": float((nn < 0.10).mean()),
            "near_duplicate_rate_d_lt_0.25": float((nn < 0.25).mean()),
            "intrinsic_dimension_median": float(np.median(dims)),
            "intrinsic_dimension_min": float(np.min(dims)),
            "intrinsic_dimension_max": float(np.max(dims)),
            "voronoi_mass_gini": gini,
            "covering_distortion_random": distortion,
        }
        print(f"[{dataset}] done", flush=True)

    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    cols = [("headroom_pp", "%9.2f"), ("probe_ba_full_pool", "%9.4f"),
            ("probe_ba_budget50", "%9.4f"), ("pool_imbalance_ratio", "%9.2f"),
            ("near_duplicate_rate_d_lt_0.10", "%9.4f"),
            ("near_duplicate_rate_d_lt_0.25", "%9.4f"),
            ("nn_distance_median", "%9.4f"), ("intrinsic_dimension_median", "%9.2f"),
            ("voronoi_mass_gini", "%9.4f")]
    for name, fmt in cols:
        print("%-32s" % name + "".join(fmt % out[d][name] for d in R.DATASETS))
    print("%-32s" % "" + "".join("%9s" % d[:8] for d in R.DATASETS))


if __name__ == "__main__":
    main()
