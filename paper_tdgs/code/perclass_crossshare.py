#!/usr/bin/env python
"""Per-class cross-class edge share of the Graph-A2 kernel K = A_sym + A_sym^2.

WHY THIS EXISTS. The mechanism claim is "coverage credit leaks across the class
boundary, and the damage scales with how much leaks". So far that has only been
tested at the DATASET level -- 5 points, and organA/organS are the same 201 LiTS
volumes, so the effective N is 2-3. But both sides of the claim are defined
per CLASS:

    x_c = share of class c's K entries that point at a different class   (a priori)
    y_c = Delta recall_c (tdgs_cls - graph_a2)                           (measured)

Computing x_c lifts the mechanism regression from N=5 to N=9+11+11+8+8 = 47.
This is zero extra GPU training -- y_c is already in tdgs_round1_harvest.json.

DEFINITION NOTE. An earlier prior table reported cross-class share on the 1-hop
adjacency A (pathmnist 0.6%, tissuemnist 58.5%). That is the wrong variable:
credit flows along the entries of K, not of A, and K includes 2-hop mass. Under
the K definition the same five datasets read 2.49 / 26.52 / 42.41 / 53.77 /
66.06 % -- the ordering is unchanged, so every rank statistic survives, but the
absolute numbers must not be mixed between definitions.

We report BOTH mass-weighted and count-based shares, because the mask removes
entries (count) but what greedy actually optimises is their value (mass).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdgs_select import K_HOPS, K_NEIGHBORS, build_graph_kernel, load_emb, load_labels


def per_class_share(K, y: np.ndarray, C: int) -> dict:
    """Row-side shares: for each class c, how much of the credit that class's
    members can hand out (or receive) crosses the boundary."""
    Kc = K.tocoo()
    rows, cols, vals = Kc.row, Kc.col, Kc.data.astype(np.float64)
    same = y[rows] == y[cols]

    out = {"per_class": [], "dataset": {}}
    for c in range(C):
        m = y[rows] == c
        n_tot, v_tot = int(m.sum()), float(vals[m].sum())
        cross = m & ~same
        out["per_class"].append(
            {
                "class": c,
                "n_pool": int((y == c).sum()),
                "nnz": n_tot,
                "cross_share_count": float(cross.sum()) / max(n_tot, 1),
                "cross_share_mass": float(vals[cross].sum()) / max(v_tot, 1e-12),
            }
        )

    out["dataset"] = {
        "nnz": int(len(vals)),
        "cross_share_count": float((~same).sum()) / max(len(vals), 1),
        "cross_share_mass": float(vals[~same].sum()) / max(vals.sum(), 1e-12),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k-neighbors", type=int, default=K_NEIGHBORS)
    ap.add_argument("--k-hops", type=int, default=K_HOPS)
    a = ap.parse_args()

    y = load_labels(a.dataset)
    Z = load_emb(a.dataset)
    C = int(y.max()) + 1
    print(f"{a.dataset}: n={len(y)} C={C} building K (k={a.k_neighbors}, "
          f"hops={a.k_hops}) ...", flush=True)
    K = build_graph_kernel(Z, a.k_neighbors, a.k_hops)
    print(f"  K nnz={K.nnz}", flush=True)

    rep = per_class_share(K, y, C)
    rep.update(
        dataset_name=a.dataset,
        C=C,
        n=int(len(y)),
        k_neighbors=a.k_neighbors,
        k_hops=a.k_hops,
        definition="K = A_sym + A_sym^2 nonzeros (NOT 1-hop A)",
    )

    os.makedirs(a.out_dir, exist_ok=True)
    p = os.path.join(a.out_dir, f"{a.dataset}_perclass_crossshare.json")
    with open(p, "w") as f:
        json.dump(rep, f, indent=1)

    print(f"\n{a.dataset}: dataset count={rep['dataset']['cross_share_count']:.4f} "
          f"mass={rep['dataset']['cross_share_mass']:.4f}")
    print(f"{'cls':>4} {'n_pool':>8} {'count':>8} {'mass':>8}")
    for r in rep["per_class"]:
        print(f"{r['class']:>4} {r['n_pool']:>8} "
              f"{r['cross_share_count']:>8.4f} {r['cross_share_mass']:>8.4f}")
    print("wrote", p)


if __name__ == "__main__":
    main()
