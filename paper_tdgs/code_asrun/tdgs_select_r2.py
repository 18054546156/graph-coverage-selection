#!/usr/bin/env python
"""TDGS round 2 -- the five arms that decide whether `tdgs_cls` is a contribution.

Round 1 (job 34970, 75 real cells) produced exactly one positive: `tdgs_cls`
beats `graph_a2` by +1.76pp BA / +3.54pp worst-class recall, 5/5 datasets same
sign, with the effect size predicted a priori by cross-class edge share. Both
demand-weighted arms lost, and the class-pair direction term was inert against
its own within-class permutation control.

`tdgs_cls` changes TWO things at once, and one rival explanation has not been
excluded. This file runs the arms that settle both.

-------------------------------------------------------------------------------
WHY NO PROBE IS NEEDED HERE  (and why this is cheap)
-------------------------------------------------------------------------------
Every arm below is free of d_i and u_i. Round 2 therefore never fits the OOF
linear head, needs no GPU for selection, and is fully deterministic given the
embeddings. The only stochastic arm is `tdgs_cls_perm`, whose permutation is
seeded explicitly.

-------------------------------------------------------------------------------
ARM 1 -- `a2_perclass`:  THE ONLY CONTROL THAT CAN KILL THE POSITIVE
-------------------------------------------------------------------------------
The author's released default is `global_selection=False`, i.e. PER-CLASS scope:
the kNN graph is built inside each class, so coverage credit is already confined
within class. `compare_global.py` is the author's own occupied ablation cell and
the paper reports global > per-class.

If per-class graph_a2 at k=50 recovers most of the +1.76pp, then `tdgs_cls` is
inside an occupied cell and is NOT a contribution. If per-class is worse (as the
paper reports) while `tdgs_cls` is better, then the novel ingredient is the
lam=0.5 BLEND -- keeping global geometry while ADDING class-compatible credit --
plus class-flat demand weighting. That distinction is the paper's claim, so it
has to be measured, not argued.

Implementation: block-diagonal K, each block built by running the author's own
global pipeline (build_knn_graph -> build_adjacency_matrix -> D^-1/2 A D^-1/2 ->
A+A^2) on that class's rows alone. Greedy over a block-diagonal kernel with an
equal per-class quota returns the SAME SET as independent per-class greedy: gains
in one block never depend on coverage in another, and the quota forces exactly
bpc picks per class. Only the interleaving order differs.

-------------------------------------------------------------------------------
ARMS 2-3 -- `tdgs_mask`, `tdgs_wcls`:  A 2x2 FACTORIAL, TWO CELLS ALREADY RUN
-------------------------------------------------------------------------------
                     | uniform demand weight | class-flat weight 1/(C n_c)
    -----------------+-----------------------+----------------------------
    global credit    |   graph_a2   (RUN)    |   tdgs_wcls   (new)
    within-class     |   tdgs_mask  (new)    |   tdgs_cls    (RUN)

The top-left cell is graph_a2 by identity, not by approximation: with no mask and
uniform weights the demand term is K/kmax with w=1, i.e. proportional to F_G, so
F_new = (1-lam)F_G/F_G(P) + lam*F_G/F_G(P) is a positive multiple of F_G and
greedy picks the same set. Verified at runtime by `--assert-a2-identity`.

So two new arms buy both main effects AND the interaction:
    mask effect        = (tdgs_mask - graph_a2)  and  (tdgs_cls - tdgs_wcls)
    weight effect      = (tdgs_wcls - graph_a2)  and  (tdgs_cls - tdgs_mask)
    interaction        = tdgs_cls - tdgs_mask - tdgs_wcls + graph_a2
A large interaction would mean the two corrections are not separable -- credit
has to be confined AND reweighted, which is a stronger and more interesting
claim than either alone.

-------------------------------------------------------------------------------
ARM 4 -- `tdgs_lam1`:  is the blend load-bearing?
-------------------------------------------------------------------------------
lam=1 drops F_G entirely: pure class-compatible, class-balanced coverage. If
lam=1 >= lam=0.5, the global term is dead weight and the method simplifies. If
lam=1 < lam=0.5 < ... then the blend is the contribution and `a2_perclass`
(which is also "within-class only") should look like lam=1, which would be a
second, independent confirmation of the blend story.

-------------------------------------------------------------------------------
ARM 5 -- `tdgs_cls_perm`:  the positive result's own null
-------------------------------------------------------------------------------
Keeps the mask STRUCTURE (credit still confined within class, same per-row
nnz, same value multiset per row) but permutes candidate identity within class:
for entries (i,j) with y_i=y_j, j is replaced by pi_c(j). Demand rows are
untouched.

This separates "the within-class kNN geometry carries information" from "merely
shrinking the candidate-demand bipartite graph helps". Column permutation rather
than value shuffling is deliberate: shuffling values leaves adjacency intact and
within-class Kbar values are fairly homogeneous, so it would be a weak null.

Every prior positive in this project that skipped its matched null later died
(Voronoi, task-geometry, FRACTAL). This arm is not optional.

Usage:
    python tdgs_select_r2.py --dataset pathmnist --ratio 0.02 \
        --out-dir /mnt/prj01/hgrp-1502-5TB/tdgs_shared/sel
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse import diags

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdgs_select import (EPS, K_HOPS, K_NEIGHBORS, LAMBDA, budget_per_class,
                         build_graph_kernel, greedy_blended, load_emb,
                         load_labels, pool_value)

ARMS_R2 = ["a2_perclass", "tdgs_mask", "tdgs_wcls", "tdgs_lam1",
           "tdgs_cls_perm"]
# regenerated on request so a 5% run does not have to borrow the archive
ARMS_BASE = ["graph_a2", "tdgs_cls"]


# ------------------------------------------------------------ per-class scope
def build_perclass_kernel(Z, y, k_neighbors=K_NEIGHBORS, k_hops=K_HOPS,
                          verbose=True):
    """Block-diagonal K: the author's per-class (`global_selection=False`) scope.

    Each class's graph is built on that class's rows only, so k is counted among
    same-class neighbours -- which is the substantive difference from masking a
    globally-built graph, where a point's 50 neighbours may be mostly other
    classes and masking leaves it with very few.
    """
    from graphcov.run.graph import build_adjacency_matrix
    from graphcov.run.selection import build_knn_graph
    n, C = len(y), int(y.max()) + 1
    blocks = []
    for c in range(C):
        idx = np.flatnonzero(y == c)
        Zc = np.ascontiguousarray(Z[idx])
        kc = min(k_neighbors, len(idx) - 1)
        ki, kd = build_knn_graph(Zc, kc, verbose=False)
        A = build_adjacency_matrix(ki, kd, len(idx))
        rs = np.asarray(A.sum(axis=1)).ravel()
        rs[rs == 0] = 1.0
        Di = diags(1.0 / np.sqrt(rs))
        As = Di @ A @ Di
        Kc = As.copy()
        Ap = As.copy()
        for _ in range(k_hops - 1):
            Ap = Ap @ As
            Kc = Kc + Ap
        Kc = Kc.tocoo()
        blocks.append((idx[Kc.row], idx[Kc.col], Kc.data.astype(np.float32)))
        if verbose:
            print(f"    class {c}: n={len(idx)} k={kc} nnz={Kc.nnz/1e6:.2f}M")
    r = np.concatenate([b[0] for b in blocks])
    c_ = np.concatenate([b[1] for b in blocks])
    v = np.concatenate([b[2] for b in blocks])
    return sp.csc_matrix((v, (r, c_)), shape=(n, n))


# ----------------------------------------------------------- demand kernels
def build_kernel_variant(K, y, mask_class, permute_within_class=False,
                         seed=42):
    """Kbar, optionally class-masked, optionally column-permuted within class.

    Returns CSC. `mask_class=False` keeps cross-class entries, which is the
    `tdgs_wcls` arm: global credit, class-flat weights.
    """
    Kc = K.tocoo()
    rows, cols = Kc.row, Kc.col
    vals = Kc.data.astype(np.float32)
    kmax = float(vals.max()) if len(vals) else 1.0
    vals = vals / (kmax + EPS)

    if mask_class:
        keep = y[rows] == y[cols]
        rows, cols, vals = rows[keep], cols[keep], vals[keep]

    if permute_within_class:
        # j -> pi_c(j) inside each class. Demand rows untouched, so every
        # demand point keeps its own degree and value multiset; only WHICH
        # candidate supplies which support is randomised.
        rng = np.random.default_rng(seed)
        relabel = np.arange(len(y))
        for c in range(int(y.max()) + 1):
            idx = np.flatnonzero(y == c)
            relabel[idx] = rng.permutation(idx)
        cols = relabel[cols]

    nz = vals > 0
    a = sp.csc_matrix((vals[nz], (rows[nz], cols[nz])), shape=K.shape)
    a.sum_duplicates()
    return a, kmax


# ------------------------------------------------------------------- driver
def run_arm(arm, K, Kpc, y, C, bpc, w_cls, w1, FG_P, lam_default, seed,
            verbose=True):
    """Return the selected indices for one arm. All arms are probe-free."""
    if arm == "graph_a2":
        a_mat = sp.csc_matrix(K.shape, dtype=np.float32)
        return greedy_blended(K, a_mat, y, bpc, w1 / FG_P,
                              np.zeros(len(y), np.float32), 0.0, verbose)

    if arm == "a2_perclass":
        # pure facility location on the block-diagonal graph, lam=0
        FG_pc = pool_value(Kpc, w1)
        a_mat = sp.csc_matrix(K.shape, dtype=np.float32)
        return greedy_blended(Kpc, a_mat, y, bpc, w1 / FG_pc,
                              np.zeros(len(y), np.float32), 0.0, verbose)

    mask = arm != "tdgs_wcls"                       # wcls keeps global credit
    perm = arm == "tdgs_cls_perm"
    a_mat, kmax = build_kernel_variant(K, y, mask, perm, seed)
    w2 = np.ones(len(y), np.float32) if arm == "tdgs_mask" else w_cls
    lam = 1.0 if arm == "tdgs_lam1" else lam_default
    Ft_P = pool_value(a_mat, w2)
    if Ft_P <= 0:
        raise SystemExit(f"F_task(P)==0 for {arm}")
    if verbose:
        print(f"    demand kernel nnz={a_mat.nnz/1e6:.1f}M  mask={mask} "
              f"perm={perm}  lam={lam}  Kbar_scale={kmax:.4f}  "
              f"F_task(P)={Ft_P:.6g}")
    return greedy_blended(K, a_mat, y, bpc, w1 / FG_P, w2 / Ft_P, lam, verbose)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ratio", type=float, default=0.02)
    ap.add_argument("--arms", nargs="*", default=ARMS_R2)
    ap.add_argument("--lam", type=float, default=LAMBDA)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--assert-a2-identity", action="store_true",
                    help="prove the (no-mask, uniform-weight) factorial cell "
                         "equals graph_a2 instead of asserting it in a comment")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.report, exist_ok=True)
    ds, ratio = a.dataset, a.ratio

    Z, emb_path = load_emb(ds)
    y = load_labels(ds)
    if len(Z) != len(y):
        raise SystemExit(f"[{ds}] truncated cache: {len(Z)} vs {len(y)}")
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + EPS)
    C = int(y.max()) + 1
    bpc = budget_per_class(len(y), ratio, C)
    print(f"\n{'='*86}\n[{ds} r={ratio}] n={len(y)} C={C} bpc={bpc} "
          f"budget={bpc*C}\n  emb <- {emb_path}\n{'='*86}")

    cls_n = np.bincount(y, minlength=C).astype(np.float32)
    w_cls = (1.0 / (C * cls_n))[y].astype(np.float32)
    w1 = np.ones(len(y), np.float32)

    t0 = time.time()
    K = build_graph_kernel(Z)
    print(f"  global kernel nnz={K.nnz/1e6:.1f}M ({time.time()-t0:.0f}s)")
    FG_P = pool_value(K, w1)

    # cross-class edge share -- the a-priori moderator the mechanism predicts
    Kc = K.tocoo()
    xshare = float((y[Kc.row] != y[Kc.col]).mean())
    print(f"  cross-class edge share = {xshare*100:.2f}%  "
          f"(mechanism predicts effect size scales with this)")
    del Kc

    Kpc = None
    if "a2_perclass" in a.arms:
        t0 = time.time()
        print("  building per-class (block-diagonal) kernel:")
        Kpc = build_perclass_kernel(Z, y)
        print(f"  per-class kernel nnz={Kpc.nnz/1e6:.1f}M "
              f"({time.time()-t0:.0f}s)")

    if a.assert_a2_identity:
        print("\n  --- identity check: (no mask, uniform w) == graph_a2 ---")
        s_ref, _ = run_arm("graph_a2", K, Kpc, y, C, bpc, w_cls, w1, FG_P,
                           a.lam, a.seed, verbose=False)
        a_id, _ = build_kernel_variant(K, y, False, False, a.seed)
        Ft = pool_value(a_id, w1)
        s_id, _ = greedy_blended(K, a_id, y, bpc, w1 / FG_P, w1 / Ft, a.lam,
                                 verbose=False)
        same = set(s_ref.tolist()) == set(s_id.tolist())
        print(f"    set_match={same}  order_match="
              f"{bool(np.array_equal(s_ref, s_id))}")
        if not same:
            raise SystemExit("identity check FAILED -- the 2x2 factorial's "
                             "top-left cell is not graph_a2; rerun it as a "
                             "real arm instead of reusing the archive")

    results = []
    for arm in a.arms:
        print(f"\n  --- arm {arm} ---")
        t0 = time.time()
        sel, gains = run_arm(arm, K, Kpc, y, C, bpc, w_cls, w1, FG_P, a.lam,
                             a.seed)
        cnt = np.bincount(y[sel], minlength=C)
        assert sel.size == bpc * C, f"{sel.size} != {bpc*C}"
        assert cnt.min() == cnt.max() == bpc, f"quota violated: {cnt}"
        name = f"{ds}_r{ratio}_{arm}_s42.npy"
        np.save(os.path.join(a.out_dir, name), sel.astype(np.int64))
        print(f"    wrote {name}  n={sel.size}  quota ok "
              f"({time.time()-t0:.0f}s)")
        results.append(dict(arm=arm, n=int(sel.size),
                            gain_first=float(gains[0]),
                            gain_last=float(gains[-1])))

    # selection-space report, including Jaccard against the round-1 arms that
    # are already trained -- a new arm that is identical to an old one costs
    # nothing to discover HERE and 15 GPU-hours to discover later.
    def jac(p, q):
        sp_, sq = set(p.tolist()), set(q.tolist())
        return len(sp_ & sq) / max(1, len(sp_ | sq))

    loaded = {}
    for arm in list(dict.fromkeys(list(a.arms) + ARMS_BASE
                                  + ["tdgs_d", "tdgs_du", "tdgs_perm"])):
        p = os.path.join(a.out_dir, f"{ds}_r{ratio}_{arm}_s42.npy")
        if os.path.exists(p):
            loaded[arm] = np.load(p)
    names = list(loaded)
    print(f"\n  pairwise Jaccard (random floor {ratio/(2-ratio):.4f}, "
          f"cross-method reference ~0.05):")
    print("    " + "".join(f"{n[:11]:>13s}" for n in names))
    jmat = {}
    for p_ in names:
        row = []
        for q_ in names:
            v = 1.0 if p_ == q_ else jac(loaded[p_], loaded[q_])
            jmat[f"{p_}|{q_}"] = v
            row.append(f"{v:>13.4f}")
        print(f"    {p_[:11]:>11s}" + "".join(row))
    for x, msg in (("a2_perclass",
                    "a2_perclass vs tdgs_cls near 1.0 => tdgs_cls IS the "
                    "author's per-class cell; the positive is not novel"),
                   ("tdgs_cls_perm",
                    "tdgs_cls_perm vs tdgs_cls near 1.0 => the permutation "
                    "did not bite and the null is uninformative")):
        if x in loaded and "tdgs_cls" in loaded:
            print(f"    NOTE {msg}  (observed "
                  f"{jmat[f'{x}|tdgs_cls']:.4f})")

    rep = os.path.join(a.report, f"{ds}_r{ratio}_r2.json")
    json.dump(dict(dataset=ds, ratio=ratio, n=len(y), C=C, bpc=bpc,
                   lam=a.lam, k_neighbors=K_NEIGHBORS, k_hops=K_HOPS,
                   cross_class_edge_share=xshare, embedding=emb_path,
                   probe_used=False, arms=results, jaccard=jmat),
              open(rep, "w"), indent=2)
    print(f"\n  wrote {rep}")


if __name__ == "__main__":
    sys.path.insert(0, "/project/prj-sis01/xuxiaoyu/graph_select")
    main()
