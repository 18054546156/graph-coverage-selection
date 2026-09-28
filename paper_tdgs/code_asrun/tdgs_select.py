#!/usr/bin/env python
"""Task-Demand Graph Selection (TDGS) -- source-only, fixed per-class quota.

THE QUESTION
------------
Graph-A2 covers SAMPLES. Two samples of the same class that sit close together
in UNI space are treated as interchangeable: covering one discharges the
obligation to cover the other. This script tests whether that substitution is
legitimate, by asking what each sample still needs to be LEARNED, and refusing
to let a candidate discharge a demand it does not actually address.

Concretely: two class-A samples can both have p(A)=0.6 -- identical difficulty --
while one is losing mass to class B and the other to class C. Under a scalar
difficulty score they are the same. They are not the same training problem.

INFORMATION BUDGET (this is the point of the method)
---------------------------------------------------
Source pool inputs + source labels ONLY. No target/test inputs, no test labels,
no test confusion matrix. This is deliberately a STRICTER budget than TD-Cover,
which reads unlabeled target inputs. TDGS is therefore a legal entry in the
author's Table 1 protocol, and its comparison set is the author's eight
baselines -- not TD-Cover, whose results are NOT transferable to this method and
are not claimed here.

THE OBJECTIVE
-------------
Out-of-fold proxy probabilities p_i (a linear head on frozen UNI, trained
without the fold containing i -- in-fold probabilities would measure
memorisation, not demand):

    d_i = 1 - p_i(y_i)                          demand STRENGTH, in [0,1)
    v_i(c) = p_i(c) for c != y_i, else 0
    u_i = v_i / (||v_i||_2 + eps)               demand DIRECTION, unit, >= 0

Demand-support relation, for a demand point i and a candidate j:

    a_ij = 1[y_i = y_j] * Kbar_ij * <u_i, u_j>

so j supports i only if it is the same class, geometrically near, AND confusable
in the same direction. Task-demand coverage:

    F_task(S) = (1/C) sum_c (1/|P_c|) sum_{i in P_c} d_i * max_{j in S_c} a_ij

Note WHERE d_i sits: on the DEMAND point, not the candidate. This is not
difficulty-ranked selection -- an easy candidate that supports many hard demands
of a given direction still wins. That distinction matters because
difficulty-as-criterion is already closed on this corpus (-6.72pp, and a
within-class permutation control at exactly zero).

Blend with the author's own coverage F_G, both normalised on the full pool:

    F_new(S) = (1-lam) * F_G(S)/F_G(P) + lam * F_task(S)/F_task(P)

Both terms are monotone non-decreasing and submodular (each is a sum of
non-negative-weighted max-coverage terms), so the sum is, and lazy greedy under
the per-class quota is the same algorithm the author already uses.

THE ABLATION LADDER (each arm adds exactly one thing)
-----------------------------------------------------
    graph_a2     lam=0. The author's method, regenerated here so the comparison
                 is not across directories.
    tdgs_cls     lam>0 with d_i = 1 and <u_i,u_j> = 1, i.e. coverage restricted
                 to same-class demand. Isolates "does using the source LABEL to
                 forbid cross-class credit help?" -- note the author's global
                 mode explicitly grants that credit (selection.py: "boundary
                 samples get credit for covering nearby other-class samples").
    tdgs_d       adds d_i. Isolates scalar demand weighting.
    tdgs_du      adds u_i. The full method.
    tdgs_perm    u_i permuted WITHIN class, d_i left aligned to its own sample.
                 Same marginal distribution of directions, destroyed pairing.
                 If tdgs_du does not beat this, the direction structure is not
                 what is working and the method must be re-described as
                 uncertainty-weighted coverage.

PRE-REGISTERED VALIDITY CHECK (decides whether the proxy is usable AT ALL)
--------------------------------------------------------------------------
The whole method assumes a light head on frozen UNI recovers the same
discriminative structure the final ResNet-18 will face. That is an assumption,
not a fact, and this corpus already contains the answer to compare against:
pre-experiment B measured the final models' modal worst class and its top
confusion partner, stable across seeds AND across four selection methods
(pathmnist worst=7; organamnist 4->5; bloodmnist 3->5; organsmnist worst=4;
tissuemnist worst=1 with a perfect four-pair structure 0<->1, 2<->5, 3<->4,
6<->7).

So the script computes the OOF confusion matrix and reports its worst class and
top partner against those. This is NOT used to build the selector -- it is a
report. If the proxy disagrees on most datasets, the honest conclusion is
proxy/learner mismatch, and the fix is a better proxy, NOT importing the test
confusion structure (which would be leakage).

Recorded in advance: the probe is expected to disagree on tissuemnist, whose
probe BA caps near 0.48.

WHAT THIS SCRIPT DOES NOT DO
    It does not train ResNet-18 and it does not claim any accuracy result.
    It writes selections. Every arm above is untested.

Usage:
    python tdgs_select.py --dataset pathmnist --ratio 0.02
"""
import argparse
import heapq
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse import diags

PROJECT = "/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab"
K_NEIGHBORS = 50          # protocol.json selection_config; confirmed by replay
K_HOPS = 2
LAMBDA = 0.5              # fixed a priori; see scale report in main()
N_FOLDS = 5
EPS = 1e-12

EMB_ROOTS = [
    f"{PROJECT}/table1_clean_only_20260922/cache/embeddings_img224_smokefull",
    "/mnt/prj01/hgrp-1502-5TB/xuxiaoyu/reliability_medmnistc_ab/"
    "table1_clean_only_20260922/cache/embeddings_img224_smokefull",
]

# From pre-experiment B (`pilots/confusion_struct.py`), final ResNet-18 models,
# stable across training seeds and across graph_a2/random/herding/facility.
FINAL_MODEL_STRUCTURE = {
    "pathmnist":   {"worst": 7, "partner": None},
    "organamnist": {"worst": 4, "partner": 5},
    "organsmnist": {"worst": 4, "partner": None},
    "bloodmnist":  {"worst": 3, "partner": 5},
    "tissuemnist": {"worst": 1, "partner": 0},
}


# ---------------------------------------------------------------- inputs
def load_emb(ds):
    want = f"{ds}_train_uni_224.npz"
    for root in EMB_ROOTS:
        p = os.path.join(root, want)
        if not os.path.exists(p):
            continue
        f = np.load(p)
        key = next(k for k in ("embeddings", "features", "emb", "arr_0")
                   if k in f.files)
        return f[key].astype(np.float32), p
    raise FileNotFoundError(f"no UNI embedding for {ds}")


def load_labels(ds):
    return np.load(f"{PROJECT}/data/medmnist/{ds}.npz")[
        "train_labels"].ravel().astype(int)


def budget_per_class(n, ratio, n_classes):
    """Author's formula. Verified: pathmnist 89996 @2% -> 1799//9 = 199, and
    199*9 = 1791 = the archived n_selected."""
    return int(n * ratio) // n_classes


# ------------------------------------------------------- OOF proxy head
def _fit_head(Zt, yt, tr, n_classes, dev, min_steps, lr, wd, bs, seed):
    """One linear head, trained for a STEP budget rather than an epoch budget.

    Epoch budgets are a trap here: the pools span 11,959 to 165,466 samples, so
    a fixed epoch count gives 3 optimiser steps on bloodmnist and 40 on
    tissuemnist. The first version of this script used 30 epochs at batch 4096
    and organsmnist got ~90 total steps, which produced an out-of-fold balanced
    accuracy of 0.25 -- that was an undertrained head, not a hard dataset.
    Inputs are L2-normalised, so per-coordinate scale is ~1/sqrt(1024) and the
    learning rate has to be correspondingly large.
    """
    import torch
    import torch.nn as nn

    g = torch.Generator(device=dev.type).manual_seed(seed)
    head = nn.Linear(Zt.shape[1], n_classes).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    Ztr, ytr = Zt[tr].to(dev), yt[tr].to(dev)
    spe = max(1, (len(tr) + bs - 1) // bs)
    epochs = max(1, -(-min_steps // spe))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * spe)
    head.train()
    for _ in range(epochs):
        perm = torch.randperm(len(tr), device=dev, generator=g)
        for s in range(0, len(tr), bs):
            b = perm[s:s + bs]
            opt.zero_grad()
            lossf(head(Ztr[b]), ytr[b]).backward()
            opt.step()
            sched.step()
    del Ztr, ytr
    return head, epochs * spe


def _predict(head, Zt, idx, dev, bs=8192):
    import torch
    head.eval()
    with torch.no_grad():
        out = []
        for s in range(0, len(idx), bs):
            zb = Zt[idx[s:s + bs]].to(dev)
            out.append(torch.softmax(head(zb), 1).cpu())
        return torch.cat(out).numpy()


def oof_probabilities(Z, y, n_classes, folds=N_FOLDS, seed=0, device="cuda",
                      min_steps=4000, lr=1e-2, wd=1e-4, bs=1024, verbose=True):
    """Stratified K-fold linear probe on frozen UNI -> out-of-fold p(y|x).

    A linear head is chosen over anything deeper on purpose: it is cheap, it is
    the same probe family this project has already characterised, and a weak
    proxy that is honestly weak is more useful than a strong one that has
    partially memorised the pool. Each fold's head never sees its own fold.

    Also fits one head on the FULL pool and returns its in-sample accuracy. That
    number is the undertraining detector: if in-sample accuracy is also low, the
    optimiser failed; if in-sample is high while out-of-fold is low, the
    representation genuinely does not separate the classes.
    """
    import torch

    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=int)
    for c in range(n_classes):          # stratify so every fold sees every class
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % folds

    P = np.zeros((len(y), n_classes), dtype=np.float32)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    Zt = torch.from_numpy(Z)
    yt = torch.from_numpy(y).long()

    t0 = time.time()
    steps = 0
    for f in range(folds):
        tr = np.flatnonzero(fold_of != f)
        te = np.flatnonzero(fold_of == f)
        head, steps = _fit_head(Zt, yt, tr, n_classes, dev, min_steps, lr, wd,
                                bs, seed * 100 + f)
        P[te] = _predict(head, Zt, te, dev)
        del head
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    allidx = np.arange(len(y))
    head, _ = _fit_head(Zt, yt, allidx, n_classes, dev, min_steps, lr, wd, bs,
                        seed)
    insample = float((_predict(head, Zt, allidx, dev).argmax(1) == y).mean())
    del head
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    if verbose:
        print(f"  OOF probe: {folds} folds x {steps} steps each  "
              f"in-sample acc {insample:.4f}  ({time.time()-t0:.0f}s)")
    return P, insample


def demand_from_probs(P, y, n_classes):
    """d_i (strength) and u_i (unit direction over competing classes)."""
    n = len(y)
    d = 1.0 - P[np.arange(n), y]
    V = P.copy()
    V[np.arange(n), y] = 0.0
    nrm = np.linalg.norm(V, axis=1, keepdims=True)
    U = V / (nrm + EPS)
    return d.astype(np.float32), U.astype(np.float32)


def report_proxy_validity(P, y, n_classes, ds, insample):
    """Compare the PROXY's confusion structure with the FINAL MODELS'.
    Reported, never used to build the selector."""
    yh = P.argmax(1)
    M = np.zeros((n_classes, n_classes))
    np.add.at(M, (y, yh), 1.0)
    nc = M.sum(1, keepdims=True)
    R = np.divide(M, nc, out=np.zeros_like(M), where=nc > 0)
    rec = np.diag(R)
    w = int(np.argmin(rec))
    off = R[w].copy(); off[w] = -1.0
    p = int(np.argmax(off))
    ba = float(rec.mean())
    exp = FINAL_MODEL_STRUCTURE.get(ds, {})
    ok_w = (exp.get("worst") == w)
    ok_p = (exp.get("partner") is None) or (exp.get("partner") == p)
    gap = insample - ba
    print(f"  PROXY VALIDITY  OOF balanced acc {ba:.4f}  "
          f"(in-sample {insample:.4f}, gap {gap:+.4f})")
    if insample < 0.60:
        print("    WARNING: in-sample accuracy is low too -- this is an "
              "UNDERTRAINED head, not a hard dataset. Fix the probe before "
              "reading anything below.")
    print(f"    worst class {w} (final models: {exp.get('worst')}) "
          f"{'MATCH' if ok_w else 'MISMATCH'};   "
          f"top partner {w}->{p} (final: {exp.get('worst')}->"
          f"{exp.get('partner')}) "
          f"{'ok' if ok_p else 'MISMATCH'}")
    print(f"    demand d_i: {d_stats(P, y)}")
    return dict(oof_ba=ba, insample_acc=insample, proxy_worst=w,
                proxy_partner=p, final_worst=exp.get("worst"),
                final_partner=exp.get("partner"),
                worst_match=bool(ok_w), partner_match=bool(ok_p))


def d_stats(P, y):
    d = 1.0 - P[np.arange(len(y)), y]
    q = np.percentile(d, [10, 50, 90])
    return (f"{d.mean():.4f}  p10/p50/p90 {q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f}  "
            f"frac(d<0.05)={float((d < 0.05).mean()):.3f}")


# ------------------------------------------------------------- kernels
def build_graph_kernel(Z, k_neighbors=K_NEIGHBORS, k_hops=K_HOPS):
    """Exactly `_select_graph_a2`'s global_selection branch: global kNN graph,
    symmetric normalisation, K = sum_{i=1}^{k_hops} A_sym^i."""
    from graphcov.run.selection import build_knn_graph
    from graphcov.run.graph import build_adjacency_matrix
    n = len(Z)
    ki, kd = build_knn_graph(Z, k_neighbors, verbose=False)
    A = build_adjacency_matrix(ki, kd, n)
    rs = np.array(A.sum(axis=1)).flatten()
    rs[rs == 0] = 1
    Di = diags(1.0 / np.sqrt(rs))
    A_sym = Di @ A @ Di
    K = A_sym.copy()
    Ap = A_sym.copy()
    for _ in range(k_hops - 1):
        Ap = Ap @ A_sym
        K = K + Ap
    return K.tocsc()


def build_demand_kernel(K, y, U, use_direction, chunk=4_000_000):
    """a_ij = 1[y_i=y_j] * Kbar_ij * <u_i,u_j>, returned CSC (same sparsity
    pattern as K, minus dropped entries).

    Chunked over nonzeros: U[rows]*U[cols] materialises nnz x C floats, which is
    3GB+ for pathmnist at k=50 if done in one shot.
    """
    Kc = K.tocoo()
    rows, cols, vals = Kc.row, Kc.col, Kc.data.astype(np.float32)
    kmax = float(vals.max()) if len(vals) else 1.0
    vals = vals / (kmax + EPS)                       # Kbar in [0,1]

    keep = y[rows] == y[cols]                        # class-compatibility
    rows, cols, vals = rows[keep], cols[keep], vals[keep]

    if use_direction:
        dots = np.empty(len(vals), dtype=np.float32)
        for s in range(0, len(vals), chunk):
            e = min(s + chunk, len(vals))
            dots[s:e] = np.einsum("ij,ij->i", U[rows[s:e]], U[cols[s:e]])
        vals = vals * np.maximum(dots, 0.0)

    nz = vals > 0
    a = sp.csc_matrix((vals[nz], (rows[nz], cols[nz])), shape=K.shape)
    return a, kmax


# -------------------------------------------------------------- greedy
def greedy_blended(K, a, y, bpc, w1, w2, lam, verbose=True):
    """Lazy greedy on  F = (1-lam)*sum_i w1_i*max_{j in S} K_ij
                        + lam    *sum_i w2_i*max_{j in S} a_ij,
    under an equal per-class quota.

    Two coverage vectors, one heap. Quota exhaustion follows the author's rule
    exactly: once a class is full, every sample of that class becomes
    ineligible (graphcov/run/selection.py).
    """
    n = len(y)
    classes = np.unique(y)
    total = bpc * len(classes)

    K = K.tocsc(); a = a.tocsc()
    kd, kr, kp = K.data.astype(np.float32), K.indices, K.indptr
    ad, ar, ap = a.data.astype(np.float32), a.indices, a.indptr
    c1 = np.float32(1.0 - lam)
    c2 = np.float32(lam)
    w1 = w1.astype(np.float32); w2 = w2.astype(np.float32)

    cov1 = np.zeros(n, np.float32)
    cov2 = np.zeros(n, np.float32)
    eligible = np.ones(n, bool)
    counts = {int(c): 0 for c in classes}
    selected, gains = [], []

    def gain_of(j):
        g = 0.0
        if c1 > 0:
            s, e = kp[j], kp[j + 1]
            r = kr[s:e]
            g += c1 * float(np.dot(w1[r],
                                   np.maximum(kd[s:e] - cov1[r], 0.0)))
        if c2 > 0:
            s, e = ap[j], ap[j + 1]
            r = ar[s:e]
            g += c2 * float(np.dot(w2[r],
                                   np.maximum(ad[s:e] - cov2[r], 0.0)))
        return g

    # Initial gains, vectorised. With cov == 0 the max(0, .) is a no-op, so the
    # gain of every column is a single bincount over the stored entries -- the
    # same shortcut graphcov uses. Doing this with n Python-level column slices
    # instead costs minutes on tissuemnist (n=165k).
    def init_gains(data, rows, indptr, w):
        cols = np.repeat(np.arange(n, dtype=np.intp), np.diff(indptr))
        return np.bincount(cols, weights=np.maximum(data, 0.0) * w[rows],
                           minlength=n)

    g0 = np.zeros(n, np.float64)
    if c1 > 0:
        g0 += c1 * init_gains(kd, kr, kp, w1)
    if c2 > 0 and a.nnz:
        g0 += c2 * init_gains(ad, ar, ap, w2)

    heap = [(-g0[j], j) for j in range(n)]
    heapq.heapify(heap)
    last_eval = np.zeros(n, np.int64)

    t0 = time.time()
    for it in range(total):
        best_j, best_g = -1, 0.0
        while heap:
            ng, j = heapq.heappop(heap)
            if not eligible[j]:
                continue
            if last_eval[j] == it:
                best_j, best_g = j, -ng
                break
            gj = gain_of(j)
            last_eval[j] = it
            heapq.heappush(heap, (-gj, j))
        if best_j == -1:
            break

        selected.append(best_j)
        gains.append(best_g)
        c = int(y[best_j])
        counts[c] += 1
        eligible[best_j] = False
        if counts[c] >= bpc:
            eligible[y == c] = False

        s, e = kp[best_j], kp[best_j + 1]
        r = kr[s:e]
        cov1[r] = np.maximum(cov1[r], kd[s:e])
        s, e = ap[best_j], ap[best_j + 1]
        r = ar[s:e]
        cov2[r] = np.maximum(cov2[r], ad[s:e])

        if verbose and (it + 1) % max(1, total // 5) == 0:
            print(f"    {it+1}/{total} picks  gain={best_g:.6g}  "
                  f"({time.time()-t0:.0f}s)")
    return np.asarray(selected, int), np.asarray(gains, float)


def pool_value(M, w):
    """F(P) = sum_i w_i * max_{j in P} M_ij -- the full-pool normaliser.

    All entries are non-negative, so scipy's sparse row-max is exactly the
    max over the pool, and it returns 0 for rows with no stored entry.
    """
    if M.nnz == 0:
        return 0.0
    mx = np.asarray(M.tocsr().max(axis=1).todense()).ravel().astype(np.float32)
    return float(np.dot(w, mx))


# ----------------------------------------------------------------- main
ARMS = ["graph_a2", "tdgs_cls", "tdgs_d", "tdgs_du", "tdgs_perm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ratio", type=float, default=0.02)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--lam", type=float, default=LAMBDA)
    ap.add_argument("--seed", type=int, default=42,
                    help="selection seed; affects the OOF fold split and the "
                         "tdgs_perm permutation only (the rest is "
                         "deterministic)")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/td_sel"))
    ap.add_argument("--report", default=os.path.expanduser("~/tdgs_report"))
    a = ap.parse_args()

    ds, ratio = a.dataset, a.ratio
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.report, exist_ok=True)

    Z, emb_path = load_emb(ds)
    y = load_labels(ds)
    if len(Z) != len(y):
        raise SystemExit(f"[{ds}] truncated cache: {len(Z)} vs {len(y)}")
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + EPS)
    C = int(y.max()) + 1
    bpc = budget_per_class(len(y), ratio, C)
    print(f"\n{'='*86}\n[{ds} r={ratio}] n={len(y)} C={C} bpc={bpc} "
          f"budget={bpc*C}\n  emb <- {emb_path}\n{'='*86}")

    # ---- demand ---------------------------------------------------------
    P, insample = oof_probabilities(Z, y, C, seed=a.seed)
    validity = report_proxy_validity(P, y, C, ds, insample)
    d, U = demand_from_probs(P, y, C)

    # ---- kernels --------------------------------------------------------
    t0 = time.time()
    K = build_graph_kernel(Z)
    print(f"  graph kernel nnz={K.nnz/1e6:.1f}M ({time.time()-t0:.0f}s)")

    # per-class demand weights: 1/(C |P_c|) makes every class contribute
    # equally, matching the balanced-accuracy metric rather than class frequency
    cls_n = np.bincount(y, minlength=C).astype(np.float32)
    w_cls = (1.0 / (C * cls_n))[y].astype(np.float32)
    w1 = np.ones(len(y), np.float32)
    FG_P = pool_value(K, w1)

    rng = np.random.default_rng(a.seed)
    results = []
    for arm in a.arms:
        print(f"\n  --- arm {arm} ---")
        t0 = time.time()
        if arm == "graph_a2":
            a_mat = sp.csc_matrix(K.shape, dtype=np.float32)
            w2 = np.zeros(len(y), np.float32)
            lam = 0.0
        else:
            use_dir = arm in ("tdgs_du", "tdgs_perm")
            Uarm = U
            if arm == "tdgs_perm":
                Uarm = U.copy()
                for c in range(C):          # permute direction WITHIN class,
                    idx = np.flatnonzero(y == c)   # leaving d_i aligned
                    Uarm[idx] = U[rng.permutation(idx)]
            a_mat, kmax = build_demand_kernel(K, y, Uarm, use_dir)
            w2 = w_cls if arm == "tdgs_cls" else (w_cls * d)
            lam = a.lam
            Ft_P = pool_value(a_mat, w2)
            if Ft_P <= 0:
                print("    F_task(P) == 0 -- skipping arm")
                continue
            w2 = w2 / Ft_P
            print(f"    demand kernel nnz={a_mat.nnz/1e6:.1f}M  "
                  f"Kbar_max_scale={kmax:.4f}  F_task(P)={Ft_P:.6g}")

        sel, gains = greedy_blended(K, a_mat, y, bpc,
                                    w1 / FG_P, w2, lam)
        cnt = np.bincount(y[sel], minlength=C)
        assert sel.size == bpc * C, f"{sel.size} != {bpc*C}"
        assert cnt.min() == cnt.max() == bpc, f"quota violated: {cnt}"

        name = f"{ds}_r{ratio}_{arm}_s42.npy"
        np.save(os.path.join(a.out_dir, name), sel.astype(np.int64))
        print(f"    wrote {name}  n={sel.size}  quota ok  "
              f"({time.time()-t0:.0f}s)")
        results.append(dict(arm=arm, n=int(sel.size), lam=float(lam),
                            gain_first=float(gains[0]),
                            gain_last=float(gains[-1])))

    # ---- selection-space report (zero cost, decides nothing) ------------
    def jac(p, q):
        sp_, sq = set(p.tolist()), set(q.tolist())
        return len(sp_ & sq) / max(1, len(sp_ | sq))
    loaded = {}
    for r in results:
        loaded[r["arm"]] = np.load(
            os.path.join(a.out_dir, f"{ds}_r{ratio}_{r['arm']}_s42.npy"))
    names = list(loaded)
    print(f"\n  pairwise Jaccard (random-draw floor {ratio/(2-ratio):.4f}; "
          f"cross-method reference ~0.05):")
    print("    " + "".join(f"{n[:9]:>11s}" for n in names))
    jmat = {}
    for p_ in names:
        row = []
        for q_ in names:
            v = 1.0 if p_ == q_ else jac(loaded[p_], loaded[q_])
            jmat[f"{p_}|{q_}"] = v
            row.append(f"{v:>11.4f}")
        print(f"    {p_[:9]:>9s}" + "".join(row))
    if "tdgs_du" in loaded and "tdgs_perm" in loaded:
        print("    NOTE tdgs_du vs tdgs_perm near 1.0 would mean the direction "
              "term is inert and the ablation cannot separate them.")

    rep = os.path.join(a.report, f"{ds}_r{ratio}.json")
    json.dump(dict(dataset=ds, ratio=ratio, n=len(y), C=C, bpc=bpc,
                   lam=a.lam, k_neighbors=K_NEIGHBORS, k_hops=K_HOPS,
                   embedding=emb_path, proxy=validity, arms=results,
                   jaccard=jmat),
              open(rep, "w"), indent=2)
    print(f"\n  wrote {rep}")


if __name__ == "__main__":
    sys.path.insert(0, "/project/prj-sis01/xuxiaoyu/graph_select")
    main()
