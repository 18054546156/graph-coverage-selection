#!/usr/bin/env python
"""Round 3 selectors: encoder-robust class-scoped coverage (route R1) and the
matched hybrid-per-class control (route R2). Probe-free, label-aware only through
the training labels that every arm in this benchmark already uses.

Per view v (a frozen encoder), the round-2 TDGS objective, normalised to [0,1]:

    F_v(S) = (1-lam) * F_G^v(S)/F_G^v(P) + lam * F_task^v(S)/F_task^v(P)
    F_G^v(S)    = sum_i max_{j in S} K^v_ij                   (Graph-A2, global kNN)
    F_task^v(S) = sum_i w_i max_{j in S} 1[y_i=y_j] Kbar^v_ij,  w_i = 1/(C n_{y_i})

Arms (all under the author's equal per-class quota, lazy greedy, author's
quota-exhaustion rule):

  hpc_cls   R2 matched control: global F_G (UNI) + task branch on the author's
            PER-CLASS-rebuilt graph (block-diagonal K_pc), class-flat weights.
            Isolates "global graph for the task credit" from "a blended global
            term": differs from tdgs_cls ONLY in which graph the task branch uses.
  a2_<v>    Graph-A2 on view v              (lam = 0)
  cls_<v>   tdgs_cls on view v              (lam = 0.5)
  cls_cat   tdgs_cls on the concatenation of the L2-normalised views, one graph
            (the naive multi-encoder baseline)
  mv_mean   (1/|V|) sum_v F_v               -- monotone submodular, greedy 1/2
  mv_rob    SATURATE-style robust variant: for each truncation level c on a
            fixed grid, greedily maximise sum_v min(F_v/F_v(S_v), c), S_v = view
            v's own single-view selection (Amendment 1); keep the S with the
            largest min_v F_v(S)/F_v(S_v). Train-pool quantity only.

Self-checks (abort on failure):
  * the generic multi-term greedy with the single UNI view reproduces
    greedy_blended bit-for-bit for graph_a2 and tdgs_cls;
  * quota exactly bpc per class for every arm.

Diagnostics written to the report JSON (no training, test split never loaded):
  per view: cross-class edge share of K, train kNN-LOO balanced accuracy on the
  k=50 graph, train->val kNN balanced accuracy (k=50 cosine, weighted vote);
  cross-view coverage matrix F_v(S_arm); pairwise Jaccard.
"""
import argparse
import heapq
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdgs_select import (EPS, LAMBDA, budget_per_class, build_graph_kernel,
                         greedy_blended, load_emb, load_labels, pool_value,
                         PROJECT)
from tdgs_select_r2 import build_kernel_variant, build_perclass_kernel

S_ROOT = "/mnt/prj01/hgrp-1502-5TB/tdgs_shared"
R3_EMB = f"{S_ROOT}/r3/emb"
VIEWS = ["uni", "dinov2", "clip"]
ARMS = ["hpc_cls", "a2_dinov2", "a2_clip", "cls_dinov2", "cls_clip",
        "cls_cat", "mv_mean", "mv_rob"]
# Amendment 1 (2026-09-28, before any R3 training): the original absolute grid
# [0.80 .. 1.0] never bound -- F_v(S) is 0.63-0.73 at 2%, so every cap gave
# sum_v F_v, i.e. mv_rob == mv_mean exactly (staging 35236, all 3 datasets).
# The cap now acts on RELATIVE coverage F_v(S)/F_v(S_v), where S_v is view v's
# own single-view greedy (cls_<v>; cls_uni == tdgs_cls). Train-pool only.
ROB_GRID = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]


# ------------------------------------------------------------------ inputs
def l2n(Z):
    return (Z / (np.linalg.norm(Z, axis=1, keepdims=True) + EPS)).astype(np.float32)


def load_view(ds, v, split="train"):
    if v == "uni" and split == "train":
        return l2n(load_emb(ds)[0])
    f = np.load(os.path.join(R3_EMB, f"{ds}_{split}_{v}_224.npz"))
    return l2n(f["embeddings"])


def val_labels(ds):
    return np.load(f"{PROJECT}/data/medmnist/{ds}.npz")["val_labels"].ravel().astype(int)


def bal_acc(y, p, C):
    r = [np.mean(p[y == c] == c) for c in range(C) if np.any(y == c)]
    return float(np.mean(r))


# ------------------------------------------------------------- diagnostics
def knn_adjacency(Z, k=50):
    """Author's symmetrised kNN adjacency (no self loops) -- the base of K."""
    from graphcov.run.graph import build_adjacency_matrix
    from graphcov.run.selection import build_knn_graph
    ki, kd = build_knn_graph(Z, k, verbose=False)
    return build_adjacency_matrix(ki, kd, len(Z))


def knn_loo_ba(A, y, C):
    """Weighted vote over the stored kNN edges (self never stored)."""
    Y = sp.csr_matrix((np.ones(len(y)), (np.arange(len(y)), y)), shape=(len(y), C))
    votes = (A.tocsr() @ Y).toarray()
    return bal_acc(y, votes.argmax(1), C)


def val_knn_ba(Ztr, ytr, Zva, yva, C, k=50):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T = torch.from_numpy(Ztr).to(dev)
    Yoh = torch.nn.functional.one_hot(torch.from_numpy(ytr).to(dev), C).float()
    pred = []
    for s in range(0, len(Zva), 2048):
        q = torch.from_numpy(Zva[s:s + 2048]).to(dev)
        sim = q @ T.T
        v, i = sim.topk(k, dim=1)
        pred.append((Yoh[i] * v.clamp_min(0).unsqueeze(-1)).sum(1).argmax(1).cpu().numpy())
    return bal_acc(yva, np.concatenate(pred), C)


# ------------------------------------------------------------------ greedy
def greedy_multi(terms, y, bpc, objective=None, verbose=True):
    """Lazy greedy on  sum_t coef_t * sum_i w_t,i * max_{j in S} M_t,ij
    (or a truncated wrapper, see `objective`), equal per-class quota,
    author's exhaustion rule.

    With terms == [(K, w1/FG, 1-lam), (a, w2/Ft, lam)] the arithmetic is the
    same float32/float64 sequence as tdgs_select.greedy_blended, which the
    driver asserts. `objective`, if given, is (V, idx_of_term -> view, c): the
    gain of view v is min(F_v + g_v, c) - min(F_v, c), which stays submodular.
    """
    n = len(y)
    classes = np.unique(y)
    total = bpc * len(classes)
    T = []
    for M, w, coef in terms:
        M = M.tocsc()
        T.append((M.data.astype(np.float32), M.indices, M.indptr,
                  w.astype(np.float32), np.float32(coef)))
    covs = [np.zeros(n, np.float32) for _ in T]
    eligible = np.ones(n, bool)
    counts = {int(c): 0 for c in classes}
    selected, gains = [], []

    if objective is not None:
        nview, view_of, cap = objective
        Fv = np.zeros(nview, np.float64)

    def raw_gains(j):
        out = []
        for (d, r_, p, w, c), cov in zip(T, covs):
            if c > 0:
                s, e = p[j], p[j + 1]
                r = r_[s:e]
                out.append(c * float(np.dot(w[r], np.maximum(d[s:e] - cov[r], 0.0))))
            else:
                out.append(0.0)
        return out

    def gain_of(j):
        g = raw_gains(j)
        if objective is None:
            tot = 0.0
            for x in g:
                tot += x
            return tot
        gv = np.zeros(nview)
        for t, x in enumerate(g):
            gv[view_of[t]] += x
        return float(np.sum(np.minimum(Fv + gv, cap) - np.minimum(Fv, cap)))

    g0 = np.zeros(n, np.float64)
    if objective is None:
        for d, r_, p, w, c in T:
            if c > 0:
                cols = np.repeat(np.arange(n, dtype=np.intp), np.diff(p))
                g0 += c * np.bincount(cols, weights=np.maximum(d, 0.0) * w[r_],
                                      minlength=n)
    else:
        gv0 = np.zeros((nview, n))
        for t, (d, r_, p, w, c) in enumerate(T):
            if c > 0:
                cols = np.repeat(np.arange(n, dtype=np.intp), np.diff(p))
                gv0[view_of[t]] += c * np.bincount(
                    cols, weights=np.maximum(d, 0.0) * w[r_], minlength=n)
        g0 = np.minimum(gv0, cap).sum(0)

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
        if objective is not None:
            g = raw_gains(best_j)
            for t, x in enumerate(g):
                Fv[view_of[t]] += x
        selected.append(best_j)
        gains.append(best_g)
        c = int(y[best_j])
        counts[c] += 1
        eligible[best_j] = False
        if counts[c] >= bpc:
            eligible[y == c] = False
        for (d, r_, p, w, cf), cov in zip(T, covs):
            s, e = p[best_j], p[best_j + 1]
            r = r_[s:e]
            cov[r] = np.maximum(cov[r], d[s:e])
        if verbose and (it + 1) % max(1, total // 5) == 0:
            print(f"    {it+1}/{total} picks  gain={best_g:.6g}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return np.asarray(selected, int), np.asarray(gains, float)


# ------------------------------------------------------------- view objects
class View:
    """Kernels and normalisers of one encoder's TDGS objective."""

    def __init__(self, name, Z, y, w_cls, keep_base=False):
        t = time.time()
        # tdgs_select.py on the cluster is the FROZEN round-1 file, whose
        # build_graph_kernel has no return_base -- so the raw kNN adjacency for
        # the LOO diagnostic is rebuilt here with the same graphcov calls.
        self.K = build_graph_kernel(Z)
        self.A = knn_adjacency(Z) if keep_base else None
        self.a, _ = build_kernel_variant(self.K, y, True, False)
        n = len(y)
        self.w1 = np.ones(n, np.float32)
        self.w2 = w_cls
        self.FG = pool_value(self.K, self.w1)
        self.Ft = pool_value(self.a, self.w2)
        Kc = self.K.tocoo()
        self.xshare = float((y[Kc.row] != y[Kc.col]).mean())
        del Kc
        self.name = name
        print(f"  view {name}: K nnz={self.K.nnz/1e6:.1f}M a nnz="
              f"{self.a.nnz/1e6:.1f}M xshare={self.xshare*100:.2f}% "
              f"({time.time()-t:.0f}s)", flush=True)

    def terms(self, lam, scale=1.0):
        return [(self.K, self.w1 / self.FG, scale * (1 - lam)),
                (self.a, self.w2 / self.Ft, scale * lam)]

    def value(self, S, lam):
        """F_v(S) in [0,1]."""
        S = np.asarray(S)
        gK = pool_value(self.K[:, S], self.w1) / self.FG
        ga = pool_value(self.a[:, S], self.w2) / self.Ft
        return float((1 - lam) * gK + lam * ga), float(gK), float(ga)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ratio", type=float, default=0.02)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--lam", type=float, default=LAMBDA)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--ref-sel", default=f"{S_ROOT}/sel",
                    help="round-1/2 selections, for the identity check and J")
    ap.add_argument("--tag", default="",
                    help="report suffix, so a partial re-run never overwrites the full report")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.report, exist_ok=True)
    ds, ratio, lam = a.dataset, a.ratio, a.lam

    y = load_labels(ds)
    C = int(y.max()) + 1
    bpc = budget_per_class(len(y), ratio, C)
    cls_n = np.bincount(y, minlength=C).astype(np.float32)
    w_cls = (1.0 / (C * cls_n))[y].astype(np.float32)
    yva = val_labels(ds)
    print(f"\n[{ds} r={ratio}] n={len(y)} C={C} bpc={bpc} lam={lam}", flush=True)

    rep = dict(dataset=ds, ratio=ratio, n=len(y), C=C, bpc=bpc, lam=lam,
               views={}, arms={}, checks={})

    Zs = {v: load_view(ds, v) for v in VIEWS}
    for v in VIEWS:
        assert len(Zs[v]) == len(y), (v, len(Zs[v]), len(y))
    views = {}
    for v in VIEWS:
        views[v] = View(v, Zs[v], y, w_cls, keep_base=True)
        Zva = load_view(ds, v, "val")
        rep["views"][v] = dict(
            xshare=views[v].xshare,
            knn_loo_ba=knn_loo_ba(views[v].A, y, C),
            val_knn_ba=val_knn_ba(Zs[v], y, Zva, yva, C))
        views[v].A = None
        print(f"    {v}: {rep['views'][v]}", flush=True)

    # ---- identity check: generic greedy == greedy_blended on the UNI view
    U = views["uni"]
    ref = {}
    for arm, l in (("graph_a2", 0.0), ("tdgs_cls", lam)):
        s_old, _ = greedy_blended(U.K, U.a if l > 0 else sp.csc_matrix(U.K.shape, dtype=np.float32),
                                  y, bpc, U.w1 / U.FG,
                                  (U.w2 / U.Ft) if l > 0 else np.zeros(len(y), np.float32),
                                  l, verbose=False)
        s_new, _ = greedy_multi(U.terms(l), y, bpc, verbose=False)
        same = bool(np.array_equal(s_old, s_new))
        p = os.path.join(a.ref_sel, f"{ds}_r{ratio}_{arm}_s42.npy")
        arch = (set(np.load(p).tolist()) == set(s_new.tolist())) if os.path.exists(p) else None
        rep["checks"][arm] = dict(order_match_greedy_blended=same,
                                  set_match_staged_selection=arch)
        print(f"  identity {arm}: greedy_blended order match={same}  "
              f"staged-set match={arch}", flush=True)
        if not same:
            raise SystemExit(f"generic greedy != greedy_blended for {arm}")
        ref[arm] = s_new

    sels = dict(ref)
    for arm in a.arms:
        t = time.time()
        print(f"\n  --- arm {arm} ---", flush=True)
        if arm == "hpc_cls":
            Kpc = build_perclass_kernel(Zs["uni"], y, verbose=False)
            apc, _ = build_kernel_variant(Kpc, y, True, False)
            del Kpc
            terms = [(U.K, U.w1 / U.FG, 1 - lam),
                     (apc, w_cls / pool_value(apc, w_cls), lam)]
            sel, _ = greedy_multi(terms, y, bpc)
            del apc, terms
        elif arm.startswith("a2_") or (arm.startswith("cls_") and arm != "cls_cat"):
            v = arm.split("_", 1)[1]
            sel, _ = greedy_multi(views[v].terms(0.0 if arm.startswith("a2_") else lam),
                                  y, bpc)
        elif arm == "cls_cat":
            Zc = np.concatenate([Zs[v] for v in VIEWS], 1) / np.sqrt(len(VIEWS))
            Vc = View("cat", Zc.astype(np.float32), y, w_cls)
            del Zc
            sel, _ = greedy_multi(Vc.terms(lam), y, bpc)
            rep["views"]["cat"] = dict(xshare=Vc.xshare)
            del Vc
        elif arm == "mv_mean":
            terms = sum((views[v].terms(lam, 1.0 / len(VIEWS)) for v in VIEWS), [])
            sel, _ = greedy_multi(terms, y, bpc)
        elif arm == "mv_rob":
            # reference S_v: this run's cls_<v>, else the file from a prior run
            refv = {}
            for v in VIEWS:
                name = "tdgs_cls" if v == "uni" else f"cls_{v}"
                s_v = sels.get(name)
                if s_v is None:
                    s_v = np.load(os.path.join(a.out_dir, f"{ds}_r{ratio}_{name}_s42.npy"))
                refv[v] = views[v].value(s_v, lam)[0]
            rep["mv_rob_ref"] = refv
            terms = sum((views[v].terms(lam, 1.0 / refv[v]) for v in VIEWS), [])
            view_of = [i // 2 for i in range(len(terms))]
            best, grid = None, []
            for cap in ROB_GRID:
                s_c, _ = greedy_multi(terms, y, bpc,
                                      objective=(len(VIEWS), view_of, cap),
                                      verbose=False)
                vals = [views[v].value(s_c, lam)[0] / refv[v] for v in VIEWS]
                grid.append(dict(cap=cap, min=min(vals), vals=vals))
                print(f"    cap={cap:.2f}  F_v/F_v(S_v)={np.round(vals, 4)}  "
                      f"min={min(vals):.4f}", flush=True)
                if best is None or min(vals) > best[0]:
                    best = (min(vals), cap, s_c)
            sel = best[2]
            rep["mv_rob_grid"] = grid
            rep["mv_rob_cap"] = best[1]
        else:
            raise SystemExit(f"unknown arm {arm}")
        cnt = np.bincount(y[sel], minlength=C)
        assert sel.size == bpc * C and cnt.min() == cnt.max() == bpc, cnt
        np.save(os.path.join(a.out_dir, f"{ds}_r{ratio}_{arm}_s42.npy"),
                sel.astype(np.int64))
        sels[arm] = sel
        print(f"    wrote {arm} ({time.time()-t:.0f}s)", flush=True)

    # ---- cross-view coverage matrix and Jaccard (selection space only)
    for arm, s in sels.items():
        rep["arms"][arm] = {v: views[v].value(s, lam) for v in VIEWS}
    for extra in ("a2_perclass", "tdgs_mask"):
        p = os.path.join(a.ref_sel, f"{ds}_r{ratio}_{extra}_s42.npy")
        if os.path.exists(p):
            sels[extra] = np.load(p)
    for arm in ARMS + ["mv_rob_v0"]:        # prior-run R3 arms, for J only
        p = os.path.join(a.out_dir, f"{ds}_r{ratio}_{arm}_s42.npy")
        if arm not in sels and os.path.exists(p):
            sels[arm] = np.load(p)
    names = list(sels)
    rep["jaccard"] = {f"{p}|{q}": len(set(sels[p]) & set(sels[q])) /
                      len(set(sels[p]) | set(sels[q]))
                      for i, p in enumerate(names) for q in names[i + 1:]}
    out = os.path.join(a.report, f"{ds}_r{ratio}_r3{a.tag}.json")
    json.dump(rep, open(out, "w"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\n  wrote {out}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, "/project/prj-sis01/xuxiaoyu/graph_select")
    main()
