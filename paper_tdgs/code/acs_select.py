#!/usr/bin/env python
"""ACS staging, E0 of report/proposal_acs_20261002.md (selections only; nothing is trained here).

ACS = per-class quantile filter (F2) -> per-class typicality/coverage budget split (F3) -> class-scoped
coverage (F1, the tdgs_cls objective on UNI, lam = 0.5, unchanged).

  agreement a_i : share of i's UNI k=50 neighbours (self removed) with i's label -- the same graph and
                  arithmetic as knnf_select.py (author build_knn_graph, CPU FAISS, author UNI cache).
  F2 (q)        : class c keeps { i in c : a_i >= Quantile_q(a_c) } (method="lower"), so every class keeps
                  >= (1-q)|c| >= bpc points; nothing is erased and nothing has to be restored.
  F3 (tau)      : k_h = floor(tau*bpc + 1/2) picks per class by the AUTHOR's _select_herding on the kept
                  points (typicality), then the remaining bpc - k_h per class by the tdgs_cls lazy greedy,
                  warm-started from the herding picks. The filter restricts who can be picked, not the
                  coverage demand (as MVF).

Identity checks (abort the (ds, ratio) on failure; files are written only after all checks pass):
  I1  greedy_init with no warm start and no filter == mv_select.greedy_multi (index order), and == the
      staged cls selection that was trained (set);                           => (q,tau) = (0,0) is cls
  I2  _select_herding on the full pool == the staged bench herding (index order); => (0,1) is herding
  I3  plurality-kept counts per class == *_knnf_report.json                  => same agreement as knnf
  I4  every staged set: bpc per class, unique, inside the filter.

(0,0) and (0,1) are not written: they are cls and herding, whose trained cells are reused.
Descriptives (no gate): per-class retention, class-balanced purity in uni/dinov2/clip, UNI typicality
(cosine to class mean; TypiClust K=20 typicality as within-class percentile), Jaccard to reference sets.
Train split only; val/test are never loaded.
"""
import argparse
import heapq
import json
import os
import sys
import time

import numpy as np

S_ROOT = "/mnt/prj01/hgrp-1502-5TB/tdgs_shared"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("GRAPH_SELECT_ROOT", "/project/prj-sis01/xuxiaoyu/graph_select"))
import tdgs_select  # noqa: E402

if f"{S_ROOT}/r3/emb" not in tdgs_select.EMB_ROOTS:          # as w1_mv_select.py
    tdgs_select.EMB_ROOTS.insert(0, f"{S_ROOT}/r3/emb")
from tdgs_select import LAMBDA, budget_per_class, load_labels  # noqa: E402
from mv_select import View, greedy_multi, load_view  # noqa: E402

K_NN = 50
QS = [0.0, 0.25, 0.5]
TAUS = [0.0, 0.5, 1.0]


def arm_name(q, t):
    return f"acs_q{int(round(q * 100)):02d}_t{int(round(t * 100)):03d}"


# ------------------------------------------------------------------ agreement / filter
def knn_agreement(Zn, y, C, use_gpu=False):
    from graphcov.run.graph import build_knn_graph
    N = len(y)
    nbr, _ = build_knn_graph(Zn, K_NN, use_gpu=use_gpu)
    valid = nbr != np.arange(N)[:, None]
    deg = valid.sum(1)
    rows = np.repeat(np.arange(N), nbr.shape[1])[valid.ravel()]
    cnt = np.zeros((N, C), np.int32)
    np.add.at(cnt, (rows, y[nbr.ravel()[valid.ravel()]]), 1)
    own = cnt[np.arange(N), y]
    return own / np.maximum(deg, 1), own == cnt.max(1)


def quantile_keep(agree, y, C, q):
    keep = np.ones(len(y), bool)
    if q <= 0:
        return keep
    for c in range(C):
        idx = np.flatnonzero(y == c)
        t = np.quantile(agree[idx], q, method="lower")
        keep[idx] = agree[idx] >= t
    return keep


# ------------------------------------------------------------------ selectors
def herd(Zraw, y, keep, k, select_herding):
    if k == 0:
        return np.zeros(0, np.int64)
    kidx = np.flatnonzero(keep)
    sub = np.asarray(select_herding(Zraw[kidx], y[kidx], k, None, seed=42), np.int64)
    return kidx[sub]


def greedy_init(terms, y, bpc, init=(), eligible0=None):
    """mv_select.greedy_multi (objective=None) plus a warm start and a candidate mask.
    With init=() and eligible0=None the arithmetic is identical; I1 asserts it."""
    n = len(y)
    classes = np.unique(y)
    total = bpc * len(classes)
    T = []
    for M, w, coef in terms:
        M = M.tocsc()
        T.append((M.data.astype(np.float32), M.indices, M.indptr,
                  w.astype(np.float32), np.float32(coef)))
    covs = [np.zeros(n, np.float32) for _ in T]
    eligible = np.ones(n, bool) if eligible0 is None else eligible0.copy()
    counts = {int(c): 0 for c in classes}
    selected, gains = [], []

    def take(j):
        c = int(y[j])
        counts[c] += 1
        eligible[j] = False
        if counts[c] >= bpc:
            eligible[y == c] = False
        for (d, r_, p, w, cf), cov in zip(T, covs):
            s, e = p[j], p[j + 1]
            r = r_[s:e]
            cov[r] = np.maximum(cov[r], d[s:e])

    for j in init:
        selected.append(int(j))
        gains.append(np.nan)
        take(int(j))

    def gain_of(j):
        tot = 0.0
        for (d, r_, p, w, c), cov in zip(T, covs):
            if c > 0:
                s, e = p[j], p[j + 1]
                r = r_[s:e]
                tot += c * float(np.dot(w[r], np.maximum(d[s:e] - cov[r], 0.0)))
        return tot

    g0 = np.zeros(n, np.float64)
    for (d, r_, p, w, c), cov in zip(T, covs):
        if c > 0:
            cols = np.repeat(np.arange(n, dtype=np.intp), np.diff(p))
            wt = (np.maximum(d, 0.0) if len(init) == 0 else np.maximum(d - cov[r_], 0.0)) * w[r_]
            g0 += c * np.bincount(cols, weights=wt, minlength=n)

    heap = [(-g0[j], j) for j in range(n)]
    heapq.heapify(heap)
    last_eval = np.zeros(n, np.int64)
    for it in range(total - len(init)):
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
        take(best_j)
    return np.asarray(selected, int), np.asarray(gains, float)


# ------------------------------------------------------------------ descriptives
def view_agreement(ds, v, y, C):
    Z = load_view(ds, v)                      # L2-normalised
    return knn_agreement(Z, y, C)[0]


def typicality_pct(Zn, y, C, k=20):
    """TypiClust typicality 1/mean-L2 to the k nearest same-class points, as within-class percentile."""
    import faiss
    out = np.zeros(len(y))
    for c in range(C):
        idx = np.flatnonzero(y == c)
        X = np.ascontiguousarray(Zn[idx], dtype=np.float32)
        kk = min(k + 1, len(idx))
        ix = faiss.IndexFlatL2(X.shape[1])
        ix.add(X)
        D, _ = ix.search(X, kk)
        typ = 1.0 / (np.sqrt(np.maximum(D[:, 1:], 0)).mean(1) + 1e-12)
        out[idx] = (np.argsort(np.argsort(typ, kind="stable"), kind="stable") + 0.5) / len(idx)
    return out


def cos_to_mean(Zn, y, C):
    out = np.zeros(len(y))
    for c in range(C):
        idx = np.flatnonzero(y == c)
        mu = Zn[idx].mean(0)
        out[idx] = Zn[idx] @ (mu / (np.linalg.norm(mu) + 1e-12))
    return out


def cb_mean(vals, S, y, C):
    return float(np.mean([vals[S[y[S] == c]].mean() for c in range(C)]))


def find_refs(ds, r):
    pat = {
        "random": [f"{S_ROOT}/runs/{t}/sel/seed_42/random/{ds}/ratio_{r}/selection_seed_42/selected_indices.npy"
                   for t in ("ambig_20261002", "table1_a100_20260929", "table1_s4246_20260928")]
                  + [f"{S_ROOT}/w1/sel/{ds}_r{r}_rand_cls_s42.npy"],
        "facility": [f"{S_ROOT}/runs/{t}/sel/seed_42/facility/{ds}/ratio_{r}/selection_seed_42/selected_indices.npy"
                     for t in ("ambig_20261002", "knnf_20261002", "table1_a100_20260929", "table1_s4246_20260928")],
        "graph_a2": [f"{S_ROOT}/runs/ambig_20261002/sel/seed_42/graph_a2/{ds}/ratio_{r}/selection_seed_42/selected_indices.npy",
                     f"{S_ROOT}/w1/sel/{ds}_r{r}_a2_uni_s42.npy", f"{S_ROOT}/sel/{ds}_r{r}_graph_a2_s42.npy"],
        "mv_mean": [f"{S_ROOT}/ambig/sel/{ds}_r{r}_mv_mean_s42.npy", f"{S_ROOT}/w1/sel/{ds}_r{r}_mv_mean_s42.npy",
                    f"{S_ROOT}/mvf/sel/{ds}_r{r}_mv_mean_s42.npy"],
        "knnf_herding": [f"{S_ROOT}/knnf/td_sel/{ds}_r{r}_knnf_herding_s42.npy"],
        "knnf_random42": [f"{S_ROOT}/knnf/td_sel/{ds}_r{r}_knnf_random42_s42.npy"],
    }
    out = {}
    for name, ps in pat.items():
        for p in ps:
            if os.path.exists(p):
                out[name] = (p, np.load(p).astype(np.int64))
                break
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--author-emb", required=True, help="author UNI cache npz (knnf/herding source)")
    ap.add_argument("--herding-ref", required=True, help="dir with ratio_<r>/selection_seed_42")
    ap.add_argument("--cls-ref", required=True, help="staged cls selection that was trained")
    ap.add_argument("--knnf-report", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--gpu-faiss-kernel", action="store_true",
                    help="build the coverage kernel with GPU FAISS (only if the staged cls was)")
    a = ap.parse_args()
    from graphcov.run.selection import _select_herding
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.report, exist_ok=True)
    ds, r = a.dataset, a.ratio
    t0 = time.time()

    y = load_labels(ds)
    C = int(y.max()) + 1
    N = len(y)
    bpc = budget_per_class(N, r, C)
    print(f"[{ds} r={r}] N={N} C={C} bpc={bpc}", flush=True)
    rep = dict(dataset=ds, ratio=r, N=N, C=C, bpc=bpc, lam=LAMBDA, grid=dict(q=QS, tau=TAUS),
               inputs=vars(a), checks={}, retention={}, arms={}, jaccard={})

    # ---- agreement on the author UNI cache, exactly as knnf_select.py
    Zraw = np.asarray(np.load(a.author_emb)["embeddings"], dtype=np.float32)
    assert len(Zraw) == N, (Zraw.shape, N)
    Zn = Zraw / np.maximum(np.linalg.norm(Zraw, axis=1, keepdims=True), 1e-12)
    agree, plural = knn_agreement(Zn, y, C)
    kr = json.load(open(a.knnf_report))
    kept_pc = [int(plural[y == c].sum()) for c in range(C)]
    rep["checks"]["I3_plurality_kept_per_class"] = dict(ours=kept_pc, knnf=kr["kept_per_class"],
                                                        match=kept_pc == kr["kept_per_class"])
    print(f"  I3 plurality kept {kept_pc} vs knnf {kr['kept_per_class']}", flush=True)
    if kept_pc != kr["kept_per_class"]:
        raise SystemExit("ABORT I3: agreement differs from knnf_select")

    # ---- I2: herding identity
    hfull = herd(Zraw, y, np.ones(N, bool), bpc, _select_herding)
    href = np.load(os.path.join(a.herding_ref, f"ratio_{r}", "selection_seed_42", "selected_indices.npy")).astype(np.int64)
    rep["checks"]["I2_herding_order_match"] = bool(np.array_equal(hfull, href))
    print(f"  I2 herding == bench herding (order): {rep['checks']['I2_herding_order_match']}", flush=True)
    if not np.array_equal(hfull, href):
        raise SystemExit("ABORT I2: herding on the full pool != staged bench herding")

    # ---- coverage view (UNI, same loader as the staged cls) and I1
    if a.gpu_faiss_kernel:
        os.environ["GRAPHCOV_USE_FAISS_GPU"] = "1"
    Zu = load_view(ds, "uni")
    rep["checks"]["uni_view_vs_author_cache_maxabs"] = float(np.abs(Zu - Zn).max())
    cls_n = np.bincount(y, minlength=C).astype(np.float32)
    w_cls = (1.0 / (C * cls_n))[y].astype(np.float32)
    U = View("uni", Zu, y, w_cls)
    terms = U.terms(LAMBDA)
    s_old, _ = greedy_multi(terms, y, bpc, verbose=False)
    s_new, _ = greedy_init(terms, y, bpc)
    cref = np.load(a.cls_ref).astype(np.int64)
    rep["checks"]["I1_order_match_greedy_multi"] = bool(np.array_equal(s_old, s_new))
    rep["checks"]["I1_set_match_staged_cls"] = set(s_new.tolist()) == set(cref.tolist())
    print(f"  I1 greedy_init==greedy_multi {rep['checks']['I1_order_match_greedy_multi']}  "
          f"== staged cls {rep['checks']['I1_set_match_staged_cls']}", flush=True)
    if not (rep["checks"]["I1_order_match_greedy_multi"] and rep["checks"]["I1_set_match_staged_cls"]):
        raise SystemExit("ABORT I1: (q,tau)=(0,0) does not reproduce the staged cls")

    # ---- grid
    sets = {"cls": s_new, "herding": hfull}
    for q in QS:
        keep = quantile_keep(agree, y, C, q)
        kc = [int(keep[y == c].sum()) for c in range(C)]
        nc = [int((y == c).sum()) for c in range(C)]
        rep["retention"][f"{q}"] = dict(kept_per_class=kc, frac=[k / n for k, n in zip(kc, nc)],
                                        erased=int(sum(k == 0 for k in kc)))
        assert min(kc) >= bpc, (q, kc, bpc)
        for t in TAUS:
            if (q, t) in ((0.0, 0.0), (0.0, 1.0)):
                continue
            ts = time.time()
            kh = int(np.floor(t * bpc + 0.5))
            hp = herd(Zraw, y, keep, kh, _select_herding)
            if kh < bpc:
                sel, _ = greedy_init(terms, y, bpc, init=hp, eligible0=keep)
            else:
                sel = hp
            cnt = np.bincount(y[sel], minlength=C)
            assert len(set(sel.tolist())) == sel.size == bpc * C and cnt.min() == cnt.max() == bpc, cnt
            assert keep[sel].all()
            name = arm_name(q, t)
            sets[name] = np.asarray(sel, np.int64)
            rep["arms"][name] = dict(q=q, tau=t, k_herd=kh)
            print(f"  {name}: k_herd={kh} ({time.time()-ts:.0f}s)", flush=True)
    rep["checks"]["I4_all_sets_valid"] = True

    for name, s in sets.items():
        if name.startswith("acs_"):
            np.save(os.path.join(a.out_dir, f"{ds}_r{r}_{name}_s42.npy"), s)

    # ---- descriptives
    refs = find_refs(ds, r)
    rep["refs"] = {k: v[0] for k, v in refs.items()}
    allsets = dict(sets, **{k: v[1] for k, v in refs.items()})
    ag = {"uni": agree}
    for v in ("dinov2", "clip"):
        try:
            ag[v] = view_agreement(ds, v, y, C)
        except Exception as e:  # missing view embedding is reported, not fatal
            rep.setdefault("missing_views", {})[v] = str(e)
    typ = typicality_pct(Zn, y, C)
    cmu = cos_to_mean(Zn, y, C)
    for name, s in allsets.items():
        d = rep["arms"].setdefault(name, {})
        d["purity"] = {v: cb_mean(x, s, y, C) for v, x in ag.items()}
        d["typ_pct"] = cb_mean(typ, s, y, C)
        d["cos_mean"] = cb_mean(cmu, s, y, C)
        d["min_class_agree_uni"] = float(min(agree[s[y[s] == c]].mean() for c in range(C)))
        print(f"  {name:16s} pur " + " ".join(f"{v}={x:.3f}" for v, x in d["purity"].items())
              + f" typ%={d['typ_pct']:.3f} cos={d['cos_mean']:.3f}", flush=True)
    names = list(allsets)
    for i, p in enumerate(names):
        for q_ in names[i + 1:]:
            A, B = set(allsets[p].tolist()), set(allsets[q_].tolist())
            rep["jaccard"][f"{p}|{q_}"] = len(A & B) / len(A | B)
    rep["seconds"] = time.time() - t0
    out = os.path.join(a.report, f"{ds}_r{r}_acs.json")
    json.dump(rep, open(out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"  wrote {out} ({rep['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
