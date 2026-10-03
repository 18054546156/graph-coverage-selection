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

E5 (report/e5_prereg_20261003.md) adds two modes; the default grid path above is unchanged (I5 checks it):
  --variants SPEC ...  SPEC = q<QQ>_t<TTT>[_l00|_l100|_mask|_wcls] -> acs_<SPEC>. The F1 term is changed only:
      l00  lam=0 (F1 off: global coverage K only, = Graph-A2 objective under the per-class quota)
      l100 lam=1 (class-masked term only)
      mask lam=.5, F1's second term class-masked but uniform demand weights   (= round-2 tdgs_mask)
      wcls lam=.5, F1's second term unmasked K, class-flat weights 1/(C n_c)   (= round-2 tdgs_wcls)
      I5  for every (q,tau) used, the lam=.5 set recomputed here == the staged grid set (bit-identical, order),
          and for tau=1 (k_herd == bpc, no coverage stage) every variant == the lam=.5 set.  Abort otherwise.
      Mediator checks (zero GPU, written to <ds>_r<r>_acs_e5.json): M1 cross-class share of each candidate's
      lam=0 gain vs 1-a (Spearman); M2 per-class demand coverage won by other-class picks; M3 purity of every
      set; M4 F2 keep-set stability over k in {20, 50, 100}.
  --new-ratio  a ratio with no staged cls / bench herding (E5c): I1 keeps the greedy_init==greedy_multi order
      check, I2 and the staged-cls set check are reported as n/a instead of aborting.
"""
import argparse
import heapq
import json
import os
import re
import sys
import time

import numpy as np

S_ROOT = "/mnt/prj01/hgrp-1502-5TB/tdgs_shared"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("GRAPH_SELECT_ROOT", "/project/prj-sis01/xuxiaoyu/graph_select"))
import tdgs_select  # noqa: E402

if f"{S_ROOT}/r3/emb" not in tdgs_select.EMB_ROOTS:          # as w1_mv_select.py
    tdgs_select.EMB_ROOTS.insert(0, f"{S_ROOT}/r3/emb")
from tdgs_select import LAMBDA, budget_per_class, load_labels, pool_value  # noqa: E402
from mv_select import View, greedy_multi, load_view  # noqa: E402
from tdgs_select_r2 import build_kernel_variant  # noqa: E402

K_NN = 50
QS = [0.0, 0.25, 0.5]
TAUS = [0.0, 0.5, 1.0]
SPEC_RE = re.compile(r"q(\d\d)_t(\d{3})(?:_(l00|l100|mask|wcls))?")


def arm_name(q, t):
    return f"acs_q{int(round(q * 100)):02d}_t{int(round(t * 100)):03d}"


def parse_spec(spec):
    m = SPEC_RE.fullmatch(spec)
    if not m:
        raise SystemExit(f"bad variant spec {spec!r}")
    return int(m[1]) / 100, int(m[2]) / 100, m[3]


def variant_terms(U, y, kind):
    """Coverage terms of one F1 variant; kind=None is the unchanged ACS F1 (U.terms(LAMBDA))."""
    if kind is None:
        return U.terms(LAMBDA)
    if kind == "l00":
        return U.terms(0.0)
    if kind == "l100":
        return U.terms(1.0)
    if kind == "mask":
        return [(U.K, U.w1 / U.FG, 1 - LAMBDA), (U.a, U.w1 / pool_value(U.a, U.w1), LAMBDA)]
    if kind == "wcls":
        g, _ = build_kernel_variant(U.K, y, False)
        return [(U.K, U.w1 / U.FG, 1 - LAMBDA), (g, U.w2 / pool_value(g, U.w2), LAMBDA)]
    raise ValueError(kind)


# ------------------------------------------------------------------ agreement / filter
def knn_agreement(Zn, y, C, use_gpu=False, k=K_NN):
    from graphcov.run.graph import build_knn_graph
    N = len(y)
    nbr, _ = build_knn_graph(Zn, k, use_gpu=use_gpu)
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


def cross_share(K, y):
    """M1: per candidate j, the share of its lam=0 (uniform-weight K) gain on an empty set that sits on
    other-class demand points: sum_{i: y_i != y_j} K_ij / sum_i K_ij."""
    K = K.tocsc()
    cols = np.repeat(np.arange(K.shape[1]), np.diff(K.indptr))
    x = y[K.indices] != y[cols]
    tot = np.bincount(cols, weights=K.data, minlength=K.shape[1])
    cr = np.bincount(cols, weights=K.data * x, minlength=K.shape[1])
    return cr / np.maximum(tot, 1e-30)


def leakage(K, S, y, C):
    """M2: per class c, the share of class-c demand coverage (max_{j in S} K_ij, summed over i in c)
    whose covering pick has another label. Returns (class-mean, per-class list)."""
    sub = K.tocsc()[:, S].tocsr()
    best = np.asarray(sub.max(axis=1).todense()).ravel()
    arg = np.asarray(sub.argmax(axis=1)).ravel()
    other = y[np.asarray(S)[arg]] != y
    per = [float((best[y == c] * other[y == c]).sum() / max(best[y == c].sum(), 1e-30)) for c in range(C)]
    return float(np.mean(per)), per


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
        # E5 additions (descriptive only): the author-pipeline Graph-A2 of the bench fill (derma/OCT) and our
        # class-quota re-implementation, which acs_q00_t000_l00 should reproduce.
        "graph_a2_author": [f"{S_ROOT}/benchfill/td_sel/{ds}_r{r}_graph_a2_s42.npy",
                            f"{S_ROOT}/runs/ambig_20261002/sel/seed_42/graph_a2/{ds}/ratio_{r}/selection_seed_42/selected_indices.npy",
                            f"{S_ROOT}/runs/table1_s4246_20260928/sel/seed_42/graph_a2/{ds}/ratio_{r}/selection_seed_42/selected_indices.npy"],
        "a2_quota_ours": [f"{S_ROOT}/w1/sel/{ds}_r{r}_a2_uni_s42.npy", f"{S_ROOT}/sel/{ds}_r{r}_graph_a2_s42.npy"],
    }
    out = {}
    for name, ps in pat.items():
        for p in ps:
            if os.path.exists(p):
                out[name] = (p, np.load(p).astype(np.int64))
                break
    return out


# ------------------------------------------------------------------ E5 variants
def save_new(path, s):
    """Write a selection; never overwrite an existing file with different content."""
    if os.path.exists(path):
        if not np.array_equal(np.load(path), s):
            raise SystemExit(f"ABORT: {path} exists with different content")
        return
    tmp = os.path.join(os.path.dirname(path), f".tmp_{os.getpid()}_{os.path.basename(path)}")
    np.save(tmp, s)
    os.rename(tmp, path)


def run_variants(a, ds, r, y, C, N, bpc, Zraw, Zn, agree, U, s_cls, s_herd, rep, t0, select_herding):
    from scipy.stats import spearmanr
    specs = [parse_spec(s) for s in a.variants]
    rep["mode"] = "e5_variants"
    rep["variants"] = a.variants

    def build(q, t, kind):
        keep = quantile_keep(agree, y, C, q)
        kh = int(np.floor(t * bpc + 0.5))
        hp = herd(Zraw, y, keep, kh, select_herding)
        if kh < bpc:
            sel, _ = greedy_init(variant_terms(U, y, kind), y, bpc, init=hp, eligible0=keep)
        else:
            sel = hp
        sel = np.asarray(sel, np.int64)
        cnt = np.bincount(y[sel], minlength=C)
        assert len(set(sel.tolist())) == sel.size == bpc * C and cnt.min() == cnt.max() == bpc, cnt
        assert keep[sel].all()
        return sel, kh

    # ---- I5: the lam=.5 path is untouched (== staged grid set, order) for every (q, tau) used
    base, i5 = {}, {}
    for q, t in sorted({(q, t) for q, t, _ in specs}):
        s, kh = build(q, t, None)
        if (q, t) == (0.0, 0.0):
            ref, src = s_cls, "cls (I1)"
        elif (q, t) == (0.0, 1.0):
            ref, src = s_herd, "herding (I2)"
        else:
            p = os.path.join(a.grid_dir or a.out_dir, f"{ds}_r{r}_{arm_name(q, t)}_s42.npy")
            if not os.path.exists(p):
                raise SystemExit(f"ABORT I5: staged grid set missing {p}")
            ref, src = np.load(p).astype(np.int64), p
        i5[arm_name(q, t)] = dict(match=bool(np.array_equal(s, ref)), ref=src, k_herd=kh)
        print(f"  I5 {arm_name(q, t)} lam=.5 recomputed == staged: {i5[arm_name(q, t)]['match']}", flush=True)
        if not np.array_equal(s, ref):
            raise SystemExit(f"ABORT I5: {arm_name(q, t)} recomputed != staged")
        base[(q, t)] = (s, kh)

    sets = {}
    for (q, t, kind), spec in zip(specs, a.variants):
        ts = time.time()
        if kind is None:
            sets[f"acs_{spec}"] = base[(q, t)][0]
            continue
        s, kh = build(q, t, kind)
        if kh == bpc:                                   # tau=1: no coverage stage, F1 cannot act
            ok = bool(np.array_equal(s, base[(q, t)][0]))
            i5[f"acs_{spec}_eq_lam05"] = dict(match=ok)
            if not ok:
                raise SystemExit(f"ABORT I5: tau=1 variant {spec} differs from the lam=.5 set")
        sets[f"acs_{spec}"] = s
        rep["arms"][f"acs_{spec}"] = dict(q=q, tau=t, k_herd=kh, kind=kind,
                                          jaccard_to_lam05=len(set(s.tolist()) & set(base[(q, t)][0].tolist()))
                                          / len(set(s.tolist()) | set(base[(q, t)][0].tolist())))
        print(f"  acs_{spec}: k_herd={kh} J(lam.5)={rep['arms'][f'acs_{spec}']['jaccard_to_lam05']:.3f} "
              f"({time.time()-ts:.0f}s)", flush=True)
    rep["checks"]["I5"] = i5
    for name, s in sets.items():
        if any(name.endswith(f"_{k}") for k in ("l00", "l100", "mask", "wcls")):
            save_new(os.path.join(a.out_dir, f"{ds}_r{r}_{name}_s42.npy"), s)

    # ---- mediator checks (descriptive; they do not change the design)
    med = {}
    chi = cross_share(U.K, y)
    rho = spearmanr(chi, 1 - agree).correlation
    med["M1"] = dict(spearman_chi_vs_1ma=float(rho), chi_mean=float(chi.mean()), pass_ge_0p7=bool(rho >= .7))
    print(f"  M1 Spearman(chi, 1-a) = {rho:.3f}", flush=True)
    refs = find_refs(ds, r)
    allsets = dict(sets, cls=s_cls, herding=s_herd, **{f"ref_{k}": v[1] for k, v in refs.items()})
    for (q, t) in base:
        allsets.setdefault(arm_name(q, t) if (q, t) not in ((0.0, 0.0), (0.0, 1.0)) else
                           ("cls" if t == 0 else "herding"), base[(q, t)][0])
    med["M2_leakage"], med["M3_purity_uni"] = {}, {}
    for name, s in allsets.items():
        if len(s) != bpc * C:              # refs of another budget/size are skipped for M2
            continue
        lm, lp = leakage(U.K, s, y, C)
        med["M2_leakage"][name] = dict(class_mean=lm, per_class=lp)
        med["M3_purity_uni"][name] = cb_mean(agree, s, y, C)
        print(f"  {name:22s} M2 leak {lm:.3f}  M3 purity {med['M3_purity_uni'][name]:.3f}", flush=True)
    keeps = {}
    for k in (20, 50, 100):
        ag_k = agree if k == K_NN else knn_agreement(Zn, y, C, k=k)[0]
        keeps[k] = quantile_keep(ag_k, y, C, 0.25)
    J = lambda u, v: float((u & v).sum() / max((u | v).sum(), 1))
    med["M4"] = {f"{k1}_{k2}": dict(kept=J(keeps[k1], keeps[k2]), removed=J(~keeps[k1], ~keeps[k2]))
                 for k1, k2 in ((20, 50), (50, 100), (20, 100))}
    med["M4"]["pass_kept_ge_0p8"] = bool(min(v["kept"] for k, v in med["M4"].items() if "_" in k) >= .8)
    print(f"  M4 {med['M4']}", flush=True)
    if "ref_a2_quota_ours" in allsets and "acs_q00_t000_l00" in sets:
        A_, B_ = set(sets["acs_q00_t000_l00"].tolist()), set(allsets["ref_a2_quota_ours"].tolist())
        med["l00_vs_our_a2_quota_jaccard"] = len(A_ & B_) / len(A_ | B_)
    rep["mediators"] = med
    rep["refs"] = {k: v[0] for k, v in refs.items()}
    rep["seconds"] = time.time() - t0
    out = os.path.join(a.report, f"{ds}_r{r}_acs_e5.json")
    json.dump(rep, open(out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"  wrote {out} ({rep['seconds']:.0f}s)", flush=True)


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
    ap.add_argument("--variants", nargs="+", default=None,
                    help="E5: stage only these F1 variants (see docstring) instead of the grid; runs I5 + M1-M4")
    ap.add_argument("--grid-dir", default=None, help="E5: where the staged grid sets live (default --out-dir)")
    ap.add_argument("--new-ratio", action="store_true",
                    help="E5c: no staged cls / bench herding exist at this ratio (I2 and the I1 set check -> n/a)")
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
    hpath = os.path.join(a.herding_ref, f"ratio_{r}", "selection_seed_42", "selected_indices.npy")
    if a.new_ratio and not os.path.exists(hpath):
        rep["checks"]["I2_herding_order_match"] = "n/a (new ratio, no bench herding)"
        print("  I2 n/a (new ratio)", flush=True)
    else:
        href = np.load(hpath).astype(np.int64)
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
    rep["checks"]["I1_order_match_greedy_multi"] = bool(np.array_equal(s_old, s_new))
    if a.new_ratio and not os.path.exists(a.cls_ref):
        rep["checks"]["I1_set_match_staged_cls"] = "n/a (new ratio, no staged cls)"
        set_ok = True
    else:
        cref = np.load(a.cls_ref).astype(np.int64)
        rep["checks"]["I1_set_match_staged_cls"] = set_ok = set(s_new.tolist()) == set(cref.tolist())
    print(f"  I1 greedy_init==greedy_multi {rep['checks']['I1_order_match_greedy_multi']}  "
          f"== staged cls {rep['checks']['I1_set_match_staged_cls']}", flush=True)
    if not (rep["checks"]["I1_order_match_greedy_multi"] and set_ok):
        raise SystemExit("ABORT I1: (q,tau)=(0,0) does not reproduce the staged cls")

    if a.variants:
        return run_variants(a, ds, r, y, C, N, bpc, Zraw, Zn, agree, U, s_new, hfull, rep, t0, _select_herding)

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
