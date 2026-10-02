#!/usr/bin/env python
"""E1 of report/proposal_acs_20261002.md: low-budget active-learning baselines, author code + author defaults.
Selections only (nothing is trained). One method per process (the two official repos both ship a `pycls`).

  typiclust  official avihu111/TypiClust  pycls.al.typiclust.TypiClust   (K_NN=20, MIN_CLUSTER_SIZE=5,
             MAX_NUM_CLUSTERS=500, clusters = budget; KMeans/MiniBatchKMeans, np.random.seed(42))
  probcover  official ProbCover greedy (unit-checked against pycls.al.prob_cover.ProbCover on a toy pool);
             delta by the paper's rule: alpha=0.95 purity of delta-balls under k-means pseudo-labels (k = C).
  maxherding official BorealisAI/uherding  pycls.al.herding.Herding (= MaxHerding, ECCV 2024) with the
             repo defaults of run_budget.sh: kernel=rbf, delta=1.0, candidate cap compute_cand_size (35000).

Versions (D3 of the proposal):
  main     : equal per-class quota bpc, the method run INSIDE each class (as every Table-1 arm).
  _nolab   : original label-free version on the whole pool, total budget bpc*C (appendix only).
Features: UNI author cache (same as the bench), L2-normalised (the official loaders normalise too).
Train split only. Fixed selection seed 42 (as all our staged selectors).
"""
import argparse
import heapq
import json
import os
import sys
import time

import numpy as np
import torch

S = "/mnt/prj01/hgrp-1502-5TB/tdgs_shared"
TP = f"{S}/transfer/third_party"
sys.path.insert(0, f"{S}/code")
from tdgs_select import budget_per_class, load_labels  # noqa: E402

ALPHA = 0.95
EDGE_CAP = 3e9
DEV = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False


class Cfg(dict):
    """dict that also answers attribute access (pycls CfgNode stand-in)."""
    def __getattr__(self, k):
        v = self[k]
        return Cfg(v) if isinstance(v, dict) else v


def l2n(Z):
    return (Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)).astype(np.float32)


# ---------------------------------------------------------------- ProbCover (official greedy semantics)
def ball_csr(Zt, delta, bs=1024):
    thr = 1 - delta ** 2 / 2          # ||x-y|| < delta  <=>  cos > 1 - delta^2/2  (unit vectors)
    ind, cnt = [], []
    for s in range(0, len(Zt), bs):
        m = (Zt[s:s + bs] @ Zt.T) > thr
        r, c = m.nonzero(as_tuple=True)
        cnt.append(torch.bincount(r, minlength=m.shape[0]).cpu().numpy())
        ind.append(c.to(torch.int32).cpu().numpy())
    cnt = np.concatenate(cnt)
    indptr = np.zeros(len(cnt) + 1, np.int64)
    np.cumsum(cnt, out=indptr[1:])
    return indptr, np.concatenate(ind)


def pc_greedy(indptr, ind, b):
    n = len(indptr) - 1
    cov = np.zeros(n, bool)
    heap = [(-(indptr[j + 1] - indptr[j]), j) for j in range(n)]
    heapq.heapify(heap)
    fresh = np.full(n, -1)
    sel = []
    for it in range(min(b, n)):
        while True:
            g, j = heapq.heappop(heap)
            if fresh[j] == it:
                break
            gj = int((~cov[ind[indptr[j]:indptr[j + 1]]]).sum())
            fresh[j] = it
            heapq.heappush(heap, (-gj, j))
        sel.append(j)
        cov[ind[indptr[j]:indptr[j + 1]]] = True
    return np.asarray(sel, np.int64)


def pc_unit_check():
    sys.path.insert(0, f"{TP}/TypiClust/deep-al")
    import pycls.datasets.utils as ds_utils
    from pycls.al.prob_cover import ProbCover
    X = l2n(np.random.default_rng(0).normal(size=(2000, 16)))
    ds_utils.load_features = lambda name, seed: X
    off, _ = ProbCover({"DATASET": {"NAME": "toy"}, "RNG_SEED": 0}, np.array([], int), np.arange(2000), 40, 0.9).select_samples()
    ip, ind = ball_csr(torch.as_tensor(X, device=DEV), 0.9)
    ours = pc_greedy(ip, ind, 40)
    return bool(np.array_equal(np.asarray(off), ours))


def pc_delta(Z, C):
    """paper rule: largest delta with >= alpha of balls pure under k-means pseudo-labels (k = C)."""
    from sklearn.cluster import KMeans
    pl = KMeans(n_clusters=C, n_init=10, random_state=42).fit_predict(Z)
    Zt, yt = torch.as_tensor(Z, device=DEV), torch.as_tensor(pl, device=DEV)
    mx = torch.empty(len(Z), device=DEV)
    for s in range(0, len(Z), 1024):
        sim = Zt[s:s + 1024] @ Zt.T
        mx[s:s + 1024] = sim.masked_fill(yt[s:s + 1024, None] == yt[None, :], -3.0).max(1).values
    d_other = torch.sqrt(torch.clamp(2 - 2 * mx, min=0)).cpu().numpy()
    return float(np.quantile(d_other, 1 - ALPHA))


def run_probcover(Z, y, C, bpc, rep):
    rep["unit_check_vs_official"] = pc_unit_check()
    if not rep["unit_check_vs_official"]:
        raise SystemExit("ABORT: ProbCover greedy != official")
    delta = pc_delta(Z, C)
    rep["delta"] = delta
    Zt = torch.as_tensor(Z, device=DEV)
    sel, ball_med, edges = [], [], 0
    for c in range(C):
        idx = np.flatnonzero(y == c)
        ip, ind = ball_csr(Zt[idx], delta)
        edges += len(ind)
        if edges > EDGE_CAP:
            raise SystemExit(f"ProbCover infeasible: edges > {EDGE_CAP:g}")
        ball_med.append(float(np.median(np.diff(ip))))
        sel.append(idx[pc_greedy(ip, ind, bpc)])
    rep["ball_median_within_class"] = ball_med
    out = {"probcover": np.concatenate(sel)}
    thr = 1 - delta ** 2 / 2
    n_edges = sum(int(((Zt[s:s + 1024] @ Zt.T) > thr).sum()) for s in range(0, len(Zt), 1024))
    rep["edges_nolab"] = n_edges
    if n_edges <= EDGE_CAP:
        ip, ind = ball_csr(Zt, delta)
        out["probcover_nolab"] = pc_greedy(ip, ind, bpc * C)
    else:
        rep["nolab_skipped"] = f"edges > {EDGE_CAP:g}"
    return out


# ---------------------------------------------------------------- TypiClust (official class)
def run_typiclust(Z, y, C, bpc, rep):
    sys.path.insert(0, f"{TP}/TypiClust/deep-al")
    import pycls.datasets.utils as ds_utils
    from pycls.al.typiclust import TypiClust
    cfg = {"DATASET": {"NAME": "medmnist"}, "RNG_SEED": 42}

    from pycls.al.typiclust import calculate_typicality

    def fallback_select(tc):
        """Official select_samples with one change: when the budget is close to the class size, the official
        round-robin revisits clusters that are too small (<= MIN_CLUSTER_SIZE, dropped) or already exhausted and
        crashes. Here the round-robin skips exhausted clusters, and if no cluster exceeds MIN_CLUSTER_SIZE all
        clusters are used. Same clustering (same seed), same typicality (K_NN=20, half-cluster rule)."""
        labels = tc.clusters.copy()
        ids, sizes = np.unique(labels, return_counts=True)
        keep = sizes > tc.MIN_CLUSTER_SIZE
        if not keep.any():
            keep[:] = True
        order = [c for _, c in sorted(zip(-sizes[keep], ids[keep]))]   # existing_count = 0 for all
        sel, i = [], 0
        while len(sel) < tc.budgetSize:
            live = [c for c in order if (labels == c).any()]
            if not live:                        # every kept cluster exhausted: open the dropped ones
                order = [c for c in ids if (labels == c).any()]
                continue
            c = live[i % len(live)]
            idx = (labels == c).nonzero()[0]
            k = min(tc.K_NN, len(idx) // 2)
            j = idx[0] if k == 0 else idx[calculate_typicality(tc.features[idx], k).argmax()]
            sel.append(j)
            labels[j] = -1
            i += 1
        return np.asarray(sel)

    def one(X, b, tag):
        ds_utils.load_features = lambda name, seed: X
        np.random.seed(42)
        tc = TypiClust(cfg, np.array([], int), np.arange(len(X)), b)
        try:
            act, _ = tc.select_samples()
        except (ZeroDivisionError, ValueError):
            rep.setdefault("fallback_skip_exhausted_clusters", []).append(tag)
            np.random.seed(42)
            act = fallback_select(TypiClust(cfg, np.array([], int), np.arange(len(X)), b))
        return np.asarray(act, np.int64)

    sel = [np.flatnonzero(y == c)[one(Z[y == c], bpc, c)] for c in range(C)]
    return {"typiclust": np.concatenate(sel), "typiclust_nolab": one(Z, bpc * C, "nolab")}


# ---------------------------------------------------------------- MaxHerding (official class)
def _stub_yacs():
    """uherding's pycls.core.config needs yacs only to build its global config tree at import; the env has no
    yacs and must not get new packages. A dict-backed CfgNode is enough (the selector never reads that tree)."""
    import copy
    import types

    class CN(dict):
        def __init__(self, init=None, **k):
            super().__init__(init or {}, **k)

        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)

        def __setattr__(self, k, v):
            self[k] = v

        def clone(self):
            return copy.deepcopy(self)

        def freeze(self):
            pass

        def defrost(self):
            pass
    m, mc = types.ModuleType("yacs"), types.ModuleType("yacs.config")
    mc.CfgNode = CN
    m.config = mc
    sys.modules.setdefault("yacs", m)
    sys.modules.setdefault("yacs.config", mc)


def run_maxherding(Z, y, C, bpc, rep, delta=1.0, kernel="rbf"):
    _stub_yacs()
    sys.path.insert(0, f"{TP}/uherding/deep-al")
    import pycls.datasets.utils as ds_utils
    import pycls.al.herding as H
    rep.update(kernel=kernel, delta=delta)
    cfg = Cfg({"DATASET": {"NAME": "medmnist"}, "RNG_SEED": 42,
               "ACTIVE_LEARNING": {"UNC_FEATURE": "uni", "FEATURE": "uni"}})

    class _DS:  # Herding asserts a dataset object; features come from the patched loader
        pass

    def one(X, b):
        ds_utils.load_features = lambda *a, **k: X
        H.ds_utils.load_features = ds_utils.load_features
        np.random.seed(42)
        torch.manual_seed(42)
        h = H.Herding(cfg, np.array([], int), np.arange(len(X)), b, delta, None, dataset=_DS(),
                      kernel=kernel, device="cuda")
        act, _ = h.select_samples()
        del h
        torch.cuda.empty_cache()
        return np.asarray(act, np.int64)

    sel = [np.flatnonzero(y == c)[one(Z[y == c], bpc)] for c in range(C)]
    out = {"maxherding": np.concatenate(sel)}
    out["maxherding_nolab"] = one(Z, bpc * C)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", required=True, choices=["typiclust", "probcover", "maxherding"])
    ap.add_argument("--emb", required=True, help="author UNI cache npz")
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.02, 0.05])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.report, exist_ok=True)
    ds = a.dataset
    y = load_labels(ds)
    C = int(y.max()) + 1
    Z = l2n(np.asarray(np.load(a.emb)["embeddings"], dtype=np.float32))
    assert len(Z) == len(y)
    R = dict(dataset=ds, method=a.method, emb=a.emb, N=len(y), C=C, ratios={})
    for r in a.ratios:
        t0 = time.time()
        bpc = budget_per_class(len(y), r, C)
        rep = dict(bpc=bpc)
        fn = {"typiclust": run_typiclust, "probcover": run_probcover, "maxherding": run_maxherding}[a.method]
        sets = fn(Z, y, C, bpc, rep)
        for name, s in sets.items():
            s = np.asarray(s, np.int64)
            assert len(set(s.tolist())) == s.size == bpc * C, (name, s.size)
            cnt = np.bincount(y[s], minlength=C)
            if not name.endswith("_nolab"):
                assert cnt.min() == cnt.max() == bpc, (name, cnt)
            rep[name] = dict(per_class=cnt.tolist())
            np.save(os.path.join(a.out_dir, f"{ds}_r{r}_{name}_s42.npy"), s)
        rep["seconds"] = round(time.time() - t0, 1)
        R["ratios"][str(r)] = rep
        print(f"[{ds} r={r}] {a.method} bpc={bpc} " + json.dumps({k: v for k, v in rep.items() if k != 'bpc'})[:400],
              flush=True)
    json.dump(R, open(os.path.join(a.report, f"{ds}_{a.method}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
