#!/usr/bin/env python
"""Does the demand proxy know what the FINAL learner will find hard?

WHY THIS EXISTS
---------------
TDGS estimates per-sample demand from a linear head on frozen UNI trained on the
whole source pool. Staging it revealed a problem that is fatal exactly where it
matters most:

    dataset       OOF BA   frac(d < 0.05)   subset-effect sd
    pathmnist     0.9974       0.981             4.07pp
    bloodmnist    0.9796       0.856             1.42pp
    organamnist   0.9355       0.617             2.29pp
    organsmnist   0.8403       0.540             0.80pp

Spearman(subset-effect sd, frac(d<0.05)) = +0.8. The datasets where subset
choice moves accuracy most are the ones where the proxy says almost nothing is
hard. On pathmnist the probe reaches 0.9974 balanced accuracy while the final
ResNet-18 trained from scratch on 1791 samples reaches about 0.815, so the
probe's notion of "hard" cannot be the final learner's.

The suspected cause is a DATA-REGIME mismatch, not a representation problem: the
probe is fitted on ~72k samples, the final learner on 1791. So this script
compares two proxy regimes under one honest validity metric.

THE TWO REGIMES
    full     train on (K-1)/K of the pool, predict the held-out fold.
             This is what tdgs_select.py currently does.
    budget   train on exactly budget_per_class samples per class -- the SAME
             data regime as the final model -- and predict everything else.
             Averaged over R independent class-balanced draws, so a sample's
             demand is its mean difficulty across many budget-sized models
             rather than one arbitrary draw.

THE VALIDITY METRIC (this is the point, and it is free)
------------------------------------------------------
The single-worst-class check used while staging is too weak: it collapses a
whole confusion structure to one label, and it scored organamnist as a MISMATCH
when the proxy had actually recovered the same pair {4,5} with the members
swapped. Replace it with a per-class comparison against the real thing:

    x_c = mean_{i in class c} d_i                  proxy's class difficulty
    y_c = 1 - recall_c  of the FINAL ResNet-18     measured class difficulty

y_c comes from the archived Table-1 runs' `predictions_clean.npz`, averaged over
the four competitive arms and all training seeds, so it is a property of the
dataset and the final learner rather than of one selection. Report
Spearman(x, y) across classes, per dataset, for both regimes.

This is the quantity the method actually needs. If it is near zero, the demand
field is aimed at the wrong target and no amount of clever coverage geometry on
top of it will help.

PRE-REGISTERED PREDICTIONS (written before the first run; do not edit after)
---------------------------------------------------------------------------
P1  `budget` has materially more dynamic range than `full`: frac(d<0.05) drops
    by at least 0.20 on pathmnist and bloodmnist, the two datasets where `full`
    exceeds 0.85.
P2  Spearman(x, y) is higher under `budget` than under `full` on at least 3 of
    the 5 datasets.
P3  Under `full`, Spearman(x, y) on pathmnist is below 0.4 -- a proxy that
    solves the task at 0.9974 cannot rank the final learner's class difficulty.

DECISION RULE, FIXED IN ADVANCE
    Round 2 (restaging the demand arms on the `budget` proxy) is worth GPU only
    if P2 holds AND `budget` reaches Spearman >= 0.6 on at least two of the
    three live units {pathmnist, organ-family, bloodmnist}.
    If both regimes are near zero, the honest conclusion is that per-sample
    demand is not estimable from frozen UNI at this budget, which would close
    the demand-weighting family rather than motivate a third proxy.

    tissuemnist is excluded from the decision in advance (probe BA caps ~0.48,
    subset-effect sd 1.07pp against a 1-2pp noise floor). It is still measured.

TRAINS NO ResNet. Linear heads only.
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

PROJECT = "/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab"
ROOTS = [f"{PROJECT}/table1_clean_only_20260922/formal",
         f"{PROJECT}/table1_clean_only_5pct_20260922/formal"]
COMPETITIVE = ["graph_a2", "random", "herding", "facility"]
DATASETS = ["bloodmnist", "organsmnist", "organamnist", "pathmnist",
            "tissuemnist"]
LIVE_UNIT = {"pathmnist": "pathmnist", "organamnist": "organ",
             "organsmnist": "organ", "bloodmnist": "bloodmnist"}
EMB_ROOTS = [
    f"{PROJECT}/table1_clean_only_20260922/cache/embeddings_img224_smokefull",
    "/mnt/prj01/hgrp-1502-5TB/xuxiaoyu/reliability_medmnistc_ab/"
    "table1_clean_only_20260922/cache/embeddings_img224_smokefull",
]
EPS = 1e-12


def load_emb(ds):
    for root in EMB_ROOTS:
        p = os.path.join(root, f"{ds}_train_uni_224.npz")
        if os.path.exists(p):
            f = np.load(p)
            k = next(x for x in ("embeddings", "features", "emb", "arr_0")
                     if x in f.files)
            return f[k].astype(np.float32)
    raise FileNotFoundError(ds)


def load_labels(ds):
    return np.load(f"{PROJECT}/data/medmnist/{ds}.npz")[
        "train_labels"].ravel().astype(int)


def final_class_difficulty(ds):
    """1 - recall_c of the final ResNet-18, averaged over the four competitive
    arms and every training seed. Dedup on the full run key because
    metrics.jsonl holds more than one row per run."""
    seen, per = set(), defaultdict(list)
    for root in ROOTS:
        for pf in glob.glob(f"{root}/**/predictions_clean.npz", recursive=True):
            if f"/{ds}/" not in pf:
                continue
            arm = next((a for a in COMPETITIVE if f"/{a}/" in pf), None)
            if arm is None:
                continue
            mf = os.path.join(os.path.dirname(pf), "metrics.jsonl")
            if not os.path.exists(mf):
                continue
            meta = None
            for line in open(mf):
                line = line.strip()
                if not line:
                    continue
                try:
                    dd = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if dd.get("corruption", "clean") == "clean" and \
                        dd.get("severity", 0) in (0, None):
                    meta = dd
                    break
            if meta is None:
                continue
            key = (meta["dataset"], meta["ratio"], meta["method"],
                   meta.get("selection_seed"), meta.get("training_seed"))
            if key in seen:
                continue
            seen.add(key)
            try:
                z = np.load(pf)
                y = z["y_true"].astype(int).ravel()
                yh = z["probs"].argmax(1).astype(int).ravel()
            except Exception:
                continue
            for c in np.unique(y):
                m = y == c
                per[int(c)].append(float((yh[m] == c).mean()))
    if not per:
        return None, 0
    C = max(per) + 1
    diff = np.full(C, np.nan)
    for c, v in per.items():
        diff[c] = 1.0 - float(np.mean(v))
    return diff, len(seen)


# ------------------------------------------------------------ probe
def fit_predict(Z, y, tr, te, C, dev, min_steps, lr, wd, bs, seed):
    import torch
    import torch.nn as nn
    g = torch.Generator(device=dev.type).manual_seed(seed)
    head = nn.Linear(Z.shape[1], C).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    Zt = torch.from_numpy(Z)
    Ztr = Zt[tr].to(dev)
    ytr = torch.from_numpy(y[tr]).long().to(dev)
    spe = max(1, (len(tr) + bs - 1) // bs)
    epochs = max(1, -(-min_steps // spe))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * spe)
    head.train()
    for _ in range(epochs):
        perm = torch.randperm(len(tr), device=dev, generator=g)
        for s in range(0, len(tr), bs):
            b = perm[s:s + bs]
            opt.zero_grad()
            lossf(head(Ztr[b]), ytr[b]).backward()
            opt.step()
            sch.step()
    head.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(te), 8192):
            out.append(torch.softmax(head(Zt[te[s:s + 8192]].to(dev)), 1).cpu())
    del Ztr, ytr, head
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate([o.numpy() for o in out], 0)


def proxy_full(Z, y, C, dev, folds=5, seed=0, **kw):
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), int)
    for c in range(C):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % folds
    P = np.zeros((len(y), C), np.float32)
    for f in range(folds):
        tr = np.flatnonzero(fold != f)
        te = np.flatnonzero(fold == f)
        P[te] = fit_predict(Z, y, tr, te, C, dev, seed=seed * 100 + f, **kw)
    return P


def proxy_budget(Z, y, C, dev, bpc, repeats=20, seed=0, **kw):
    """Train on exactly bpc samples per class; predict everything else.
    Averaged over `repeats` independent class-balanced draws."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((len(y), C), np.float64)
    cnt = np.zeros(len(y), np.int64)
    for r in range(repeats):
        tr = np.concatenate([rng.choice(np.flatnonzero(y == c), bpc,
                                        replace=False) for c in range(C)])
        mask = np.ones(len(y), bool)
        mask[tr] = False
        te = np.flatnonzero(mask)
        acc[te] += fit_predict(Z, y, tr, te, C, dev, seed=seed * 1000 + r, **kw)
        cnt[te] += 1
    cnt = np.maximum(cnt, 1)
    return (acc / cnt[:, None]).astype(np.float32)


def summarise(P, y, C, final_diff, tag):
    d = 1.0 - P[np.arange(len(y)), y]
    yh = P.argmax(1)
    rec = np.array([float((yh[y == c] == c).mean()) for c in range(C)])
    x = np.array([float(d[y == c].mean()) for c in range(C)])
    rho, p = (float("nan"), float("nan"))
    if final_diff is not None:
        ok = ~np.isnan(final_diff[:C])
        if ok.sum() >= 3:
            rho, p = spearmanr(x[ok], final_diff[:C][ok])
    q = np.percentile(d, [10, 50, 90])
    print(f"    {tag:7s} probeBA {rec.mean():.4f}  d̄ {d.mean():.4f}  "
          f"p10/50/90 {q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f}  "
          f"frac(d<.05) {float((d < 0.05).mean()):.3f}  "
          f"rho(class d, final 1-recall) {rho:+.3f} (p={p:.3f})")
    return dict(tag=tag, probe_ba=float(rec.mean()), d_mean=float(d.mean()),
                frac_easy=float((d < 0.05).mean()), rho=float(rho),
                rho_p=float(p), class_d=x.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--ratio", type=float, default=0.02)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--min-steps", type=int, default=4000)
    ap.add_argument("--out", default=os.path.expanduser("~/proxy_regime.json"))
    a = ap.parse_args()

    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kw = dict(min_steps=a.min_steps, lr=1e-2, wd=1e-4, bs=1024)

    out = []
    for ds in a.datasets:
        try:
            Z = load_emb(ds)
            y = load_labels(ds)
        except Exception as e:
            print(f"[{ds}] SKIP {e}")
            continue
        if len(Z) != len(y):
            print(f"[{ds}] SKIP truncated cache {len(Z)} vs {len(y)}")
            continue
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + EPS)
        C = int(y.max()) + 1
        bpc = int(len(y) * a.ratio) // C
        fd, nruns = final_class_difficulty(ds)
        print(f"\n[{ds}] n={len(y)} C={C} bpc={bpc}  "
              f"final-model class difficulty from {nruns} archived runs")
        if fd is not None:
            print(f"    final 1-recall per class: "
                  + " ".join(f"{v:.3f}" for v in fd[:C]))

        t0 = time.time()
        Pf = proxy_full(Z, y, C, dev, **kw)
        rf = summarise(Pf, y, C, fd, "full")
        Pb = proxy_budget(Z, y, C, dev, bpc, repeats=a.repeats, **kw)
        rb = summarise(Pb, y, C, fd, "budget")
        print(f"    ({time.time()-t0:.0f}s)")
        out.append(dict(dataset=ds, C=C, bpc=bpc, n_archived_runs=nruns,
                        final_class_difficulty=(fd[:C].tolist()
                                                if fd is not None else None),
                        full=rf, budget=rb))

    # ---------------- pre-registered checks --------------------------------
    print(f"\n{'='*92}\nP1  frac(d<.05) drop, full -> budget "
          f"(predicted >= 0.20 on pathmnist and bloodmnist)\n{'='*92}")
    for r in out:
        drop = r["full"]["frac_easy"] - r["budget"]["frac_easy"]
        print(f"  {r['dataset']:14s} {r['full']['frac_easy']:.3f} -> "
              f"{r['budget']['frac_easy']:.3f}   drop {drop:+.3f}")

    print(f"\n{'='*92}\nP2/P3  rho(class demand, final class difficulty)"
          f"\n{'='*92}")
    print(f"  {'dataset':14s}{'full':>9s}{'budget':>9s}{'better?':>10s}")
    nbetter = 0
    for r in out:
        b = r["budget"]["rho"] > r["full"]["rho"]
        nbetter += int(b)
        print(f"  {r['dataset']:14s}{r['full']['rho']:>9.3f}"
              f"{r['budget']['rho']:>9.3f}{'budget' if b else 'full':>10s}")
    print(f"\n  P2 budget better on {nbetter}/{len(out)} "
          f"(predicted >= 3/5)")
    pm = next((r for r in out if r["dataset"] == "pathmnist"), None)
    if pm:
        print(f"  P3 pathmnist full rho = {pm['full']['rho']:+.3f} "
              f"(predicted < 0.40)")

    print(f"\n{'='*92}\nDECISION\n{'='*92}")
    units = {}
    for r in out:
        u = LIVE_UNIT.get(r["dataset"])
        if u:
            units[u] = max(units.get(u, -9.0), r["budget"]["rho"])
    for u, v in sorted(units.items()):
        print(f"  unit {u:10s} budget rho {v:+.3f} -> "
              f"{'PASS' if v >= 0.6 else 'fail'}")
    npass = sum(1 for v in units.values() if v >= 0.6)
    go = (nbetter >= 3) and (npass >= 2)
    print(f"\n  P2 {nbetter}/{len(out)}, live units at rho>=0.6: {npass}/3")
    print(f"  VERDICT: round 2 on the budget-matched proxy "
          f"{'IS WORTH GPU' if go else 'IS NOT WORTH GPU'}")
    if not go:
        print("  If BOTH regimes are near zero, the honest reading is that "
              "per-sample demand is not estimable from frozen UNI at this "
              "budget -- that closes the demand-weighting family, it does not "
              "motivate a third proxy.")

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    sys.path.insert(0, "/project/prj-sis01/xuxiaoyu/graph_select")
    main()
