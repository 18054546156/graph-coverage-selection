#!/usr/bin/env python
"""Zero-GPU observations behind report/acs_principles_20261003.md (dev seeds s42-46 only; nothing new is trained).

  O1  hyperparameter landscape: every (q, tau) cell of the E2 grid per row (BA, worst recall), and the 12
      candidate rules (b* in {0,5,10,20} x q in {0,.25,.5}) as 10-row means -> how flat is the optimum.
  O2  selected-set purity (UNI k=50 same-class share, class-balanced) of random (~ pool), herding, cls,
      Graph-A2 and the ACS cells, from results/acs/report/*_acs.json.
  O3  per-class plurality-loss rate eps_c = 1 - (plurality-kept / n_c) (the knnf filter's removal rate),
      the principled scale for q.
Run from paper_tdgs/:  python code/acs_principles_obs.py > results/acs/principles_obs.txt
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acs_eval as E  # noqa: E402

C, _, _ = E.load()
rows = [(d, r) for d in E.HIGH for r in E.RATIOS]
S5 = E.S5
have = lambda d, r, a: all((d, r, a, s) in C for s in S5)
V = lambda d, r, a, ep: np.array([C[(d, r, a, s)][ep] for s in S5])
RULE = {tuple(k.rsplit("_", 1)): v for k, v in json.load(open("results/acs/e2_selected_rule.json"))["arms"].items()}

print("## O1a  E2 grid, seed-mean BA / worst (s42-46); * = arm chosen by the frozen rule (b*=10, q*=.25)")
cells = [(q, t) for q in E.QS for t in E.TAUS]
print("row          bpc  " + " ".join(f"q{int(q*100):02d}t{int(t*100):03d}".rjust(13) for q, t in cells))
for d, r in rows:
    out = []
    for q, t in cells:
        a = E.arm(q, t)
        mark = "*" if RULE[(d, str(r))] == a else " "
        out.append(f"{V(d, r, a, 'ba').mean():5.1f}/{V(d, r, a, 'worst').mean():5.1f}{mark}".rjust(13)
                   if have(d, r, a) else "--".rjust(13))
    print(f"{d[:6]} {r:<5} {E.BPC[(d, r)]:5d} " + " ".join(out))

print("\n## O1b  12 candidate rules, 10-row mean BA / worst (dev; the rule was chosen here)")
res = []
for bstar in (0, 5, 10, 20):
    for q in E.QS:
        arms = {(d, r): E.arm(q, E.tau(E.BPC[(d, r)], bstar)) for d, r in rows}
        res.append((bstar, q, np.mean([V(d, r, arms[k], "ba").mean() for k in rows for d, r in [k]]),
                    np.mean([V(d, r, arms[k], "worst").mean() for k in rows for d, r in [k]])))
top = max(x[2] for x in res)
for b, q, ba, w in sorted(res, key=lambda x: -x[2]):
    print(f"  b*={b:2d} q={q:.2f}  BA {ba:6.2f} ({ba - top:+.2f})  worst {w:6.2f}")

print("\n## O1c  per-row oracle (q,tau) cell vs rule cell (BA)")
for d, r in rows:
    best = max(((q, t) for q, t in cells if have(d, r, E.arm(q, t))), key=lambda c: V(d, r, E.arm(*c), "ba").mean())
    ra = RULE[(d, str(r))]
    print(f"  {d[:6]} {r}: oracle {E.arm(*best):13s} {V(d, r, E.arm(*best), 'ba').mean():6.2f} | rule {ra:13s} "
          f"{V(d, r, ra, 'ba').mean():6.2f} | regret {V(d, r, E.arm(*best), 'ba').mean() - V(d, r, ra, 'ba').mean():+.2f}")

print("\n## O2  selected-set purity (UNI), class-balanced; random ~ pool")
names = ["random", "graph_a2", "herding", "cls", "acs_q25_t100", "acs_q25_t050", "acs_q25_t000", "acs_q50_t000", "acs_q50_t100"]
print("row          " + " ".join(n.replace("acs_", "")[:10].rjust(10) for n in names))
for d, r in rows:
    rep = json.load(open(f"results/acs/report/{d}_r{r}_acs.json"))
    print(f"{d[:6]} {r:<5} " + " ".join(
        (f"{rep['arms'][n]['purity']['uni']:10.3f}" if n in rep["arms"] and "purity" in rep["arms"][n] else "--".rjust(10))
        for n in names))

print("\n## O3  per-class plurality-loss rate eps_c (share of class with own label not the kNN plurality)")
for d in E.HIGH:
    rep = json.load(open(f"results/acs/report/{d}_r0.02_acs.json"))
    kept = np.array(rep["checks"]["I3_plurality_kept_per_class"]["ours"], float)
    ret = rep["retention"]["0.25"]
    n = np.array(ret["kept_per_class"], float) / np.array(ret["frac"], float)
    eps = 1 - kept / n
    print(f"  {d[:6]} C={len(n)}  eps_c " + " ".join(f"{e:.2f}" for e in eps)
          + f" | mean {eps.mean():.2f} median {np.median(eps):.2f} | classes with eps>.25: {(eps > .25).sum()}/{len(n)}")
