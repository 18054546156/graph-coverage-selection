#!/usr/bin/env python
"""ACS (E2-selected rule) vs the PUBLISHED Table-1 methods only, per row, paired over s42-46.

acs_eval.py §4.6 compares against every arm we have (incl. our own earlier variants cls / mv_mean / knnf_*);
this script restricts the opponents to the 8 Table-1 methods (graph_a2, herding, facility, random, fps, eva,
el2n_top, forgetting), whichever exist for the row. On derma/OCT the Graph-A2 column is `a2_uni` (same
implementation, W1 tree). Run from paper_tdgs/:  python code/acs_vs_table1.py
"""
import json
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, "code")
import acs_eval as E  # noqa: E402

C, _, _ = E.load()
have = lambda d, r, a: all((d, r, a, s) in C for s in E.S5)
V = lambda d, r, a, ep="ba": np.array([C[(d, r, a, s)][ep] for s in E.S5])
sel = json.load(open("results/acs/e2_selected_rule.json"))["arms"]

print(f"{'row':12s} {'ACS arm':14s} {'ACS':>6s} {'A2':>6s} {'ACS-A2':>7s} | {'best T1':10s} {'T1':>6s} "
      f"{'d':>6s} {'p':>6s} {'wins':>5s} | worst: {'best T1':10s} {'d':>6s} {'p':>6s} | n_T1")
for d in E.HIGH:
    for r in E.RATIOS:
        a = sel[f"{d}_{r}"]
        t1 = [b for b in E.T1B if have(d, r, b)] + (["a2_uni"] if have(d, r, "a2_uni") else [])
        a2 = "graph_a2" if have(d, r, "graph_a2") else "a2_uni"
        out = []
        for ep in ("ba", "worst"):
            bt = max(t1, key=lambda b: V(d, r, b, ep).mean())
            dd = V(d, r, a, ep) - V(d, r, bt, ep)
            out.append((bt, dd.mean(), stats.ttest_1samp(dd, 0).pvalue, int((dd > 0).sum())))
        (b1, d1, p1, w1), (b2, d2, p2, _) = out
        print(f"{d[:6]} {r:<5} {a:14s} {V(d, r, a).mean():6.2f} {V(d, r, a2).mean():6.2f} "
              f"{V(d, r, a).mean() - V(d, r, a2).mean():+7.2f} | {b1:10s} {V(d, r, b1).mean():6.2f} "
              f"{d1:+6.2f} {p1:6.3f} {w1:3d}/5 | worst: {b2:10s} {d2:+6.2f} {p2:6.3f} | {len(t1)}")
