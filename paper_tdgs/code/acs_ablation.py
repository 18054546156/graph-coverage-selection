#!/usr/bin/env python
"""Component ablation read off the E2 factorial (no new training), 10 high-ambiguity rows, s42-46, BA.
ACS = rule (b*=10, q*=0.25); -F2 = same tau, q=0; -F3 = same q, tau=0; cls = F1 only; A2 = Graph-A2 (no F1).
Run from paper_tdgs/:  python code/acs_ablation.py"""
import sys
import numpy as np
from scipy import stats
sys.path.insert(0, "code")
import acs_eval as E
C, _, _ = E.load()
have = lambda d, r, a: all((d, r, a, s) in C for s in E.S5)
V = lambda d, r, a, ep: np.array([C[(d, r, a, s)][ep] for s in E.S5])
for ep in ("ba", "worst"):
    print(f"\n## {ep}: ACS / ACS-F2 / ACS-F3 / cls (F1 only) / Graph-A2")
    acc = {k: [] for k in ("acs", "noF2", "noF3", "cls", "a2")}
    for d in E.HIGH:
        for r in E.RATIOS:
            t = E.tau(E.BPC[(d, r)], 10)
            arms = dict(acs=E.arm(.25, t), noF2=E.arm(0, t), noF3=E.arm(.25, 0), cls=E.arm(0, 0),
                        a2="graph_a2" if have(d, r, "graph_a2") else "a2_uni")
            v = {k: V(d, r, a, ep) for k, a in arms.items()}
            for k in acc:
                acc[k].append(v[k])
            print(f"  {d[:6]} {r:<5} tau={t:.1f} " + " ".join(f"{k}={v[k].mean():6.2f}" for k in acc))
    m = {k: np.mean([x.mean() for x in acc[k]]) for k in acc}
    def ab(a, b):
        dd = np.concatenate(acc[a]) - np.concatenate(acc[b])
        rows = sum(x.mean() > y.mean() for x, y in zip(acc[a], acc[b]))
        return f"{dd.mean():+.2f}pp, rows {rows}/10, paired-cell p={stats.ttest_1samp(dd, 0).pvalue:.3g}"
    print("  mean " + " ".join(f"{k}={m[k]:.2f}" for k in m))
    print(f"  F2 (ACS vs -F2): {ab('acs', 'noF2')}")
    print(f"  F3 (ACS vs -F3): {ab('acs', 'noF3')}")
    print(f"  F1 (cls vs A2) : {ab('cls', 'a2')}")
    print(f"  all (ACS vs A2): {ab('acs', 'a2')}")
