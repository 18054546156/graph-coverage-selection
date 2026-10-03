#!/usr/bin/env python
"""Leave-one-seed-out (LOSO) check of the E2 rule selection: is the dev 9/10 a winner's-curse artifact?

The rule (b*, q*) was chosen on s42-46 from the 12 preregistered candidates (E2 prereg §3), and the headline
ACS-vs-per-row-SOTA table is reported on the same s42-46.  Here, for each held-out seed s:
  1. re-run the E2 §3 selection (max 10-row mean BA, ties < .25pp broken by worst recall, then small q, small b*)
     on the other 4 seeds only;
  2. per row, pick the comparator = best published method (8 Table-1 + TypiClust/ProbCover/MaxHerding) on the
     same 4 seeds;
  3. score ACS(rule_s) - comparator_s on the held-out seed only.
Every held-out number is out-of-sample for both the rule and the comparator choice.  Zero GPU, dev cells only.
Run from paper_tdgs/:  python code/acs_loso.py > results/acs/loso.txt
"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acs_eval as E  # noqa: E402

C, _, _ = E.load()
rows = [(d, r) for d in E.HIGH for r in E.RATIOS]
S5 = E.S5
pub = lambda d, r: [b for b in E.T1B + E.NEWB if all((d, r, b, s) in C for s in S5)]
v = lambda d, r, a, seeds, ep="ba": np.array([C[(d, r, a, s)][ep] for s in seeds])


def select(seeds):
    res = []
    for bstar in (0, 5, 10, 20):
        for q in E.QS:
            arms = {(d, r): E.arm(q, E.tau(E.BPC[(d, r)], bstar)) for d, r in rows}
            ba = np.mean([v(d, r, arms[k], seeds).mean() for k in rows for d, r in [k]])
            wr = np.mean([v(d, r, arms[k], seeds, "worst").mean() for k in rows for d, r in [k]])
            res.append(dict(bstar=bstar, q=q, ba=ba, worst=wr, arms=arms))
    top = max(x["ba"] for x in res)
    return sorted([x for x in res if top - x["ba"] < .25], key=lambda x: (-x["worst"], x["q"], x["bstar"]))[0]


full = select(S5)
print(f"in-sample rule (all 5 seeds): b*={full['bstar']} q*={full['q']}")
D = {k: [] for k in rows}            # LOSO held-out differences, per row, one per seed
for s in S5:
    tr = [x for x in S5 if x != s]
    w = select(tr)
    picks = []
    for d, r in rows:
        comp = max(pub(d, r), key=lambda b: v(d, r, b, tr).mean())
        a = w["arms"][(d, r)]
        D[(d, r)].append(v(d, r, a, [s])[0] - v(d, r, comp, [s])[0])
        picks.append(f"{d[:4]}{int(r*100)}:{a.replace('acs_', '')}>{comp}")
    print(f"held-out s{s}: rule b*={w['bstar']} q*={w['q']}  | " + " ".join(picks))

print("\n## per row: in-sample (rule + comparator chosen on all 5 seeds) vs LOSO (both chosen on the other 4)")
print(f"{'row':12s} {'in-sample d':>11s} {'p':>6s} | {'LOSO d':>7s} {'p':>6s} {'wins':>5s} | optimism")
ins_first = loso_first = 0
opt = []
for d, r in rows:
    a = full["arms"][(d, r)]
    comp = max(pub(d, r), key=lambda b: v(d, r, b, S5).mean())
    di = v(d, r, a, S5) - v(d, r, comp, S5)
    dl = np.array(D[(d, r)])
    pi = stats.ttest_1samp(di, 0).pvalue
    pl = stats.ttest_1samp(dl, 0).pvalue
    ins_first += di.mean() >= 0
    loso_first += dl.mean() >= 0
    opt.append(di.mean() - dl.mean())
    print(f"{d[:6]} {r:<5} {di.mean():+11.2f} {pi:6.3f} | {dl.mean():+7.2f} {pl:6.3f} {int((dl > 0).sum()):3d}/5 | {opt[-1]:+.2f}")
print(f"\nfirst (mean d >= 0): in-sample {ins_first}/10, LOSO {loso_first}/10; "
      f"mean optimism {np.mean(opt):+.2f}pp (median {np.median(opt):+.2f})")
allin = np.mean([(v(d, r, full['arms'][(d, r)], S5) - v(d, r, max(pub(d, r), key=lambda b: v(d, r, b, S5).mean()), S5)).mean()
                 for d, r in rows])
print(f"10-row mean d: in-sample {allin:+.2f}, LOSO {np.mean([np.mean(D[k]) for k in rows]):+.2f}")

print("\n## isolate the rule's own selection optimism: comparator fixed (in-sample, best on all 5), only the rule is LOSO")
ro, rfirst = [], 0
for d, r in rows:
    comp = max(pub(d, r), key=lambda b: v(d, r, b, S5).mean())
    dl = [v(d, r, select([x for x in S5 if x != s])["arms"][(d, r)], [s])[0] - v(d, r, comp, [s])[0] for s in S5]
    di = (v(d, r, full["arms"][(d, r)], S5) - v(d, r, comp, S5)).mean()
    ro.append(di - np.mean(dl))
    rfirst += np.mean(dl) >= 0
    print(f"{d[:6]} {r:<5} in-sample {di:+6.2f}  rule-LOSO {np.mean(dl):+6.2f}  optimism {ro[-1]:+.2f}")
print(f"rule-selection optimism: mean {np.mean(ro):+.2f}pp, max {np.max(ro):+.2f}pp; "
      f"first under rule-LOSO {rfirst}/10")
