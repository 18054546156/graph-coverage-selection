"""Recompute every 2%/5% paired contrast from raw per-cell harvest (R1 + R2 + 2b).

Paired by (dataset, seed). Reports pooled mean, per-dataset means, sign count,
a dataset-level mean (each dataset weighted equally), and a hierarchical
bootstrap CI (resample datasets, then seeds within dataset) so the interval
reflects that there are only 5 datasets.
"""
import json, sys, itertools
import numpy as np
from collections import defaultdict

files = sys.argv[1:]
cells = {}
for f in files:
    for r in json.load(open(f)):
        key = (r['ds'], round(r['ratio'], 3), r['arm'], int(r['seed']))
        cells[key] = r                        # later files override earlier
DS = ['pathmnist', 'bloodmnist', 'organamnist', 'organsmnist', 'tissuemnist']

def paired(a, b, ratio, metric):
    out = {}
    for ds in DS:
        seeds = sorted({k[3] for k in cells if k[:3] == (ds, ratio, a)} &
                       {k[3] for k in cells if k[:3] == (ds, ratio, b)})
        out[ds] = {s: 100 * (cells[(ds, ratio, a, s)][metric] -
                             cells[(ds, ratio, b, s)][metric]) for s in seeds}
    return out

def boot(d, B=20000, seed=0):
    rng = np.random.default_rng(seed)
    dss = [k for k in d if d[k]]
    arr = [np.array(list(d[k].values())) for k in dss]
    m = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(arr), len(arr))
        m[b] = np.mean([rng.choice(arr[i], len(arr[i])).mean() for i in pick])
    return np.percentile(m, [2.5, 97.5])

def report(a, b, ratio, metric, label=''):
    d = paired(a, b, ratio, metric)
    per = {k: np.mean(list(v.values())) for k, v in d.items() if v}
    ns = {k: len(v) for k, v in d.items()}
    allv = [x for v in d.values() for x in v.values()]
    dm = np.mean(list(per.values()))
    ex = np.mean([per[k] for k in per if k != 'tissuemnist'])
    lo, hi = boot(d)
    pos = sum(v > 0 for v in per.values())
    cpos = sum(x > 0 for x in allv)
    print(f"{a:>14s} - {b:<14s} r={ratio} {metric:5s} n/ds={sorted(set(ns.values()))} "
          f"dsmean {dm:+.2f} [{lo:+.2f},{hi:+.2f}] exTis {ex:+.2f} "
          f"sign {pos}/{len(per)} cells {cpos}/{len(allv)} | " +
          " ".join(f"{k[:5]} {per[k]:+.2f}" for k in DS if k in per))
    return per

if __name__ == '__main__':
    arms = sorted({k[2] for k in cells})
    print('arms:', arms)
    for ratio in (0.02, 0.05):
        for arm in arms:
            for ds in DS:
                s = sorted(k[3] for k in cells if k[:3] == (ds, ratio, arm))
                if s: pass
        cnt = defaultdict(int)
        for k in cells:
            if k[1] == ratio: cnt[k[2]] += 1
        print(f'ratio {ratio} counts:', dict(cnt))
    pairs = [('tdgs_cls', 'graph_a2'), ('a2_perclass', 'graph_a2'),
             ('tdgs_cls', 'a2_perclass'), ('tdgs_mask', 'graph_a2'),
             ('tdgs_wcls', 'graph_a2'), ('tdgs_cls', 'tdgs_mask'),
             ('tdgs_cls', 'tdgs_wcls'), ('tdgs_cls', 'tdgs_cls_perm'),
             ('tdgs_cls_perm', 'graph_a2'), ('tdgs_lam1', 'tdgs_cls'),
             ('tdgs_lam1', 'graph_a2'), ('tdgs_lam1', 'a2_perclass'),
             ('tdgs_mask', 'a2_perclass'), ('tdgs_d', 'tdgs_cls'),
             ('tdgs_du', 'tdgs_perm')]
    for metric in ('ba', 'worst'):
        print(f'\n===== ratio 0.02 {metric} =====')
        for a, b in pairs:
            report(a, b, 0.02, metric)
        # interaction on common (ds, seed) basis
        print('-- interaction cls - mask - wcls + a2 (common basis) --')
        per = {}
        for ds in DS:
            ss = set.intersection(*[{k[3] for k in cells if k[:3] == (ds, 0.02, arm)}
                                    for arm in ('tdgs_cls', 'tdgs_mask', 'tdgs_wcls', 'graph_a2')])
            v = [100 * (cells[(ds, .02, 'tdgs_cls', s)][metric] - cells[(ds, .02, 'tdgs_mask', s)][metric]
                        - cells[(ds, .02, 'tdgs_wcls', s)][metric] + cells[(ds, .02, 'graph_a2', s)][metric]) for s in ss]
            per[ds] = (np.mean(v), len(v))
        print('   ', ' '.join(f"{k[:5]} {m:+.2f}(n{n})" for k, (m, n) in per.items()),
              f"| dsmean {np.mean([m for m, _ in per.values()]):+.2f}")
        print(f'\n===== ratio 0.05 {metric} =====')
        report('tdgs_cls', 'graph_a2', 0.05, metric)
