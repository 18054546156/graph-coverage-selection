#!/usr/bin/env python
"""E7 (report/e7_main4_ablation_prereg_20261004.md): module ablation on the 4 low-overlap Table-1 datasets
= the negative-control / dose-response gate of the problem -> module -> ablation chain. Seeds s42-46, A100 only.

  python e7_eval.py --cluster   -> (on luhpc, in $S/code) harvest runs/acs_e7_20261004 -> e7_cells.json
  python code/e7_eval.py test   -> preregistered tests (§4) -> stdout (saved as results/acs/e7_test.txt)

Unit of analysis = ROW (dataset x ratio; seed-mean first) or DATASET (mean of its two rows), never the seed:
the 5 seeds of a row share one selected subset, so seed-level tests are pseudo-replication.
"""
import collections
import glob
import itertools
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TREE = "/mnt/prj01/hgrp-1502-5TB/tdgs_shared/runs/acs_e7_20261004"
LOGS = "/mnt/prj01/hgrp-1502-5TB/tdgs_shared/acs/logs"
S5 = [42, 43, 44, 45, 46]
R2 = (0.02, 0.05)
HIGH = ["dermamnist", "retinamnist", "tissuemnist", "breastmnist", "octmnist"]
LOW = ["organsmnist", "organamnist", "bloodmnist", "pathmnist"]
# label overlap = 1 - leave-one-out kNN BA on UNI train features (k=50); report/dataset_scope_20261003.md §3,
# fixed 10-03, before E2 / main-4 / E5 training
AMBIG = dict(dermamnist=.716, retinamnist=.641, tissuemnist=.603, breastmnist=.376, octmnist=.287,
             organsmnist=.227, organamnist=.103, bloodmnist=.051, pathmnist=.004)
T0HIGH = [(d, r) for d in ("dermamnist", "octmnist", "tissuemnist") for r in R2]   # E5 tau=0 rows with _l00 arms


def harvest():
    out, seen = [], set()
    for f in sorted(glob.glob(f"{TREE}/formal/ratio_*/*/*/seed_*/results/**/metrics.jsonl", recursive=True)):
        m = re.search(r"ratio_([0-9.]+)/(\w+)/(\w+)/seed_(\d+)/", f)
        key = (m[2], float(m[1]), m[3], int(m[4]))
        if key in seen:
            continue
        j = [json.loads(l) for l in open(f) if l.strip()]
        j = [x for x in j if x.get("corruption", "clean") == "clean"]
        if j:
            seen.add(key)
            out.append(dict(ds=key[0], ratio=key[1], arm=key[2], seed=key[3],
                            ba=100 * j[-1]["ba"], worst=100 * j[-1]["worst_recall"]))
    gpu = {}
    for lf in glob.glob(f"{LOGS}/acse7_*.out"):
        txt = open(lf).read()
        g = "H100" if "H100" in txt else ("A100" if "A100" in txt else "?")
        for key in re.findall(r"ok (\w+_r[0-9.]+_\w+_s\d+)", txt):
            gpu[key] = g
    for x in out:
        x["gpu"] = gpu.get(f"{x['ds']}_r{x['ratio']}_{x['arm']}_s{x['seed']}", "?")
    json.dump(out, open("e7_cells.json", "w"), indent=0)
    print(f"wrote e7_cells.json ({len(out)} cells; gpu {collections.Counter(x['gpu'] for x in out)})")


def load():
    """Dev cells (acs_eval via e5_eval: E2 grid, bench, E5 arms) + main-4 ACS (A100) + E7 arms (A100 only)."""
    import e5_eval
    C, _, _ = e5_eval.load()
    for x in json.load(open("results/acs/acs_main4_cells.json")):
        if x["gpu"] == "A100":
            C[(x["ds"], round(x["ratio"], 4), x["arm"], x["seed"])] = dict(ba=x["ba"], worst=x["worst"])
    new = json.load(open("results/acs/e7_cells.json"))
    bad = [x for x in new if x["gpu"] != "A100"]
    for x in new:
        if x not in bad:
            C[(x["ds"], round(float(x["ratio"]), 4), x["arm"], int(x["seed"]))] = dict(ba=x["ba"], worst=x["worst"])
    return C, new, bad


def test():
    C, new, bad = load()
    both_off = json.load(open("results/acs/e7_identity.json"))   # {"<ds>_<r>": "graph_a2" | "acs_q00_t000_l00"}
    print(f"E7 cells {len(new)}; excluded (not A100): {len(bad)} {[(x['ds'], x['ratio'], x['arm'], x['seed']) for x in bad]}")

    def rv(d, r, a, ep):
        k = [(d, r, a, s) for s in S5]
        return None if not all(x in C for x in k) else float(np.mean([C[x][ep] for x in k]))

    def off2(d, r):   # F1 off and F2 off: the per-class-quota Graph-A2 objective
        return both_off.get(f"{d}_{r}", "graph_a2") if d in LOW else "graph_a2"

    def dF1(d, r, ep):        # F1 alone (F2 off, tau=0): cls - Graph-A2   (= E2 proxy on the high rows)
        a, b = rv(d, r, "cls", ep), rv(d, r, off2(d, r), ep)
        return None if a is None or b is None else a - b

    def dF2noF1(d, r, ep):    # F2 alone (F1 off): ACS_l00 - both-off
        b = "acs_q00_t000_l00" if d in HIGH else off2(d, r)
        a, b = rv(d, r, "acs_q25_t000_l00", ep), rv(d, r, b, ep)
        return None if a is None or b is None else a - b

    rng = np.random.default_rng(20261004)
    ps = {}
    for ep in ("ba", "worst"):
        print(f"\n## {ep}  (row = seed-mean; positive = module helps)")
        print(f"{'row':16s} {'overlap':>7s} {'dF1':>7s} {'dF2|noF1':>9s} {'ACS-cls':>8s} {'ACS-ACS_l00':>11s}")
        rows = {}
        for d in HIGH + LOW:
            for r in R2:
                f1, f2 = dF1(d, r, ep), dF2noF1(d, r, ep)
                acs = "acs_q25_t000"
                g = lambda a, b: (None if rv(d, r, a, ep) is None or rv(d, r, b, ep) is None
                                  else rv(d, r, a, ep) - rv(d, r, b, ep))
                rows[(d, r)] = dict(f1=f1, f2=f2, f2g1=g(acs, "cls") if d in LOW or (d, r) in T0HIGH else None,
                                    f1g2=g(acs, "acs_q25_t000_l00"))
                fmt = lambda v, w: f"{'—':>{w}s}" if v is None else f"{v:+{w}.2f}"
                q = rows[(d, r)]
                print(f"{d[:-5]:8s} {r:5.2f}   {AMBIG[d]:7.3f} {fmt(q['f1'], 7)} {fmt(q['f2'], 9)} "
                      f"{fmt(q['f2g1'], 8)} {fmt(q['f1g2'], 11)}")

        # H1 dose-response at dataset level: Spearman(overlap, dataset-mean dF1), exact over 9! orderings
        dsv = {d: np.mean([rows[(d, r)]["f1"] for r in R2]) for d in HIGH + LOW
               if all(rows[(d, r)]["f1"] is not None for r in R2)}
        ds = sorted(dsv)
        x = np.array([AMBIG[d] for d in ds]); y = np.array([dsv[d] for d in ds])
        rk = lambda v: np.argsort(np.argsort(v)).astype(float)
        rx, ry = rk(x), rk(y)
        rho = np.corrcoef(rx, ry)[0, 1]
        n_ge = tot = 0
        for p in itertools.permutations(range(len(ds))):
            tot += 1; n_ge += np.corrcoef(rx, ry[list(p)])[0, 1] >= rho - 1e-12
        print(f"\nH1 dose (datasets n={len(ds)}): Spearman(overlap, dF1) = {rho:+.3f}; exact one-sided p = {n_ge / tot:.4f}")

        def two_sample(hi, lo, key, name):
            a = np.array([rows[k][key] for k in hi if rows[k][key] is not None])
            b = np.array([rows[k][key] for k in lo if rows[k][key] is not None])
            T = a.mean() - b.mean(); z = np.concatenate([a, b]); n = len(a)
            cnt = tot = 0
            for idx in itertools.combinations(range(len(z)), n):
                m = np.zeros(len(z), bool); m[list(idx)] = True
                tot += 1; cnt += z[m].mean() - z[~m].mean() >= T - 1e-12
            se = b.std(ddof=1) / np.sqrt(len(b))
            from scipy import stats
            tq = stats.t.ppf(.975, len(b) - 1)
            print(f"{name}: high mean {a.mean():+.2f} (n={len(a)}, rows+ {int((a > 0).sum())}) vs low mean "
                  f"{b.mean():+.2f} (n={len(b)}, rows+ {int((b > 0).sum())}); diff {T:+.2f}; exact one-sided p = "
                  f"{cnt / tot:.4f}; low 95% CI [{b.mean() - tq * se:+.2f}, {b.mean() + tq * se:+.2f}]"
                  f"{'  -> within ±1pp' if abs(b.mean()) + tq * se <= 1 else ''}")
            return cnt / tot

        hi10 = [(d, r) for d in HIGH for r in R2]; lo8 = [(d, r) for d in LOW for r in R2]
        p2 = two_sample(hi10, lo8, "f1", "H2 F1 negative control (dF1)")
        p3 = two_sample(T0HIGH, lo8, "f2", "H3 F2 negative control (dF2|noF1)")
        two_sample(T0HIGH, lo8, "f2g1", "D1 F2 given F1 (ACS-cls), descriptive")
        two_sample(T0HIGH, lo8, "f1g2", "D2 F1 given F2 (ACS-ACS_l00), descriptive")
        ps[ep] = dict(H1=n_ge / tot, H2=p2, H3=p3)

    p = ps["ba"]; order = sorted(p, key=p.get); m = len(order); adj, run = {}, 0
    for i, k in enumerate(order):
        run = max(run, min(1, (m - i) * p[k])); adj[k] = run
    print("\n## Holm (BA, family H1-H3)")
    for k in ("H1", "H2", "H3"):
        print(f"  {k}: p {p[k]:.4f}  Holm {adj[k]:.4f}  {'PASS' if adj[k] < .05 else 'FAIL'}")


if __name__ == "__main__":
    harvest() if "--cluster" in sys.argv else test()
