#!/usr/bin/env python
"""ACS main-4 run (report/acs_main4_prereg_20261003.md, stamped 2026-10-03 06:04:39 UTC): harvest + §3 analysis.

  python acs_main4_eval.py --cluster   -> (on luhpc, in $S/code) harvest the 40 acs_q25_t000 cells -> acs_main4_cells.json
  python code/acs_main4_eval.py        -> §3 analysis from results/acs/acs_main4_cells.json (run from paper_tdgs/)

§3 (fixed in the prereg): opponent pool = published methods with all 5 paired A100 seeds in the row
(8 Table-1 + TypiClust/ProbCover/MaxHerding); primary ACS - SOTA (SOTA = best seed-mean BA), two-sided paired t, raw + BH
over the 8 rows; same on worst-class recall (SOTA re-chosen); ACS rank; secondary ACS - Graph-A2.
Exploratory, NOT in the prereg (written after the cells landed, labelled as such): ACS - cls on these rows = the F2 effect
where the rule switches F3 off (E5d in report/acs_causal_chain_20261003.md); cls = tdgs_cls from round1/round2,
the same trees as the Table-1 Graph-A2 column.
"""
import glob
import json
import os
import re
import sys

import numpy as np

TREE = "/mnt/prj01/hgrp-1502-5TB/tdgs_shared/runs/acs_20261002"
LOGS = "/mnt/prj01/hgrp-1502-5TB/tdgs_shared/acs/logs"
ARM = "acs_q25_t000"
MAIN = ["bloodmnist", "organamnist", "organsmnist", "pathmnist"]


def harvest():
    gpu = {}
    for lf in glob.glob(f"{LOGS}/acsm4_*.out"):
        txt = open(lf).read()
        g = "H100" if "H100" in txt else ("A100" if "A100" in txt else "?")
        for key in re.findall(r"ok (\w+_r[0-9.]+_\w+_s\d+)", txt):
            gpu[key] = g
    out, seen = [], set()
    for d in MAIN:
        # each cell writes metrics.jsonl twice (results/ and results/<ds>/precomputed/...); keep the first, as acs_eval
        for f in sorted(glob.glob(f"{TREE}/formal/ratio_*/{d}/{ARM}/seed_*/results/**/metrics.jsonl", recursive=True)):
            m = re.search(r"ratio_([0-9.]+)/(\w+)/(\w+)/seed_(\d+)/", f)
            if (d, m[1], m[4]) in seen:
                continue
            j = [json.loads(l) for l in open(f) if l.strip()]
            j = [x for x in j if x.get("corruption", "clean") == "clean"]
            if not j:
                continue
            seen.add((d, m[1], m[4]))
            x = dict(ds=d, ratio=float(m[1]), arm=ARM, seed=int(m[4]), ba=100 * j[-1]["ba"],
                     worst=100 * j[-1]["worst_recall"])
            x["gpu"] = gpu.get(f"{d}_r{m[1]}_{ARM}_s{m[4]}", "?")
            out.append(x)
    json.dump(out, open("acs_main4_cells.json", "w"), indent=0)
    print(f"wrote acs_main4_cells.json ({len(out)} cells; gpu {sorted(set(x['gpu'] for x in out))})")


def analyse():
    from scipy import stats
    sys.path.insert(0, "code")
    import acs_eval as E
    C, _, _ = E.load()
    new = json.load(open("results/acs/acs_main4_cells.json"))
    bad = [x for x in new if x["gpu"] != "A100"]
    print(f"main-4 ACS cells {len(new)} (expected 40); not A100: {len(bad)}")
    for x in new:
        if x not in bad:
            C[(x["ds"], round(x["ratio"], 4), "acs", x["seed"])] = dict(ba=x["ba"], worst=x["worst"])
    cls = {}
    for f in ("results/round1/tdgs_round1_harvest.json", "results/round2/round2_harvest_final_2b.json"):
        for e in json.load(open(f)):
            if e["arm"] == "tdgs_cls" and e["ds"] in MAIN:
                cls[(e["ds"], round(float(e["ratio"]), 4), "cls", int(e["seed"]))] = dict(ba=100 * e["ba"],
                                                                                        worst=100 * e["worst"])
    C.update(cls)
    have = lambda d, r, a: all((d, r, a, s) in C for s in E.S5)
    V = lambda d, r, a, ep="ba": np.array([C[(d, r, a, s)][ep] for s in E.S5])
    pool_all = E.T1B + E.NEWB

    res = {}
    for ep in ("ba", "worst"):
        rows = []
        for d in MAIN:
            for r in E.RATIOS:
                if not have(d, r, "acs"):
                    print(f"  {d} {r}: ACS incomplete -> row not analysed")
                    continue
                pool = [b for b in pool_all if have(d, r, b)]
                sota = max(pool, key=lambda b: V(d, r, b, ep).mean())
                dd = V(d, r, "acs", ep) - V(d, r, sota, ep)
                means = sorted([V(d, r, b, ep).mean() for b in pool] + [V(d, r, "acs", ep).mean()], reverse=True)
                rank = means.index(V(d, r, "acs", ep).mean()) + 1
                a2 = V(d, r, "acs", ep) - V(d, r, "graph_a2", ep) if have(d, r, "graph_a2") else None
                c = V(d, r, "acs", ep) - V(d, r, "cls", ep) if have(d, r, "cls") else None
                rows.append(dict(ds=d, r=r, acs=V(d, r, "acs", ep).mean(), sota=sota, sota_v=V(d, r, sota, ep).mean(),
                                 d=dd.mean(), p=stats.ttest_rel(V(d, r, "acs", ep), V(d, r, sota, ep)).pvalue,
                                 wins=int((dd > 0).sum()), rank=rank, n=len(pool) + 1,
                                 d_a2=None if a2 is None else a2.mean(),
                                 p_a2=None if a2 is None else stats.ttest_1samp(a2, 0).pvalue,
                                 d_cls=None if c is None else c.mean(),
                                 p_cls=None if c is None else stats.ttest_1samp(c, 0).pvalue, pool=pool))
        q = E.bh([x["p"] for x in rows]) if rows else []
        for x, qq in zip(rows, q):
            x["q"] = qq
        res[ep] = rows
        print(f"\n## {ep}: ACS ({ARM}) vs per-row SOTA (published pool), paired s42-46, A100")
        print(f"{'row':13s} {'ACS':>6s} {'SOTA':11s} {'SOTA_v':>6s} {'d':>6s} {'p':>6s} {'q_BH':>6s} {'wins':>5s} "
              f"{'rank':>6s} | {'ACS-A2':>7s} {'p':>6s} | {'ACS-cls*':>8s} {'p':>6s}")
        for x in rows:
            fa = f"{x['d_a2']:+7.2f} {x['p_a2']:6.3f}" if x["d_a2"] is not None else f"{'--':>7s} {'':6s}"
            fc = f"{x['d_cls']:+8.2f} {x['p_cls']:6.3f}" if x["d_cls"] is not None else f"{'--':>8s} {'':6s}"
            print(f"{x['ds'][:6]} {x['r']:<6} {x['acs']:6.2f} {x['sota']:11s} {x['sota_v']:6.2f} {x['d']:+6.2f} "
                  f"{x['p']:6.3f} {x['q']:6.3f} {x['wins']:3d}/5 {x['rank']:2d}/{x['n']:<3d} | {fa} | {fc}")
        if rows:
            print(f"  ACS first: {sum(x['rank'] == 1 for x in rows)}/{len(rows)}; ahead of SOTA: "
                  f"{sum(x['d'] > 0 for x in rows)}/{len(rows)}; sig behind (BH q<.05): "
                  f"{sum(x['d'] < 0 and x['q'] < .05 for x in rows)}; sig ahead (BH): "
                  f"{sum(x['d'] > 0 and x['q'] < .05 for x in rows)}; |d|<1pp: {sum(abs(x['d']) < 1 for x in rows)}; "
                  f"mean d {np.mean([x['d'] for x in rows]):+.2f}; mean rank {np.mean([x['rank'] for x in rows]):.2f}")
            dc = [x["d_cls"] for x in rows if x["d_cls"] is not None]
            if dc:
                print(f"  * exploratory (not preregistered) ACS - cls = F2 where F3 is off: mean {np.mean(dc):+.2f}, "
                      f"{sum(v > 0 for v in dc)}/{len(dc)} rows positive")
    json.dump(res, open("results/acs/acs_main4_analysis.json", "w"), indent=1)


if __name__ == "__main__":
    harvest() if "--cluster" in sys.argv else analyse()
