#!/usr/bin/env python
"""ACS E2 dev screen (report/acs_e2_prereg_20261002.md, stamped 2026-10-02 13:58:53 UTC).

  python acs_eval.py --cluster   -> (on luhpc, in $S/code) harvest runs/acs_20261002 -> acs_cells.json
  python code/acs_eval.py test   -> rule selection (§3) + descriptives (§4) from results/acs/acs_cells.json

Reused cells (same hardware within a row, prereg §2) come from knnf_eval.load():
  retina/breast H100 (ambig), derma/OCT H100 (w1 + knnf), tissue A100 (Table-1 / round1-2 / knnf).
Exception (prereg §2): tissue 2% cls is the NEW A100 tdgs_cls cell in the acs tree, not round1/r3hw.
Main 4 datasets (new baselines only): A100 Table-1 bench as d2_paired_table.py; mv_mean as there.
"""
import collections
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TREE = "/mnt/prj01/hgrp-1502-5TB/tdgs_shared/runs/acs_20261002"
LOGS = "/mnt/prj01/hgrp-1502-5TB/tdgs_shared/acs/logs"
HIGH = ["retinamnist", "breastmnist", "dermamnist", "octmnist", "tissuemnist"]
MAIN = ["bloodmnist", "organamnist", "organsmnist", "pathmnist"]
RATIOS = (0.02, 0.05)
S5 = [42, 43, 44, 45, 46]
BPC = {("retinamnist", .02): 4, ("retinamnist", .05): 10, ("breastmnist", .02): 5, ("breastmnist", .05): 13,
       ("dermamnist", .02): 20, ("dermamnist", .05): 50, ("octmnist", .02): 487, ("octmnist", .05): 1218,
       ("tissuemnist", .02): 413, ("tissuemnist", .05): 1034}
GPU = {d: "H100" for d in HIGH[:4]} | {d: "A100" for d in ["tissuemnist"] + MAIN}
QS, TAUS = (0, .25, .5), (0, .5, 1)
NEWB = ["typiclust", "probcover", "maxherding"]
T1B = ["graph_a2", "herding", "facility", "random", "fps", "eva", "el2n_top", "forgetting"]


def arm(q, t):
    if q == 0 and t == 0:
        return "cls"
    if q == 0 and t == 1:
        return "herding"
    return f"acs_q{int(round(q * 100)):02d}_t{int(round(t * 100)):03d}"


def tau(b, bstar):
    if b <= bstar:
        return 1
    return .5 if b < 2 * bstar else 0


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
    for lf in glob.glob(f"{LOGS}/acse2_*.out"):
        txt = open(lf).read()
        g = "H100" if "H100" in txt else ("A100" if "A100" in txt else "?")
        for key in re.findall(r"ok (\w+_r[0-9.]+_\w+_s\d+)", txt):
            gpu[key] = g
    for x in out:
        x["gpu"] = gpu.get(f"{x['ds']}_r{x['ratio']}_{x['arm']}_s{x['seed']}", "?")
    json.dump(out, open("acs_cells.json", "w"), indent=0)
    print(f"wrote acs_cells.json ({len(out)} cells; gpu {collections.Counter(x['gpu'] for x in out)})")


def load():
    import knnf_eval
    C, _ = knnf_eval.load()
    C = {k: v for k, v in C.items() if not (k[0] == "tissuemnist" and k[1] == .02 and k[2] == "cls")}
    # main-dataset Table-1 A100 bench (same sources/order as d2_paired_table.py)
    for f in ("results/table1/t1_s4246_harvest.json", "results/table1/t1_a100_rerun_harvest.json",
              "results/round1/tdgs_round1_harvest.json", "results/round2/round2_harvest_final_2b.json"):
        for e in json.load(open(f)):
            if "round" in f and e["arm"] != "graph_a2":
                continue
            if e["ds"] in MAIN + ["tissuemnist"]:
                C[(e["ds"], round(float(e["ratio"]), 4), e["arm"], int(e["seed"]))] = dict(ba=100 * e["ba"],
                                                                                          worst=100 * e["worst"])
    PREF = ['plus_a100_20260930', 'mvf_20260930', 'tune_20260929', 'tune2_4090_20260929', 'plus_scr_20260930']
    lines = [l.rstrip("\n").split("\t") for l in open("results/plus/ba_1500.tsv")]
    for t in PREF:
        for tr, r, ds, a, s, ba, w in lines:
            k = (ds, round(float(r.split("_")[1]), 4), "mv_mean", int(s.split("_")[1]))
            if tr == t and a == "mv_mean" and ds in MAIN and k not in C:
                C[k] = dict(ba=100 * float(ba), worst=100 * float(w))
    new = json.load(open("results/acs/acs_cells.json"))
    bad = [x for x in new if x["gpu"] != GPU[x["ds"]]]
    for x in new:
        if x in bad:
            continue
        a = "cls" if x["arm"] == "tdgs_cls" else x["arm"]
        C[(x["ds"], round(float(x["ratio"]), 4), a, int(x["seed"]))] = dict(ba=x["ba"], worst=x["worst"])
    # bench fill (report/bench_fill_20261003.md, code/bench_fill_eval.py): AUTHOR-pipeline cells, H100, s42-46.
    # derma/OCT: knnf_eval mapped OUR rand_cls to "random"; keep it as "rand_cls" and let the author random /
    # graph_a2 take the published-method names (a2_uni stays under its own name). E2 rule selection uses ACS arms only.
    bf = "results/acs/bench_fill_cells.json"
    if os.path.exists(bf):
        for k in [k for k in C if k[0] in ("dermamnist", "octmnist") and k[2] == "random"]:
            C[(k[0], k[1], "rand_cls", k[3])] = C.pop(k)
        for x in json.load(open(bf)):
            if x["gpu"] == GPU[x["ds"]]:
                C[(x["ds"], round(float(x["ratio"]), 4), x["arm"], int(x["seed"]))] = dict(ba=x["ba"], worst=x["worst"])
    return C, new, bad


def bh(ps):
    ps = np.asarray(ps, float); n = len(ps); o = np.argsort(ps); q = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, ps[o[i]] * n / (i + 1)); q[o[i]] = prev
    return q


def test():
    C, new, bad = load()
    print(f"new cells {len(new)} (expected 625); excluded for wrong GPU: {len(bad)} {[(x['ds'], x['arm'], x['seed'], x['gpu']) for x in bad][:10]}")
    rows = [(d, r) for d in HIGH for r in RATIOS]
    have = lambda d, r, a: all((d, r, a, s) in C for s in S5)
    V = lambda d, r, a, ep: np.array([C[(d, r, a, s)][ep] for s in S5])
    miss = [(d, r, arm(q, t)) for d, r in rows for q in QS for t in TAUS if not have(d, r, arm(q, t))]
    print(f"missing factorial cells (row, arm): {len(miss)} {miss[:12]}")

    # ---- §4.1 factorial table
    for ep in ("ba", "worst"):
        print(f"\n## factorial, seed-mean {ep} (s42-46)")
        print("row            bpc " + " ".join(f"q{int(q*100):02d}t{int(t*100):03d}" for q in QS for t in TAUS))
        for d, r in rows:
            cells = [f"{V(d, r, arm(q, t), ep).mean():8.2f}" if have(d, r, arm(q, t)) else "      --" for q in QS for t in TAUS]
            print(f"{d[:6]} {r:<5} {BPC[(d, r)]:5d} " + " ".join(cells))

    if miss:
        print("\nfactorial incomplete -> rule selection not run")
        return
    # ---- §3 rule selection
    print("\n## rule selection (mean over 10 rows of seed-mean BA)")
    res = []
    for bstar in (0, 5, 10, 20):
        for qs in QS:
            arms = {(d, r): arm(qs, tau(BPC[(d, r)], bstar)) for d, r in rows}
            ba = np.mean([V(d, r, arms[(d, r)], "ba").mean() for d, r in rows])
            wr = np.mean([V(d, r, arms[(d, r)], "worst").mean() for d, r in rows])
            res.append(dict(bstar=bstar, q=qs, ba=ba, worst=wr, arms=arms))
    top = max(x["ba"] for x in res)
    for x in sorted(res, key=lambda x: -x["ba"]):
        print(f"  b*={x['bstar']:2d} q*={x['q']:.2f}  BA {x['ba']:.3f} (Δ to top {x['ba'] - top:+.3f})  worst {x['worst']:.2f}")
    tied = [x for x in res if top - x["ba"] < .25]
    win = sorted(tied, key=lambda x: (-x["worst"], x["q"], x["bstar"]))[0]
    print(f"tied (<0.25pp): {[(x['bstar'], x['q']) for x in tied]}  -> SELECTED b*={win['bstar']} q*={win['q']}"
          f"{'  == cls: ACS does NOT go to E4' if (win['bstar'], win['q']) == (0, 0) else ''}")
    json.dump(dict(bstar=win["bstar"], q=win["q"], arms={f"{d}_{r}": a for (d, r), a in win["arms"].items()}),
              open("results/acs/e2_selected_rule.json", "w"), indent=1)

    # ---- §4.3 oracle vs rule, §4.6 vs all bench
    for ep in ("ba", "worst"):
        print(f"\n## selected rule vs row-best bench ({ep}); paired over s42-46")
        ps, lines, first = [], [], 0
        for d, r in rows:
            a = win["arms"][(d, r)]
            oracle = max((arm(q, t) for q in QS for t in TAUS), key=lambda x: V(d, r, x, ep).mean())
            bench = [b for b in T1B + ["mv_mean", "knnf_herding", "knnf_random", "a2_uni"] + NEWB
                     if b != a and have(d, r, b)]
            if a != "cls":
                bench.append("cls")
            if a != "herding" and "herding" not in bench and have(d, r, "herding"):
                bench.append("herding")
            xb = max(bench, key=lambda b: V(d, r, b, ep).mean())
            dd = V(d, r, a, ep) - V(d, r, xb, ep)
            p = stats.ttest_1samp(dd, 0).pvalue if dd.std() > 0 else 1.0
            ps.append(p if dd.mean() < 0 else 1.0)
            first += dd.mean() >= 0
            lines.append(f"  {d[:6]} {r}: rule {a:13s} {V(d, r, a, ep).mean():6.2f} | best bench {xb:13s} "
                         f"{V(d, r, xb, ep).mean():6.2f} | d {dd.mean():+5.2f} p {p:.3f} | oracle {oracle} "
                         f"{V(d, r, oracle, ep).mean() - V(d, r, a, ep).mean():+.2f} | n_bench {len(bench)}")
        q = bh(ps)
        for l, p, qq in zip(lines, ps, q):
            print(l + f"  {'BEHIND(raw)' if p < .05 else ''}{' BEHIND(BH)' if qq < .05 else ''}")
        print(f"  first {first}/10; significantly behind raw {sum(p < .05 for p in ps)}, BH {sum(x < .05 for x in q)}")

    # ---- §4.4 new baselines on all 18 rows, BA
    print("\n## new baselines vs row best (BA, seed mean)")
    for d in HIGH + MAIN:
        for r in RATIOS:
            pool = [b for b in T1B + ["mv_mean", "cls", "knnf_herding", "knnf_random", "a2_uni"] + NEWB if have(d, r, b)]
            if not pool:
                continue
            best = max(pool, key=lambda b: V(d, r, b, "ba").mean())
            s = " ".join(f"{b}:{V(d, r, b, 'ba').mean() - V(d, r, best, 'ba').mean():+.2f}" for b in NEWB if have(d, r, b))
            print(f"  {d[:6]} {r}: best {best:12s} {V(d, r, best, 'ba').mean():6.2f} | {s}")


if __name__ == "__main__":
    {"--cluster": harvest, "test": test}[sys.argv[1]]()
