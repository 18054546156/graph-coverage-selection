#!/usr/bin/env python
"""Amendment 1 of report/ambig_law_prereg_20261002.md: kNN-filter baselines, K1 / K2.

  python knnf_eval.py --cluster  -> (on luhpc, in $S/code) harvest runs/knnf_20261002 -> knnf_cells.json
  python code/knnf_eval.py test  -> K1/K2 (locally, from paper_tdgs/)

Comparison cells (same hardware within a dataset, prereg amendment 1):
  retina/breast  cls_uni, mv_mean, herding, ...   results/ambig/ambig_cells.json   (H100, s42-51)
  derma/OCT      cls_uni, mv_mean, a2_uni, rand_cls results/w1/w1_cells.json        (H100, s42-46)
  tissue         tdgs_cls (round1 + round2 harvests, as make_table1.py), mv_mean (plus/ba_1500.tsv, plus_a100),
                 herding (Table-1 A100 harvests)                                     (A100, s42-46)
  knnf arms      results/ambig/knnf_cells.json (H100; tissue A100)
"""
import glob
import json
import re
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

TREE = "/mnt/prj01/hgrp-1502-5TB/tdgs_shared/runs/knnf_20261002"
HIGH = ["retinamnist", "breastmnist", "dermamnist", "octmnist", "tissuemnist"]
SEEDS = {"retinamnist": range(42, 52), "breastmnist": range(42, 52)}
RATIOS = (0.02, 0.05)


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
    for lf in glob.glob("/mnt/prj01/hgrp-1502-5TB/tdgs_shared/ambig/logs/knnftrain_*.out"):
        txt = open(lf).read()
        g = "H100" if "H100" in txt else ("A100" if "A100" in txt else "?")
        for key in re.findall(r"ok (\w+_r[0-9.]+_\w+_s\d+)", txt):
            gpu[key] = g
    for x in out:
        x["gpu"] = gpu.get(f"{x['ds']}_r{x['ratio']}_{x['arm']}_s{x['seed']}", "?")
    json.dump(out, open("knnf_cells.json", "w"), indent=0)
    print(f"wrote knnf_cells.json ({len(out)} cells)")


def load():
    C = {}

    def put(ds, r, arm, s, ba, w):
        arm = "random" if arm.startswith("random") or arm == "rand_cls" else arm
        arm = "knnf_random" if arm.startswith("knnf_random") else arm
        arm = "cls" if arm in ("cls_uni", "tdgs_cls") else arm
        C[(ds, round(float(r), 4), arm, int(s))] = dict(ba=ba, worst=w)

    for x in json.load(open("results/ambig/ambig_cells.json")):
        put(x["ds"], x["ratio"], x["arm"], x["seed"], x["ba"], x["worst"])
    for x in json.load(open("results/w1/w1_cells.json")):
        if x["ds"] in ("dermamnist", "octmnist"):
            put(x["ds"], x["ratio"], x["arm"], x["seed"], x["ba"], x["worst"])
    for f in ("results/round1/tdgs_round1_harvest.json", "results/round2/round2_harvest_final_2b.json"):
        for x in json.load(open(f)):
            if x["ds"] == "tissuemnist" and x["arm"] == "tdgs_cls" and 42 <= x["seed"] <= 46:
                put(x["ds"], x["ratio"], x["arm"], x["seed"], 100 * x["ba"], 100 * x["worst"])
    for f in ("results/table1/t1_s4246_harvest.json", "results/table1/t1_a100_rerun_harvest.json"):
        for x in json.load(open(f)):
            if x["ds"] == "tissuemnist" and x["arm"] in ("herding", "facility", "random"):
                put(x["ds"], x["ratio"], x["arm"], x["seed"], 100 * x["ba"], 100 * x["worst"])
    # tissue mv_mean: same campaign preference as first_place_math.py / d2_paired_table.py (descriptive only;
    # 5% is not in plus_a100, so it comes from the next campaign in the list -- mixed hardware, K2 only)
    PREF = ['plus_a100_20260930', 'mvf_20260930', 'tune_20260929', 'tune2_4090_20260929', 'plus_scr_20260930']
    lines = [l.rstrip("\n").split("\t") for l in open("results/plus/ba_1500.tsv")]
    for t in PREF:
        for tr, r, ds, arm, s, ba, w in lines:
            key = ("tissuemnist", round(float(r.split("_")[1]), 4), "mv_mean", int(s.split("_")[1]))
            if tr == t and ds == "tissuemnist" and arm == "mv_mean" and key not in C:
                put(ds, r.split("_")[1], arm, s.split("_")[1], 100 * float(ba), 100 * float(w))
    kn = json.load(open("results/ambig/knnf_cells.json"))
    for x in kn:
        put(x["ds"], x["ratio"], x["arm"], x["seed"], x["ba"], x["worst"])
    return C, kn


def test():
    C, kn = load()
    print(f"knnf cells: {len(kn)} (expected 180); gpu by ds: "
          f"{ {d: sorted({x['gpu'] for x in kn if x['ds'] == d}) for d in HIGH} }")
    for ep in ("ba", "worst"):
        print(f"\n######## endpoint {ep} ########")
        k1 = 0
        for ds in HIGH:
            S = list(SEEDS.get(ds, range(42, 47)))
            get = lambda a: np.array([C[(ds, r, a, s)][ep] for r in RATIOS for s in S])
            kh, kr = get("knnf_herding"), get("knnf_random")
            best = "knnf_herding" if kh.mean() >= kr.mean() else "knnf_random"
            F = get("cls") - get(best)
            p = stats.ttest_1samp(F, 0, alternative="greater").pvalue
            ok = F.mean() > 0 and p < .05
            k1 += ok
            line = (f"{ds[:6]} F=cls-{best:12s} {F.mean():+6.2f} (n={len(F)}) one-sided p {p:.3f} {'PASS' if ok else 'fail'}"
                    f" | mv-best_knnf {(get('mv_mean') - get(best)).mean():+6.2f}"
                    f" | knnf_herding-herding {(kh - get('herding')).mean():+6.2f}"
                    f" | knnf_random-random {(kr - get('random')).mean():+6.2f}")
            print(line)
            for r in RATIOS:
                arms = [a for a in ("cls", "mv_mean", "herding", "facility", "random", "graph_a2", "a2_uni",
                                    "knnf_herding", "knnf_random") if all((ds, r, a, s) in C for s in S)]
                V = {a: np.mean([C[(ds, r, a, s)][ep] for s in S]) for a in arms}
                print(f"      {r}: " + " > ".join(f"{a}:{V[a]:.1f}" for a in sorted(V, key=lambda a: -V[a])))
        if ep == "ba":
            print(f"K1: {k1}/5 datasets with cls > best knnf (one-sided p<.05); need >=3 -> {'PASS' if k1 >= 3 else 'FAIL'}")


if __name__ == "__main__":
    {"--cluster": harvest, "test": test}[sys.argv[1]]()
