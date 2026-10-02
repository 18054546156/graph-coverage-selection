#!/usr/bin/env python
"""Summarise E0 (acs_select.py reports) and E1 (al_baselines_select.py reports). Descriptive only.
  python code/acs_e0_summary.py [--dir results/acs] > results/acs/e0_summary.txt"""
import argparse
import glob
import json
import os

ORDER = ["retinamnist", "breastmnist", "dermamnist", "octmnist", "tissuemnist",
         "bloodmnist", "organamnist", "organsmnist", "pathmnist"]
SHOW = ["cls", "herding", "knnf_herding", "knnf_random42", "graph_a2", "random", "facility", "mv_mean",
        "acs_q00_t050", "acs_q25_t000", "acs_q25_t050", "acs_q25_t100",
        "acs_q50_t000", "acs_q50_t050", "acs_q50_t100"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/acs")
    a = ap.parse_args()
    reps = {}
    for f in glob.glob(os.path.join(a.dir, "report", "*_acs.json")):
        R = json.load(open(f))
        reps[(R["dataset"], R["ratio"])] = R
    keys = sorted(reps, key=lambda k: (ORDER.index(k[0]), k[1]))

    print("== identity checks (I1 cls, I2 herding, I3 knnf agreement) ==")
    for k in keys:
        c = reps[k]["checks"]
        print(f"{k[0]:12s} {k[1]:.2f} bpc={reps[k]['bpc']:5d}  I1 order={c['I1_order_match_greedy_multi']} "
              f"set={c['I1_set_match_staged_cls']}  I2={c['I2_herding_order_match']}  "
              f"I3={c['I3_plurality_kept_per_class']['match']}  uni-vs-cache maxabs={c['uni_view_vs_author_cache_maxabs']:.2e}")

    print("\n== F2 retention: min class keep fraction / erased classes; plurality filter (knnf) for contrast ==")
    for k in keys:
        R = reps[k]
        if k[1] != min(r for d, r in keys if d == k[0]):
            continue
        kp = R["checks"]["I3_plurality_kept_per_class"]["ours"]
        tot = R["retention"]["0.0"]["kept_per_class"]
        knnf_frac = [x / t for x, t in zip(kp, tot)]
        s = "  ".join(f"q={q}: min {min(v['frac']):.2f} erased {v['erased']}" for q, v in R["retention"].items())
        print(f"{k[0]:12s} {s}  | knnf: min {min(knnf_frac):.2f} erased {sum(x == 0 for x in kp)}")

    print("\n== purity (class-balanced UNI/DINOv2/CLIP k=50 same-label share), typicality pct, J to cls / herding ==")
    for k in keys:
        R = reps[k]
        print(f"-- {k[0]} {k[1]} bpc={R['bpc']}")
        J = R["jaccard"]

        def jac(p, q):
            return J.get(f"{p}|{q}", J.get(f"{q}|{p}", 1.0 if p == q else float("nan")))
        for name in SHOW:
            d = R["arms"].get(name)
            if not d or "purity" not in d:
                continue
            p = d["purity"]
            print(f"   {name:14s} pur u/d/c {p.get('uni', float('nan')):.3f}/{p.get('dinov2', float('nan')):.3f}/"
                  f"{p.get('clip', float('nan')):.3f}  minclass {d['min_class_agree_uni']:.3f}  typ% {d['typ_pct']:.3f}"
                  f"  cos {d['cos_mean']:.3f}  J(cls) {jac(name, 'cls'):.3f}  J(herd) {jac(name, 'herding'):.3f}")

    al = glob.glob(os.path.join(a.dir, "al_report", "*.json"))
    if al:
        print("\n== E1 AL baselines ==")
        for f in sorted(al):
            R = json.load(open(f))
            for r, X in R["ratios"].items():
                extra = {k: X[k] for k in ("delta", "unit_check_vs_official", "edges_nolab", "nolab_skipped") if k in X}
                print(f"{R['dataset']:12s} {r} {R['method']:10s} bpc={X['bpc']} sec={X['seconds']} {extra}")


if __name__ == "__main__":
    main()
