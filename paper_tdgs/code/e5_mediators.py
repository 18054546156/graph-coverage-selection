#!/usr/bin/env python
"""E5 zero-GPU mediator checks (report/e5_prereg_20261003.md §3) from the staging reports
results/acs/e5_report/<ds>_r<r>_acs_e5.json (copied from $S/acs/e5_report). Predictions are the ones stamped.
Run from paper_tdgs/:  python code/e5_mediators.py > results/acs/e5_mediators.txt
"""
import json
import os

ROWS = [("breastmnist", 0.05)] + [(d, r) for d in ("dermamnist", "octmnist", "tissuemnist") for r in (0.02, 0.05)]
RULE = {("breastmnist", 0.05): "acs_q25_t050"}
T0 = ROWS[1:]
REP = "results/acs/e5_report"

R = {}
for d, r in ROWS:
    f = f"{REP}/{d}_r{r}_acs_e5.json"
    if os.path.exists(f):
        R[(d, r)] = json.load(open(f))
print(f"reports {len(R)}/{len(ROWS)}")
tag = lambda d, r: f"{d[:6]} {r:<5}"

print("\n## I5 (must all be True)")
for k, rep in R.items():
    print(f"  {tag(*k)} " + " ".join(f"{a}:{v['match']}" for a, v in rep["checks"]["I5"].items()))

print("\n## M1  Spearman(chi_j, 1 - a_j) over the pool; predicted >= .7 in every row")
m1 = [R[k]["mediators"]["M1"]["spearman_chi_vs_1ma"] for k in R]
for k in R:
    m = R[k]["mediators"]["M1"]
    print(f"  {tag(*k)} rho {m['spearman_chi_vs_1ma']:.3f}  mean chi {m['chi_mean']:.3f}")
print(f"  -> {sum(x >= .7 for x in m1)}/{len(m1)} rows >= .7")

print("\n## M2  demand-side leakage (class-mean share of class coverage won by other-class picks)")
print("     predicted on the 6 tau=0 rows: ACS_l00 > ACS and cls_l00 > cls")
ok = 0
for k in R:
    L = {a: v["class_mean"] for a, v in R[k]["mediators"]["M2_leakage"].items()}
    acs = RULE.get(k, "acs_q25_t000")
    line = (f"  {tag(*k)} cls {L.get('cls', float('nan')):.3f} cls_l00 {L.get('acs_q00_t000_l00', float('nan')):.3f} | "
            f"ACS {L.get(acs, float('nan')):.3f} ACS_l00 {L.get(acs + '_l00', float('nan')):.3f} | "
            f"mask {L.get(acs + '_mask', float('nan')):.3f} wcls {L.get(acs + '_wcls', float('nan')):.3f} "
            f"l100 {L.get(acs + '_l100', float('nan')):.3f} | A2(author) {L.get('ref_graph_a2_author', float('nan')):.3f}")
    if k in T0:
        hit = L["acs_q25_t000_l00"] > L["acs_q25_t000"] and L["acs_q00_t000_l00"] > L["cls"]
        ok += hit
        line += "  HIT" if hit else "  miss"
    print(line)
print(f"  -> {ok}/{sum(k in R for k in T0)} tau=0 rows as predicted")

print("\n## M3  purity (UNI k=50 same-class share, class-balanced); predicted on tau=0: cls_l00 < ACS_l00, cls_l00 < cls <= ACS")
ok = 0
for k in R:
    P = R[k]["mediators"]["M3_purity_uni"]
    acs = RULE.get(k, "acs_q25_t000")
    g = lambda a: P.get(a, float("nan"))
    line = (f"  {tag(*k)} random {g('ref_random'):.3f} A2(author) {g('ref_graph_a2_author'):.3f} A2(ours) "
            f"{g('ref_a2_quota_ours'):.3f} | cls_l00 {g('acs_q00_t000_l00'):.3f} ACS_l00 {g(acs + '_l00'):.3f} "
            f"cls {g('cls'):.3f} ACS {g(acs):.3f} | mask {g(acs + '_mask'):.3f} wcls {g(acs + '_wcls'):.3f} "
            f"l100 {g(acs + '_l100'):.3f}")
    if k in T0:
        hit = g("acs_q00_t000_l00") < g("acs_q25_t000_l00") and g("acs_q00_t000_l00") < g("cls") <= g("acs_q25_t000")
        ok += hit
        line += "  HIT" if hit else "  miss"
    print(line)
print(f"  -> {ok}/{sum(k in R for k in T0)} tau=0 rows as predicted")

print("\n## T4  author Graph-A2 purity vs pool (random), derma/OCT (P1: A2 below pool)")
for k in R:
    if k[0] in ("dermamnist", "octmnist"):
        P = R[k]["mediators"]["M3_purity_uni"]
        a2, rnd = P.get("ref_graph_a2_author"), P.get("ref_random")
        if a2 is not None and rnd is not None:
            print(f"  {tag(*k)} A2(author) {a2:.3f} random {rnd:.3f}  diff {a2 - rnd:+.3f}  {'below pool' if a2 < rnd else 'NOT below'}"
                  f"  [{R[k]['refs'].get('graph_a2_author', '?').split('/')[-4] if 'graph_a2_author' in R[k]['refs'] else '?'}]")

print("\n## M4  F2 keep-set (q=.25) Jaccard over k in {20, 50, 100}; predicted kept >= .8 (random-75% baseline .6)")
for k in R:
    m = R[k]["mediators"]["M4"]
    print(f"  {tag(*k)} " + "  ".join(f"k{p}: kept {v['kept']:.3f} removed {v['removed']:.3f}"
                                      for p, v in m.items() if isinstance(v, dict))
          + f"  -> {'PASS' if m['pass_kept_ge_0p8'] else 'FAIL'}")

print("\n## side: acs_q00_t000_l00 vs our earlier class-quota Graph-A2 (Jaccard)")
for k in R:
    print(f"  {tag(*k)} {R[k]['mediators'].get('l00_vs_our_a2_quota_jaccard', 'n/a')}  ref {R[k]['refs'].get('a2_quota_ours', 'none')}")
