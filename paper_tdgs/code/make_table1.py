#!/usr/bin/env python
"""Assemble Table 1 on the author seed grid (42-46) and set it against the paper.

Columns: the 7 baselines from the t1 re-run (t1_harvest.py), Graph-A2 and TDGS
(tdgs_cls) from rounds 1+2. Every column is 5 training seeds 42-46, i.e. the
author's own seeds (graphcov --seed 42, seed = seed + trial, --trials 5), and
selection_seed == training_seed as in the author's code.

Printed per (dataset, ratio): ours mean+-sd, paper mean+-sd, ours-paper; then
the rank of every method in our table vs in the paper's, and the paired
TDGS - Graph-A2 delta (paired by seed).

sd is the sample sd (ddof=1). The paper does not say which it uses; with n=5
the two differ by a factor 1.118, which is below the rounding in its table.
A cell with fewer than 5 seeds is printed with its n and never silently padded.
"""
import argparse
import json
from collections import defaultdict

import numpy as np

DS = [("organsmnist", "OrganS"), ("organamnist", "OrganA"), ("pathmnist", "Path"),
      ("tissuemnist", "Tissue"), ("bloodmnist", "Blood")]
COLS = [("random", "Random"), ("el2n_top", "EL2N"), ("forgetting", "Forg."),
        ("eva", "EVA"), ("facility", "Facility"), ("fps", "FPS"),
        ("herding", "Herding"), ("graph_a2", "Graph-A2"), ("tdgs_cls", "TDGS")]
SEEDS = (42, 43, 44, 45, 46)

# arXiv:2606.22002 Table 1, (mean, sd); the last column is the paper's "Ours" = Graph-A2
PAPER = {
    ("organsmnist", 0.02): [(57.2, 4.7), (39.5, 2.0), (42.5, 1.4), (52.3, 1.3), (63.3, 1.2), (59.6, 2.7), (63.2, 0.6), (63.7, 0.7)],
    ("organsmnist", 0.05): [(66.0, 1.2), (51.4, 0.9), (56.6, 1.1), (66.0, 0.9), (67.7, 1.1), (66.0, 0.8), (68.1, 0.7), (68.4, 1.0)],
    ("organamnist", 0.02): [(83.6, 2.1), (63.1, 0.6), (72.9, 1.8), (68.6, 2.2), (84.9, 1.1), (83.3, 1.5), (86.3, 0.7), (86.5, 0.9)],
    ("organamnist", 0.05): [(89.8, 1.2), (80.9, 0.8), (84.6, 0.2), (81.0, 2.0), (90.5, 0.9), (90.7, 0.6), (90.4, 0.7), (91.9, 0.3)],
    ("pathmnist", 0.02): [(77.5, 4.6), (36.8, 1.0), (53.5, 3.9), (53.5, 5.7), (77.0, 1.6), (73.0, 3.8), (78.0, 2.1), (80.9, 0.9)],
    ("pathmnist", 0.05): [(84.0, 2.4), (49.9, 2.4), (58.9, 4.2), (58.7, 3.5), (82.5, 2.2), (83.0, 1.6), (84.6, 2.7), (85.9, 1.7)],
    ("tissuemnist", 0.02): [(43.1, 1.0), (10.1, 0.4), (39.3, 0.6), (39.8, 0.6), (44.7, 1.0), (35.6, 0.9), (42.9, 0.9), (42.6, 0.6)],
    ("tissuemnist", 0.05): [(49.1, 0.4), (15.2, 0.2), (43.6, 1.0), (46.7, 1.0), (48.1, 0.6), (41.0, 1.1), (48.3, 0.4), (49.5, 1.3)],
    ("bloodmnist", 0.02): [(83.2, 1.5), (48.3, 6.9), (67.5, 3.3), (75.9, 2.5), (82.7, 1.6), (78.8, 2.1), (83.1, 3.0), (84.5, 1.7)],
    ("bloodmnist", 0.05): [(90.3, 0.5), (72.5, 1.7), (81.4, 2.1), (86.6, 1.7), (88.2, 2.7), (89.0, 2.5), (92.3, 0.6), (93.4, 0.7)],
}


def load(paths, arms=None):
    cells = defaultdict(dict)
    for p in paths:
        for r in json.load(open(p)):
            if arms and r["arm"] not in arms:
                continue
            if r["seed"] in SEEDS:
                cells[(r["ds"], round(r["ratio"], 4), r["arm"])][r["seed"]] = 100 * r["ba"]
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", required=True, help="t1_harvest.py output")
    ap.add_argument("--round2", required=True)
    ap.add_argument("--round1", required=True)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    cells = load([a.baselines], arms={c for c, _ in COLS[:7]})
    for k, v in load([a.round1, a.round2], arms={"graph_a2", "tdgs_cls"}).items():
        cells[k].update(v)

    out, absdiff, missing = [], [], 0
    for ratio in (0.02, 0.05):
        print(f"\n===== ratio {ratio:.0%}  (ours = seeds 42-46, BA %) =====")
        print(f"{'dataset':8s} " + " ".join(f"{n:>13s}" for _, n in COLS))
        for ds, dn in DS:
            ours, line_o, line_p, line_d = [], [], [], []
            for j, (arm, _) in enumerate(COLS):
                v = cells.get((ds, ratio, arm), {})
                xs = np.array([v[s] for s in SEEDS if s in v])
                m = xs.mean() if len(xs) else np.nan
                sd = xs.std(ddof=1) if len(xs) > 1 else np.nan
                ours.append(m)
                tag = "" if len(xs) == 5 else f"[n{len(xs)}]"
                missing += 5 - len(xs)
                line_o.append(f"{m:5.1f}+-{sd:3.1f}{tag}" if len(xs) else "--")
                if j < 8:
                    pm, psd = PAPER[(ds, ratio)][j]
                    line_p.append(f"{pm:5.1f}+-{psd:3.1f}")
                    line_d.append(f"{m - pm:+5.2f}" if len(xs) else "--")
                    if len(xs):
                        absdiff.append((abs(m - pm), ds, ratio, arm))
                else:
                    line_p.append("(new)")
                    line_d.append("")
                out.append(dict(ds=ds, ratio=ratio, arm=arm, n=int(len(xs)),
                                mean=None if np.isnan(m) else round(float(m), 3),
                                sd=None if np.isnan(sd) else round(float(sd), 3),
                                paper=PAPER[(ds, ratio)][j] if j < 8 else None,
                                seeds={int(s): round(v[s], 3) for s in SEEDS if s in v}))
            print(f"{dn:8s} " + " ".join(f"{x:>13s}" for x in line_o))
            print(f"{'  paper':8s} " + " ".join(f"{x:>13s}" for x in line_p))
            print(f"{'  diff':8s} " + " ".join(f"{x:>13s}" for x in line_d))
            o = np.array(ours[:8])
            if not np.isnan(o).any():
                p = np.array([x[0] for x in PAPER[(ds, ratio)]])
                ro = (-o).argsort().argsort() + 1
                rp = (-p).argsort().argsort() + 1
                print(f"{'  rank':8s} ours " + " ".join(map(str, ro)) +
                      "   paper " + " ".join(map(str, rp)))
            g = cells.get((ds, ratio, "graph_a2"), {})
            t = cells.get((ds, ratio, "tdgs_cls"), {})
            pair = [t[s] - g[s] for s in SEEDS if s in g and s in t]
            if pair:
                print(f"{'  TDGS-A2':8s} paired {np.mean(pair):+.2f} (n={len(pair)}, "
                      f"{sum(x > 0 for x in pair)}/{len(pair)} seeds > 0)")

    if absdiff:
        d = np.array([x[0] for x in absdiff])
        print(f"\nours vs paper over {len(d)} cells: mean |diff| {d.mean():.2f}pp, "
              f"median {np.median(d):.2f}, within 2pp {int((d <= 2).sum())}, "
              f"within 4pp {int((d <= 4).sum())}")
        for x in sorted(absdiff, reverse=True)[:6]:
            print(f"  largest: {x[1]} r{x[2]} {x[3]}  |diff| {x[0]:.2f}")
    print(f"missing seed-cells: {missing}")
    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
