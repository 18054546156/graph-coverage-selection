#!/usr/bin/env python
"""Full results table: every PUBLISHED method + ACS, every dataset x ratio, seed-mean over s42-46 (paired cells only).
BA and worst-class recall, markdown. Bold = row best. "—" = not run for that row (fewer than 5 seeds).
Run from paper_tdgs/:  python code/acs_full_table.py > results/acs/full_table.md
"""
import json
import sys

import numpy as np

sys.path.insert(0, "code")
import acs_eval as E  # noqa: E402

C, _, _ = E.load()
for x in json.load(open("results/acs/acs_main4_cells.json")):  # main-4 run (acs_main4_prereg_20261003.md), A100
    if x["gpu"] == "A100":
        C[(x["ds"], round(x["ratio"], 4), x["arm"], x["seed"])] = dict(ba=x["ba"], worst=x["worst"])
have = lambda d, r, a: all((d, r, a, s) in C for s in E.S5)
V = lambda d, r, a, ep: np.array([C[(d, r, a, s)][ep] for s in E.S5])
sel = json.load(open("results/acs/e2_selected_rule.json"))["arms"]
MAIN = ["bloodmnist", "organamnist", "organsmnist", "pathmnist"]
DS = MAIN + ["tissuemnist", "retinamnist", "breastmnist", "dermamnist", "octmnist"]
PUB = E.T1B + E.NEWB
NAME = dict(graph_a2="Graph-A2", herding="Herding", facility="Facility", random="Random", fps="FPS", eva="EVA",
            el2n_top="EL2N", forgetting="Forgetting", typiclust="TypiClust", probcover="ProbCover",
            maxherding="MaxHerding", acs="**ACS (ours)**")


def arm(d, r, m):
    if m == "acs":
        return sel.get(f"{d}_{r}", "acs_q25_t000")
    if m == "graph_a2" and not have(d, r, "graph_a2"):
        return "a2_uni"  # fallback only: our re-implementation; the bench fill supplies author graph_a2 for derma/OCT
    return m


for ep, title in (("ba", "Balanced accuracy (%)"), ("worst", "Worst-class recall (%)")):
    print(f"\n### {title}, mean over seeds 42-46\n")
    cols = PUB + ["acs"]
    print("| dataset | ratio | " + " | ".join(NAME[c] for c in cols) + " | ACS rank |")
    print("|---|---|" + "---|" * (len(cols) + 1))
    for d in DS:
        for r in E.RATIOS:
            m = {c: V(d, r, arm(d, r, c), ep).mean() for c in cols if have(d, r, arm(d, r, c))}
            best = max(m.values())
            cell = lambda c: "—" if c not in m else (f"**{m[c]:.2f}**" if m[c] == best else f"{m[c]:.2f}")
            rank = (f"{1 + sum(v > m['acs'] for k, v in m.items() if k != 'acs')}/{len(m)}" if "acs" in m
                    else "pending")
            print(f"| {d.replace('mnist', '')} | {int(r * 100)}% | " + " | ".join(cell(c) for c in cols) + f" | {rank} |")
