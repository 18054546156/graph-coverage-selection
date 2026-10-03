#!/usr/bin/env python
"""Bench fill (report/bench_fill_20261003.md): author-pipeline selections -> td_sel files + H100 worklist.

  runs/bench_fill_20261003/sel/seed_<s>/<m>/<ds>/ratio_<r>/selection_seed_<s>/selected_indices.npy
    -> benchfill/td_sel/<ds>_r<r>_<m><s>_s42.npy   trained at seed s only (selection seed == training seed)
    -> benchfill/td_sel/<ds>_r<r>_graph_a2_s42.npy (seed 42 only; seed-independent), trained at 42-46
Indices copied verbatim (int64, author order). Written to a temp name then renamed (tune_pack never reads a partial file).
Never overwrites an existing td_sel file with different content (abort). Idempotent: re-run after late staging.
The worklist is written once (all 200 cells); cells whose selection is not linked yet are skipped by tune_pack.
"""
import os
import sys

import numpy as np

S = "/mnt/prj01/hgrp-1502-5TB/tdgs_shared"
SEL = f"{S}/runs/bench_fill_20261003/sel"
TD = f"{S}/benchfill/td_sel"
WL = f"{S}/work/bench_fill_h100.txt"
SEEDS = range(42, 47)
RATIOS = ("0.02", "0.05")
PLAN = {d: ["fps", "eva", "el2n_top", "forgetting"] for d in ("retinamnist", "breastmnist")}
PLAN.update({d: ["fps", "eva", "el2n_top", "forgetting", "random", "graph_a2"] for d in ("dermamnist", "octmnist")})
# cheap datasets last would leave OCT for the tail; put OCT first so the long cells start early
ORDER = ["octmnist", "dermamnist", "breastmnist", "retinamnist"]


def cells():
    for s in SEEDS:
        for d in ORDER:
            for r in RATIOS:
                for m in PLAN[d]:
                    arm = m if m == "graph_a2" else f"{m}{s}"
                    src_seed = 42 if m == "graph_a2" else s
                    yield d, r, m, arm, s, src_seed


def main():
    os.makedirs(TD, exist_ok=True)
    linked = missing = 0
    for d, r, m, arm, s, ss in cells():
        dst = f"{TD}/{d}_r{r}_{arm}_s42.npy"
        src = f"{SEL}/seed_{ss}/{m}/{d}/ratio_{r}/selection_seed_{ss}/selected_indices.npy"
        if not os.path.exists(src):
            missing += 1
            continue
        a = np.load(src).astype(np.int64)
        assert a.ndim == 1 and len(np.unique(a)) == len(a), (src, a.shape)
        if os.path.exists(dst):
            if not np.array_equal(np.load(dst), a):
                sys.exit(f"ABORT: {dst} exists with different content")
            continue
        tmp = f"{TD}/.tmp_{os.getpid()}_{os.path.basename(dst)}"
        np.save(tmp, a)
        os.rename(tmp, dst)
        linked += 1
    if not os.path.exists(WL):
        with open(WL, "w") as f:
            for d, r, m, arm, s, ss in cells():
                f.write(f"{d} {r} {arm} {s}\n")
        os.chmod(WL, 0o664)
        print("worklist written", WL)
    n_cells = sum(1 for _ in cells())
    have = sum(os.path.exists(f"{TD}/{d}_r{r}_{arm}_s42.npy") for d, r, m, arm, s, ss in cells())
    print(f"linked now {linked}; cells with selection {have}/{n_cells}; missing sources {missing}")


if __name__ == "__main__":
    main()
