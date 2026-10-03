#!/usr/bin/env python
"""Bench fill (report/bench_fill_20261003.md, stamped 06:46:54Z): harvest the 200 author-pipeline H100 cells.

  python bench_fill_eval.py --cluster   -> (on luhpc, in $S/code) -> bench_fill_cells.json
Arms: graph_a2 (seed-independent selection, trained at 42-46) and <method><seed> (selection seed == training seed);
method = arm with the trailing seed stripped. Each cell writes metrics.jsonl twice (results/ and results/<ds>/precomputed/...);
the first per (ds, ratio, method, seed) is kept, as acs_main4_eval.py. GPU type from the bfill_*.out logs.
Merged into acs_eval.load(): derma/OCT graph_a2 / random become the AUTHOR cells (replacing a2_uni / rand_cls).
"""
import glob
import json
import re

TREE = "/mnt/prj01/hgrp-1502-5TB/tdgs_shared/runs/bench_fill_20261003"
LOGS = "/mnt/prj01/hgrp-1502-5TB/tdgs_shared/benchfill/logs"
METHODS = ("graph_a2", "fps", "eva", "el2n_top", "forgetting", "random")


def harvest():
    gpu = {}
    for lf in glob.glob(f"{LOGS}/bfill_*.out"):
        txt = open(lf).read()
        g = "H100" if "H100" in txt else ("A100" if "A100" in txt else "?")
        for key in re.findall(r"ok (\w+_r[0-9.]+_\w+_s\d+)", txt):
            gpu[key] = g
    out, seen = [], set()
    for f in sorted(glob.glob(f"{TREE}/formal/ratio_*/*/*/seed_*/results/**/metrics.jsonl", recursive=True)):
        m = re.search(r"ratio_([0-9.]+)/(\w+)/(\w+)/seed_(\d+)/", f)
        r, d, arm, s = m[1], m[2], m[3], int(m[4])
        meth = arm if arm == "graph_a2" else re.sub(rf"{s}$", "", arm)
        assert meth in METHODS, (arm, s)
        if (d, r, meth, s) in seen:
            continue
        j = [json.loads(l) for l in open(f) if l.strip()]
        j = [x for x in j if x.get("corruption", "clean") == "clean"]
        if not j:
            continue
        seen.add((d, r, meth, s))
        out.append(dict(ds=d, ratio=float(r), arm=meth, run_arm=arm, seed=s, ba=100 * j[-1]["ba"],
                        worst=100 * j[-1]["worst_recall"], gpu=gpu.get(f"{d}_r{r}_{arm}_s{s}", "?")))
    json.dump(out, open("bench_fill_cells.json", "w"), indent=0)
    print(f"wrote bench_fill_cells.json ({len(out)} cells; gpu {sorted(set(x['gpu'] for x in out))})")


if __name__ == "__main__":
    harvest()
