"""Collect corruption-inference output into the two JSON lists shift_battery.py reads.

  S1 : python harvest_s1.py                                   (blood+path, original ckpts)
  S1b: python harvest_s1.py --tag s1b --datasets organamnist organsmnist tissuemnist \
           --out-root /mnt/prj01/hgrp-1502-5TB/tdgs_shared/runs/corr_s1b_20261002
  S2 : python harvest_s1.py --tag s2 --tree s2
Writes <tag>_corr.json and <tag>_clean.json next to this script.
Corruption rows come from <out-root>/**/metrics_corr.jsonl (filtered to the requested
datasets); clean rows come from each source run's own metrics.jsonl (via discover()).
"""
import argparse
import glob
import json
import os

import corr_rescore_s1 as S

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="s1")
ap.add_argument("--tree", choices=["s1", "s2"], default="s1")
ap.add_argument("--datasets", nargs="+", default=S.DATASETS)
ap.add_argument("--out-root", default=None)
args = ap.parse_args()
out_root = args.out_root or (S.S2_OUT_ROOT if args.tree == "s2" else S.OUT_ROOT)

corr_rows = []
for ds in args.datasets:
    for path in sorted(glob.glob(os.path.join(out_root, ds, "**", "metrics_corr.jsonl"), recursive=True)):
        for line in open(path):
            line = line.strip()
            if line:
                corr_rows.append(json.loads(line))

clean_rows = []
for rec in S.discover(set(args.datasets), S.RATIOS, set(S.ARMS), set(S.SEEDS), args.tree):
    row = dict(rec["clean"])
    cfg = rec["config"]
    row.update(dataset=cfg["dataset"], ratio=cfg["ratio"], method=cfg["method"],
               training_seed=cfg["training_seed"], corruption="clean", severity=0,
               source_ckpt=rec["ckpt"])
    clean_rows.append(row)

here = os.path.dirname(os.path.abspath(__file__))
json.dump(corr_rows, open(os.path.join(here, args.tag + "_corr.json"), "w"))
json.dump(clean_rows, open(os.path.join(here, args.tag + "_clean.json"), "w"))
print("{}: corr rows {} from {}, clean rows {} (expect {} ckpts)".format(
    args.tag, len(corr_rows), out_root, len(clean_rows), len(args.datasets) * 2 * 9 * 5))
