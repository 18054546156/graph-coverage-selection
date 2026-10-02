"""Build the S2 worklist: one line per S1 checkpoint, pointing at that run's own selection.

Output ~/corr/s2_worklist.tsv: dataset ratio arm seed selection_file
Ordered longest-first (path 5% -> blood 2%) so the packer's tail is short.
"""
import os

import numpy as np

import corr_rescore_s1 as S

records = S.discover(set(S.DATASETS), S.RATIOS, set(S.ARMS), set(S.SEEDS))
cost = {("pathmnist", 0.05): 0, ("pathmnist", 0.02): 1, ("bloodmnist", 0.05): 2, ("bloodmnist", 0.02): 3}
lines = []
for rec in records:
    cfg = rec["config"]
    sel = os.path.join(rec["run_dir"], "selected_local_indices.npy")
    idx = np.load(sel)
    if len(idx) != cfg["n_selected"] or len(np.unique(idx)) != len(idx):
        raise RuntimeError("bad selection file " + sel)
    lines.append((cost[(cfg["dataset"], cfg["ratio"])], cfg["dataset"], cfg["ratio"],
                  cfg["method"], cfg["training_seed"], sel))
lines.sort()
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2_worklist.tsv"), "w") as handle:
    for _, ds, ratio, arm, seed, sel in lines:
        handle.write("{}\t{}\t{}\t{}\t{}\n".format(ds, ratio, arm, seed, sel))
print("wrote {} cells".format(len(lines)))
