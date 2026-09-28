#!/usr/bin/env python
"""Harvest one cell per (dataset, ratio, arm, training seed) from a TDGS run tree.

Two traps this encodes, both of which have already cost real time here:

 1. The arm identity is NOT in the run metadata. td_run_one.sh writes
    `methods: [precomputed]` into every config, so `method` is the string
    "precomputed" in all 145 cells. The arm only exists in the OUTPUT PATH
    (.../ratio_R/<dataset>/<arm>/seed_<s>/...). Read it from there or every arm
    collapses into one.

 2. `selection_seed` is pinned to 42 for every cell while the TRAINING seed
    varies. Keying on selection_seed silently deduplicates all seeds down to one
    row per arm. The key is (dataset, ratio, arm, train_seed).

Balanced accuracy and per-class recall are recomputed from predictions_clean.npz
rather than read from metrics.jsonl, so that every number in the paper comes from
one code path regardless of which harness version wrote the cell.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

PAT = re.compile(r"/ratio_([0-9.]+)/([a-z]+mnist)/([A-Za-z0-9_]+)/seed_(\d+)/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows, bad = [], []
    for d in sorted(glob.glob(f"{a.exp_root}/formal/ratio_*/**/aug_0",
                              recursive=True)):
        m = PAT.search(d.replace(os.sep, "/"))
        if not m:
            bad.append(d)
            continue
        ratio, ds, arm, seed = (float(m.group(1)), m.group(2), m.group(3),
                                int(m.group(4)))
        npz_p = os.path.join(d, "predictions_clean.npz")
        if not os.path.exists(npz_p):
            continue
        z = np.load(npz_p)
        y = z["y_true"].ravel()
        p = z["y_pred"].ravel() if "y_pred" in z else z["logits"].argmax(1)
        C = int(y.max()) + 1
        rec = np.array([(p[y == c] == c).mean() if (y == c).sum() else np.nan
                        for c in range(C)])
        rows.append(dict(ds=ds, ratio=ratio, arm=arm, seed=seed,
                         ba=float(np.nanmean(rec)), worst=float(np.nanmin(rec)),
                         acc=float((p == y).mean()),
                         recalls=[round(float(x), 6) for x in rec],
                         path=d))

    key = lambda r: (r["ds"], r["ratio"], r["arm"], r["seed"])
    seen, uniq = set(), []
    for r in rows:
        if key(r) in seen:
            print(f"DUPLICATE KEY skipped: {key(r)}  {r['path']}")
            continue
        seen.add(key(r))
        uniq.append(r)

    json.dump(uniq, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(uniq)} cells "
          f"({len(rows)-len(uniq)} duplicates dropped, {len(bad)} unparsable)")
    arms = sorted({r["arm"] for r in uniq})
    for arm in arms:
        sub = [r for r in uniq if r["arm"] == arm]
        rr = sorted({r["ratio"] for r in sub})
        print(f"  {arm:16s} {len(sub):3d} cells  ratios={rr}  "
              f"seeds={sorted({r['seed'] for r in sub})}")


if __name__ == "__main__":
    main()
