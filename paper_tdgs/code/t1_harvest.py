#!/usr/bin/env python
"""Harvest Table-1 baseline cells from archive-layout trees.

Layout (the archive's, which t1_run_one.sh mirrors):
  <root>/formal/<ds>/seed_<s>/results/<ds>/<method>/ratio_<r>/
        selection_seed_<s>/train_seed_<t>/aug_0/predictions_clean.npz

The ratio is read from the path, never from the tree name: the 2% and 5%
archive trees are separate directories but a single t1 tree holds both.

BA / worst / per-class recall are recomputed from predictions_clean.npz with the
same code as harvest.py, so Table 1 and the TDGS ladder share one metric path.

--compare-archive: for every (ds, ratio, method, seed) present in BOTH the new
tree and an archive tree, report whether the predictions are bit-identical.
That is the harness-identity check: same code, same seed, same selection,
DETERMINISTIC_TRAINING=1 in both -> the logits must match exactly.
"""
import argparse
import glob
import hashlib
import json
import os
import re

import numpy as np

PAT = re.compile(r"/formal/([a-z]+mnist)/seed_(\d+)/results/\1/([A-Za-z0-9_]+)/"
                 r"ratio_([0-9.]+)/selection_seed_(\d+)/train_seed_(\d+)/aug_0$")


def scan(root, tag):
    out = {}
    for d in glob.glob(f"{root}/formal/*/seed_*/results/*/*/ratio_*/"
                       f"selection_seed_*/train_seed_*/aug_0"):
        m = PAT.search(d.replace(os.sep, "/"))
        if not m:
            continue
        ds, s, method, ratio, sel, tr = m.groups()
        npz_p = os.path.join(d, "predictions_clean.npz")
        if not os.path.exists(npz_p):
            continue
        z = np.load(npz_p)
        y = z["y_true"].ravel()
        p = z["y_pred"].ravel() if "y_pred" in z else z["logits"].argmax(1)
        C = int(y.max()) + 1
        rec = np.array([(p[y == c] == c).mean() if (y == c).sum() else np.nan
                        for c in range(C)])
        key = (ds, float(ratio), method, int(tr))
        out[key] = dict(ds=ds, ratio=float(ratio), arm=method, seed=int(tr),
                        sel_seed=int(sel), ba=float(np.nanmean(rec)),
                        worst=float(np.nanmin(rec)), acc=float((p == y).mean()),
                        recalls=[round(float(x), 6) for x in rec],
                        logits_sha=hashlib.sha256(
                            np.ascontiguousarray(z["logits"]).tobytes()).hexdigest(),
                        src=tag, path=d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare-archive", nargs="*", default=[])
    ap.add_argument("--fill-from", nargs="*", default=[],
                    help="OPTIONAL fallback only: archive roots whose seed-42..46 "
                         "cells fill gaps the new tree does not have. The t1 "
                         "worklist re-runs all 350 cells, so normally unused")
    a = ap.parse_args()

    new = scan(a.root, "t1_s4246")
    merged = dict(new)
    for root in a.fill_from:
        for k, r in scan(root, root).items():
            if 42 <= k[3] <= 46 and k not in merged and k[2] != "graph_a2":
                merged[k] = r
    rows = sorted(merged.values(), key=lambda r: (r["ds"], r["ratio"], r["arm"], r["seed"]))
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(rows)} cells "
          f"({len(new)} new tree, {len(rows) - len(new)} reused from archive)")
    for ratio in (0.02, 0.05):
        for arm in sorted({r["arm"] for r in rows}):
            sub = [r for r in rows if r["ratio"] == ratio and r["arm"] == arm]
            if sub:
                print(f"  r{ratio} {arm:11s} {len(sub):3d}  "
                      f"seeds={sorted({r['seed'] for r in sub})}  "
                      f"ds={len({r['ds'] for r in sub})}")

    if a.compare_archive:
        old = {}
        for root in a.compare_archive:
            old.update(scan(root, root))
        both = sorted(set(new) & set(old))
        same = [k for k in both if new[k]["logits_sha"] == old[k]["logits_sha"]]
        print(f"\nharness identity vs archive: {len(same)}/{len(both)} "
              f"overlapping cells bit-identical")
        for k in both:
            if k not in same:
                print(f"  DIFFERS {k}: ba new {new[k]['ba']:.4f} "
                      f"archive {old[k]['ba']:.4f}")


if __name__ == "__main__":
    main()
