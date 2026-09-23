"""Does the linear probe rank selections the same way real training does?

This project has now had four instances of "offline gate passed, real training did
not agree" (FRACTAL, tissuemnist Voronoi, task-geometry, and as of today the
weighting x selection interaction, gate_training.py -> REFUTED with pooled
-0.064 pp against a probe prediction of +1.05/+2.92/+1.60). Every one of them has
the same shape: the criterion is computed on UNI embeddings, the endpoint comes from
a ResNet-18 trained from scratch on pixels.

That is a hypothesis about the *instrument*, not about any particular method, and it
is directly testable with what is already on disk: the 360 real-training runs stand
on 60 distinct selections, and a probe score can be computed for each of them.

If probe BA does not rank those 60 selections the way real training does, then no
probe-gated conclusion in this project -- including the functional screen, whose
endpoint is probe BA -- means what it was taken to mean.

Two things are deliberately separated:
* **level agreement** (does the probe predict the training BA value) -- expected to
  fail trivially, different model class and different eval split; not interesting.
* **rank agreement within a cell** (at fixed dataset, does a selection the probe
  prefers also train better) -- this is what a screen actually needs, and it is
  computed with dataset and replicate structure held fixed.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np

import build_training_cells as B
import run_linear_probe as R

WEIGHTING = "equal"  # compare like with like: unweighted probe vs unweighted training


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", type=Path, required=True)
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--embedding-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--pool-fraction", type=float, default=0.80)
    p.add_argument("--size", type=int, default=224)
    args = p.parse_args()

    # real training BA, averaged over the two train seeds
    train_ba = collections.defaultdict(list)
    for path in args.results:
        for line in Path(path).open(encoding="utf-8"):
            r = json.loads(line)
            if r["weighting"] == WEIGHTING:
                train_ba[(r["dataset"], r["replicate"], r["arm"])].append(r["balanced_accuracy"])

    cells = {}
    for line in args.cells.open(encoding="utf-8"):
        c = json.loads(line)
        if c["weighting"] == WEIGHTING:
            cells[(c["dataset"], c["replicate"], c["arm"])] = c["selected_indices"]

    rows = []
    for dataset in sorted({k[0] for k in cells}):
        emb = args.embedding_root / f"{dataset}_train_uni_{args.size}.npz"
        x = np.asarray(np.load(emb, allow_pickle=False)["embeddings"], dtype=np.float32)
        y = R.read_train_labels(emb, args.data_root / f"{dataset}_{args.size}.npz")
        # build_training_cells indexes into R.DATASETS (all five), NOT its own
        # three-element DATASETS. Using the wrong list silently shifts the seed
        # schedule for every dataset except bloodmnist, which is index 0 in both.
        di = R.DATASETS.index(dataset)
        z = R.normalize(x)
        del x

        for (ds, rep, arm), sel in sorted(cells.items()):
            if ds != dataset:
                continue
            # Same seed schedule as build_training_cells, so the audit split is the
            # exact complement of the pool that produced this selection.
            rseed = args.seed + 100003 * di + 1009 * rep
            pool = B.replicate_pool(y, args.pool_fraction, rseed)
            audit = np.setdiff1d(np.arange(len(y)), pool, assume_unique=False)
            sel = np.asarray(sel, dtype=np.int64)
            if not np.isin(sel, pool).all():
                raise RuntimeError(f"{ds} rep{rep} {arm}: selection is not inside the "
                                   "reconstructed pool -- the seed schedule drifted")
            m = R.metrics(z[sel], y[sel], z[audit], y[audit], rseed)
            rows.append({
                "dataset": ds, "replicate": rep, "arm": arm,
                "probe_ba": m["balanced_accuracy"],
                "train_ba": float(np.mean(train_ba[(ds, rep, arm)])),
                "n_train_seeds": len(train_ba[(ds, rep, arm)]),
            })
            print(f"  {ds:<13} rep{rep:<3} {arm:<8} probe {m['balanced_accuracy']:.4f} "
                  f"train {np.mean(train_ba[(ds, rep, arm)]):.4f}", flush=True)
        del z

    print(f"\n{'='*72}\nRANK AGREEMENT, dataset held fixed  (n per row = 20 selections)")
    print(f"{'dataset':<14}{'n':>4}{'spearman':>11}{'pearson':>10}{'t':>8}   arms pooled")
    summary = {}
    for ds in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds]
        a = np.array([r["probe_ba"] for r in sub])
        b = np.array([r["train_ba"] for r in sub])
        rho, pr = spearman(a, b), float(np.corrcoef(a, b)[0, 1])
        t = pr * math.sqrt((len(a) - 2) / max(1 - pr * pr, 1e-12))
        summary[ds] = {"n": len(a), "spearman": rho, "pearson": pr, "t": t}
        print(f"{ds:<14}{len(a):>4}{rho:>+11.3f}{pr:>+10.3f}{t:>+8.2f}")

    # within arm as well: pooling random+herding lets a between-arm difference
    # masquerade as rank agreement
    print(f"\n{'dataset':<14}{'arm':<10}{'n':>4}{'spearman':>11}{'pearson':>10}{'t':>8}")
    for ds in sorted({r["dataset"] for r in rows}):
        for arm in ("random", "herding"):
            sub = [r for r in rows if r["dataset"] == ds and r["arm"] == arm]
            if len(sub) < 5:
                continue
            a = np.array([r["probe_ba"] for r in sub])
            b = np.array([r["train_ba"] for r in sub])
            pr = float(np.corrcoef(a, b)[0, 1])
            t = pr * math.sqrt((len(a) - 2) / max(1 - pr * pr, 1e-12))
            summary[f"{ds}/{arm}"] = {"n": len(a), "spearman": spearman(a, b),
                                      "pearson": pr, "t": t}
            print(f"{ds:<14}{arm:<10}{len(a):>4}{spearman(a,b):>+11.3f}{pr:>+10.3f}{t:>+8.2f}")

    print(f"\n{'='*72}")
    within = [v for k, v in summary.items() if "/" in k]
    pooled_rho = float(np.mean([v["spearman"] for v in within]))
    print(f"mean within-(dataset,arm) Spearman = {pooled_rho:+.3f} over {len(within)} cells")
    print("If this is near zero, probe BA does not rank selections the way real")
    print("training does, and every probe-gated result in this project -- including")
    print("the functional screen's endpoint -- is measuring the probe, not accuracy.")

    args.output.write_text(json.dumps({"rows": rows, "summary": summary,
                                       "mean_within_cell_spearman": pooled_rho},
                                      indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
