#!/usr/bin/env python
"""Paired analysis of the TDGS ladder, with the round-2 gates stated as code.

Reporting rules this file enforces, all of them from measured properties of this
harness rather than convention:

 * Pair by TRAINING SEED. Training is bit-deterministic per seed, so the same
   seed on two arms shares its initialisation exactly and the difference is
   attributable to the subset. Unpaired comparison throws that away.

 * Never collapse to a single p-value. Per-cell MDE at n=3 is 2.4-17.6pp while
   the effect outside tissuemnist is +0.81pp, so p>0.05 carries no information
   here. SIGN CONSISTENCY across datasets is the primary evidence; the sign test
   over 5 datasets has its own exact p (1/32 = 0.031 for 5/5).

 * Report pooled AND pooled-excluding-tissuemnist. tissuemnist carries the
   round-1 pooled number (+5.55 of the +1.76pp average), and hiding that behind a
   single figure would be the same mistake as the "10/10 calibration" unanimity
   that turned out to be 1.31 effective dimensions.

 * Datasets are not independent: organamnist and organsmnist are the same 201
   LiTS volumes, so effective N is 2-3, not 5. The pooled SE is printed but
   flagged, never used as if n=15 were 15 independent observations.
"""
import argparse
import json
import itertools
import math
from collections import defaultdict

import numpy as np

DS = ["pathmnist", "organamnist", "bloodmnist", "organsmnist", "tissuemnist"]
# cross-class share of K = A_sym + A_sym^2 (job 35062). The earlier 0.6/9.9/20.3/
# 32.1/58.5 was the 1-hop A share: same order, different values -- never mix them.
XSHARE = {"pathmnist": 2.49, "bloodmnist": 26.52, "organamnist": 42.41,
          "organsmnist": 53.77, "tissuemnist": 66.06}
DS_BY_SHARE = sorted(DS, key=XSHARE.get)

CONTRASTS = [
    # (hi, lo, what it decides)
    ("tdgs_cls", "graph_a2", "round-1 headline, re-measured at n=5 / 5%"),
    ("a2_perclass", "graph_a2",
     "P0-a: does the AUTHOR'S per-class scope already get the gain?"),
    ("tdgs_cls", "a2_perclass",
     "P0-a: is the lam blend better than pure per-class scope?"),
    ("tdgs_mask", "graph_a2", "P0-b factorial: MASK main effect"),
    ("tdgs_wcls", "graph_a2", "P0-b factorial: class-flat WEIGHT main effect"),
    ("tdgs_cls", "tdgs_mask", "P0-b factorial: weight effect given mask"),
    ("tdgs_cls", "tdgs_wcls", "P0-b factorial: mask effect given weight"),
    ("tdgs_cls", "tdgs_cls_perm",
     "P1-a: THE NULL. If ~0, the win is not the within-class geometry."),
    ("tdgs_cls_perm", "graph_a2",
     "P1-a: how much does confining credit buy with geometry destroyed?"),
    ("tdgs_lam1", "tdgs_cls", "P1-b: is the global term dead weight?"),
    ("tdgs_lam1", "graph_a2", "P1-b: pure class-compatible coverage vs author"),
]


def load(paths):
    rows = []
    for p in paths:
        if p:
            rows += json.load(open(p))
    by = defaultdict(dict)
    for r in rows:
        by[(r["ds"], r["ratio"], r["seed"])][r["arm"]] = r
    return rows, by


def paired(by, hi, lo, met, ratio):
    """Return {dataset: [per-seed deltas in pp]}."""
    out = {}
    for ds in DS:
        dd = []
        for (d2, r2, s) in sorted(by):
            if d2 != ds or abs(r2 - ratio) > 1e-9:
                continue
            cell = by[(d2, r2, s)]
            if hi in cell and lo in cell:
                dd.append((cell[hi][met] - cell[lo][met]) * 100)
        if dd:
            out[ds] = dd
    return out


def sign_test_p(k, n):
    """Two-sided exact sign test."""
    if n == 0:
        return float("nan")
    k = max(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def report(by, met, ratio):
    print(f"\n{'='*100}\n  {met.upper()}  @ ratio {ratio}   (all values pp; "
          f"paired by training seed)\n{'='*100}")
    for hi, lo, why in CONTRASTS:
        per = paired(by, hi, lo, met, ratio)
        if not per:
            continue
        means = {k: np.mean(v) for k, v in per.items()}
        alld = [x for v in per.values() for x in v]
        pooled = np.mean(alld)
        sd = np.std(alld, ddof=1) if len(alld) > 1 else float("nan")
        se = sd / np.sqrt(len(alld)) if len(alld) > 1 else float("nan")
        ex = [x for k, v in per.items() if k != "tissuemnist" for x in v]
        npos_ds = sum(v > 0 for v in means.values())
        npos_cell = sum(x > 0 for x in alld)
        print(f"\n  {hi} - {lo}")
        print(f"    {why}")
        print(f"    pooled {pooled:+6.2f}  (sd {sd:4.2f}, se {se:4.2f}, "
              f"n={len(alld)} cells over {len(per)} datasets -- effective "
              f"dataset N is 2-3, not {len(per)})")
        if ex:
            print(f"    pooled EXCLUDING tissuemnist {np.mean(ex):+6.2f}  "
                  f"(n={len(ex)})")
        print(f"    sign: {npos_ds}/{len(means)} datasets, "
              f"{npos_cell}/{len(alld)} cells  "
              f"(dataset sign test p={sign_test_p(npos_ds, len(means)):.3f})")
        print("    per dataset (ordered by cross-class edge share): " +
              "  ".join(f"{d[:6]}({XSHARE[d]:.1f}%){means[d]:+6.2f}"
                        for d in DS_BY_SHARE if d in means))
        xs = [XSHARE[d] for d in DS if d in means]
        ys = [means[d] for d in DS if d in means]
        if len(xs) >= 4:
            rx = np.argsort(np.argsort(xs))
            ry = np.argsort(np.argsort(ys))
            rho = np.corrcoef(rx, ry)[0, 1]
            print(f"    Spearman(cross-class share, effect) = {rho:+.2f}   "
                  "(mechanism predicts strongly positive for mask-based arms)")


def factorial(by, met, ratio):
    """2x2: rows = global/within-class credit, cols = uniform/class-flat weight."""
    cells = {"graph_a2": "global+uniform", "tdgs_mask": "mask+uniform",
             "tdgs_wcls": "global+classflat", "tdgs_cls": "mask+classflat"}
    print(f"\n{'='*100}\n  2x2 FACTORIAL on {met.upper()} @ ratio {ratio}"
          f"\n{'='*100}")
    have = set()
    for (d, r, s), cell in by.items():
        if abs(r - ratio) < 1e-9:
            have |= set(cell)
    if not set(cells) <= have:
        print(f"  incomplete: missing {sorted(set(cells) - have)}")
        return
    print(f"  {'dataset':14s}{'a2':>9s}{'mask':>9s}{'wcls':>9s}{'cls':>9s}"
          f"{'mask_eff':>10s}{'wt_eff':>9s}{'inter':>9s}")
    inters = []
    for ds in DS:
        v = {}
        for arm in cells:
            xs = [by[(d, r, s)][arm][met] * 100
                  for (d, r, s) in by
                  if d == ds and abs(r - ratio) < 1e-9 and arm in by[(d, r, s)]]
            if xs:
                v[arm] = np.mean(xs)
        if len(v) < 4:
            continue
        me = v["tdgs_mask"] - v["graph_a2"]
        we = v["tdgs_wcls"] - v["graph_a2"]
        it = v["tdgs_cls"] - v["tdgs_mask"] - v["tdgs_wcls"] + v["graph_a2"]
        inters.append(it)
        print(f"  {ds:14s}{v['graph_a2']:9.2f}{v['tdgs_mask']:9.2f}"
              f"{v['tdgs_wcls']:9.2f}{v['tdgs_cls']:9.2f}"
              f"{me:+10.2f}{we:+9.2f}{it:+9.2f}")
    if inters:
        print(f"\n  mean interaction {np.mean(inters):+.2f}pp. A large positive "
              "interaction means the two corrections are NOT separable --\n"
              "  credit must be confined AND reweighted -- which is a stronger "
              "claim than either main effect alone.")


GATES = """
{sep}
  PRE-REGISTERED GATES (written before round 2 ran; read these, not the p-values)
{sep}
  G1  KILL GATE.  If (a2_perclass - graph_a2) recovers >= 60% of
      (tdgs_cls - graph_a2) on >= 3/5 datasets, then tdgs_cls sits inside the
      author's own compare_global.py ablation cell and is NOT a contribution.
      Consequence: drop the coverage-alignment framing, keep only the Act-2
      flatness result, and the paper becomes a negative/diagnostic paper.

  G2  NULL GATE.  If (tdgs_cls - tdgs_cls_perm) <= 0 pooled, the win comes from
      confining credit within class AT ALL, not from within-class kNN geometry.
      That is still publishable but must be described that way -- it makes the
      method a class-scoping correction, not a graph method.

  G3  FACTORIAL GATE.  If the mask main effect is >= 3x the weight main effect,
      report the method as class-compatible coverage and demote the weighting to
      an implementation detail. If they are comparable, both are required and the
      interaction term decides whether they are separable.

  G4  POWER GATE.  The headline (tdgs_cls - graph_a2) must survive at n=5 seeds
      AND at ratio 0.05 with the same sign on >= 4/5 datasets. If it holds only
      at 2%, the claim is budget-specific and the abstract must say so.

  G5  TISSUE GATE.  If excluding tissuemnist drops the pooled effect below
      +0.5pp, the honest headline is "one dataset, mechanistically predicted"
      plus a 5-dataset directional trend -- NOT a 5-dataset mean improvement.
{sep}
""".format(sep="=" * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", required=True)
    ap.add_argument("--round1", default=None)
    ap.add_argument("--ratios", type=float, nargs="*", default=[0.02, 0.05])
    a = ap.parse_args()

    rows, by = load([a.round1, a.harvest])
    arms = sorted({r["arm"] for r in rows})
    print(f"loaded {len(rows)} cells, arms: {arms}")

    for ratio in a.ratios:
        sub = {k: v for k, v in by.items() if abs(k[1] - ratio) < 1e-9}
        if not sub:
            continue
        print(f"\n\n{'#'*100}\n#  RATIO {ratio}\n{'#'*100}")
        print(f"\n  per-arm means (BA %, then worst-class recall %)")
        present = [x for x in ["graph_a2", "a2_perclass", "tdgs_mask",
                               "tdgs_wcls", "tdgs_cls", "tdgs_cls_perm",
                               "tdgs_lam1", "tdgs_d", "tdgs_du", "tdgs_perm"]
                   if x in arms]
        for met in ("ba", "worst"):
            print(f"\n  [{met}] {'dataset':14s}"
                  + "".join(f"{x[:11]:>12s}" for x in present))
            for ds in DS:
                line = f"        {ds:14s}"
                for arm in present:
                    xs = [c[arm][met] * 100 for k, c in sub.items()
                          if k[0] == ds and arm in c]
                    line += f"{np.mean(xs):12.2f}" if xs else f"{'--':>12s}"
                print(line)
        for met in ("ba", "worst"):
            report(sub, met, ratio)
        factorial(sub, "ba", ratio)
        factorial(sub, "worst", ratio)
    print(GATES)


if __name__ == "__main__":
    main()
