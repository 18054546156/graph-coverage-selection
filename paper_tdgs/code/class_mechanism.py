#!/usr/bin/env python
"""Class-level mechanism regression: does Delta recall_c track class c's
cross-class edge share?

WHY. The mechanism claim -- coverage credit leaks across the class boundary and
the damage scales with the leak -- has so far only been tested at the dataset
level. That is 5 points, and organamnist/organsmnist are the same 201 LiTS
volumes, so the effective N is 2-3. Both sides of the claim are however defined
per CLASS, so descending one level gives 9+11+11+8+8 = 47 points at zero extra
training cost: y_c comes from recalls[] already in the round-1 harvest, x_c from
perclass_crossshare.py.

WHAT THIS IS NOT. 47 class-points are not 47 independent observations -- classes
within a dataset share a training run, a pool, and a graph. The honest model is
therefore a random-intercept-per-dataset fit, reported next to the naive pooled
fit so the shrinkage is visible. Both are printed; the paper should quote the
within-dataset one.

Usage:
  python class_mechanism.py --harvest results/round1/tdgs_round1_harvest.json \
      [--crossshare <dir with *_perclass_crossshare.json>]
Without --crossshare it runs y-side only, which is enough to check that the
per-class deltas are large and reproducible before spending anything on x.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

TREAT, BASE = "tdgs_cls", "graph_a2"


def load_harvest(p: str):
    rows = json.load(open(p))
    # key on (ds, ratio, arm, TRAIN seed) -- selection_seed is pinned to 42 in
    # this harness while the training seed varies, so 'seed' here is the training
    # seed and is the only legal pairing key.
    by = {}
    for r in rows:
        by[(r["ds"], r["ratio"], r["arm"], r["seed"])] = r
    return rows, by


def per_class_deltas(by, ratio=0.02):
    """Paired per-class recall delta, averaged over the seeds present in BOTH arms."""
    seeds = defaultdict(list)
    for ds, rt, arm, sd in by:
        if rt == ratio and arm in (TREAT, BASE):
            seeds[(ds, sd)].append(arm)
    paired = sorted({(ds, sd) for (ds, sd), a in seeds.items() if len(set(a)) == 2})

    acc = defaultdict(list)  # (ds, class) -> [delta per seed]
    sat = defaultdict(list)  # (ds, class) -> [both arms already at recall 1.0?]
    for ds, sd in paired:
        t = np.asarray(by[(ds, ratio, TREAT, sd)]["recalls"], float)
        b = np.asarray(by[(ds, ratio, BASE, sd)]["recalls"], float)
        if t.shape != b.shape:
            raise SystemExit(f"class count mismatch {ds} s{sd}: {t.shape} vs {b.shape}")
        for c, d in enumerate(100.0 * (t - b)):
            acc[(ds, c)].append(float(d))
            sat[(ds, c)].append(bool(t[c] >= 1.0 and b[c] >= 1.0))

    out = []
    for (ds, c), v in sorted(acc.items()):
        v = np.asarray(v)
        out.append(dict(ds=ds, cls=c, n_seed=len(v), mean=v.mean(),
                        sd=v.std(ddof=1) if len(v) > 1 else float("nan"),
                        saturated=all(sat[(ds, c)]),
                        per_seed=v.tolist()))
    return out, paired


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / den) if den else float("nan")


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(x) - 2
    if dof <= 0:
        return beta[1], float("nan"), float("nan")
    s2 = (resid**2).sum() / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = float(np.sqrt(cov[1, 1]))
    return float(beta[1]), se, float(beta[1] / se) if se else float("nan")


def demean_by_group(x, g):
    """Within-dataset centring -- the fixed-effects equivalent of a random
    intercept per dataset, done without a mixed-model dependency."""
    x = np.asarray(x, float)
    out = x.copy()
    for k in set(g):
        m = np.array([gi == k for gi in g])
        out[m] = x[m] - x[m].mean()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", required=True)
    ap.add_argument("--crossshare", default=None)
    ap.add_argument("--ratio", type=float, default=0.02)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    _, by = load_harvest(a.harvest)
    rows, paired = per_class_deltas(by, a.ratio)

    print(f"paired (dataset, seed) cells: {len(paired)}  "
          f"class-points: {len(rows)}")
    print(f"\n== y side: Delta recall_c ({TREAT} - {BASE}), pp ==")
    print(f"{'dataset':>12} {'cls':>4} {'mean':>8} {'sd':>7}   per-seed")
    for r in rows:
        print(f"{r['ds']:>12} {r['cls']:>4} {r['mean']:>8.2f} {r['sd']:>7.2f}   "
              + " ".join(f"{v:+6.2f}" for v in r["per_seed"]))

    y = np.array([r["mean"] for r in rows])
    ds = [r["ds"] for r in rows]
    print(f"\nclass-level delta: mean {y.mean():+.2f}pp  sd {y.std(ddof=1):.2f}pp  "
          f"range [{y.min():+.2f}, {y.max():+.2f}]  positive {int((y > 0).sum())}/{len(y)}")

    # How much of the class-level spread is BETWEEN datasets vs WITHIN? If it is
    # nearly all between, the 47 points carry little more information than the 5
    # dataset means and the whole exercise is cosmetic.
    grand = y.mean()
    between = sum(((y[[i for i, d in enumerate(ds) if d == k]].mean() - grand) ** 2)
                  * sum(1 for d in ds if d == k) for k in set(ds))
    total = ((y - grand) ** 2).sum()
    print(f"variance decomposition: between-dataset {between/total:.1%}, "
          f"within-dataset {1 - between/total:.1%}")

    # seed reproducibility of the per-class delta -- if sd >> |mean| the class
    # points are noise and no x can rescue them
    sds = np.array([r["sd"] for r in rows])
    print(f"within-class seed sd: median {np.median(sds):.2f}pp  "
          f"|mean|/sd median {np.median(np.abs(y) / np.maximum(sds, 1e-9)):.2f}")

    # ---- is the within-dataset spread REAL class heterogeneity, or seed noise?
    # The class means are averages over n_seed seeds, so each carries sampling
    # variance sd_seed^2 / n_seed. Subtract it from the observed within-dataset
    # variance to get the signal component. If that comes out <= 0 the 47 points
    # are cosmetic and the regression must not be run.
    n_seed = np.array([r["n_seed"] for r in rows], float)
    var_noise = float(np.mean(sds**2 / n_seed))
    var_within_obs = float((1 - between / total) * total / len(y))
    var_signal = var_within_obs - var_noise
    print(f"\nwithin-dataset variance: observed {var_within_obs:.2f} "
          f"= signal {var_signal:.2f} + seed noise {var_noise:.2f}  "
          f"(sd: {np.sqrt(max(var_signal,0)):.2f}pp real vs "
          f"{np.sqrt(var_noise):.2f}pp noise)")
    print(f"  -> R^2 ceiling for any x: {max(var_signal,0)/var_within_obs:.2f}")
    print("  NOTE noise in y does not bias an OLS slope, only inflates its se;")
    print("       x (cross-class share) is computed exactly from K, so there is")
    print("       no errors-in-variables attenuation. The regression is legal.")

    # which classes individually survive their own seed noise
    tstat = np.abs(y) / np.maximum(sds / np.sqrt(n_seed), 1e-9)
    strong = [(rows[i]["ds"], rows[i]["cls"], y[i], tstat[i])
              for i in np.argsort(-tstat)[:10]]
    print(f"\ntop class-points by |t| over seeds ({int((tstat > 2).sum())}/{len(y)} "
          f"exceed |t|=2):")
    for d, c, v, t in strong:
        print(f"  {d:>12} cls {c:<3} {v:+7.2f}pp  |t|={t:.2f}")

    # Classes where BOTH arms already sit at recall 1.000 on every seed have zero
    # headroom: their delta is exactly 0 by construction, not by measurement, and
    # including them biases the regression toward the origin. Flag and drop.
    n_sat = sum(1 for r in rows if r["saturated"])
    if n_sat:
        print(f"\nsaturated class-points (both arms recall=1.000 on all seeds): "
              f"{n_sat} -> "
              + ", ".join(f"{r['ds']}/{r['cls']}" for r in rows if r["saturated"]))
        print("  these are excluded from the regression: zero headroom by "
              "construction, not a measured null")

    out = dict(ratio=a.ratio, treat=TREAT, base=BASE,
               paired_cells=[list(p) for p in paired], classes=rows)

    if a.crossshare:
        xs = {}
        for f in glob.glob(os.path.join(a.crossshare, "*_perclass_crossshare.json")):
            d = json.load(open(f))
            for r in d["per_class"]:
                xs[(d["dataset_name"], r["class"])] = r
        miss = [(r["ds"], r["cls"]) for r in rows if (r["ds"], r["cls"]) not in xs]
        if miss:
            print(f"\n!! missing cross-share for {len(miss)} class-points: "
                  f"{sorted({m[0] for m in miss})}")
        keep = [r for r in rows
                if (r["ds"], r["cls"]) in xs and not r["saturated"]]
        if not keep:
            print("no joinable points; run tdgs_crossshare.slurm first")
        else:
            for r in keep:
                r.update(xs[(r["ds"], r["cls"])])
            print(f"\n== joined mechanism regression, N={len(keep)} class-points ==")
            yk = np.array([r["mean"] for r in keep])
            dsk = [r["ds"] for r in keep]
            for tag in ("cross_share_mass", "cross_share_count"):
                xk = np.array([r[tag] for r in keep])
                b, se, t = ols(xk, yk)
                bw, sew, tw = ols(demean_by_group(xk, dsk),
                                  demean_by_group(yk, dsk))
                print(f"  {tag}:")
                print(f"    pooled   slope {b:+.2f} pp per unit  se {se:.2f}  "
                      f"t {t:+.2f}  rho {spearman(xk, yk):+.3f}")
                print(f"    within-ds slope {bw:+.2f}  se {sew:.2f}  t {tw:+.2f}  "
                      f"(dataset intercepts removed -- quote THIS one)")
            out["joined"] = keep

    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
