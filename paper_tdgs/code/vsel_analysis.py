#!/usr/bin/env python
"""Amendment 2/3 analysis: per-dataset configuration chosen on VAL BA,
evaluated on TEST BA with leave-one-seed-out (the held-out seed's val never
enters its own choice, which removes the shared seed-level training noise).

Inputs: <val_eval>/val_metrics.jsonl (val_eval.py); the test metrics are read
from the metrics.jsonl the harness wrote next to each final.pt.

Endpoints (fixed in the amendment before any val BA was read):
  vsel - tdgs_cls   dataset mean >= +0.5pp, >= 3/5 datasets same sign
  vsel - mean(C)    > 0          (the value of choosing at all)
  reported, never used for a choice: oracle (argmax test BA over C per seed)
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

DSL = ["bloodmnist", "organamnist", "organsmnist", "pathmnist", "tissuemnist"]
SETS = {
    "C5_n5":  (["graph_a2", "a2_perclass", "tdgs_mask", "tdgs_cls"], [42, 43, 44, 45, 46]),
    "C6_n3":  (["graph_a2", "a2_perclass", "tdgs_mask", "tdgs_cls", "tdgs_lam1"], [42, 43, 44]),
}


def test_metrics(ckpt):
    with open(os.path.join(os.path.dirname(ckpt), "metrics.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("corruption", "clean") == "clean":
                return r["ba"], r["worst_recall"]
    raise ValueError(ckpt)


def hboot(d, B=20000, seed=0):
    """d: {ds: array over seeds}. Resample datasets, then seeds within."""
    rng = np.random.default_rng(seed)
    keys = list(d)
    out = np.empty(B)
    for b in range(B):
        ks = rng.choice(len(keys), len(keys))
        out[b] = np.mean([rng.choice(d[keys[k]], len(d[keys[k]])).mean() for k in ks])
    return np.percentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-eval", required=True)
    ap.add_argument("--ratio", type=float, default=0.02)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    V = defaultdict(dict)                 # (ds, arm) -> seed -> (val, test, twr)
    for line in open(os.path.join(a.val_eval, "val_metrics.jsonl")):
        r = json.loads(line)
        if abs(r["ratio"] - a.ratio) > 1e-9:
            continue
        tb, tw = test_metrics(r["ckpt"])
        V[(r["dataset"], r["arm"])][r["seed"]] = (r["ba"], tb, tw, r["worst_recall"])
    rep = {}
    for name, (C, seeds) in SETS.items():
        per_ds, lines = {}, []
        for ds in DSL:
            if not all(s in V.get((ds, c), {}) for c in C for s in seeds):
                lines.append(f"  {ds}: incomplete, skipped")
                continue
            rows = []
            for s in seeds:
                other = [t for t in seeds if t != s]
                score = {c: np.mean([V[(ds, c)][t][0] for t in other]) for c in C}
                cstar = max(C, key=lambda c: score[c])
                test = {c: V[(ds, c)][s][1] for c in C}
                rows.append(dict(seed=s, choice=cstar, vsel=test[cstar],
                                 tdgs_cls=test["tdgs_cls"], graph_a2=test["graph_a2"],
                                 meanC=float(np.mean(list(test.values()))),
                                 oracle=max(test.values()),
                                 vsel_worst=V[(ds, cstar)][s][2],
                                 cls_worst=V[(ds, "tdgs_cls")][s][2]))
            per_ds[ds] = rows
            # val-test agreement across all (arm, seed) of this dataset
            vt = np.array([V[(ds, c)][s][:2] for c in C for s in seeds])
            rho = float(np.corrcoef(vt[:, 0], vt[:, 1])[0, 1])
            d = lambda k: 100 * np.mean([r["vsel"] - r[k] for r in rows])
            lines.append(f"  {ds:12s} choices={[r['choice'] for r in rows]}  "
                         f"vsel-cls={d('tdgs_cls'):+.2f} vsel-a2={d('graph_a2'):+.2f} "
                         f"vsel-meanC={d('meanC'):+.2f} oracle-cls="
                         f"{100*np.mean([r['oracle']-r['tdgs_cls'] for r in rows]):+.2f}  "
                         f"r(val,test)={rho:.2f}")
        res = {}
        for k in ("tdgs_cls", "graph_a2", "meanC"):
            dd = {ds: 100 * np.array([r["vsel"] - r[k] for r in rows]) for ds, rows in per_ds.items()}
            if not dd:
                continue
            means = {ds: float(v.mean()) for ds, v in dd.items()}
            res[f"vsel-{k}"] = dict(
                dataset_mean=float(np.mean(list(means.values()))),
                ci95=hboot(dd).tolist(),
                n_pos=sum(m > 0 for m in means.values()), n_ds=len(means),
                excl_tissue=float(np.mean([m for ds, m in means.items() if ds != "tissuemnist"])),
                per_ds=means)
        dw = {ds: 100 * np.array([r["vsel_worst"] - r["cls_worst"] for r in rows]) for ds, rows in per_ds.items()}
        if dw:
            res["worst_recall vsel-tdgs_cls"] = float(np.mean([v.mean() for v in dw.values()]))
        g = res.get("vsel-tdgs_cls")
        if g:
            res["gate"] = ("PASS" if g["dataset_mean"] >= 0.5 and g["n_pos"] >= 3
                           and res["vsel-meanC"]["dataset_mean"] > 0 else "FAIL")
        rep[name] = dict(summary=res, per_dataset=per_ds)
        print(f"\n== {name}  C={C} seeds={seeds}")
        print("\n".join(lines))
        for k, v in res.items():
            print(f"  {k}: {v}")
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
