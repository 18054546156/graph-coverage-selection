#!/usr/bin/env python
"""Round 3 gates, exactly as preregistered (report/r3_prereg_20260928.md §4 and
Amendment 4). Reads harvests only; trains nothing, never reads val BA of a
trained model (the val-selected single view uses the zero-training train->val
kNN BA logged by mv_select.py in r3/report/<ds>_r0.02_r3.json).

Reference tdgs_cls = the H100 copy (P4-e, runs/r3hw_20260928), because every R3
arm trained on H100 and deterministic training is not bit-identical across GPU
types. The A100 copy (round 2) is reported as a sensitivity row.

  G-R2          tdgs_cls - hpc_cls           n=5   mean >= +0.5, >= 4/5 same sign, excl tissue > +0.3
  G-MV stage 1  mv_rob - tdgs_cls            n=3   mean >= +0.5, >= 3/5 pos, excl tissue >= +0.3,
                                                   and mv_rob - valsel_view >= 0      -> GO
                                                   mean <= 0                         -> STOP
                                                   otherwise                         -> PIVOT checks
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

DSL = ["bloodmnist", "organamnist", "organsmnist", "pathmnist", "tissuemnist"]
VIEW_ARM = {"uni": "tdgs_cls", "dinov2": "cls_dinov2", "clip": "cls_clip"}


def load(path, tag=""):
    T = defaultdict(dict)
    for r in json.load(open(path)):
        if abs(r["ratio"] - 0.02) < 1e-9:
            T[(r["ds"], r["arm"] + tag)][r["seed"]] = (100 * r["ba"], 100 * r["worst"])
    return T


def hboot(d, B=20000, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(d)
    out = np.empty(B)
    for b in range(B):
        ks = rng.choice(len(keys), len(keys))
        out[b] = np.mean([rng.choice(d[keys[k]], len(d[keys[k]])).mean() for k in ks])
    return [float(x) for x in np.percentile(out, [2.5, 97.5])]


def contrast(T, a, b, seeds, idx=0):
    d = {}
    for ds in DSL:
        if all(s in T.get((ds, a), {}) and s in T.get((ds, b), {}) for s in seeds):
            d[ds] = np.array([T[(ds, a)][s][idx] - T[(ds, b)][s][idx] for s in seeds])
    if not d:
        return None
    m = {ds: float(v.mean()) for ds, v in d.items()}
    return dict(a=a, b=b, seeds=seeds, n_ds=len(m),
                dataset_mean=float(np.mean(list(m.values()))), ci95=hboot(d),
                n_pos=sum(v > 0 for v in m.values()),
                excl_tissue=float(np.mean([v for k, v in m.items() if k != "tissuemnist"])),
                per_ds=m)


def fmt(c):
    if c is None:
        return "  (missing cells)"
    per = " ".join(f"{k[:6]}={v:+.2f}" for k, v in c["per_ds"].items())
    return (f"  {c['a']} - {c['b']}  n={len(c['seeds'])}: {c['dataset_mean']:+.2f} "
            f"[{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]  pos {c['n_pos']}/{c['n_ds']}  "
            f"exTis {c['excl_tissue']:+.2f}\n    {per}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r3", required=True, help="harvest of runs/r3_20260928")
    ap.add_argument("--r3hw", required=True, help="harvest of runs/r3hw_20260928 (tdgs_cls on H100)")
    ap.add_argument("--round1", required=True, help="tdgs_round1_harvest.json (A100, 2%% seeds 42-44)")
    ap.add_argument("--round2", required=True, help="round2_harvest.json (A100, 2%% seeds 45-46 of tdgs_cls/graph_a2)")
    ap.add_argument("--r3-report", required=True, help="dir with <ds>_r0.02_r3.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    T = load(a.r3)
    for k, v in load(a.r3hw).items():                 # tdgs_cls (H100) = reference
        T[k] = v
    for f in (a.round1, a.round2):                    # tdgs_cls@A100, graph_a2@A100, ...
        for k, v in load(f, "@A100").items():
            T[k].update(v)

    # val-selected single view: argmax train->val kNN BA (zero training)
    valsel = {}
    for ds in DSL:
        rep = json.load(open(os.path.join(a.r3_report, f"{ds}_r0.02_r3.json")))
        vk = {v: rep["views"][v]["val_knn_ba"] for v in VIEW_ARM}
        v = max(vk, key=vk.get)
        valsel[ds] = dict(view=v, arm=VIEW_ARM[v], val_knn_ba=vk)
        T[(ds, "valsel_view")] = T.get((ds, VIEW_ARM[v]), {})

    S5, S3 = [42, 43, 44, 45, 46], [42, 43, 44]
    R = {"valsel": valsel}
    lines = []

    # hardware noise: same arm, same selection, same seed, H100 vs A100
    hw = []
    for ds in DSL:
        for s in S5:
            h, g = T.get((ds, "tdgs_cls"), {}).get(s), T.get((ds, "tdgs_cls@A100"), {}).get(s)
            if h and g:
                hw.append(dict(ds=ds, seed=s, h100=h[0], a100=g[0], d=h[0] - g[0]))
    dd = np.array([r["d"] for r in hw])
    R["hardware_noise"] = dict(n=len(hw), mean_signed=float(dd.mean()), mean_abs=float(np.abs(dd).mean()),
                               sd=float(dd.std(ddof=1)), max_abs=float(np.abs(dd).max()), cells=hw)
    lines.append(f"== hardware noise tdgs_cls H100 - A100 (n={len(hw)}): mean {dd.mean():+.2f}  "
                 f"mean|d| {np.abs(dd).mean():.2f}  sd {dd.std(ddof=1):.2f}  max|d| {np.abs(dd).max():.2f} pp")

    # G-R2
    g = contrast(T, "tdgs_cls", "hpc_cls", S5)
    g_s = contrast(T, "tdgs_cls@A100", "hpc_cls", S5)
    g["verdict"] = ("PASS" if g["dataset_mean"] >= 0.5 and g["n_pos"] >= 4
                    and g["excl_tissue"] > 0.3 else "FAIL")
    R["G_R2"] = dict(primary=g, sensitivity_A100=g_s,
                     worst_recall=contrast(T, "tdgs_cls", "hpc_cls", S5, idx=1))
    lines += ["== G-R2 (tdgs_cls[H100] - hpc_cls)", fmt(g), f"  verdict: {g['verdict']}",
              "  sensitivity (A100 tdgs_cls):", fmt(g_s),
              "  worst-class recall (secondary):", fmt(R["G_R2"]["worst_recall"])]

    # G-MV stage 1
    p = contrast(T, "mv_rob", "tdgs_cls", S3)
    v = contrast(T, "mv_rob", "valsel_view", S3)
    if p["dataset_mean"] >= 0.5 and p["n_pos"] >= 3 and p["excl_tissue"] >= 0.3 and v["dataset_mean"] >= 0:
        verdict = "GO"
    elif p["dataset_mean"] <= 0:
        verdict = "STOP"
    else:
        verdict = "PIVOT"
    piv = dict(valsel_minus_cls=contrast(T, "valsel_view", "tdgs_cls", S3),
               mv_mean_minus_mv_rob=contrast(T, "mv_mean", "mv_rob", S3),
               cls_cat_minus_mv_rob=contrast(T, "cls_cat", "mv_rob", S3))
    R["G_MV_stage1"] = dict(primary=p, mv_rob_minus_valsel=v, verdict=verdict, pivot_checks=piv,
                            sensitivity_A100=contrast(T, "mv_rob", "tdgs_cls@A100", S3),
                            worst_recall=contrast(T, "mv_rob", "tdgs_cls", S3, idx=1))
    lines += ["== G-MV stage 1 (mv_rob - tdgs_cls[H100], seeds 42-44)", fmt(p),
              "  mv_rob - val-selected single view:", fmt(v), f"  verdict: {verdict}",
              "  pivot checks:"] + [fmt(c) for c in piv.values()] + [
              "  sensitivity (A100 tdgs_cls):", fmt(R["G_MV_stage1"]["sensitivity_A100"]),
              "  worst-class recall (secondary):", fmt(R["G_MV_stage1"]["worst_recall"])]

    # descriptive: every R3 arm vs tdgs_cls[H100] and vs graph_a2 (A100; hardware-confounded)
    desc = {}
    lines.append("== descriptive, seeds 42-44 (vs tdgs_cls[H100]; vs graph_a2@A100 is hardware-confounded)")
    for arm in ["hpc_cls", "cls_dinov2", "cls_clip", "cls_cat", "mv_mean", "mv_rob", "a2_dinov2", "a2_clip"]:
        c1, c2 = contrast(T, arm, "tdgs_cls", S3), contrast(T, arm, "graph_a2@A100", S3)
        desc[arm] = dict(vs_tdgs_cls=c1, vs_graph_a2_A100=c2)
        lines.append(f"  {arm:11s} vs cls {c1['dataset_mean']:+.2f} ({c1['n_pos']}/5)   "
                     f"vs a2@A100 {c2['dataset_mean']:+.2f} ({c2['n_pos']}/5)")
    R["descriptive"] = desc
    lines.append("== val-selected view per dataset: " +
                 ", ".join(f"{ds[:6]}={valsel[ds]['view']}" for ds in DSL))

    print("\n".join(lines))
    json.dump(R, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
