#!/usr/bin/env python
"""E5 (report/e5_prereg_20261003.md, revision 4: dev seeds s42-46 only) harvest + preregistered analysis.

  python e5_eval.py --cluster   -> (on luhpc, in $S/code) harvest runs/acs_e5_20261003 -> e5_cells.json
  python code/e5_eval.py test   -> §4 tests from results/acs/e5_cells.json + the dev cells (acs_eval.load())

New cells are paired with the existing dev cells of the same row, seed and GPU type (H100: breast/derma/OCT;
A100: tissue and the E5c rows). A new cell on the wrong GPU type is excluded and listed.
"""
import collections
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TREE = "/mnt/prj01/hgrp-1502-5TB/tdgs_shared/runs/acs_e5_20261003"
LOGS = "/mnt/prj01/hgrp-1502-5TB/tdgs_shared/acs/logs"
S5 = [42, 43, 44, 45, 46]
T0 = [(d, r) for d in ("dermamnist", "octmnist", "tissuemnist") for r in (0.02, 0.05)]   # rule tau = 0
F1ROWS = [("breastmnist", 0.05)] + T0                                                    # F1 acts (tau < 1)
RULE = {("breastmnist", 0.05): "acs_q25_t050"} | {k: "acs_q25_t000" for k in T0}
E5C = {("organsmnist", 0.005): 6, ("bloodmnist", 0.005): 7, ("organsmnist", 0.01): 12,
       ("bloodmnist", 0.01): 14, ("organamnist", 0.005): 15, ("organamnist", 0.01): 31}
GPU = {"breastmnist": "H100", "dermamnist": "H100", "octmnist": "H100", "tissuemnist": "A100",
       "organsmnist": "A100", "bloodmnist": "A100", "organamnist": "A100"}
B_PERM, B_BOOT = 100_000, 10_000


def harvest():
    out, seen = [], set()
    for f in sorted(glob.glob(f"{TREE}/formal/ratio_*/*/*/seed_*/results/**/metrics.jsonl", recursive=True)):
        m = re.search(r"ratio_([0-9.]+)/(\w+)/(\w+)/seed_(\d+)/", f)
        key = (m[2], float(m[1]), m[3], int(m[4]))
        if key in seen:
            continue
        j = [json.loads(l) for l in open(f) if l.strip()]
        j = [x for x in j if x.get("corruption", "clean") == "clean"]
        if j:
            seen.add(key)
            out.append(dict(ds=key[0], ratio=key[1], arm=key[2], seed=key[3],
                            ba=100 * j[-1]["ba"], worst=100 * j[-1]["worst_recall"]))
    gpu = {}
    for lf in glob.glob(f"{LOGS}/acse5_*.out"):
        txt = open(lf).read()
        g = "H100" if "H100" in txt else ("A100" if "A100" in txt else "?")
        for key in re.findall(r"ok (\w+_r[0-9.]+_\w+_s\d+)", txt):
            gpu[key] = g
    for x in out:
        x["gpu"] = gpu.get(f"{x['ds']}_r{x['ratio']}_{x['arm']}_s{x['seed']}", "?")
    json.dump(out, open("e5_cells.json", "w"), indent=0)
    print(f"wrote e5_cells.json ({len(out)} cells; gpu {collections.Counter(x['gpu'] for x in out)})")


def load():
    import acs_eval
    C, _, _ = acs_eval.load()
    new = json.load(open("results/acs/e5_cells.json"))
    bad = [x for x in new if x["gpu"] != GPU[x["ds"]]]
    for x in new:
        if x not in bad:
            C[(x["ds"], round(float(x["ratio"]), 4), x["arm"], int(x["seed"]))] = dict(ba=x["ba"], worst=x["worst"])
    return C, new, bad


def perm_p(D, rng):
    """One-sided sign-flip permutation p for T = mean over rows of the seed-mean difference.
    D: rows x seeds paired differences; each (row, seed) pair flips independently."""
    D = np.asarray(D, float)
    T = D.mean()
    flips = rng.choice([-1.0, 1.0], size=(B_PERM,) + D.shape)
    Ts = (flips * D).mean(axis=(1, 2))
    return T, (1 + (Ts >= T).sum()) / (1 + B_PERM)


def boot_ci(D, rng):
    D = np.asarray(D, float)
    idx = rng.integers(0, D.shape[1], size=(B_BOOT, D.shape[1]))
    bs = np.stack([D[:, i].mean(axis=1) for i in idx]).mean(axis=1)   # resample seeds within each row
    return np.percentile(bs, [2.5, 97.5])


def holm(ps):
    o = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def test():
    C, new, bad = load()
    rng = np.random.default_rng(20261003)
    print(f"new cells {len(new)} (expected 275); excluded for wrong GPU: {len(bad)} "
          f"{[(x['ds'], x['ratio'], x['arm'], x['seed'], x['gpu']) for x in bad][:10]}")
    have = lambda d, r, a: all((d, r, a, s) in C for s in S5)
    V = lambda d, r, a, ep="ba": np.array([C[(d, r, a, s)][ep] for s in S5])

    def contrast(rows, f, ep="ba"):
        """rows x seeds matrix of a contrast; None if any cell is missing (row then reported missing)."""
        out, miss = [], []
        for d, r in rows:
            try:
                out.append(f(d, r, ep))
            except KeyError as e:
                miss.append((d, r, str(e)))
        return np.array(out), miss

    fam = {
        "H1 C2-F1: ACS - ACS_l00 (6 tau=0 rows)":
            (T0, lambda d, r, ep: V(d, r, RULE[(d, r)], ep) - V(d, r, "acs_q25_t000_l00", ep)),
        "H2 P-int: (cls - cls_l00) - (ACS - ACS_l00) (6 tau=0 rows)":
            (T0, lambda d, r, ep: (V(d, r, "cls", ep) - V(d, r, "acs_q00_t000_l00", ep))
             - (V(d, r, "acs_q25_t000", ep) - V(d, r, "acs_q25_t000_l00", ep))),
        "H3 E5b: ACS - ACS_wcls (7 F1 rows)":
            (F1ROWS, lambda d, r, ep: V(d, r, RULE[(d, r)], ep) - V(d, r, RULE[(d, r)] + "_wcls", ep)),
    }
    print("\n## §4.1 primary family (one-sided sign-flip permutation, Holm over 3), BA")
    res = []
    for name, (rows, f) in fam.items():
        D, miss = contrast(rows, f)
        if miss or len(D) != len(rows):
            print(f"  {name}: MISSING {miss} -> not tested")
            res.append((name, np.nan, 1.0, None, D, rows))
            continue
        T, p = perm_p(D, rng)
        res.append((name, T, p, boot_ci(D, rng), D, rows))
    adj = holm([x[2] for x in res])
    for (name, T, p, ci, D, rows), pa in zip(res, adj):
        if ci is None:
            continue
        rm = D.mean(axis=1)
        print(f"  {name}\n    effect {T:+.2f}pp  95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]  p {p:.4f}  Holm {pa:.4f}  "
              f"{'PASS' if pa < .05 else 'not significant'}  | rows > 0: {(rm > 0).sum()}/{len(rm)}")
        print("    per row: " + "  ".join(f"{d[:5]}{int(r * 100)} {m:+.2f}" for (d, r), m in zip(rows, rm)))

    print("\n## §4.2 effect + 95% CI only (BA unless stated)")
    ci_only = {
        "P-F2only: ACS_l00 - cls_l00 (q25 vs q0 at lam=0; 6 rows; predicted > 0)":
            (T0, lambda d, r, ep: V(d, r, "acs_q25_t000_l00", ep) - V(d, r, "acs_q00_t000_l00", ep)),
        "mask-only - ACS (7 rows; predicted ~ 0)":
            (F1ROWS, lambda d, r, ep: V(d, r, RULE[(d, r)] + "_mask", ep) - V(d, r, RULE[(d, r)], ep)),
        "lam=1 - ACS (7 rows; landscape, no prediction of sign)":
            (F1ROWS, lambda d, r, ep: V(d, r, RULE[(d, r)] + "_l100", ep) - V(d, r, RULE[(d, r)], ep)),
        "worst recall, H1 contrast": (T0, fam["H1 C2-F1: ACS - ACS_l00 (6 tau=0 rows)"][1], "worst"),
        "worst recall, H3 contrast": (F1ROWS, fam["H3 E5b: ACS - ACS_wcls (7 F1 rows)"][1], "worst"),
    }
    for name, spec in ci_only.items():
        rows, f = spec[0], spec[1]
        ep = spec[2] if len(spec) > 2 else "ba"
        D, miss = contrast(rows, f, ep)
        if miss or len(D) != len(rows):
            print(f"  {name}: MISSING {miss}")
            continue
        ci = boot_ci(D, rng)
        rm = D.mean(axis=1)
        print(f"  {name}: {D.mean():+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]  rows > 0 {(rm > 0).sum()}/{len(rm)}  | "
              + " ".join(f"{d[:5]}{int(r * 100)} {m:+.2f}" for (d, r), m in zip(rows, rm)))

    print("\n## §4.2 breast 5% (tau=.5): 2x2x2 q x tau x lam, seed-mean BA / worst")
    d, r = "breastmnist", 0.05
    for q in ("q00", "q25"):
        for t in ("t000", "t050"):
            base = "cls" if (q, t) == ("q00", "t000") else f"acs_{q}_{t}"
            cells = []
            for a in (base, f"acs_{q}_{t}_l00"):
                cells.append(f"{a:18s} {V(d, r, a).mean():6.2f}/{V(d, r, a, 'worst').mean():6.2f}" if have(d, r, a)
                             else f"{a:18s}   --")
            print("  " + "   ".join(cells))
    if have(d, r, "acs_q25_t050") and have(d, r, "acs_q25_t050_l00"):
        dd = V(d, r, "acs_q25_t050") - V(d, r, "acs_q25_t050_l00")
        ci = boot_ci(dd[None, :], rng)
        print(f"  F1@transition ACS - ACS_l00 = {dd.mean():+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] (predicted >= 0, small)")

    print("\n## §4.3 E5c switch point at budgets the rule never saw (q=.25), seed-mean BA")
    tau_arm = {0: "acs_q25_t000", .5: "acs_q25_t050", 1: "acs_q25_t100"}
    hit, gaps, bpcs = 0, [], []
    import acs_eval
    for (d, r), bpc in sorted(E5C.items(), key=lambda x: x[1]):
        if not all(have(d, r, a) for a in tau_arm.values()):
            print(f"  {d[:6]} {r}: MISSING")
            continue
        m = {t: V(d, r, a).mean() for t, a in tau_arm.items()}
        rule_t = acs_eval.tau(bpc, 10)
        best = max(m, key=m.get)
        hit += best == rule_t
        gaps.append(m[1] - m[0]); bpcs.append(bpc)
        print(f"  {d[:6]} {r:<5} bpc {bpc:3d} rule tau {rule_t:<3}  t000 {m[0]:6.2f} t050 {m[.5]:6.2f} t100 {m[1]:6.2f}"
              f"  best {best:<3} {'HIT' if best == rule_t else 'miss'}  t100-t000 {m[1] - m[0]:+.2f}")
    if len(gaps) == len(E5C):
        rho = stats.spearmanr(bpcs, gaps).correlation
        print(f"  P-cut: rule tau best in {hit}/6 (criterion >= 4), Spearman(bpc, t100 - t000) = {rho:+.2f} "
              f"(criterion < 0) -> {'PASS' if hit >= 4 and rho < 0 else 'FAIL'}")
        dev = [(acs_eval.BPC[k], V(*k, "acs_q25_t100").mean() - V(*k, "acs_q25_t000").mean())
               for k in [(d, r) for d in acs_eval.HIGH for r in acs_eval.RATIOS]]
        allb = bpcs + [b for b, _ in dev]
        allg = gaps + [g for _, g in dev]
        print(f"  descriptive, E5c 6 + dev 10 rows: Spearman(bpc, t100 - t000) = {stats.spearmanr(allb, allg).correlation:+.2f}; "
              f"sign of t100 - t000 for bpc <= 10: {sum(g > 0 for b, g in zip(allb, allg) if b <= 10)}/"
              f"{sum(b <= 10 for b in allb)} positive, bpc >= 20: {sum(g < 0 for b, g in zip(allb, allg) if b >= 20)}/"
              f"{sum(b >= 20 for b in allb)} negative")

    print("\n## §4.4 determinism side-check: acs_q00_t000_l00 vs our earlier class-quota Graph-A2 cells (same set?)")
    for d, r in T0:
        old = "a2_uni" if d != "tissuemnist" else "graph_a2"
        if have(d, r, "acs_q00_t000_l00") and have(d, r, old):
            print(f"  {d[:6]} {r}: l00 {V(d, r, 'acs_q00_t000_l00').mean():6.2f} vs {old} {V(d, r, old).mean():6.2f} "
                  f"(meaningful only if the e5 report shows Jaccard 1.0 to ref_a2_quota_ours)")


if __name__ == "__main__":
    {"--cluster": harvest, "test": test}[sys.argv[1]]()
