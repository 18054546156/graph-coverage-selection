"""Analyse the functional screen: which functional predicts accuracy at fixed budget?

Four analyses, in increasing order of how much they are worth believing:

A. **Across the whole library.** Dominated by pathological selections (impure, hard,
   sparse). Any functional correlated with "is this garbage" wins here. Reported for
   completeness; it does not tell you how to beat herding.

B. **Competitive regime, defined by covariate.** Restricted to families a sensible
   method could emit (random / herding / kcenter / kmedoids / dense) and their small
   perturbations. This conditions on a pre-outcome covariate, so it is not a collider.

C. **Within family.** Centre BA and the functional inside each (dataset, block,
   family, m). This removes the method identity entirely, and is the exact analogue
   of the within-cell test that killed covering distortion on results/ladder.json.
   A functional that survives here has an effect of its own rather than being a
   marker for which generator produced the set.

D. **Cross-block replication.** Sign and significance must agree on both independent
   pool/audit splits, per dataset. This project's recurring failure mode is findings
   that do not replicate, so nothing is reported as real without it.

The outcome-conditioned version of B (keep selections within X pp of the block's
best) is computed too but flagged: selecting on the endpoint induces range
restriction and collider bias, so it can manufacture correlations that are not there.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
from pathlib import Path

import numpy as np

# Families a real method could plausibly emit. The last four are published coreset
# baselines (Toneva forgetting, Paul EL2N, Zheng CCS, Xia Moderate) and belong here
# for the same reason herding does. forget_low / aum_high / aum_low are deliberately
# NOT competitive: they are the ablation directions of those methods, kept in the
# library so the perturbation ladders span the dynamics functionals, but no one
# publishes "select the easiest points".
COMPETITIVE_FAMILIES = ("random", "herding", "kcenter", "kmedoids", "dense",
                        "forget_high", "el2n_high", "ccs", "moderate")
# Every anchor whose unperturbed BA is a baseline this project has to beat.
BASELINE_FAMILIES = COMPETITIVE_FAMILIES + ("forget_low", "aum_low", "aum_high")

# Pure hard-example selection is published at retention fractions of 30-90%, and
# both papers report it degrading at aggressive pruning -- which is the regime here
# (25/class). So there is a literature-based, outcome-independent case that no tuned
# method would emit these sets at this budget, and keeping them in the competitive
# regime would smuggle back the same "avoid garbage" signal that regime exists to
# exclude. Rather than decide it from this screen's own endpoint (that is the
# collider bias analysis E warns about), the decision cell is reported BOTH ways.
HARD_EXAMPLE_FAMILIES = ("forget_high", "el2n_high")
ENDPOINT = "ba_equal"


def load(paths) -> list[dict]:
    rows = []
    for p in paths:
        for line in Path(p).open(encoding="utf-8"):
            rows.append(json.loads(line))
    return rows


def pearson_t(x: np.ndarray, y: np.ndarray):
    if len(x) < 5 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan"), float("nan"), len(x)
    r = float(np.corrcoef(x, y)[0, 1])
    r = max(min(r, 1 - 1e-12), -1 + 1e-12)
    t = r * math.sqrt((len(x) - 2) / (1 - r * r))
    return r, t, len(x)


def centred(rows, keys, func, endpoint=None):
    """Centre functional and endpoint within every group defined by `keys`."""
    endpoint = endpoint or ENDPOINT
    g = collections.defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r)
    fx, fy = [], []
    for _, v in g.items():
        if len(v) < 3:
            continue
        a = np.array([v_[func] for v_ in v], dtype=float)
        b = np.array([100.0 * v_[endpoint] for v_ in v], dtype=float)
        if not np.isfinite(a).all() or a.std() < 1e-12:
            continue
        fx += list(a - a.mean())
        fy += list(b - b.mean())
    return np.array(fx), np.array(fy)


def report(title, rows, keys, funcs, note=""):
    print(f"\n{'='*78}\n{title}   (n={len(rows)})")
    if note:
        print(f"  {note}")
    print(f"{'functional':<16}{'n':>7}{'r':>9}{'t':>9}   per-dataset r (replicated?)")
    out = {}
    for f in funcs:
        x, y = centred(rows, keys, f)
        r, t, n = pearson_t(x, y)
        per = []
        for ds in sorted({row["dataset"] for row in rows}):
            sub = [row for row in rows if row["dataset"] == ds]
            rs = []
            for blk in sorted({row["block"] for row in sub}):
                xb, yb = centred([row for row in sub if row["block"] == blk], keys, f)
                rb, tb, nb = pearson_t(xb, yb)
                rs.append((rb, tb))
            ok = (len(rs) == 2 and all(np.isfinite(v[0]) for v in rs)
                  and rs[0][0] * rs[1][0] > 0 and all(abs(v[1]) > 2 for v in rs))
            per.append(f"{ds[:6]}:{np.mean([v[0] for v in rs]):+.2f}{'*' if ok else ' '}")
        out[f] = {"r": r, "t": t, "n": n}
        print(f"{f[2:]:<16}{n:>7}{r:>+9.3f}{t:>+9.2f}   {' '.join(per)}")
    print("  * = same sign AND |t|>2 on BOTH independent blocks for that dataset")
    return out


def baselines(rows):
    """Every published baseline's unperturbed BA against the random distribution.

    This is the leaderboard the project has to beat, and until the dynamics anchors
    were added it did not exist: the screen carried only geometric methods, so
    "we beat the baselines" could only ever have meant "we beat herding".

    z is (anchor - mean random) / sd(random) within the same (dataset, block), so a
    method has to clear the spread of random draws at the same budget, not just the
    mean. Blocks are averaged only after z-scoring, since their audit sets differ.
    """
    print(f"\n{'='*78}\nBASELINE LEADERBOARD -- unperturbed anchors vs the random draws")
    cells = collections.defaultdict(dict)
    rnd = {}
    for r in rows:
        if r["m"] != 0:
            continue
        key = (r["dataset"], r["block"])
        if r["family"] == "random":
            rnd.setdefault(key, []).append(100 * r[ENDPOINT])
        elif r["family"] in BASELINE_FAMILIES:
            cells[key][r["family"]] = 100 * r[ENDPOINT]

    fams = [f for f in BASELINE_FAMILIES if f != "random"]
    datasets = sorted({k[0] for k in cells})
    print(f"{'method':<13}" + "".join(f"{d[:6]:>9}" for d in datasets) + f"{'mean z':>9}")
    out = {}
    for f in fams:
        zs, cols = [], []
        for d in datasets:
            per = []
            for (ds, blk), v in cells.items():
                if ds != d or f not in v or len(rnd.get((ds, blk), [])) < 5:
                    continue
                a = np.array(rnd[(ds, blk)])
                per.append((v[f] - a.mean()) / max(a.std(), 1e-9))
            cols.append(f"{np.mean(per):>+9.2f}" if per else f"{'--':>9}")
            zs += per
        out[f] = float(np.mean(zs)) if zs else float("nan")
        print(f"{f:<13}" + "".join(cols) + f"{out[f]:>+9.2f}")
    print("  units = sd of the random draws in the same (dataset, block).")
    print("  A method that is not clearly >0 here does not beat random at this budget.")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--competitive-pp", type=float, default=2.0)
    p.add_argument("--endpoint", default="ba_equal",
                   help="'ba_equal' = linear-probe screen, 'ba_real' = real-training screen")
    args = p.parse_args()

    global ENDPOINT
    ENDPOINT = args.endpoint

    rows = load(args.results)
    rows = [r for r in rows if r.get(ENDPOINT) is not None]
    funcs = sorted(k for k in rows[0] if k.startswith("F_"))
    print(f"loaded {len(rows)} selections | datasets "
          f"{sorted({r['dataset'] for r in rows})} | families {sorted({r['family'] for r in rows})}")

    for ds in sorted({r["dataset"] for r in rows}):
        b = np.array([100 * r[ENDPOINT] for r in rows if r["dataset"] == ds])
        comp = np.array([100 * r[ENDPOINT] for r in rows
                         if r["dataset"] == ds and r["family"] in COMPETITIVE_FAMILIES])
        print(f"  {ds:<13} BA all {b.min():5.2f}-{b.max():5.2f}  "
              f"competitive-family {comp.min():5.2f}-{comp.max():5.2f} (sd {comp.std():.2f})")

    leaderboard = baselines(rows)

    A = report("A. WHOLE LIBRARY -- includes pathological sets; 'avoid garbage' dominates",
               rows, ["dataset", "block"], funcs)

    comp = [r for r in rows if r["family"] in COMPETITIVE_FAMILIES]
    B = report("B. COMPETITIVE REGIME (families a real method could emit) -- pre-outcome covariate",
               comp, ["dataset", "block"], funcs)

    C = report("C. WITHIN FAMILY x PERTURBATION LEVEL -- method identity removed",
               rows, ["dataset", "block", "family", "m"], funcs,
               note="the analogue of the test that killed covering distortion")

    Ccomp = report("D. WITHIN FAMILY, COMPETITIVE FAMILIES ONLY -- the decision-relevant cell",
                   comp, ["dataset", "block", "family", "m"], funcs)

    soft = [r for r in comp if r["family"] not in HARD_EXAMPLE_FAMILIES]
    Dsoft = report("D2. AS D, MINUS PURE HARD-EXAMPLE FAMILIES (see HARD_EXAMPLE_FAMILIES)",
                   soft, ["dataset", "block", "family", "m"], funcs,
                   note="a functional that only survives in D is probably marking "
                        "'this budget is too small for hard-example selection'")

    # outcome-conditioned, reported but not trusted
    keep = []
    for (ds, blk), v in itertools.groupby(
            sorted(rows, key=lambda r: (r["dataset"], r["block"])),
            key=lambda r: (r["dataset"], r["block"])):
        v = list(v)
        best = max(100 * r[ENDPOINT] for r in v)
        keep += [r for r in v if 100 * r[ENDPOINT] >= best - args.competitive_pp]
    E = report(f"E. [BIASED] selections within {args.competitive_pp}pp of the block best",
               keep, ["dataset", "block"], funcs,
               note="conditions on the OUTCOME -- range restriction and collider bias; "
                    "do not use this to pick a functional")

    print(f"\n{'='*78}\nSURVIVORS: |t|>3 in BOTH D and D2 (the decision cell, either definition)")
    surv = [f for f in funcs if abs(Ccomp[f]["t"]) > 3 and abs(Dsoft[f]["t"]) > 3
            and Ccomp[f]["r"] * Dsoft[f]["r"] > 0]
    print("  " + (", ".join(f[2:] for f in surv) if surv else
                  "NONE -- no functional of the selected set predicts accuracy among "
                  "competitive selections at fixed budget"))

    if args.output:
        args.output.write_text(json.dumps(
            {"baseline_leaderboard_z": leaderboard,
             "whole_library": A, "competitive": B, "within_family": C,
             "within_family_competitive": Ccomp,
             "within_family_competitive_no_hard": Dsoft,
             "outcome_conditioned_biased": E,
             "survivors": surv}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
