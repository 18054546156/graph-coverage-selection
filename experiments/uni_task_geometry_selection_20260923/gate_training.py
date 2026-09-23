"""Frozen decision rule for the real-training factorial (protocol_training.json).

Written before any training result existed. Editing the thresholds below is a
protocol amendment and must be recorded as one.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from pathlib import Path

TREATMENT = "voronoi_within"
CONTROL = "equal"
PERMUTED = "permuted_within"
ENDPOINT = "balanced_accuracy"

CONFIRM_PP = 1.0
REFUTE_PP = 0.5
MIN_DATASETS = 2
MAX_NEGATIVE_DATASETS = 2
PERMUTED_MAX_FRACTION = 0.5


def load(paths):
    cells = collections.defaultdict(dict)
    for path in paths:
        for line in Path(path).open(encoding="utf-8"):
            r = json.loads(line)
            key = (r["dataset"], r["replicate"], r["train_seed"], r["arm"])
            cells[key][r["weighting"]] = r[ENDPOINT]
    return cells


def did_units(cells, treatment):
    """Per (dataset, replicate, seed): DiD = delta(random) - delta(herding)."""
    out = collections.defaultdict(list)
    units = {(d, rep, s) for (d, rep, s, _) in cells}
    for d, rep, s in sorted(units):
        try:
            r, h = cells[(d, rep, s, "random")], cells[(d, rep, s, "herding")]
            out[d].append(100.0 * ((r[treatment] - r[CONTROL]) - (h[treatment] - h[CONTROL])))
        except KeyError:
            continue
    return out


def summarise(units):
    rows = {}
    for d, v in units.items():
        se = st.stdev(v) / len(v) ** 0.5 if len(v) > 1 else float("nan")
        rows[d] = {"n": len(v), "mean_pp": st.mean(v), "se_pp": se,
                   "n_positive": sum(1 for x in v if x > 0)}
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    cells = load(args.results)
    real = summarise(did_units(cells, TREATMENT))
    perm = summarise(did_units(cells, PERMUTED))
    if not real:
        raise SystemExit("no complete random/herding pairs found")

    pooled = st.mean(v["mean_pp"] for v in real.values())
    pooled_perm = st.mean(v["mean_pp"] for v in perm.values()) if perm else float("nan")
    at_threshold = sum(1 for v in real.values() if v["mean_pp"] >= CONFIRM_PP)
    negative = sum(1 for v in real.values() if v["mean_pp"] < 0)
    control_clean = (perm and abs(pooled_perm) < PERMUTED_MAX_FRACTION * abs(pooled))

    if pooled >= CONFIRM_PP and at_threshold >= MIN_DATASETS and control_clean:
        verdict, action = "CONFIRMED", "the interaction survives real training; write it up with the permuted control as the headline comparison"
    elif pooled < REFUTE_PP or negative >= MAX_NEGATIVE_DATASETS or (perm and not control_clean):
        verdict, action = "REFUTED", "the probe effect does not survive real training; report as a negative and close the branch"
    else:
        verdict, action = "INCONCLUSIVE", "extend replicates 10 -> 20 and re-apply once; a second INCONCLUSIVE resolves to REFUTED"

    print("%-14s %4s %14s %9s %10s" % ("dataset", "n", "DiD (pp)", "se", "units>0"))
    for d in sorted(real):
        v = real[d]
        print("%-14s %4d %14.3f %9.3f %7d/%d" % (d, v["n"], v["mean_pp"], v["se_pp"],
                                                 v["n_positive"], v["n"]))
    print("\npermuted-weight control (should be near zero):")
    for d in sorted(perm):
        v = perm[d]
        print("%-14s %4d %14.3f %9.3f" % (d, v["n"], v["mean_pp"], v["se_pp"]))
    print(f"\npooled DiD {pooled:+.3f} pp | permuted {pooled_perm:+.3f} pp | "
          f"{at_threshold}/{len(real)} datasets >= {CONFIRM_PP} pp | {negative} negative")
    print(f"VERDICT: {verdict}\n  -> {action}")

    if args.output:
        args.output.write_text(json.dumps({
            "per_dataset": real, "per_dataset_permuted": perm,
            "pooled_did_pp": pooled, "pooled_permuted_did_pp": pooled_perm,
            "verdict": verdict, "action": action,
        }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
