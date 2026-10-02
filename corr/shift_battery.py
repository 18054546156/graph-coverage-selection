"""Shift-robustness metric battery (proposal_shift_robustness_20261002).

Input: one JSON list of corruption rows (dataset, ratio, method, training_seed,
corruption, severity, ba, be, worst_recall, recall_*, ece15, brier, nll, aurc,
risk_at_80, clean_ba_drop, clean_worst_recall_drop) and optionally one JSON list
of clean rows (same keys, corruption == "clean"). Writes a markdown report.

Only seeds where EVERY method has the full 11x5 grid in a (dataset, ratio) cell
are used for that cell, so all comparisons are paired.

Usage:
  python corr/shift_battery.py --corr X_corr.json [--clean X_clean.json] --out report.md
"""
import argparse
import json
from collections import defaultdict
from itertools import combinations
from statistics import mean, pstdev

# RGB (blood, path): EXP = colour/exposure. Grayscale (organA/S, tissue): EXP = intensity
# (brightness, contrast, gamma). STR = everything structural (blur, pixelate, jpeg, noise, artefacts).
EXP = {"brightness_down", "brightness_up", "contrast_down", "contrast_up", "saturate",
       "gamma_corr_up", "gamma_corr_down"}
STR = {"defocus_blur", "motion_blur", "pixelate", "jpeg_compression", "bubble", "stain_deposit",
       "gaussian_blur", "gaussian_noise", "speckle_noise", "impulse_noise", "shot_noise"}
FAMILIES = [("EXP", EXP), ("STR", STR)]
MAIN_SEV = {1, 2, 3}
CALIB = ["ece15", "brier", "nll", "aurc", "risk_at_80"]
ORDER = ["mv_mean", "graph_a2", "herding", "facility", "fps", "random", "eva", "el2n_top", "forgetting"]


def kendall_tau(order_a, order_b):
    pos_a = {m: i for i, m in enumerate(order_a)}
    pos_b = {m: i for i, m in enumerate(order_b)}
    conc = disc = 0
    for x, y in combinations(order_a, 2):
        prod = (pos_a[x] - pos_a[y]) * (pos_b[x] - pos_b[y])
        conc += prod > 0
        disc += prod < 0
    pairs = len(order_a) * (len(order_a) - 1) / 2
    return (conc - disc) / pairs


def ranked(values):
    return sorted(values, key=lambda m: -values[m])


def tail(values, frac):
    """Lower-tail mean (CVaR) over the worst ceil(frac*count) values; p10 = empirical 10% quantile."""
    ordered = sorted(values)
    k = max(1, int(round(frac * len(ordered))))
    return mean(ordered[:k])


def p10(values):
    ordered = sorted(values)
    pos = 0.1 * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def pp(x):
    return "{:+.2f}".format(100 * x)


def pct(x):
    return "{:.1f}".format(100 * x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corr", required=True)
    ap.add_argument("--clean")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Shift-robustness battery")
    args = ap.parse_args()

    corr_rows = json.load(open(args.corr, encoding="utf-8"))
    clean_rows = json.load(open(args.clean, encoding="utf-8")) if args.clean else []

    grid = defaultdict(dict)  # (ds, ratio, method, seed) -> {(corr, sev): row}
    for r in corr_rows:
        key = (r["dataset"], round(float(r["ratio"]), 2), r["method"], int(r["training_seed"]))
        grid[key][(r["corruption"], int(r["severity"]))] = r
    clean = {}
    for r in clean_rows:
        if r.get("corruption", "clean") != "clean":
            continue
        clean[(r["dataset"], round(float(r["ratio"]), 2), r["method"], int(r["training_seed"]))] = r

    unknown = {c for k in grid for (c, _) in grid[k]} - EXP - STR
    if unknown:
        raise RuntimeError("corruptions not assigned to a family: {}".format(sorted(unknown)))
    cells = sorted({(k[0], k[1]) for k in grid})
    usable = {}
    for ds, ratio in cells:
        methods = sorted({k[2] for k in grid if k[:2] == (ds, ratio)},
                         key=lambda m: ORDER.index(m) if m in ORDER else 99)
        seeds = sorted({k[3] for k in grid if k[:2] == (ds, ratio)})
        n_corr = len({c for k in grid if k[0] == ds for (c, _) in grid[k]})
        full = [s for s in seeds if all(len(grid.get((ds, ratio, m, s), {})) == 5 * n_corr for m in methods)]
        usable[(ds, ratio)] = (methods, full)

    def clean_metric(ds, ratio, m, s, field):
        key = (ds, ratio, m, s)
        if key in clean and field in clean[key]:
            return clean[key][field]
        rows = list(grid[key].values())
        if field == "ba":
            return mean(x["ba"] + x["clean_ba_drop"] for x in rows)
        if field == "worst_recall":
            return mean(x["worst_recall"] + x["clean_worst_recall_drop"] for x in rows)
        return None

    def fam(ds, ratio, m, s, family, field="ba", sevs=MAIN_SEV):
        vals = [r[field] for (c, v), r in grid[(ds, ratio, m, s)].items() if c in family and v in sevs]
        return mean(vals)

    out = ["# " + args.title, "",
           "Source: `{}`{}".format(args.corr, " + `{}`".format(args.clean) if args.clean else ""),
           "EXP = " + ", ".join(sorted(EXP)) + "; STR = " + ", ".join(sorted(STR)) + ".",
           "Main metrics average severities 1-3 (4-5 floor out). Only seeds with the full 11x5 grid "
           "for every method in a cell are used, so all comparisons are paired.", ""]
    out.append("| dataset | ratio | methods | seeds used |")
    out.append("|---|---|---|---|")
    for (ds, ratio), (methods, seeds) in usable.items():
        out.append("| {} | {} | {} | {} |".format(ds, ratio, len(methods), seeds))
    out.append("")

    # ---------- 1. diagnostics ----------
    out += ["## 1. Diagnostics (is there a problem?)", "",
            "Rankings use the seed-mean BA per method. tau = Kendall tau(clean ranking, shifted ranking). "
            "Regret = BA(best under shift) - BA(clean winner) under shift.", "",
            "| dataset | ratio | tau_EXP | tau_STR | EXP<STR | #1 clean | #1 EXP | #1 STR | regret EXP (pp) | regret STR (pp) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    gate_tau = gate_flip = 0
    seed_level = []
    for (ds, ratio), (methods, seeds) in usable.items():
        if not seeds:
            continue
        c = {m: mean(clean_metric(ds, ratio, m, s, "ba") for s in seeds) for m in methods}
        e = {m: mean(fam(ds, ratio, m, s, EXP) for s in seeds) for m in methods}
        t = {m: mean(fam(ds, ratio, m, s, STR) for s in seeds) for m in methods}
        tau_e, tau_s = kendall_tau(ranked(c), ranked(e)), kendall_tau(ranked(c), ranked(t))
        win = ranked(c)[0]
        gate_tau += tau_e < tau_s
        gate_flip += ranked(e)[0] != win
        out.append("| {} | {} | {:+.3f} | {:+.3f} | {} | {} | {} | {} | {} | {} |".format(
            ds, ratio, tau_e, tau_s, "yes" if tau_e < tau_s else "no", win, ranked(e)[0], ranked(t)[0],
            pp(max(e.values()) - e[win]), pp(max(t.values()) - t[win])))
        for s in seeds:
            cs = {m: clean_metric(ds, ratio, m, s, "ba") for m in methods}
            es = {m: fam(ds, ratio, m, s, EXP) for m in methods}
            ts = {m: fam(ds, ratio, m, s, STR) for m in methods}
            w = ranked(cs)[0]
            seed_level.append((ds, ratio, s, kendall_tau(ranked(cs), ranked(es)),
                               kendall_tau(ranked(cs), ranked(ts)), w, ranked(es)[0],
                               max(es.values()) - es[w]))
    n_cells = sum(1 for v in usable.values() if v[1])
    out += ["", "**H1 gate (proposal S1):** tau_EXP < tau_STR in {}/{} cells (need >=3/4); "
            "#1 changes clean->EXP in {}/{} cells (need >=2/4) -> **{}**".format(
                gate_tau, n_cells, gate_flip, n_cells,
                "PASS" if gate_tau >= 3 and gate_flip >= 2 else "FAIL"), ""]
    out += ["Per-seed (rankings within one seed):", "",
            "| dataset | ratio | seed | tau_EXP | tau_STR | #1 clean | #1 EXP | regret EXP (pp) |",
            "|---|---|---|---|---|---|---|---|"]
    for row in seed_level:
        out.append("| {} | {} | {} | {:+.3f} | {:+.3f} | {} | {} | {} |".format(
            row[0], row[1], row[2], row[3], row[4], row[5], row[6], pp(row[7])))
    if seed_level:
        out += ["", "Per-seed summary: mean tau_EXP {:+.3f}, mean tau_STR {:+.3f}, EXP<STR {}/{}, "
                "#1 flips {}/{}, mean regret {} pp, regret>0.5pp {}/{}.".format(
                    mean(r[3] for r in seed_level), mean(r[4] for r in seed_level),
                    sum(r[3] < r[4] for r in seed_level), len(seed_level),
                    sum(r[5] != r[6] for r in seed_level), len(seed_level),
                    pp(mean(r[7] for r in seed_level)),
                    sum(r[7] > 0.005 for r in seed_level), len(seed_level))]
    out.append("")

    # ---------- 2. main: BA/BE by family x severity ----------
    out += ["## 2. Main: BA by family x severity (mean +- sd over seeds, %)", "",
            "BE = 1 - BA. rBE not computable (no AlexNet reference checkpoint)."]
    for (ds, ratio), (methods, seeds) in usable.items():
        if not seeds:
            continue
        out += ["", "### {} r={}".format(ds, ratio), "",
                "| method | clean | " + " | ".join("{} s{}".format(f, v) for f, _ in FAMILIES for v in sorted(MAIN_SEV))
                + " | EXP s1-3 BE | STR s1-3 BE | clean_ba_drop EXP |",
                "|---" * (5 + 2 * len(MAIN_SEV)) + "|"]
        for m in methods:
            cells_txt = [pct(mean(clean_metric(ds, ratio, m, s, "ba") for s in seeds))]
            for fname, family in FAMILIES:
                for v in sorted(MAIN_SEV):
                    vals = [fam(ds, ratio, m, s, family, sevs={v}) for s in seeds]
                    cells_txt.append("{}+-{}".format(pct(mean(vals)), pct(pstdev(vals))))
            be_e = mean(1 - fam(ds, ratio, m, s, EXP) for s in seeds)
            be_s = mean(1 - fam(ds, ratio, m, s, STR) for s in seeds)
            drop = mean(fam(ds, ratio, m, s, EXP, "clean_ba_drop") for s in seeds)
            out.append("| {} | {} | {} | {} | {} |".format(m, " | ".join(cells_txt), pct(be_e), pct(be_s), pp(drop)))
    out.append("")

    # ---------- 3. worst class ----------
    out += ["## 3. Worst-class recall (severity 1-3, mean over seeds, %)", ""]
    for (ds, ratio), (methods, seeds) in usable.items():
        if not seeds:
            continue
        any_row = next(iter(grid[(ds, ratio, methods[0], seeds[0])].values()))
        rec_keys = ["recall_{}".format(i) for i in range(20) if "recall_{}".format(i) in any_row]
        out += ["### {} r={}".format(ds, ratio), "",
                "| method | worst clean | worst EXP | worst STR | worst drop EXP (pp) | most-hurt class under EXP |",
                "|---|---|---|---|---|---|"]
        for m in methods:
            wc = mean(clean_metric(ds, ratio, m, s, "worst_recall") for s in seeds)
            we = mean(fam(ds, ratio, m, s, EXP, "worst_recall") for s in seeds)
            ws = mean(fam(ds, ratio, m, s, STR, "worst_recall") for s in seeds)
            drops = {}
            for rk in rec_keys:
                cvals = [clean_metric(ds, ratio, m, s, rk) for s in seeds]
                if None in cvals:
                    continue
                drops[rk] = mean(cvals) - mean(fam(ds, ratio, m, s, EXP, rk) for s in seeds)
            hurt = max(drops, key=drops.get) if drops else None
            hurt_txt = "{} (-{}pp)".format(hurt, pct(drops[hurt])) if hurt else "n/a (no clean per-class)"
            out.append("| {} | {} | {} | {} | {} | {} |".format(m, pct(wc), pct(we), pct(ws), pp(wc - we), hurt_txt))
        out.append("")

    # ---------- 3b. method x shift interaction (pre-registered quantity) ----------
    out += ["## 3b. Method x EXP interaction I(m) (PREREG_s1b_s2_20261002)", "",
            "I(m) = [BA(m) - BA(random)]_EXP - [BA(m) - BA(random)]_clean, severity 1-3, per seed; "
            "mean (t over seeds). Negative = method loses its clean standing under EXP.", ""]
    inter_methods = [m for m in ORDER if m != "random"]
    out.append("| dataset | ratio | " + " | ".join(inter_methods) + " |")
    out.append("|---" * (2 + len(inter_methods)) + "|")
    for (ds, ratio), (methods, seeds) in usable.items():
        if not seeds or "random" not in methods:
            continue
        cells_txt = []
        for m in inter_methods:
            if m not in methods:
                cells_txt.append("-")
                continue
            vals = [(fam(ds, ratio, m, s, EXP) - fam(ds, ratio, "random", s, EXP))
                    - (clean_metric(ds, ratio, m, s, "ba") - clean_metric(ds, ratio, "random", s, "ba"))
                    for s in seeds]
            sd = pstdev(vals) * (len(vals) / max(len(vals) - 1, 1)) ** 0.5
            t = mean(vals) / (sd / len(vals) ** 0.5) if sd > 1e-12 else float("inf")
            cells_txt.append("{} (t {:+.1f})".format(pp(mean(vals)), t))
        out.append("| {} | {} | {} |".format(ds, ratio, " | ".join(cells_txt)))
    out.append("")

    # ---------- 4. risk: seed spread, p10, CVaR ----------
    out += ["## 4. Risk across seeds (H2)", "",
            "Paired gap = method - random, same seed. sd ratio = sd(gap under EXP) / sd(gap clean); >1 means "
            "shift amplifies the seed lottery. p10 = 10% quantile, CVaR20 = mean of worst 20% of seeds. "
            "With <=5 seeds these tails are near the minimum; read as format until more seeds exist.", ""]
    for (ds, ratio), (methods, seeds) in usable.items():
        if not seeds or "random" not in methods:
            continue
        out += ["### {} r={}".format(ds, ratio), "",
                "| method | gap clean mean | gap EXP mean | sd clean | sd EXP | sd ratio | p10 EXP | CVaR20 EXP | gap STR mean |",
                "|---|---|---|---|---|---|---|---|---|"]
        mean_rank, tail_rank = {}, {}
        for m in methods:
            ge = [fam(ds, ratio, m, s, EXP) - fam(ds, ratio, "random", s, EXP) for s in seeds]
            mean_rank[m] = mean([fam(ds, ratio, m, s, EXP) for s in seeds])
            tail_rank[m] = tail([fam(ds, ratio, m, s, EXP) for s in seeds], 0.2)
            if m == "random":
                continue
            gc = [clean_metric(ds, ratio, m, s, "ba") - clean_metric(ds, ratio, "random", s, "ba") for s in seeds]
            gs = [fam(ds, ratio, m, s, STR) - fam(ds, ratio, "random", s, STR) for s in seeds]
            sdc, sde = pstdev(gc), pstdev(ge)
            out.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                m, pp(mean(gc)), pp(mean(ge)), pct(sdc), pct(sde),
                "{:.2f}".format(sde / sdc) if sdc > 1e-9 else "inf", pp(p10(ge)), pp(tail(ge, 0.2)), pp(mean(gs))))
        out += ["", "tau(EXP mean-BA ranking, EXP CVaR20 ranking) = {:+.3f}; #1 by mean = {}, #1 by CVaR20 = {}".format(
            kendall_tau(ranked(mean_rank), ranked(tail_rank)), ranked(mean_rank)[0], ranked(tail_rank)[0]), ""]

    # ---------- 5. calibration appendix ----------
    out += ["## 5. Appendix: calibration under shift (severity 1-3, mean over seeds)", ""]
    for (ds, ratio), (methods, seeds) in usable.items():
        if not seeds:
            continue
        out += ["### {} r={}".format(ds, ratio), "",
                "| method | " + " | ".join("{} clean/EXP/STR".format(f) for f in CALIB) + " |",
                "|---" * (1 + len(CALIB)) + "|"]
        for m in methods:
            parts = []
            for f in CALIB:
                cv = [clean_metric(ds, ratio, m, s, f) for s in seeds]
                ctxt = "{:.3f}".format(mean(cv)) if None not in cv else "-"
                parts.append("{} / {:.3f} / {:.3f}".format(
                    ctxt, mean(fam(ds, ratio, m, s, EXP, f) for s in seeds),
                    mean(fam(ds, ratio, m, s, STR, f) for s in seeds)))
            out.append("| {} | {} |".format(m, " | ".join(parts)))
        out.append("")

    open(args.out, "w", encoding="utf-8").write(chr(10).join(out) + chr(10))
    print("wrote " + args.out)
    for line in out:
        if line.startswith("**H1") or line.startswith("Per-seed summary"):
            print(line)


if __name__ == "__main__":
    main()
