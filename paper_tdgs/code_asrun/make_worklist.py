#!/usr/bin/env python
"""Generate the round-2 worklist, ordered by scientific priority.

ORDER MATTERS. tdgs_pack.slurm shards the list by a global stride, so all 15
jobs start near the front. If the allocation dies or is stopped early, what is
finished is a PREFIX of this list -- so the list is ordered so that the prefix
is always the most decisive experiment available, not an arbitrary slice.

Priority order, and what each block decides:

  P0-a  a2_perclass                      the only control that can kill the
        (5 ds x 3 seeds = 15)            round-1 positive: if the author's own
                                         per-class scope reproduces +1.76pp,
                                         tdgs_cls is inside compare_global.py

  P0-b  tdgs_mask, tdgs_wcls             completes the 2x2 factorial whose other
        (5 ds x 2 arms x 3 = 30)         two cells (graph_a2, tdgs_cls) are
                                         already trained -> both main effects
                                         plus the interaction

  P1-a  tdgs_cls_perm                    the positive's own matched null. Every
        (5 ds x 3 = 15)                  prior positive here that skipped its
                                         null later died.

  P1-b  tdgs_lam1                        is the global term load-bearing, or is
        (5 ds x 3 = 15)                  pure class-compatible coverage enough?

  P1-c  graph_a2, tdgs_cls @ seeds 45,46  lifts the headline contrast from n=3
        (5 ds x 2 arms x 2 = 20)         to n=5 at 2%. Needed because outside
                                         tissuemnist the effect is +0.81pp while
                                         the per-cell MDE at n=3 is 2.4-17.6pp.

  P1-d  graph_a2, tdgs_cls @ ratio 0.05   the second budget. The paper protocol
        (5 ds x 2 arms x 5 seeds = 50)   is 2% AND 5%; a 2%-only result is half
                                         a table. Run 5 seeds directly.

  Total 145 cells. At ~30 min/cell and 30 concurrent slots (15 jobs x 2 GPUs)
  this is roughly 2.5-3 h wall clock, tissuemnist being the slow tail.

ROUND 2b (added 2026-09-28, user-approved; written to a SEPARATE worklist):

  P2-a  a2_perclass, tdgs_cls_perm @ seeds 45,46   G1 and G2 controls to n=5,
        (5 ds x 2 arms x 2 = 20)                   so the controls are not the
                                                   weakest link next to an n=5
                                                   headline
  P2-b  tdgs_mask, tdgs_wcls @ seeds 45,46         the 2x2 factorial to n=5
        (5 ds x 2 arms x 2 = 20)

  Total 40 cells. tdgs_lam1 s45/46 deliberately NOT included: lam1 already
  loses to tdgs_cls (-1.11pp at n=9), more seeds cannot change the decision.
  Selections are seed-independent (td_run_one.sh reads ${DS}_r${R}_${ARM}_s42.npy
  for every seed), so round 2b needs NO staging.

  Generate with:  make_worklist.py --only P2 --out .../work/round2b.txt
  and regenerate round 2 (if ever) with:  --only P0 P1 --out .../work/round2.txt

NOT included, deliberately:
  - (superseded 2026-09-28 by TABLE 1 below) the 8 baselines were first taken
    from the Table-1 archive (400/400 on seeds 0-4) with a footnote.
  - ratio 0.05 for the round-2 control arms: gated on the 2% factorial. If the
    mask effect is zero at 2% there is nothing to confirm at 5%.

TABLE 1 on the author seed grid (added 2026-09-28, user decision: "table1 要一致"):

  P3-a/b/c  7 baselines x 5 ds x {2%, 5%} x seeds 42-46 = 350 cells, run by
            t1_run_one.sh into their own tree with the archive's own pipeline
            and config (only the seed differs). Selections are staged first by
            t1_select.slurm (selection_seed == training_seed, as in the author's
            code), because random/fps/el2n/forgetting/eva depend on the seed.
            graph_a2 is excluded: its selection is seed-independent and it is
            already trained on 42-46 in rounds 1/2.
  P3-a comes first because organs/blood/organa seed 42 at 2% also exist in the
  archive: those cells must come out bit-identical, which proves the new tree
  runs the archive's harness and not a drifted copy.

  Generate with:  make_worklist.py --only P3 --out .../work/table1_s4246.txt
"""
import argparse

DS = ["pathmnist", "organamnist", "bloodmnist", "organsmnist", "tissuemnist"]
T1_METHODS = ["random", "el2n_top", "forgetting", "eva", "facility", "fps", "herding"]

BLOCKS = [
    ("P0-a  a2_perclass: author per-class scope control",
     [("a2_perclass", 0.02, s) for s in (42, 43, 44)]),
    ("P0-b  2x2 factorial completion (mask x weight)",
     [(arm, 0.02, s) for arm in ("tdgs_mask", "tdgs_wcls")
      for s in (42, 43, 44)]),
    ("P1-a  tdgs_cls_perm: the positive's matched null",
     [("tdgs_cls_perm", 0.02, s) for s in (42, 43, 44)]),
    ("P1-b  tdgs_lam1: is the global blend load-bearing",
     [("tdgs_lam1", 0.02, s) for s in (42, 43, 44)]),
    ("P1-c  headline contrast to n=5 seeds at 2%",
     [(arm, 0.02, s) for arm in ("graph_a2", "tdgs_cls") for s in (45, 46)]),
    ("P1-d  second budget: ratio 0.05, 5 seeds",
     [(arm, 0.05, s) for arm in ("graph_a2", "tdgs_cls")
      for s in (42, 43, 44, 45, 46)]),
    # ---- round 2b: never mix into round2.txt, use --only P2 ----
    ("P2-a  round 2b: G1/G2 controls to n=5 at 2%",
     [(arm, 0.02, s) for arm in ("a2_perclass", "tdgs_cls_perm")
      for s in (45, 46)]),
    ("P2-b  round 2b: 2x2 factorial to n=5 at 2%",
     [(arm, 0.02, s) for arm in ("tdgs_mask", "tdgs_wcls")
      for s in (45, 46)]),
    # ---- Table-1 baselines on the author seed grid: --only P3, runner
    # t1_run_one.sh (the <arm> column is the archive pipeline's method name) ----
    ("P3-a  table1: 7 baselines @2% seed 42 (archive overlap = identity check)",
     [(m, 0.02, 42) for m in T1_METHODS]),
    ("P3-b  table1: 7 baselines @2% seeds 43-46",
     [(m, 0.02, s) for s in (43, 44, 45, 46) for m in T1_METHODS]),
    ("P3-c  table1: 7 baselines @5% seeds 42-46",
     [(m, 0.05, s) for s in (42, 43, 44, 45, 46) for m in T1_METHODS]),
    # ---- round 3 (09-28, report/r3_prereg_20260928.md): --only P4, own tree
    # runs/r3_20260928, selections r3/sel. Stage 1 = seeds 42-44 at 2%. ----
    ("P4-a  r3: matched hybrid-perclass control (hpc_cls) @2% seeds 42-46",
     [("hpc_cls", 0.02, s) for s in (42, 43, 44, 45, 46)]),
    ("P4-b  r3: multi-encoder objectives @2% seeds 42-44",
     [(arm, 0.02, s) for arm in ("mv_rob", "mv_mean") for s in (42, 43, 44)]),
    ("P4-c  r3: single-view and concatenation references @2% seeds 42-44",
     [(arm, 0.02, s) for arm in ("cls_dinov2", "cls_clip", "cls_cat")
      for s in (42, 43, 44)]),
    ("P4-d  r3: Graph-A2 on the other encoders @2% seeds 42-44",
     [(arm, 0.02, s) for arm in ("a2_dinov2", "a2_clip") for s in (42, 43, 44)]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None,
                    help="block prefixes to keep, e.g. P0")
    a = ap.parse_args()

    lines, n = [], 0
    for title, cells in BLOCKS:
        if a.only and not any(title.startswith(p) for p in a.only):
            continue
        lines.append(f"# ===== {title} =====")
        # dataset-major inside a block so each stride residue sees a mix of
        # cheap (bloodmnist n=11959) and expensive (tissuemnist n=165466) cells
        for arm, ratio, seed in cells:
            for ds in DS:
                lines.append(f"{ds} {ratio} {arm} {seed}")
                n += 1
    with open(a.out, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {a.out}: {n} cells")
    for title, cells in BLOCKS:
        if a.only and not any(title.startswith(p) for p in a.only):
            continue
        print(f"  {len(cells)*len(DS):>4d}  {title}")


if __name__ == "__main__":
    main()
