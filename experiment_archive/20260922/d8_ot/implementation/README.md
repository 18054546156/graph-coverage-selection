# D8: Equal-Mass Class-Conditional OT Selector

This is an isolated Auto Research implementation artifact for the D8 proposal. The detailed prior-art comparison is in `LITERATURE_REVIEW.md`; the mathematical objective, heuristic, and clean-benchmark migration recipe are in `METHOD_AND_MIGRATION.md`; novelty questions are in `NOVELTY_DOSSIER.md`; citations are in `references.bib`.

The prototype reads a frozen embedding matrix and source-train labels, applies classwise uniform-mass entropic OT, updates support points by weighted geometric medians, then projects supports to distinct observed rows. The output contains original row indices and solver diagnostics.

The D8 benchmark now runs through an isolated copy of the clean-only pipeline. It does not modify the active Table 1 source or outputs, and the D8 benchmark uses clean train and clean test only. The algorithm is an approximate heuristic, not an exact minimizer of the discrete Wasserstein subset objective. FDMat (AAAI 2024) and OT representative subsampling (Zhang et al., 2023) are direct prior-art comparisons; no novelty or SOTA claim is established.

## Input

An `.npz` file with arrays `embeddings` shaped `[N,D]` and `labels` shaped `[N]` or `[N,1]`. Inputs must be frozen source-training features only. The caller supplies the audited `budget_per_class`; the selector does not invent quota rounding.

## Local synthetic check

```powershell
python -m unittest discover -s d8_class_conditional_ot_20260922/tests -v
python d8_class_conditional_ot_20260922/select_d8.py --help
```

## Current execution

- Full OrganSMNIST selection smoke: job `33459`, complete.
- One-epoch pipeline integration smoke: job `33466`, complete with checkpoint, clean predictions, metrics, and `run_complete.json`.
- Formal 5-dataset × 8-seed comparison: array `33468`, running with five A100 workers.
- See `RUN_METADATA.json` and `AUTORESEARCH_TRACE.md` for configurations, code/data hashes, progress snapshots, and startup-fix history.

## Research status

**Novelty unconfirmed; no SOTA claim yet.** FDMat (AAAI 2024) and OT representative subsampling (Zhang et al., 2023) are direct prior-art threats. The implementation is a heuristic approximation. The active clean-only benchmark source/results remain unchanged; D8 training outputs are isolated under the D8 project root.
