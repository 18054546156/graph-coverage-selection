# D8 Novelty Dossier

## Proposed delta

At a fixed, externally supplied per-class budget and with unchanged equal-weight downstream training, select actual source-training examples by approximately minimizing the classwise (W_1) distance from the complete source-training empirical feature distribution to the uniform empirical measure on the selected examples. Current implementation uses entropic Sinkhorn, weighted geometric-median updates, and projection to distinct observed samples.

## Claims to attack

1. Is full empirical equal-mass OT selection already present in Zhang et al. (2023) or another representative-subsampling paper?
2. Is the proposed objective meaningfully different from FDMat's class-center approximation and existing class-balanced FDMat variants?
3. Does any difference survive comparison with Graph-A2, Herding, balanced Random, and a source-defined subtype-proportional control?
4. Is the selected set's equal-mass distribution a plausible mechanism for downstream clean BA, or only an auxiliary feature-space score?
5. Does the proposed approximate solver optimize the displayed objective closely enough to justify the method name? Compare Sinkhorn diagnostics to exact small-problem OT and report sensitivity to initialization/epsilon.
6. Does the clean-only five-dataset setting support a useful finding, and what additional locked evaluation would be required before claiming unseen-domain robustness?

## Prior art and verification state

- FDMat, AAAI 2024, DOI `10.1609/aaai.v38i8.28771`: local paper PDF and implementation reviewed; direct threat.
- Zhang et al., JCGS 2023, DOI `10.1080/10618600.2022.2084404`: Crossref metadata verified; full algorithm comparison pending.
- Herding, ICML 2009, DOI `10.1145/1553374.1553517`: bibliographic metadata verified; strong mean-matching baseline.
- GraphCov, arXiv `2606.22002`: local official implementation README and pinned code available; use as the project's direct comparator. Current publisher status should be independently confirmed.
- `verify_papers.py` returned `verify_pending` on its Semantic Scholar title checks. Prior Auto Research trace says arXiv search ran and Semantic Scholar was HTTP 429. No exhaustive recent-work search or independent cross-model verdict is claimed here.

## Required verdict standard

Do not call D8 novel solely because it is class-conditional, uses MedMNIST, or is compared with Graph-A2. A defensible delta must survive the named OT-subsampling and FDMat comparisons, and must establish either a distinct algorithmic result or a non-obvious, reproducible finding about equal-weight training under extreme compression. Until then the correct status is **PROCEED WITH CAUTION / NOVELTY UNCONFIRMED**.
