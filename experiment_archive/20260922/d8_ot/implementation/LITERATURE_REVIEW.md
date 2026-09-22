# D8 Literature Review and Evidence Boundaries

Review date: 2026-09-22. This is a targeted review, not an exhaustive novelty search. The installed `verify_papers.py` returned `verify_pending` for title-only lookups because its Semantic Scholar path was transiently unavailable. Crossref title/DOI metadata and the project's locally archived PDF/code were used where noted. The earlier Auto Research run records an arXiv search and an HTTP 429 from Semantic Scholar. Novelty therefore remains **unconfirmed**.

## Closest work

| Work | What it does | Relation to D8 | Evidence/status |
|---|---|---|---|
| Xiao et al., **Feature Distribution Matching by Optimal Transport for Effective and Robust Coreset Selection**, AAAI 2024 | Uses learned feature representations, class-wise feature processing, a class-center approximation, an OT/Sinkhorn score, and selects low-cost training examples. Evaluates PathMNIST as well as CIFAR-10/100 and Tiny-ImageNet. The paper also studies noisy/imbalanced settings. | **Direct prior-art threat.** OT + feature-distribution coreset + medical image data are already present. D8 cannot claim any of those ingredients as new. The candidate difference is matching the full empirical class distribution with the final equal-weight selected support, instead of ranking against a class-center proxy. Exact author-code variant and balancing flag must be pinned in the comparison. | Full text archived at `reliability_medmnistc_ab/research/20260915_robust_selection/sources/fdmat_pdf_text.txt`; implementation audit at `.../sources/fdmat_method.py`; Crossref resolves DOI `10.1609/aaai.v38i8.28771`. [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28771) · [code](https://github.com/successhaha/FDMat) |
| Zhang et al., **An Optimal Transport Approach for Selecting a Representative Subsample with Application in Efficient Kernel Density Estimation**, *Journal of Computational and Graphical Statistics* 32(1), 2023, 329–339 | Studies OT-based selection of a representative subsample for kernel density estimation. | **Closest objective-level threat.** It means “choose a subsample that represents a distribution under OT” is not a new general principle. The D8 paper must compare the exact support/mass constraints, algorithm, and downstream supervised-learning setting; the title or application difference alone is insufficient. | Crossref-confirmed DOI `10.1080/10618600.2022.2084404`; exact theorem/algorithm-level comparison remains pending full-text audit. [Paper](https://doi.org/10.1080/10618600.2022.2084404) |
| Welling, **Herding Dynamical Weights to Learn**, ICML 2009 | Greedily constructs a representative sequence that matches feature/kernel means. | Strong distribution-representation baseline. It is related through moment/mean matching, but does not in general minimize the full Wasserstein distance between empirical measures. | Bibliographic record in the archived FDMat references; Crossref DOI `10.1145/1553374.1553517`. [Paper](https://proceedings.mlr.press/v5/welling09a.html) |
| Rustamov et al., **One-Shot Data Selection for Medical Image Classification via Graph Coverage**, 2026 preprint / project reports MICCAI 2026 | Builds a kNN graph over frozen foundation-model embeddings and uses the two-term heat/diffusion kernel (A_{sym}+A_{sym}^2) in class-budgeted greedy facility location. | The project's direct benchmark comparator. It maximizes graph coverage; it does not directly constrain the selected set's empirical class-conditional mass to match the source training distribution. | Local official implementation README cites arXiv `2606.22002`; code snapshot is `official-graph-coverage-8cf757a/` at the project's recorded commit. Venue/status should be checked against the current publisher record before publication. [arXiv](https://arxiv.org/abs/2606.22002) · [code](https://github.com/zahiriddin-rustamov/graph-coverage-selection) |

## Mechanism distinction

For class (c), with source-train embeddings (z_i), D8 defines

\[
P_c=\frac1{n_c}\sum_{i:y_i=c}\delta_{z_i},\qquad
Q_{S,c}=\frac1{q_c}\sum_{j\in S_c}\delta_{z_j},\qquad
J(S)=\frac1C\sum_c W_1(P_c,Q_{S,c}).
\]

For a fixed selected set, the transport plan has row mass (1/n_c), column mass (1/q_c), and ground cost (\|z_i-z_j\|_2). This explicitly makes every selected example carry the same mass, matching ordinary unweighted downstream training. It asks for a distributional match, not only broad coverage or closeness to a single class centroid.

However, the code in this folder is **not an exact solver of the combinatorial objective above**. It uses entropic Sinkhorn plans, weighted-geometric-median updates, and projection to distinct observed examples. It is a proposed heuristic whose objective and convergence behavior need empirical and algorithmic validation. Its `entropic_transport_cost` diagnostic is the ground-cost expectation under the regularized plan, not the exact unregularized (W_1).

## What can and cannot be claimed

- Supported: Graph-A2 and D8 optimize different stated selection objectives.
- Supported: FDMat and OT representative subsampling make a broad “first OT coreset” claim untenable.
- Candidate, not established: replacing FDMat's center approximation with equal-mass matching to the full empirical class distribution may better preserve within-class modes when the final subset is trained uniformly.
- Not established: D8 is novel, beats Graph-A2, improves clean BA, or improves unseen-domain/corruption robustness.
- A clean-only Table 1 comparison can test clean performance. It cannot establish corruption robustness; that needs a later locked, source-only-designed shift evaluation.

## Verification follow-up

Before a paper claim, obtain and compare the full Zhang et al. method and supplement; pin and run the FDMat author-code configuration; verify GraphCov's publisher/venue status; search 2024–2026 papers on Wasserstein subset selection and equal-weight coreset training. Semantic Scholar is currently rate-limited, and the verifier's title-only checks remain pending. Do not write “first” or “novel” until this is closed.
