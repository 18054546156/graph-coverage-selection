# PathMNIST 5% fixed-selection paired A/B

Status: complete. Paired array `29860`; dependent summary `29866`.

The two frozen inputs passed file SHA, index SHA, strict-order, uniqueness, bounds, and equal-quota checks. Both arms used the same downstream seeds `42-46`, the same training entry point, and the final epoch official-test balanced accuracy as the primary endpoint.

Result: Random with one frozen seed-42 selection averaged `87.7058+/-0.8362%`; Graph-A2 averaged `85.8074+/-1.5881%`. Graph-A2 minus Random was `-1.8984 pp`, with Random winning 4/5 paired seeds. The fixed-selection comparison therefore does not reproduce the paper's PathMNIST 5% ranking; changing the Random selection seed is not a sufficient explanation for the earlier reversal.

`best_balanced_accuracy` is audit-only and was not used for the decision or aggregate.
