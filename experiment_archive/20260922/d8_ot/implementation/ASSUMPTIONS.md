# Assumption Ledger — D8 equal-mass class-conditional OT
<!-- ASK mode: never -->

| ID | Under-determined by the request | Chosen | Class | Source |
|----|---------------------------------|--------|-------|--------|
| A-001 | Feature representation and scaling are unspecified | Consume frozen source-train embeddings; L2-normalize each vector inside the selector | semantic | default |
| A-002 | Class quotas and rounding are unspecified | Require the caller to pass the already-audited equal per-class budget; do not compute quotas here | semantic | default |
| A-003 | The objective is specified but no discrete optimizer is | Use log-domain entropic Sinkhorn coupling, weighted geometric-median updates, and deterministic projection to distinct observed feature rows | semantic | default |
| A-004 | Entropic approximation tolerance/scale is unspecified | Expose epsilon and iteration limits; prototype defaults are epsilon=0.05 in normalized Euclidean feature space and 60 Sinkhorn iterations | semantic | default |
| A-005 | Seed semantics for a deterministic selector are unspecified | Use the selection seed only for initialization; same input/config/seed must return identical row indices | semantic | default |
| A-006 | How to judge an early implementation is unspecified | Unit/synthetic tests establish interface and numerical behavior only; clean test performance is not used for development or tuning | semantic | default |
| A-007 | Whether to integrate or submit jobs now is unspecified by the method proposal and conflicts with the active benchmark resource gate | Keep this prototype isolated; no pipeline edits, synchronization, or Slurm submission in this turn | interface | default |

## Notes

- **A-003** is deliberately a heuristic: after each entropic coupling, each selected support point is updated toward a weighted geometric median and snapped back to a distinct source observation. It does not certify a global minimum of the discrete subset objective.
- **A-004** is a provisional numerical default, not a tuned research hyperparameter. It must be frozen before any confirmatory run and varied only on source-train/source-validation data under a predeclared protocol.
- The active clean-only benchmark is left untouched. Existing clean test metrics may be reported as a frozen comparison later, but must not be repeatedly queried to tune D8.
