# M2: training-free relation gate

M2 retains equal class quota and the official global Graph-A2 kernel, then applies
`K_ij <- K_ij * Lambda[y_i, y_j]` before the official greedy solver.

Modes:

- `baseline`: all-one gate, exact original behavior.
- `scalar`: one tunable `cross_credit` for every off-diagonal class pair.
- `same_class`: zero cross-class credit.
- `class_pair`: estimate a symmetric pair gate from graph-mass lift relative to
  class priors, then interpolate with the all-one gate using `relation_strength`.

This is a training-free coreset adaptation of compatibility modeling; it does
not claim to be the first class-pair compatibility matrix.

