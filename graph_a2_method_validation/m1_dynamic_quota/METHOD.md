# M1: safe normalized dynamic quota

M1 retains the official global Graph-A2 kernel and greedy coverage objective but
allows different class budgets. It first guarantees
`q_floor = max(1, floor(floor_ratio * b_equal))`, then allocates remaining units
to the class with the largest normalized next coverage gain.

`self_optimum` normalization divides each class marginal by its own coverage at
the equal-budget reference. `mass` divides by total class client mass. Setting
`floor_ratio=1.0` bypasses M1 allocation and calls the official greedy solver,
which is the exact baseline-equivalence control.

