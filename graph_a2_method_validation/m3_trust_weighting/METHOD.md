# M3: purity and local-support trust weighting

M3 retains the original Graph-A2 kernel, equal quota, and solver. It changes only
the value of covered clients in the facility objective.

- `purity`: `w_i = p_i ^ purity_exponent`.
- `adaptive`: `w_i = p_i ^ (purity_exponent * (1 - support_i))`, where support is
  the within-class percentile of same-class graph mass.

The optional weight floor prevents complete suppression. Setting
`purity_exponent=0` produces all-one trust weights and exact baseline behavior.

