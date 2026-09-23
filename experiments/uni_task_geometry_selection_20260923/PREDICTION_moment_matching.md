# Pre-registered prediction: the herding degradation is broken moment matching

Written 2026-09-23, **before** any first-moment error was computed from
`results/run_33659/`. The balanced-accuracy numbers were already known; the
quantity below (`eps_mean`) had never been looked at, so the predicted signs
and orderings are genuinely at risk.

## Mechanism claimed

Herding's objective is `min || (1/k) * sum_{i in S_c} z_i - mu_c ||`: the set it
returns is constructed so that its **unweighted** mean matches the class mean.
Applying Voronoi mass weights destroys exactly that invariant. Random's
unweighted mean is an unbiased but high-variance estimate of `mu_c`, and mass
weighting is importance weighting toward the true density, so it repairs it.
Facility-location / graph_a2 optimise coverage, which is orthogonal to the first
moment, so weighting should do approximately nothing.

Define, per class `c`, on L2-normalised UNI features (the same space herding
operates in):

    mu_c    = mean over the selection POOL points of class c
    eps_c   = || sum_i w_i z_i / sum_i w_i  -  mu_c ||      for i in S_c
    eps_mean = mean over classes of eps_c

with `w_i = 1` for the equal arm and the recorded Voronoi masses for the
weighted arm.

## Predictions

**P1 (sign, per arm).** `delta_eps = eps_mean(voronoi) - eps_mean(equal)` is
**positive for herding** and **negative for random** and
**negative for random_plus_pilot**, on at least 4 of 5 datasets each.

**P2 (the link to accuracy).** Across the 8 arms, Spearman(`delta_eps`,
`delta_BA`) is **negative** on at least 3 of 5 datasets, where `delta_BA` is the
already-known voronoi-minus-equal balanced-accuracy gain.

**P3 (implementation sanity).** `eps_mean(equal)` is **smallest for herding**
among all 8 arms on at least 4 of 5 datasets. This is herding's defining
property. If P3 fails, my reading of what herding does is wrong at the root and
P1/P2 carry no weight regardless of how they come out.

## What refutes this

- P3 fails -> the whole mechanism story is discarded, not patched.
- P1's herding sign is negative or null -> discarded.
- P2's sign is positive -> the first moment is not the channel; look elsewhere.

## Status

This is a mechanism probe on an endpoint that the frozen protocol classifies as
**secondary**. A positive result here does NOT promote the weighting finding to
a headline claim. It earns one thing only: the right to spend CPU on the
out-of-sample replication in `PLAN_weighting_interaction.md`.
