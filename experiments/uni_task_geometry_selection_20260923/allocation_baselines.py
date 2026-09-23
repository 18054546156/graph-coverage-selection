"""Direct marginal-distortion allocation baselines.

This module is deliberately independent of the frozen-UNI pilot runner.  It
implements the empirical comparator for the later regional-budget study:
measure how much adding the next representative reduces held-out regional
covering distortion, then allocate a fixed total budget by those measured
marginal gains.

The caller must keep calibration points separate from the points used for
final selection and source-validation evaluation.  No model labels or target
test data are required by this geometry-only baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np


@dataclass(frozen=True)
class RegionCurve:
    """Empirical distortion curve for one region.

    ``centers[k]`` contains local indices of the k selected representatives.
    ``distortion[k]`` is measured on the region's calibration points and is
    therefore the quantity used for the direct allocation comparator.

    ``demand_source`` records which split the curve was measured on. It exists
    so that an oracle curve -- one measured on the evaluation set -- cannot be
    fed to a deployable allocator by accident. This project has already shipped
    two silent-leak bugs of exactly this shape, and neither was caught by any
    downstream gate.
    """

    region_id: str
    centers: tuple[tuple[int, ...], ...]
    distortion: tuple[float, ...]
    marginal_gain: tuple[float, ...]
    demand_source: str = "calibration"


def _pairwise_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    return np.sqrt(np.maximum(aa + bb - 2.0 * (a @ b.T), 0.0))


def covering_distortion(
    demand: np.ndarray,
    representatives: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Return weighted mean nearest-representative distance."""
    demand = np.asarray(demand, dtype=np.float32)
    representatives = np.asarray(representatives, dtype=np.float32)
    if len(representatives) == 0:
        return float("inf")
    nearest = _pairwise_l2(demand, representatives).min(axis=1)
    if weights is None:
        return float(nearest.mean())
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) != len(nearest) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be nonnegative and match demand rows")
    return float(np.average(nearest, weights=weights))


def greedy_region_curve(
    selection_features: np.ndarray,
    calibration_features: np.ndarray,
    max_representatives: int,
    region_id: str,
    calibration_weights: np.ndarray | None = None,
    forced_local: np.ndarray | None = None,
    demand_source: str = "calibration",
) -> RegionCurve:
    """Build a nested greedy facility curve for one region.

    Centers are chosen from ``selection_features``.  The direct marginal
    gains are evaluated on the disjoint ``calibration_features`` set.  This
    is intentionally a greedy empirical oracle, not a fitted dimension law.
    """
    selection_features = np.asarray(selection_features, dtype=np.float32)
    calibration_features = np.asarray(calibration_features, dtype=np.float32)
    if len(selection_features) == 0 or len(calibration_features) == 0:
        raise ValueError("selection and calibration regions must be non-empty")
    if demand_source not in ("calibration", "evaluation"):
        raise ValueError("demand_source must be 'calibration' or 'evaluation'")
    k = min(int(max_representatives), len(selection_features))
    forced = np.asarray(forced_local if forced_local is not None else [], dtype=np.int64)
    if len(np.unique(forced)) != len(forced) or np.any(forced < 0) or np.any(forced >= len(selection_features)):
        raise ValueError("forced_local must contain unique valid local indices")
    if len(forced) > k:
        raise ValueError("forced_local exceeds max_representatives")

    distances = _pairwise_l2(calibration_features, selection_features)
    nearest = np.full(len(calibration_features), np.inf, dtype=np.float32)
    chosen: list[int] = []
    centers_by_k: list[tuple[int, ...]] = [tuple()]
    curve: list[float] = [float("inf")]
    for idx in forced.tolist():
        chosen.append(int(idx))
        nearest = np.minimum(nearest, distances[:, idx])
        centers_by_k.append(tuple(chosen))
        if calibration_weights is None:
            curve.append(float(nearest.mean()))
        else:
            curve.append(float(np.average(nearest, weights=calibration_weights)))

    # Add all non-forced centers by training-side coverage gain.  The demand
    # side is the calibration set, so the curve is evaluated out of sample.
    while len(chosen) < k:
        eligible = np.ones(len(selection_features), dtype=bool)
        eligible[np.asarray(chosen, dtype=np.int64)] = False
        gains = np.maximum(nearest[:, None] - distances, 0.0)
        if calibration_weights is None:
            scores = gains.sum(axis=0)
        else:
            scores = (gains * np.asarray(calibration_weights)[:, None]).sum(axis=0)
        scores[~eligible] = -np.inf
        pick = int(np.argmax(scores))
        if not np.isfinite(scores[pick]):
            break
        chosen.append(pick)
        nearest = np.minimum(nearest, distances[:, pick])
        centers_by_k.append(tuple(chosen))
        if calibration_weights is None:
            value = float(nearest.mean())
        else:
            value = float(np.average(nearest, weights=calibration_weights))
        curve.append(value)

    marginal = [float("nan")]
    for left, right in zip(curve[:-1], curve[1:]):
        marginal.append(float(left - right))
    return RegionCurve(
        region_id=str(region_id),
        centers=tuple(centers_by_k),
        distortion=tuple(curve),
        marginal_gain=tuple(marginal),
        demand_source=demand_source,
    )


def allocate_by_direct_marginal(
    curves: dict[str, RegionCurve],
    total_budget: int,
    minimum_per_region: int = 1,
    maximum_per_region: int | None = None,
    region_weights: dict[str, float] | None = None,
    allow_oracle: bool = False,
) -> dict[str, int]:
    """Allocate a fixed budget by measured next-example distortion reduction.

    The priority queue implements the empirical greedy solution to the
    separable diminishing-return allocation problem.  ``region_weights`` can
    express a pre-registered population mass weight; it must not be tuned on
    final validation performance.

    Curves whose demand side is the evaluation set are rejected unless
    ``allow_oracle`` is set -- see :func:`allocate_oracle`.
    """
    if total_budget < 0 or minimum_per_region < 1:
        raise ValueError("total_budget must be nonnegative and minimum_per_region must be at least 1")
    leaked = sorted(n for n, c in curves.items() if c.demand_source != "calibration")
    if leaked and not allow_oracle:
        raise ValueError(
            f"regions {leaked} carry evaluation-measured curves; this allocator would leak "
            "the evaluation set into a deployable allocation. Use allocate_oracle() if an "
            "upper bound is what you actually want."
        )
    if not curves:
        if total_budget:
            raise ValueError("cannot allocate a positive budget to no regions")
        return {}
    names = sorted(curves)
    allocations = {name: 0 for name in names}
    caps = {}
    for name, curve in curves.items():
        available = len(curve.centers) - 1
        cap = available if maximum_per_region is None else min(available, maximum_per_region)
        if cap < minimum_per_region:
            raise ValueError(f"region {name} cannot satisfy minimum budget")
        caps[name] = cap

    if total_budget < minimum_per_region * len(names):
        raise ValueError("total_budget is below the regional minimum floor")
    if total_budget > sum(caps.values()):
        raise ValueError("total_budget exceeds available regional curve budget")

    queue: list[tuple[float, str, int]] = []

    def push_next(name: str) -> None:
        current = allocations[name]
        next_k = current + 1
        if next_k <= caps[name] and next_k < len(curves[name].marginal_gain):
            raw = curves[name].marginal_gain[next_k]
            weight = 1.0 if region_weights is None else float(region_weights.get(name, 1.0))
            if weight < 0:
                raise ValueError("region weights must be nonnegative")
            heapq.heappush(queue, (-weight * raw, name, next_k))

    for name in names:
        while allocations[name] < minimum_per_region:
            allocations[name] += 1
        push_next(name)

    remaining = total_budget - sum(allocations.values())
    while remaining:
        if not queue:
            raise ValueError("regional curves cannot supply the requested budget")
        _, name, expected_next = heapq.heappop(queue)
        if allocations[name] + 1 != expected_next:
            continue
        allocations[name] += 1
        remaining -= 1
        push_next(name)
    return allocations


def allocate_oracle(
    curves: dict[str, RegionCurve],
    total_budget: int,
    minimum_per_region: int = 1,
    maximum_per_region: int | None = None,
    region_weights: dict[str, float] | None = None,
) -> dict[str, int]:
    """Upper bound on what any allocator could achieve (decision H).

    Identical greedy, but the curves must have been measured on the evaluation
    set itself.  This is deliberately not a method: it cannot be deployed,
    because it reads the answer.  Its only job is to answer the question that
    comes *before* "which allocator is better" -- namely how much headroom
    allocation has at all.

    If mass / dimension-law / direct-marginal all land inside noise, this tells
    you whether that means "allocation does not matter here" (oracle is also
    flat) or "all three allocators are bad" (oracle is far above them).  Without
    it, a three-way tie is uninterpretable.

    Report it as a dashed ceiling line, never as a competing arm.
    """
    wrong = sorted(n for n, c in curves.items() if c.demand_source != "evaluation")
    if wrong:
        raise ValueError(
            f"regions {wrong} were measured on '{curves[wrong[0]].demand_source}', not the "
            "evaluation set. An oracle bound built on calibration curves is not an oracle; "
            "build the curves with greedy_region_curve(..., demand_source='evaluation')."
        )
    return allocate_by_direct_marginal(
        curves,
        total_budget,
        minimum_per_region=minimum_per_region,
        maximum_per_region=maximum_per_region,
        region_weights=region_weights,
        allow_oracle=True,
    )


def summarize_curve(curve: RegionCurve) -> list[dict[str, float | int | str]]:
    """Convert a curve to JSON/CSV-friendly records."""
    rows = []
    for k, (distortion, gain) in enumerate(zip(curve.distortion, curve.marginal_gain)):
        rows.append({
            "region_id": curve.region_id,
            "demand_source": curve.demand_source,
            "n_representatives": k,
            "distortion": float(distortion),
            "marginal_gain": float(gain) if np.isfinite(gain) else None,
        })
    return rows
