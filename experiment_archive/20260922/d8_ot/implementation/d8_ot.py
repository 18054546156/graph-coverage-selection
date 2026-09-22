"""Heuristic equal-mass class-conditional entropic OT subset selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class ClassDiagnostics:
    label: int
    population: int
    budget: int
    entropic_transport_cost: float
    marginal_error: float
    outer_iterations: int


def _pairwise_euclidean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_sq = np.einsum("ij,ij->i", x, x)[:, None]
    y_sq = np.einsum("ij,ij->i", y, y)[None, :]
    squared = np.maximum(x_sq + y_sq - 2.0 * (x @ y.T), 0.0)
    return np.sqrt(squared, out=squared)


def _sinkhorn_uniform(
    cost: np.ndarray,
    epsilon: float,
    iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, float]:
    n, q = cost.shape
    log_kernel = -cost / epsilon
    log_a = -np.log(n)
    log_b = -np.log(q)
    log_u = np.zeros(n, dtype=np.float64)
    log_v = np.zeros(q, dtype=np.float64)
    error = np.inf

    for step in range(iterations):
        log_u = log_a - logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_b - logsumexp(log_kernel + log_u[:, None], axis=0)
        if step % 10 == 9 or step == iterations - 1:
            plan = np.exp(log_kernel + log_u[:, None] + log_v[None, :])
            error = max(
                float(np.max(np.abs(plan.sum(axis=1) - 1.0 / n))),
                float(np.max(np.abs(plan.sum(axis=0) - 1.0 / q))),
            )
            if error <= tolerance:
                break

    plan = np.exp(log_kernel + log_u[:, None] + log_v[None, :])
    return plan, error


def _weighted_geometric_median(
    x: np.ndarray,
    weights: np.ndarray,
    initial: np.ndarray,
    iterations: int = 30,
) -> np.ndarray:
    y = initial.astype(np.float64, copy=True)
    weights = np.maximum(weights.astype(np.float64, copy=False), 0.0)
    if not np.any(weights):
        return y

    for _ in range(iterations):
        distances = np.linalg.norm(x - y[None, :], axis=1)
        exact = distances <= 1e-12
        if np.any(exact):
            exact_weights = weights[exact]
            return np.average(x[exact], axis=0, weights=exact_weights)
        adjusted = weights / np.maximum(distances, 1e-12)
        updated = (adjusted[:, None] * x).sum(axis=0) / adjusted.sum()
        if np.linalg.norm(updated - y) <= 1e-7:
            y = updated
            break
        y = updated
    return y


def _project_distinct(medians: np.ndarray, x: np.ndarray) -> np.ndarray:
    distances = _pairwise_euclidean(medians, x)
    nearest = distances.min(axis=1)
    order = np.argsort(nearest, kind="stable")
    chosen = np.zeros(len(x), dtype=bool)
    result = np.empty(len(medians), dtype=np.int64)
    for slot in order:
        costs = distances[slot].copy()
        costs[chosen] = np.inf
        index = int(np.argmin(costs))
        result[slot] = index
        chosen[index] = True
    return result


def select_equal_mass_ot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    budget_per_class: int,
    seed: int,
    *,
    epsilon: float = 0.05,
    sinkhorn_iterations: int = 60,
    sinkhorn_tolerance: float = 1e-7,
    outer_iterations: int = 8,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Select source observations with equal classwise mass.

    The caller supplies the class quota so this function cannot silently change
    the benchmark's rounding rule. The returned indices address the original
    input rows. The optimizer is a deterministic heuristic, not an exact
    solution of the discrete W1 subset problem.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels).reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or len(x) == 0:
        raise ValueError("embeddings must be a non-empty 2D array aligned with labels")
    if not np.isfinite(x).all():
        raise ValueError("embeddings must contain only finite values")
    if budget_per_class < 1:
        raise ValueError("budget_per_class must be positive")
    if epsilon <= 0 or sinkhorn_iterations < 1 or outer_iterations < 1:
        raise ValueError("epsilon and iteration limits must be positive")

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-12)
    rng = np.random.default_rng(seed)
    chosen_global: list[int] = []
    diagnostics: list[dict[str, Any]] = []

    for label in np.unique(y):
        class_global = np.flatnonzero(y == label)
        class_x = x[class_global]
        n = len(class_global)
        q = min(int(budget_per_class), n)
        if q == n:
            selected_local = np.arange(n, dtype=np.int64)
            diagnostics.append(asdict(ClassDiagnostics(
                label=int(label), population=n, budget=q,
                entropic_transport_cost=0.0, marginal_error=0.0,
                outer_iterations=0,
            )))
            chosen_global.extend(class_global[selected_local].tolist())
            continue

        selected_local = rng.choice(n, size=q, replace=False)
        marginal_error = np.inf
        transport_cost = np.inf
        used_outer = 0

        for used_outer in range(1, outer_iterations + 1):
            selected_x = class_x[selected_local]
            cost = _pairwise_euclidean(class_x, selected_x)
            plan, marginal_error = _sinkhorn_uniform(
                cost, epsilon, sinkhorn_iterations, sinkhorn_tolerance
            )
            transport_cost = float(np.sum(plan * cost))
            medians = np.vstack([
                _weighted_geometric_median(class_x, plan[:, j], selected_x[j])
                for j in range(q)
            ])
            updated = _project_distinct(medians, class_x)
            if np.array_equal(updated, selected_local):
                break
            selected_local = updated

        final_cost = _pairwise_euclidean(class_x, class_x[selected_local])
        final_plan, marginal_error = _sinkhorn_uniform(
            final_cost, epsilon, sinkhorn_iterations, sinkhorn_tolerance
        )
        transport_cost = float(np.sum(final_plan * final_cost))
        chosen_global.extend(class_global[selected_local].tolist())
        diagnostics.append(asdict(ClassDiagnostics(
            label=int(label), population=n, budget=q,
            entropic_transport_cost=transport_cost,
            marginal_error=marginal_error,
            outer_iterations=used_outer,
        )))

    return np.asarray(chosen_global, dtype=np.int64), diagnostics
