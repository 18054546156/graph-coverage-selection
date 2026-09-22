"""Adapter for the GraphCov method registry; copy beside d8_ot.py in a fork."""

from __future__ import annotations

from typing import Any, Callable

try:
    from .d8_ot import select_equal_mass_ot
except ImportError:
    from d8_ot import select_equal_mass_ot


def register_d8_method(register_method: Callable[..., Any]) -> Callable[..., Any]:
    """Register D8 without importing or mutating the live GraphCov checkout."""

    @register_method(
        "d8_ot",
        needs={"embeddings"},
        importance="optional",
        kwargs={
            "epsilon": 0.05,
            "outer_iterations": 8,
            "sinkhorn_iterations": 60,
        },
        description="Equal-mass class-conditional entropic OT heuristic",
    )
    def _select_d8_ot(
        embeddings,
        labels,
        budget_per_class,
        importance=None,
        seed=42,
        verbose=False,
        _verbose_level=1,
        **kwargs,
    ):
        selected, diagnostics = select_equal_mass_ot(
            embeddings,
            labels,
            budget_per_class,
            seed,
            epsilon=kwargs.get("epsilon", 0.05),
            outer_iterations=kwargs.get("outer_iterations", 8),
            sinkhorn_iterations=kwargs.get("sinkhorn_iterations", 60),
        )
        if verbose:
            print(f"[D8-OT] diagnostics={diagnostics}")
        return selected.tolist()

    return _select_d8_ot
