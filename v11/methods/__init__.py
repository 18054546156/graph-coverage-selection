"""Selection primitives used by the v11 experiments."""

from .selection import (
    MarginResult,
    SelectionResult,
    build_diffusion_kernel,
    build_or_load_knn,
    build_weighted_adjacency,
    compute_exact_margins,
    coverage_metrics,
    greedy_facility,
    kernel_metrics,
    knn_purity,
    l2_normalize,
    posthoc_margin_repair,
    selection_margin_metrics,
    symmetric_normalize,
)

__all__ = [
    "MarginResult",
    "SelectionResult",
    "build_diffusion_kernel",
    "build_or_load_knn",
    "build_weighted_adjacency",
    "compute_exact_margins",
    "coverage_metrics",
    "greedy_facility",
    "kernel_metrics",
    "knn_purity",
    "l2_normalize",
    "posthoc_margin_repair",
    "selection_margin_metrics",
    "symmetric_normalize",
]
