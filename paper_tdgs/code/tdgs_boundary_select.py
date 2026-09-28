"""Source-only boundary-aware candidate for the TDGS follow-up.

This module deliberately does not run downstream training. It computes a
global class-mixing score from the already-built propagation kernel, converts
it to a within-class rank weight, and can call the frozen Round-2 masked greedy
selector. The caller must preregister the transform and evaluate the returned
IDs against Graph-A2 and tdgs_mask.
"""

import numpy as np


def cross_class_share(K, y):
    """Return q_i: the K-mass from demand i reaching another class."""
    K = K.tocoo()
    y = np.asarray(y)
    if K.shape[0] != len(y):
        raise ValueError("K and y have incompatible sizes")
    total = np.bincount(K.row, weights=np.maximum(K.data, 0.0),
                        minlength=len(y))
    cross = np.bincount(
        K.row[y[K.row] != y[K.col]],
        weights=np.maximum(K.data[y[K.row] != y[K.col]], 0.0),
        minlength=len(y),
    )
    return cross / np.maximum(total, 1e-12)


def class_rank_factor(q, y, floor=0.5):
    """Return positive within-class rank factors with mean one per class.

    The floor prevents zero weight for the least mixed point. Ties receive the
    same average rank. No downstream labels or predictions are used.
    """
    q = np.asarray(q, dtype=float)
    y = np.asarray(y)
    if q.ndim != 1 or len(q) != len(y):
        raise ValueError("q and y must be one-dimensional and aligned")
    out = np.empty(len(y), dtype=float)
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        order = np.argsort(q[idx], kind="mergesort")
        ranks = np.empty(len(idx), dtype=float)
        sorted_q = q[idx][order]
        starts = np.r_[0, 1 + np.flatnonzero(sorted_q[1:] != sorted_q[:-1])]
        ends = np.r_[starts[1:], len(idx)]
        for start, end in zip(starts, ends):
            ranks[order[start:end]] = np.mean(np.arange(start, end)) + floor
        # Average rank factor is one within every class, preserving class mass.
        out[idx] = ranks / ranks.mean()
    return out


def boundary_demand_weights(K, y, classes=None):
    """Return class-balanced, boundary-ranked demand weights.

    The class total is 1/C when ``classes`` is omitted, matching the current
    class-flat convention before the kernel normalizer is applied.
    """
    y = np.asarray(y)
    if classes is None:
        classes = int(y.max()) + 1
    q = cross_class_share(K, y)
    factor = class_rank_factor(q, y)
    counts = np.bincount(y, minlength=classes).astype(float)
    if np.any(counts == 0):
        raise ValueError("empty class in labels")
    weights = factor / (classes * counts[y])
    return weights, q, factor


def select_boundary(K, y, bpc, lam=0.5, classes=None, verbose=False):
    """Select IDs with the fixed boundary-aware candidate objective.

    This imports the existing greedy implementation lazily so the pure score
    functions above remain usable without the project package on ``sys.path``.
    ``K`` must be the global, nonnegative ``A_sym + A_sym**2`` kernel.
    """
    from tdgs_select import greedy_blended, pool_value
    from tdgs_select_r2 import build_kernel_variant

    y = np.asarray(y)
    if classes is None:
        classes = int(y.max()) + 1
    w2, q, factor = boundary_demand_weights(K, y, classes)
    masked, kmax = build_kernel_variant(K, y, mask_class=True,
                                        permute_within_class=False)
    w1 = np.ones(len(y), dtype=np.float32)
    selected, gains = greedy_blended(
        K, masked, y, bpc,
        w1 / pool_value(K, w1),
        w2.astype(np.float32) / pool_value(masked, w2),
        lam, verbose,
    )
    return selected, gains, {"q": q, "rank_factor": factor,
                             "kernel_scale": kmax}


__all__ = ["cross_class_share", "class_rank_factor",
           "boundary_demand_weights", "select_boundary"]
