"""Frozen-UNI selection pilot with a train-only audit split.

The script deliberately keeps this experiment independent from the previous
selection and test-only directories. It reads cached train UNI embeddings and
train labels, creates a per-block selection/audit split, runs the four Round 1
methods and the four Round 2 task-geometry methods, and evaluates a fixed
multinomial linear probe on the held-out train audit points.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

DATASETS = ("bloodmnist", "organamnist", "organsmnist", "pathmnist", "tissuemnist")
ROUND1 = ("random", "herding", "graph_a2", "geometry")
ROUND2 = ("random_plus_pilot", "geometry_plus_pilot", "task_geometry_plus_pilot", "permuted_task_geometry_plus_pilot")

# Training-weight factor (decision E). The bound |R_D - R_{S,w}| <= L*D(S) only
# holds when each representative carries the mass of the points assigned to it.
# Equal-weight training does not test the object the theory motivates, so both
# are run for every arm and 'training_weighting' becomes a crossed factor.
# 'voronoi' rescales the assignment masses globally, so a class with a larger
# pool carries more total weight. Under a frequency-flat endpoint (balanced
# accuracy) that is a class-prior shift, not coverage repair, and the measured
# pool imbalance is 1.63 on organsmnist -- the dataset with the largest observed
# weighting gain. 'voronoi_within' rescales to mean 1 *inside each class*, so
# the class priors match 'equal' exactly. The decomposition is then:
#   voronoi_within - equal   = coverage repair (the effect of interest)
#   voronoi - voronoi_within = class-prior shift (a confound to be subtracted)
WEIGHTINGS = ("equal", "voronoi", "voronoi_within")

# Pre-registered constants (decision G). tau is NOT a free parameter: it must be
# fixed before any audit result is read, otherwise the whole comparison degrades
# into tuning-until-significant. Overriding requires --unfreeze-tau, which also
# stamps tau_frozen=false into run_metadata.json so the deviation is visible in
# the artifact rather than only in a shell history.
PREREGISTERED_TAU = 0.5

# Primary contrast (decision F). task_features() reads every pool label via
# errors[i, y_i] -= 1, while geometry_plus_pilot reads none, so their difference
# confounds "task geometry helps" with "label access helps". The within-class
# permutation control consumes exactly the same labels and destroys only the
# point-to-gradient correspondence, so it is the only clean counterfactual.
PRIMARY_CONTRAST = ("task_geometry_plus_pilot", "permuted_task_geometry_plus_pilot")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.int64).tobytes()).hexdigest()


def read_train_labels(embedding_path: Path, archive_path: Path | None) -> np.ndarray:
    data = np.load(embedding_path, allow_pickle=False)
    for key in ("labels", "targets", "y"):
        if key in data.files:
            return np.asarray(data[key]).reshape(-1).astype(np.int64)
    if archive_path is None:
        raise FileNotFoundError(f"no labels in {embedding_path} and no archive was supplied")
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open("train_labels.npy") as handle:
            return np.load(handle, allow_pickle=False).reshape(-1).astype(np.int64)


def cap_per_class(labels: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    parts = []
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        rng = np.random.default_rng(seed + 1009 * int(label))
        take = min(max_per_class, len(ids))
        parts.append(np.sort(rng.choice(ids, size=take, replace=False)))
    return np.sort(np.concatenate(parts)).astype(np.int64)


def split_block(candidate_ids: np.ndarray, labels: np.ndarray, audit_fraction: float, seed: int):
    selection, audit = [], []
    for label in np.unique(labels[candidate_ids]):
        ids = candidate_ids[labels[candidate_ids] == label].copy()
        rng = np.random.default_rng(seed + 7919 * int(label))
        rng.shuffle(ids)
        n_audit = max(1, int(round(len(ids) * audit_fraction)))
        audit.extend(ids[:n_audit].tolist())
        selection.extend(ids[n_audit:].tolist())
    return np.sort(np.asarray(selection, dtype=np.int64)), np.sort(np.asarray(audit, dtype=np.int64))


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def pairwise_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance without materialising an (n, m, d) tensor.

    The naive ``norm(a[:, None, :] - b[None, :, :], axis=2)`` allocates
    n*m*d floats. At UNI's d=1024 with ~800 points per class that is 2.6 GB
    per call, which cannot fit the 4G qos-normal per-user cap. Same identity
    as allocation_baselines._pairwise_l2.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    return np.sqrt(np.maximum(aa + bb - 2.0 * (a @ b.T), 0.0))


def voronoi_weights(pool_x: np.ndarray, pool_y: np.ndarray, local_selected: np.ndarray):
    """Assignment mass w_i and covering distortion D(S) for one selection.

    Implements w_i = |{j : pi(j) = i}| from the generalisation bound
    ``|R_D(theta) - R_{S,w}(theta)| <= L * D(S)``, together with the D(S) that
    appears on its right-hand side, so the bound can be checked empirically
    instead of only cited as motivation.

    The assignment ``pi`` is taken **within class**. A label-blind nearest
    representative would let pi(j) cross a class boundary, and then
    ``|l(theta, x_i, y_i) - l(theta, x_j, y_j)|`` is not controlled by
    ``||z_i - z_j||`` at all -- the Lipschitz step in the derivation silently
    fails. Within-class assignment is also what the per-class fixed quota
    already imposes on the selection side.

    Returns ``(w_global, w_within, distortion)``. Both weight vectors are
    aligned row-for-row with ``local_selected``. ``w_global`` is rescaled to
    mean 1 over the whole selection; ``w_within`` to mean 1 inside each class,
    which leaves the class priors identical to equal weighting. Rescaling at all
    matters: sklearn does not renormalise the L2 penalty by
    ``sum(sample_weight)``, so raw counts would make the weighted arm far less
    regularised and confound weighting with effective C.
    """
    local_selected = np.asarray(local_selected, dtype=np.int64)
    if len(np.unique(local_selected)) != len(local_selected):
        raise RuntimeError("selection contains duplicate pool indices")
    pool_z = normalize(pool_x)
    position = {int(i): p for p, i in enumerate(local_selected)}
    w = np.zeros(len(local_selected), dtype=np.float64)
    w_within = np.zeros(len(local_selected), dtype=np.float64)
    distance_sum, n_assigned = 0.0, 0
    for label in np.unique(pool_y):
        ids = np.flatnonzero(pool_y == label)
        reps = local_selected[pool_y[local_selected] == label]
        if len(reps) == 0:
            raise RuntimeError(f"class {int(label)} has no representative")
        d = pairwise_l2(pool_z[ids], pool_z[reps])
        nearest = d.argmin(axis=1)
        distance_sum += float(d[np.arange(len(ids)), nearest].sum())
        n_assigned += len(ids)
        counts = np.bincount(nearest, minlength=len(reps)).astype(np.float64)
        if counts.sum() <= 0:
            raise RuntimeError(f"class {int(label)} received no assignment")
        scaled = counts / counts.sum() * len(reps)
        for rep, raw, within in zip(reps.tolist(), counts.tolist(), scaled.tolist()):
            w[position[rep]] = raw
            w_within[position[rep]] = within
    if w.sum() <= 0:
        raise RuntimeError("empty Voronoi assignment")
    return w / w.sum() * len(w), w_within, distance_sum / max(n_assigned, 1)


def assert_labels_aligned(x: np.ndarray, y: np.ndarray, dataset: str, seed: int,
                          min_balanced_accuracy: float) -> float:
    """Guard against silent embedding/label misalignment.

    The embedding .npz stores only 'embeddings'; labels are read from a
    separate MedMNIST archive and alignment is positional by assumption.
    A length check cannot catch a permutation. A probe on correctly aligned
    UNI features scores far above chance; on misaligned ones it collapses to
    chance. This is the failure mode that an offline gate provably cannot
    detect downstream, so it is checked here at the source.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    tr, te = [], []
    for label in classes:
        ids = np.flatnonzero(y == label)
        take = min(200, len(ids))
        pick = rng.choice(ids, size=take, replace=False)
        cut = max(1, take // 2)
        tr.extend(pick[:cut].tolist())
        te.extend(pick[cut:].tolist())
    tr = np.asarray(tr, dtype=np.int64)
    te = np.asarray(te, dtype=np.int64)
    head = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)
    head.fit(normalize(x[tr]), y[tr])
    score = float(balanced_accuracy_score(y[te], head.predict(normalize(x[te]))))
    chance = 1.0 / len(classes)
    if score < min_balanced_accuracy:
        raise RuntimeError(
            f"{dataset}: label-alignment probe balanced accuracy {score:.3f} is below "
            f"the {min_balanced_accuracy:.3f} floor (chance={chance:.3f}). Embeddings and "
            f"labels are probably misaligned -- they come from different files."
        )
    return score


def load_graphcov(root: Path):
    # The cluster's FAISS probe can abort during import before GraphCov's own
    # ImportError fallback. Blocking only this optional module selects CPU.
    sys.modules.setdefault("faiss", None)
    sys.path.insert(0, str(root))
    from graphcov.run.selection import select
    return select


def random_balanced(x: np.ndarray, y: np.ndarray, budget: int, seed: int) -> np.ndarray:
    out = []
    for label in np.unique(y):
        ids = np.flatnonzero(y == label)
        rng = np.random.default_rng(seed + 17 * int(label))
        out.extend(rng.choice(ids, size=min(budget, len(ids)), replace=False).tolist())
    return np.asarray(out, dtype=np.int64)


def random_fill_with_forced(
    y: np.ndarray, forced: np.ndarray, budget: int, seed: int
) -> np.ndarray:
    """Keep forced examples and fill each class to exactly ``budget``."""
    forced = np.asarray(forced, dtype=np.int64)
    out = []
    for label in np.unique(y):
        class_ids = np.flatnonzero(y == label)
        class_forced = np.intersect1d(class_ids, forced, assume_unique=False)
        if len(class_forced) > budget:
            raise ValueError(
                f"forced examples exceed budget for class {label}: "
                f"{len(class_forced)} > {budget}"
            )
        need = budget - len(class_forced)
        available = np.setdiff1d(class_ids, class_forced, assume_unique=False)
        if len(available) < need:
            raise ValueError(
                f"not enough examples to fill class {label}: "
                f"need {need}, available {len(available)}"
            )
        rng = np.random.default_rng(seed + 17 * int(label))
        fill = rng.choice(available, size=need, replace=False)
        out.extend(class_forced.tolist())
        out.extend(fill.tolist())
    return np.asarray(out, dtype=np.int64)


def herding(x: np.ndarray, y: np.ndarray, budget: int) -> np.ndarray:
    out = []
    x = normalize(x)
    for label in np.unique(y):
        ids = np.flatnonzero(y == label)
        z = x[ids]
        k = min(budget, len(ids))
        target = z.mean(axis=0)
        running = np.zeros_like(target)
        chosen = []
        for step in range(k):
            residual = (step + 1) * target - running
            scores = z @ residual
            if chosen:
                scores[np.asarray(chosen)] = -np.inf
            pick = int(np.argmax(scores))
            chosen.append(pick)
            running += z[pick]
        out.extend(ids[np.asarray(chosen)].tolist())
    return np.asarray(out, dtype=np.int64)


def facility_from_features(features: np.ndarray, labels: np.ndarray, budget: int, forced: np.ndarray | None = None) -> np.ndarray:
    """Greedy facility location with uniform demand and Euclidean distance."""
    out = []
    forced = np.asarray(forced if forced is not None else [], dtype=np.int64)
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        z = np.asarray(features[ids], dtype=np.float32)
        k = min(budget, len(ids))
        local_forced = [int(np.flatnonzero(ids == i)[0]) for i in forced if i in set(ids.tolist())]
        local_forced = list(dict.fromkeys(local_forced))[:k]
        d = pairwise_l2(z, z)
        selected = list(local_forced)
        if selected:
            nearest = d[:, selected].min(axis=1)
        else:
            nearest = np.full(len(ids), np.inf, dtype=np.float32)
        while len(selected) < k:
            gains = nearest[:, None] - d
            gains = np.maximum(gains, 0).sum(axis=0)
            if selected:
                gains[np.asarray(selected)] = -np.inf
            pick = int(np.argmax(gains))
            selected.append(pick)
            nearest = np.minimum(nearest, d[:, pick])
        out.extend(ids[np.asarray(selected)].tolist())
    return np.asarray(out, dtype=np.int64)


def task_features(z: np.ndarray, y: np.ndarray, pilot: np.ndarray, seed: int, projection_dim: int) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    z = normalize(z)
    classes = np.unique(y)
    class_to_col = {int(c): i for i, c in enumerate(classes)}
    # Newer scikit-learn versions infer multinomial behavior from lbfgs and
    # no longer accept the removed multi_class keyword.
    head = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)
    head.fit(z[pilot], y[pilot])
    probabilities = head.predict_proba(z)
    errors = np.zeros_like(probabilities)
    for i, label in enumerate(y):
        errors[i] = probabilities[i]
        errors[i, class_to_col[int(label)]] -= 1.0
    augmented = np.concatenate([z, np.ones((len(z), 1), dtype=np.float32)], axis=1)
    gradient = np.einsum("nc,nd->ncd", errors, augmented, optimize=True).reshape(len(z), -1)
    rng = np.random.default_rng(seed)
    projection = rng.normal(0, 1 / np.sqrt(projection_dim), size=(gradient.shape[1], projection_dim)).astype(np.float32)
    return gradient @ projection


def metrics(train_x, train_y, audit_x, audit_y, seed: int, sample_weight=None) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, recall_score

    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)
    # Only the training side is weighted. The audit split stands in for the
    # target distribution D, so reweighting it would move the estimand.
    clf.fit(normalize(train_x), train_y, sample_weight=sample_weight)
    probs = clf.predict_proba(normalize(audit_x))
    pred = clf.classes_[np.argmax(probs, axis=1)]
    recalls = recall_score(audit_y, pred, labels=clf.classes_, average=None, zero_division=0)
    one_hot = np.eye(len(clf.classes_), dtype=np.float32)[np.searchsorted(clf.classes_, audit_y)]
    confidence = probs.max(axis=1)
    correct = (pred == audit_y).astype(np.float32)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        mask = (confidence >= lo) & ((confidence < hi) if hi < 1 else (confidence <= hi))
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return {
        "accuracy": float(accuracy_score(audit_y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(audit_y, pred)),
        "nll": float(log_loss(audit_y, probs, labels=clf.classes_)),
        "brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "ece15": float(ece),
        "worst_class_recall": float(np.min(recalls)),
        "per_class_recall": {str(int(c)): float(v) for c, v in zip(clf.classes_, recalls)},
    }


def run_dataset(args, dataset: str, select_fn, output_dir: Path, all_rows: list, manifest_handle):
    emb_path = args.embedding_root / f"{dataset}_train_uni_{args.size}.npz"
    archive = args.data_root / f"{dataset}_{args.size}.npz" if args.data_root else None
    data = np.load(emb_path, allow_pickle=False)
    x_full = np.asarray(data["embeddings"], dtype=np.float32)
    y_full = read_train_labels(emb_path, archive)
    if len(x_full) != len(y_full):
        raise RuntimeError(f"{dataset}: embeddings={len(x_full)} labels={len(y_full)}")
    align_ba = assert_labels_aligned(
        x_full, y_full, dataset, args.seed + 500009 * DATASETS.index(dataset),
        args.min_alignment_ba,
    )
    print(f"[{dataset}] label-alignment probe balanced_accuracy={align_ba:.4f}", flush=True)
    candidate = cap_per_class(y_full, args.max_per_class, args.seed + 100003 * DATASETS.index(dataset))
    dataset_meta = {"dataset": dataset, "embedding_path": str(emb_path), "embedding_sha256": sha256_file(emb_path), "candidate_sha256": sha256_array(candidate), "candidate_n": int(len(candidate)), "label_alignment_balanced_accuracy": align_ba}
    for block in range(args.blocks):
        block_seed = args.seed + 100003 * DATASETS.index(dataset) + 1009 * block
        select_ids, audit_ids = split_block(candidate, y_full, args.audit_fraction, block_seed)
        local_y = y_full[select_ids]
        local_x = x_full[select_ids]
        audit_x, audit_y = x_full[audit_ids], y_full[audit_ids]
        for method in ROUND1:
            t0 = time.time()
            if method == "random":
                local_selected = random_balanced(local_x, local_y, args.budget_per_class, block_seed)
            elif method == "herding":
                local_selected = herding(local_x, local_y, args.budget_per_class)
            elif method == "geometry":
                local_selected = facility_from_features(normalize(local_x), local_y, args.budget_per_class)
            else:
                local_selected = np.asarray(select_fn(method="graph_a2", labels=local_y, budget_per_class=args.budget_per_class, embeddings=local_x, seed=block_seed, verbose=False, global_selection=False, sparse_cpu=True, k_neighbors=args.graph_k_neighbors, k_hops=args.graph_k_hops, _verbose_level=0), dtype=np.int64)
            selection_seconds = time.time() - t0
            selected_ids = select_ids[local_selected]
            weights, weights_within, distortion = voronoi_weights(local_x, local_y, local_selected)
            by_weighting = {"equal": None, "voronoi": weights, "voronoi_within": weights_within}
            for weighting in WEIGHTINGS:
                result = metrics(x_full[selected_ids], y_full[selected_ids], audit_x, audit_y, block_seed,
                                 sample_weight=by_weighting[weighting])
                all_rows.append({"round": "round1", "dataset": dataset, "block": block, "method": method, "training_weighting": weighting, "budget_per_class": args.budget_per_class, "selection_n": len(selected_ids), "selection_seconds": selection_seconds, "covering_distortion": distortion, "weight_max_over_mean": float(weights.max()), "selected_sha256": sha256_array(np.sort(selected_ids)), "audit_sha256": sha256_array(audit_ids), **result})
            manifest_handle.write(json.dumps({"round": "round1", "dataset": dataset, "block": block, "method": method, "selected_ids": selected_ids.tolist(), "voronoi_weights": weights.tolist(), "voronoi_weights_within": weights_within.tolist(), "covering_distortion": distortion, "audit_ids": audit_ids.tolist(), "meta": dataset_meta}) + "\n")
        for method in ROUND2:
            t0 = time.time()
            pilot = random_balanced(local_x, local_y, args.pilot_per_class, block_seed + 700001)
            if method == "random_plus_pilot":
                chosen = random_fill_with_forced(
                    local_y, pilot, args.budget_per_class, block_seed + 700003
                )
            else:
                z = normalize(local_x)
                if method == "geometry_plus_pilot":
                    features = z
                else:
                    g = task_features(z, local_y, pilot, block_seed + 700005, args.gradient_projection_dim)
                    if method == "permuted_task_geometry_plus_pilot":
                        g = g.copy()
                        for label in np.unique(local_y):
                            ids = np.flatnonzero(local_y == label)
                            rng = np.random.default_rng(block_seed + 800003 + int(label))
                            g[ids] = g[rng.permutation(ids)]
                    # Random pairs, not adjacent indices. select_ids is sorted
                    # and MedMNIST train order is largely class-grouped, so
                    # g[1::2]-g[0::2] would pair mostly same-class neighbours
                    # and underestimate the scale, silently inflating tau.
                    scale_rng = np.random.default_rng(block_seed + 900007)
                    n_pair = min(len(g), 1000) // 2 * 2
                    perm = scale_rng.permutation(len(g))[:n_pair]
                    scale = np.median(
                        np.linalg.norm(g[perm[1::2]] - g[perm[0::2]], axis=1)
                    )
                    scale = max(float(scale), 1e-6)
                    features = np.concatenate([np.sqrt(1 - args.tau) * z, np.sqrt(args.tau) * g / scale], axis=1)
                chosen = facility_from_features(features, local_y, args.budget_per_class, forced=pilot)
            selection_seconds = time.time() - t0
            selected_ids = select_ids[chosen]
            counts = np.bincount(
                y_full[selected_ids].astype(np.int64),
                minlength=int(np.max(y_full)) + 1,
            )
            expected = args.budget_per_class
            if any(counts[int(label)] != expected for label in np.unique(local_y)):
                raise RuntimeError(
                    f"{dataset} block={block} {method}: invalid per-class counts "
                    f"{counts.tolist()}"
                )
            weights, weights_within, distortion = voronoi_weights(local_x, local_y, chosen)
            by_weighting = {"equal": None, "voronoi": weights, "voronoi_within": weights_within}
            for weighting in WEIGHTINGS:
                result = metrics(x_full[selected_ids], y_full[selected_ids], audit_x, audit_y, block_seed,
                                 sample_weight=by_weighting[weighting])
                all_rows.append({"round": "round2", "dataset": dataset, "block": block, "method": method, "training_weighting": weighting, "budget_per_class": args.budget_per_class, "pilot_per_class": args.pilot_per_class, "selection_n": len(selected_ids), "selection_seconds": selection_seconds, "covering_distortion": distortion, "weight_max_over_mean": float(weights.max()), "selected_sha256": sha256_array(np.sort(selected_ids)), "audit_sha256": sha256_array(audit_ids), **result})
            manifest_handle.write(json.dumps({"round": "round2", "dataset": dataset, "block": block, "method": method, "selected_ids": selected_ids.tolist(), "voronoi_weights": weights.tolist(), "voronoi_weights_within": weights_within.tolist(), "covering_distortion": distortion, "pilot_ids": select_ids[pilot].tolist(), "audit_ids": audit_ids.tolist(), "meta": dataset_meta}) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--graphcov-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--max-per-class", type=int, default=1000)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--budget-per-class", type=int, default=50)
    p.add_argument("--pilot-per-class", type=int, default=10)
    p.add_argument("--audit-fraction", type=float, default=0.2)
    p.add_argument("--graph-k-neighbors", type=int, default=10)
    p.add_argument("--graph-k-hops", type=int, default=2)
    p.add_argument("--tau", type=float, default=PREREGISTERED_TAU)
    p.add_argument("--unfreeze-tau", action="store_true",
                   help="permit --tau to differ from the pre-registered value; stamps "
                        "tau_frozen=false into run_metadata.json")
    p.add_argument("--gradient-projection-dim", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--min-alignment-ba", type=float, default=0.30,
                   help="floor for the embedding/label alignment probe (chance is 1/C ~ 0.09-0.12)")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    tau_frozen = abs(args.tau - PREREGISTERED_TAU) <= 1e-12
    if not tau_frozen and not args.unfreeze_tau:
        raise SystemExit(
            f"--tau={args.tau} differs from the pre-registered {PREREGISTERED_TAU}. "
            "tau was frozen before any audit result was read; re-tuning it on the "
            "audit split is the failure mode this protocol exists to prevent. "
            "Pass --unfreeze-tau if this is a deliberate, separately-reported sensitivity run."
        )
    if args.smoke:
        args.datasets = [args.datasets[0]]
        args.blocks = 1
        args.max_per_class = min(args.max_per_class, 160)
        args.budget_per_class = min(args.budget_per_class, 10)
        args.pilot_per_class = min(args.pilot_per_class, 2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    select_fn = load_graphcov(args.graphcov_root)
    rows, start = [], time.time()
    manifest_path = args.output_dir / "selection_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for dataset in args.datasets:
            run_dataset(args, dataset, select_fn, args.output_dir, rows, manifest)
    with (args.output_dir / "linear_probe_results.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    fields = sorted({k for row in rows for k in row if k != "per_class_recall"})
    with (args.output_dir / "linear_probe_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})
    metadata = vars(args).copy()
    metadata.update({"created_unix": time.time(), "wall_seconds": time.time() - start, "official_validation_accessed": False, "official_test_accessed": False, "target_classifier_training": False, "tau_frozen": tau_frozen, "preregistered_tau": PREREGISTERED_TAU, "primary_contrast": list(PRIMARY_CONTRAST), "primary_endpoint": "balanced_accuracy", "primary_weighting": "equal", "training_weightings": list(WEIGHTINGS)})
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir), "wall_seconds": time.time() - start}, indent=2))


if __name__ == "__main__":
    main()
