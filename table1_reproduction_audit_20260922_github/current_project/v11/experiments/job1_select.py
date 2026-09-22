#!/usr/bin/env python3
"""Job 1: generate and audit frozen selected-index sets.

This program reads training embeddings and training labels only. It never loads
validation/test images and never trains a classifier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v11.methods.selection import (  # noqa: E402
    MarginResult,
    SelectionResult,
    build_diffusion_kernel,
    build_or_load_knn,
    build_weighted_adjacency,
    compute_exact_margins,
    coverage_metrics,
    greedy_facility,
    json_ready,
    kernel_metrics,
    knn_purity,
    l2_normalize,
    posthoc_margin_repair,
    selection_margin_metrics,
    symmetric_normalize,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--only-dataset")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str, project_root: Path, config_dir: Path) -> Path:
    expanded = os.path.expandvars(value.replace("{project_root}", str(project_root)))
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def ratio_key(ratio: float) -> str:
    return f"r{ratio:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    header = f"{contiguous.dtype}|{contiguous.shape}".encode("ascii")
    return sha256_bytes(header + contiguous.tobytes())


def load_embeddings(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    with np.load(path) as payload:
        if "embeddings" not in payload:
            raise KeyError(f"{path} has no embeddings array")
        return np.asarray(payload["embeddings"], dtype=np.float32)


def load_train_labels(archive: Path, key: str = "train_labels") -> np.ndarray:
    with np.load(archive) as payload:
        if key not in payload:
            raise KeyError(f"{archive} has no {key}")
        labels = np.asarray(payload[key], dtype=np.int64).reshape(-1)
    return labels


def load_or_compute_margins(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cache_path: Path,
    config: dict,
    force: bool,
) -> tuple[MarginResult, str]:
    if cache_path.exists() and not force:
        with np.load(cache_path) as payload:
            result = MarginResult(
                np.asarray(payload["same_distance"], dtype=np.float32),
                np.asarray(payload["different_distance"], dtype=np.float32),
                np.asarray(payload["ratio"], dtype=np.float32),
                np.asarray(payload["gap"], dtype=np.float32),
                np.asarray(payload["unsafe"], dtype=bool),
            )
            backend = str(payload["backend"].item()) if "backend" in payload else "cache"
        return result, f"cache:{backend}"
    result, backend = compute_exact_margins(
        embeddings,
        labels,
        backend=str(config.get("backend", "auto")),
        gpu=config.get("gpu", 0),
        batch_size=int(config.get("batch_size", 512)),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        same_distance=result.same_distance,
        different_distance=result.different_distance,
        ratio=result.ratio,
        gap=result.gap,
        unsafe=result.unsafe,
        backend=np.asarray(backend),
    )
    return result, backend


def validate_job(job: dict) -> None:
    variants = job.get("variants", [])
    if not variants:
        raise ValueError(f"{job.get('dataset', '<unknown>')}: variants must not be empty")
    if not job.get("ratios"):
        raise ValueError(f"{job.get('dataset', '<unknown>')}: ratios must not be empty")
    ids = [str(variant["id"]) for variant in variants]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate variant id in {job['dataset']}")
    if str(job["reference_variant"]) not in ids:
        raise ValueError(f"reference variant missing in {job['dataset']}")
    reference_mode = str(job.get("reference_mode", "generated"))
    if reference_mode not in {"generated", "frozen"}:
        raise ValueError(
            f"{job['dataset']}: reference_mode must be generated or frozen"
        )
    if reference_mode == "frozen":
        frozen = job.get("frozen_reference_indices", {})
        missing = [
            ratio_key(float(ratio))
            for ratio in job["ratios"]
            if ratio_key(float(ratio)) not in frozen
        ]
        if missing:
            raise ValueError(
                f"{job['dataset']}: frozen reference missing ratios {missing}"
            )
    seen: set[str] = set()
    for variant in variants:
        if int(variant["k"]) < 1:
            raise ValueError(f"{variant['id']}: k must be positive")
        kernel = variant.get("kernel", {})
        if int(kernel.get("hops", 0)) < 1:
            raise ValueError(f"{variant['id']}: kernel.hops must be positive")
        constraint = variant.get("constraint", {"mode": "none"})
        mode = str(constraint.get("mode", "none"))
        if mode not in {"none", "hard_cap", "posthoc_repair"}:
            raise ValueError(f"{variant['id']}: unknown constraint mode {mode}")
        for dependency_key in ("source_variant", "base_variant"):
            dependency = constraint.get(dependency_key)
            if dependency is not None and dependency not in seen:
                raise ValueError(
                    f"{variant['id']} depends on {dependency}, which must appear earlier"
                )
        seen.add(str(variant["id"]))


def kernel_signature(variant: dict) -> str:
    payload = {
        "k": int(variant["k"]),
        "kernel": variant["kernel"],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def unsafe_caps_from_reference(
    reference: SelectionResult,
    labels: np.ndarray,
    margins: MarginResult,
    budget: int,
    multiplier: float,
) -> dict[int, int]:
    caps = {}
    for cls in np.unique(labels):
        subset = reference.indices[labels[reference.indices] == cls]
        observed = int(np.sum(margins.unsafe[subset]))
        caps[int(cls)] = min(budget, int(np.ceil(multiplier * observed)))
    return caps


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = len(np.union1d(left, right))
    return float(len(np.intersect1d(left, right)) / union) if union else 1.0


def audit_frozen_reference(
    selected: np.ndarray,
    spec: dict | None,
    project_root: Path,
    config_dir: Path,
) -> dict[str, object] | None:
    if spec is None:
        return None
    path = resolve_path(str(spec["path"]), project_root, config_dir)
    required = bool(spec.get("required", False))
    if not path.exists():
        if required:
            raise FileNotFoundError(f"required frozen reference is missing: {path}")
        return {"path": str(path), "status": "missing_optional"}
    frozen = np.asarray(np.load(path), dtype=np.int64).reshape(-1)
    exact_set_match = bool(np.array_equal(np.sort(selected), np.sort(frozen)))
    result = {
        "path": str(path),
        "status": "checked",
        "required": required,
        "n_frozen": int(len(frozen)),
        "exact_set_match": exact_set_match,
        "jaccard": jaccard(selected, frozen),
        "frozen_index_sha256": sha256_array(np.sort(frozen)),
    }
    if required and not exact_set_match:
        raise AssertionError(
            f"generated reference does not reproduce frozen indices: {path}; "
            f"Jaccard={result['jaccard']:.6f}"
        )
    return result


def load_frozen_reference(
    spec: dict,
    *,
    labels: np.ndarray,
    margins: MarginResult,
    budget: int,
    project_root: Path,
    config_dir: Path,
) -> tuple[SelectionResult, dict[str, object]]:
    """Load an official frozen subset without regenerating it in v11."""
    path = resolve_path(str(spec["path"]), project_root, config_dir)
    if not path.exists():
        raise FileNotFoundError(f"required frozen reference is missing: {path}")

    selected = np.asarray(np.load(path), dtype=np.int64).reshape(-1)
    classes = np.unique(labels)
    expected = int(budget * len(classes))
    if len(selected) != expected or len(np.unique(selected)) != expected:
        raise ValueError(
            f"invalid frozen reference size or duplicates: {len(selected)} != {expected}"
        )
    if expected and (int(selected.min()) < 0 or int(selected.max()) >= len(labels)):
        raise ValueError(f"frozen reference contains an out-of-range index: {path}")
    class_counts = {
        int(cls): int(np.sum(labels[selected] == cls)) for cls in classes
    }
    if any(value != budget for value in class_counts.values()):
        raise ValueError(f"frozen reference class quota violation: {class_counts}")

    order_path = path.with_name("selection_order.npy")
    if order_path.exists():
        order = np.asarray(np.load(order_path), dtype=np.int64).reshape(-1)
        if not np.array_equal(np.sort(order), np.sort(selected)):
            raise ValueError(
                f"frozen selection order does not contain the selected set: {order_path}"
            )
    else:
        order = selected.copy()

    unsafe_counts = {
        int(cls): int(np.sum(margins.unsafe[selected[labels[selected] == cls]]))
        for cls in classes
    }
    result = SelectionResult(selected.copy(), order.copy(), class_counts, unsafe_counts)
    audit = {
        "path": str(path),
        "order_path": str(order_path) if order_path.exists() else None,
        "status": "reused_frozen",
        "required": True,
        "direct_reuse": True,
        "n_frozen": int(len(selected)),
        "exact_array_match": True,
        "exact_set_match": True,
        "jaccard": 1.0,
        "frozen_index_sha256": sha256_array(selected),
        "frozen_set_sha256": sha256_array(np.sort(selected)),
    }
    return result, audit


def selection_summary(
    *,
    variant: dict,
    ratio: float,
    budget: int,
    selection: SelectionResult,
    labels: np.ndarray,
    margins: MarginResult,
    kernel,
    reference_kernel,
    reference_selection: SelectionResult,
    knn_distances: np.ndarray,
    kernel_build: dict,
    kernel_audit: dict,
    unsafe_caps: dict[int, int] | None,
) -> dict[str, object]:
    selected = selection.indices
    class_counts = {
        int(cls): int(np.sum(labels[selected] == cls)) for cls in np.unique(labels)
    }
    if any(value != budget for value in class_counts.values()):
        raise AssertionError(f"class quota violation: {class_counts}")
    return {
        "schema": "graphcov-v11/selection-v1",
        "variant": variant,
        "ratio": float(ratio),
        "budget_per_class": int(budget),
        "n_selected": int(len(selected)),
        "class_counts": class_counts,
        "unsafe_caps": unsafe_caps,
        "index_sha256": sha256_array(selected),
        "jaccard_with_reference": jaccard(selected, reference_selection.indices),
        "margin": selection_margin_metrics(margins, labels, selected),
        "self_kernel_coverage": coverage_metrics(kernel, labels, selected),
        "reference_kernel_coverage": coverage_metrics(reference_kernel, labels, selected),
        "selected_knn_squared_distance_mean": float(np.mean(knn_distances[selected])),
        "selected_knn_squared_distance_p90": float(np.quantile(knn_distances[selected], 0.9)),
        "kernel_build": kernel_build,
        "kernel_audit": kernel_audit,
    }


def build_kernel_for_variant(
    variant: dict,
    embeddings: np.ndarray,
    cache_root: Path,
    dataset_name: str,
    knn_config: dict,
    force: bool,
):
    k = int(variant["k"])
    cache_path = cache_root / "knn" / f"{dataset_name}_train_uni224_k{k}.npz"
    indices, distances, knn_metadata = build_or_load_knn(
        embeddings,
        k,
        cache_path,
        backend=str(knn_config.get("backend", "auto")),
        gpu=knn_config.get("gpu", 0),
        force=force,
    )
    adjacency = build_weighted_adjacency(indices, distances, len(embeddings))
    normalized = symmetric_normalize(adjacency)
    kernel_config = variant["kernel"]
    kernel, build_metadata = build_diffusion_kernel(
        normalized,
        hops=int(kernel_config["hops"]),
        weight_mode=str(kernel_config.get("weight_mode", "uniform")),
        hop_decay=float(kernel_config.get("hop_decay", 1.0)),
        truncation=str(kernel_config.get("truncation", "none")),
        max_degree=kernel_config.get("max_degree"),
        symmetrize_after=bool(kernel_config.get("symmetrize_after", False)),
    )
    build_metadata["k"] = k
    build_metadata["knn"] = knn_metadata
    return kernel, indices, distances, build_metadata


def process_job(
    *,
    job: dict,
    dataset_spec: dict,
    output_root: Path,
    cache_root: Path,
    project_root: Path,
    config_dir: Path,
    global_config: dict,
    force: bool,
) -> list[dict[str, object]]:
    validate_job(job)
    dataset_name = str(job["dataset"])
    dataset_output = output_root / dataset_name
    embedding_path = resolve_path(dataset_spec["embedding"], project_root, config_dir)
    label_archive = resolve_path(dataset_spec["label_archive"], project_root, config_dir)
    embeddings = load_embeddings(embedding_path)
    labels = load_train_labels(label_archive, str(dataset_spec.get("label_key", "train_labels")))
    if len(embeddings) != len(labels):
        raise ValueError(f"{dataset_name}: embedding and label counts differ")
    classes = np.unique(labels)
    expected_samples = dataset_spec.get("n_train")
    expected_classes = dataset_spec.get("n_classes")
    expected_dimension = dataset_spec.get("embedding_dim")
    if expected_samples is not None and len(labels) != int(expected_samples):
        raise ValueError(
            f"{dataset_name}: expected {expected_samples} train samples, got {len(labels)}"
        )
    if expected_classes is not None and len(classes) != int(expected_classes):
        raise ValueError(
            f"{dataset_name}: expected {expected_classes} classes, got {len(classes)}"
        )
    if expected_dimension is not None and embeddings.shape[1] != int(expected_dimension):
        raise ValueError(
            f"{dataset_name}: expected embedding dimension {expected_dimension}, "
            f"got {embeddings.shape[1]}"
        )
    ratios = [float(value) for value in job["ratios"]]
    budgets = {ratio: int(len(labels) * ratio / len(classes)) for ratio in ratios}
    if any(value < 1 for value in budgets.values()):
        raise ValueError(f"{dataset_name}: a ratio yields zero budget")

    label_cache = cache_root / "labels" / f"{dataset_name}_train.npy"
    label_cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(label_cache, labels)

    margin_config = job.get("margin", global_config.get("margin", {}))
    margin_cache = cache_root / "margins" / f"{dataset_name}_train_uni224_exact.npz"
    margins, margin_backend = load_or_compute_margins(
        embeddings, labels, margin_cache, margin_config, force
    )
    train_margin = {
        "backend": margin_backend,
        "unsafe_ratio": float(np.mean(margins.unsafe)),
        "ratio_mean": float(np.mean(margins.ratio)),
        "ratio_p10": float(np.quantile(margins.ratio, 0.1)),
        "gap_mean": float(np.mean(margins.gap)),
        "gap_p10": float(np.quantile(margins.gap, 0.1)),
    }

    variants = list(job["variants"])
    variant_by_id = {str(variant["id"]): variant for variant in variants}
    reference_id = str(job["reference_variant"])
    reference_mode = str(job.get("reference_mode", "generated"))
    signatures: list[str] = []
    grouped: dict[str, list[dict]] = {}
    for variant in variants:
        signature = kernel_signature(variant)
        if signature not in grouped:
            signatures.append(signature)
            grouped[signature] = []
        grouped[signature].append(variant)
    reference_signature = kernel_signature(variant_by_id[reference_id])
    signatures.remove(reference_signature)
    signatures.insert(0, reference_signature)

    selections: dict[tuple[float, str], SelectionResult] = {}
    frozen_reference_audits: dict[str, dict[str, object]] = {}
    summaries: list[dict[str, object]] = []
    reference_kernel = None
    reference_kernel_metadata = None
    reference_distances = None
    knn_config = global_config.get("knn", {})

    for signature in signatures:
        group = grouped[signature]
        if signature == reference_signature:
            group = sorted(group, key=lambda item: str(item["id"]) != reference_id)
        representative = group[0]
        print(
            f"[v11:job1] dataset={dataset_name} kernel={representative['id']}", flush=True
        )
        started = time.perf_counter()
        kernel, knn_indices, knn_distances, build_metadata = build_kernel_for_variant(
            representative,
            embeddings,
            cache_root,
            dataset_name,
            knn_config,
            force,
        )
        audit = kernel_metrics(kernel, labels)
        audit["build_seconds"] = float(time.perf_counter() - started)
        build_metadata["knn_purity"] = knn_purity(knn_indices, labels)
        if signature == reference_signature:
            reference_kernel = kernel
            reference_kernel_metadata = {"build": build_metadata, "audit": audit}
            reference_distances = knn_distances

        for ratio in ratios:
            budget = budgets[ratio]
            for variant in group:
                variant_id = str(variant["id"])
                constraint = variant.get("constraint", {"mode": "none"})
                mode = str(constraint.get("mode", "none"))
                unsafe_caps = None
                frozen_audit = None
                if variant_id == reference_id and reference_mode == "frozen":
                    frozen_specs = job["frozen_reference_indices"]
                    result, frozen_audit = load_frozen_reference(
                        frozen_specs[ratio_key(ratio)],
                        labels=labels,
                        margins=margins,
                        budget=budget,
                        project_root=project_root,
                        config_dir=config_dir,
                    )
                    frozen_reference_audits[ratio_key(ratio)] = frozen_audit
                elif mode == "none":
                    result = greedy_facility(
                        kernel, labels, budget, unsafe=margins.unsafe
                    )
                elif mode == "hard_cap":
                    source_id = str(constraint["source_variant"])
                    source = selections[(ratio, source_id)]
                    unsafe_caps = unsafe_caps_from_reference(
                        source,
                        labels,
                        margins,
                        budget,
                        float(constraint.get("multiplier", 1.0)),
                    )
                    result = greedy_facility(
                        kernel,
                        labels,
                        budget,
                        unsafe=margins.unsafe,
                        unsafe_caps=unsafe_caps,
                    )
                elif mode == "posthoc_repair":
                    source_id = str(constraint["source_variant"])
                    base_id = str(constraint["base_variant"])
                    source = selections[(ratio, source_id)]
                    base = selections[(ratio, base_id)]
                    unsafe_caps = unsafe_caps_from_reference(
                        source,
                        labels,
                        margins,
                        budget,
                        float(constraint.get("multiplier", 1.0)),
                    )
                    result = posthoc_margin_repair(
                        kernel, labels, base, margins.unsafe, unsafe_caps
                    )
                else:
                    raise ValueError(f"unknown constraint mode: {mode}")
                selections[(ratio, variant_id)] = result

                if variant_id == reference_id and reference_mode == "generated":
                    frozen_specs = job.get("frozen_reference_indices", {})
                    frozen_audit = audit_frozen_reference(
                        result.indices,
                        frozen_specs.get(ratio_key(ratio)),
                        project_root,
                        config_dir,
                    )
                    if frozen_audit is not None:
                        frozen_reference_audits[ratio_key(ratio)] = frozen_audit

                if reference_kernel is None or reference_distances is None:
                    raise AssertionError("reference kernel must be processed first")
                summary = selection_summary(
                    variant=variant,
                    ratio=ratio,
                    budget=budget,
                    selection=result,
                    labels=labels,
                    margins=margins,
                    kernel=kernel,
                    reference_kernel=reference_kernel,
                    reference_selection=selections[(ratio, reference_id)],
                    knn_distances=knn_distances,
                    kernel_build=build_metadata,
                    kernel_audit=audit,
                    unsafe_caps=unsafe_caps,
                )
                is_frozen_reference = (
                    variant_id == reference_id and reference_mode == "frozen"
                )
                summary["selection_source"] = (
                    "frozen_table1" if is_frozen_reference else "v11_generated"
                )
                if is_frozen_reference:
                    summary["frozen_reference"] = frozen_audit
                run_dir = dataset_output / ratio_key(ratio) / variant_id
                run_dir.mkdir(parents=True, exist_ok=True)
                np.save(run_dir / "selected_indices.npy", result.indices)
                np.save(run_dir / "selection_order.npy", result.order)
                write_json(run_dir / "selection_metrics.json", summary)
                write_json(run_dir / "variant_config.json", variant)
                summaries.append({"dataset": dataset_name, **summary})

        if signature != reference_signature:
            del kernel

    diagnostics = {
        "schema": "graphcov-v11/dataset-diagnostics-v1",
        "dataset": dataset_name,
        "n_samples": int(len(labels)),
        "n_classes": int(len(classes)),
        "class_counts": {
            int(cls): int(np.sum(labels == cls)) for cls in classes
        },
        "ratios": ratios,
        "budget_per_class": budgets,
        "embedding_path": str(embedding_path),
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "label_archive": str(label_archive),
        "train_margin": train_margin,
        "reference_variant": reference_id,
        "reference_mode": reference_mode,
        "reference_kernel": reference_kernel_metadata,
        "frozen_reference_audits": frozen_reference_audits,
    }
    write_json(dataset_output / "dataset_diagnostics.json", diagnostics)
    return summaries


def flatten_summary(summary: dict[str, object]) -> dict[str, object]:
    margin = summary["margin"]
    own = summary["self_kernel_coverage"]
    reference = summary["reference_kernel_coverage"]
    audit = summary["kernel_audit"]
    return {
        "dataset": summary["dataset"],
        "ratio": summary["ratio"],
        "variant": summary["variant"]["id"],
        "k": summary["variant"]["k"],
        "hops": summary["variant"]["kernel"]["hops"],
        "weight_mode": summary["variant"]["kernel"].get("weight_mode", "uniform"),
        "hop_decay": summary["variant"]["kernel"].get("hop_decay", 1.0),
        "truncation": summary["variant"]["kernel"].get("truncation", "none"),
        "max_degree": summary["variant"]["kernel"].get("max_degree"),
        "constraint": summary["variant"].get("constraint", {}).get("mode", "none"),
        "n_selected": summary["n_selected"],
        "jaccard_reference": summary["jaccard_with_reference"],
        "unsafe_ratio": margin["unsafe_ratio"],
        "margin_gap_mean": margin["gap_mean"],
        "self_macro": own["macro_class_mean"],
        "self_p10": own["p10"],
        "reference_macro": reference["macro_class_mean"],
        "reference_p10": reference["p10"],
        "reference_worst": reference["worst_class_mean"],
        "reference_worst_p10": reference["worst_class_p10"],
        "kernel_nnz": audit["nnz"],
        "kernel_cross_mass": audit["cross_class_mass_ratio"],
        "kernel_asymmetry": audit["sampled_relative_asymmetry"],
        "selected_knn_distance": summary["selected_knn_squared_distance_mean"],
    }


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = load_json(config_path)
    if config.get("schema") != "graphcov-v11/job1-config-v1":
        raise ValueError("unexpected job1 config schema")
    config_dir = config_path.parent
    datasets_path = resolve_path(config["datasets_file"], project_root, config_dir)
    datasets = load_json(datasets_path)["datasets"]
    output_root = resolve_path(config["output_root"], project_root, config_dir)
    cache_root = resolve_path(config["cache_root"], project_root, config_dir)
    jobs = [job for job in config["jobs"] if not args.only_dataset or job["dataset"] == args.only_dataset]
    for job in jobs:
        validate_job(job)
    if args.dry_run:
        print(json.dumps(json_ready(jobs), indent=2))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    all_summaries: list[dict[str, object]] = []
    for job in jobs:
        dataset_name = str(job["dataset"])
        if dataset_name not in datasets:
            raise KeyError(f"dataset missing from registry: {dataset_name}")
        all_summaries.extend(
            process_job(
                job=job,
                dataset_spec=datasets[dataset_name],
                output_root=output_root,
                cache_root=cache_root,
                project_root=project_root,
                config_dir=datasets_path.parent,
                global_config=config,
                force=args.force,
            )
        )
    rows = [flatten_summary(summary) for summary in all_summaries]
    results_name = f"results_{args.only_dataset}.csv" if args.only_dataset else "results.csv"
    if rows:
        with (output_root / results_name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "schema": "graphcov-v11/job1-manifest-v1",
        "config": str(config_path),
        "config_sha256": sha256_bytes(config_path.read_bytes()),
        "datasets_file": str(datasets_path),
        "output_root": str(output_root),
        "cache_root": str(cache_root),
        "n_completed_selections": len(rows),
    }
    manifest_name = f"manifest_{args.only_dataset}.json" if args.only_dataset else "manifest.json"
    write_json(output_root / manifest_name, manifest)
    print(f"[v11:job1] complete: {output_root / results_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
