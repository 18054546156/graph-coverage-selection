#!/usr/bin/env python3
"""Collect Graph-A2 raw CSVs and selection artifacts into auditable outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def array_sha(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(f"{values.dtype}|{values.shape}".encode("ascii"))
    digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def ratio_key(value: float) -> str:
    return f"r{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def load_rows(raw_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(raw_root.glob("*/results.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("base_method") == "graph_a2":
                    row["source_csv"] = str(path)
                    rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    rows = load_rows(root / "results" / "raw")
    expected = {
        (dataset, f"{ratio:.2f}", str(seed))
        for dataset in protocol["datasets"]
        for ratio in protocol["ratios"]
        for seed in protocol["training_seeds"]
    }
    actual = {(r["dataset"], f"{float(r['ratio']):.2f}", r["seed"]) for r in rows}
    missing = sorted(expected - actual)
    duplicates = len(rows) - len(actual)

    report_dir = root / "reports"
    report_dir.mkdir(exist_ok=True)
    with (report_dir / "ours_raw_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        if rows:
            fields = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    selection_records = []
    selection_root = root / "selection"
    selection_root.mkdir(exist_ok=True)
    for source in sorted((selection_root).glob("**/selected_indices.npy")):
        relative = source.relative_to(selection_root)
        if len(relative.parts) != 3:
            continue
        dataset, ratio_dir, _ = relative.parts
        ratio = float(ratio_dir[1:].replace("p", "."))
        values = np.asarray(np.load(source), dtype=np.int64).reshape(-1)
        metrics_path = source.parent / "selection_metrics.json"
        record = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        selection_records.append({
            **record,
            "dataset": dataset,
            "ratio": ratio,
            "ordered_index_sha256": array_sha(values),
            "file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "local_path": str(source),
        })
    for source in sorted((root / "results" / "raw").glob("**/selection_artifacts/*.npy")):
        name = source.stem
        parts = name.split("_")
        if len(parts) < 6:
            continue
        dataset = parts[0]
        ratio = next((part[1:].replace("p", ".") for part in parts if part.startswith("r")), None)
        seed = next((part[1:] for part in parts if part.startswith("s")), None)
        if dataset not in protocol["datasets"] or ratio is None or seed is None:
            continue
        values = np.asarray(np.load(source), dtype=np.int64).reshape(-1)
        destination = selection_root / dataset / ratio_key(float(ratio)) / f"seed{seed}.npy"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        selection_records.append({
            "dataset": dataset,
            "ratio": float(ratio),
            "selection_seed": int(seed),
            "n_selected": int(values.size),
            "ordered_index_sha256": array_sha(values),
            "file_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "source": str(source),
            "local_path": str(destination),
        })

    # Array workers may each write a partial manifest; rebuild one deterministic
    # manifest from the synchronized per-condition artifacts during finalization.
    selection_records.sort(key=lambda item: (item.get("dataset", ""), float(item.get("ratio", 0.0)), int(item.get("selection_seed", 0))))
    (selection_root / "selection_manifest.json").write_text(
        json.dumps(
            {"schema": "graphcov-table1-ours-selection-manifest-v1", "records": selection_records},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    summary_rows = []
    paper = protocol["paper_ours_pct"]
    for dataset in protocol["datasets"]:
        for ratio in protocol["ratios"]:
            subset = [r for r in rows if r["dataset"] == dataset and abs(float(r["ratio"]) - ratio) < 1e-12]
            values = [100.0 * float(r["balanced_accuracy"]) for r in subset]
            mean = float(np.mean(values)) if values else None
            std = float(np.std(values, ddof=1)) if len(values) > 1 else None
            paper_mean, paper_std = paper[dataset][f"{ratio:.2f}"]
            summary_rows.append({
                "dataset": dataset,
                "ratio": ratio,
                "n_trials": len(values),
                "training_seeds": sorted(int(r["seed"]) for r in subset),
                "reproduction_mean_pct": mean,
                "reproduction_std_pct": std,
                "paper_mean_pct": paper_mean,
                "paper_std_pct": paper_std,
                "delta_pp": mean - paper_mean if mean is not None else None,
                "within_4pp": abs(mean - paper_mean) <= 4.0 if mean is not None else None,
            })

    output = {
        "schema": "graphcov-table1-ours-summary-v1",
        "protocol": str(root / "protocol.json"),
        "expected_trials": len(expected),
        "actual_graph_a2_rows": len(rows),
        "missing_trials": missing,
        "duplicate_row_count": duplicates,
        "selection_artifacts": selection_records,
        "conditions": summary_rows,
    }
    (report_dir / "summary.json").write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    with (report_dir / "ours_table1.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "dataset", "ratio", "n_trials", "training_seeds",
            "reproduction_mean_pct", "reproduction_std_pct",
            "paper_mean_pct", "paper_std_pct", "delta_pp", "within_4pp",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            output_row = dict(row)
            output_row["training_seeds"] = ",".join(map(str, row["training_seeds"]))
            writer.writerow(output_row)

    lines = [
        "# Graph-A2 Ours Table 1 completion",
        "",
        f"Rows found: {len(rows)}/{len(expected)}; missing: {len(missing)}; duplicate rows: {duplicates}.",
        "",
        "The endpoint is the vendor runner final `balanced_accuracy` field. `best_balanced_accuracy` is retained only as an audit field.",
        "",
        "| Dataset | Ratio | Trials | Reproduction BA | Paper BA | Delta pp |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        rep = "NA" if row["reproduction_mean_pct"] is None else f"{row['reproduction_mean_pct']:.2f}+/-{row['reproduction_std_pct']:.2f}"
        lines.append(f"| {row['dataset']} | {row['ratio']:.0%} | {row['n_trials']} | {rep} | {row['paper_mean_pct']:.1f}+/-{row['paper_std_pct']:.1f} | {row['delta_pp']:.2f} |" if row["delta_pp"] is not None else f"| {row['dataset']} | {row['ratio']:.0%} | {row['n_trials']} | {rep} | {row['paper_mean_pct']:.1f}+/-{row['paper_std_pct']:.1f} | NA |")
    if missing:
        lines.extend(["", "Missing keys:", *[f"- `{item}`" for item in missing]])
    (root / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "expected": len(expected), "missing": len(missing), "selections": len(selection_records)}, indent=2))
    return 0 if not missing and duplicates == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
