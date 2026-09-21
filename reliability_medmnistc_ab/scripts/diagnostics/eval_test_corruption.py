#!/usr/bin/env python
"""Re-evaluate existing abc_select checkpoints on held-out test split, clean
and MedMNIST-C corrupted. Zero new training: loads final.pt, runs inference
only. Answers item 1 of the 2026-09-20 diagnostic plan -- the 80 checkpoints
so far were only ever evaluated on eval_split='val' with no corruption BA,
despite the corruption-BA axis being the whole point of the project.

Reuses model/data/metric plumbing directly from reliability_medmnistc's
pipeline module so results are bit-for-bit comparable to the three-arm
multiseed metrics.jsonl schema (ba, worst_recall, ece15, clean_ba_drop, ...).
"""
from __future__ import annotations

import json
import ast
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

PIPE_REPO = Path(os.environ["PIPE_REPO_ROOT"])
sys.path.insert(0, str(PIPE_REPO))

# pipeline.py has no `if __name__ == "__main__"` guard. Importing it would
# execute its full select/train/evaluate loop. Load only the definitions that
# appear before the main execution block, while keeping its globals/imports.
ABC_ROOT = Path(os.environ.get("ABC_ROOT", "/project/prj-sis01/xuxiaoyu/abc_select"))
DATASET = os.environ["EVAL_DATASET"]
os.environ["DATASETS"] = DATASET
os.environ["PHASE"] = "validate"
os.environ["RUN_FULL_TRAIN"] = "0"

pipeline_path = PIPE_REPO / "reliability_medmnistc" / "reliability" / "pipeline.py"
pipeline_tree = ast.parse(pipeline_path.read_text(encoding="utf-8"), filename=str(pipeline_path))
main_index = next(
    index for index, node in enumerate(pipeline_tree.body)
    if isinstance(node, ast.If)
    and isinstance(node.test, ast.Compare)
    and isinstance(node.test.left, ast.Name)
    and node.test.left.id == "PHASE"
    and any(isinstance(op, ast.Eq) for op in node.test.ops)
    and any(isinstance(value, ast.Constant) and value.value == "summarize" for value in node.test.comparators)
)
pipeline_tree.body = pipeline_tree.body[:main_index]
pipeline_module = types.ModuleType("reliability_medmnistc.reliability.pipeline_definitions")
pipeline_module.__file__ = str(pipeline_path)
exec(compile(pipeline_tree, str(pipeline_path), "exec"), pipeline_module.__dict__)
P = pipeline_module

ARMS = ["baseline_graph_a2", "A_metric_eta025", "A_metric_eta050", "A_metric_eta075"]
SEEDS = [42, 43, 44, 45, 46]


def main():
    info = P.INFO[DATASET]
    num_classes = len(info["label"])
    in_channels = info["n_channels"]

    test_ds, _ = P.load_med(DATASET, "test", size=P.SIZE, source_override="clean")
    y_test = P.flat_labels(test_ds)
    test_indices = np.arange(len(y_test), dtype=np.int64)
    corruptions = P.ensure_corruptions(DATASET, y_test)

    results = []
    for arm in ARMS:
        for seed in SEEDS:
            run_dir = ABC_ROOT / "runs" / "train" / arm / DATASET / f"seed_{seed}"
            ckpt_path = run_dir / "final.pt"
            if not ckpt_path.is_file():
                print(f"MISSING checkpoint: {ckpt_path}", flush=True)
                continue

            checkpoint = torch.load(ckpt_path, map_location=P.DEVICE)
            model = P.ResNet18WithFeatures(num_classes, in_channels, pretrained=False).to(P.DEVICE)
            model.load_state_dict(checkpoint["model"])
            model.eval()

            meta = {"dataset": DATASET, "arm": arm, "training_seed": seed,
                     "source_checkpoint": str(ckpt_path)}

            y, logits = P.predict(model, test_ds)
            if not np.array_equal(y, y_test):
                raise RuntimeError(f"clean label order mismatch: {run_dir}")
            clean_row, _, _ = P.metric_row(y, logits, num_classes)
            clean_row.update(meta, corruption="clean", severity=0)
            results.append(clean_row)

            for corruption in corruptions:
                corruption_dataset = P.CorruptedMedMNIST(
                    DATASET, corruption, norm_mean=[0.5] * in_channels,
                    norm_std=[0.5] * in_channels, root=str(P.CORR_ROOT),
                    as_rgb=P.DATASET_RGB[DATASET], mmap_mode="r",
                )
                view, _, _ = P.make_corruption_view(corruption_dataset, test_indices, len(y_test))
                cy, clogits = P.predict(model, view)
                if not np.array_equal(cy, np.tile(y_test, 5)):
                    raise RuntimeError(f"corruption label order mismatch: {DATASET}/{corruption}")
                for severity in range(1, 6):
                    start, end = (severity - 1) * len(y_test), severity * len(y_test)
                    crow, _, _ = P.metric_row(cy[start:end], clogits[start:end], num_classes)
                    crow.update(meta, corruption=corruption, severity=severity)
                    crow["clean_ba_drop"] = float(clean_row["ba"] - crow["ba"])
                    crow["clean_worst_recall_drop"] = float(clean_row["worst_recall"] - crow["worst_recall"])
                    results.append(crow)
                del corruption_dataset, view

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"done arm={arm} seed={seed} clean_test_ba={clean_row['ba']:.4f}", flush=True)

    out_path = ABC_ROOT / "results_test_corruption" / f"{DATASET}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in results:
            f.write(json.dumps(row, default=str) + "\n")
    print(f"wrote {len(results)} rows to {out_path}")


if __name__ == "__main__":
    main()
