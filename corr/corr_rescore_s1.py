#!/usr/bin/env python
"""S1 (proposal_shift_robustness_20261002.md): MedMNIST-C re-score of the
FORMAL seed 42-46 checkpoints for all 9 methods (8 bench + mv_mean), blood
and path, both ratios. Pure inference: trains nothing.

WHY A SEPARATE SCRIPT FROM corr_rescore.py
-------------------------------------------
corr_rescore.py's `discover()` globs a single flat archive
(table1_clean_only_20260922) whose universal seed set is {0,1,2,3,4,5,6,2026};
42-44 are sporadic there and 45/46 do not exist at all. The formal 42-46
checkpoints this project actually reports in the paper (table1_s4246_a100.json,
bench8_per_seed_a100) live in FIVE different run trees with TWO different
directory conventions, because graph_a2 and mv_mean were trained in separate
rounds (round1/round2/mvf/plus_a100) from the 7 other baselines (table1
s4246 + its A100 rerun). This script's discover() walks all five trees.

TREE 1 -- 7 baselines (el2n_top, eva, facility, forgetting, fps, herding,
random), convention formal/<ds>/seed_<s>/results/<ds>/<arm>/ratio_<r>/
selection_seed_*/train_seed_<s>/aug_0/final.pt:
  table1_a100_20260929  (A100 rerun of the 140 cells that were originally
                          H100 -- PREFERRED when present, matches
                          fig_data.py's a100_only_table1 override)
  table1_s4246_20260928 (original mixed A100/H100 tree -- fallback)

TREE 2 -- graph_a2 and mv_mean, convention formal/ratio_<r>/<ds>/<arm>/
seed_<s>/results/<ds>/precomputed/ratio_<r>/selection_seed_*/train_seed_<s>/
aug_0/final.pt (arm name is a path segment; run_config.json's "method" field
says "precomputed" for all of these, so arm is read from the path, not cfg):
  tdgs_round1_20260927  graph_a2, ratio 0.02, seeds 42-44
  round2_20260927       graph_a2, ratio 0.02 seeds 45-46 + ratio 0.05 all seeds
  mvf_20260930          mv_mean,  blood r0.05 + path r0.02, seeds 42-46
  plus_a100_20260930    mv_mean,  blood r0.02 + path r0.05, seeds 42-46
Verified by filesystem walk 2026-10-02: each (dataset, ratio, arm, seed) cell
in the 2x2x9x5=180 grid resolves to exactly one final.pt across these trees;
no cell is missing, none needed retraining.

Everything else (pipeline import neutralisation, metric computation, output
schema, resumability) is unchanged from corr_rescore.py so the rows are
directly comparable / mergeable with the existing precheck_all_corr.json.
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np

BASE = "/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab"
SRC = BASE + "/sources/table1_clean_only_20260922"
SHARED = "/mnt/prj01/hgrp-1502-5TB/tdgs_shared"

BASELINE_ARMS = ["graph_a2", "herding", "facility", "fps", "random", "eva",
                  "el2n_top", "forgetting"]
ARMS = BASELINE_ARMS + ["mv_mean"]
DATASETS = ["pathmnist", "bloodmnist"]
RATIOS = [0.02, 0.05]
SEEDS = [42, 43, 44, 45, 46]

# Tree 1: 7 non-graph_a2 baselines. NEW (A100 rerun) takes priority over OLD.
TREE1_NEW = SHARED + "/runs/table1_a100_20260929/formal"
TREE1_OLD = SHARED + "/runs/table1_s4246_20260928/formal"
TREE1_ARM_RE = re.compile(
    r"formal/(?P<dataset>[^/]+)/seed_(?P<seed>\d+)/results/[^/]+/(?P<arm>[^/]+)/"
    r"ratio_(?P<ratio>[\d.]+)/selection_seed_\d+/train_seed_(?P<trainseed>\d+)/"
    r"aug_\d+/final\.pt$")

# Tree 2: graph_a2 + mv_mean, "flat" convention. Order does not matter --
# cells are disjoint by construction (see module docstring), verified below.
FLAT_ROOTS = [
    BASE + "/tdgs_round1_20260927/formal",
    SHARED + "/runs/round2_20260927/formal",
    SHARED + "/runs/mvf_20260930/formal",
    SHARED + "/runs/plus_a100_20260930/formal",
]
FLAT_ARM_RE = re.compile(
    r"formal/ratio_(?P<ratio>[\d.]+)/(?P<dataset>[^/]+)/(?P<arm>[^/]+)/"
    r"seed_(?P<seed>\d+)/results/[^/]+/precomputed/ratio_[\d.]+/"
    r"selection_seed_\d+/train_seed_(?P<trainseed>\d+)/aug_\d+/final\.pt$")

# CORR_OUT_ROOT overrides the output tree. Only xiaoyuxu2 can write under BASE, so
# jobs on qiangzeng/danranwang must point it at the group-writable shared tree.
OUT_ROOT = os.environ.get("CORR_OUT_ROOT", BASE + "/table1_corruption_s1_20261002")

# S2 (colour-augmented retraining): every arm is trained via `precomputed` replaying
# the S1 run's own selected_local_indices.npy, so all 9 arms use the flat layout.
S2_ROOT = SHARED + "/runs/s2_coloraug_20261002/formal"
S2_OUT_ROOT = os.environ.get("CORR_OUT_ROOT", SHARED + "/runs/corr_s2_coloraug_20261002")
NEWLINE = chr(10)


def candidate_from_match(ckpt_path, match, datasets, ratios, arms, seeds):
    dataset, arm = match.group("dataset"), match.group("arm")
    ratio = float(match.group("ratio"))
    seed = int(match.group("seed"))
    trainseed = int(match.group("trainseed"))
    if dataset not in datasets or arm not in arms or seed not in seeds:
        return None
    if trainseed != seed:
        return None
    if not any(abs(ratio - want) < 1e-9 for want in ratios):
        return None
    run_dir = os.path.dirname(ckpt_path)
    config_path = os.path.join(run_dir, "run_config.json")
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    clean_pred_path = os.path.join(run_dir, "predictions_clean.npz")
    if not (os.path.exists(config_path) and os.path.exists(metrics_path)
            and os.path.exists(clean_pred_path)):
        print("  SKIP (missing sibling file): " + run_dir)
        return None
    config = dict(json.load(open(config_path)))
    config["dataset"] = dataset
    config["method"] = arm
    config["ratio"] = round(ratio, 2)
    config["training_seed"] = seed
    clean_row = None
    for line in open(metrics_path):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (parsed.get("corruption", "clean") == "clean"
                and parsed.get("severity", 0) in (0, None)):
            clean_row = parsed
            break
    if clean_row is None:
        print("  SKIP (no clean row in metrics.jsonl): " + run_dir)
        return None
    return dict(ckpt=ckpt_path, run_dir=run_dir, config=config,
                clean=clean_row, clean_pred=clean_pred_path)


def discover(datasets, ratios, arms, seeds, tree="s1"):
    cells = {}  # (dataset, ratio, arm, seed) -> record; first writer wins (tree priority order)

    def maybe_add(record):
        key = (record["config"]["dataset"], record["config"]["ratio"],
               record["config"]["method"], record["config"]["training_seed"])
        if key not in cells:
            cells[key] = record

    if tree == "s2":
        for ckpt_path in glob.glob(S2_ROOT + "/**/final.pt", recursive=True):
            match = FLAT_ARM_RE.search(ckpt_path.replace(chr(92), "/"))
            if not match:
                continue
            record = candidate_from_match(ckpt_path, match, datasets, ratios, set(arms), seeds)
            if record:
                maybe_add(record)
        return sorted(cells.values(), key=lambda rec: (
            rec["config"]["dataset"], rec["config"]["ratio"],
            rec["config"]["method"], rec["config"]["training_seed"]))

    # graph_a2 comes only from the flat trees (same source as the mv_mean paired tables)
    baseline_arms = (set(arms) & set(BASELINE_ARMS)) - {"graph_a2"}
    if baseline_arms:
        for root in (TREE1_NEW, TREE1_OLD):  # NEW first -> NEW wins ties
            for ckpt_path in glob.glob(root + "/**/final.pt", recursive=True):
                match = TREE1_ARM_RE.search(ckpt_path.replace(chr(92), "/"))
                if not match:
                    continue
                record = candidate_from_match(ckpt_path, match, datasets, ratios,
                                              baseline_arms, seeds)
                if record:
                    maybe_add(record)

    flat_arms = set(arms) & {"graph_a2", "mv_mean"}
    if flat_arms:
        for root in FLAT_ROOTS:
            for ckpt_path in glob.glob(root + "/**/final.pt", recursive=True):
                match = FLAT_ARM_RE.search(ckpt_path.replace(chr(92), "/"))
                if not match:
                    continue
                record = candidate_from_match(ckpt_path, match, datasets, ratios,
                                              flat_arms, seeds)
                if record:
                    maybe_add(record)

    records = list(cells.values())
    records.sort(key=lambda rec: (rec["config"]["dataset"], rec["config"]["ratio"],
                                   rec["config"]["method"], rec["config"]["training_seed"]))
    return records


def out_dir_for(config):
    return os.path.join(OUT_ROOT, config["dataset"], "r" + str(config["ratio"]),
                        config["method"], "seed_" + str(config["training_seed"]))


def load_existing_rows(path):
    rows = {}
    if not os.path.exists(path):
        return rows
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (str(parsed.get("corruption")), int(parsed.get("severity", 0)))
        rows[key] = parsed
    return rows


def save_rows(path, rows):
    tmp_path = path + ".tmp"
    handle = open(tmp_path, "w")
    for key in sorted(rows, key=lambda item: (item[0], item[1])):
        handle.write(json.dumps(rows[key], sort_keys=True))
        handle.write(NEWLINE)
    handle.close()
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--ratios", nargs="+", type=float, default=RATIOS)
    parser.add_argument("--arms", nargs="+", default=ARMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--limit-corruptions", type=int, default=0,
                        help="debug only: cap number of corruption types evaluated")
    parser.add_argument("--discover-only", action="store_true",
                        help="list discovered checkpoints and exit (no pipeline import, no GPU)")
    parser.add_argument("--tree", choices=["s1", "s2"], default="s1",
                        help="s1 = original checkpoints; s2 = colour-augmented retrains")
    args = parser.parse_args()
    global OUT_ROOT
    if args.tree == "s2":
        OUT_ROOT = S2_OUT_ROOT

    if args.discover_only:
        records = discover(set(args.datasets), args.ratios, set(args.arms), set(args.seeds), args.tree)
        want = {(d, r, a, s) for d in args.datasets for r in args.ratios
                for a in args.arms for s in args.seeds}
        have = {(rec["config"]["dataset"], rec["config"]["ratio"], rec["config"]["method"],
                 rec["config"]["training_seed"]) for rec in records}
        print("discovered {} of {} wanted checkpoints".format(len(have & want), len(want)))
        for key in sorted(have):
            match = [rec for rec in records if (rec["config"]["dataset"], rec["config"]["ratio"],
                     rec["config"]["method"], rec["config"]["training_seed"]) == key][0]
            print("  HAVE {} {} {:10s} s{}  {}".format(key[0], key[1], key[2], key[3], match["ckpt"]))
        for key in sorted(want - have):
            print("  MISSING {} {} {:10s} s{}".format(*key))
        return

    os.environ.setdefault("MEDMNIST_ROOT", BASE + "/data/medmnist")
    os.environ.setdefault("MEDMNISTC_ROOT", BASE + "/data/medmnistc")
    os.environ.setdefault("MEDC_ROOT", SRC + "/third_party/medmnistc")

    os.environ["PHASE"] = "evaluate"
    os.environ["METHODS"] = ""
    os.environ["RUN_FULL_TRAIN"] = "0"
    os.environ["CLEAN_ONLY"] = "1"
    os.environ["AUTO_CONSOLIDATE"] = "0"
    os.environ["DATASETS"] = "bloodmnist"
    sys.path.insert(0, SRC)
    sys.path.insert(0, "/project/prj-sis01/xuxiaoyu/graph_select")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import corr_compat
    print("wand backend: " + str(corr_compat.install()))
    import_start = time.time()
    from reliability_medmnistc.reliability import pipeline as P
    print("pipeline imported in {:.0f}s".format(time.time() - import_start))
    if P.METHODS or P.PHASE != "evaluate" or P.RUN_FULL_TRAIN:
        raise RuntimeError(
            "pipeline import was NOT neutralised (PHASE={!r}, METHODS={}, "
            "RUN_FULL_TRAIN={}). Refusing.".format(P.PHASE, P.METHODS, P.RUN_FULL_TRAIN))
    from medmnistc.dataset import CorruptedMedMNIST
    from medmnistc.corruptions.registry import CORRUPTIONS_DS, DATASET_RGB

    records = discover(set(args.datasets), args.ratios, set(args.arms), set(args.seeds), args.tree)
    want_total = len(args.datasets) * len(args.ratios) * len(args.arms) * len(args.seeds)
    print("discovered {} checkpoints (want {}x{}x{}x{}={})".format(
        len(records), len(args.datasets), len(args.ratios), len(args.arms),
        len(args.seeds), want_total))
    per_cell = Counter((rec["config"]["dataset"], rec["config"]["ratio"],
                       rec["config"]["method"]) for rec in records)
    thin = {key: count for key, count in per_cell.items() if count < len(args.seeds)}
    if thin:
        print("  WARNING: {} (dataset,ratio,arm) cells have fewer than {} seeds: {}".format(
            len(thin), len(args.seeds), sorted(thin.items())))
    by_dataset = {}
    for rec in records:
        by_dataset.setdefault(rec["config"]["dataset"], []).append(rec)
    for dataset, recs in by_dataset.items():
        print("  {:14s} {} checkpoints".format(dataset, len(recs)))

    overall_start = time.time()
    images_scored_total = 0
    for dataset, dataset_records in by_dataset.items():
        clean_dataset, info = P.load_med(dataset, "test")
        y_test = P.flat_labels(clean_dataset)
        num_classes = len(info["label"])
        corruptions = sorted(CORRUPTIONS_DS[dataset])
        if args.limit_corruptions:
            corruptions = corruptions[:args.limit_corruptions]
        print(NEWLINE + "=" * 92)
        print("{}: test_count={} classes={} corruptions={}".format(
            dataset, len(y_test), num_classes, corruptions))
        print("=" * 92)

        for rec in dataset_records:
            pred_file = np.load(rec["clean_pred"], allow_pickle=False)
            rec["test_idx"] = np.asarray(pred_file["clean_id"], np.int64)
            stored_y = np.asarray(pred_file["y_true"], np.int64)
            pred_file.close()
            if not np.array_equal(y_test[rec["test_idx"]], stored_y):
                raise RuntimeError(
                    "clean_id/y_true disagree with dataset labels at " + rec["run_dir"]
                    + " -- cannot pair corrupted to clean")
        idx_sets = {hash(rec["test_idx"].tobytes()) for rec in dataset_records}
        print("  distinct clean test index sets across runs: {} (1 is expected)".format(
            len(idx_sets)))

        for corruption in corruptions:
            todo = []
            for rec in dataset_records:
                out_dir = out_dir_for(rec["config"])
                metrics_path = os.path.join(out_dir, "metrics_corr.jsonl")
                conf_path = os.path.join(out_dir, "conf_" + corruption + ".npz")
                rows = load_existing_rows(metrics_path)
                needed = {(corruption, sev) for sev in range(1, 6)}
                if needed.issubset(rows) and os.path.exists(conf_path):
                    continue
                todo.append((rec, out_dir, metrics_path, conf_path, rows))
            if not todo:
                print("  [{}] already complete for all {} runs, skipping".format(
                    corruption, len(dataset_records)))
                continue
            corruption_start = time.time()
            corrupted_dataset = CorruptedMedMNIST(
                dataset, corruption, norm_mean=[0.5] * info["n_channels"],
                norm_std=[0.5] * info["n_channels"],
                root=os.environ["MEDMNISTC_ROOT"],
                as_rgb=DATASET_RGB[dataset], mmap_mode="r")
            print("  [{}] opened in {:.0f}s; {} run(s) to evaluate".format(
                corruption, time.time() - corruption_start, len(todo)))

            for item_index, (rec, out_dir, metrics_path, conf_path, rows) in enumerate(todo):
                config = rec["config"]
                os.makedirs(out_dir, exist_ok=True)
                view, clean_ids, severities = P.make_corruption_view(
                    corrupted_dataset, rec["test_idx"], len(y_test))
                model = P.model_from_checkpoint(
                    rec["ckpt"], num_classes, info["n_channels"], config["config_hash"])
                corrupted_y, corrupted_logits = P.predict(model, view)
                expected_y = np.tile(y_test[rec["test_idx"]], 5)
                if not np.array_equal(corrupted_y, expected_y):
                    raise RuntimeError(
                        "corruption label order mismatch: {}/{} at {}".format(
                            dataset, corruption, rec["run_dir"]))
                corrupted_probs = P.probs_from_logits(corrupted_logits)
                eval_count = len(rec["test_idx"])
                for severity in range(1, 6):
                    lo, hi = (severity - 1) * eval_count, severity * eval_count
                    metric_row, _, _ = P.metric_row(
                        corrupted_y[lo:hi], corrupted_logits[lo:hi], num_classes)
                    base_row = dict(
                        dataset=dataset, ratio=config["ratio"], method=config["method"],
                        selection_seed=config["selection_seed"],
                        training_seed=config["training_seed"],
                        n_selected=config["n_selected"],
                        budget_per_class=config["budget_per_class"],
                        checkpoint="final.pt",
                        corruption=corruption, severity=severity)
                    base_row.update(metric_row)
                    base_row["clean_ba_drop"] = float(
                        rec["clean"]["ba"] - metric_row["ba"])
                    base_row["clean_worst_recall_drop"] = float(
                        rec["clean"]["worst_recall"] - metric_row["worst_recall"])
                    base_row["be"] = float(1.0 - metric_row["ba"])
                    rows[(corruption, severity)] = base_row
                save_rows(metrics_path, rows)
                predicted_label = corrupted_probs.argmax(1).astype(np.int16)
                confidence = corrupted_probs.max(1).astype(np.float32)
                true_class_prob = corrupted_probs[
                    np.arange(len(corrupted_y)), corrupted_y].astype(np.float32)
                tmp_conf_path = conf_path + ".tmp.npz"
                np.savez_compressed(
                    tmp_conf_path, clean_id=clean_ids.astype(np.int64),
                    severity=severities.astype(np.int8),
                    y_true=corrupted_y.astype(np.int16), pred=predicted_label,
                    conf=confidence, true_prob=true_class_prob)
                os.replace(tmp_conf_path, conf_path)
                images_scored_total += len(corrupted_y)
                del model, view
                if (item_index + 1) % 10 == 0 or item_index + 1 == len(todo):
                    elapsed = time.time() - corruption_start
                    print("    {}/{}  {:.0f}s elapsed, {:.1f}s/run, {:.1f}M images scored total".format(
                        item_index + 1, len(todo), elapsed, elapsed / (item_index + 1),
                        images_scored_total / 1e6))
            del corrupted_dataset

    total_elapsed = time.time() - overall_start
    print(NEWLINE + "done: {:.1f}M images in {:.2f}h ({:.0f} img/s)".format(
        images_scored_total / 1e6, total_elapsed / 3600,
        images_scored_total / max(total_elapsed, 1)))
    print("output tree: " + OUT_ROOT)


if __name__ == "__main__":
    main()
