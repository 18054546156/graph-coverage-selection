"""S0 of proposal_shift_robustness_20261002: is MedMNIST-C EXP the right SIZE of shift?

Per-image colour statistics (224px):
  R, G, B means; HSV saturation and value means; hue as a circular mean (cos, sin);
  luminance std (contrast proxy).
Gap between two image sets = standardized mean difference per feature, using the
TRAIN set's per-feature sd, and its L2 norm over features ("gap").

Reference gaps:
  floor   train -> val   (same centre, sampling noise only)
  real    train -> test  (pathmnist: test is a different centre; blood: same centre)
Synthetic gaps:
  clean test -> corrupted test, each EXP corruption x severity 1-5
Verdict per dataset: which severities fall inside the real cross-centre gap.

CPU only. Usage: python s0_colour_gap.py [--datasets pathmnist bloodmnist] [--per-set 3000]
"""
import argparse
import json
import os
import zipfile

import numpy as np

DATA = "/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/data"
EXP = ["brightness_down", "brightness_up", "contrast_down", "contrast_up", "saturate"]
FEATURES = ["R", "G", "B", "sat", "val", "hue_cos", "hue_sin", "lum_std"]


def load_member(path, name):
    with zipfile.ZipFile(path) as archive, archive.open(name + ".npy") as handle:
        return np.lib.format.read_array(handle)


def member_shape(path, name):
    with zipfile.ZipFile(path) as archive, archive.open(name + ".npy") as handle:
        version = np.lib.format.read_magic(handle)
        shape, _, _ = np.lib.format._read_array_header(handle, version)
    return shape


def stream_features(path, name, rows, chunk=256):
    """Features of the given (sorted) rows of a zipped uint8 image array, streamed in
    chunks so peak memory is one chunk, not the whole array (qos-normal caps at 4G)."""
    rows = np.asarray(rows)
    out = np.empty((len(rows), len(FEATURES)))
    with zipfile.ZipFile(path) as archive, archive.open(name + ".npy") as handle:
        version = np.lib.format.read_magic(handle)
        shape, fortran, dtype = np.lib.format._read_array_header(handle, version)
        if fortran or dtype != np.uint8:
            raise RuntimeError("unexpected array layout in {}:{}".format(path, name))
        row_shape = shape[1:]
        row_bytes = int(np.prod(row_shape))
        cursor = 0
        for start in range(0, shape[0], chunk):
            count = min(chunk, shape[0] - start)
            buf = np.frombuffer(handle.read(count * row_bytes), np.uint8).reshape((count,) + row_shape)
            hit = rows[(rows >= start) & (rows < start + count)]
            if len(hit):
                block = buf[hit - start]
                if block.ndim == 3:
                    block = np.repeat(block[..., None], 3, axis=-1)
                out[cursor:cursor + len(hit)] = image_features(block)
                cursor += len(hit)
    if cursor != len(rows):
        raise RuntimeError("stream read {} of {} rows from {}".format(cursor, len(rows), path))
    return out


def image_features(images):
    """images: (N, H, W, 3) uint8 -> (N, 8) float64."""
    out = np.empty((len(images), len(FEATURES)))
    for start in range(0, len(images), 250):
        x = images[start:start + 250].astype(np.float32) / 255.0
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        cmax, cmin = x.max(-1), x.min(-1)
        delta = cmax - cmin
        sat = np.where(cmax > 0, delta / np.maximum(cmax, 1e-8), 0.0)
        safe = np.maximum(delta, 1e-8)
        hue = np.where(cmax == r, ((g - b) / safe) % 6,
              np.where(cmax == g, (b - r) / safe + 2, (r - g) / safe + 4)) / 6.0
        weight = (delta > 1e-3).astype(np.float32)
        angle = 2 * np.pi * hue
        wsum = np.maximum(weight.sum((1, 2)), 1.0)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        block = slice(start, start + len(x))
        out[block, 0] = r.mean((1, 2))
        out[block, 1] = g.mean((1, 2))
        out[block, 2] = b.mean((1, 2))
        out[block, 3] = sat.mean((1, 2))
        out[block, 4] = cmax.mean((1, 2))
        out[block, 5] = (np.cos(angle) * weight).sum((1, 2)) / wsum
        out[block, 6] = (np.sin(angle) * weight).sum((1, 2)) / wsum
        out[block, 7] = lum.std((1, 2))
    return out


def smd(reference, target, scale):
    return (target.mean(0) - reference.mean(0)) / scale


def fmt(vec):
    return " ".join("{:+.2f}".format(v) for v in vec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["pathmnist", "bloodmnist"])
    ap.add_argument("--per-set", type=int, default=3000)
    ap.add_argument("--out", default=os.path.expanduser("~/corr/s0_colour_gap.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    report = {}
    for ds in args.datasets:
        clean_path = "{}/medmnist/{}_224.npz".format(DATA, ds)
        feats = {}
        for split in ["train", "val", "test"]:
            total = member_shape(clean_path, split + "_images")[0]
            pick = np.sort(rng.choice(total, min(args.per_set, total), replace=False))
            if split == "test":
                test_pick, test_count = pick, total
            feats[split] = stream_features(clean_path, split + "_images", pick)
            print(ds, split, "features done", flush=True)
        scale = feats["train"].std(0) + 1e-8
        floor = smd(feats["train"], feats["val"], scale)
        real = smd(feats["train"], feats["test"], scale)
        entry = {"features": FEATURES,
                 "floor_train_val": {"smd": floor.tolist(), "gap": float(np.linalg.norm(floor))},
                 "real_train_test": {"smd": real.tolist(), "gap": float(np.linalg.norm(real))},
                 "exp": {}}
        print("\n{}  features {}".format(ds, FEATURES))
        print("  floor train->val   gap {:.3f}  [{}]".format(entry["floor_train_val"]["gap"], fmt(floor)))
        print("  real  train->test  gap {:.3f}  [{}]".format(entry["real_train_test"]["gap"], fmt(real)))
        for corruption in EXP:
            corr_path = "{}/medmnistc/{}/{}.npz".format(DATA, ds, corruption)
            if not os.path.exists(corr_path):
                continue
            labels = load_member(corr_path, "test_labels").ravel()
            clean_labels = load_member(clean_path, "test_labels").ravel()
            if len(labels) != 5 * test_count or not np.array_equal(labels, np.tile(clean_labels, 5)):
                raise RuntimeError("unexpected severity layout in " + corr_path)
            all_rows = np.concatenate([(s - 1) * test_count + test_pick for s in range(1, 6)])
            all_feats = stream_features(corr_path, "test_images", all_rows)
            entry["exp"][corruption] = {}
            for severity in range(1, 6):
                f = all_feats[(severity - 1) * len(test_pick):severity * len(test_pick)]
                vs_clean = smd(feats["test"], f, scale)
                vs_train = smd(feats["train"], f, scale)
                entry["exp"][corruption][severity] = {
                    "smd_vs_clean_test": vs_clean.tolist(), "gap_vs_clean_test": float(np.linalg.norm(vs_clean)),
                    "smd_vs_train": vs_train.tolist(), "gap_vs_train": float(np.linalg.norm(vs_train))}
                print("  {:16s} s{}  gap(clean test->corr) {:.3f}  gap(train->corr) {:.3f}  [{}]".format(
                    corruption, severity, np.linalg.norm(vs_clean), np.linalg.norm(vs_train), fmt(vs_clean)),
                    flush=True)
        report[ds] = entry
    json.dump(report, open(args.out, "w"), indent=1)
    print("wrote " + args.out)


if __name__ == "__main__":
    main()
