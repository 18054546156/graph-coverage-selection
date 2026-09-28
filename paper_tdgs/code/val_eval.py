#!/usr/bin/env python
"""Val-split evaluation of already-trained cells (prereg Amendment 2).

Every cell saved its final-epoch weights (final.pt). This script re-runs those
weights on the MedMNIST *val* split, so that a per-dataset configuration can be
chosen by val BA without retraining and without touching test labels for the
choice.

Fidelity to the harness (reliability/pipeline.py predict + graphcov data.py):
  model      graphcov ResNet18WithFeatures, re-declared here with identical
             modules and key names ("resnet.*"); conv1 replaced iff in_ch != 3
  transform  ToTensor (uint8 -> float /255, HWC -> CHW) + Normalize(0.5, 0.5)
  data       medmnist `{ds}_224.npz`, as_rgb=False (clean eval path)
  loop       model.eval(), no_grad, sequential batches of 256
The transform is applied to the whole split once (same float32 ops, same order).

Self-check (abort on failure): for the first checkpoint of every dataset the
same code is run on the TEST split and must reproduce predictions_clean.npz
(label order exact; argmax agreement >= --min-agree and |BA - harness BA| <=
--ba-tol). NOT bit-exact across GPUs: on H100, bloodmnist gave max|dlogit|
1.9e-2 with argmax identical, organamnist 3.4e-3 with >=1 near-tie argmax flip
(cuDNN TF32 convolutions; the cells were trained and tested on A100). The
slurm script therefore runs on gpu-a100.
Test outputs are used for nothing else.

Writes one JSON line per checkpoint to <out>/val_metrics.jsonl (resumable) and
val logits to <out>/logits/<key>.npz. Never writes into a run directory.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

PROJECT = os.environ.get("TDGS_PROJECT",
                         "/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab")
DATA = os.environ.get("MEDMNIST_ROOT", f"{PROJECT}/data/medmnist")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NCH = {"bloodmnist": 3, "pathmnist": 3, "organamnist": 1, "organsmnist": 1,
       "tissuemnist": 1, "dermamnist": 3}


class ResNet18WithFeatures(nn.Module):
    """Identical to graphcov/run/embeddings.py:59-76 (pretrained=False)."""

    def __init__(self, num_classes, in_channels=3):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                                          stride=2, padding=3, bias=False)
        self.feature_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        return self.resnet(x)


def load_split(ds, split):
    d = np.load(os.path.join(DATA, f"{ds}_224.npz"))
    x = d[f"{split}_images"]
    y = d[f"{split}_labels"].reshape(-1).astype(np.int64)
    t = torch.from_numpy(x)
    t = t.unsqueeze(1) if t.ndim == 3 else t.permute(0, 3, 1, 2)
    t = t.contiguous().float().div(255)          # ToTensor
    t.sub_(0.5).div_(0.5)                        # Normalize([.5]*c, [.5]*c)
    return t, y


@torch.no_grad()
def predict(model, X, bs=256):
    model.eval()
    out = []
    for s in range(0, len(X), bs):
        out.append(model(X[s:s + bs].to(DEV, non_blocking=True)).cpu().numpy())
    return np.concatenate(out)


def metrics(y, logits, C):
    p = logits.argmax(1)
    rec = [float(np.mean(p[y == c] == c)) for c in range(C) if np.any(y == c)]
    return dict(ba=float(np.mean(rec)), acc=float(np.mean(p == y)),
                worst_recall=float(np.min(rec)), recall=rec)


PAT = [re.compile(r"/ratio_(?P<ratio>[0-9.]+)/(?P<ds>[a-z]+)/(?P<arm>[A-Za-z0-9_]+)/seed_(?P<seed>\d+)/"),
       re.compile(r"/formal/(?P<ds>[a-z]+)/seed_(?P<seed>\d+)/results/[a-z]+/(?P<arm>[A-Za-z0-9_]+)/ratio_(?P<ratio>[0-9.]+)/")]


def cell_of(path):
    for p in PAT:
        m = p.search(path)
        if m:
            g = m.groupdict()
            return g["ds"], float(g["ratio"]), g["arm"], int(g["seed"])
    raise ValueError(f"cannot parse cell from {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--min-agree", type=float, default=0.999,
                    help="min test argmax agreement with the harness predictions")
    ap.add_argument("--ba-tol", type=float, default=1e-3,
                    help="max |test BA - harness test BA| in the self-check")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.makedirs(os.path.join(a.out, "logits"), exist_ok=True)
    out_f = os.path.join(a.out, "val_metrics.jsonl")
    done = set()
    if os.path.exists(out_f):
        done = {json.loads(l)["ckpt"] for l in open(out_f)}

    ckpts = sorted(set(sum((glob.glob(f"{r}/**/final.pt", recursive=True)
                            for r in a.roots), [])))
    by_ds = {}
    for c in ckpts:
        ds = cell_of(c)[0]
        if a.datasets is None or ds in a.datasets:
            by_ds.setdefault(ds, []).append(c)
    print(f"{len(ckpts)} checkpoints; todo per ds:",
          {d: sum(c not in done for c in v) for d, v in by_ds.items()}, flush=True)

    for ds, lst in sorted(by_ds.items()):
        todo = [c for c in lst if c not in done]
        if not todo:
            continue
        t0 = time.time()
        Xva, yva = load_split(ds, "val")
        Xva = Xva.pin_memory() if DEV == "cuda" else Xva
        C = int(max(yva.max(), 0)) + 1
        print(f"\n[{ds}] val {tuple(Xva.shape)} C={C} load {time.time()-t0:.0f}s",
              flush=True)
        checked = False
        for c in todo:
            ds_, ratio, arm, seed = cell_of(c)
            ck = torch.load(c, map_location="cpu", weights_only=False)
            C_ck = ck["state_dict"]["resnet.fc.weight"].shape[0]
            m = ResNet18WithFeatures(C_ck, NCH[ds]).to(DEV)
            m.load_state_dict(ck["state_dict"])
            rec = dict(ckpt=c, dataset=ds, ratio=ratio, arm=arm, seed=seed,
                       ckpt_sha256_16=hashlib.sha256(open(c, "rb").read()).hexdigest()[:16])
            if not checked:        # fidelity check against the harness's test run
                pred_f = os.path.join(os.path.dirname(c), "predictions_clean.npz")
                ref = np.load(pred_f)
                Xte, yte = load_split(ds, "test")
                ids = ref["clean_id"]
                lt = predict(m, Xte[ids])
                del Xte
                same_y = bool(np.array_equal(yte[ids], ref["y_true"]))
                agree = float(np.mean(lt.argmax(1) == ref["logits"].argmax(1)))
                Cte = ref["logits"].shape[1]
                dba = abs(metrics(ref["y_true"], lt, Cte)["ba"]
                          - metrics(ref["y_true"], ref["logits"], Cte)["ba"])
                dmax = float(np.abs(lt - ref["logits"]).max())
                rec["test_check"] = dict(label_order=same_y, argmax_agree=agree,
                                         abs_dBA=dba, max_abs_logit_diff=dmax,
                                         n=int(len(ids)))
                print(f"  CHECK {ds}: labels={same_y} argmax_agree={agree:.6f} "
                      f"|dBA|={dba:.2e} max|dlogit|={dmax:.2e}", flush=True)
                if not (same_y and agree >= a.min_agree and dba <= a.ba_tol):
                    raise SystemExit(f"fidelity check failed on {c}")
                checked = True
            lv = predict(m, Xva)
            rec.update(metrics(yva, lv, C))
            key = hashlib.sha1(c.encode()).hexdigest()[:16]
            np.savez_compressed(os.path.join(a.out, "logits", f"{key}.npz"),
                                logits=lv.astype(np.float32), y=yva)
            rec["logits_key"] = key
            with open(out_f, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  {ds} r{ratio} {arm:14s} s{seed}  valBA={rec['ba']*100:.2f}",
                  flush=True)
        del Xva
    print("done", flush=True)


if __name__ == "__main__":
    main()
