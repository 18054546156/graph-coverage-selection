#!/usr/bin/env python
"""Frozen-encoder embeddings for the encoder-robustness route (round 3).

Writes  <out>/<ds>_<split>_<enc>_224.npz  with key 'embeddings' (float32, n x d),
row order == the MedMNIST+ 224 array order, the same convention as the archived
UNI cache (<ds>_train_uni_224.npz), so every downstream selector is unchanged.

Preprocessing mirrors graphcov/run/embeddings.py for UNI: uint8 -> [0,1],
grayscale repeated to 3 channels, per-encoder mean/std (timm's own config),
native 224 input, no crop, fp32. Feature = timm pooled pre-head output
(num_classes=0): the CLS token for both ViT-L encoders.

UNI is extracted for val only in production (train already archived); a
--limit smoke on train reproduces the archived UNI rows as a preprocessing check.

Only train and val splits are extracted. The test split is never touched.
"""
import argparse
import os
import time

import numpy as np
import torch

ENCODERS = {
    "uni":    "hf-hub:MahmoodLab/UNI",               # histopathology (H&E), SSL; val split only
    "dinov2": "vit_large_patch14_dinov2.lvd142m",   # natural images, SSL
    "clip":   "vit_large_patch14_clip_224.openai",   # natural images, language-supervised
}
DATA = os.environ.get("MEDMNIST_ROOT", "/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/data/medmnist")


def load_split(ds, split):
    f = np.load(os.path.join(DATA, f"{ds}_224.npz"))
    return f[f"{split}_images"]


@torch.no_grad()
def embed(model, imgs, mean, std, bs, dev):
    mean = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(std, device=dev).view(1, 3, 1, 1)
    out = []
    for s in range(0, len(imgs), bs):
        x = torch.from_numpy(np.ascontiguousarray(imgs[s:s + bs])).to(dev)
        x = x.float().div_(255.0)
        x = x.unsqueeze(1).repeat(1, 3, 1, 1) if x.ndim == 3 else x.permute(0, 3, 1, 2)
        x = (x - mean) / std
        f = model.forward_features(x)[:, 0]         # CLS token, as graphcov does for UNI
        out.append(f.float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def main():
    import timm
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS))
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N rows")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=256)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    for enc in a.encoders:
        if enc == "uni":   # exactly graphcov.run.embeddings.get_uni_model
            m = timm.create_model(ENCODERS[enc], pretrained=True,
                                  init_values=1e-5, dynamic_img_size=True)
            cfg = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        else:
            m = timm.create_model(ENCODERS[enc], pretrained=True, num_classes=0,
                                  img_size=224)
            cfg = timm.data.resolve_data_config({}, model=m)
        m = m.eval().to(dev)
        for ds in a.datasets:
            for split in a.splits:
                p = os.path.join(a.out, f"{ds}_{split}_{enc}_224.npz")
                if os.path.exists(p):
                    print("exists", p); continue
                t = time.time()
                imgs = load_split(ds, split)
                if a.limit:
                    imgs = imgs[:a.limit]
                    p = p.replace("_224.npz", f"_224_smoke{a.limit}.npz")
                Z = embed(m, imgs, cfg["mean"], cfg["std"], a.bs, dev)
                assert len(Z) == len(imgs) and np.isfinite(Z).all()
                tmp = p + ".tmp.npz"
                np.savez(tmp, embeddings=Z)
                os.replace(tmp, p)
                print(f"{enc} {ds} {split} {Z.shape} {time.time()-t:.0f}s -> {p}",
                      flush=True)
                del imgs
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
