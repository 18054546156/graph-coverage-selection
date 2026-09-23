"""Which functional of a selected set predicts downstream accuracy at FIXED budget?

This inverts the methodology that produced this project's eight dead ideas. Those
all ran: invent a score -> build a method -> gate it -> fail. None of them ever
established that the score had *any* relationship to accuracy once the budget was
held fixed. When that was finally tested directly (results/ladder.json, within-cell
regression of BA on covering distortion at fixed dataset/cap/budget/arm) the answer
was r = +0.014, t = 0.21, against a test with power to see r = 0.36. Geometric
coverage does not move accuracy at fixed budget. The 107/120 sign agreement that
looked supportive was herding being better at everything, not distortion causing BA.

So: measure first, build second.

Design notes that matter:

* **The endpoint has no noise.** Probe BA is a deterministic function of the
  selected set -- convex logistic regression, fixed audit split, fixed solver seed.
  So this is function approximation, not a noisy correlation, and the power is far
  better than the ladder's 240-point within-cell test.

* **Perturbation generation is the whole point.** Comparing methods (random vs
  herding) confounds every functional with "which method". Each structured anchor is
  therefore perturbed by swapping m of its points per class for random same-class
  pool points, m in {1,2,5,10,25}. That sweeps the functionals continuously *inside*
  a method family, which is the only way to see a functional's own effect.

* **Two blocks, always.** This project's recurring failure is findings that do not
  replicate. The screen's own conclusion has to replicate across independent
  pool/audit splits before it is worth anything.

* Point-level quantities (purity, margin, density) do not depend on the selection,
  so they are precomputed once per pool and then averaged over the selected points.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import run_linear_probe as R

DATASETS = ("bloodmnist", "organamnist", "organsmnist", "pathmnist", "tissuemnist")
BUDGET = 25
POOL_FRACTION = 0.80
BLOCKS = 2
KNN = 20

FUNCTIONALS = (
    "dist_mean", "dist_p90", "dist_max", "moment1", "moment2", "mmd2",
    "mass_gini", "spread", "centrality", "logdet",
    "purity", "margin", "density", "sel_margin_min", "xclass_sep",
)

# Training-dynamics functionals. The geometric half of the coreset literature is
# what the first version of this screen covered; Forgetting / EL2N / GraNd / AUM /
# CCS / Moderate are all dynamics-based and were missing entirely, which made any
# claim about "what predicts accuracy" answer only half the question.
DYN_FUNCTIONALS = (
    "forget_mean", "forget_sd", "forget_frac_zero",
    "aum_mean", "aum_sd", "el2n_final_mean", "el2n_early_mean",
    "difficulty_ks", "difficulty_w1", "frac_hardest_decile", "frac_easiest_decile",
)


# --------------------------------------------------------------------------- #
# pool-level precomputation (selection-independent)
# --------------------------------------------------------------------------- #

def pool_precompute(z: torch.Tensor, y: np.ndarray, knn: int = KNN) -> dict:
    """Per-point purity / margin / density, and per-class moments.

    Done in row blocks so the n x n distance matrix is never materialised.
    """
    n = z.shape[0]
    yt = torch.as_tensor(y, device=z.device)
    purity = torch.empty(n, device=z.device)
    margin = torch.empty(n, device=z.device)
    density = torch.empty(n, device=z.device)

    step = max(1, int(2e8 // n))
    for lo in range(0, n, step):
        hi = min(n, lo + step)
        d = torch.cdist(z[lo:hi], z)
        d[torch.arange(hi - lo, device=z.device), torch.arange(lo, hi, device=z.device)] = float("inf")
        nd, ni = torch.topk(d, knn, largest=False)
        purity[lo:hi] = (yt[ni] == yt[lo:hi, None]).float().mean(1)
        density[lo:hi] = 1.0 / nd.mean(1).clamp_min(1e-9)
        same = d.clone()
        same[yt[None, :] != yt[lo:hi, None]] = float("inf")
        other = d.clone()
        other[yt[None, :] == yt[lo:hi, None]] = float("inf")
        margin[lo:hi] = other.min(1).values - same.min(1).values
        del d, same, other

    per_class = {}
    for c in np.unique(y):
        idx = torch.as_tensor(np.flatnonzero(y == c), device=z.device)
        pc = z[idx]
        # median pairwise distance on a subsample -> RBF bandwidth, fixed per class
        sub = pc[torch.randperm(len(pc), device=z.device)[:2000]]
        sigma = torch.cdist(sub, sub).median().clamp_min(1e-6)
        per_class[int(c)] = {
            "idx": idx,
            "mean": pc.mean(0),
            "trace_cov": pc.var(0, unbiased=False).sum(),
            "sigma": sigma,
            # E_{p,p'}[k] is constant across selections; kept so mmd2 is a real MMD
            "kpp": torch.exp(-(torch.cdist(sub, sub) ** 2) / (2 * sigma ** 2)).mean(),
        }
    return {"purity": purity, "margin": margin, "density": density, "per_class": per_class}


# --------------------------------------------------------------------------- #
# functionals of one selection
# --------------------------------------------------------------------------- #

def gini(v: torch.Tensor) -> float:
    v = torch.sort(v.double())[0]
    n = len(v)
    if n == 0 or v.sum() <= 0:
        return 0.0
    i = torch.arange(1, n + 1, device=v.device, dtype=torch.float64)
    return float((2 * (i * v).sum()) / (n * v.sum()) - (n + 1) / n)


def functionals(z: torch.Tensor, y: np.ndarray, sel: np.ndarray, pre: dict):
    """All functionals for one selection, plus the within-class Voronoi weights.

    Class-equal averaging, not point-weighted: the endpoint is balanced accuracy,
    so a frequency-weighted functional would be measuring a different target than
    the one being predicted.

    The weights come back from here rather than from R.voronoi_weights because the
    Voronoi assignment is already computed below; calling the reference version
    would copy the whole pool to host memory once per selection.
    """
    acc = {k: [] for k in FUNCTIONALS}
    sel_t = torch.as_tensor(sel, device=z.device)
    weights = np.ones(len(sel), dtype=np.float64)
    for c, info in pre["per_class"].items():
        at = np.flatnonzero(y[sel] == c)
        local = sel_t[torch.as_tensor(at, device=z.device)]
        if len(local) < 2:
            continue
        S = z[local]
        P = z[info["idx"]]
        d = torch.cdist(P, S)                      # n_c x k_c
        nd, ni = d.min(1)[0], d.argmin(1)

        acc["dist_mean"].append(float(nd.mean()))
        acc["dist_p90"].append(float(torch.quantile(nd.float(), 0.90)))
        acc["dist_max"].append(float(nd.max()))
        acc["moment1"].append(float(torch.linalg.vector_norm(S.mean(0) - info["mean"])))
        acc["moment2"].append(float(
            (S.var(0, unbiased=False).sum() - info["trace_cov"]).abs() / info["trace_cov"].clamp_min(1e-9)))

        sig = info["sigma"]
        kss = torch.exp(-(torch.cdist(S, S) ** 2) / (2 * sig ** 2)).mean()
        ksp = torch.exp(-(d.T ** 2) / (2 * sig ** 2)).mean()
        acc["mmd2"].append(float(kss - 2 * ksp + info["kpp"]))

        counts = torch.bincount(ni, minlength=len(local)).float()
        acc["mass_gini"].append(gini(counts))
        # within-class, mean 1 -- identical to R.voronoi_weights' w_within
        cnp = counts.double().cpu().numpy()
        weights[at] = cnp / cnp.sum() * len(cnp) if cnp.sum() > 0 else 1.0

        dss = torch.cdist(S, S)
        k = len(local)
        acc["spread"].append(float(dss.sum() / (k * (k - 1))))
        acc["centrality"].append(float(torch.linalg.vector_norm(S - info["mean"], dim=1).mean()))
        # log-volume of the selected set (DPP diversity); jitter keeps it finite
        G = torch.exp(-(dss ** 2) / (2 * sig ** 2)).double()
        acc["logdet"].append(float(torch.linalg.slogdet(G + 1e-4 * torch.eye(k, device=z.device, dtype=torch.float64))[1]))

        acc["purity"].append(float(pre["purity"][local].mean()))
        acc["margin"].append(float(pre["margin"][local].mean()))
        acc["density"].append(float(pre["density"][local].mean()))
        acc["sel_margin_min"].append(float(pre["margin"][local].min()))

        other = sel_t[torch.as_tensor(y[sel] != c, device=z.device)]
        acc["xclass_sep"].append(float(torch.cdist(S, z[other]).min(1)[0].mean()) if len(other) else 0.0)

    return {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}, weights


# --------------------------------------------------------------------------- #
# training dynamics (the other half of the coreset literature)
# --------------------------------------------------------------------------- #

def load_dynamics(root: Path, dataset: str, n_train: int, early_epoch: int = 20) -> dict:
    """Per-sample training dynamics from a ResNet trained on pixels for 200 epochs.

    These come from a full 200-epoch run on the whole train split, so using them to
    select is only legal under the training-cost framing the rest of this project
    already assumes -- exactly as in Forgetting / EL2N / GraNd, which all require
    the same thing. No test data is involved.

    The 28px variant is used because all five datasets have it; pathmnist also has
    224 but mixing resolutions across datasets would confound the comparison.
    """
    path = root / f"{dataset}_train_dynamics_28_e200_s42.npz"
    d = np.load(path, allow_pickle=False)
    l2 = d["all_l2_scores"]                      # (epochs, N)
    if l2.shape[1] != n_train:
        raise RuntimeError(f"{dataset}: dynamics N={l2.shape[1]} != train N={n_train}")
    return {
        "forget": np.asarray(d["forgetting_scores"], dtype=np.float64),
        "aum": np.asarray(d["aum_scores"], dtype=np.float64),
        "el2n_final": np.asarray(l2[-1], dtype=np.float64),
        "el2n_early": np.asarray(l2[early_epoch], dtype=np.float64),
    }


def dyn_functionals(dyn_pool: dict, sel: np.ndarray, y: np.ndarray) -> dict:
    """Dynamics functionals of one selection, averaged over classes with equal weight.

    `difficulty_ks` / `difficulty_w1` measure how well the selection's difficulty
    distribution matches the pool's. That is CCS's actual claim -- a good coreset
    spans the difficulty range rather than taking the hard tail -- and it is the
    dynamics analogue of a coverage functional, so it belongs in the screen as much
    as dist_mean does.
    """
    acc = {k: [] for k in DYN_FUNCTIONALS}
    diff = dyn_pool["el2n_final"]
    for c in np.unique(y):
        at = np.flatnonzero(y[sel] == c)
        if len(at) < 2:
            continue
        loc = sel[at]
        pool_c = np.flatnonzero(y == c)
        f, a = dyn_pool["forget"][loc], dyn_pool["aum"][loc]
        acc["forget_mean"].append(f.mean())
        acc["forget_sd"].append(f.std())
        acc["forget_frac_zero"].append(float((f == 0).mean()))
        acc["aum_mean"].append(a.mean())
        acc["aum_sd"].append(a.std())
        acc["el2n_final_mean"].append(dyn_pool["el2n_final"][loc].mean())
        acc["el2n_early_mean"].append(dyn_pool["el2n_early"][loc].mean())

        ds, dp = np.sort(diff[loc]), np.sort(diff[pool_c])
        grid = np.concatenate([ds, dp])
        cs = np.searchsorted(ds, grid, "right") / len(ds)
        cp = np.searchsorted(dp, grid, "right") / len(dp)
        acc["difficulty_ks"].append(float(np.abs(cs - cp).max()))
        q = np.linspace(0, 1, 101)
        acc["difficulty_w1"].append(float(np.abs(np.quantile(ds, q) - np.quantile(dp, q)).mean()))
        hi, lo = np.quantile(dp, 0.9), np.quantile(dp, 0.1)
        acc["frac_hardest_decile"].append(float((diff[loc] >= hi).mean()))
        acc["frac_easiest_decile"].append(float((diff[loc] <= lo).mean()))
    return {f"F_{k}": (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}


def ccs_select(y: np.ndarray, score: np.ndarray, budget: int, seed: int, bins: int = 5):
    """Coverage-centric stratified selection: equal quota per difficulty stratum."""
    rng = np.random.default_rng(seed)
    out = []
    for c in np.unique(y):
        ids = np.flatnonzero(y == c)
        edges = np.quantile(score[ids], np.linspace(0, 1, bins + 1))
        edges[-1] += 1e-9
        per = max(1, budget // bins)
        picked = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = ids[(score[ids] >= lo) & (score[ids] < hi)]
            if len(m):
                picked.append(rng.choice(m, size=min(per, len(m)), replace=False))
        picked = np.concatenate(picked) if picked else ids[:budget]
        if len(picked) < budget:
            rest = np.setdiff1d(ids, picked)
            picked = np.concatenate([picked, rng.choice(rest, size=min(budget - len(picked), len(rest)), replace=False)])
        out.append(picked[:budget])
    return np.sort(np.concatenate(out))


def moderate_select(z: torch.Tensor, y: np.ndarray, budget: int) -> np.ndarray:
    """Moderate coreset: points nearest the MEDIAN distance to the class centroid."""
    out = []
    for c in np.unique(y):
        ids = np.flatnonzero(y == c)
        P = z[torch.as_tensor(ids, device=z.device)]
        d = torch.linalg.vector_norm(P - P.mean(0), dim=1)
        order = torch.argsort((d - d.median()).abs())[:budget]
        out.append(ids[order.cpu().numpy()])
    return np.sort(np.concatenate(out))


# --------------------------------------------------------------------------- #
# selection generators
# --------------------------------------------------------------------------- #

def by_score(y: np.ndarray, score: np.ndarray, budget: int, largest: bool) -> np.ndarray:
    out = []
    for c in np.unique(y):
        ids = np.flatnonzero(y == c)
        order = np.argsort(-score[ids] if largest else score[ids], kind="stable")
        out.append(ids[order[:budget]])
    return np.sort(np.concatenate(out))


def k_center(z: torch.Tensor, y: np.ndarray, budget: int) -> np.ndarray:
    """Greedy k-center (farthest-point) per class. Minimises dist_max by design."""
    out = []
    for c in np.unique(y):
        ids = np.flatnonzero(y == c)
        P = z[torch.as_tensor(ids, device=z.device)]
        chosen = [int(torch.linalg.vector_norm(P - P.mean(0), dim=1).argmin())]
        dmin = torch.cdist(P, P[chosen[:1]]).squeeze(1)
        for _ in range(budget - 1):
            nxt = int(dmin.argmax())
            chosen.append(nxt)
            dmin = torch.minimum(dmin, torch.cdist(P, P[nxt:nxt + 1]).squeeze(1))
        out.append(ids[np.array(chosen)])
    return np.sort(np.concatenate(out))


def k_medoids(z: torch.Tensor, y: np.ndarray, budget: int, seed: int, iters: int = 15) -> np.ndarray:
    """Lloyd-style medoid refinement per class. Minimises dist_mean by design."""
    out = []
    for c in np.unique(y):
        ids = np.flatnonzero(y == c)
        P = z[torch.as_tensor(ids, device=z.device)]
        g = torch.Generator(device="cpu").manual_seed(seed + int(c))
        cur = torch.randperm(len(ids), generator=g)[:budget].to(z.device)
        for _ in range(iters):
            assign = torch.cdist(P, P[cur]).argmin(1)
            new = cur.clone()
            for j in range(budget):
                m = assign == j
                if not bool(m.any()):
                    continue
                members = torch.nonzero(m, as_tuple=True)[0]
                sub = P[members]
                new[j] = members[torch.cdist(sub, sub).sum(1).argmin()]
            if bool((new == cur).all()):
                break
            cur = new
        out.append(ids[cur.cpu().numpy()])
    return np.sort(np.concatenate(out))


def perturb(sel: np.ndarray, y: np.ndarray, m: int, seed: int) -> np.ndarray:
    """Swap m selected points per class for random same-class pool points."""
    rng = np.random.default_rng(seed)
    keep, ysel = list(sel), y[sel]
    out = []
    for c in np.unique(y):
        cur = np.array([i for i in keep if y[i] == c])
        pool = np.setdiff1d(np.flatnonzero(y == c), cur, assume_unique=False)
        mm = min(m, len(cur), len(pool))
        if mm == 0:
            out.append(cur)
            continue
        drop = rng.choice(len(cur), size=mm, replace=False)
        add = rng.choice(pool, size=mm, replace=False)
        cur = cur.copy()
        cur[drop] = add
        out.append(cur)
    return np.sort(np.concatenate(out))


def build_library(z, y, pre, budget, seed, n_random, n_perturb_seeds, dyn=None):
    """(name, anchor_family, m, indices) for every selection in the library.

    `dyn` is the pool-subset dynamics dict. When it is present the six published
    dynamics baselines join the library as anchors. That matters for two separate
    reasons: they are the actual SOTA this project has to beat (and had never been
    run here), and their perturbation ladders sweep the dynamics functionals inside
    a fixed method family, which is the only way to read a functional's own effect.
    """
    lib = []
    for s in range(n_random):
        lib.append((f"random_{s}", "random", 0, R.random_balanced(None, y, budget, seed + s)))

    dens = pre["density"].cpu().numpy()
    marg = pre["margin"].cpu().numpy()
    pur = pre["purity"].cpu().numpy()
    anchors = {
        "herding": R.herding(z.cpu().numpy(), y, budget),
        "kcenter": k_center(z, y, budget),
        "kmedoids": k_medoids(z, y, budget, seed),
        "dense": by_score(y, dens, budget, True),
        "sparse": by_score(y, dens, budget, False),
        "hard": by_score(y, marg, budget, False),
        "easy": by_score(y, marg, budget, True),
        "impure": by_score(y, pur, budget, False),
    }
    if dyn is not None:
        # EL2N is taken at the early epoch because that is what Paul et al. specify;
        # the final-epoch norm is nearly zero for everything the model has fit.
        # AUM appears in both directions: low AUM is the mislabel/hard end (Pleiss),
        # high AUM the easy end, and which one helps at budget 25 is exactly the
        # open question.
        anchors.update({
            "forget_high": by_score(y, dyn["forget"], budget, True),
            "forget_low": by_score(y, dyn["forget"], budget, False),
            "el2n_high": by_score(y, dyn["el2n_early"], budget, True),
            "aum_low": by_score(y, dyn["aum"], budget, False),
            "aum_high": by_score(y, dyn["aum"], budget, True),
            "ccs": ccs_select(y, dyn["el2n_early"], budget, seed),
            "moderate": moderate_select(z, y, budget),
        })
    for name, sel in anchors.items():
        lib.append((name, name, 0, sel))
        for m in (1, 2, 5, 10, budget):
            for s in range(n_perturb_seeds):
                lib.append((f"{name}_m{m}_s{s}", name, m,
                            perturb(sel, y, m, seed + 7919 * m + s)))
    return lib


# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    # optional so the script still runs where the 200-epoch dynamics caches are not
    # staged; without it the dynamics anchors and functionals are simply absent.
    p.add_argument("--dynamics-root", type=Path, default=None)
    p.add_argument("--budget", type=int, default=BUDGET)
    p.add_argument("--blocks", type=int, default=BLOCKS)
    p.add_argument("--n-random", type=int, default=100)
    p.add_argument("--n-perturb-seeds", type=int, default=7)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--size", type=int, default=224)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for dataset in args.datasets:
            emb = args.embedding_root / f"{dataset}_train_uni_{args.size}.npz"
            x = np.asarray(np.load(emb, allow_pickle=False)["embeddings"], dtype=np.float32)
            y_all = R.read_train_labels(emb, args.data_root / f"{dataset}_{args.size}.npz")
            di = DATASETS.index(dataset)
            align = R.assert_labels_aligned(x, y_all, dataset, args.seed + 500009 * di, 0.30)
            z_all = R.normalize(x)
            del x
            dyn_all = (load_dynamics(args.dynamics_root, dataset, len(y_all))
                       if args.dynamics_root is not None else None)
            print(f"[{dataset}] n={len(y_all)} align_ba={align:.4f} "
                  f"dynamics={'yes' if dyn_all else 'no'}", flush=True)

            for block in range(args.blocks):
                bseed = args.seed + 100003 * di + 1009 * block
                pool_ids, audit_ids = R.split_block(np.arange(len(y_all)), y_all,
                                                    1.0 - POOL_FRACTION, bseed)
                zp = torch.as_tensor(z_all[pool_ids], device=dev)
                yp = y_all[pool_ids]
                audit_x, audit_y = z_all[audit_ids], y_all[audit_ids]

                # dynamics are indexed by full-train position; the library and the
                # functionals both work in pool-local indices, so subset once here.
                dyn = ({k: v[pool_ids] for k, v in dyn_all.items()}
                       if dyn_all is not None else None)

                t0 = time.time()
                pre = pool_precompute(zp, yp)
                lib = build_library(zp, yp, pre, args.budget, bseed,
                                    args.n_random, args.n_perturb_seeds, dyn)
                print(f"  block {block}: pool={len(pool_ids)} audit={len(audit_ids)} "
                      f"library={len(lib)} precompute={time.time()-t0:.0f}s", flush=True)

                for n, (name, family, m, sel) in enumerate(lib, 1):
                    F, w_within = functionals(zp, yp, sel, pre)
                    tx, ty = z_all[pool_ids[sel]], yp[sel]
                    eq = R.metrics(tx, ty, audit_x, audit_y, bseed)
                    vw = R.metrics(tx, ty, audit_x, audit_y, bseed, sample_weight=w_within)
                    out.write(json.dumps({
                        "dataset": dataset, "block": block, "budget": args.budget,
                        "name": name, "family": family, "m": m, "n_selected": int(len(sel)),
                        **{f"F_{k}": v for k, v in F.items()},
                        **(dyn_functionals(dyn, sel, yp) if dyn is not None else {}),
                        "ba_equal": eq["balanced_accuracy"], "acc_equal": eq["accuracy"],
                        "wcr_equal": eq["worst_class_recall"], "ece_equal": eq["ece15"],
                        "ba_voronoi": vw["balanced_accuracy"],
                    }, sort_keys=True) + "\n")
                    out.flush()
                    if n % 50 == 0:
                        print(f"    {n}/{len(lib)} ({time.time()-t0:.0f}s)", flush=True)
            del z_all
    print(json.dumps({"output": str(args.output)}))


if __name__ == "__main__":
    main()
