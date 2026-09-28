import json, glob, os, re, numpy as np
from collections import defaultdict
P="/project/prj-sis01/xuxiaoyu/reliability_medmnistc_ab/tdgs_round1_20260927"
rows=[]
for d in sorted(glob.glob(f"{P}/formal/ratio_*/**/aug_0", recursive=True)):
    m=re.search(r"/ratio_([0-9.]+)/([a-z]+mnist)/([a-z0-9_]+)/seed_(\d+)/", d)
    if not m: continue
    ratio,ds,arm,seed=m.group(1),m.group(2),m.group(3),int(m.group(4))
    mt=os.path.join(d,"metrics.jsonl")
    if not os.path.exists(mt): continue
    recs=[json.loads(l) for l in open(mt) if l.strip()]
    r=recs[-1]
    npz=np.load(os.path.join(d,"predictions_clean.npz"))
    y=npz["y_true"].ravel(); p=npz["y_pred"].ravel() if "y_pred" in npz else npz["logits"].argmax(1)
    C=int(y.max())+1
    rec=np.array([ (p[y==c]==c).mean() if (y==c).sum() else np.nan for c in range(C)])
    rows.append(dict(ds=ds,ratio=float(ratio),arm=arm,seed=seed,
                     ba=float(np.nanmean(rec)), worst=float(np.nanmin(rec)),
                     acc=float((p==y).mean()), recalls=rec.round(6).tolist()))
json.dump(rows, open(os.path.expanduser("~/tdgs_round1_harvest.json"),"w"), indent=1)
print("n_rows",len(rows))
# ---- paired ladder ----
by=defaultdict(dict)
for r in rows: by[(r["ds"],r["seed"])][r["arm"]]=r
LADDER=[("tdgs_cls","graph_a2"),("tdgs_d","tdgs_cls"),("tdgs_du","tdgs_d"),
        ("tdgs_du","tdgs_perm"),("tdgs_du","graph_a2"),("tdgs_d","graph_a2")]
DS=["pathmnist","organamnist","bloodmnist","organsmnist","tissuemnist"]
print("\n=== per-arm BA (mean over 3 seeds), %% ===")
print(f"{'dataset':14s}"+"".join(f"{a:>11s}" for a in ["graph_a2","tdgs_cls","tdgs_d","tdgs_du","tdgs_perm"]))
for ds in DS:
    line=f"{ds:14s}"
    for a in ["graph_a2","tdgs_cls","tdgs_d","tdgs_du","tdgs_perm"]:
        v=[by[(ds,s)][a]["ba"]*100 for s in (42,43,44) if a in by[(ds,s)]]
        line+=f"{np.mean(v):11.2f}" if v else f"{'--':>11s}"
    print(line)
print("\n=== per-arm WORST-CLASS RECALL (mean over 3 seeds), %% ===")
print(f"{'dataset':14s}"+"".join(f"{a:>11s}" for a in ["graph_a2","tdgs_cls","tdgs_d","tdgs_du","tdgs_perm"]))
for ds in DS:
    line=f"{ds:14s}"
    for a in ["graph_a2","tdgs_cls","tdgs_d","tdgs_du","tdgs_perm"]:
        v=[by[(ds,s)][a]["worst"]*100 for s in (42,43,44) if a in by[(ds,s)]]
        line+=f"{np.mean(v):11.2f}" if v else f"{'--':>11s}"
    print(line)
for met in ("ba","worst"):
    print(f"\n=== PAIRED DELTAS on {met} (pp), per dataset then pooled ===")
    for hi,lo in LADDER:
        ds_d={}; alld=[]
        for ds in DS:
            dd=[(by[(ds,s)][hi][met]-by[(ds,s)][lo][met])*100 for s in (42,43,44)
                if hi in by[(ds,s)] and lo in by[(ds,s)]]
            if dd: ds_d[ds]=np.mean(dd); alld+=dd
        if not alld: continue
        pooled=np.mean(alld); sd=np.std(alld,ddof=1); se=sd/np.sqrt(len(alld))
        print(f"{hi:>10s}-{lo:<10s} pooled {pooled:+6.2f}  (sd {sd:4.2f} se {se:4.2f} n={len(alld)})  "
              +" ".join(f"{k[:5]}{v:+6.2f}" for k,v in ds_d.items()))
