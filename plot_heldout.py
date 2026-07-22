# plot_heldout.py
import os, glob, re, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H=os.environ["HOME"]

for tag in ["ns05","wide"]:
    rows=[]
    for p in glob.glob(f"{H}/scratch/plast_{tag}/cells/*.csv"):
        if os.path.getsize(p)==0: continue
        try: d=pd.read_csv(p)
        except Exception: continue
        if d.empty: continue
        fn=os.path.basename(p)[:-4]
        m=re.match(r"adapt_(l2|spectral|wass)_(inv|saturating)_sens([0-9eE.+-]+)_(none|grad)_lr([0-9eE.+-]+)_seed(\d+)", fn)
        if not m: continue
        d=d.iloc[[0]].copy()
        d["rtype"],d["scale"],d["coeff"],d["sched"],d["lr"],d["seed"]=\
            m.group(1),m.group(2),float(m.group(3)),m.group(4),float(m.group(5)),int(m.group(6))
        rows.append(d)
    if not rows: print(f"{tag}: no cells"); continue
    df=pd.concat(rows,ignore_index=True)
    print(f"\n=== {tag}: {len(df)} cells")
    print(df.groupby(["rtype","scale"]).size().to_string())

    LRS=sorted(df.lr.unique()); SCH=["none","grad"]
    for metric in ["auc","slope"]:
        for rt in sorted(df.rtype.unique()):
            fig,ax=plt.subplots(len(LRS),len(SCH),figsize=(5.5*len(SCH),4*len(LRS)),squeeze=False)
            for i,lr in enumerate(LRS):
                for j,sc in enumerate(SCH):
                    a=ax[i][j]
                    for scale,col in [("inv","#c44"),("saturating","#268")]:
                        q=df[(df.rtype==rt)&(df.scale==scale)&(df.lr==lr)&(df.sched==sc)]
                        if q.empty: continue
                        g=q.groupby("coeff")[metric].agg(["mean","std","count"]).reset_index().sort_values("coeff")
                        sem=g["std"]/np.sqrt(g["count"].clip(lower=1))
                        a.plot(g["coeff"],g["mean"],"-o",color=col,label=scale,ms=4)
                        a.fill_between(g["coeff"],g["mean"]-sem,g["mean"]+sem,alpha=.2,color=col)
                    a.set_xscale("log"); a.grid(alpha=.3); a.set_title(f"lr={lr:g}, sched={sc}")
                    if metric=="slope": a.axhline(0,color="k",lw=.5)
                    if i==len(LRS)-1: a.set_xlabel("coefficient")
                    if j==0: a.set_ylabel(metric)
                    if i==0 and j==0: a.legend(fontsize=8,frameon=False)
            fig.suptitle(f"[{tag}] {rt}: {metric}, inv vs saturating",y=1.00)
            fig.tight_layout(); fig.savefig(f"{H}/scratch/heldout_{tag}_{metric}_{rt}.png",dpi=200,bbox_inches="tight")
            print(f"saved heldout_{tag}_{metric}_{rt}.png")

    # numeric verdict: high-coefficient behaviour, where the bounding hypothesis lives
    print(f"\n--- {tag}: mean AUC at high coefficients (>=1e-2), inv vs saturating")
    hi=df[df.coeff>=1e-2]
    for rt in sorted(hi.rtype.unique()):
        for lr in LRS:
            for sc in SCH:
                q=hi[(hi.rtype==rt)&(hi.lr==lr)&(hi.sched==sc)]
                if q.empty: continue
                v=q.groupby("scale")["auc"].mean()
                if len(v)<2: continue
                w="SATURATING better" if v.get("saturating",0)>v.get("inv",0) else "inv better"
                print(f"  {rt:9} lr={lr:<7g} {sc:5} inv={v.get('inv',float('nan')):.4f} sat={v.get('saturating',float('nan')):.4f}  {w}")
