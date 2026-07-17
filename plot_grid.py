# plot_grid.py  — run on the login node in ~/scratch/continual-plast
import os, glob, re, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

H=os.environ["HOME"]; CELLS=f"{H}/scratch/plast/cells"
rows=[]
for p in glob.glob(f"{CELLS}/*.csv"):
    if os.path.getsize(p)==0: continue
    try: d=pd.read_csv(p)
    except Exception: continue
    if d.empty: continue
    fn=os.path.basename(p)[:-4]
    d=d.iloc[[0]].copy(); d["fname"]=fn
    d["arm"]  = "adaptive" if fn.startswith("adapt_") else ("static" if fn.startswith("static_") else "vanilla")
    d["rtype"]= (re.search(r"^(?:adapt|static)_(l2|spectral)_", fn) or [None,None])[1] if d["arm"].iloc[0]!="vanilla" else "none"
    m=re.search(r"_(none|grad|svar|reset)_lr", fn); d["sched"]= m.group(1) if m else "none"
    m=re.search(r"_lr([0-9eE.\-+]+)_seed", fn);      d["lr"]   = float(m.group(1)) if m else np.nan
    m=re.search(r"(?:sens|lam)([0-9eE.\-+]+)_", fn); d["coeff"]= float(m.group(1)) if m else np.nan
    rows.append(d)
df=pd.concat(rows, ignore_index=True)
df.to_csv(f"{H}/scratch/plast/grid_all.csv", index=False)
print("cells loaded:", len(df))
print(df.groupby(["arm","sched","lr"]).size().to_string())

LRS=sorted(df["lr"].dropna().unique())
SCHEDS=["none","grad"]

# ===== FIG 1: the money figure. AUC + slope vs coefficient, grid of (lr x sched), adaptive vs static
for metric, better in [("auc","higher"), ("slope","flatter/positive")]:
  for rt in ["l2","spectral"]:                      # <-- new
    fig,axes=plt.subplots(len(LRS), len(SCHEDS), figsize=(5.5*len(SCHEDS), 4*len(LRS)), squeeze=False)
    for i,lr in enumerate(LRS):
        for j,sch in enumerate(SCHEDS):
            ax=axes[i][j]
            sub=df[(df.lr==lr)&(df.sched==sch)&(df.rtype==rt)]
            for arm,color in [("adaptive","#268"),("static","#c44")]:
                a=sub[(sub.arm==arm)&sub.coeff.notna()]
                if a.empty: continue
                g=a.groupby("coeff")[metric].agg(["mean","std","count"]).reset_index().sort_values("coeff")
                sem=g["std"]/np.sqrt(g["count"].clip(lower=1))
                ax.plot(g["coeff"],g["mean"],"-o",color=color,label=arm,ms=4)
                ax.fill_between(g["coeff"],g["mean"]-sem,g["mean"]+sem,alpha=.2,color=color)
            v=df[(df.arm=="vanilla")&(df.lr==lr)&(df.sched==sch)]
            if not v.empty: ax.axhline(v[metric].mean(),color="gray",ls="--",lw=1,label="vanilla")
            if metric=="slope": ax.axhline(0,color="k",lw=.5)
            ax.set_xscale("log"); ax.grid(alpha=.3)
            ax.set_title(f"lr={lr:g}, sched={sch}")
            if i==len(LRS)-1: ax.set_xlabel("coefficient")
            if j==0: ax.set_ylabel(metric)
            if i==0 and j==0: ax.legend(fontsize=8,frameon=False)
    fig.suptitle(f"{metric.upper()} vs coefficient  ({better} = better)", y=1.00)
    fig.tight_layout(); fig.savefig(f"{H}/scratch/plast/grid_{metric}_{rt}.png",dpi=200,bbox_inches="tight")
    print("saved grid_"+metric+".png")

# ===== FIG 2: robustness quantified. spread across the coefficient axis, per arm
print("\n=== ROBUSTNESS: std of mean-AUC across coefficients (lower = flatter = more robust) ===")
print(f"{'lr':>8} {'sched':>6} {'rtype':>9} {'adaptive':>10} {'static':>10}  verdict")
recs=[]
for lr in LRS:
    for sch in SCHEDS:
        for rt in ["l2","spectral"]:
            r={}
            for arm in ["adaptive","static"]:
                a=df[(df.lr==lr)&(df.sched==sch)&(df.arm==arm)&(df.rtype==rt)&df.coeff.notna()]
                r[arm]= a.groupby("coeff")["auc"].mean().std() if len(a) else np.nan
            if np.isnan(r.get("adaptive",np.nan)) or np.isnan(r.get("static",np.nan)): continue
            verdict = "ADAPTIVE flatter" if r["adaptive"]<r["static"] else "static flatter"
            print(f"{lr:>8g} {sch:>6} {rt:>9} {r['adaptive']:>10.4f} {r['static']:>10.4f}  {verdict}")
            recs.append(dict(lr=lr,sched=sch,rtype=rt,**r))

# ===== FIG 3: peak vs robustness. best-case AUC per arm (the honest tradeoff table)
print("\n=== PEAK: best mean-AUC over coefficients (who wins when well-tuned) ===")
print(f"{'lr':>8} {'sched':>6} {'rtype':>9} {'adaptive':>10} {'static':>10}  verdict")
for lr in LRS:
    for sch in SCHEDS:
        for rt in ["l2","spectral"]:
            r={}
            for arm in ["adaptive","static"]:
                a=df[(df.lr==lr)&(df.sched==sch)&(df.arm==arm)&(df.rtype==rt)&df.coeff.notna()]
                r[arm]= a.groupby("coeff")["auc"].mean().max() if len(a) else np.nan
            if np.isnan(r.get("adaptive",np.nan)) or np.isnan(r.get("static",np.nan)): continue
            verdict = "ADAPTIVE higher" if r["adaptive"]>r["static"] else "static higher"
            print(f"{lr:>8g} {sch:>6} {rt:>9} {r['adaptive']:>10.4f} {r['static']:>10.4f}  {verdict}")

# ===== FIG 4: trajectories, seed-averaged, faceted by coefficient (fixes the spaghetti)
COEFFS=sorted(df["coeff"].dropna().unique())
for lr in LRS:
    fig,axes=plt.subplots(1,len(COEFFS),figsize=(3.1*len(COEFFS),3.4),sharey=True,squeeze=False)
    for k,c in enumerate(COEFFS):
        ax=axes[0][k]
        for arm,color in [("adaptive","#268"),("static","#c44")]:
            a=df[(df.arm==arm)&(df.lr==lr)&(df.sched=="none")&(df.coeff==c)&(df.rtype=="l2")]
            if a.empty: continue
            T=[[float(v) for v in str(r).split(";")] for r in a["task_acc_traj"]]
            L=min(len(t) for t in T); T=np.array([t[:L] for t in T])
            m=T.mean(0); s=T.std(0)/np.sqrt(len(T))
            ax.plot(m,color=color,lw=1.2,label=arm)
            ax.fill_between(range(L),m-s,m+s,color=color,alpha=.2)
        v=df[(df.arm=="vanilla")&(df.lr==lr)&(df.sched=="none")]
        if not v.empty:
            T=[[float(x) for x in str(r).split(";")] for r in v["task_acc_traj"]]
            L=min(len(t) for t in T); ax.plot(np.array([t[:L] for t in T]).mean(0),color="gray",ls="--",lw=1)
        ax.set_title(f"coeff={c:g}",fontsize=9); ax.grid(alpha=.3); ax.set_xlabel("task")
        if k==0: ax.set_ylabel("task_acc"); ax.legend(fontsize=7,frameon=False)
    fig.suptitle(f"Per-task trajectory, l2, sched=none, lr={lr:g}  (mean±SEM over seeds)",y=1.03)
    fig.tight_layout(); fig.savefig(f"{H}/scratch/plast/traj_lr{lr:g}.png",dpi=200,bbox_inches="tight")
    print(f"saved traj_lr{lr:g}.png")
