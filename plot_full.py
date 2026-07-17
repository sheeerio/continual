# plot_full.py  — run on the login node in ~/scratch/continual-plast
import os, glob, re, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

H=os.environ["HOME"]; CELLS=f"{H}/scratch/plast/cells"
rows=[]
for p in glob.glob(f"{CELLS}/*.csv"):
    if os.path.getsize(p)==0: continue
    try: d=pd.read_csv(p)
    except Exception: continue
    if d.empty: continue
    fn=os.path.basename(p)[:-4]; d=d.iloc[[0]].copy(); d["fname"]=fn
    d["arm"]="adaptive" if fn.startswith("adapt_") else ("static" if fn.startswith("static_") else "vanilla")
    m=re.search(r"^(?:adapt|static)_(l2_loss|l2|spectral|wass|parseval)_", fn)
    d["rtype"]= m.group(1) if m else "none"
    m=re.search(r"_(none|grad|svar|reset)_lr", fn); d["sched"]= m.group(1) if m else "none"
    m=re.search(r"_lr([0-9eE.\-+]+)_seed", fn);      d["lr"]= float(m.group(1)) if m else np.nan
    m=re.search(r"(?:sens|lam)([0-9eE.\-+]+)_", fn); d["coeff"]= float(m.group(1)) if m else np.nan
    rows.append(d)
df=pd.concat(rows, ignore_index=True)
df.to_csv(f"{H}/scratch/plast/full_all.csv", index=False)
print("cells:", len(df))
print(df.groupby(["arm","rtype","lr","sched"]).size().to_string())

LRS=sorted(df.lr.dropna().unique()); SCHEDS=["none","grad"]
# compare adaptive vs its static counterpart. For l2, use l2_loss as the honest static baseline.
PAIRS=[("l2","l2_loss"),("spectral","spectral"),("wass","wass"),("parseval","parseval")]

# ===== per-regularizer AUC + slope grids =====
for metric in ["auc","slope"]:
    for atype, stype in PAIRS:
        fig,axes=plt.subplots(len(LRS),len(SCHEDS),figsize=(5.5*len(SCHEDS),4*len(LRS)),squeeze=False)
        for i,lr in enumerate(LRS):
            for j,sch in enumerate(SCHEDS):
                ax=axes[i][j]
                for arm,rt,color,lab in [("adaptive",atype,"#268","adaptive"),("static",stype,"#c44",f"static ({stype})")]:
                    a=df[(df.arm==arm)&(df.rtype==rt)&(df.lr==lr)&(df.sched==sch)&df.coeff.notna()]
                    if a.empty: continue
                    g=a.groupby("coeff")[metric].agg(["mean","std","count"]).reset_index().sort_values("coeff")
                    sem=g["std"]/np.sqrt(g["count"].clip(lower=1))
                    ax.plot(g["coeff"],g["mean"],"-o",color=color,label=lab,ms=4)
                    ax.fill_between(g["coeff"],g["mean"]-sem,g["mean"]+sem,alpha=.2,color=color)
                v=df[(df.arm=="vanilla")&(df.lr==lr)&(df.sched==sch)]
                if not v.empty: ax.axhline(v[metric].mean(),color="gray",ls="--",lw=1,label="vanilla")
                if metric=="slope": ax.axhline(0,color="k",lw=.5)
                ax.set_xscale("log"); ax.grid(alpha=.3); ax.set_title(f"lr={lr:g}, sched={sch}")
                if i==len(LRS)-1: ax.set_xlabel("coefficient")
                if j==0: ax.set_ylabel(metric)
                if i==0 and j==0: ax.legend(fontsize=8,frameon=False)
        fig.suptitle(f"{atype.upper()}: {metric} vs coefficient",y=1.00)
        fig.tight_layout(); fig.savefig(f"{H}/scratch/plast/full_{metric}_{atype}.png",dpi=200,bbox_inches="tight")
        print(f"saved full_{metric}_{atype}.png")

# ===== TABLE 1: robustness (std of mean-AUC across coeff; lower=flatter=more robust) =====
print("\n=== ROBUSTNESS: std of mean-AUC across coefficients (lower = more robust) ===")
print(f"{'rtype':>9} {'lr':>7} {'sched':>6} {'adaptive':>10} {'static':>10}  verdict")
for atype,stype in PAIRS:
    for lr in LRS:
        for sch in SCHEDS:
            a=df[(df.arm=='adaptive')&(df.rtype==atype)&(df.lr==lr)&(df.sched==sch)&df.coeff.notna()]
            st=df[(df.arm=='static')&(df.rtype==stype)&(df.lr==lr)&(df.sched==sch)&df.coeff.notna()]
            if a.empty or st.empty: continue
            ra=a.groupby("coeff")["auc"].mean().std(); rs=st.groupby("coeff")["auc"].mean().std()
            v="ADAPTIVE flatter" if ra<rs else "static flatter"
            print(f"{atype:>9} {lr:>7g} {sch:>6} {ra:>10.4f} {rs:>10.4f}  {v}")

# ===== TABLE 2: peak (best mean-AUC over coeff; who wins when tuned) =====
print("\n=== PEAK: best mean-AUC over coefficients (who wins when well-tuned) ===")
print(f"{'rtype':>9} {'lr':>7} {'sched':>6} {'adaptive':>10} {'static':>10}  verdict")
for atype,stype in PAIRS:
    for lr in LRS:
        for sch in SCHEDS:
            a=df[(df.arm=='adaptive')&(df.rtype==atype)&(df.lr==lr)&(df.sched==sch)&df.coeff.notna()]
            st=df[(df.arm=='static')&(df.rtype==stype)&(df.lr==lr)&(df.sched==sch)&df.coeff.notna()]
            if a.empty or st.empty: continue
            pa=a.groupby("coeff")["auc"].mean().max(); ps=st.groupby("coeff")["auc"].mean().max()
            v="ADAPTIVE higher" if pa>ps else "static higher"
            print(f"{atype:>9} {lr:>7g} {sch:>6} {pa:>10.4f} {ps:>10.4f}  {v}")
