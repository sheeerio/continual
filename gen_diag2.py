# TASKS 2c, 3, 4 -- 6 cells.
#  0 median2000 re-run (crash fixed)          -> Task 2c
#  1 median100 re-run with event logging      -> Task 2c
#  2 fixed re-run with event logging          -> Task 2c
#  3 static, COST_PROFILE (all 9 sites)       -> Task 3a/3b
#  4 sigma2 N=32 with per-step sigma2 logged  -> Task 4a
#  5 sigma2 N=64                              -> Task 4a
import os, shlex
H=os.environ["HOME"]; ROOT=f"{H}/scratch/diag2"; CD=f"{ROOT}/cells"
os.makedirs(CD, exist_ok=True)
PY=f"{H}/venv/continual/bin/python"
BASE=["--optimizer","adam","--activation","relu","--runs","20","--epochs","50",
      "--dataset","MNIST","--model","MLP","--hidden","256","--lr","0.001",
      "--batch_size","256","--log_interval","100","--ns","1.0",
      "--diagnostics","full","--track_coherence","--coherence_window","20",
      "--hessian_tol","1e-2","--hessian_max_iters","50","--seed","0"]
ADAPT=["--reg","none","--adaptive_reg","--adaptive_type","spectral",
       "--adaptive_scale","saturating","--sat_kappa","1","--reg_sensitivity","1e-3"]
STATIC=["--reg","spectral","--spectral_lambda","1e-3"]
CELLS=[("ev_median2000", ADAPT+["--tau_ref_mode","median","--tau_ref_window","2000"]),
       ("ev_median100",  ADAPT+["--tau_ref_mode","median","--tau_ref_window","100"]),
       ("ev_fixed",      ADAPT+["--tau_ref_mode","fixed"]),
       ("prof_static2",  STATIC),
       ("sig_n32",       STATIC+["--sigma2_subsample","32"]),
       ("sig_n64",       STATIC+["--sigma2_subsample","64"])]
cmds=[]
for nm,fl in CELLS:
    out=f"{CD}/{nm}.csv"
    cmds.append((out,[PY,"implicit_regularization.py",*BASE,*fl,"--name",nm,
        "--exp_name","diag2","--results_csv",out,
        "--taskdiag_csv",f"{CD}/{nm}_taskdiag.csv","--diag_csv",f"{CD}/{nm}_diag.csv"]))
open(f"{ROOT}/commands.txt","w").write("\n".join(" ".join(shlex.quote(a) for a in c) for _,c in cmds)+"\n")
open(f"{ROOT}/csvs.txt","w").write("\n".join(o for o,_ in cmds)+"\n")
print(f"{len(cmds)} cells -> {ROOT}"); print(f"array range: 0-{len(cmds)-1}")
