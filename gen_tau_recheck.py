# TASK 4: re-measure tau with the finalized estimator on the calibrated testbed.
import os, shlex
H=os.environ["HOME"]; ROOT=f"{H}/scratch/tau_recheck"; CD=f"{ROOT}/cells"
os.makedirs(CD, exist_ok=True)
PY=f"{H}/venv/continual/bin/python"
BASE=["--optimizer","adam","--activation","relu","--runs","20","--epochs","50",
      "--dataset","MNIST","--model","MLP","--hidden","256","--lr","0.001",
      "--batch_size","256","--log_interval","100","--ns","1.0",
      "--diagnostics","full","--track_coherence","--coherence_window","20",
      "--hessian_tol","1e-2","--hessian_max_iters","50"]
ARMS=[("static_c1e-3",["--reg","spectral","--spectral_lambda","1e-3"]),
      ("adaptive_sat_c1e-3",["--reg","none","--adaptive_reg","--adaptive_type","spectral",
                             "--adaptive_scale","saturating","--sat_kappa","1",
                             "--reg_sensitivity","1e-3"])]
cmds=[]
for nm,fl in ARMS:
    out=f"{CD}/{nm}.csv"
    cmds.append((out,[PY,"implicit_regularization.py",*BASE,*fl,"--seed","0","--name",nm,
        "--exp_name","tau_recheck","--results_csv",out,
        "--taskdiag_csv",f"{CD}/{nm}_taskdiag.csv","--diag_csv",f"{CD}/{nm}_diag.csv"]))
open(f"{ROOT}/commands.txt","w").write("\n".join(" ".join(shlex.quote(a) for a in c) for _,c in cmds)+"\n")
open(f"{ROOT}/csvs.txt","w").write("\n".join(o for o,_ in cmds)+"\n")
print(f"{len(cmds)} cells -> {ROOT}")
