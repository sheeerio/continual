# TASK 2 -- fc2 dead-layer characterization. 3 arms, same seed/config.
#  0 adaptive_sat median100   1 static spectral c=1e-3   2 vanilla (--reg none)
import os, shlex
H=os.environ["HOME"]; ROOT=f"{H}/scratch/fc2dead"; CD=f"{ROOT}/cells"
os.makedirs(CD, exist_ok=True)
PY=f"{H}/venv/continual/bin/python"
BASE=["--optimizer","adam","--activation","relu","--runs","20","--epochs","50",
      "--dataset","MNIST","--model","MLP","--hidden","256","--lr","0.001",
      "--batch_size","256","--log_interval","100","--ns","1.0",
      "--diagnostics","full","--track_coherence","--coherence_window","20",
      "--hessian_tol","1e-2","--hessian_max_iters","50","--seed","0"]
CELLS=[("fc2_adaptive", ["--reg","none","--adaptive_reg","--adaptive_type","spectral",
                         "--adaptive_scale","saturating","--sat_kappa","1",
                         "--reg_sensitivity","1e-3","--tau_ref_mode","median",
                         "--tau_ref_window","100"]),
       ("fc2_static",   ["--reg","spectral","--spectral_lambda","1e-3"]),
       ("fc2_vanilla",  ["--reg","none"])]
cmds=[]
for nm,fl in CELLS:
    out=f"{CD}/{nm}.csv"
    cmds.append((out,[PY,"implicit_regularization.py",*BASE,*fl,"--name",nm,
        "--exp_name","fc2dead","--results_csv",out,
        "--taskdiag_csv",f"{CD}/{nm}_taskdiag.csv","--diag_csv",f"{CD}/{nm}_diag.csv"]))
open(f"{ROOT}/commands.txt","w").write("\n".join(" ".join(shlex.quote(a) for a in c) for _,c in cmds)+"\n")
open(f"{ROOT}/csvs.txt","w").write("\n".join(o for o,_ in cmds)+"\n")
print(f"{len(cmds)} cells -> {ROOT}"); print(f"array range: 0-{len(cmds)-1}")
