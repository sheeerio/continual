# TASK 3 loader modes (with a fair RNG comparison) + fc2 re-run.
#
# The gpu loader replaces the DataLoader sampler with torch.randperm on device,
# so it consumes RNG differently and trajectories diverge by construction. A
# single-seed delta cannot separate "different shuffling" from "bug in the
# indexing path", so gpu AND workers4 are each run on 3 seeds and compared as
# distributions. workers4 seeds are run under CURRENT code rather than reusing
# the 93-cell grid's std, which predates the estimator fix.
import os, shlex
H=os.environ["HOME"]; ROOT=f"{H}/scratch/ldr"; CD=f"{ROOT}/cells"
os.makedirs(CD, exist_ok=True)
PY=f"{H}/venv/continual/bin/python"
BASE=["--optimizer","adam","--activation","relu","--runs","20","--epochs","50",
      "--dataset","MNIST","--model","MLP","--hidden","256","--lr","0.001",
      "--batch_size","256","--log_interval","100","--ns","1.0",
      "--diagnostics","full","--track_coherence","--coherence_window","20",
      "--hessian_tol","1e-2","--hessian_max_iters","50"]
ST=["--reg","spectral","--spectral_lambda","1e-3"]
AD=["--reg","none","--adaptive_reg","--adaptive_type","spectral","--adaptive_scale",
    "saturating","--sat_kappa","1","--reg_sensitivity","1e-3",
    "--tau_ref_mode","median","--tau_ref_window","100"]
CELLS=[
  ("ldr_workers4_prof",  ST+["--loader_mode","workers4","--seed","0"]),
  ("ldr_persistent_prof",ST+["--loader_mode","persistent","--seed","0"]),
  ("ldr_gpu_prof",       ST+["--loader_mode","gpu","--seed","0"]),
  ("ldr_gpu_s0",         ST+["--loader_mode","gpu","--seed","0"]),
  ("ldr_gpu_s1",         ST+["--loader_mode","gpu","--seed","1"]),
  ("ldr_gpu_s2",         ST+["--loader_mode","gpu","--seed","2"]),
  ("ldr_w4_s0",          ST+["--loader_mode","workers4","--seed","0"]),
  ("ldr_w4_s1",          ST+["--loader_mode","workers4","--seed","1"]),
  ("ldr_w4_s2",          ST+["--loader_mode","workers4","--seed","2"]),
  ("fc2b_adaptive",      AD+["--seed","0"]),
  ("fc2b_static",        ST+["--seed","0"]),
  ("fc2b_vanilla",       ["--reg","none","--seed","0"]),
]
cmds=[]
for nm,fl in CELLS:
    out=f"{CD}/{nm}.csv"
    cmds.append((out,[PY,"implicit_regularization.py",*BASE,*fl,"--name",nm,
        "--exp_name","ldr","--results_csv",out,
        "--taskdiag_csv",f"{CD}/{nm}_taskdiag.csv","--diag_csv",f"{CD}/{nm}_diag.csv"]))
open(f"{ROOT}/commands.txt","w").write("\n".join(" ".join(shlex.quote(a) for a in c) for _,c in cmds)+"\n")
open(f"{ROOT}/csvs.txt","w").write("\n".join(o for o,_ in cmds)+"\n")
print(f"{len(cmds)} cells -> {ROOT}"); print("array range: 0-%d"%(len(cmds)-1))
