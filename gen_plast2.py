# gen_plast2.py
import os, shlex, itertools
H=os.environ["HOME"]
ROOT=f"{H}/scratch/plast"; CSVDIR=f"{ROOT}/cells"; os.makedirs(CSVDIR, exist_ok=True)
PY=f"{H}/venv/continual/bin/python"; SCRIPT="implicit_regularization.py"
base=["--optimizer","adam","--activation","relu","--runs","100","--epochs","100",
      "--dataset","MNIST","--model","MLP","--batch_size","256","--log_interval","400","--ns","1.0"]
COEFFS=["1e-4","3e-4","1e-3","3e-3","1e-2","3e-2","1e-1"]
LRS=["1e-2","1e-3"]
SEEDS=["1","2","3"]

def sched_flags(s):
    if s=="none":  return [], "none"
    if s=="grad":  return ["--lr_schedule","pl_lyapunov","--param","t"], "grad"
    if s=="reset": return ["--reset_model"], "reset"

cmds=[]
# adaptive: sensitivity/tau, local
for atype,c,lr,sched,seed in itertools.product(["l2","spectral"],COEFFS,LRS,["none","grad"],SEEDS):
    sf,st=sched_flags(sched)
    name=f"adapt_{atype}_sens{c}_{st}_lr{lr}_seed{seed}"; out=f"{CSVDIR}/{name}.csv"
    cmds.append((out,[PY,SCRIPT,*base,"--lr",lr,"--adaptive_reg","--adaptive_type",atype,
                      "--reg_sensitivity",c,*sf,"--seed",seed,"--name",name,
                      "--exp_name","plast2","--results_csv",out]))
# static
for rtype,c,lr,sched,seed in itertools.product(["l2","spectral"],COEFFS,LRS,["none","grad"],SEEDS):
    sf,st=sched_flags(sched)
    name=f"static_{rtype}_lam{c}_{st}_lr{lr}_seed{seed}"; out=f"{CSVDIR}/{name}.csv"
    cc=[PY,SCRIPT,*base,"--lr",lr,"--reg",rtype,*sf,"--seed",seed,"--name",name,
        "--exp_name","plast2","--results_csv",out]
    cc += ["--l2_lambda",c] if rtype=="l2" else ["--spectral_lambda",c]
    cmds.append((out,cc))
# vanilla
for lr,sched,seed in itertools.product(LRS,["none","grad","reset"],SEEDS):
    sf,st=sched_flags(sched)
    name=f"none_{st}_lr{lr}_seed{seed}"; out=f"{CSVDIR}/{name}.csv"
    cmds.append((out,[PY,SCRIPT,*base,"--lr",lr,*sf,"--seed",seed,"--name",name,
                      "--exp_name","plast2","--results_csv",out]))

with open(f"{ROOT}/commands.txt","w") as fc, open(f"{ROOT}/csvs.txt","w") as fv:
    for out,c in cmds:
        fc.write(" ".join(shlex.quote(x) for x in c)+"\n"); fv.write(out+"\n")
done = sum(1 for o,_ in cmds if os.path.exists(o) and os.path.getsize(o)>0)
print(f"{len(cmds)} cells, {done} already done, {len(cmds)-done} to run")
