# gen_plast.py
import os, shlex, itertools
H=os.environ["HOME"]
ROOT=f"{H}/scratch/plast"; CSVDIR=f"{ROOT}/cells"; os.makedirs(CSVDIR, exist_ok=True)
PY=f"{H}/venv/continual/bin/python"; SCRIPT="implicit_regularization.py"
base=["--optimizer","adam","--activation","relu","--runs","100","--epochs","100",
      "--dataset","MNIST","--model","MLP","--lr","1e-2","--batch_size","256","--log_interval","400"]
seeds=[1,2,3]
PARAM_FLAG="--param"   # change to --sched_param if the config check says so

def sched_flags(s):
    if s=="none":  return [], "none"
    if s=="grad":  return ["--lr_schedule","pl_lyapunov",PARAM_FLAG,"t"], "grad"
    if s=="svar":  return ["--lr_schedule","pl_lyapunov",PARAM_FLAG,"svar10"], "svar"
    if s=="reset": return ["--reset_model"], "reset"

cmds=[]
# adaptive: old working sensitivity/tau formula, local scope
for atype,sens,sched,seed in itertools.product(["l2","spectral"],["1e-4","1e-3","1e-2"],["none","grad"],seeds):
    sf,stag=sched_flags(sched)
    name=f"adapt_{atype}_sens{sens}_{stag}_seed{seed}"; out=f"{CSVDIR}/{name}.csv"
    cmds.append((out,[PY,SCRIPT,*base,"--adaptive_reg","--adaptive_type",atype,"--reg_sensitivity",sens,
                      *sf,"--seed",str(seed),"--name",name,"--exp_name","plast","--results_csv",out]))
# static control
for rtype,lam,sched,seed in itertools.product(["l2","spectral"],["1e-4","1e-3","1e-2"],["none","grad","svar"],seeds):
    sf,stag=sched_flags(sched)
    name=f"static_{rtype}_lam{lam}_{stag}_seed{seed}"; out=f"{CSVDIR}/{name}.csv"
    c=[PY,SCRIPT,*base,"--reg",rtype,*sf,"--seed",str(seed),"--name",name,"--exp_name","plast","--results_csv",out]
    c += ["--l2_lambda",lam] if rtype=="l2" else ["--spectral_lambda",lam]
    cmds.append((out,c))
# vanilla baselines
for sched,seed in itertools.product(["none","grad","reset"],seeds):
    sf,stag=sched_flags(sched)
    name=f"none_{stag}_seed{seed}"; out=f"{CSVDIR}/{name}.csv"
    cmds.append((out,[PY,SCRIPT,*base,*sf,"--seed",str(seed),"--name",name,"--exp_name","plast","--results_csv",out]))

with open(f"{ROOT}/commands.txt","w") as fc, open(f"{ROOT}/csvs.txt","w") as fv:
    for out,c in cmds:
        fc.write(" ".join(shlex.quote(x) for x in c)+"\n"); fv.write(out+"\n")
print(len(cmds),"cells")   # 36 + 54 + 9 = 99
