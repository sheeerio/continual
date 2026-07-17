import os, shlex, itertools
H=os.environ["HOME"]
ROOT=f"{H}/scratch/plast"; CSVDIR=f"{ROOT}/cells"; os.makedirs(CSVDIR, exist_ok=True)
PY=f"{H}/venv/continual/bin/python"; SCRIPT="implicit_regularization.py"
base=["--optimizer","adam","--activation","relu","--runs","100","--epochs","100",
      "--dataset","MNIST","--model","MLP","--batch_size","256","--log_interval","400","--ns","1.0"]
COEFFS=["1e-4","3e-4","1e-3","3e-3","1e-2","3e-2","1e-1"]
LRS=["1e-2","1e-3"]; SEEDS=["1","2","3"]
ADAPT_TYPES=["l2","spectral","wass","parseval"]
STATIC_TYPES=["l2","spectral","wass","parseval","l2_loss"]
LAMFLAG={"l2":"--l2_lambda","l2_loss":"--l2_lambda","spectral":"--spectral_lambda",
         "wass":"--wass_lambda","parseval":"--parseval_lambda"}

def sched(s):
    if s=="none":  return [], "none"
    if s=="grad":  return ["--lr_schedule","pl_lyapunov","--param","t"], "grad"
    if s=="reset": return ["--reset_model"], "reset"

cmds=[]
for t,c,lr,sc,sd in itertools.product(ADAPT_TYPES,COEFFS,LRS,["none","grad"],SEEDS):
    sf,st=sched(sc); n=f"adapt_{t}_sens{c}_{st}_lr{lr}_seed{sd}"; o=f"{CSVDIR}/{n}.csv"
    cmds.append((o,[PY,SCRIPT,*base,"--lr",lr,"--adaptive_reg","--adaptive_type",t,
                    "--reg_sensitivity",c,*sf,"--seed",sd,"--name",n,"--exp_name","plast3","--results_csv",o]))
for t,c,lr,sc,sd in itertools.product(STATIC_TYPES,COEFFS,LRS,["none","grad"],SEEDS):
    sf,st=sched(sc); n=f"static_{t}_lam{c}_{st}_lr{lr}_seed{sd}"; o=f"{CSVDIR}/{n}.csv"
    cmds.append((o,[PY,SCRIPT,*base,"--lr",lr,"--reg",t,LAMFLAG[t],c,*sf,
                    "--seed",sd,"--name",n,"--exp_name","plast3","--results_csv",o]))
for lr,sc,sd in itertools.product(LRS,["none","grad","reset"],SEEDS):
    sf,st=sched(sc); n=f"none_{st}_lr{lr}_seed{sd}"; o=f"{CSVDIR}/{n}.csv"
    cmds.append((o,[PY,SCRIPT,*base,"--lr",lr,*sf,"--seed",sd,"--name",n,"--exp_name","plast3","--results_csv",o]))

with open(f"{ROOT}/commands.txt","w") as fc, open(f"{ROOT}/csvs.txt","w") as fv:
    for o,c in cmds:
        fc.write(" ".join(shlex.quote(x) for x in c)+"\n"); fv.write(o+"\n")
done=sum(1 for o,_ in cmds if os.path.exists(o) and os.path.getsize(o)>0)
print(f"{len(cmds)} cells, {done} already done, {len(cmds)-done} to run")
