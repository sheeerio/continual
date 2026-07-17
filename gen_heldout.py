import os, sys, shlex, itertools
H=os.environ["HOME"]
tag=sys.argv[1]; NS=sys.argv[2]; MODEL=sys.argv[3]
WIDTH=sys.argv[4] if len(sys.argv)>4 else None
ROOT=f"{H}/scratch/plast_{tag}"; CSVDIR=f"{ROOT}/cells"; os.makedirs(CSVDIR, exist_ok=True)
PY=f"{H}/venv/continual/bin/python"; SCRIPT="implicit_regularization.py"
base=["--optimizer","adam","--activation","relu","--runs","100","--epochs","100",
      "--dataset","MNIST","--model",MODEL,"--batch_size","256","--log_interval","400","--ns",NS]
if WIDTH: base += ["--hidden", WIDTH]
COEFFS=["1e-4","3e-4","1e-3","3e-3","1e-2","3e-2","1e-1"]
LRS=["1e-3","1e-2"]; SEEDS=["1","2"]
def sched(x):
    return ([],"none") if x=="none" else (["--lr_schedule","pl_lyapunov","--param","t"],"grad")
STATIC_REG={"l2":"l2_loss","spectral":"spectral","wass":"wass"}
LAMFLAG={"l2_loss":"--l2_lambda","spectral":"--spectral_lambda","wass":"--wass_lambda"}
cmds=[]
for scale,t,c,lr,sc,sd in itertools.product(["inv","saturating"],["l2","spectral","wass"],COEFFS,LRS,["none","grad"],SEEDS):
    sf,st=sched(sc); n=f"adapt_{t}_{scale}_sens{c}_{st}_lr{lr}_seed{sd}"; o=f"{CSVDIR}/{n}.csv"
    cmds.append((o,[PY,SCRIPT,*base,"--lr",lr,"--adaptive_reg","--adaptive_type",t,"--adaptive_scale",scale,
                    "--reg_sensitivity",c,*sf,"--seed",sd,"--name",n,"--exp_name",f"plast_{tag}","--results_csv",o]))
for t,c,lr,sc,sd in itertools.product(["l2","spectral","wass"],COEFFS,LRS,["none","grad"],SEEDS):
    reg=STATIC_REG[t]; sf,st=sched(sc); n=f"static_{t}_sens{c}_{st}_lr{lr}_seed{sd}"; o=f"{CSVDIR}/{n}.csv"
    cmds.append((o,[PY,SCRIPT,*base,"--lr",lr,"--reg",reg,LAMFLAG[reg],c,*sf,
                    "--seed",sd,"--name",n,"--exp_name",f"plast_{tag}","--results_csv",o]))
for lr,sc,sd in itertools.product(LRS,["none","grad"],SEEDS):
    sf,st=sched(sc); n=f"none_{st}_lr{lr}_seed{sd}"; o=f"{CSVDIR}/{n}.csv"
    cmds.append((o,[PY,SCRIPT,*base,"--lr",lr,*sf,"--seed",sd,"--name",n,"--exp_name",f"plast_{tag}","--results_csv",o]))
with open(f"{ROOT}/commands.txt","w") as fc, open(f"{ROOT}/csvs.txt","w") as fv:
    for o,c in cmds: fc.write(" ".join(shlex.quote(x) for x in c)+"\n"); fv.write(o+"\n")
print(f"{tag}: {len(cmds)} cells -> {ROOT}")
