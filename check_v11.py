# Compare the v1.1 verification cell against the stored pre-fix row for the
# identical config (unit/tui_N5_s0, git_sha 449dcc0-dirty). Both fixes are
# claimed to be exact no-ops, so this must be bitwise, not within-tolerance.
import csv, os, sys, json

H = os.environ["HOME"]
OLD = f"{H}/scratch/unit/cells/tui_N5_s0.csv"
NEW = f"{H}/scratch/v11/cells/v11_tui_N5_s0.csv"

if not os.path.exists(NEW):
    sys.exit(f"missing {NEW} -- cell has not written yet")

o = list(csv.DictReader(open(OLD)))[0]
n = list(csv.DictReader(open(NEW)))[0]

print(f"old git_sha : {o['git_sha']}")
print(f"new git_sha : {n['git_sha']}")
print()

fails = []


def cmp(label, a, b, exact=True):
    ok = (a == b) if exact else (abs(float(a) - float(b)) <= 1e-12)
    print(f"{label:<26} {'MATCH' if ok else 'DIFFER'}")
    if not ok:
        print(f"    old {a}\n    new {b}")
        fails.append(label)


cmp("task_acc_traj", o["task_acc_traj"], n["task_acc_traj"])
cmp("auc", o["auc"], n["auc"])
cmp("slope", o["slope"], n["slope"])
cmp("final_acc", o["final_acc"], n["final_acc"])

# every per-task diagnostic trajectory that both rows carry
for k in sorted(set(o) & set(n)):
    if k.endswith("_traj") and k != "task_acc_traj":
        cmp(k, o[k], n[k])

# config must differ ONLY by the fields the fixes introduced
co, cn = json.loads(o["full_config"]), json.loads(n["full_config"])
diff = {k for k in set(co) | set(cn) if co.get(k) != cn.get(k)}
expected = {"make_plots", "name", "results_csv", "taskdiag_csv", "diag_csv", "exp_name"}
print(f"\nconfig keys differing: {sorted(diff)}")
unexpected = diff - expected
if unexpected:
    print(f"  UNEXPECTED: {sorted(unexpected)}")
    fails.append("config")

print("\nVERDICT:", "PASS (bitwise identical)" if not fails else f"FAIL on {fails}")
