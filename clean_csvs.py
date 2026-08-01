# TASK 2a: recover the stringified-tensor columns from existing CSVs.
#
# sharp, mu, tau, cv were written as "tensor(0.9315, device='cuda:0')" because
# get_norm_sharpness returned a 0-dim tensor. The values are present and
# recoverable -- this rewrites cleaned copies ALONGSIDE the originals
# (*.clean.csv), never overwriting, so the raw record is preserved.
import csv, os, re, sys, glob

TENSOR_RE = re.compile(r"tensor\(\s*([-+0-9.eEinfa]+)")


def tolerant_float(v):
    """Return (value, was_tensor_repr) or (None, False) if unparseable."""
    if v is None:
        return None, False
    s = v.strip()
    if s == "":
        return None, False
    try:
        return float(s), False
    except ValueError:
        pass
    m = TENSOR_RE.match(s)
    if m:
        try:
            return float(m.group(1)), True
        except ValueError:
            return None, True
    return None, False


def clean_file(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    cols = list(rows[0].keys())
    recovered = {}
    unparseable = {}
    out = []
    for r in rows:
        nr = {}
        for c in cols:
            v = r.get(c, "")
            # Leave known non-numeric columns untouched.
            if c in ("layer", "git_sha", "full_config", "model", "dataset", "reg",
                     "adaptive_type", "adaptive_scale", "lr_schedule", "sched_param",
                     "task_acc_traj") or c.endswith("_traj"):
                nr[c] = v
                continue
            fv, was_tensor = tolerant_float(v)
            if was_tensor:
                recovered[c] = recovered.get(c, 0) + 1
            if fv is None:
                nr[c] = v
                if v.strip() != "":
                    unparseable[c] = unparseable.get(c, 0) + 1
            else:
                nr[c] = repr(fv)
        out.append(nr)
    dest = path[:-4] + ".clean.csv"
    with open(dest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    return dest, len(rows), recovered, unparseable


DIRS = [
    "~/scratch/testbed/cells",
    "~/scratch/testbed_extend/cells",
    "~/scratch/stability_probe/cells",
    "~/scratch/efflr_probe/cells",
    "~/scratch/centered_check/cells",
]

total_files = 0
total_recovered = 0
for d in DIRS:
    d = os.path.expanduser(d)
    if not os.path.isdir(d):
        print(f"SKIP (missing): {d}")
        continue
    files = [f for f in sorted(glob.glob(f"{d}/*.csv")) if not f.endswith(".clean.csv")]
    dir_rec = {}
    dir_unp = {}
    for f in files:
        res = clean_file(f)
        if res is None:
            continue
        _, n, rec, unp = res
        total_files += 1
        for k, v in rec.items():
            dir_rec[k] = dir_rec.get(k, 0) + v
            total_recovered += v
        for k, v in unp.items():
            dir_unp[k] = dir_unp.get(k, 0) + v
    print(f"\n{d}")
    print(f"  files cleaned: {len(files)}")
    if dir_rec:
        print(f"  recovered tensor-repr cells: " +
              ", ".join(f"{k}={v}" for k, v in sorted(dir_rec.items())))
    else:
        print("  recovered tensor-repr cells: none (already clean)")
    if dir_unp:
        print(f"  STILL UNPARSEABLE: " +
              ", ".join(f"{k}={v}" for k, v in sorted(dir_unp.items())))

print(f"\nTOTAL: {total_files} files, {total_recovered} tensor-repr values recovered")
