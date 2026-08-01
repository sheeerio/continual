# STEP 1 analysis: which (tasks, epochs) cell preserves the reference ordering?
#
# Absolute AUC is NOT comparable to the 100-task reference -- fewer epochs/task
# means less fitting per task, so every arm's AUC drops. Only the ORDERING is
# comparable, which is what the reference is used for here.
import csv, glob, os, statistics as st

ROOT = os.path.expanduser("~/scratch/calib")
CELLDIR = f"{ROOT}/cells"

# Completed 100-task reference run, ns=1.0 MNIST, 100 epochs/task, seeds 0/1/2.
REFERENCE = {"spectral": 0.742, "l2_loss": 0.625, "wass": 0.606, "vanilla": 0.117}
ARMS = ["spectral", "l2_loss", "wass", "vanilla"]
REF_ORDER = sorted(ARMS, key=lambda a: -REFERENCE[a])


def ranks(vals):
    """Ascending competition ranks with ties averaged."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else float("nan")


def load(tasks, epochs, arm):
    aucs, slopes, times = [], [], []
    for seed in (0, 1, 2):
        f = f"{CELLDIR}/t{tasks}_e{epochs}_{arm}_seed{seed}.csv"
        if not (os.path.exists(f) and os.path.getsize(f) > 0):
            continue
        row = list(csv.DictReader(open(f)))[0]
        aucs.append(float(row["auc"]))
        slopes.append(float(row["slope"]))
    return aucs, slopes


# Wall clock per cell, scraped from the harness CELLTIME lines.
celltime = {}
for log in glob.glob(f"{ROOT}/logs/*.out"):
    for line in open(log):
        if line.startswith("CELLTIME") and "rc=0" in line:
            parts = dict(p.split("=", 1) for p in line.split() if "=" in p)
            key = os.path.basename(parts.get("csv", "")).replace(".csv", "")
            if key:
                celltime[key] = int(parts["elapsed_s"])

GRID = [(t, e) for t in (50, 30, 20) for e in (50, 25, 10)]

print(f"Reference ordering (100 tasks x 100 epochs): {' > '.join(REF_ORDER)}")
print(f"  {REFERENCE}\n")
hdr = f"{'cell':<10} {'wall/cell':>10} {'n':>3} " + " ".join(f"{a:>16}" for a in ARMS) + f" {'rho':>6}  ordering"
print(hdr)
print("-" * len(hdr))

results = []
for tasks, epochs in GRID:
    means, stds, n_seeds = [], [], []
    for arm in ARMS:
        aucs, _ = load(tasks, epochs, arm)
        means.append(st.mean(aucs) if aucs else float("nan"))
        stds.append(st.stdev(aucs) if len(aucs) > 1 else 0.0)
        n_seeds.append(len(aucs))
    if any(n == 0 for n in n_seeds):
        print(f"t{tasks}_e{epochs:<5} {'INCOMPLETE':>10}")
        continue
    ref_vals = [REFERENCE[a] for a in ARMS]
    rho = spearman(means, ref_vals)
    got_order = " > ".join(sorted(ARMS, key=lambda a: -means[ARMS.index(a)]))
    times = [celltime.get(f"t{tasks}_e{epochs}_{a}_seed{s}", 0)
             for a in ARMS for s in (0, 1, 2)]
    times = [t for t in times if t]
    wall = f"{st.mean(times)/60:.1f}m" if times else "?"
    cells = f"t{tasks}_e{epochs}"
    body = " ".join(f"{m:>9.4f}+-{s:<5.4f}" for m, s in zip(means, stds))
    print(f"{cells:<10} {wall:>10} {min(n_seeds):>3} {body} {rho:>6.3f}  {got_order}")
    results.append((tasks, epochs, rho, means, stds, st.mean(times) if times else 0,
                    got_order == " > ".join(REF_ORDER)))

print()
print("Resolvability check: can 3 seeds resolve a 10% AUC difference?")
print("  criterion: 1.96 * std * sqrt(2/3) < 0.10 * mean_AUC  (two-sample, per arm)")
for tasks, epochs, rho, means, stds, wall, exact in results:
    ok = all(1.96 * s * (2 / 3) ** 0.5 < 0.10 * m for m, s in zip(means, stds) if m == m)
    worst = max((1.96 * s * (2 / 3) ** 0.5) / (0.10 * m) for m, s in zip(means, stds) if m)
    print(f"  t{tasks}_e{epochs:<4} exact_order={str(exact):<5} rho={rho:+.3f} "
          f"wall={wall/60:5.1f}m resolvable={str(ok):<5} (worst arm at {worst:.2f}x the budget)")

ok_cells = [r for r in results if r[6] and
            all(1.96 * s * (2 / 3) ** 0.5 < 0.10 * m for m, s in zip(r[3], r[4]) if m)]
if ok_cells:
    best = min(ok_cells, key=lambda r: r[0] * r[1])
    print(f"\nCHEAPEST cell with exact ordering AND resolvable seeds: "
          f"t{best[0]}_e{best[1]} ({best[0]*best[1]} task-epochs, {best[5]/60:.1f} min/cell)")
else:
    print("\nNo cell satisfies both exact ordering and the resolvability criterion.")
