"""Build the merged gene-trajectory table: nine datasets, TEN-SEED point estimate + CI.

TEN-SEED, 2026-07-20. This table previously reported seed 0 while main Table 1 reported the
ten-seed mean of the same quantity, so one claim carried two numbers (Pancreas .742 here
against .844 there, E7.5 .062 against .345). Every stochastic row is now the ten-seed mean,
so the two tables agree by construction.

The table carries TWO different uncertainties, in separate columns, because they answer
different questions and a reader must never have to guess which a bracket refers to:

  |rho|            mean over ten model fits          "what does the method score?"
  s_seed           sd over those same ten fits       "how much does the fit matter?"
  boot. mean, CI   POOLED marker bootstrap           "how much does the marker panel matter,
                   over (seed, marker resample)       having also averaged over fits?"

The interval is a MARKER bootstrap pooled across seeds: each of B=5000 replicates draws a
seed uniformly from the ten and resamples the marker set with replacement, scoring within
that one seed. It is a different axis from s_seed, and from the seed sd in main Table 1.
Attaching the old seed-0 interval to a ten-seed point estimate would re-create exactly the
mismatch this rewrite removes.

The point estimate and the bootstrap mean are BOTH shown because they remain different
quantities: the bootstrap mean is an average over resampled marker sets and drifts from the
point estimate when the marker set is small.

Blank cells are real, and there are three distinct reasons for them:
  1. The root-supervised DPT baseline is undefined where no root cell is defined (Dentate
     Gyrus, Endoderm, Paul15).
  2. The pooled bootstrap needs a method's ordering to be complete on every marker in every
     seed, so GeneTrajectory (extract) has an interval only where its optimal-transport
     extraction assigned all canonical markers to a trajectory. Same rule as the previous
     single-seed bootstrap, applied across ten seeds instead of one.
  3. The best-trajectory variant has NO interval, because it is scored from
     gt_full_trajectories on a per-trajectory marker SUBSET that varies with the seed, so
     there is no fixed marker panel to resample. It does now carry a ten-seed mean and sd.

DPT and expression magnitude are seed-invariant (measured, sd = 0.0000), so their pooled
bootstrap reduces exactly to the single-seed bootstrap already published and their intervals
are carried forward bit-unchanged. See scripts/build_traj_gene_tenseed.py.

Inputs : outputs/trajectory/tenseed_2026-07-20/gene_traj_tenseed.csv
Output : results/tables/tab_traj_gene_9ds_ci.csv
         One row per (dataset, method): rho (ten-seed mean), s_seed, boot_mean, ci_lo,
         ci_hi, n_panel. All scores to 3 decimals; empty cells are the blank cells
         explained above.
"""
import argparse
from pathlib import Path

import pandas as pd

WS = Path(__file__).resolve().parent.parent
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
# release layout: results/tables/, created on demand (scripts/paths.py)
from paths import tables_dir  # noqa: E402
TRAJ = WS / "outputs" / "trajectory"
TENSEED = TRAJ / "tenseed_2026-07-20" / "gene_traj_tenseed.csv"
DEFAULT_OUT = tables_dir() / "tab_traj_gene_9ds_ci.csv"

DS = [
    ("pancreas", "Pancreas"),
    ("gastrulation", "Gastrulation"),
    ("gastrulation_erythroid", "Gastr. Erythroid"),
    ("hemogenic_endothelium", "Hemogenic Endo."),
    ("bonemarrow", "Bonemarrow"),
    ("paul15", "Paul15"),
    ("dentategyrus", "Dentate Gyrus"),
    ("endoderm_diff", "Endoderm"),
    ("gastrulation_e75", "Gastrulation E7.5"),
]

# (display name, method key in the ten-seed CSV)
METHODS = [
    ("PRISM-GEP Step (ii)", "PRISM_K5_StepII"),
    ("GeneTrajectory (best traj.)", "GT_best_traj"),
    ("GeneTrajectory (extract)", "GeneTrajectory"),
    ("GeneTrajectory (EV)", "GeneTrajectory_EV"),
    ("DPT-weighted-mean", "DPT_weighted_mean"),
    ("Expression magnitude", "Expression_magnitude"),
]


def f3(x):
    """Three decimals as a plain number; blank cells stay blank (None)."""
    if pd.isna(x):
        return None
    return round(float(x), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    T = pd.read_csv(TENSEED).set_index(["dataset", "method"])

    missing = [(d, k) for d, _ in DS for _, k in METHODS if (d, k) not in T.index]
    if missing:
        raise SystemExit(f"ten-seed rows missing for: {missing}")

    rows = []
    n_ci = 0
    for ds, label in DS:
        for name, key in METHODS:
            r = T.loc[(ds, key)]
            row = dict(dataset=ds, dataset_label=label, method=name,
                       rho=f3(r["mean"]), s_seed=f3(r["sd"]),
                       boot_mean=None, ci_lo=None, ci_hi=None, n_panel=None)
            if pd.notna(r["boot_mean"]):
                row["boot_mean"] = f3(r["boot_mean"])
                row["ci_lo"], row["ci_hi"] = f3(r["ci_lo"]), f3(r["ci_hi"])
                # n is the marker PANEL size, the meaning this column has always carried.
                row["n_panel"] = int(r["n_panel"])
                n_ci += 1
            rows.append(row)

    # Point estimates and s_seed are TEN-SEED; the interval is a marker bootstrap
    # pooled over those same ten seeds. See scripts/build_traj_gene_tenseed.py.
    df = pd.DataFrame(rows)
    df["n_panel"] = df["n_panel"].astype("Int64")
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    print(f"{len(DS)} datasets x {len(METHODS)} methods = {len(DS)*len(METHODS)} rows")
    print(f"pooled/carried CI available on {n_ci} of {len(DS)*len(METHODS)} cells")


if __name__ == "__main__":
    main()
