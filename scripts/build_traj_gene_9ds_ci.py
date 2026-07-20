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
Output : paper/figures/tab_traj_gene_9ds_ci.tex
"""
import argparse
from pathlib import Path

import pandas as pd

WS = Path(__file__).resolve().parent.parent
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
# release layout: paper/figures/, created on demand (scripts/paths.py)
from paths import figures_dir  # noqa: E402
TRAJ = WS / "outputs" / "trajectory"
TENSEED = TRAJ / "tenseed_2026-07-20" / "gene_traj_tenseed.csv"
DEFAULT_OUT = figures_dir() / "tab_traj_gene_9ds_ci.tex"

DS = [
    ("pancreas", "Pancreas"),
    ("gastrulation", "Gastrulation"),
    ("gastrulation_erythroid", "Gastr.\\ Erythroid"),
    ("hemogenic_endothelium", "Hemogenic Endo."),
    ("bonemarrow", "Bonemarrow"),
    ("paul15", "Paul15"),
    ("dentategyrus", "Dentate Gyrus"),
    ("endoderm_diff", "Endoderm"),
    ("gastrulation_e75", "Gastrulation E7.5$^{\\S}$"),
]

# (display name, method key in the ten-seed CSV)
METHODS = [
    ("PRISM-GEP Step (ii)", "PRISM_K5_StepII"),
    ("GeneTrajectory (best traj.)", "GT_best_traj"),
    ("GeneTrajectory (extract)", "GeneTrajectory"),
    ("GeneTrajectory (EV)", "GeneTrajectory_EV"),
    ("DPT-weighted-mean$^{\\dagger}$", "DPT_weighted_mean"),
    ("Expression magnitude", "Expression_magnitude"),
]


def f3(x):
    if pd.isna(x):
        return "---"
    return f"{x:.3f}".lstrip("0") if 0 <= x < 1 else f"{x:.3f}"


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

    lines = []
    n_ci = 0
    for di, (ds, label) in enumerate(DS):
        if di:
            lines.append("\\midrule")
        n = len(METHODS)
        for mi, (name, key) in enumerate(METHODS):
            first = f"\\multirow{{{n}}}{{*}}{{{label}}}" if mi == 0 else ""
            r = T.loc[(ds, key)]
            point, sd = f3(r["mean"]), f3(r["sd"])
            if pd.notna(r["boot_mean"]):
                bm = f3(r["boot_mean"])
                cistr = f"[{f3(r['ci_lo'])}, {f3(r['ci_hi'])}]"
                # n is the marker PANEL size, the meaning this column has always carried.
                nm = str(int(r["n_panel"]))
                n_ci += 1
            else:
                bm, cistr, nm = "---", "---", "---"
            lines.append(f"{first} & {name} & {point} & {sd} & {bm} & {cistr} & {nm} \\\\")

    frag = "\n".join([
        "% AUTO-GENERATED by scripts/build_traj_gene_9ds_ci.py -- do not hand-edit.",
        "% Point estimates and s_seed are TEN-SEED; the interval is a marker bootstrap",
        "% pooled over those same ten seeds. See scripts/build_traj_gene_tenseed.py.",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Dataset & Method & $|\\rho|$ & $s_{\\mathrm{seed}}$ & boot.\\ mean "
        "& 95\\% CI$_{\\mathrm{marker}}$ & $n$ \\\\",
        "\\midrule",
        *lines,
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ])
    out.write_text(frag, encoding="utf-8")
    print(f"wrote {out}")
    print(f"{len(DS)} datasets x {len(METHODS)} methods = {len(DS)*len(METHODS)} rows")
    print(f"pooled/carried CI available on {n_ci} of {len(DS)*len(METHODS)} cells")


if __name__ == "__main__":
    main()
