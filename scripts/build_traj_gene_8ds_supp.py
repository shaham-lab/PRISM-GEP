"""Main-text Figure 6: mean/median gene-trajectory recovery across nine datasets.

TEN-SEED, 2026-07-20. Every stochastic row is now the mean over the same ten model fits that
main Table 1 reports, so the figure's in-image "mean"/"med" annotations agree with Table 1
instead of contradicting it on the same page spread (PRISM mean was .706 from seed 0 alone,
it is .752 over ten seeds).

Numbers come from outputs/trajectory/tenseed_2026-07-20/gene_traj_tenseed.csv, which is built
by scripts/build_traj_gene_tenseed.py and is the SAME file behind Supplementary Table S11, so
the figure and that table cannot drift apart.

The chance level is read from the ten-seed run's null (10,000 random orderings per dataset),
which is the identical series printed in main Table 1's "random (chance level)" row. It used
to be recomputed here at n_iter=1000 via random_baseline_mean_rho, which disagreed both with
Table 1 and with the surrounding prose that says 10,000.

Outputs:
  paper/figures/traj_gene_8ds_meanmedian.pdf    (main-text Figure 6, mean/median panel)

The LaTeX fragment this script once emitted (tab_traj_gene_8ds.tex) is NO LONGER BUILT
as of 2026-07-20. Neither main.tex nor supplementary.tex ever inputs it. The live
per-dataset table is figures/tab_traj_gene_9ds_ci.tex from scripts/build_traj_gene_9ds_ci.py.
See the comment block in build_table() before re-enabling anything.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

WS = Path(__file__).resolve().parent.parent
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
# release layout: paper/figures/, created on demand (scripts/paths.py)
from paths import figures_dir, tables_dir  # noqa: E402
sys.path.insert(0, str(WS / "scripts"))
from figsafe import save_and_deploy  # noqa: E402

# ---------------------------------------------------------------------------
# Print legibility (2026-07-20)
# ---------------------------------------------------------------------------
# This is main-text Figure 6, placed at 0.82\linewidth of a single column, i.e.
# 208pt against a 245pt drawing (0.85x). The in-panel mean/median values were
# set at 5.6pt and therefore printed at 4.8pt. Every size below is raised so the
# smallest text clears 5.5pt on the page; the canvas is unchanged, so the figure
# occupies the same box.
FIG_W_IN, FIG_H_IN = 3.5, 3.0
F_ANNOT = 6.9     # in-panel "mean 0.752" / "med 0.781"   (was 5.6)
F_YTICK = 7.1     # method row labels                     (was 6.0)
F_XTICK = 6.9     # x tick labels                         (was 6.0)
F_XLABEL = 7.4    # x axis label                          (was 6.4)

TEN = pd.read_csv(WS / "outputs" / "trajectory" / "tenseed_2026-07-20" / "gene_traj_tenseed.csv")
V = TEN.pivot(index="dataset", columns="method", values="mean")
DS = ["pancreas", "gastrulation", "gastrulation_erythroid", "hemogenic_endothelium", "bonemarrow",
      "paul15", "dentategyrus", "endoderm_diff", "gastrulation_e75"]
HEAD = ["Panc.", "Gastr.", "Eryth.", "Hemog.", "Bone.", "Paul15", "DG", "Endo.", "E75$^{\\S}$"]

# Chance level: the 10,000-draw null from the ten-seed run, i.e. literally the same numbers
# main Table 1 prints. NOT recomputed here, so the two can never disagree.
_null = pd.read_csv(WS / "outputs" / "gene_embedding_ablation" / "tenseed_2026-07-20"
                    / "perseed_prism_and_random.csv")
_null = _null[_null.method == "random_full"].set_index("dataset")["spearman_abs"]
EXPR = {d: V.loc[d, "Expression_magnitude"] for d in DS}
RAND = {d: float(_null[d]) for d in DS}

# rows: (label, per-ds value list); "real" methods count toward per-column best
ROWS = [
    ("PRISM-GEP $K{=}5$ Step (ii)",      [V.loc[d, "PRISM_K5_StepII"] for d in DS],   True),
    ("GeneTrajectory (best trajectory)", [V.loc[d, "GT_best_traj"] for d in DS],      True),
    ("GeneTrajectory (extract, native)", [V.loc[d, "GeneTrajectory"] for d in DS],    True),
    (r"GeneTrajectory (EV)$^{\ddagger}$", [V.loc[d, "GeneTrajectory_EV"] for d in DS], True),
    ("Expression magnitude",             [EXPR[d] for d in DS],                       False),
    ("Random chance",                    [RAND[d] for d in DS],                       False),
]


# Gastrulation E7.5 is ill-posed for gene-trajectory recovery (a multi-lineage snapshot with no
# single developmental axis). It STAYS in the table, but every aggregate is reported both over all
# datasets and, in parentheses, over the datasets with E7.5 excluded, so nothing is hidden.
E75 = "gastrulation_e75"
KEEP = [j for j, d in enumerate(DS) if d != E75]


def agg(v):
    a = [x for x in v if pd.notna(x)]
    return (np.mean(a) if a else np.nan, np.median(a) if a else np.nan)


def agg_red(v):
    """Same aggregate, but over the datasets minus Gastrulation E7.5."""
    return agg([v[j] for j in KEEP])


def fmt_cell(x):
    if pd.isna(x):
        return "---"
    return f"{x:.3f}".lstrip("0") if 0 <= x < 1 else f"{x:.3f}"


def build_table():
    means = {i: agg(v)[0] for i, (_, v, _) in enumerate(ROWS)}
    meds = {i: agg(v)[1] for i, (_, v, _) in enumerate(ROWS)}
    means_r = {i: agg_red(v)[0] for i, (_, v, _) in enumerate(ROWS)}
    meds_r = {i: agg_red(v)[1] for i, (_, v, _) in enumerate(ROWS)}
    # per-column best among REAL methods only
    real = [i for i, (_, _, r) in enumerate(ROWS) if r]
    col_best = []
    for j in range(len(DS)):
        vals = [ROWS[i][1][j] for i in real if pd.notna(ROWS[i][1][j])]
        col_best.append(max(vals) if vals else None)
    mean_best = max(means[i] for i in real)
    med_best = max(meds[i] for i in real)

    # This script produces main-text Figure 6. The numbers behind it are written here as a
    # CSV so the figure can be checked against the values it plots. Aggregates are reported
    # over all datasets and, in the *_excl_e75 columns, with Gastrulation E7.5 left out.
    # This repository generates no LaTeX. The manuscript's table fragments are produced
    # outside it, from the CSVs in results/tables/.
    recs = []
    for i, (name, v, isreal) in enumerate(ROWS):
        r = {"method": name, "is_real_method": bool(isreal)}
        for j, ds in enumerate(DS):
            r[ds] = None if pd.isna(v[j]) else round(float(v[j]), 3)
        r["mean"] = round(float(means[i]), 3)
        r["median"] = round(float(meds[i]), 3)
        r["mean_excl_e75"] = round(float(means_r[i]), 3)
        r["median_excl_e75"] = round(float(meds_r[i]), 3)
        r["best_mean"] = bool(isreal and abs(means[i] - mean_best) < 1e-9)
        r["best_median"] = bool(isreal and abs(meds[i] - med_best) < 1e-9)
        recs.append(r)
    out = tables_dir() / "tab_traj_gene_8ds_meanmedian.csv"
    pd.DataFrame(recs).to_csv(out, index=False)
    print("wrote", out)
    print("means      (full):", {ROWS[i][0][:14]: round(means[i], 3) for i in range(len(ROWS))})
    print("means (excl E7.5):", {ROWS[i][0][:14]: round(means_r[i], 3) for i in range(len(ROWS))})
    print("medians     (full):", {ROWS[i][0][:14]: round(meds[i], 3) for i in range(len(ROWS))})
    print("medians(excl E7.5):", {ROWS[i][0][:14]: round(meds_r[i], 3) for i in range(len(ROWS))})


def build_viz():
    labels = ["PRISM-GEP\n$K{=}5$", "GeneTraj.\n(best traj.)", "GeneTraj.\n(extract)",
              "GeneTraj.\n(EV)", "Expression\nmagnitude", "Random\nchance"]
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#e377c2", "#7f7f7f", "#bcbcbc"]
    # Sized for ONE column of the two-column OUP layout (~3.5in), where this figure
    # now lives in the main text. Drawing it at its final printed size keeps the
    # labels legible instead of scaling an 8.6in canvas down to 40%.
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    y = np.arange(len(ROWS))[::-1]
    for i, (_, v, _) in enumerate(ROWS):
        pts = np.array([x for x in v if pd.notna(x)])
        mean, med = agg(v)
        ax.barh(y[i], mean, height=0.62, color=colors[i], alpha=0.32, zorder=1)
        # per-dataset dots
        ax.scatter(pts, np.full_like(pts, y[i]) + np.linspace(-0.16, 0.16, len(pts)),
                   s=8, color=colors[i], edgecolor="white", linewidth=0.3, zorder=3)
        # mean (solid bar edge marker) + median (diamond)
        ax.plot([mean, mean], [y[i] - 0.31, y[i] + 0.31], color=colors[i], lw=1.4, zorder=4)
        ax.scatter([med], [y[i]], marker="D", s=22, facecolor="white",
                   edgecolor=colors[i], linewidth=1.1, zorder=5)
        ax.text(mean + 0.015, y[i] + 0.21, f"mean {mean:.3f}", fontsize=F_ANNOT, va="center", color=colors[i])
        ax.text(med + 0.015, y[i] - 0.23, f"med {med:.3f}", fontsize=F_ANNOT, va="center", color="0.35")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=F_YTICK)
    ax.tick_params(axis="x", labelsize=F_XTICK, length=2, pad=1.5)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.set_xlabel(r"$|\mathrm{Spearman}\ \rho|$ vs canonical marker order  (9 datasets)", fontsize=F_XLABEL)
    ax.set_xlim(0, 1.0); ax.axvline(0, color="0.8", lw=0.8)
    # in-figure descriptive title removed 2026-07-10 -> moved to LaTeX caption
    ax.grid(axis="x", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_and_deploy(fig, figures_dir() / "traj_gene_8ds_meanmedian.pdf", bbox_inches="tight")
    print("wrote traj_gene_8ds_meanmedian.pdf")


if __name__ == "__main__":
    build_table()
    build_viz()
