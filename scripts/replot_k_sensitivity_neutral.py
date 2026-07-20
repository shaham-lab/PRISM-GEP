"""Re-plot paper/figures/k_sensitivity.pdf from outputs/k_sweep/summary.csv.

Plots only, no metric recompute: the curves come straight from the sweep's summary CSV.

Blue  = beta recomputed at each k (the sweep).
Green = default (k=15) baseline +/- 1 std.

Legend labels describe what each series IS rather than where it came from, so the
figure reads without knowing the analysis history.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

WS = Path(__file__).resolve().parent.parent
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
# release layout: inputs from results/, figures to paper/figures/
# (scripts/paths.py); the authoring tree uses different roots.
from paths import figures_dir  # noqa: E402
sys.path.insert(0, str(WS / "scripts"))
from figsafe import save_and_deploy  # noqa: E402

# ---------------------------------------------------------------------------
# Print legibility (2026-07-20)
# ---------------------------------------------------------------------------
# supplementary.tex places this with height=0.6	extheight and keepaspectratio,
# and 	extheight here is 683pt, so the HEIGHT cap of 410pt is what binds, not
# the width. Drawn 16.4in (1181pt) tall the artwork was shrunk to 0.35x and this
# script never set a font size, so matplotlib's 10pt default landed at 3.5pt on
# paper. A shorter, narrower figure lifts the shrink to ~0.68x; the explicit
# sizes below are then close to what the reader gets.
FIG_W_IN = 9.70
ROW_IN = 1.05            # per-dataset row height (was 2.05)
F_TICK = 8.5
F_YLABEL = 8.8
F_TITLE = 10.5
F_LEGEND = 9.0

# breast_cancer omitted from the sensitivity figure: it is unlabeled and yields
# no significant GO-BP enrichment, so its Coverage/Strength panels are empty
# ("no values"). The eight enrichment-bearing datasets are shown; breast_cancer
# is still in the main benchmark table.
DATASETS = (
    "pbmc3k", "zeisel_brain", "pancreas", "bonemarrow",
    "hemogenic_endothelium", "gastrulation", "gastrulation_e75",
    "gastrulation_erythroid",
)
# Default (k=15) baseline, 10-seed mean (std) — same values previously labelled
# "paper beta"; relabelled here as the default operating point.
DEFAULT_BASELINE = {
    "breast_cancer": {"coh": (0.2677, 0.0012), "cov": (None, None),     "str": (None, None)},
    "pbmc3k":        {"coh": (0.3229, 0.0100), "cov": (0.0760, 0.0021), "str": (12.06, 0.51)},
    "zeisel_brain":  {"coh": (0.5202, 0.0384), "cov": (0.0758, 0.0024), "str": (2.69, 0.14)},
    "pancreas":      {"coh": (0.3118, 0.0396), "cov": (0.0573, 0.0049), "str": (10.03, 1.44)},
    "bonemarrow":    {"coh": (0.2696, 0.0069), "cov": (0.1094, 0.0224), "str": (2.87, 0.17)},
    "hemogenic_endothelium":  {"coh": (0.6216, 0.0152), "cov": (0.0769, 0.0026), "str": (6.69, 0.59)},
    "gastrulation":           {"coh": (0.4027, 0.0208), "cov": (0.0810, 0.0113), "str": (3.13, 0.02)},
    "gastrulation_e75":       {"coh": (0.6367, 0.0084), "cov": (0.0691, 0.0041), "str": (5.38, 0.14)},
    "gastrulation_erythroid": {"coh": (0.4246, 0.0130), "cov": (0.0514, 0.0025), "str": (1.78, 0.02)},
}
# Row labels are drawn rotated, so their length competes with the row HEIGHT.
# At the 2026-07-20 figure size a row is ~60pt tall, so the three longest names
# are broken over two lines rather than running into the neighbouring rows.
LABELS = {"breast_cancer": "BreastCancer", "pbmc3k": "PBMC3k",
          "zeisel_brain": "Zeisel_brain", "pancreas": "Pancreas",
          "bonemarrow": "Bonemarrow", "hemogenic_endothelium": "Hemogenic\nEndo.",
          "gastrulation": "Gastrulation", "gastrulation_e75": "Gastrulation\nE7.5",
          "gastrulation_erythroid": "Gastrulation\nEryth."}
K_VALUES = (5, 10, 15, 20, 30, 50)


def main() -> int:
    df = pd.read_csv(WS / "outputs" / "k_sweep" / "summary.csv")
    n_ds = len(DATASETS)
    fig, axes = plt.subplots(n_ds, 3, figsize=(FIG_W_IN, ROW_IN * n_ds), sharex=True)
    metrics = [("coh", "Coherence"), ("cov", "Coverage"), ("str", "Strength")]
    for col, (key, label) in enumerate(metrics):
        for row, ds in enumerate(DATASETS):
            ax = axes[row, col]
            sub = df[df["dataset"] == ds]
            ax.errorbar(sub["K"].values, sub[f"{key}_mean"].values,
                        yerr=sub[f"{key}_std"].values, marker="o", linestyle="-",
                        color="C0", capsize=2, markersize=3.0, linewidth=1.0,
                        elinewidth=0.8,
                        label=r"$\beta$ recomputed at each $k$")
            # green baseline = the sweep's OWN k=15 point (mean +/- 1 std), so it is
            # on the SAME metric basis as the blue curve and the k=15 sample sits on
            # the line. (Previously an external production baseline on a different
            # Strength basis, which sat far below the blue top-1 Strength points.)
            k15 = sub[sub["K"] == 15]
            bm = float(k15[f"{key}_mean"].iloc[0]) if len(k15) else float("nan")
            bs = float(k15[f"{key}_std"].iloc[0]) if len(k15) else 0.0
            if bm == bm:  # not NaN
                ax.axhline(bm, color="green", linestyle="-", linewidth=1.1,
                           label=r"default ($k$=15)")
                if bs and bs > 0:
                    ax.axhspan(bm - bs, bm + bs, color="green", alpha=0.12)
            ax.set_xticks(K_VALUES)
            ax.tick_params(labelsize=F_TICK, length=2.2, pad=1.5)
            ax.locator_params(axis="y", nbins=4)
            for sp in ax.spines.values():
                sp.set_linewidth(0.6)
            if row == n_ds - 1:
                ax.set_xlabel(r"$k$ (kNN-over-cells neighborhood size)", fontsize=F_YLABEL)
            if col == 0:
                ax.set_ylabel(LABELS[ds], fontsize=F_YLABEL, linespacing=1.0)
            if row == 0:
                ax.set_title(label, fontsize=F_TITLE)
            ax.grid(alpha=0.3, lw=0.5)
    # one shared legend (was 24 per-axes legends that overlapped the data)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=F_LEGEND,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    # in-figure descriptive title removed 2026-07-10 -> moved to LaTeX caption
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=0.5, w_pad=0.9)
    save_and_deploy(fig, figures_dir() / "k_sensitivity.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
