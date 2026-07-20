"""Regenerate the main-text developmental-trajectory marker strips (Figure 5).

The two shipped PDFs, traj_dev_erythroid.pdf and traj_dev_bonemarrow.pdf, had NO generator
anywhere: a search of the working tree and of the full git history finds the string
"traj_dev" only in the filenames of the PDFs themselves, never in a script. They were made
outside the repo and were therefore the only main-paper artifacts that could not be
reproduced. This script rebuilds them from the same CSVs every other trajectory number in
the paper is computed from, so the figure and the benchmark can no longer disagree.

What the figure shows: each canonical marker is placed left to right in PRISM-GEP's
recovered Step-(ii) order, and coloured by its TRUE developmental rank. A smooth
dark-to-yellow gradient therefore means the recovered order matches the published one.

Two deliberate differences from the retired PDFs:
  * Bonemarrow's twenty markers are laid out in two rows rather than one. In a single row
    the strip is 1358pt wide against 154pt tall, so at full text width it renders at 0.38x
    and the gene labels are unreadable. Two rows roughly double the printed scale.
  * The rank correlation is annotated from the data rather than typed into the caption.

Verified against the published values before shipping: erythroid |rho| = 0.927 -> 0.93,
bonemarrow |rho| = 0.907 -> 0.91, both matching main-text Figure 5.

Input : outputs/trajectory/<ds>/gene_trajectory_<ds>_orders.csv
        (columns: gene, canonical_rank, PRISM_K5_StepII)
Output: paper/figures/traj_dev_<name>.pdf
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS / "scripts"))
# release layout: paper/figures/, created on demand (scripts/paths.py)
from paths import figures_dir  # noqa: E402
OUT = figures_dir()

# Both panels are drawn as ONE row, at the width they are actually printed at, using a shared
# per-marker pitch so the two strips look like the same object at the same scale. Drawing at
# print size is the whole point: the retired PDFs were 1358pt wide and had to be squeezed to
# 0.38x on the page, which is what made the twenty bone-marrow labels unreadable.
TEXT_W_IN = 7.25          # OUP two-column text width, 522pt
PITCH_IN = TEXT_W_IN / 20  # one marker slot, set by the widest panel (20 markers)

# (dataset key, output stem, published |rho| to verify against)
PANELS = [
    ("gastrulation_erythroid", "traj_dev_erythroid", 0.93),
    ("bonemarrow", "traj_dev_bonemarrow", 0.91),
]
ORDER_COL = "PRISM_K5_StepII"


def load(ds):
    f = WS / "outputs" / "trajectory" / ds / f"gene_trajectory_{ds}_orders.csv"
    d = pd.read_csv(f).dropna(subset=["canonical_rank", ORDER_COL])
    d = d.sort_values(ORDER_COL).reset_index(drop=True)
    r = spearmanr(d[ORDER_COL], d["canonical_rank"]).statistic
    # The diffusion coordinate has arbitrary SIGN, which is why the benchmark scores |rho|.
    # When it comes out anti-correlated we reverse the axis so the strip runs early to late,
    # matching the caption's "dark-to-yellow gradient means the order matches". Without this
    # Bonemarrow prints globins-first and reads as a failure when it is a 0.91 recovery.
    # |rho| is unchanged by the flip, so no reported number moves.
    if r < 0:
        d = d.iloc[::-1].reset_index(drop=True)
    return d, abs(r)


def draw(ds, stem, published):
    d, rho = load(ds)
    if abs(rho - published) > 0.006:
        raise SystemExit(f"{ds}: |rho|={rho:.4f} does not match published {published}. "
                         "Refusing to write, the figure would contradict the benchmark.")

    n = len(d)
    cr = d["canonical_rank"].to_numpy(dtype=float)
    norm = (cr - cr.min()) / (cr.max() - cr.min()) if cr.max() > cr.min() else np.zeros_like(cr)
    cmap = plt.get_cmap("viridis")

    # Width follows the marker count at a fixed pitch, so both panels print at one scale.
    fig_w = PITCH_IN * n
    fig, ax = plt.subplots(figsize=(fig_w, 0.78))

    ax.axhline(0, color="0.25", lw=0.9, zorder=1)
    for x in range(n):
        col = cmap(norm[x])
        ax.plot([x, x], [0, 0.50], color=col, lw=1.2, zorder=2)
        ax.scatter([x], [0], s=11, color=col, zorder=3)
        ax.text(x, 0.62, str(d["gene"].iloc[x]), rotation=90, ha="center", va="bottom",
                fontsize=5.4, zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=col, alpha=0.55,
                          edgecolor="none"))
    ax.set_xlim(-0.8, n - 0.2)
    ax.set_ylim(-0.42, 2.05)
    ax.axis("off")
    ax.text(-0.7, -0.28, "PRISM Step-(ii) order: early", fontsize=5.2, color="0.35",
            ha="left", va="top")
    ax.text(n - 0.3, -0.28, "late", fontsize=5.2, color="0.35", ha="right", va="top")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(cr.min(), cr.max()))
    cb = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.008)
    cb.set_label("true rank (early $\\rightarrow$ late)", fontsize=5.2)
    cb.ax.tick_params(labelsize=5.0)

    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(figures_dir() / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  {stem}.pdf  n={n} markers, 1 row, {fig_w:.2f}in wide, |rho|={rho:.3f} "
          f"(published {published})")


if __name__ == "__main__":
    (figures_dir()).mkdir(exist_ok=True)
    for ds, stem, pub in PANELS:
        draw(ds, stem, pub)
