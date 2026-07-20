"""Build the 9-dataset cell-ordering table fragment for the supplement.

Roster: the nine datasets that carry a canonical marker order, i.e. exactly the roster
of the main-text gene-trajectory figure. Fixing it that way matters -- it is inherited
from a table we already ship rather than chosen after seeing these numbers, so it is not
open to the charge that the datasets were picked to flatter the result. It also excludes,
by rule rather than by score, the three datasets whose "ordering" we invented for lack of
a developmental axis (Zeisel, PBMC3k, PBMC68k).

All methods are scored against the identical rank vector on the identical kept-cell subset.
PRISM is K_topics=5 on every dataset (the only K available for all nine), which differs from
the older three-row table's K=#published-cell-types -- do not mix values across the two.

PCA-1 is included deliberately: on several rows the first principal component of the
expression matrix lands within .05 of the best method, and without that column a reader
cannot tell a real recovery from variance-tracking.

Inputs : outputs/trajectory/cell_traj_all_datasets.csv   (PRISM, 10 seeds)
         outputs/trajectory/cell_traj_baselines_all.csv  (Slingshot, DPT, PCA-1)
Output : paper/figures/tab_traj_cell_9ds.tex
"""
import argparse
from pathlib import Path

import pandas as pd

WS = Path(__file__).resolve().parent.parent
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
# release layout: paper/figures/, created on demand (scripts/paths.py)
from paths import figures_dir  # noqa: E402
# Default writes to the release figure directory, paper/figures/, which figures_dir()
# creates on demand. --out overrides it, which is how the authoring repo redirects the
# fragment into its own manuscript tree. The parent is created either way rather than
# failing late, after the numbers have already been computed.
DEFAULT_OUT = figures_dir() / "tab_traj_cell_9ds.tex"

# (key, printed name) in the display order used by the gene-trajectory table
NINE = [
    ("pancreas", "Pancreas"),
    ("gastrulation", "Gastrulation"),
    ("gastrulation_erythroid", "Gast.\\ Eryth."),
    ("hemogenic_endothelium", "Hemogenic$^{\\dagger}$"),
    ("bonemarrow", "Bonemarrow"),
    ("paul15", "Paul15"),
    ("dentategyrus", "Dentate Gyrus"),
    ("endoderm_diff", "Endoderm$^{\\dagger}$"),
    ("gastrulation_e75", "Gastr.\\ E7.5"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    OUT = Path(args.out)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    a = pd.read_csv(WS / "outputs" / "trajectory" / "cell_traj_all_datasets.csv")
    b = pd.read_csv(WS / "outputs" / "trajectory" / "cell_traj_baselines_all.csv")
    m = a.merge(b[["dataset", "PCA_1", "DPT", "Slingshot"]], on="dataset")
    m = m.set_index("dataset")

    missing = [k for k, _ in NINE if k not in m.index]
    if missing:
        raise SystemExit(f"missing datasets in the CSVs: {missing}")

    # Sanity: every method must have been scored on the same cells as PRISM.
    for k, _ in NINE:
        n_p = int(a.loc[a.dataset == k, "n_cells"].iloc[0])
        n_b = int(b.loc[b.dataset == k, "n_cells"].iloc[0])
        if n_p != n_b:
            raise SystemExit(f"{k}: PRISM scored {n_p} cells, baselines {n_b} -- not comparable")

    METHODS = [("prism_mean", "PRISM-GEP (JS diffusion-map)"),
               ("Slingshot", "Slingshot"),
               ("DPT", "DPT / PAGA-DPT"),
               ("PCA_1", "PCA-1 \\emph{(trivial floor)}")]

    def fmt(v, best):
        s = f"{v:.3f}".lstrip("0")
        return f"\\best{{{s}}}" if best else s

    keys = [k for k, _ in NINE]
    cols = [c for c, _ in METHODS]
    # Mean rank over the four methods, computed per dataset then averaged. It is reported
    # alongside the arithmetic mean because the two can disagree: one catastrophic dataset
    # drags a mean down while barely moving a rank, and Gastrulation E7.5 is exactly that
    # case for the gene-side analysis. Showing both lets the reader see which is happening.
    ranks = m.loc[keys, cols].rank(axis=1, ascending=False)
    means = {c: m.loc[keys, c].mean() for c in cols}
    meds = {c: m.loc[keys, c].median() for c in cols}
    mranks = {c: ranks[c].mean() for c in cols}
    best_mean, best_med, best_rank = max(means.values()), max(meds.values()), min(mranks.values())

    rows = []
    for col, name in METHODS:
        cells = []
        for k, _ in NINE:
            v = m.loc[k, col]
            top = max(m.loc[k, c] for c in cols)
            cells.append(fmt(v, abs(v - top) < 1e-12))
        cells.append(fmt(means[col], abs(means[col] - best_mean) < 5e-4))
        cells.append(fmt(meds[col], abs(meds[col] - best_med) < 5e-4))
        mr = f"{mranks[col]:.2f}"
        cells.append(f"\\best{{{mr}}}" if abs(mranks[col] - best_rank) < 1e-9 else mr)
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")

    hdr = ("\\textbf{Method} & " + " & ".join(n for _, n in NINE)
           + " & \\textbf{Mean} & \\textbf{Med.} & \\textbf{Rank} \\\\")
    frag = "\n".join([
        "% AUTO-GENERATED by scripts/build_traj_cell_9ds.py -- do not hand-edit.",
        "\\begin{tabular}{l" + "c" * len(NINE) + "|ccc}",
        "\\toprule", hdr, "\\midrule",
        rows[0], "\\midrule", *rows[1:],
        "\\bottomrule", "\\end{tabular}", "",
    ])
    OUT.write_text(frag, encoding="utf-8")
    print(f"wrote {OUT}")

    sub = m.loc[[k for k, _ in NINE]]
    ranks = sub[["prism_mean", "Slingshot", "DPT", "PCA_1"]].rank(axis=1, ascending=False)
    print(f"PRISM mean rho {sub.prism_mean.mean():.3f} | mean rank {ranks.prism_mean.mean():.2f}/4"
          f" | firsts {(ranks.prism_mean == 1).sum()} | lasts {(ranks.prism_mean == 4).sum()}")
    print("2-rank (non-discriminating) rows:",
          [k for k, _ in NINE if int(sub.loc[k, "n_ranks"]) == 2])


if __name__ == "__main__":
    main()
