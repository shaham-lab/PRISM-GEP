"""Doubly-ordered gene x cell "cascade" heatmap in the GeneTrajectory idiom -- the
doubly-ordered gene x cell "cascade" heatmap. Cells are ordered along a pseudotime axis and marker genes are
ordered by their inferred gene-pseudo-order; a clean diagonal wave = the ordering recovers the developmental
program.

The NON-CIRCULAR design: cells are ordered by **Slingshot** -- a cell pseudotime PRISM never saw -- while
genes are ordered by each method. A diagonal therefore validates a method's gene order against an INDEPENDENT
cell axis. Panels side by side compare PRISM Step-ii | GeneTrajectory (best-matching trajectory) |
GeneTrajectory (leading-EV) | DPT-weighted-mean (root-supervised) | canonical ground truth.

Parameterized over any cascade-ready dataset (both pseudotimes.csv + gene orders.csv on disk):
  python scripts/viz_cascade_heatmap.py --dataset pancreas
  python scripts/viz_cascade_heatmap.py --dataset pancreas --panels core   # 3-panel PRISM/GT-EV/Canonical

Input : data/<ds>/filtered_<ds>_cells_x_genes.csv               (cells x genes, raw)
        outputs/trajectory/<ds>/<ds>_pseudotimes.csv            (Slingshot, lineage_rank, lineage)
        outputs/trajectory/<ds>/gene_trajectory_<ds>_orders.csv (gene orders + canonical_rank)
        outputs/trajectory/<ds>/gt_full_trajectories.csv        (for the GT best-trajectory panel)
Output: figures/<ds>_cascade_heatmap.pdf
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

WS = Path(__file__).resolve().parent.parent

import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
# release layout: paper/figures/, created on demand (scripts/paths.py)
from paths import figures_dir  # noqa: E402
CELL_AXIS = "Slingshot"      # independent of PRISM

# panel sets: (column, label). "GT_best_traj" is computed below from gt_full_trajectories.csv.
PANELS_EXTENDED = [
    ("canonical_rank", "Canonical\n(ground truth)"),
    ("PRISM_K5_StepII", "PRISM Step-ii\n(gene-topic diffusion)"),
    ("GT_best_traj", "GeneTrajectory\n(best trajectory)"),
    ("GeneTrajectory_EV", "GeneTrajectory\n(leading eigenvector)"),
    ("DPT_weighted_mean", "DPT-weighted mean\n(root-supervised)"),
]
PANELS_CORE = [
    ("PRISM_K5_StepII", "PRISM Step-ii\n(gene-topic diffusion)"),
    ("GeneTrajectory_EV", "GeneTrajectory\n(leading eigenvector)"),
    ("canonical_rank", "Canonical\n(ground truth)"),
]

TITLES = {
    "pancreas": "Pancreas endocrinogenesis",
    "gastrulation": "Mouse gastrulation (E6.5–E8.5)",
    "gastrulation_erythroid": "Gastrulation erythroid lineage",
}


def gt_best_traj_column(ds, orders):
    """Per-gene ordering from GeneTrajectory's BEST-matching extracted trajectory
    (mirrors scripts/score_gt_3variants.best_trajectory). Genes not in that trajectory -> NaN."""
    gf = WS / "outputs" / "trajectory" / ds / "gt_full_trajectories.csv"
    out = pd.Series(np.nan, index=orders["gene"].astype(str))
    if not gf.exists():
        return out.values
    gt = pd.read_csv(gf)
    lut = {str(g).upper(): g for g in gt["gene"]}
    mk = orders[["gene", "canonical_rank"]].copy()
    mk["gt_gene"] = mk["gene"].map(lambda g: lut.get(str(g).upper()))
    mg = mk.dropna(subset=["gt_gene"]).merge(gt, left_on="gt_gene", right_on="gene", suffixes=("", "_gt"))
    pcols = [c for c in gt.columns if c.lower().startswith("pseudoorder")]
    best_rho, best_lab = -1.0, None
    for pc in pcols:
        lab = f"Trajectory-{pc.split('-')[-1]}"
        sub = mg[mg["selected"] == lab]
        if len(sub) >= 3:
            a = pd.to_numeric(sub[pc], errors="coerce"); b = pd.to_numeric(sub["canonical_rank"], errors="coerce")
            m = a.notna() & b.notna()
            if m.sum() >= 3 and a[m].nunique() >= 2:
                r = abs(spearmanr(a[m], b[m])[0])
                if r > best_rho:
                    best_rho, best_lab, best_pc = r, lab, pc
    if best_lab is None:
        return out.values
    sub = mg[mg["selected"] == best_lab]
    for _, row in sub.iterrows():
        out[str(row["gene"])] = row[best_pc]
    return out.values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pancreas")
    ap.add_argument("--panels", choices=["core", "extended"], default="extended")
    args = ap.parse_args()
    ds = args.dataset
    ORDERINGS = PANELS_CORE if args.panels == "core" else PANELS_EXTENDED

    EXPR = WS / "data" / ds / f"filtered_{ds}_cells_x_genes.csv"
    PT = WS / "outputs" / "trajectory" / ds / f"{ds}_pseudotimes.csv"
    ORD = WS / "outputs" / "trajectory" / ds / f"gene_trajectory_{ds}_orders.csv"
    OUT = figures_dir() / f"{ds}_cascade_heatmap.pdf"
    title = TITLES.get(ds, ds.replace("_", " "))

    d = pd.read_csv(ORD)
    if "GT_best_traj" in [c for c, _ in ORDERINGS]:
        d["GT_best_traj"] = gt_best_traj_column(ds, d)
    genes = [str(g) for g in d["gene"].tolist()]
    can = d["canonical_rank"].values.astype(float)

    pt = pd.read_csv(PT).dropna(subset=[CELL_AXIS]).copy()
    pt["cell_id"] = pt["cell_id"].astype(str)

    expr = pd.read_csv(EXPR, index_col=0)
    expr.index = expr.index.astype(str)
    cols = {c.lower(): c for c in expr.columns}
    gcols = [cols[g.lower()] for g in genes if g.lower() in cols]
    gnames = [g for g in genes if g.lower() in cols]

    common = [c for c in pt["cell_id"].tolist() if c in expr.index]
    pt = pt[pt["cell_id"].isin(common)].copy()
    E = expr.loc[pt["cell_id"].values, gcols]
    Z = np.log1p(E.values.astype(float))
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)

    cell_pt = pt[CELL_AXIS].values.astype(float)
    if "lineage_rank" in pt.columns and spearmanr(cell_pt, pt["lineage_rank"].values).correlation < 0:
        cell_pt = -cell_pt
    cell_order = np.argsort(cell_pt)
    Zc = Z[cell_order, :]
    lin_rank = pt["lineage_rank"].values[cell_order] if "lineage_rank" in pt.columns else None

    n_cells = Zc.shape[0]
    SMOOTH = int(max(25, min(250, n_cells // 50)))

    def smooth(v, w):
        if w <= 1:
            return v
        return np.convolve(v, np.ones(w) / w, mode="same")
    Zs = np.column_stack([smooth(Zc[:, j], SMOOTH) for j in range(Zc.shape[1])])

    dmap = d.set_index("gene")
    can_map = dict(zip(gnames, [can[genes.index(g)] for g in gnames]))
    cr_all = np.array([can_map[g] for g in gnames])

    pw = 3.55 if len(ORDERINGS) >= 5 else 4.7
    fig = plt.figure(figsize=(pw * len(ORDERINGS), 6.8))
    gs = GridSpec(2, len(ORDERINGS), height_ratios=[0.05, 1], hspace=0.05, wspace=0.20,
                  top=0.80, bottom=0.09, left=0.05, right=0.99)
    im = None

    for k, (col, label) in enumerate(ORDERINGS):
        vals_all = dmap.reindex(gnames)[col].values.astype(float)
        valid = np.isfinite(vals_all)
        axL = fig.add_subplot(gs[0, k])
        if lin_rank is not None:
            axL.imshow(lin_rank[None, :], aspect="auto", cmap="cividis")
        axL.set_xticks([]); axL.set_yticks([]); axL.set_title(label, fontsize=10)
        ax = fig.add_subplot(gs[1, k])

        if valid.sum() < 3:
            ax.text(0.5, 0.5,
                    "GeneTrajectory best\ntrajectory undefined here\n\n"
                    "no single extracted\ntrajectory holds 3+\nof the canonical markers\n"
                    "(most fall in 'Other'),\nso a single-lineage order\ncannot be read off",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9, color="0.3")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("0.7"); s.set_linestyle((0, (4, 4)))
            ax.set_xlabel(f"cells ordered by {CELL_AXIS} pseudotime (n={Zs.shape[0]})", fontsize=8)
            continue

        vals = vals_all[valid]
        vnames = [g for g, v in zip(gnames, valid) if v]
        vcr = cr_all[valid]
        vidx = np.where(valid)[0]
        r = spearmanr(vals, vcr).correlation
        if r is not None and r < 0:
            vals = -vals
        g_order = np.argsort(vals)                          # early genes first
        M = Zs[:, vidx][:, g_order].T                        # genes(ordered) x cells
        rho = abs(spearmanr(vals, vcr).correlation)

        im = ax.imshow(M, aspect="auto", cmap="magma", vmin=-1.5, vmax=1.8, interpolation="nearest")
        ax.set_yticks(range(len(vnames)))
        ax.set_yticklabels([vnames[i] for i in g_order], fontsize=7)
        ax.set_xticks([])
        ax.set_xlabel(f"cells ordered by {CELL_AXIS} pseudotime (n={M.shape[1]})", fontsize=8)
        tag = f"|rho|={rho:.2f}" + (f"  (n={len(vnames)})" if len(vnames) < len(gnames) else "")
        ax.text(0.015, 0.985, f"gene order vs canonical  {tag}",
                transform=ax.transAxes, fontsize=8.5, color="white", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.55, ec="none"))

    cax = fig.add_axes([0.05, 0.035, 0.24, 0.014])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("smoothed log1p expression (per-gene z-score)", fontsize=7.5)
    cb.ax.tick_params(labelsize=6)

    fig.savefig(OUT, bbox_inches="tight")
    print(f"WROTE {OUT}  ({Zs.shape[0]} cells x {len(gnames)} genes, {len(ORDERINGS)} panels, smooth={SMOOTH})")


if __name__ == "__main__":
    main()
