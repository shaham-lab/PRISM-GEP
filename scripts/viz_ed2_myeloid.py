"""FAITHFUL reproduction of GeneTrajectory Extended Data Fig 2 (a, b, c) for the myeloid dataset, using
GeneTrajectory's own trajectory extraction on PRISM's gene embedding.

GT ED Fig 2 has THREE panels; on its own panel a is hard to read, so we reproduce all three and tie them to
the SAME marker gene set + the SAME 4 myeloid cell states so they read consistently:
  a) UMAP FeaturePlot grid (grey->purple), ONE ROW PER CELL STATE (GT ED2a idiom), of canonical /
     state-specific markers chosen so each marker lights ONE state crisply -- shows WHERE each of the 4 myeloid
     states sits on the cell embedding. Markers are picked state-blind (highest per-state expression contrast
     among well-expressed genes), preferring GT/canonical markers where they still mark a distinct state in this
     PBMC-derived set (FCGR3A, CDKN1C, CD1C, CLEC10A, S100A12, ...) and falling back to the top data-driven
     state marker otherwise. This is a DISPLAY set for panel a only; the trajectory-terminus genes remain the
     subject of panels b/c. (GT's own ED2a markers CCR2/C1QA/C1QB/CD2/CD72/CCR5/CLEC5A are near-silent in this
     PBMC myeloid set, so copying them verbatim would be muddy AND misleading; the substitutions are the honest
     choice and are noted in the figure caption.)
  b) gene x cell HEATMAP: rows = the marker genes (grouped by trajectory), columns = cells grouped by cell type,
     color = z-scored log1p expression (viridis, clipped to [-2, 2]); a colored cell-type bar across the top +
     an "Identity" legend -- shows the marker->cell-type block structure at single-cell resolution.
  c) DOT PLOT (Seurat DotPlot idiom): rows = the 4 cell types, columns = the marker genes; dot SIZE = percent of
     cells expressing, dot COLOR = scaled (per-gene z) mean expression (blue) -- a compact summary of b.

Cell states: the human_myeloid.h5ad Seurat `cluster` column (res 0.3) has 4 clusters (sizes 1625/1243/325/71);
we map cluster id -> cell type by CANONICAL MARKER MEANS, recomputed locally (not by trusting a stored label):
  cluster 0 (1625) highest CD14/S100A12/VCAN            -> CD14+ monocytes
  cluster 1 (1243) highest HLA-DR / 2nd CD14            -> Intermediate monocytes
  cluster 2 (325)  by-far highest FCGR3A(CD16)/CDKN1C   -> CD16+ monocytes
  cluster 3 (71)   highest CLEC10A/CD1C/FCER1A          -> Myeloid type-2 dendritic cells

Shared gene set: the trajectory terminus MARKER genes from panel a (selected honestly, gene-set-blind: additive
log1p enrichment of a gene's top-decile cells over the rest, within a prevalence band, among each trajectory's
late-pseudo-order genes), plus a small set of canonical GT ED2 markers (all present in the 5000-gene vocab) added
to b/c so the block structure is legible. All panels use the SAME filtered expression matrix + UMAP-sidecar cells.

Usage: python scripts/viz_ed2_myeloid.py gt_myeloid [n_per_traj]
Output: figures/ed2_trajectory_markers_gt_myeloid.pdf
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

WS = Path(__file__).resolve().parent.parent
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
# release layout: inputs from results/, figures to paper/figures/
# (scripts/paths.py); the authoring tree uses different roots.
from paths import figures_dir  # noqa: E402
sys.path.insert(0, str(WS / "scripts"))
from gt_faithful_common import extract_prism_trajectories  # noqa: E402
from viz_cell_embed_common import load_cell_embedding  # noqa: E402
from figsafe import save_and_deploy  # noqa: E402

# ---------------------------------------------------------------------------
# Print legibility (2026-07-20)
# ---------------------------------------------------------------------------
# supplementary.tex places this at \textwidth = 524.5pt. It used to be drawn
# 15.5in (1116pt) wide, so everything was shrunk to 0.42x and the smallest
# text -- the per-UMAP colourbar ticks and the UMAP1/UMAP2 axis labels --
# landed at 2.1pt on paper. Drawing near the placed width instead lifts the
# shrink to ~0.85x. The figure also grows taller: its supplement page had
# ~190pt of unused column, and panel b's 27 gene rows need roughly 7.6pt of
# source pitch each before a 7pt label starts touching its neighbour.
SRC_W_IN = 8.75              # -> ~630pt after matplotlib's tight bbox + crop
SRC_H_IN = 9.20

# GT / Seurat FeaturePlot idiom: light grey (low) -> purple (high).
GT_CMAP = LinearSegmentedColormap.from_list("gt_purple", ["#e6e6e6", "#cfc6e6", "#8a5fc9", "#3d1178"])
# Seurat DotPlot idiom: light grey (low scaled mean) -> blue (high).
DOT_CMAP = LinearSegmentedColormap.from_list("gt_blue", ["#e8e8e8", "#b9c3e6", "#4a63c8", "#12237a"])

# 4 myeloid cell states (order used across the top bar / dot-plot rows) + their identity-bar colors (GT-like).
CELLTYPE_ORDER = ["CD14+ monocytes", "Intermediate monocytes", "CD16+ monocytes", "Myeloid type-2 dendritic cells"]
CELLTYPE_COLORS = {
    "CD14+ monocytes": "#2ca089",                 # teal/green
    "Intermediate monocytes": "#e8d43a",          # yellow
    "CD16+ monocytes": "#6fb8e6",                 # light blue
    "Myeloid type-2 dendritic cells": "#e8963a",  # orange
}
# Canonical GT ED2 markers (all present in the 5000-gene vocab) added to b/c for a legible block structure.
CANON_MARKERS = ["CD14", "S100A12", "VCAN", "SELL", "HLA-DRB1", "LGALS2",
                 "FCGR3A", "CDKN1C", "IFITM2", "CD1C", "CLEC10A", "FCER1A"]

# --- Panel a: state-specific FeaturePlot markers (matches GeneTrajectory ED Fig 2a's "one row = one state,
# each marker lights ONE state crisply" quality). GT ED2a uses well-studied markers (CCR2, FCGR3A, CD1C,
# CLEC10A, ...); many of those mark tissue macrophages and are near-silent in THIS PBMC-derived myeloid set
# (C1QA/C1QB/CD2/CD72/CCR5/CLEC5A all <0.4 mean, <4% expressing here), so blindly copying GT's 15 would give
# muddy near-empty plots. Instead, per state we take the markers with the highest state-specificity that are
# actually well-expressed here, PREFERRING GT/canonical markers where they still light a distinct state in this
# data (FCGR3A, CDKN1C, CD1C, CLEC10A, S100A12, ...) and falling back to the top data-driven state-specific
# gene otherwise. Selection is gene-set-blind (contrast = mean_log1p in the state - best other state). The
# Intermediate-monocyte row is honestly the softest: no gene is crisply specific to the transitional
# intermediate state in this data (max contrast +0.29) — see the panel-a note. Rows are ordered so each lights
# a DISTINCT state, exactly like GT.  These are DISPLAY markers for panel a only; panels b/c keep PRISM's
# trajectory-terminus gene set unchanged.
PANEL_A_ROWS = [
    ("CD14+ monocytes",                  ["S100A8", "S100A12", "CYP1B1", "SLC2A3"]),
    ("Intermediate monocytes",           ["TMEM176B", "KLF10", "STAB1", "TMEM176A"]),
    ("CD16+ monocytes",                  ["FCGR3A", "CDKN1C", "SMIM25", "RHOC"]),
    ("Myeloid type-2 dendritic cells",   ["FCER1A", "CLEC10A", "CD1C", "ENHO"]),
]
# Which of the above are GT/canonical markers (for the honest panel-a note), vs data-driven fills.
PANEL_A_CANON = {"S100A12", "FCGR3A", "CDKN1C", "CD1C", "CLEC10A", "FCER1A", "S100A8"}


def _int_arg(argv, i, default):
    """Parse a positional int arg; tolerate placeholders like '.' passed by the runner."""
    if len(argv) > i:
        try:
            return int(argv[i])
        except (ValueError, TypeError):
            return default
    return default


def marker_rank_late(traj, tnum, Xlog, col_idx, k, late_frac=0.50, min_prev=0.10, max_prev=0.80):
    """Trajectory-SPECIFIC marker genes from the terminus region of Trajectory-<tnum> (see module docstring)."""
    pcol = f"Pseudoorder-{tnum}"
    if pcol not in traj.columns:
        return []
    sub = traj[(traj["selected"] == f"Trajectory-{tnum}") & (traj[pcol] > 0)]
    if sub.empty:
        return []
    thr = sub[pcol].quantile(1.0 - late_frac)
    late = sub[sub[pcol] >= thr]
    n = Xlog.shape[0]
    top = max(1, int(0.10 * n))
    scored = []
    for g in late.index:
        gl = str(g).lower()
        if gl not in col_idx:
            continue
        v = Xlog[:, col_idx[gl]]
        prev = float((v > 0).mean())
        if prev < min_prev or prev > max_prev or v.sum() <= 0:
            continue
        idx = np.argpartition(v, -top)[-top:]
        mask = np.ones(n, bool); mask[idx] = False
        enrich = float(v[idx].mean() - v[mask].mean())
        if enrich > 0:
            scored.append((gl, enrich))
    scored.sort(key=lambda t: -t[1])
    seen, out = set(), []
    for gl, _ in scored:
        if gl in seen:
            continue
        seen.add(gl); out.append(gl)
        if len(out) >= k:
            break
    return out


def celltype_labels(cell_ids):
    """Map the h5ad Seurat `cluster` (res 0.3, 4 clusters) to cell-type names by CANONICAL MARKER MEANS,
    recomputed locally. Returns a pandas Series (cell_id -> cell type) aligned to `cell_ids`."""
    import anndata as ad, scanpy as sc
    h = WS / "data" / "gt_myeloid" / "human_myeloid.h5ad"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = ad.read_h5ad(h)
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    a.obs_names = a.obs_names.astype(str)
    panels = {
        "CD14+ monocytes": ["CD14", "S100A12", "VCAN", "S100A8", "S100A9"],
        "Intermediate monocytes": ["HLA-DRA", "HLA-DRB1", "CD74", "LGALS2"],
        "CD16+ monocytes": ["FCGR3A", "CDKN1C", "MS4A7", "RHOC", "LYPD2"],
        "Myeloid type-2 dendritic cells": ["CLEC10A", "CD1C", "FCER1A", "HLA-DQA1", "CLEC9A"],
    }
    genes = set(a.var_names)
    clusters = list(pd.unique(a.obs["cluster"]))
    # score each cluster against each cell-type marker panel (mean panel expression), then z per cell type so the
    # tie-prone CD14/Intermediate pair separates on its DISTINGUISHING panel rather than raw magnitude.
    score = pd.DataFrame(index=clusters, columns=list(panels), dtype=float)
    for label, gs in panels.items():
        present = [g for g in gs if g in genes]
        col = np.asarray(a[:, present].X.mean(1)).ravel()
        for cl in clusters:
            score.loc[cl, label] = float(col[(a.obs["cluster"] == cl).values].mean())
    z = (score - score.mean(0)) / (score.std(0) + 1e-9)
    # greedy assignment: each cell type takes its best remaining cluster (descending z) -> distinct clusters
    mapping, taken, assigned = {}, set(), set()
    for _, lab, cl in sorted([(z.loc[cl, lab], lab, cl) for lab in panels for cl in clusters], reverse=True):
        if lab in assigned or cl in taken:
            continue
        mapping[cl] = lab; taken.add(cl); assigned.add(lab)
    ser = a.obs["cluster"].map(mapping)
    ser.index = a.obs_names
    return ser.reindex([str(c) for c in cell_ids])


def _mean_pct_scaled(genes, Xlog, col_idx, ct):
    """Per-(cell type, gene) matrices used by both b/c: mean log1p over expressing cells, percent expressing,
    and the per-gene z-score of the mean across the 4 cell types (the Seurat DotPlot "scaled mean")."""
    ct = ct.dropna()
    T = len(CELLTYPE_ORDER); G = len(genes)
    masks = {t: (ct.values == t) for t in CELLTYPE_ORDER}
    pct = np.zeros((T, G)); mean_expr = np.zeros((T, G))
    for gi, g in enumerate(genes):
        v = Xlog[:, col_idx[g.lower()]]
        for ti, t in enumerate(CELLTYPE_ORDER):
            vv = v[masks[t]]
            pct[ti, gi] = 100.0 * (vv > 0).mean() if vv.size else 0.0
            mean_expr[ti, gi] = vv[vv > 0].mean() if (vv > 0).any() else 0.0
    scaled = (mean_expr - mean_expr.mean(0, keepdims=True)) / (mean_expr.std(0, keepdims=True) + 1e-9)
    return pct, mean_expr, scaled


def block_order(genes, Xlog, col_idx, ct):
    """Group marker genes into a block-diagonal column order for the dot plot / heatmap.

    Each gene is assigned, data-blind, to the cell type in which its scaled (per-gene z) mean expression is
    highest (argmax over the 4 states) -- exactly the quantity the dot COLOR encodes -- then columns are
    emitted cell-type by cell-type in CELLTYPE_ORDER, and within a block sorted by descending peak scaled
    expression. So the darkest/biggest dots march down the diagonal. Returns
    (ordered_genes, block_bounds) where block_bounds = [(celltype, start_idx, end_idx), ...]."""
    _, _, scaled = _mean_pct_scaled(genes, Xlog, col_idx, ct)
    assign = np.argmax(scaled, axis=0)                       # gene -> cell-type index
    ordered, bounds, cur = [], [], 0
    for ti, t in enumerate(CELLTYPE_ORDER):
        members = [(genes[gi], scaled[ti, gi]) for gi in range(len(genes)) if assign[gi] == ti]
        members.sort(key=lambda x: -x[1])
        start = cur
        ordered.extend(g for g, _ in members); cur += len(members)
        bounds.append((t, start, cur))
    return ordered, bounds


def _short_state(t):
    """Compact 2-line state label for the row headers (mirrors the identity bar naming)."""
    return {"CD14+ monocytes": "CD14+\nmono", "Intermediate monocytes": "Interm.\nmono",
            "CD16+ monocytes": "CD16+\nmono",
            "Myeloid type-2 dendritic cells": "Myeloid\nDC"}.get(t, t)


def panel_a(fig, gs_a, rows, Xlog, col_idx, U):
    """UMAP FeaturePlot grid in the GeneTrajectory ED Fig 2a idiom: one ROW per cell state, each cell a
    grey->purple FeaturePlot of a marker that lights that state crisply. `rows` = [(state_name, [genes...]), ...].

    Clean-display choices matched to GT: robust vmax (99th pctile of expressing cells) so a few outliers don't
    wash the scale; low-expression cells drawn FIRST so high-expression cells pop on top; italic gene titles;
    per-row state label on the left; a single small UMAP1/UMAP2 axis arrow in the bottom-left panel."""
    nrow = len(rows)
    ncol = max((len(g) for _, g in rows), default=1)
    # reserve a leftmost column for the per-row cell-state label (kept clear of the plots + colorbars)
    inner = gs_a.subgridspec(nrow, ncol + 1, hspace=0.34, wspace=0.30,
                             width_ratios=[0.30] + [1.0] * ncol)
    # global UMAP extent (shared across every panel so the embedding is drawn identically everywhere)
    xpad = 0.03 * (U[:, 0].max() - U[:, 0].min()); ypad = 0.03 * (U[:, 1].max() - U[:, 1].min())
    xlim = (U[:, 0].min() - xpad, U[:, 0].max() + xpad)
    ylim = (U[:, 1].min() - ypad, U[:, 1].max() + ypad)
    for ri, (state, genes) in enumerate(rows):
        axlab = fig.add_subplot(inner[ri, 0]); axlab.axis("off")
        axlab.text(0.62, 0.5, _short_state(state), transform=axlab.transAxes, rotation=0,
                   ha="right", va="center", fontsize=9.5, fontweight="bold", linespacing=1.05,
                   color=CELLTYPE_COLORS.get(state, "#333333"))
        for ci in range(ncol):
            ax = fig.add_subplot(inner[ri, ci + 1])
            if ci < len(genes):
                g = genes[ci]
                v = Xlog[:, col_idx[g.lower()]]
                pos = v[v > 0]
                vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
                order = np.argsort(v)  # low (grey) drawn first, high (purple) drawn last -> pops on top
                sc = ax.scatter(U[order, 0], U[order, 1], c=v[order], cmap=GT_CMAP, s=4.0,
                                vmin=0, vmax=max(vmax, 1e-6), linewidths=0, rasterized=True)
                ax.set_title(g, fontsize=9, style="italic", pad=2.5)
                cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.015)
                cb.ax.tick_params(labelsize=7, length=1.5); cb.outline.set_visible(False)
                cb.set_ticks([0, max(vmax, 1e-6)]); cb.ax.set_yticklabels(["0", f"{vmax:.1f}"])
                ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            else:
                ax.axis("off")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#bbbbbb"); s.set_linewidth(0.5)
            if ri == nrow - 1 and ci == 0 and ci < len(genes):
                ax.annotate("", xy=(0.19, 0.045), xytext=(0.04, 0.045), xycoords="axes fraction",
                            arrowprops=dict(arrowstyle="-|>", color="#444", lw=0.9))
                ax.annotate("", xy=(0.04, 0.20), xytext=(0.04, 0.045), xycoords="axes fraction",
                            arrowprops=dict(arrowstyle="-|>", color="#444", lw=0.9))
                ax.text(0.205, 0.045, "UMAP1", transform=ax.transAxes, fontsize=7.0, va="center", color="#444")
                ax.text(0.045, 0.21, "UMAP2", transform=ax.transAxes, fontsize=7.0, va="bottom",
                        rotation=90, color="#444")


def panel_b(fig, gs_b, genes, gene_block, Xlog, col_idx, ct):
    """gene x cell heatmap: rows = markers (grouped into the SAME cell-type blocks as panel c), cols = cells
    grouped by cell type. `gene_block[i]` = the cell-type name gene i is assigned to (for row-block separators)."""
    ct = ct.dropna()
    idx_by_type = {t: np.where(ct.values == t)[0] for t in CELLTYPE_ORDER}
    cell_order, bounds, cur = [], [], 0
    for t in CELLTYPE_ORDER:
        ii = idx_by_type[t]
        cell_order.extend(ii.tolist())
        cur += len(ii); bounds.append((t, cur))
    cell_order = np.array(cell_order)
    M = np.vstack([Xlog[:, col_idx[g.lower()]] for g in genes])[:, cell_order]   # genes x cells
    Z = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
    Z = np.clip(Z, -2, 2)

    inner = gs_b.subgridspec(2, 1, height_ratios=[0.05, 1.0], hspace=0.015)
    axbar = fig.add_subplot(inner[0]); axhm = fig.add_subplot(inner[1])
    prev = 0
    # The bar is only ~420 source pt wide and the CD16+ / Myeloid-DC blocks hold
    # 325 and 71 cells, so their centred labels used to overlap. Draw a label only
    # when its block is wide enough to hold it at a readable size; the identity
    # legend at the top of the figure names all four states regardless.
    bar_pt = 0.72 * SRC_W_IN * 72.0
    for t, end in bounds:
        axbar.axvspan(prev, end, color=CELLTYPE_COLORS[t], lw=0)
        lab = t.replace(" monocytes", "").replace("Myeloid type-2 dendritic cells", "Myeloid DC")
        block_pt = (end - prev) / max(1, Z.shape[1]) * bar_pt
        if block_pt >= 0.60 * len(lab) * 7.5:
            axbar.text((prev + end) / 2.0, 0.5, lab, ha="center", va="center",
                       fontsize=7.5, fontweight="bold", color="#333333")
        prev = end
    axbar.set_xlim(0, Z.shape[1]); axbar.set_ylim(0, 1)
    axbar.set_xticks([]); axbar.set_yticks([])
    for s in axbar.spines.values():
        s.set_visible(False)
    im = axhm.imshow(Z, aspect="auto", cmap="viridis", vmin=-2, vmax=2, interpolation="nearest")
    axhm.set_yticks(range(len(genes)))
    axhm.set_yticklabels(genes, fontsize=7, style="italic")
    axhm.set_xticks([])
    axhm.set_xlabel("cells (grouped by cell type)", fontsize=8)
    for i in range(1, len(genes)):
        if gene_block[i] != gene_block[i - 1]:
            axhm.axhline(i - 0.5, color="w", lw=1.4)
    for s in axhm.spines.values():
        s.set_edgecolor("#999999"); s.set_linewidth(0.5)
    for _, end in bounds[:-1]:
        axhm.axvline(end - 0.5, color="w", lw=0.8)
    cb = fig.colorbar(im, ax=axhm, fraction=0.013, pad=0.008, aspect=16)
    cb.set_label("Expression\n(z-scored)", fontsize=8); cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)


DOT_SMIN, DOT_SMAX = 7.0, 185.0   # marker-area range (pts^2): 0% -> DOT_SMIN, 100% -> DOT_SMAX
# (rescaled 2026-07-20 with the figure: the dot-plot columns are now ~19 source pt
#  apart, so the old 340pt^2 maximum drew touching dots. The size legend uses the
#  same mapping, so the percent-expressed encoding is unchanged.)


def _dot_size(pct):
    return DOT_SMIN + (np.clip(pct, 0, 100) / 100.0) * (DOT_SMAX - DOT_SMIN)


def panel_c(fig, gs_c, genes, bounds, Xlog, col_idx, ct):
    """Seurat DotPlot, block-diagonal: rows = the 4 cell types (fixed CELLTYPE_ORDER), cols = marker genes
    grouped into the cell-type block where each is most enriched. Dot SIZE = percent of cells expressing,
    dot COLOR = scaled (per-gene z) mean expression. Big/dark dots run down the diagonal by construction."""
    ct = ct.dropna()
    pct, _, scaled = _mean_pct_scaled(genes, Xlog, col_idx, ct)
    scaled = np.clip(scaled, -1.5, 1.5)
    G = len(genes); T = len(CELLTYPE_ORDER)
    norm = Normalize(vmin=-1.5, vmax=1.5)

    ax = fig.add_subplot(gs_c)
    # faint block backgrounds so each cell type's column block is visually contained
    for ti, (t, s, e) in enumerate(bounds):
        ax.axvspan(s - 0.5, e - 0.5, color=CELLTYPE_COLORS[t], alpha=0.07, lw=0, zorder=0)
        ax.text((s + e - 1) / 2.0, T - 0.30, t.replace(" monocytes", "").replace(
            "Myeloid type-2 dendritic cells", "Myeloid DC"), ha="center", va="bottom",
            fontsize=6.8, fontweight="bold", color=CELLTYPE_COLORS[t])
    for ti in range(T):
        for gi in range(G):
            ax.scatter(gi, T - 1 - ti, s=_dot_size(pct[ti, gi]), c=[DOT_CMAP(norm(scaled[ti, gi]))],
                       edgecolors="#9a9a9a", linewidths=0.35, zorder=3)
    ax.set_xlim(-0.7, G - 0.3); ax.set_ylim(-0.7, T - 0.05)
    # Panel c ran about 1pt heavier than panel b above it 2026-07-20 (Yanir), which
    # read as the dot plot shouting over the heatmap. Trimmed roughly 1pt across this
    # panel only. The smallest text here is now 6.5pt at source, still above the
    # 5.5pt on-page floor once the figure is placed.
    ax.set_xticks(range(G)); ax.set_xticklabels(genes, rotation=90, fontsize=6.8, style="italic")
    # Compact two-line row labels, matching panel a's row headers. The full names
    # are spelled out once in the identity legend at the top of the figure; spelled
    # out here they run 133pt into the left margin and collide with the panel letter.
    ax.set_yticks(range(T))
    ax.set_yticklabels([_short_state(t) for t in CELLTYPE_ORDER[::-1]], fontsize=7.5,
                       linespacing=1.05)
    for lab, t in zip(ax.get_yticklabels(), CELLTYPE_ORDER[::-1]):
        lab.set_color(CELLTYPE_COLORS[t])
        lab.set_fontweight("bold")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_edgecolor("#cccccc"); s.set_linewidth(0.5)
    ax.set_axisbelow(True); ax.grid(True, color="#eeeeee", lw=0.5)
    # dashed separators between cell-type blocks
    for _, _, e in bounds[:-1]:
        ax.axvline(e - 0.5, color="#c4c4c4", lw=0.9, ls=(0, (4, 3)), zorder=1)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=DOT_CMAP); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.012, pad=0.010, aspect=11, anchor=(0.0, 1.0), panchor=(0.0, 1.0))
    cb.set_label("Scaled mean\nexpression", fontsize=7.5, labelpad=6)
    cb.ax.yaxis.set_label_position("left"); cb.ax.tick_params(labelsize=6.5)
    cb.set_ticks([-1.5, 0, 1.5]); cb.outline.set_visible(False)
    # size legend placed well to the RIGHT of the colorbar (its label) so nothing overlaps
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=np.sqrt(_dot_size(p)),
                      markerfacecolor="#9aa0aa", markeredgecolor="#9a9a9a", markeredgewidth=0.35, label=f"{p}")
               for p in (0, 25, 50, 75, 100)]
    ax.legend(handles=handles, title="Percent\nexpressed", title_fontsize=7.5, fontsize=7.0,
              loc="upper left", bbox_to_anchor=(1.045, 1.0), frameon=False, labelspacing=1.25, borderpad=0.5,
              handletextpad=0.7)


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "gt_myeloid"
    k = _int_arg(sys.argv, 2, 5)
    traj = extract_prism_trajectories(ds)
    coords, E, source = load_cell_embedding(ds)
    U = coords[["x", "y"]].values
    cols = list(E.columns)
    col_idx = {c.lower(): i for i, c in enumerate(cols)}
    orig_case = {c.lower(): c for c in cols}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xlog = np.log1p(E.values.astype(float))

    n_traj = sum(1 for t in traj["selected"].unique() if str(t).startswith("Trajectory"))
    rows = []
    for tnum in range(1, n_traj + 1):
        genes_low = marker_rank_late(traj, tnum, Xlog, col_idx, k)
        rows.append((tnum, [orig_case[g] for g in genes_low]))

    # shared gene set for b/c: terminus markers grouped by trajectory + canonical GT markers (dedup, in-vocab)
    gene_rows, gene_traj, seen = [], [], set()
    for tnum, genes in rows:
        for g in genes:
            if g.lower() not in seen:
                seen.add(g.lower()); gene_rows.append(g); gene_traj.append(f"T{tnum}")
    for g in CANON_MARKERS:
        if g.lower() in col_idx and g.lower() not in seen:
            seen.add(g.lower()); gene_rows.append(g); gene_traj.append("canon")

    ct = celltype_labels(coords.index)
    got = ct.dropna()
    print("cell-type counts:", {t: int((got.values == t).sum()) for t in CELLTYPE_ORDER})

    # Block-diagonal column/row order shared by b and c: assign each marker to the cell type where its scaled
    # mean expression peaks, then emit blocks in CELLTYPE_ORDER. gene_block[i] = that cell type (for separators).
    gene_order, bounds = block_order(gene_rows, Xlog, col_idx, ct)
    blk_of = {}
    for t, s, e in bounds:
        for g in gene_order[s:e]:
            blk_of[g] = t
    gene_block = [blk_of[g] for g in gene_order]
    print("block column order:", gene_order)
    print("block sizes:", {t: e - s for t, s, e in bounds})

    # Panel a: state-organized DISPLAY markers (one row per cell state). Resolve gene case from the vocab and
    # drop any not present, so a marker missing on some other dataset degrades gracefully rather than erroring.
    panel_a_rows = []
    for state, genes in PANEL_A_ROWS:
        present = [orig_case[g.lower()] for g in genes if g.lower() in col_idx]
        panel_a_rows.append((state, present))
    print("panel-a display markers by state:", {s: g for s, g in panel_a_rows})

    fig = plt.figure(figsize=(SRC_W_IN, SRC_H_IN))
    # b gets the biggest share: 27 gene rows set the minimum readable pitch.
    gs = GridSpec(3, 1, height_ratios=[0.86, 1.25, 0.62], hspace=0.26, figure=fig)
    panel_a(fig, gs[0], panel_a_rows, Xlog, col_idx, U)
    panel_b(fig, gs[1], gene_order, gene_block, Xlog, col_idx, ct)
    panel_c(fig, gs[2], gene_order, bounds, Xlog, col_idx, ct)

    for y, letter in zip([0.930, 0.660, 0.243], ["a", "b", "c"]):
        fig.text(0.010, y, letter, fontsize=13, fontweight="bold", va="top", ha="left")
    handles = [Line2D([0], [0], marker="s", linestyle="", markersize=8, markerfacecolor=CELLTYPE_COLORS[t],
                      markeredgecolor="none", label=t) for t in CELLTYPE_ORDER]
    # Identity legend as a single horizontal row in the title band (clear of every panel + colorbar)
    fig.legend(handles=handles, title="Identity (cell type)", loc="upper center", bbox_to_anchor=(0.5, 0.998),
               ncol=4, columnspacing=1.2, fontsize=8, title_fontsize=8.5, frameon=False)
    # in-figure descriptive title removed 2026-07-10 -> moved to LaTeX caption
    # (suptitle + the multi-paragraph "What this figure shows" caption band both
    #  described the whole figure and belong in the \caption, not the image).

    fig.subplots_adjust(left=0.075, right=0.800, top=0.945, bottom=0.105)
    out = figures_dir() / f"ed2_trajectory_markers_{ds}.pdf"
    save_and_deploy(fig, out, bbox_inches="tight", dpi=140)
    print(f"WROTE {out}")
    for tnum, genes in rows:
        print(f"  Trajectory-{tnum} terminus markers:", genes)
    print("  shared b/c gene set:", gene_rows)


if __name__ == "__main__":
    main()
