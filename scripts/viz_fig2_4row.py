"""PRISM analogue of GeneTrajectory Figure 2, on GeneTrajectory's OWN simulation model (their
simulate_{cyclic,bifurcation,cylinder,coral}, ported in gt_simulate.py). 4 rows: a cell cycle · b bifurcation ·
c linear+cell-cycle (cylinder) · d multilayered+cell-cycle (coral).

Columns: cell embedding (t-SNE a,b / UMAP c,d, as in the paper; PCA-oriented like their layout) · process
schematic with g's placed along a single lineage path · gene-expression panels (g1-g5; g6-g10 for the 2nd
process of c,d) · PRISM gene embedding colored by gene pseudo-order (per lineage). For the concurrent rows the
last column is SPLIT into two panels — Process-1 (differentiation) and Process-2 (cell cycle) — so the two
distinct gene trajectories are each shown, matching the paper. Deconvolution silhouette is measured in the
top-K diffusion components, K chosen by the eigengap (data-driven, not by matching GT).

Output: figures/fig2_simulations.pdf
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
from itertools import combinations
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

WS = Path(__file__).resolve().parent.parent
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
# release layout: inputs from results/, figures to paper/figures/
# (scripts/paths.py); the authoring tree uses different roots.
from paths import figures_dir  # noqa: E402
sys.path.insert(0, str(WS / "scripts"))
from gt_faithful_common import load_prism_gene_space  # noqa: E402

DIFF_C, CYC_C = "#1f77b4", "#d62728"
NG = 5
MAX_CELLS = 6000  # subsample cap for the display embedding; higher = fuller coral/cylinder
# (dataset, label, cell-method, orientation, [ (process label, shape, process key, path-branches) ])
SCEN = [
    ("gtcc_a", "a  Cell cycle", "tsne", "h", [("", "ring", "cyc", None)]),
    ("gtcc_b", "b  Bifurcation", "tsne", "v", [("", "tree", "diff", [0, 1])]),
    ("gtcc_c", "c  Linear + cell cycle", "umap", "h",
     [("Process 1 (differentiation)", "line", "diff", [0]), ("Process 2 (cell cycle)", "ring", "cyc", None)]),
    ("gtcc_d", "d  Multilayered + cell cycle", "t-SNE", "v",
     [("Process 1 (differentiation)", "coral", "diff", [0, 1, 3]), ("Process 2 (cell cycle)", "ring", "cyc", None)]),
]
# schematic line segments + a lineage path (for placing g's) per shape
# tree/coral geometry (all segments drawn; the lineage path is routed to the arm chosen FROM THE DATA)
TREE_TIPS = [(-0.75, 0.98), (0.75, 0.98)]                         # 0=left, 1=right
TREE_SEGS = [((0, -1.05), (0, 0)), ((0, 0), TREE_TIPS[0]), ((0, 0), TREE_TIPS[1])]
CORAL_LM, CORAL_RM = (-0.55, 0.1), (0.55, 0.1)
CORAL_TIPS = [(-0.9, 0.75), (-0.22, 0.75), (0.22, 0.75), (0.9, 0.75)]   # 0=leftmost .. 3=rightmost
CORAL_SEGS = [((0, -1.1), (0, -0.35)), ((0, -0.35), CORAL_LM), ((0, -0.35), CORAL_RM),
              (CORAL_LM, CORAL_TIPS[0]), (CORAL_LM, CORAL_TIPS[1]),
              (CORAL_RM, CORAL_TIPS[2]), (CORAL_RM, CORAL_TIPS[3])]


def path_target(shape, U, counts_u, branch, mask, path_branches):
    """Which drawn arm the lineage occupies — the one where its terminal genes actually express in the cell
    embedding U. Arms are ordered LEFT->RIGHT by their angular position around the embedding centre (captures a
    fan of arms far better than x alone). Returns tree side (0/1) or coral tip index (0-3)."""
    term = int(path_branches[-1])
    cen = U.mean(0)

    def ang(b):
        gi = np.where(mask & (branch == b))[0]
        if len(gi) == 0:
            return 0.0
        w = np.clip(counts_u[:, gi].sum(1), 0, None)
        c = (w[:, None] * U).sum(0) / (w.sum() + 1e-9) - cen
        return float(np.arctan2(c[1], c[0]))       # angle; larger = more counter-clockwise (upper-left)
    arms = [1, 2] if shape == "tree" else [3, 4, 5, 6]
    return sorted(arms, key=lambda b: -ang(b)).index(term)   # descending angle = left -> right


DIM3 = set()                                        # display embeddings are now all 2-D (cached & pre-oriented)


def load_display_U(ds):
    """The cached 2-D display embedding for this row (see prep_fig2_embeddings.py): one row per cell in original
    CSV order, already oriented to match GeneTrajectory (a ring · b root-down tree · c dense square · d coral)."""
    f = figures_dir() / "fig2_cache" / f"U_{ds}.npy"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing — run: python scripts/prep_fig2_embeddings.py")
    U = np.load(f)
    if ds == "gtcc_d":                                   # tilt the coral ~35 deg onto its side (user preference)
        th = np.deg2rad(35.0)
        U = U @ np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    return U


def load_display_U_cyc(ds):
    """Cached CELL-CYCLE cell-view (cells embedded by the cell-cycle program's genes -> a ring), so the
    cell-cycle genes light distinct rotating arcs. Same cell order as load_display_U. See prep_cyc_ring.py."""
    f = figures_dir() / "fig2_cache" / f"U_cyc_{ds}.npy"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing — run: python scripts/prep_cyc_ring.py")
    return np.load(f)


def cell_embed(counts, method, ndim=2):
    from sklearn.decomposition import PCA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = np.log1p(counts)
        npc = min(30, X.shape[1] - 1, X.shape[0] - 1)
        Xp = PCA(n_components=npc, random_state=0).fit_transform(X)
        if method == "tsne":
            from sklearn.manifold import TSNE
            return TSNE(n_components=ndim, random_state=0, init="pca",
                        perplexity=min(100, (X.shape[0] - 1) // 3)).fit_transform(Xp)
        import umap
        nn, md = (30, 0.6) if ndim == 3 else (15, 0.3)   # fuller manifold in 3-D (thicker branches/tube)
        return umap.UMAP(n_components=ndim, random_state=0, n_neighbors=nn, min_dist=md).fit_transform(Xp)


def orient3d(U, cell_pt):
    U = U - U.mean(0)
    _, V = np.linalg.eigh(U.T @ U)
    R = U @ V[:, ::-1]                               # PC1..PC3
    if np.corrcoef(R[:, 0], cell_pt)[0, 1] < 0:      # PC1 = main axis, increasing with pseudotime
        R[:, 0] = -R[:, 0]
    return R


def sq3d(ax, U):
    c = U.mean(0); hw = (U.max(0) - U.min(0)).max() / 2 * 1.05
    ax.set_xlim(c[0] - hw, c[0] + hw); ax.set_ylim(c[1] - hw, c[1] + hw); ax.set_zlim(c[2] - hw, c[2] + hw)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.view_init(elev=16, azim=-68)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def orient(U, cell_pt, axis):
    U = U - U.mean(0)
    _, V = np.linalg.eigh(U.T @ U)
    R = U @ V[:, ::-1]
    cp = np.asarray(cell_pt, float)
    if axis == "h":
        if np.corrcoef(R[:, 0], cp)[0, 1] < 0:
            R[:, 0] = -R[:, 0]
    else:
        R = R[:, ::-1]
        if np.corrcoef(R[:, 1], cp)[0, 1] < 0:
            R[:, 1] = -R[:, 1]
    return R


def sq(ax, x, y, pct=0.5):
    cx, cy = np.median(x), np.median(y)
    hw = max(np.percentile(x, 100 - pct) - np.percentile(x, pct),
             np.percentile(y, 100 - pct) - np.percentile(y, pct)) / 2 * 1.25 + 1e-9
    ax.set_xlim(cx - hw, cx + hw); ax.set_ylim(cy - hw, cy + hw); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def sq_full(ax, x, y, m=0.12):
    cx = (x.min() + x.max()) / 2; cy = (y.min() + y.max()) / 2
    hw = max(x.max() - x.min(), y.max() - y.min()) / 2 * (1 + m) + 1e-9
    ax.set_xlim(cx - hw, cx + hw); ax.set_ylim(cy - hw, cy + hw); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def porder01(pt):
    r = np.argsort(np.argsort(pt)).astype(float)
    return r / (len(r) - 1 + 1e-9)


def eigengap_K(E, kmax=6):
    """K = number of leading diffusion components before the largest drop in per-component scale (embedding.npy
    columns are eigenvalue-scaled eigenvectors, so column std tracks the eigenvalue)."""
    s = E[:, :kmax].std(0)
    if len(s) < 3:
        return len(s)
    drops = s[:-1] - s[1:]
    return int(np.argmax(drops[1:]) + 2)   # at least 2 components


def interp_path(path, n):
    P = np.array(path, float)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    t = np.linspace(0, seg[-1], n)
    return [(np.interp(ti, seg, P[:, 0]), np.interp(ti, seg, P[:, 1])) for ti in t]


def draw_schematic(ax, shape, gene_pts, labels, ptmax, target=0):
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect("equal"); ax.axis("off")
    if shape == "ring":
        th = np.linspace(0, 2 * np.pi, 120)
        ax.plot(np.cos(th), np.sin(th), "k", lw=1.2)
        frac = np.clip(np.asarray(gene_pts) / (ptmax + 1e-9), 0, 1)
        pos = [(np.cos(2 * np.pi * f), np.sin(2 * np.pi * f)) for f in frac]
    elif shape == "line":
        ax.plot([-1.05, 1.05], [0, 0], "k", lw=1.2)
        pos = interp_path([(-1.0, 0), (1.0, 0)], len(labels))
    elif shape == "tree":
        for (a, b) in TREE_SEGS:
            ax.plot([a[0], b[0]], [a[1], b[1]], "k", lw=1.2)
        pos = interp_path([(0, -1.05), (0, 0), TREE_TIPS[target]], len(labels))
    else:  # coral: route trunk -> matching mid-fork -> chosen tip
        for (a, b) in CORAL_SEGS:
            ax.plot([a[0], b[0]], [a[1], b[1]], "k", lw=1.2)
        mid = CORAL_LM if target < 2 else CORAL_RM
        pos = interp_path([(0, -1.1), (0, -0.35), mid, CORAL_TIPS[target]], len(labels))
    for (x, y), lab in zip(pos, labels):
        ax.plot(x, y, "o", ms=7, color="#555")
        ax.text(x, y + 0.22, lab, fontsize=6.5, ha="center", va="bottom")


def pick_path_genes(gene_pt, branch, mask, path_branches, n, cyclic=False):
    """Genes along a SINGLE lineage path, spaced by ABSOLUTE pseudotime (segment index * maxT + peak) so
    consecutive g's advance smoothly along one arm — the last picks (g4,g5) land together on the terminal
    branch rather than one near the fork and one at the tip on a different arm.

    cyclic: for a cell-cycle (ring) lineage the peak phase lives on a CIRCLE, so min and max phase are the
    same point. Sampling both endpoints with linspace picks two near-identical genes (e.g. g6==g10, expr
    corr ~0.5) and leaves the ring unevenly covered; instead take n phases evenly AROUND the circle, dropping
    the wrap-around duplicate, so the five genes are distinct and equispaced on the ring."""
    idx = np.where(mask)[0]
    if path_branches is None:                       # linear / cyclic: single segment
        abspt = gene_pt[idx]
    else:
        order_of = {b: k for k, b in enumerate(path_branches)}
        idx = np.array([i for i in idx if branch[i] in order_of])
        ptmax = float(gene_pt[idx].max()) + 1e-9
        abspt = np.array([order_of[branch[i]] * ptmax + gene_pt[i] for i in idx])
    if len(idx) < n:
        return idx[np.argsort(abspt)]
    order = np.argsort(abspt); idx, abspt = idx[order], abspt[order]
    lo, hi = abspt[0], abspt[-1]
    if cyclic:                                      # n points around the circle; no min/max endpoint duplicate
        targets = lo + (hi - lo) * (np.arange(n) / n)
    else:
        targets = np.linspace(lo, hi, n)            # even spacing by VALUE, not rank
    return np.array([idx[np.argmin(np.abs(abspt - t))] for t in targets])


def best_pair(Eg, kmax=6):
    """The 2 diffusion components (of the first kmax) with the most spread for this gene set — for a
    linear/tree lineage that surfaces the trajectory."""
    v = Eg[:, :kmax].var(0)
    a, b = np.argsort(v)[::-1][:2]
    return (int(min(a, b)), int(max(a, b)))


def ringness(x, y):
    """Unsupervised score (no ground truth): high for a round, hollow, angularly-complete loop."""
    cx, cy = x.mean(), y.mean(); r = np.hypot(x - cx, y - cy); a = np.arctan2(y - cy, x - cx)
    if r.mean() < 1e-9:
        return 0.0
    roundness = 1.0 / (1.0 + r.std() / r.mean())              # low radius CV
    hollow = float(np.mean(r > 0.5 * np.median(r)))           # few points near the centre
    h, _ = np.histogram(a, bins=12, range=(-np.pi, np.pi))    # all angular sectors occupied
    return roundness * hollow * float(np.mean(h > 0))


def pick_ring(Eg, kmax=10):
    """The component pair that best forms a RING (unsupervised) — a cyclic lineage's ring can live in higher
    diffusion components than the top-variance pair (esp. when a complex differentiation dominates variance,
    e.g. the coral). Searches more components than best_pair; selection uses no ground-truth order."""
    K = min(kmax, Eg.shape[1]); best = (-1.0, 0, 1)
    for i in range(K):
        for j in range(i + 1, K):
            s = ringness(Eg[:, i], Eg[:, j])
            if s > best[0]:
                best = (s, i, j)
    return (best[1], best[2])


def gene_embed_panel(ax, E, mask, gene_pt, glabels, gpick, cyclic=False):
    """Draw one process's genes in the gene embedding (best 2-D slice), colored by within-lineage pseudo-order,
    with the g labels placed at their gene positions. Cyclic lineages use the ring-forming slice (pick_ring);
    linear/tree lineages use the max-spread slice (best_pair)."""
    Eg = E[mask]
    p = pick_ring(Eg) if cyclic else best_pair(Eg)
    XY = E[:, list(p)]
    col = porder01(gene_pt[mask])
    ax.scatter(XY[mask, 0], XY[mask, 1], c=col, cmap="magma", vmin=0, vmax=1, s=9, linewidths=0)
    sq_full(ax, XY[mask, 0], XY[mask, 1])           # fix the square frame FIRST so labels can be kept inside it
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    inset = 0.055 * (x1 - x0)                        # keep every label at least this far inside the frame
    off = 0.13 * (x1 - x0)                           # nominal outward label offset (data units)
    cen = XY[mask].mean(0)
    placed = []                                      # already-placed label centres, to de-overlap crowded pairs
    for gi, lab in zip(gpick, glabels):
        pt = XY[gi]; d = pt - cen; ang = np.arctan2(d[1], d[0]) if np.any(d) else 0.0
        lx, ly = pt[0] + off * np.cos(ang), pt[1] + off * np.sin(ang)
        # reflect the offset inward whenever the outward label would spill past the frame -- this is what keeps
        # bottom-vertex genes (g3 on the arc, g9 on the ring) from dropping below the panel into the next title.
        if not (x0 + inset <= lx <= x1 - inset): lx = pt[0] - off * np.cos(ang)
        if not (y0 + inset <= ly <= y1 - inset): ly = pt[1] - off * np.sin(ang)
        # nudge apart labels that land almost on top of a previous one (terminal pairs g4/g5, cycle-wrap g6/g10)
        for px, py in placed:
            if abs(lx - px) < 0.9 * off and abs(ly - py) < 0.55 * off:
                ly += 0.9 * off * (1 if ly >= py else -1)
        lx = min(max(lx, x0 + inset), x1 - inset); ly = min(max(ly, y0 + inset), y1 - inset)
        placed.append((lx, ly))
        ax.annotate(lab, xy=(pt[0], pt[1]), xytext=(lx, ly), textcoords="data",
                    fontsize=6.5, fontweight="bold", ha="center", va="center",
                    color="black", arrowprops=dict(arrowstyle="-", lw=0.4, color="#555", shrinkA=0, shrinkB=1))
    return p


def main():
    GROWS = [len(s[4]) for s in SCEN]; G = sum(GROWS); NC = 2 + NG + 1; S = 2.05
    fig = plt.figure(figsize=(NC * S, G * S))
    gs = GridSpec(G, NC, figure=fig, hspace=0.32, wspace=0.10)
    grow = 0
    for si, (ds, label, method, axis, subrows) in enumerate(SCEN):
        meta = np.load(WS / "data" / ds / "sim_meta.npz", allow_pickle=True)
        gene_pt = meta["truth_peak"]; process = meta["process"]; branch = meta["gene_branch"]
        counts = pd.read_csv(WS / "data" / ds / f"filtered_{ds}_cells_x_genes.csv", index_col=0).values.astype(float)
        cell_pt = meta["cell_pt"]
        counts_u, cell_pt_u = counts, cell_pt          # all cells; display embedding is cached & pre-oriented
        d3 = False
        U = load_display_U(ds)
        assert U.shape[0] == counts_u.shape[0], f"{ds}: cached U {U.shape} vs counts {counts_u.shape}"
        E, names, _ = load_prism_gene_space(ds)
        is_cyc = process == "cyc"; is_diff = process == "diff"; concurrent = len(subrows) == 2

        axc = fig.add_subplot(gs[grow, 0], projection="3d" if d3 else None)
        if d3:
            axc.scatter(U[:, 0], U[:, 1], U[:, 2], color="#222222", s=3, depthshade=True, linewidths=0, rasterized=True)
            sq3d(axc, U)
        else:
            axc.scatter(U[:, 0], U[:, 1], color="#222222", s=4, linewidths=0, rasterized=True)
            sq(axc, U[:, 0], U[:, 1])
        axc.set_ylabel(label, fontsize=9.5)
        axc.set_title(("cell embedding\n" if si == 0 else "") + f"({method})", fontsize=8.3)

        gcount = 1
        for subi, (proc_label, shape, pkey, pbr) in enumerate(subrows):
            r = grow + subi
            mask = is_diff if pkey == "diff" else is_cyc
            cyc_view = False        # REVERTED: both sub-rows share the main cell embedding; the cell cycle is
            Usub = U                # deconvolved in the last column's gene embedding (a ring), not by a separate view
            gpick = pick_path_genes(gene_pt, branch, mask, pbr, NG, cyclic=(pkey == "cyc"))
            glabels = [f"g{gcount + j}" for j in range(len(gpick))]
            ptmax = float(np.nanmax(gene_pt[mask])) if mask.any() else 1.0
            tgt = path_target(shape, U, counts_u, branch, mask, pbr) if shape in ("tree", "coral") else 0
            if shape == "coral":
                tgt = 1  # Yanir 2026-07-10: route the differentiation lineage to the second-left arm so g4,g5 land there
            axs = fig.add_subplot(gs[r, 1]); draw_schematic(axs, shape, gene_pt[gpick], glabels, ptmax, tgt)
            if si == 0 and subi == 0: axs.set_title("process", fontsize=8.5)
            if proc_label:
                axs.text(0.5, -0.08, proc_label, transform=axs.transAxes, ha="center", va="top",
                         fontsize=7, color=DIFF_C if pkey == "diff" else CYC_C)
            if cyc_view:                                      # the cyc sub-row draws its OWN (ring) cell embedding
                axcc = fig.add_subplot(gs[r, 0])
                axcc.scatter(Usub[:, 0], Usub[:, 1], color="#222222", s=4, linewidths=0, rasterized=True)
                sq(axcc, Usub[:, 0], Usub[:, 1])
                axcc.set_title("cell-cycle view\n(cells by cell-cycle genes)", fontsize=7.5)
            for j, gi in enumerate(gpick):
                axg = fig.add_subplot(gs[r, 2 + j])
                v = np.log1p(counts_u[:, gi]); vmax = max(np.percentile(v, 99), 1e-6)
                o = np.argsort(v)
                axg.scatter(Usub[o, 0], Usub[o, 1], c=v[o], cmap="viridis", s=5, vmin=0, vmax=vmax,
                            linewidths=0, rasterized=True)
                sq(axg, Usub[:, 0], Usub[:, 1])
                axg.set_title(glabels[j], fontsize=8.5)

            # per-process gene-embedding panel in this sub-row's last column
            axe = fig.add_subplot(gs[r, 2 + NG])
            gene_embed_panel(axe, E, mask, gene_pt, glabels, gpick, cyclic=(pkey == "cyc"))
            if concurrent:
                axe.set_title(f"gene embedding\n{'diff.' if pkey == 'diff' else 'cell cycle'} lineage", fontsize=8)
            else:
                rr = spearmanr(E[:, 0], gene_pt)[0]
                tag = " (ring)" if meta["is_cyclic"] else ""
                axe.set_title(f"PRISM gene embedding\nby pseudo-order (|rho|={abs(rr):.2f}){tag}", fontsize=8)
            gcount += len(gpick)

        # deconv-silhouette annotation removed per Yanir 2026-07-10 (not relevant; it cluttered the panels).
        grow += len(subrows)

    # No in-figure title: the caption lives in the LaTeX \caption (Yanir 2026-07-10).
    fig.tight_layout(rect=[0, 0, 0.99, 1.0])
    out = figures_dir() / "fig2_simulations.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=140)
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
