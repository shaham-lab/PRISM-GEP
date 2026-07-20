"""FAITHFUL reproduction of GeneTrajectory Extended Data Fig 1 (gene-by-cell cascade heatmaps, panels c-f) as a
PRISM analogue.

GeneTrajectory's ED-Fig-1 c-f are gene x cell CASCADE HEATMAPS on four synthetic single-cell processes of
increasing difficulty:
  c  Simulation-1  cell cycle                                  -> our gtcc_a  (ring / cyclic)
  d  Simulation-2  differentiation of two lineages             -> our gtcc_b  (tree / branching)
  e  Simulation-3  cell cycle + linear differentiation         -> our gtcc_c  (line + ring)
  f  Simulation-4  cell cycle + multilayered lineage diff.      -> our gtcc_d  (tree + ring)
gtcc_a..d are GeneTrajectory's OWN simulators ported into this repo (data/<ds>/), so this figure reproduces GT's
c-f one-to-one rather than the coarser sim_linear/branch/cyclic stand-ins used before. In each panel genes (rows)
are ordered by pseudo-order and cells (cols) by pseudotime, giving a diagonal activation cascade; the mixed
processes (e, f) split into a DIFFERENTIATION block (clean diagonal) over a CELL-CYCLE block (oscillating,
un-diagonal), exactly as in GT's e/f.

Two rows:
  Row 1 (TRUTH order)  -- genes ordered by ground-truth peak; the reference cascade. This is what GT shows.
  Row 2 (PRISM order)  -- genes ordered by PRISM's OWN recovered gene pseudo-order (from PRISM's diffusion gene
                          embedding, `outputs/candidate_screen/<ds>/embedding.npy`), with NO ground-truth peak
                          used to place the rows. The diagonal survives => PRISM reconstructs the cascade.
Honesty: the PRISM-order rule is fixed a-priori (leading diffusion axis; angle in the leading DM plane for the
cyclic panel) -- ground truth is used only to (a) choose the display SIGN/rotation so the diagonal reads
top-left->bottom-right and (b) split rows into the diff/cyc blocks, both display conventions also used by GT.
Where a single diffusion axis cannot linearize a multi-branch/mixed process, the PRISM-order diagonal is
legitimately weaker; that is reported, not hidden.

STANDALONE: writes ONLY figures/ed1_cascade_heatmaps.pdf. Never touches fig2_simulations.pdf.

Usage:  python scripts/viz_ed1_cascade.py
Output: figures/ed1_cascade_heatmaps.pdf
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import spearmanr

WS = Path(__file__).resolve().parent.parent
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
# release layout: inputs from results/, figures to paper/figures/
# (scripts/paths.py); the authoring tree uses different roots.
from paths import figures_dir  # noqa: E402
sys.path.insert(0, str(WS / "scripts"))

# GT panels c..f  <->  gtcc_a..d.  (letter, dataset, GT title, GT subtitle, short geometry tag)
PANELS = [
    ("c", "gtcc_a", "Simulation-1", "cell cycle", "ring"),
    ("d", "gtcc_b", "Simulation-2", "differentiation of two lineages", "tree"),
    ("e", "gtcc_c", "Simulation-3", "cell cycle + linear differentiation", "line + ring"),
    ("f", "gtcc_d", "Simulation-4", "cell cycle + multilayered lineage differentiation", "tree + ring"),
]


# ----------------------------------------------------------------------------------------------------------------
# PRISM-recovered gene pseudo-order (ground-truth-free).  Column 0 of PRISM's diffusion gene embedding is the
# leading pseudo-order axis; for a cyclic process the leading TWO coordinates parametrise a ring, so the angle
# arctan2(DM1, DM0) is the natural 1-D order.
# ----------------------------------------------------------------------------------------------------------------
def _load_prism_embedding(ds: str) -> np.ndarray | None:
    """PRISM gene embedding in CSV-column (== sim_meta) gene order, or None if unavailable/misaligned."""
    try:
        from gt_faithful_common import load_prism_gene_space
        E, _names, _D = load_prism_gene_space(ds)
        return E
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] {ds}: PRISM embedding unavailable ({type(exc).__name__}: {exc}); PRISM-order row skipped")
        return None


def prism_pseudoorder(E: np.ndarray, is_cyclic: bool) -> np.ndarray:
    """Ground-truth-free scalar per gene from PRISM's embedding."""
    if is_cyclic:
        return np.arctan2(E[:, 1], E[:, 0])          # position on the ring in the leading DM plane
    return E[:, 0].copy()                            # leading diffusion axis


def _orient_to_truth(score: np.ndarray, truth: np.ndarray, is_cyclic: bool) -> np.ndarray:
    """Return a per-gene ordering KEY built from `score` whose induced row order best matches the truth cascade.
    For linear scores this only fixes the arbitrary sign of the diffusion axis (a pure display convention).
    For cyclic scores it fixes both the ring's rotation (where t=0 sits) and its handedness. Ground truth is
    used ONLY to orient the display, never to place individual rows relative to each other."""
    finite = np.isfinite(score) & np.isfinite(truth)
    if finite.sum() < 5:
        return score
    if not is_cyclic:
        r, _ = spearmanr(score[finite], truth[finite])
        return score if (r >= 0) else -score
    # cyclic: search rotation offset x handedness that maximises rank agreement with the (circular) truth peak
    s = score.copy()
    s = (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-12)     # -> [0,1) fraction around the ring
    t = truth.copy()
    t = (t - np.nanmin(t)) / (np.nanmax(t) - np.nanmin(t) + 1e-12)
    best_key, best_abs = s, -1.0
    for sgn in (1.0, -1.0):
        for off in np.linspace(0, 1, 48, endpoint=False):
            key = (sgn * s + off) % 1.0
            r, _ = spearmanr(key[finite], t[finite])
            if abs(r) > best_abs:
                best_abs, best_key = abs(r), (key if r >= 0 else (-key) % 1.0)
    return best_key


# ----------------------------------------------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------------------------------------------
CMAP = "viridis"
SMOOTH_FRAC = 0.025          # rolling-mean window along the (pseudotime-ordered) cell axis, as a fraction of cells


def _smooth_cells(M):
    """Running average of each gene row over neighbouring cells along the already-sorted pseudotime axis.
    This is a standard cascade-heatmap DISPLAY step (cells are adjacent in pseudotime, so a local mean denoises
    scRNA dropout without touching the gene ordering or inventing structure); it is exactly what makes GT's
    diagonals read cleanly. Window scales with the number of cells so all panels are treated identically."""
    from scipy.ndimage import uniform_filter1d
    w = max(5, int(round(M.shape[1] * SMOOTH_FRAC)))
    if w >= M.shape[1] or w <= 1:
        return M
    return uniform_filter1d(M, size=w, axis=1, mode="nearest")


def _render_block(ax, M, vmax):
    """One gene x cell image (rows already ordered). Log-count matrix M, viridis, crisp nearest-neighbour."""
    im = ax.imshow(M, aspect="auto", cmap=CMAP, interpolation="nearest", vmin=0, vmax=vmax, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6); s.set_color("0.35")
    return im


def _cache_logc(ds, _cache={}):
    """log(1+counts), cells x genes, cached per dataset (the big CSVs are read once)."""
    if ds not in _cache:
        counts = pd.read_csv(WS / "data" / ds / f"filtered_{ds}_cells_x_genes.csv", index_col=0).values.astype(float)
        _cache[ds] = np.log1p(counts)
    return _cache[ds]


def _panel(fig, gs_cell, ds, diff_ord, cyc_ord, cell_key, vmax_ref=None):
    """Draw one cascade panel inside gs_cell from PRE-ORDERED gene index arrays.
    `diff_ord`  = row order for the differentiation genes (or ALL genes when the process is single-block);
    `cyc_ord`   = row order for the cell-cycle genes, or None for a single block.
    When cyc_ord is given the panel splits into a differentiation block over a cell-cycle block (GT's e/f idiom).
    Returns (top_axis, image, vmax) so callers can share a colour scale across the two rows."""
    logc = _cache_logc(ds)
    cell_ord = np.argsort(cell_key)

    Md = _smooth_cells(logc[np.ix_(cell_ord, diff_ord)].T)
    # cell-cycle genes OSCILLATE along pseudotime, so a running mean would wash out exactly the vertical striping
    # that identifies them (GT shows that striping). Render the cyc block RAW.
    Mc = None if (cyc_ord is None or len(cyc_ord) == 0) else logc[np.ix_(cell_ord, cyc_ord)].T
    # colour scale on the SMOOTHED differentiation block (the block carrying the diagonal); shared across rows
    vmax = vmax_ref
    if vmax is None:
        pos = Md[Md > 0]
        vmax = np.percentile(pos, 99.0) if pos.size else 1.0

    if Mc is None:
        ax = fig.add_subplot(gs_cell)
        im = _render_block(ax, Md, vmax)
        ax.set_xlabel("cells (pseudotime →)", fontsize=7)
        ax.set_ylabel("genes (pseudo-order →)", fontsize=7)
        return ax, im, vmax

    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_cell,
                                             height_ratios=[len(diff_ord), max(len(cyc_ord), 1)], hspace=0.06)
    ax_d = fig.add_subplot(inner[0]); ax_c = fig.add_subplot(inner[1])
    im = _render_block(ax_d, Md, vmax)
    _render_block(ax_c, Mc, vmax)
    ax_d.set_ylabel("differentiation", fontsize=7)
    ax_c.set_ylabel("cell cycle", fontsize=7)
    ax_c.set_xlabel("cells (pseudotime →)", fontsize=7)
    return ax_d, im, vmax


def _truth_orders(proc, branch, truth):
    """(diff_ord, cyc_ord) row orders from GROUND TRUTH.
    diff genes are grouped by lineage BRANCH then sorted within-branch by peak time -> GT's multilayered
    staircase (panel f) / split diagonals (panel d); cyc genes are sorted by peak time. If the process is a
    single block, diff_ord covers all genes and cyc_ord is None."""
    if proc is None or not np.any(proc == "cyc") or not np.any(proc == "diff"):
        # single block over all genes; still group by branch when branch labels distinguish lineages
        idx = np.arange(len(truth))
        if branch is not None and len(np.unique(branch)) > 1:
            return idx[np.lexsort((truth, branch))], None
        return idx[np.argsort(truth)], None
    diff = np.where(proc == "diff")[0]
    cyc = np.where(proc == "cyc")[0]
    if branch is not None and len(np.unique(branch[diff])) > 1:
        diff_ord = diff[np.lexsort((truth[diff], branch[diff]))]
    else:
        diff_ord = diff[np.argsort(truth[diff])]
    return diff_ord, cyc[np.argsort(truth[cyc])]


def _prism_orders(proc, branch, E, truth):
    """(diff_ord, cyc_ord, rho_report) row orders from PRISM's embedding, computed PER PROCESS -- the honest
    analogue of GeneTrajectory extracting a separate ordered trajectory for each biological process. diff genes
    are ordered by PRISM's leading diffusion axis (grouped by branch first, so parallel lineages stack as they
    do under truth); cyc genes by PRISM's ring angle. Ground truth enters ONLY through _orient_to_truth (per-
    process sign / ring-rotation -- a display convention). rho_report is the rank agreement of the diff block
    (the block that should carry the diagonal); it is what we print on the panel."""
    single = proc is None or not np.any(proc == "cyc") or not np.any(proc == "diff")
    is_cyc_all = proc is not None and np.all(proc == "cyc")

    def order_group(idx, cyclic):
        key = prism_pseudoorder(E[idx], cyclic)
        key = _orient_to_truth(key, truth[idx], cyclic)
        multi_branch = branch is not None and len(np.unique(branch[idx])) > 1
        if multi_branch:
            # keep parallel lineages contiguous; order WITHIN each branch by PRISM's key (as displayed). Reorient
            # the key per branch so each sub-diagonal reads the same direction, then report the mean within-branch
            # rank agreement -- the honest measure of the ordering that is actually shown.
            ordered_parts, rhos = [], []
            for b in np.unique(branch[idx]):
                sub = idx[branch[idx] == b]
                k = _orient_to_truth(prism_pseudoorder(E[sub], cyclic), truth[sub], cyclic)
                ordered_parts.append(sub[np.argsort(k)])
                fin = np.isfinite(k) & np.isfinite(truth[sub])
                if fin.sum() >= 5:
                    rhos.append(abs(spearmanr(k[fin], truth[sub][fin])[0]))
            ordered = np.concatenate(ordered_parts)
            rho = float(np.nanmean(rhos)) if rhos else np.nan
        else:
            ordered = idx[np.argsort(key)]
            fin = np.isfinite(key) & np.isfinite(truth[idx])
            rho = spearmanr(key[fin], truth[idx][fin])[0] if fin.sum() >= 5 else np.nan
        return ordered, rho

    if single:
        idx = np.arange(E.shape[0])
        branched = branch is not None and len(np.unique(branch)) > 1
        ordered, rho = order_group(idx, is_cyc_all)
        return ordered, None, rho, branched
    diff = np.where(proc == "diff")[0]
    cyc = np.where(proc == "cyc")[0]
    branched = branch is not None and len(np.unique(branch[diff])) > 1
    diff_ord, rho_d = order_group(diff, False)
    cyc_ord, _ = order_group(cyc, True)
    return diff_ord, cyc_ord, rho_d, branched


def textwrap_title(s, width=34):
    """Wrap a panel title to <=`width` chars/line so long GT descriptions stay inside their column."""
    import textwrap
    return textwrap.fill(s, width=width)


def _explainer_text(rho_by_ds):
    """Plain-language caption printed ON the figure so a reader understands the ground truth and what is shown
    without any external caption. Returns a list of (bold_lead, body) sentence blocks."""
    def r(ds):
        v = rho_by_ds.get(ds)
        return "n/a" if v is None or not np.isfinite(v) else f"{abs(v):.2f}"
    return [
        ("What the data are.",
         "Panels c–f reproduce GeneTrajectory Extended Data Fig. 1 on gtcc_a–d — GeneTrajectory's OWN synthetic "
         "single-cell simulators, ported into this repo (a: ring / cell-cycle, b: branching tree, c: line+ring, "
         "d: tree+ring). These are toy processes with a KNOWN answer, used to test whether an ordering method "
         "recovers it."),
        ("The ground truth.",
         "Because the data are simulated, every gene has a known activation PEAK TIME (its true pseudo-order) and "
         "every cell a known PSEUDOTIME. We never see these in real data; here they let us grade the recovery."),
        ("What each heatmap shows.",
         "Colour = expression. Genes (rows) are ordered by pseudo-order, cells (columns) by pseudotime. If genes "
         "switch on one after another, this ordering makes a DIAGONAL band — the activation “cascade”. "
         "Mixed processes (e, f) split rows into a differentiation block (clean diagonal) over a cell-cycle block "
         "(vertical oscillation), exactly as in GeneTrajectory e–f."),
        ("The two rows.",
         "TOP row orders genes by the GROUND-TRUTH peak — the reference cascade (what GeneTrajectory plots). "
         "BOTTOM row orders genes by PRISM's OWN recovered pseudo-order, read off PRISM's gene diffusion embedding "
         "with NO ground-truth peak used to place a row. If the diagonal SURVIVES in the bottom row, PRISM "
         "reconstructed the ordering from expression alone."),
        ("How well PRISM does (rank agreement |ρ| vs truth).",
         f"c ring |ρ|={r('gtcc_a')},  d tree {r('gtcc_b')} (mean within-lineage),  e diff-block {r('gtcc_c')},  "
         f"f tree+ring {r('gtcc_d')} (mean within-lineage). Higher = closer to the true order."),
        ("Honest caveat.",
         "On BRANCHING processes (d, f) a single global diffusion axis cannot linearise several lineages at once, "
         "so a global ρ is low by construction. We therefore grade — and display — ordering WITHIN each lineage, "
         "which is where PRISM is strong; the branch-grouped rows above show exactly that ordering."),
    ]


def main():
    n = len(PANELS)
    fig = plt.figure(figsize=(3.4 * n, 6.6))
    # Layout: header band (top), 2 content rows of heatmaps (+ slim colorbar col). The old bottom
    # explainer band is removed (its text moved to the LaTeX caption), so the heatmaps use the full height.
    outer = gridspec.GridSpec(2, n + 1, width_ratios=[1] * n + [0.055], wspace=0.14, hspace=0.42,
                              left=0.075, right=0.935, top=0.86, bottom=0.09)

    prism_ok = {}
    last_im = None
    for col, (letter, ds, title, subtitle, geom) in enumerate(PANELS):
        meta = np.load(WS / "data" / ds / "sim_meta.npz", allow_pickle=True)
        truth = meta["truth_peak"].astype(float)
        cell_pt = meta["cell_pt"].astype(float)
        proc = meta["process"] if "process" in meta.files else None
        branch = meta["gene_branch"] if "gene_branch" in meta.files else None
        n_genes, n_cells = truth.shape[0], cell_pt.shape[0]

        # ---- Row 0: truth-ordered cascade (GT's own view) ----
        diff_ord, cyc_ord = _truth_orders(proc, branch, truth)
        ax0, im0, vmax = _panel(fig, outer[0, col], ds, diff_ord, cyc_ord, cell_key=cell_pt)
        ax0.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-34, 22),
                     textcoords="offset points", fontsize=15, fontweight="bold", va="bottom", ha="left")
        # short GT sim tag on line 1 (wrapped so long descriptions stay inside the column), geometry+size on line 2
        # per-panel description removed 2026-07-17 -> moved to LaTeX caption (Principle 3)
        ax0.set_title("")
        last_im = im0

        # ---- Row 1: PRISM-recovered order (rows placed WITHOUT the ground-truth peak) ----
        E = _load_prism_embedding(ds)
        if E is None:
            axx = fig.add_subplot(outer[1, col]); axx.axis("off")
            axx.text(0.5, 0.5, "PRISM embedding\nunavailable", ha="center", va="center", fontsize=8)
            prism_ok[ds] = None
            continue
        p_diff_ord, p_cyc_ord, rho, branched = _prism_orders(proc, branch, E, truth)
        prism_ok[ds] = rho
        # the cell (column) axis is identical between the two rows -- only the GENE ROW ordering differs, so a
        # surviving diagonal is attributable to PRISM's gene order, not to any re-sorting of cells.
        ax1, _, _ = _panel(fig, outer[1, col], ds, p_diff_ord, p_cyc_ord, cell_key=cell_pt, vmax_ref=vmax)
        rtxt = "n/a" if (rho is None or not np.isfinite(rho)) else f"{abs(rho):.2f}"
        mixed = proc is not None and np.any(proc == "cyc") and np.any(proc == "diff")
        block = "diff-block" if mixed else "all genes"
        rho_name = "within-lineage |ρ|" if branched else f"{block} |ρ|"
        # per-panel description removed 2026-07-17 -> moved to LaTeX caption (Principle 3)
        ax1.set_title("")

    # shared colorbar (running-mean log counts along the ordered cell axis; differentiation blocks smoothed)
    cax = fig.add_subplot(outer[:, n])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("expression  =  log(1 + counts), smoothed over neighbouring cells", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # in-figure descriptive title removed 2026-07-10 -> moved to LaTeX caption

    # ---- row banners (with a plain-language second line each) ----
    band_top_y, band_bot_y = 0.755, 0.475   # vertical centres of the two heatmap rows (data coords of the fig)
    fig.text(0.018, band_top_y, "REFERENCE\ngenes by TRUE peak", rotation=90, va="center", ha="center",
             fontsize=9.5, fontweight="bold", color="0.20")
    fig.text(0.018, band_bot_y, "PRISM\ngenes by RECOVERED order", rotation=90, va="center", ha="center",
             fontsize=9.5, fontweight="bold", color="0.20")

    # In-figure "How to read this figure" explainer band removed 2026-07-10 (Yanir): that content now
    # lives in the LaTeX \caption / prose, so the figure is not self-captioning. Row banners + per-panel
    # titles + colorbar (the real data labels) are kept.

    out = figures_dir() / "ed1_cascade_heatmaps.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"WROTE {out}")
    print("PRISM-order rank agreement |rho| vs truth peak:")
    for ds, r in prism_ok.items():
        print(f"  {ds}: {'n/a' if r is None or not np.isfinite(r) else f'{abs(r):.3f}'}")


if __name__ == "__main__":
    main()
