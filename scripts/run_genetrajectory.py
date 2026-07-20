"""Run the OFFICIAL Kluger-lab GeneTrajectory (Qu et al. 2025) Python port
on the nine gene-trajectory benchmark datasets and add a `GeneTrajectory`
column to each dataset's gene-trajectory orders CSV, so the existing bootstrap
scorer (scripts/bootstrap_gene_trajectory_ci.py) picks it up automatically.

GeneTrajectory pipeline (matches the package tutorial):
  expression -> scanpy preprocess -> PCA -> run_dm (cell diffusion map)
  -> get_graph_distance (cell-cell graph distance)
  -> coarse_grain to metacells over a gene panel (variable genes U markers)
  -> cal_ot_mat (gene-gene Wasserstein/OT distance)
  -> get_gene_embedding + extract_gene_trajectory -> per-gene Pseudoorder

The marker genes' Pseudoorder is then compared to canonical_rank via the
same |Spearman rho| metric used for every other gene-ordering method.
Pseudoorder direction is arbitrary; the scorer uses |rho| so direction does
not matter.

IMPORTANT: POT eagerly imports jax as an optional backend; this env has an
incompatible jax/numpy pair, so we disable the jax backend before importing.

Run:
    python scripts/run_genetrajectory.py                 # all 9
    python scripts/run_genetrajectory.py --dataset pancreas
"""
from __future__ import annotations

import os
os.environ.setdefault("POT_BACKEND_DISABLE_JAX", "1")  # before ot/gene_trajectory import

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS))

from gene_trajectory.run_dm import run_dm  # noqa: E402
from gene_trajectory.get_graph_distance import get_graph_distance  # noqa: E402
from gene_trajectory.coarse_grain import coarse_grain_adata, select_top_genes  # noqa: E402
from gene_trajectory.gene_distance_shared import cal_ot_mat  # noqa: E402
from gene_trajectory.extract_gene_trajectory import (  # noqa: E402
    get_gene_embedding, extract_gene_trajectory,
)

DATASET_FILES = {
    "pancreas":               "filtered_pancreas_cells_x_genes.csv",
    "gastrulation":           "filtered_gastrulation_cells_x_genes.csv",
    "gastrulation_erythroid": "filtered_gastrulation_erythroid_cells_x_genes.csv",
    "hemogenic_endothelium":  "filtered_hemogenic_endothelium_cells_x_genes.csv",
    "bonemarrow":             "filtered_bonemarrow_cells_x_genes.csv",
    "gastrulation_e75":       "filtered_gastrulation_e75_cells_x_genes.csv",
    "paul15":                 "filtered_paul15_cells_x_genes.csv",
    "dentategyrus":           "filtered_dentategyrus_cells_x_genes.csv",
    "endoderm_diff":          "filtered_endoderm_diff_cells_x_genes.csv",
}

TRAJ = WS / "outputs" / "trajectory"
# Defaults tuned for SPEED: we only need to order ~15 marker genes, so a few
# hundred panel genes give a meaningful gene diffusion embedding while keeping
# cal_ot_mat (O(panel^2) OT solves, each O(metacells^3)) tractable. The full
# 2000-gene / 500-metacell tutorial config takes hours; 300 / 150 takes mins.
N_VARIABLE = 500      # gene panel size for the OT graph (variable genes U markers)
N_METACELLS = 250     # coarse-grain target (OT cost-matrix dimension)
N_GENE_EV = 10        # gene diffusion-map eigenvectors
N_TRAJ = 3            # number of gene trajectories to extract
OT_NUM_ITER_MAX = 20000  # EMD solver cap per gene pair


def orders_csv(ds: str) -> Path:
    return TRAJ / ds / f"gene_trajectory_{ds}_orders.csv"


def build_adata(ds: str) -> ad.AnnData:
    csv = WS / "data" / ds / DATASET_FILES[ds]
    df = pd.read_csv(csv, index_col=0)
    a = ad.AnnData(X=df.values.astype(np.float32))
    a.obs_names = [str(i) for i in df.index]
    a.var_names = [str(c) for c in df.columns]
    return a


def preprocess(a: ad.AnnData) -> None:
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.layers["log1p"] = a.X.copy()
    sc.pp.highly_variable_genes(a, n_top_genes=N_VARIABLE, flavor="seurat")
    sc.pp.scale(a, max_value=10)
    n_pcs = min(50, min(a.shape) - 1)
    sc.pp.pca(a, n_comps=n_pcs)
    # restore log1p as X for OT (OT works on non-negative expression)
    a.X = a.layers["log1p"].copy()


def run_one(ds: str) -> dict:
    print(f"\n=== GeneTrajectory: {ds} ===", flush=True)
    oc = orders_csv(ds)
    if not oc.exists():
        print(f"  no orders CSV at {oc}; skip")
        return {}
    orders = pd.read_csv(oc)
    markers = [str(g) for g in orders["gene"].tolist()]

    a = build_adata(ds)
    preprocess(a)

    # Gene panel = top variable genes UNION marker genes (so OT graph is rich
    # but markers are guaranteed present for the comparison). Marker matching
    # is CASE-INSENSITIVE: marker lists use mouse casing (Sox17) but some
    # datasets (e.g. human hemogenic_endothelium) use uppercase symbols
    # (SOX17). We map each orders-CSV marker to its real var_name and keep a
    # reverse map so the GeneTrajectory column re-aligns to the orders CSV.
    var_genes = list(a.var_names[a.var["highly_variable"].values])
    var_lut = {str(c).upper(): str(c) for c in a.var_names}
    marker_to_real = {g: var_lut[g.upper()] for g in markers if g.upper() in var_lut}
    real_to_marker = {v: k for k, v in marker_to_real.items()}
    present_markers = list(marker_to_real.values())  # real var_names
    missing = [g for g in markers if g.upper() not in var_lut]
    if missing:
        print(f"  markers missing from expr (skipped): {missing}")
    panel = sorted(set(var_genes) | set(present_markers))
    print(f"  panel={len(panel)} genes ({len(present_markers)}/{len(markers)} markers present)")

    print("  run_dm + get_graph_distance (cell-cell; k-retry until connected) ...", flush=True)
    cell_graph_dist = None
    for kk in (10, 25, 40, 60, 100, 150):
        try:
            run_dm(a, reduction="X_pca", k=kk, n_components=30)
            cell_graph_dist = get_graph_distance(a, reduction="X_dm", k=kk, dims=5)
            if kk > 10:
                print(f"    (raised k to {kk} to connect the cell kNN graph)")
            break
        except RuntimeError as e:
            if "disconnected" in str(e) and kk < 150:
                continue
            raise

    print(f"  coarse_grain -> {N_METACELLS} metacells ...", flush=True)
    gene_expr, graph_dist = coarse_grain_adata(
        a, cell_graph_dist, features=panel, n=N_METACELLS, reduction="X_dm",
        dims=5, random_seed=1,
    )
    # gene_expr: (n_panel_genes x n_metacells); graph_dist: (n_meta x n_meta)

    n_pairs = len(panel) * (len(panel) - 1) // 2
    print(f"  cal_ot_mat (gene-gene Wasserstein, {n_pairs} pairs on "
          f"{graph_dist.shape[0]}x{graph_dist.shape[0]} cost) ... [slowest step]",
          flush=True)
    gene_dist = cal_ot_mat(graph_dist, gene_expr, show_progress_bar=True,
                           num_iter_max=OT_NUM_ITER_MAX,
                           processes=max(1, (os.cpu_count() or 2) - 2))

    print("  get_gene_embedding + extract_gene_trajectory ...", flush=True)
    gene_emb, _ = get_gene_embedding(gene_dist, k=10, n_ev=N_GENE_EV, t=1)
    gt = extract_gene_trajectory(
        gene_emb, gene_dist, gene_names=panel,
        t_list=[3] * N_TRAJ, dims=5, k=10,
    )
    # gt is indexed by gene name with a 'selected' trajectory label and
    # 'Pseudoorder-<i>' columns (one per trajectory). For each gene take the
    # pseudoorder of the trajectory it was assigned to.
    print(f"  extract_gene_trajectory cols: {list(gt.columns)}")
    print(f"  trajectory assignment counts:\n{gt['selected'].value_counts().to_string()}")
    # Save FULL trajectory table (gene -> selected label + every Pseudoorder-i) so the
    # 'best-trajectory' scoring variant can pick the single extracted trajectory that best
    # matches the canonical markers -- the fair scoring for a MULTI-program method.
    gt.reset_index().rename(columns={"index": "gene"}).to_csv(
        TRAJ / ds / "gt_full_trajectories.csv", index=False)
    pseudo_cols = [c for c in gt.columns if c.lower().startswith("pseudoorder")]

    def _gene_pseudoorder(row) -> float:
        sel = row["selected"]
        # 'selected' is e.g. 'Trajectory-2' -> use 'Pseudoorder-2'
        if isinstance(sel, str) and sel.startswith("Trajectory-"):
            col = f"Pseudoorder-{sel.split('-')[1]}"
            if col in row and pd.notna(row[col]):
                return float(row[col])
        # BUGFIX: 'Other'/unassigned genes must be NaN, NOT max(vals).
        # The library zero-fills Pseudoorder for genes outside a trajectory's subset, so the old
        # max(vals) returned 0.0 for every 'Other' gene -> a spurious tie-block that DEPRESSES GT's
        # |Spearman| (measured: extract < leading-EV on 7/9 datasets). Return NaN so they are excluded.
        return np.nan

    gt_value = gt.apply(_gene_pseudoorder, axis=1)  # indexed by real var_name

    # Map marker -> GeneTrajectory pseudoorder value via real var_name.
    marker_val = {}
    for g in markers:
        real = marker_to_real.get(g)
        marker_val[g] = float(gt_value.get(real, np.nan)) if real is not None else np.nan
    orders["GeneTrajectory"] = [marker_val.get(str(g), np.nan) for g in orders["gene"]]
    # APPLES-TO-APPLES GT order = the leading gene-diffusion eigenvector emb[:,0] (one global axis),
    # which is the SAME single-eigenvector ordering PRISM Step-(ii) uses (evaluate_supp.py:393). The
    # multi-trajectory extract column above is a DIFFERENT ordering method (native GT output) and is
    # DEPRESSED by the 'Other' 0.0-collapse on 7/9 datasets -- so extract<EV nearly everywhere.
    # CORRECTION 2026-07-02: an earlier comment here wrongly said the 0.0-block INFLATES GT; measured
    # direction is the opposite (it depresses GT). Use GeneTrajectory_EV for the fair head-to-head.
    gene_emb_arr = np.asarray(gene_emb)
    panel_idx = {g: i for i, g in enumerate(panel)}
    ev_val = {}
    for g in markers:
        real = marker_to_real.get(g)
        ev_val[g] = (float(gene_emb_arr[panel_idx[real], 0])
                     if (real is not None and real in panel_idx) else np.nan)
    orders["GeneTrajectory_EV"] = [ev_val.get(str(g), np.nan) for g in orders["gene"]]
    orders.to_csv(oc, index=False)
    print(f"  WROTE GeneTrajectory + GeneTrajectory_EV columns -> {oc}")

    # Quick in-script |Spearman| for sanity
    from scipy.stats import spearmanr
    sub = orders.dropna(subset=["GeneTrajectory"])
    if len(sub) >= 3 and sub["GeneTrajectory"].nunique() > 1:
        rho, _ = spearmanr(sub["GeneTrajectory"], sub["canonical_rank"])
        print(f"  |Spearman(GeneTrajectory, canonical_rank)| = {abs(rho):.3f} "
              f"(n={len(sub)} markers)")
        return {"dataset": ds, "rho_abs": abs(rho), "n": len(sub)}
    print("  (too few non-NaN markers for Spearman)")
    return {"dataset": ds, "rho_abs": float("nan"), "n": len(sub)}


def main() -> int:
    global N_VARIABLE, N_METACELLS
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASET_FILES))
    ap.add_argument("--n-variable", type=int, default=N_VARIABLE)
    ap.add_argument("--n-metacells", type=int, default=N_METACELLS)
    args = ap.parse_args()
    N_VARIABLE = args.n_variable
    N_METACELLS = args.n_metacells
    datasets = [args.dataset] if args.dataset else list(DATASET_FILES)
    results = []
    for ds in datasets:
        try:
            r = run_one(ds)
            if r:
                results.append(r)
        except Exception as e:
            import traceback
            print(f"  ERROR on {ds}: {e!r}")
            traceback.print_exc()
    print("\n=== GeneTrajectory summary (|Spearman rho| vs canonical_rank) ===")
    for r in results:
        print(f"  {r['dataset']:24s} {r['rho_abs']:.3f}  (n={r['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
