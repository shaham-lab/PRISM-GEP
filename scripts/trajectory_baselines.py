"""Trajectory validation, generalized across datasets.

For each dataset that has an ordinal ground-truth axis (developmental stage
or lineage rank), compute cell pseudotime from several methods and score
|Spearman rho| vs the ground-truth rank.

Datasets currently supported (see DATASET_SPECS):
  * pancreas               -- coarse-lineage rank (5 stages)
  * gastrulation_erythroid -- developmental stage E7.0 .. E8.5 (7 stages)
  * gastrulation           -- developmental stage E6.5 .. E8.5 (9 stages)

Methods:
  * PRISM_GEP_PHATE1   -- PHATE-1 of K=5 GEP attributions
  * PRISM_GEP_dominant -- dominant-GEP id mapped to its mean lineage rank
  * PCA_1              -- 1st PC of log1p(CP10K) on HVG (naive baseline)
  * DPT                -- scanpy diffusion pseudotime (root = rank-0 centroid)
  * PAGA_DPT           -- DPT through PAGA's coarse-cluster graph

Outputs (under outputs/trajectory/<dataset>/):
  * <ds>_trajectory_scores.json
  * <ds>_trajectory_panel.pdf
  * <ds>_pseudotimes.csv

Usage:
  python workspace/scripts/trajectory_baselines.py --datasets pancreas gastrulation_erythroid
  python workspace/scripts/trajectory_baselines.py --datasets all
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS))


# --- Per-dataset specs ----------------------------------------------------

class DatasetSpec:
    """How to load a dataset's cells + the ordinal ground-truth axis."""
    def __init__(self, name, h5ad=None, csv=None, label_csv=None,
                 label_col=None, lineage_order=None, root_cluster=None):
        self.name = name
        self.h5ad = h5ad             # optional AnnData
        self.csv = csv               # filtered counts CSV (fallback / for PRISM align)
        self.label_csv = label_csv   # cell_id -> cell_type csv (fallback)
        self.label_col = label_col   # column name in adata.obs to use
        self.lineage_order = lineage_order   # list of labels in rank order
        self.root_cluster = root_cluster      # for DPT / PAGA-DPT

DATASET_SPECS = {
    "pancreas": DatasetSpec(
        "pancreas",
        h5ad=WS / "data" / "pancreas" / "endocrinogenesis_day15.h5ad",
        csv=WS / "data" / "pancreas" / "filtered_pancreas_cells_x_genes.csv",
        label_col="clusters_coarse",
        lineage_order=["Ductal", "Ngn3 low EP", "Ngn3 high EP",
                       "Pre-endocrine", "Endocrine"],
        root_cluster="Ductal",
    ),
    "gastrulation_erythroid": DatasetSpec(
        "gastrulation_erythroid",
        csv=WS / "data" / "gastrulation_erythroid"
            / "filtered_gastrulation_erythroid_cells_x_genes.csv",
        label_csv=WS / "data" / "gastrulation_erythroid"
            / "cell_type_labels.csv",
        lineage_order=["E7.0", "E7.25", "E7.5", "E7.75", "E8.0", "E8.25", "E8.5"],
        root_cluster="E7.0",
    ),
    "gastrulation": DatasetSpec(
        "gastrulation",
        csv=WS / "data" / "gastrulation"
            / "filtered_gastrulation_cells_x_genes.csv",
        label_csv=WS / "data" / "gastrulation"
            / "cell_type_labels.csv",
        lineage_order=["E6.5", "E6.75", "E7.0", "E7.25", "E7.5",
                       "E7.75", "E8.0", "E8.25", "E8.5"],
        root_cluster="E6.5",
    ),
    # Added 2026-07-19 so the Slingshot cell axis exists for the Bonemarrow cascade
    # heatmap, which pairs with the main-text Figure 5(b) lineage. The lineage order is
    # taken verbatim from traj_cell_all_datasets.ORDERINGS["bonemarrow"], the same
    # definition the 14-dataset cell-trajectory table scores against, so the two
    # analyses cannot disagree about what the ground-truth ordering is.
    "bonemarrow": DatasetSpec(
        "bonemarrow",
        csv=WS / "data" / "bonemarrow"
            / "filtered_bonemarrow_cells_x_genes.csv",
        label_csv=WS / "data" / "bonemarrow" / "cell_type_labels.csv",
        lineage_order=["HSC_1", "Ery_1", "Ery_2"],
        root_cluster="HSC_1",
    ),
}


# --- Loading --------------------------------------------------------------

def load_dataset(spec: DatasetSpec):
    """Return (adata, rank, cell_ids_in_order).

    `adata` always has a `lineage` obs column with the canonical label,
    and `rank` is an int array of the corresponding ordinal rank.
    """
    import anndata as ad
    import scipy.sparse as sp

    rank_map = {n: i for i, n in enumerate(spec.lineage_order)}

    if spec.h5ad and spec.h5ad.exists():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            adata = ad.read_h5ad(spec.h5ad)
        # use the requested column
        if spec.label_col not in adata.obs.columns:
            raise KeyError(f"{spec.label_col} not in adata.obs of {spec.h5ad}")
        adata.obs["lineage"] = adata.obs[spec.label_col].astype(str)
    else:
        if spec.csv is None or not spec.csv.exists():
            raise FileNotFoundError(spec.csv)
        df = pd.read_csv(spec.csv, index_col=0)
        # Build AnnData from CSV (raw counts, HVG-filtered).
        adata = ad.AnnData(X=sp.csr_matrix(df.values.astype(np.float32)))
        adata.obs_names = df.index.astype(str)
        adata.var_names = df.columns.astype(str)
        if spec.label_csv is None or not spec.label_csv.exists():
            raise FileNotFoundError(spec.label_csv)
        ldf = pd.read_csv(spec.label_csv)
        id_to_lab = dict(zip(ldf["cell_id"].astype(str),
                             ldf["cell_type"].astype(str)))
        adata.obs["lineage"] = pd.Series(
            [id_to_lab.get(c, None) for c in adata.obs_names],
            index=adata.obs_names,
        )

    keep = adata.obs["lineage"].isin(rank_map)
    adata = adata[keep.values].copy()
    rank = adata.obs["lineage"].map(rank_map).astype(int).values
    return adata, rank


def load_prism_doc_topics(spec: DatasetSpec, adata,
                          layout: str = "seed0") -> np.ndarray | None:
    """Pull doc_topics.txt from outputs/<ds>/<layout>; align to adata cells."""
    seed_dir = WS / "outputs" / spec.name / layout
    dt = seed_dir / "doc_topics.txt"
    if not dt.exists() or spec.csv is None or not spec.csv.exists():
        return None
    df = pd.read_csv(dt, sep="\t", header=None)
    csv_ids = pd.read_csv(spec.csv, index_col=0).index.astype(str).tolist()
    probs = df.iloc[:, 2:].values
    if probs.shape[0] != len(csv_ids):
        print(f"  warn: doc_topics rows={probs.shape[0]} vs csv rows={len(csv_ids)}")
        return None
    id_to_prob = dict(zip(csv_ids, probs))
    aligned = []
    for cid in adata.obs_names.astype(str):
        if cid in id_to_prob:
            aligned.append(id_to_prob[cid])
        else:
            aligned.append(np.full(probs.shape[1], np.nan))
    return np.array(aligned)


# --- Methods --------------------------------------------------------------

def phate_first_coord(P, random_state=42):
    import phate
    op = phate.PHATE(n_components=2, random_state=random_state, verbose=0, n_jobs=1)
    return op.fit_transform(P)[:, 0]


def pca_first_coord(X):
    from sklearn.decomposition import PCA
    return PCA(n_components=1, random_state=42).fit_transform(X)[:, 0]


def _scanpy_prep(adata, n_hvg=2000):
    import scanpy as sc
    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, flavor="seurat",
                                subset=True)
    sc.pp.pca(a, n_comps=30)
    sc.pp.neighbors(a, n_neighbors=15)
    return a


def dpt_pseudotime(adata, root_cluster):
    import scanpy as sc
    a = _scanpy_prep(adata)
    sc.tl.diffmap(a)
    root_idx = np.where(a.obs["lineage"] == root_cluster)[0][0]
    a.uns["iroot"] = int(root_idx)
    sc.tl.dpt(a)
    return a.obs["dpt_pseudotime"].values


def paga_dpt_pseudotime(adata, root_cluster):
    import scanpy as sc
    a = _scanpy_prep(adata)
    sc.tl.paga(a, groups="lineage")
    sc.pl.paga(a, show=False, plot=False)
    sc.tl.draw_graph(a, init_pos="paga")
    root_idx = np.where(a.obs["lineage"] == root_cluster)[0][0]
    a.uns["iroot"] = int(root_idx)
    sc.tl.dpt(a)
    return a.obs["dpt_pseudotime"].values


def slingshot_pseudotime(adata, root_cluster):
    """Slingshot via pyslingshot (Python port of the R package).

    Slingshot fits a tree of principal curves through the cluster centroids
    in a low-dim embedding, then projects each cell onto its lineage to get
    pseudotime. Standard cell-trajectory tool.
    """
    import scanpy as sc
    from pyslingshot import Slingshot
    a = _scanpy_prep(adata)
    # Slingshot needs a 2D embedding for the principal-curves step
    sc.tl.umap(a)
    # Cluster index for the root
    clusters = a.obs["lineage"].astype(str).values
    uniq = list(pd.unique(clusters))
    # Encode cluster as integer 0..C-1; start_node = index of root_cluster
    if root_cluster not in uniq:
        raise ValueError(f"root cluster {root_cluster!r} not in {uniq}")
    cluster_to_int = {c: i for i, c in enumerate(uniq)}
    start_node = cluster_to_int[root_cluster]
    a.obs["lineage_int"] = pd.Categorical(
        [cluster_to_int[c] for c in clusters],
        categories=list(range(len(uniq))), ordered=True)
    sling = Slingshot(a, celltype_key="lineage_int", obsm_key="X_umap",
                      start_node=start_node)
    sling.fit(num_epochs=10)
    return sling.unified_pseudotime


def spearman_abs(pt, rank):
    valid = np.isfinite(pt)
    if valid.sum() < 10:
        return float("nan")
    rho, _ = spearmanr(pt[valid], rank[valid])
    return float(abs(rho))


# --- One-dataset driver ---------------------------------------------------

def _add_prism(pseudotimes, adata, rank, spec, layout, tag):
    print(f"  [PRISM/{layout}] PHATE-1 + dominant ...")
    dt = load_prism_doc_topics(spec, adata, layout=layout)
    if dt is None:
        print(f"    (no doc_topics at {layout}; skipping)")
        return
    finite = ~np.isnan(dt).any(axis=1)
    pt = np.full(adata.n_obs, np.nan)
    pt[finite] = phate_first_coord(dt[finite])
    pseudotimes[f"PRISM_{tag}_PHATE1"] = pt
    dom = np.argmax(dt[finite], axis=1)
    gep_to_meanrank = {g: rank[finite][dom == g].mean()
                       for g in np.unique(dom)}
    sorted_geps = sorted(gep_to_meanrank.items(), key=lambda x: x[1])
    gep_to_pt = {g: i for i, (g, _) in enumerate(sorted_geps)}
    pt2 = np.full(adata.n_obs, np.nan)
    pt2[finite] = np.array([gep_to_pt[g] for g in dom])
    pseudotimes[f"PRISM_{tag}_dominant"] = pt2


def run_one(spec: DatasetSpec, out_dir: Path, *, prism_layouts: list[tuple[str, str]]):
    """prism_layouts: list of (layout_subdir, tag) to evaluate (e.g.
    [("seed0", "K5"), ("K8/seed0", "K8")])."""
    print(f"\n========== {spec.name} ==========")
    adata, rank = load_dataset(spec)
    print(f"  {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  rank counts: {dict(pd.Series(rank).value_counts().sort_index())}")

    pseudotimes = {}
    for layout, tag in prism_layouts:
        _add_prism(pseudotimes, adata, rank, spec, layout, tag)

    # PCA-1
    print(f"  [3] PCA-1 ...")
    import scanpy as sc
    a3 = adata.copy()
    sc.pp.normalize_total(a3, target_sum=1e4)
    sc.pp.log1p(a3)
    X = a3.X.toarray() if hasattr(a3.X, "toarray") else np.asarray(a3.X)
    pseudotimes["PCA_1"] = pca_first_coord(X)

    # DPT
    print(f"  [4] DPT (root={spec.root_cluster}) ...")
    try:
        pseudotimes["DPT"] = dpt_pseudotime(adata, spec.root_cluster)
    except Exception as e:
        print(f"    DPT failed: {e}")

    # PAGA-DPT
    print(f"  [5] PAGA-DPT (root={spec.root_cluster}) ...")
    try:
        pseudotimes["PAGA_DPT"] = paga_dpt_pseudotime(adata, spec.root_cluster)
    except Exception as e:
        print(f"    PAGA-DPT failed: {e}")

    # Slingshot
    print(f"  [6] Slingshot (root={spec.root_cluster}) ...")
    try:
        pseudotimes["Slingshot"] = slingshot_pseudotime(adata, spec.root_cluster)
    except Exception as e:
        print(f"    Slingshot failed: {e}")
        import traceback; traceback.print_exc()

    scores = {name: spearman_abs(pt, rank) for name, pt in pseudotimes.items()}
    print(f"\n  === {spec.name}: |Spearman rho| ===")
    for name, s in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"    {name:30s}  {s:.4f}")

    df = pd.DataFrame({"cell_id": adata.obs_names.astype(str),
                       "lineage_rank": rank,
                       "lineage": adata.obs["lineage"].values,
                       **pseudotimes})
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{spec.name}_pseudotimes.csv", index=False)
    (out_dir / f"{spec.name}_trajectory_scores.json").write_text(
        json.dumps({"spearman_abs_vs_rank": scores,
                    "n_cells": int(adata.n_obs),
                    "lineage_order": spec.lineage_order}, indent=2))

    plot_panel(spec, adata, pseudotimes, scores,
               out_dir / f"{spec.name}_trajectory_panel.pdf")
    return scores


# --- Plot -----------------------------------------------------------------

def plot_panel(spec, adata, pseudotimes, scores, out_path: Path):
    import scanpy as sc
    a = _scanpy_prep(adata)
    sc.tl.umap(a)
    coords = a.obsm["X_umap"]

    method_names = list(pseudotimes.keys())
    n_methods = len(method_names)
    # Grid sized to fit all methods + 1 bar chart.
    n_cells = n_methods + 1
    ncols = 3
    nrows = (n_cells + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 5.25))
    axes_flat = axes.flatten() if nrows > 1 else axes

    for i, name in enumerate(method_names):
        ax = axes_flat[i]
        pt = pseudotimes[name]
        valid = np.isfinite(pt)
        sc_obj = ax.scatter(coords[valid, 0], coords[valid, 1],
                            c=pt[valid], cmap="viridis", s=4, alpha=0.8)
        rho = scores.get(name, float("nan"))
        ax.set_title(f"{name}\n|Spearman ρ|={rho:.3f}", fontsize=10)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        plt.colorbar(sc_obj, ax=ax, label="pseudotime")

    ax = axes_flat[n_methods]
    # Hide remaining unused axes
    for j in range(n_methods + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    ordered = sorted(scores.items(), key=lambda x: -x[1])
    names = [n for n, _ in ordered]
    vals = [s for _, s in ordered]
    bars = ax.barh(names, vals,
                   color=["#1f77b4" if n.startswith("PRISM") else "#ff7f0e"
                          for n in names])
    ax.set_xlabel("|Spearman ρ| vs lineage rank")
    ax.set_title(f"Trajectory recovery — {spec.name}")
    for b, v in zip(bars, vals):
        ax.text(v + 0.005, b.get_y() + b.get_height()/2, f"{v:.3f}",
                va="center", fontsize=9)
    ax.set_xlim(0, max(vals) * 1.15 if vals else 1.0)

    fig.suptitle(f"Cell-pseudotime trajectory recovery on {spec.name}\n"
                 f"({len(spec.lineage_order)} ordinal stages: "
                 f"{spec.lineage_order[0]} → {spec.lineage_order[-1]})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# --- Driver ---------------------------------------------------------------

# Per-dataset K=#types map (mirrors run_k_eq_types.K_TARGETS)
K_TYPES = {"pancreas": 8, "gastrulation_erythroid": 7, "gastrulation": 9}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=list(DATASET_SPECS.keys()))
    p.add_argument("--also-k-types", action="store_true",
                   help="Also evaluate PRISM at K=#published-cell-types "
                        "(reads outputs/<ds>/K<n>/seed0/doc_topics.txt).")
    args = p.parse_args()
    if args.datasets == ["all"]:
        args.datasets = list(DATASET_SPECS.keys())

    all_scores = {}
    for ds in args.datasets:
        if ds not in DATASET_SPECS:
            print(f"[{ds}] no spec -- SKIP")
            continue
        spec = DATASET_SPECS[ds]
        out = WS / "outputs" / "trajectory" / ds
        # Always include K=5 (the existing seed0).
        layouts = [("seed0", "K5")]
        if args.also_k_types and ds in K_TYPES:
            n = K_TYPES[ds]
            ktypes_root = WS / "outputs" / ds / f"K{n}"
            # Include every available seed directory (K{n}/seed*) so we can
            # report mean ± std across PRISM seeds at K=#types.
            seed_dirs = sorted([p for p in ktypes_root.glob("seed*")
                                 if (p / "doc_topics.txt").exists()])
            if not seed_dirs:
                print(f"[{ds}] K{n}/seed*/doc_topics.txt absent -- skip K{n}")
            for sd in seed_dirs:
                tag = f"K{n}_{sd.name}"  # e.g. K8_seed0, K8_seed1
                layouts.append((f"K{n}/{sd.name}", tag))
        try:
            all_scores[ds] = run_one(spec, out, prism_layouts=layouts)
        except Exception as e:
            print(f"[{ds}] FAILED: {e}")
            import traceback; traceback.print_exc()

    if len(all_scores) > 1:
        print("\n\n========== CROSS-DATASET SUMMARY ==========")
        # one row per method, one column per dataset
        all_methods = sorted({m for s in all_scores.values() for m in s})
        rows = []
        for m in all_methods:
            row = {"method": m}
            for ds, s in all_scores.items():
                row[ds] = s.get(m, np.nan)
            rows.append(row)
        df = pd.DataFrame(rows)
        df["mean"] = df[list(all_scores)].mean(axis=1, skipna=True)
        df = df.sort_values("mean", ascending=False).reset_index(drop=True)
        out_csv = WS / "outputs" / "trajectory" / "all_trajectory_scores.csv"
        df.to_csv(out_csv, index=False)
        print(df.to_string(index=False))
        print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
