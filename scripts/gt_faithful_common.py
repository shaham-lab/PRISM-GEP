"""Shared primitive for FAITHFUL GeneTrajectory-idiom reproductions using GeneTrajectory's OWN algorithms
(`extract_gene_trajectory`, `add_gene_bin_score`) applied to PRISM's gene geometry. This fixes the two fidelity
gaps identified from the paper (PMC11452571):
  1. gene-bin score = *proportion of a bin's genes expressed per cell* (GT's `add_gene_bin_score`), not mean z-expr;
  2. gene trajectories are *extracted* (terminus + random walk on the gene graph via `extract_gene_trajectory`),
     yielding multiple ordered trajectories -- not GMM clusters.

PRISM contributes the gene embedding: `outputs/candidate_screen/<ds>/embedding.npy` is the diffusion map of the
gene-gene co-occurrence (PPMI) affinity (`bio/pipeline.py::diffusion_embedding`) -- the analogue of GT's DM of
gene-gene Wasserstein distances. We feed PRISM's embedding into GT's extraction/scoring code.

Vocabulary alignment: the embedding rows align to the `*_K<K>_alpha.csv.genes.tsv` sidecar (verified by a
ribosomal-gene coherence test; the plain `beta_prism.csv.genes.tsv` for pbmc3k is a DIFFERENT order and is wrong).
"""
from __future__ import annotations
import glob, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform

WS = Path(__file__).resolve().parent.parent


def prism_gene_order(ds: str) -> list[str]:
    """Gene names in the order embedding.npy / ppmi.npz / p_g.npy use.

    CRITICAL: bio/pipeline.py:305 sets `gene_names = list(df.columns)` and builds PPMI + the diffusion
    embedding in that filtered-CSV COLUMN order. The `*.genes.tsv` sidecars are written by write_beta_csv in
    MALLET FIRST-ENCOUNTER order (a different permutation) — using them scrambles the gene->embedding mapping.
    So the ONLY correct source of names for embedding.npy is the filtered-CSV header. (Verified by ribosomal-
    coherence: CSV-column order co-locates ribosomal genes; the sidecar order randomizes them.)"""
    csv = WS / "data" / ds / f"filtered_{ds}_cells_x_genes.csv"
    return pd.read_csv(csv, index_col=0, nrows=0).columns.astype(str).tolist()


def load_prism_gene_space(ds: str, dist_dims: int = 10):
    """-> (embedding[V,20], gene_names[V] (original case, CSV-column order), gene_dist[V,V] Euclidean)."""
    base = WS / "outputs" / "candidate_screen" / ds
    E = np.load(base / "embedding.npy")
    names = prism_gene_order(ds)
    if len(names) != E.shape[0]:
        raise ValueError(f"{ds}: CSV columns ({len(names)}) != embedding rows ({E.shape[0]}); the filtered CSV "
                         f"on disk may differ from the one used to build the embedding.")
    D = squareform(pdist(E[:, :dist_dims]))
    return E, names, D


def extract_prism_trajectories(ds: str, t_list=(3, 3, 3), dims: int = 5, k: int = 10, quantile: float = 0.02):
    """Run GeneTrajectory's extract_gene_trajectory on PRISM's gene embedding.
    t_list = diffusion steps per trajectory (its length = number of trajectories); balanced defaults
    (3,3,3) give ~400-700 genes/trajectory with the rest as 'Other', matching GT's balanced extraction
    (uneven t_list makes one trajectory a catch-all). -> trajectory DataFrame (DM_1..DM_dims, 'selected',
    'Pseudoorder-i').

    Robustness: some PRISM embeddings (e.g. pancreas — a single tight lineage, gene distances all ~0.06)
    make GT's terminus/random-walk extraction return an empty trajectory, which crashes its pseudoorder step.
    We escalate t_list and drop to fewer trajectories before giving up with a clear message."""
    from gene_trajectory.extract_gene_trajectory import extract_gene_trajectory
    E, names, D = load_prism_gene_space(ds)
    n_traj = len(t_list)
    base = int(round(np.mean(t_list)))
    configs = [list(t_list)]
    for step in (base, base + 2, base + 4):                 # escalate diffusion steps
        for nt in (n_traj, n_traj - 1, 2):                  # then fall back to fewer trajectories
            if nt >= 2:
                configs.append([step] * nt)
    last = None
    for cfg in configs:
        try:
            return extract_gene_trajectory(E[:, :dims], D, gene_names=names, t_list=cfg, n=None,
                                           dims=dims, k=k, quantile=quantile)
        except Exception as e:  # noqa: BLE001
            last = e
            continue
    raise RuntimeError(f"{ds}: GeneTrajectory extraction failed for all configs (PRISM gene embedding may be "
                       f"a single tight lineage that does not decompose into multiple trajectories). Last: {last}")


def build_adata(ds: str):
    """AnnData (cells x genes), X = library-normalized log1p, var_names = gene symbols (original case)."""
    import anndata as ad, scanpy as sc
    expr = pd.read_csv(WS / "data" / ds / f"filtered_{ds}_cells_x_genes.csv", index_col=0)
    expr.index = expr.index.astype(str)
    a = ad.AnnData(X=expr.values.astype(np.float32),
                   obs=pd.DataFrame(index=expr.index),
                   var=pd.DataFrame(index=expr.columns.astype(str)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    return a


def add_bin_scores(adata, traj: pd.DataFrame, n_bins: int = 5, n_traj: int = 3):
    """GT's add_gene_bin_score (proportion of bin genes expressed per cell). Harmonizes gene-name case between
    the trajectory index (embedding vocab) and adata.var (expression vocab). Writes obs columns in place;
    returns the list of (trajectory_index, [bin_obs_columns])."""
    from gene_trajectory.add_gene_bin_score import add_gene_bin_score
    # derive the true number of extracted trajectories (extract_gene_trajectory may early-stop) so we never
    # ask add_gene_bin_score for a Pseudoorder column that does not exist.
    n_present = sum(1 for c in traj.columns if str(c).startswith("Pseudoorder"))
    n_traj = min(n_traj, n_present)
    # harmonize case: lower-case both sides so the join hits
    var_map = {v.lower(): v for v in adata.var_names}
    tl = traj.copy()
    tl.index = [str(g).lower() for g in tl.index]
    keep = [g for g in tl.index if g in var_map]
    tl = tl.loc[keep]
    a2 = adata[:, [var_map[g] for g in keep]].copy()
    a2.var_names = keep
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        add_gene_bin_score(a2, tl, n_bins=n_bins, trajectories=n_traj, prefix="Trajectory")
    bin_cols = [c for c in a2.obs.columns if c.startswith("Trajectory")]
    # copy scores back onto a cell-indexed frame
    scores = a2.obs[bin_cols].copy()
    scores.index = adata.obs_names
    return scores, bin_cols
