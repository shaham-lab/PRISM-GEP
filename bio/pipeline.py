"""End-to-end Stages A→D orchestrator for PRISM-GEP.

Produces the Dirichlet β prior CSV that gets passed to the patched MALLET via
``--beta-file``: a single CSV row of V comma-separated floats, no header, no index.

Pipeline:
    A. kNN-over-cells PPMI       → bio.ppmi_knn_over_cells
    B. Diffusion-map embedding   → implemented inline
    C. GMM soft clustering       → sklearn.GaussianMixture
       + Bayes inversion         → uses neighborhood-marginal p(g) from Stage A
       + soft_predictions smoothing
    D. Method-of-moments β̂      → prism_lib.methods_of_moments.dirichlet_moments

Usage:
    python -m bio.pipeline --dataset breast_cancer
    python -m bio.pipeline --dataset pbmc3k --K 5 --m 20

The script expects:
    data/<dataset>/filtered_<dataset>_cells_x_genes.csv

And writes:
    outputs/<dataset>/{ppmi.npz, p_g.npy, embedding.npy,
                       prior_KxV.npy, beta_prism.csv}
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigs
from sklearn.mixture import GaussianMixture

# Make ``prism_lib`` importable regardless of cwd.
_HERE = Path(__file__).resolve().parent
_WORKSPACE = _HERE.parent
sys.path.insert(0, str(_WORKSPACE))

from bio.ppmi_knn_over_cells import compute_gene_gene_ppmi, PPMIResult  # noqa: E402
from prism_lib.methods_of_moments import dirichlet_moments  # noqa: E402


def first_encounter_order(counts: np.ndarray, gene_names: list[str]) -> list[int]:
    """Column indices ordered as MALLET assigns type ids (first-encounter).

    Genes that never appear in any cell never enter MALLET's alphabet; they are
    excluded so the result length equals MALLET's numTypes (V).
    """
    int_counts = np.rint(np.maximum(np.asarray(counts), 0)).astype(np.int64)
    seen = np.zeros(len(gene_names), dtype=bool)
    order: list[int] = []
    for i in range(int_counts.shape[0]):
        for j in np.nonzero(int_counts[i])[0]:
            if not seen[j]:
                seen[j] = True
                order.append(int(j))
    n_unseen = int((~seen).sum())
    if n_unseen:
        print(f"[beta] WARNING: {n_unseen} gene(s) never expressed -> excluded "
              f"from beta (they are absent from MALLET's alphabet).")
    return order


def reorder_beta_to_first_encounter(
    beta: np.ndarray, counts: np.ndarray, gene_names: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Permute a column-ordered beta into MALLET first-encounter order.

    Returns (beta_reordered, genes_in_alphabet_order).
    """
    order = first_encounter_order(counts, gene_names)
    return np.asarray(beta)[order], [gene_names[j] for j in order]


def write_beta_csv(
    beta_csv: Path, beta: np.ndarray, counts: np.ndarray, gene_names: list[str]
) -> Path:
    """Write beta_prism.csv in MALLET first-encounter order + labeled sidecar."""
    beta_re, genes_re = reorder_beta_to_first_encounter(beta, counts, gene_names)
    beta_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(beta_csv, [beta_re], delimiter=",", fmt="%.6f")
    side = beta_csv.with_suffix(beta_csv.suffix + ".genes.tsv")
    with side.open("w", encoding="utf-8") as fh:
        fh.write("type_id\tgene\tbeta\n")
        for i, (g, bv) in enumerate(zip(genes_re, beta_re)):
            fh.write(f"{i}\t{g}\t{bv:.6f}\n")
    return beta_csv


def soc_affinity(matrix: sp.spmatrix | np.ndarray) -> np.ndarray:
    """Second-order cosine affinity, faithful to text get_cosine_similarity.

    Equivalent to ``sklearn.metrics.pairwise.cosine_similarity(matrix)``: each
    ROW of ``matrix`` (a gene's PPMI context vector) is L2-normalized, then the
    Gram matrix of the normalized rows is returned. Rows with zero norm map to
    zero vectors (cosine 0 with everything, including a 0 self-similarity), which
    is exactly sklearn's behavior.

    Returns a dense (V, V) ndarray. No thresholding is applied (matches text).
    """
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(matrix)


def diffusion_embedding(
    matrix: sp.spmatrix | np.ndarray, m: int = 20
) -> np.ndarray:
    """Diffusion-map embedding of a similarity matrix.

    Parameters
    ----------
    matrix : (V, V) similarity matrix (sparse or dense)
    m : int, default 20
        Number of diffusion components (PRISM-GEP paper, p.5: "empirically m=20").

    Returns
    -------
    np.ndarray, shape (V, m)
        Diffusion-map embeddings: top-m non-trivial eigenvectors of P=D⁻¹W,
        scaled by their corresponding eigenvalues.
    """
    W = csr_matrix(matrix)
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    D_inv = diags(np.where(row_sums != 0, 1.0 / row_sums, 0))
    P = D_inv @ W

    # eigs needs k+1 to skip the trivial eigenvector with eigenvalue 1.
    eigenvalues, eigenvectors = eigs(P, k=m + 1, which="LR")
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Skip trivial first eigenvector, take next m, scale by eigenvalues.
    top_eigenvalues = eigenvalues[1 : m + 1]
    top_eigenvectors = eigenvectors[:, 1 : m + 1]
    scaled = top_eigenvectors * top_eigenvalues
    return scaled


def gmm_soft_cluster(
    embeddings: np.ndarray, K: int, *, random_state: int = 42,
    n_init: int = 1, covariance_type: str = "full",
) -> tuple[np.ndarray, np.ndarray]:
    """GMM soft clustering over the gene diffusion embedding.

    Defaults are the published settings (n_init=1, covariance="full").
    ``n_init``/``covariance_type`` are exposed so the parameter search can test
    a more robust fit (a single random init can land in a poor local optimum,
    which propagates to a poor beta prior).

    Returns
    -------
    p_z_given_g : (V, K) — posterior responsibilities
    p_z         : (K,)   — mixture weights
    """
    gmm = GaussianMixture(
        n_components=K,
        random_state=random_state,
        n_init=n_init,
        covariance_type=covariance_type,
    )
    gmm.fit(embeddings)
    p_z_given_g = gmm.predict_proba(embeddings)  # (V, K)
    return p_z_given_g, gmm.weights_


def bayes_invert(
    p_z_given_g: np.ndarray, p_g: np.ndarray, p_z: np.ndarray
) -> np.ndarray:
    """Bayes' rule: p(g|z) = p(z|g) * p(g) / p(z).

    Per PRISM-GEP paper p.5, p(g) is the empirical gene frequency from the
    *neighborhood co-occurrence* statistics (Stage A), NOT raw gene-total
    frequency.

    Returns
    -------
    p_g_given_z : (V, K) — each column is a multinomial over genes for one GEP.
    """
    # p(g|z) = p(z|g) * p(g) / p(z)
    # shapes: (V,K) * (V,1) / (1,K) → (V,K)
    return (p_z_given_g * p_g[:, None]) / p_z[None, :]


def soft_predictions(probs: np.ndarray, eps: float = 1e-6, axis: int = 0) -> np.ndarray:
    """Add eps and renormalize along ``axis``.

    axis=0 (default, our shipped "bio convention"): per-COLUMN — each column = one
    GEP's multinomial over genes.
    axis=1: per-ROW — each gene's K topic-probs sum to 1. Used by the
    ``use_row_norm`` path in ``run_pipeline``.
    """
    smoothed = probs + eps
    smoothed = smoothed / smoothed.sum(axis=axis, keepdims=True)
    return smoothed



def run_pipeline(
    counts_csv: Path,
    output_dir: Path,
    *,
    K: int = 5,
    m: int = 20,
    n_neighbors: int = 15,
    n_pca: int = 50,
    expression_threshold: float = 0.0,
    neighborhood_min_support: int = 1,
    binary_neighborhood_indicator: bool = True,
    use_soc: bool = False,
    use_row_norm: bool = False,
    random_state: int = 42,
) -> Path:
    """Run Stages A through D end-to-end. Returns path to written β CSV.

    Parameters
    ----------
    counts_csv : Path
        Path to the cell×gene CSV (e.g., from data/<dataset>/).
        Format: first column is cell ID, remaining columns are gene names,
        rows are cells, values are raw UMI counts.
    output_dir : Path
        Where to write artifacts.
    use_soc : bool, default False
        If True, apply the published-PRISM second-order-cosine transform
        (``soc_affinity`` == text ``get_cosine_similarity``) to the PPMI matrix
        BEFORE the diffusion embedding. Default False keeps the existing scRNA
        bio path (raw PPMI -> diffusion) byte-identical.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {counts_csv}")
    df = pd.read_csv(counts_csv, index_col=0)
    counts = df.values
    gene_names = list(df.columns)
    print(f"       cells={counts.shape[0]}, genes={counts.shape[1]}")

    print(f"[A] PPMI (n_neighbors={n_neighbors}, n_pca={n_pca}, "
          f"thr={expression_threshold}, "
          f"tau_nhood={neighborhood_min_support}, "
          f"binary={binary_neighborhood_indicator}) ...")
    ppmi_result: PPMIResult = compute_gene_gene_ppmi(
        counts,
        gene_names,
        n_neighbors=n_neighbors,
        n_pca=n_pca,
        expression_threshold=expression_threshold,
        neighborhood_min_support=neighborhood_min_support,
        binary_neighborhood_indicator=binary_neighborhood_indicator,
        random_state=random_state,
    )
    sp.save_npz(output_dir / "ppmi.npz", ppmi_result.ppmi)
    np.save(output_dir / "p_g.npy", ppmi_result.p_g)

    # --use-soc routes PPMI rows through cosine similarity before the diffusion
    # embedding. Off by default, which passes raw PPMI straight through.
    if use_soc:
        print("[A.5] SOC: cosine_similarity(PPMI) before diffusion ...")
        affinity = soc_affinity(ppmi_result.ppmi)
    else:
        affinity = ppmi_result.ppmi

    print(f"[B] diffusion embedding (m={m}, use_soc={use_soc}) ...")
    embedding = diffusion_embedding(affinity, m=m)
    np.save(output_dir / "embedding.npy", embedding)
    print(f"    embedding shape: {embedding.shape}")

    print(f"[C] GMM (K={K}) + Bayes inversion ...")
    p_z_given_g, p_z = gmm_soft_cluster(embedding, K=K, random_state=random_state)
    p_g_given_z = bayes_invert(p_z_given_g, ppmi_result.p_g, p_z)
    if use_row_norm:
        # per-ROW softpred over the K topics per gene (axis=1), then transpose to
        # (K,V) and re-row-normalize before the method-of-moments fit.
        print("[C] use_row_norm=True -> per-ROW softpred (axis=1) + pre-MoM row renorm")
        p_g_given_z = soft_predictions(p_g_given_z, axis=1)
        multinomial_rows = p_g_given_z.T  # (K, V)
        multinomial_rows = multinomial_rows / multinomial_rows.sum(axis=1, keepdims=True)
    else:
        p_g_given_z = soft_predictions(p_g_given_z)  # per-COLUMN (shipped bio convention)
        # Each column of p_g_given_z is one GEP's distribution over genes (sum=1).
        # dirichlet_moments expects (K, V) where each ROW is a multinomial.
        multinomial_rows = p_g_given_z.T  # (K, V)
    np.save(output_dir / "prior_KxV.npy", multinomial_rows)

    print("[D] method-of-moments beta-hat ...")
    beta = dirichlet_moments(multinomial_rows)
    print(f"    beta shape: {beta.shape}, min={beta.min():.6f}, "
          f"max={beta.max():.6f}, mean={beta.mean():.6f}")

    # MALLET consumes beta positionally by type id, not by gene name, so the
    # vector MUST be in first-encounter order, not data-CSV column order.
    beta_csv = output_dir / "beta_prism.csv"
    write_beta_csv(beta_csv, beta, counts, gene_names)
    print(f"[done] wrote {beta_csv} (MALLET first-encounter order + .genes.tsv sidecar)")
    return beta_csv



DATASETS = {
    "breast_cancer": "filtered_breast_cancer_cells_x_genes.csv",
    "pbmc3k": "filtered_pbmc3k_cells_x_genes.csv",
    "zeisel_brain": "filtered_zeisel_brain_cells_x_genes.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="PRISM-GEP Stages A-D pipeline")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name. Any name is accepted; the counts matrix is "
                             "looked up at data/<dataset>/filtered_<dataset>_cells_x_genes.csv")
    parser.add_argument("--K", type=int, default=5,
                        help="Number of GEPs/topics (paper: 5 for all bio datasets)")
    parser.add_argument("--m", type=int, default=20,
                        help="Diffusion components (paper: 20)")
    parser.add_argument("--n_neighbors", type=int, default=15,
                        help="kNN size for cell graph (Supp S1; default scanpy=15)")
    parser.add_argument("--n_pca", type=int, default=50,
                        help="PCA components for cell embedding (default scanpy=50)")
    parser.add_argument("--expression_threshold", type=float, default=2.0,
                        help="Per-cell count cutoff for the neighborhood indicator "
                             "(production value: 2.0)")
    parser.add_argument("--neighborhood_min_support", type=int, default=1,
                        help="Minimum neighborhood support (production value: 1)")
    parser.add_argument("--use-soc", dest="use_soc", action="store_true",
                        help="Apply published-PRISM second-order-cosine "
                             "transform to PPMI before diffusion (default off).")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--workspace", type=Path, default=_WORKSPACE)
    args = parser.parse_args()

    fname = DATASETS.get(args.dataset, f"filtered_{args.dataset}_cells_x_genes.csv")
    counts_csv = args.workspace / "data" / args.dataset / fname
    output_dir = args.workspace / "outputs" / args.dataset
    if not counts_csv.exists():
        raise FileNotFoundError(counts_csv)
    run_pipeline(
        counts_csv=counts_csv,
        output_dir=output_dir,
        K=args.K,
        m=args.m,
        n_neighbors=args.n_neighbors,
        n_pca=args.n_pca,
        expression_threshold=args.expression_threshold,
        neighborhood_min_support=args.neighborhood_min_support,
        use_soc=args.use_soc,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
