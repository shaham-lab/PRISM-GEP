"""Stage A — kNN-over-cells PPMI gene-gene affinity for PRISM-GEP.

Builds neighborhoods with a standard kNN graph in a PCA embedding of cells and
treats each neighborhood as a context window for gene co-occurrence, yielding a
gene-gene PPMI matrix (input to Stage B) and the neighborhood-marginal p(g).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class PPMIResult:
    """Output of Stage A.

    Attributes
    ----------
    ppmi : scipy.sparse.csr_matrix, shape (V, V)
        Gene-gene PPMI affinity matrix. Symmetric, non-negative, zero diagonal.
        Used as the input to Stage B (`prism_lib.graph_utils.diffusion_embedding`).
    p_g : np.ndarray, shape (V,)
        Marginal probability p(g) of seeing gene g in any neighborhood
        co-occurrence event. Sums to 1. Used as the empirical gene frequency
        in Stage C's Bayes inversion: p(g|z) = p(z|g) * p(g) / p(z).

        IMPORTANT: this is the *neighborhood-marginal* p(g), not the raw
        gene-total frequency that ``prism_lib.graph_utils.compute_bio_px``
        computes. The PRISM-GEP paper (p.5) specifies "empirical gene frequency
        computed in previous steps", i.e., from the same co-occurrence stats
        that produced the PPMI matrix.
    gene_names : tuple[str, ...]
        Vocabulary order. Index i in ``ppmi`` and ``p_g`` corresponds to
        ``gene_names[i]``.
    """

    ppmi: sp.csr_matrix
    p_g: np.ndarray
    gene_names: tuple[str, ...]


def compute_gene_gene_ppmi(
    counts: np.ndarray | sp.spmatrix,
    gene_names: list[str] | tuple[str, ...],
    *,
    n_neighbors: int = 15,
    n_pca: int = 50,
    expression_threshold: float = 0.0,
    neighborhood_min_support: int = 1,
    binary_neighborhood_indicator: bool = True,
    eps: float = 1e-10,
    random_state: int = 42,
) -> PPMIResult:
    """Compute the PRISM-GEP kNN-over-cells PPMI gene-gene affinity matrix.

    Parameters
    ----------
    counts : array-like, shape (n_cells, n_genes)
        Raw UMI counts. Dense or sparse.
    gene_names : sequence of str, length n_genes
        Vocabulary in column order of ``counts``.
    n_neighbors : int, default 15
        kNN size for the cell graph (scanpy default).
    n_pca : int, default 50
        Number of PCA components used as the cell embedding for the kNN graph
        (scanpy default).
    expression_threshold : float, default 0.0
        Per-cell expression cutoff before neighborhood aggregation.
        The paper's τ_nhood is not this value; τ_nhood is represented by
        ``neighborhood_min_support`` below.
    neighborhood_min_support : int, default 1
        τ_nhood from Appendix §C.2. For each cell neighborhood, a gene is
        considered present if at least this many cells in the neighborhood
        express the gene after applying ``expression_threshold``.
    binary_neighborhood_indicator : bool, default True
        If True, each neighborhood contributes 1 to C[g1,g2] whenever both
        genes are expressed somewhere in the neighborhood. If False,
        contribution is weighted by the product of summed expression
        magnitudes within the neighborhood.
    eps : float
        Small constant to avoid log(0) in PMI.
    random_state : int
        Seed for PCA / scanpy.

    Returns
    -------
    PPMIResult
        See class docstring.

    Notes
    -----
    Algorithm (paper main text + scanpy/Seurat convention):

    1. Normalize counts: log1p( CP10K ) — standard scRNA-seq preprocessing.
    2. PCA → ``n_pca`` components on cells.
    3. Build kNN graph over cells (k = ``n_neighbors``) in the PCA space.
    4. For each cell c, define ``N(c) = {c} ∪ kNN(c)`` (size k+1).
    5. For each neighborhood, build support counts
       ``Z[n,g] = # cells in N(c) expressing gene g`` and indicator vector
       ``e[g] = 1`` iff ``Z[n,g] >= τ_nhood``.
    6. Co-occurrence count: ``C[g1, g2] += e[g1] * e[g2]`` for every nbhd.
       Zero out the diagonal.
    7. PPMI: ``PPMI[g1,g2] = max(log( P(g1,g2) / (P(g1)*P(g2)) ), 0)``
       where probabilities are normalized over the upper triangle.
    8. p(g) = row marginal of co-occurrence matrix, normalized to sum to 1.
    """
    if sp.issparse(counts):
        counts = counts.toarray()
    counts = np.asarray(counts, dtype=np.float32)
    n_cells, n_genes = counts.shape
    if len(gene_names) != n_genes:
        raise ValueError(
            f"gene_names length {len(gene_names)} does not match counts column count {n_genes}"
        )
    gene_names = tuple(gene_names)

    # CP10K = counts per 10,000, then log1p (standard scanpy normalization).
    cell_totals = counts.sum(axis=1, keepdims=True)
    cell_totals = np.maximum(cell_totals, 1.0)
    cp10k = counts * (1e4 / cell_totals)
    log_cp10k = np.log1p(cp10k)

    # Use sklearn so we don't require scanpy to be installed.
    from sklearn.decomposition import PCA

    n_components = min(n_pca, min(n_cells, n_genes) - 1)
    pca = PCA(n_components=n_components, random_state=random_state)
    cell_embedding = pca.fit_transform(log_cp10k)  # shape (n_cells, n_components)

    from sklearn.neighbors import NearestNeighbors

    # NearestNeighbors returns the cell itself as one neighbor; request k+1 to
    # match scanpy's convention of k neighbors excluding self.
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="auto")
    knn.fit(cell_embedding)
    _, nbhd_indices = knn.kneighbors(cell_embedding)  # shape (n_cells, k+1)

    if neighborhood_min_support < 1:
        raise ValueError("neighborhood_min_support must be >= 1")

    # Neighborhood gene-expression indicator:
    # Z[c, g] = number of cells in nbhd N(c) with count > expression_threshold;
    # E[c, g] = 1 iff Z[c, g] >= neighborhood_min_support.
    # The per-neighborhood gather is (n_cells, k+1, n_genes); done in batches.
    expressed = counts > expression_threshold  # bool, shape (n_cells, n_genes)

    if binary_neighborhood_indicator:
        nbhd_expressed = _batched_neighborhood_support(
            expressed,
            nbhd_indices,
            min_support=neighborhood_min_support,
        )
        # nbhd_expressed: bool, shape (n_cells, n_genes)
        # Co-occurrence counts C = E^T @ E, zero diagonal.
        ne = nbhd_expressed.astype(np.float32)
        co_occ = ne.T @ ne  # shape (V, V), float
        np.fill_diagonal(co_occ, 0.0)
    else:
        # Weighted variant: each neighborhood contributes the outer product of
        # its summed expression vector.
        nbhd_summed = _batched_neighborhood_sum(log_cp10k, nbhd_indices)
        co_occ = nbhd_summed.T @ nbhd_summed
        np.fill_diagonal(co_occ, 0.0)

    # PPMI = max(log(P(g1,g2) / (P(g1) P(g2))), 0), over the co-occurrence matrix.
    total = co_occ.sum()
    if total <= 0:
        raise RuntimeError(
            "Co-occurrence matrix is all zero. Check input counts and "
            "expression_threshold."
        )
    p_wc = co_occ / total
    p_w = p_wc.sum(axis=1)  # marginal over rows
    p_c = p_wc.sum(axis=0)  # marginal over cols
    expected = np.outer(p_w, p_c)

    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((p_wc + eps) / (expected + eps))
    ppmi = np.maximum(pmi, 0.0)
    np.fill_diagonal(ppmi, 0.0)

    # p(g) = row marginal of the co-occurrence matrix, normalized to sum to 1.
    gene_marginal = co_occ.sum(axis=1)
    gene_marginal_total = gene_marginal.sum()
    if gene_marginal_total <= 0:
        raise RuntimeError("Gene marginal is all zero.")
    p_g = gene_marginal / gene_marginal_total

    return PPMIResult(
        ppmi=sp.csr_matrix(ppmi),
        p_g=p_g.astype(np.float64),
        gene_names=gene_names,
    )


def _batched_neighborhood_support(
    expressed: np.ndarray,
    nbhd_indices: np.ndarray,
    *,
    min_support: int,
    batch_size: int = 256,
) -> np.ndarray:
    """For each cell, threshold neighborhood expression support counts.

    Returns a bool array of shape (n_cells, n_genes).
    """
    n_cells, n_genes = expressed.shape
    out = np.zeros((n_cells, n_genes), dtype=bool)
    for start in range(0, n_cells, batch_size):
        stop = min(start + batch_size, n_cells)
        # gather shape (batch, k+1, n_genes), then support count along axis=1
        gathered = expressed[nbhd_indices[start:stop]]  # bool
        out[start:stop] = gathered.sum(axis=1) >= min_support
    return out


def _batched_neighborhood_sum(
    values: np.ndarray,
    nbhd_indices: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """For each row (cell), sum the rows of ``values`` indexed by nbhd_indices."""
    n_cells, n_features = values.shape
    out = np.zeros((n_cells, n_features), dtype=values.dtype)
    for start in range(0, n_cells, batch_size):
        stop = min(start + batch_size, n_cells)
        gathered = values[nbhd_indices[start:stop]]
        out[start:stop] = gathered.sum(axis=1)
    return out


def quick_self_test() -> None:
    """Tiny end-to-end smoke test on a hand-built 4-gene, 6-cell matrix.

    Genes A and B are co-expressed in cells 0,1,2,3 (4 of 6) → high PPMI.
    Genes C and D are co-expressed in cells 4,5 only (2 of 6) → moderate PPMI.
    Genes A and C never co-expressed → 0 in PPMI.
    """
    counts = np.array(
        [
            [3, 2, 0, 0],  # A,B
            [1, 4, 0, 0],  # A,B
            [2, 1, 0, 0],  # A,B
            [4, 3, 0, 0],  # A,B
            [0, 0, 5, 6],  # C,D
            [0, 0, 2, 1],  # C,D
        ],
        dtype=np.float32,
    )
    gene_names = ["A", "B", "C", "D"]
    res = compute_gene_gene_ppmi(
        counts, gene_names, n_neighbors=2, n_pca=2
    )
    print("PPMI matrix:")
    print(res.ppmi.toarray())
    print("p_g:", res.p_g)
    print("gene_names:", res.gene_names)


if __name__ == "__main__":
    quick_self_test()
