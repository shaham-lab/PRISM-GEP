"""Stage F — within-GEP diffusion-based gene ordering for PRISM-GEP."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class GeneOrdering:
    """Result of within-GEP ordering.

    Attributes
    ----------
    ordered_genes : tuple[str, ...]
        Top-N genes of the GEP, reordered by 1D diffusion pseudotime.
    pseudotime : np.ndarray, shape (N,)
        The 1D coordinate (2nd diffusion component) corresponding to
        ``ordered_genes``. Monotone-increasing.
    """

    ordered_genes: tuple[str, ...]
    pseudotime: np.ndarray


def order_genes_in_gep(
    top_genes: list[str] | tuple[str, ...],
    ppmi: sp.csr_matrix,
    gene_to_idx: dict[str, int],
    *,
    n_diffusion_components: int = 2,
) -> GeneOrdering:
    """Order the top-N genes of a GEP along a 1D diffusion pseudotime.

    Parameters
    ----------
    top_genes : sequence of str
        The top-N genes of one GEP (typically N=20), as produced by
        ``bio.extract_top_genes``.
    ppmi : scipy.sparse.csr_matrix, shape (V, V)
        The full gene-gene PPMI matrix from Stage A. Used as the source of the
        within-GEP similarity kernel.
    gene_to_idx : dict[str, int]
        Maps gene name to its row/column index in ``ppmi``.
    n_diffusion_components : int, default 2
        How many diffusion components to compute. We use the 2nd
        (first non-trivial) as the 1D ordering coordinate.

    Returns
    -------
    GeneOrdering
        Genes reordered by pseudotime.

    Notes
    -----
    v0 algorithm (paper main text + Qu et al. 2025 procedure; **swap on Supp S2**):

    1. Subset the PPMI matrix to the rows/cols indexed by ``top_genes``.
    2. Apply diffusion-map embedding to the sub-matrix.
    3. Take the 2nd component (first non-trivial) as the 1D pseudotime.
    4. Sort genes by pseudotime ascending.
    """
    if len(top_genes) < 3:
        raise ValueError(
            f"Need at least 3 genes for diffusion ordering, got {len(top_genes)}"
        )
    indices = np.array([gene_to_idx[g] for g in top_genes], dtype=np.int64)

    # Subset PPMI to the GEP's top genes.
    sub = ppmi[indices, :][:, indices]
    sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub)

    # Symmetrize and zero diagonal (defensive).
    sub = 0.5 * (sub + sub.T)
    np.fill_diagonal(sub, 0.0)

    # Diffusion map: P = D^{-1} W; eigendecompose P; take 2nd eigenvector.
    row_sums = sub.sum(axis=1)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    P = sub / row_sums[:, None]

    eigenvalues, eigenvectors = np.linalg.eig(P)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sort_idx]

    # Skip trivial first eigenvector (constant for stochastic matrix).
    if n_diffusion_components < 2:
        n_diffusion_components = 2
    pseudotime = eigenvectors[:, 1]

    # Sort genes ascending by pseudotime; the sign of the eigenvector is
    # arbitrary, so caller may want to flip if needed for visualization.
    order = np.argsort(pseudotime)
    ordered_genes = tuple(top_genes[i] for i in order)

    return GeneOrdering(
        ordered_genes=ordered_genes,
        pseudotime=pseudotime[order],
    )
