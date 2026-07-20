"""Cell-neighborhood alternatives for the Stage-A graph.

The PRISM-GEP pipeline currently defines a neighborhood per cell as
{c} ∪ kNN(c) on a PCA(50) embedding. The Stage-A PPMI then aggregates
gene co-occurrence over these neighborhoods.

This module exposes a single interface that returns the neighborhood
membership matrix H ∈ {0,1}^(M × N) where M is the number of neighborhoods
and N the number of cells. H[m, c] = 1 iff cell c belongs to neighborhood
m. The rest of the pipeline (PPMI co-occurrence → diffusion → GMM → MOM
→ MALLET) is run unchanged with the chosen H.

Methods:
  - "kNN"       : the existing PRISM default (M = N, square)
  - "Metacells" : aggregate cells into compact metacells via the
                  metacells package (Baran et al. 2019). M << N.
  - "SEACells"  : sparse cell archetypes (Persad et al. 2023). M << N.
  - "Milo"      : (placeholder) overlapping neighborhoods via Milo's
                  k-NN graph. Not yet implemented here; left as a stub.

The standard convention is: row-stochastic? NO — H is BINARY and may have
arbitrary row sums (size of the neighborhood). Stage A reads H, builds
Z[m, g] = #cells in nbhd m that express gene g, and PPMI from there.
"""
from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import scipy.sparse as sp


METHODS = Literal["kNN", "Metacells", "SEACells", "Milo", "MiloLike"]


def build_H(
    counts: np.ndarray | sp.spmatrix,
    method: METHODS = "kNN",
    *,
    n_neighbors: int = 15,
    n_pca: int = 50,
    random_state: int = 42,
    metacell_size_target: int = 50,
    seacells_n: int | None = None,
    milo_prop: float = 0.1,
) -> sp.csr_matrix:
    """Return the neighborhood-cell membership matrix H ∈ {0,1}^(M × N).

    Parameters
    ----------
    counts : (N, V) raw UMI counts. Sparse or dense.
    method : which neighborhood algorithm.
    n_neighbors, n_pca : passed to scanpy's PCA + kNN graph for kNN method.
    metacell_size_target : approximate target metacell size for Metacells.
    seacells_n : number of archetypes for SEACells. Default = max(50, N/50).

    Returns
    -------
    H : sparse (M, N) binary matrix.
    """
    if method == "kNN":
        return _build_H_knn(counts, n_neighbors=n_neighbors, n_pca=n_pca,
                            random_state=random_state)
    if method == "Metacells":
        return _build_H_metacells(counts, metacell_size_target=metacell_size_target,
                                  random_state=random_state)
    if method == "SEACells":
        return _build_H_seacells(counts, n_archetypes=seacells_n,
                                 n_neighbors=n_neighbors, n_pca=n_pca,
                                 random_state=random_state)
    if method == "MiloLike":
        return _build_H_milo_like(counts, n_neighbors=n_neighbors, n_pca=n_pca,
                                  prop=milo_prop, random_state=random_state)
    raise ValueError(f"unknown method {method!r}; choose from {METHODS.__args__}")


def _build_H_milo_like(counts, *, n_neighbors=15, n_pca=50, prop=0.1,
                        random_state=42) -> sp.csr_matrix:
    """Milo-like overlapping-kNN-ball surrogate.

    Milo (Dann et al. 2022) builds neighborhoods by sampling index cells at
    proportion `prop` (default 0.1) and using each index cell's kNN ball as a
    neighborhood for differential-abundance testing. We mirror that
    construction and emit a binary H matrix the rest of Stage A can ingest
    unchanged: H rows = sampled index cells, columns = all N cells, H[m,n]=1
    iff cell n is in the m-th index cell's kNN ball (including itself).

    This is NOT a full Milo benchmark — Milo's downstream is DA testing, not
    PPMI aggregation — but it does swap Stage A's "every cell is its own
    neighborhood" for Milo's "overlapping sampled-index-cell neighborhoods",
    isolating the geometry that Milo introduces.
    """
    import scanpy as sc
    import anndata as ad
    if sp.issparse(counts):
        X = counts.toarray()
    else:
        X = np.asarray(counts)
    a = ad.AnnData(X=X.astype(np.float32))
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.pca(a, n_comps=min(n_pca, min(a.shape) - 1), random_state=random_state)
    sc.pp.neighbors(a, n_neighbors=n_neighbors, random_state=random_state)
    conn = a.obsp["connectivities"].tocsr()  # (N, N)
    N = conn.shape[0]
    # Sample index cells uniformly at proportion `prop`
    rng = np.random.default_rng(random_state)
    n_index = max(2, int(round(prop * N)))
    idx = rng.choice(N, size=n_index, replace=False)
    # Build H: rows = sampled index cells; their kNN ball + self
    rows, cols = [], []
    for m, i in enumerate(idx):
        nbrs = conn[i].nonzero()[1].tolist()
        nbrs.append(int(i))
        for n in set(nbrs):
            rows.append(m)
            cols.append(n)
    data = np.ones(len(rows), dtype=np.int8)
    H = sp.csr_matrix((data, (rows, cols)), shape=(n_index, N), dtype=np.int8)
    return H



def _build_H_knn(counts, *, n_neighbors=15, n_pca=50, random_state=42) -> sp.csr_matrix:
    """{c} ∪ kNN(c) over PCA(n_pca) on log1p(CP10K). One neighborhood per cell."""
    import scanpy as sc
    import anndata as ad
    if sp.issparse(counts):
        X = counts.toarray()
    else:
        X = np.asarray(counts)
    a = ad.AnnData(X=X.astype(np.float32))
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.pca(a, n_comps=min(n_pca, min(a.shape) - 1), random_state=random_state)
    sc.pp.neighbors(a, n_neighbors=n_neighbors, random_state=random_state)
    conn = a.obsp["connectivities"]  # (N, N) sparse, kNN graph
    # Make symmetric binary: H[i, j] = 1 if i==j OR j is a neighbor of i.
    H = (conn > 0).astype(np.int8)
    # Add self-loops
    N = H.shape[0]
    diag = sp.eye(N, dtype=np.int8, format="csr")
    H = (H + diag).astype(bool).astype(np.int8)  # collapse to binary
    return H.tocsr()


def _build_H_metacells(counts, *, metacell_size_target=50, random_state=42
                        ) -> sp.csr_matrix:
    """Aggregate cells into metacells via the metacells package.

    Each metacell is a row in H; cells assigned to it have H[m, c]=1.
    Falls back to a fast k-means proxy if the metacells package is missing,
    so the pipeline still runs end-to-end.
    """
    if sp.issparse(counts):
        X = counts.toarray().astype(np.float32)
    else:
        X = np.asarray(counts, dtype=np.float32)
    N = X.shape[0]
    n_meta = max(2, N // metacell_size_target)

    try:
        # metacells 0.9.x:
        #   - X must be float32 (downsample_cells asserts it),
        #   - pipeline lives under .pipeline (not .tl),
        #   - requires set_name + top_level + the var/obs masks
        #     (lateral_gene, noisy_gene, excluded_gene, excluded_cell) to exist
        #     even if empty, otherwise tl.combine_masks raises KeyError.
        import metacells.pipeline as mc_pl
        import metacells.utilities as mc_ut
        import anndata as ad
        Xfloat = X.astype(np.float32)
        adata = ad.AnnData(X=sp.csr_matrix(Xfloat))
        n_cells, n_genes = Xfloat.shape
        adata.var_names = [f"g{i}" for i in range(n_genes)]
        adata.obs_names = [f"c{i}" for i in range(n_cells)]
        mc_ut.set_name(adata, "prism")
        mc_ut.top_level(adata)
        adata.var["lateral_gene"] = np.zeros(n_genes, dtype=bool)
        adata.var["noisy_gene"] = np.zeros(n_genes, dtype=bool)
        adata.var["excluded_gene"] = np.zeros(n_genes, dtype=bool)
        adata.obs["excluded_cell"] = np.zeros(n_cells, dtype=bool)
        median_umis = float(np.median(Xfloat.sum(axis=1)))
        target_umis = max(1000, int(round(median_umis * metacell_size_target)))
        mc_pl.divide_and_conquer_pipeline(
            adata,
            target_metacell_size=metacell_size_target,
            target_metacell_umis=target_umis,
            random_seed=int(random_state),
            quick_and_dirty=True,
        )
        labels = np.asarray(adata.obs["metacell"].values, dtype=np.int64)
        # Outliers come back as -1; give each its own singleton bin so we
        # don't collapse heterogeneous outliers into a single fake metacell.
        if (labels < 0).any():
            n_outliers = int((labels < 0).sum())
            new_ids = labels.max() + 1 + np.arange(n_outliers)
            labels = labels.copy()
            labels[labels < 0] = new_ids
    except Exception as e:
        if __import__("os").environ.get("PRISM_REQUIRE_REAL"):
            raise RuntimeError(
                f"REAL metacells required (PRISM_REQUIRE_REAL set) but it failed: {e}. "
                f"Install the matching version (pip install metacells==0.9.5) and rerun; "
                f"refusing to silently fall back to a KMeans proxy.") from e
        warnings.warn(f"metacells package unavailable or failed ({e}); "
                      f"falling back to MiniBatchKMeans({n_meta}) proxy.")
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.decomposition import PCA
        totals = np.maximum(X.sum(axis=1, keepdims=True), 1.0)
        cp10k = X * (1e4 / totals)
        logx = np.log1p(cp10k)
        pcs = PCA(n_components=min(50, min(logx.shape) - 1),
                  random_state=random_state).fit_transform(logx)
        labels = MiniBatchKMeans(n_clusters=n_meta, random_state=random_state,
                                 n_init=10, batch_size=256).fit_predict(pcs)

    rows = labels.astype(np.int64)
    cols = np.arange(N, dtype=np.int64)
    data = np.ones(N, dtype=np.int8)
    H = sp.csr_matrix((data, (rows, cols)), shape=(int(labels.max()) + 1, N))
    return H


def compute_ppmi_from_H(
    counts: np.ndarray | sp.spmatrix,
    gene_names: list[str] | tuple[str, ...],
    H: sp.csr_matrix,
    *,
    expression_threshold: float = 2.0,
    neighborhood_min_support: int = 1,
    eps: float = 1e-10,
):
    """Compute the kNN-over-cells PPMI affinity matrix using an arbitrary
    H ∈ {0,1}^(M × N) neighborhood-cell membership matrix.

    This is the "neighborhood-agnostic" version of
    bio.ppmi_knn_over_cells.compute_gene_gene_ppmi -- swap H to ablate the
    neighborhood definition (kNN vs Metacells vs SEACells vs Milo).
    """
    from bio.ppmi_knn_over_cells import PPMIResult
    if sp.issparse(counts):
        counts = counts.toarray()
    counts = np.asarray(counts, dtype=np.float32)
    n_cells, n_genes = counts.shape
    if len(gene_names) != n_genes:
        raise ValueError("gene_names length mismatch")
    if H.shape[1] != n_cells:
        raise ValueError(f"H has {H.shape[1]} cols, expected {n_cells} cells")

    # Per-cell binary expression indicator after threshold.
    expressed = (counts > expression_threshold).astype(np.float32)  # (N, V)

    # Z[m, g] = #cells in nbhd m where gene g is expressed
    # H is (M, N), expressed is (N, V) -> sparse @ dense -> dense (M, V).
    Z = H @ expressed  # (M, V) float
    # E[m, g] = 1 if Z[m, g] >= tau
    E = (Z >= neighborhood_min_support).astype(np.float32)

    # Co-occurrence over neighborhoods: C[g1, g2] = sum_m E[m, g1] * E[m, g2]
    co_occ = E.T @ E  # (V, V)
    np.fill_diagonal(co_occ, 0.0)

    total = co_occ.sum()
    if total <= 0:
        raise RuntimeError("Co-occurrence is all zero. Check threshold / tau.")
    p_wc = co_occ / total
    p_w = p_wc.sum(axis=1)
    p_c = p_wc.sum(axis=0)
    expected = np.outer(p_w, p_c)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((p_wc + eps) / (expected + eps))
    ppmi = np.maximum(pmi, 0.0)
    np.fill_diagonal(ppmi, 0.0)

    gene_marginal = co_occ.sum(axis=1)
    p_g = gene_marginal / max(gene_marginal.sum(), 1e-12)

    return PPMIResult(ppmi=sp.csr_matrix(ppmi),
                      p_g=p_g.astype(np.float64),
                      gene_names=tuple(gene_names))


def _build_H_seacells(counts, *, n_archetypes=None, n_neighbors=15, n_pca=50,
                      random_state=42) -> sp.csr_matrix:
    """SEACells archetypes. Falls back to a k-means proxy if package missing."""
    if sp.issparse(counts):
        X = counts.toarray().astype(np.float32)
    else:
        X = np.asarray(counts, dtype=np.float32)
    N = X.shape[0]
    K = n_archetypes or max(50, N // 50)

    try:
        import SEACells                              # noqa: F401
        import anndata as ad
        import scanpy as sc
        adata = ad.AnnData(X=X)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.pca(adata, n_comps=min(n_pca, min(adata.shape) - 1),
                  random_state=random_state)
        model = SEACells.core.SEACells(adata, build_kernel_on="X_pca",
                                       n_SEACells=K, n_waypoint_eigs=10,
                                       convergence_epsilon=1e-5)
        model.construct_kernel_matrix()
        model.initialize_archetypes()
        model.fit(max_iter=50)
        A = model.A_                                  # soft membership
        # Different SEACells releases shape A as (N, K) or (K, N); also may
        # return a sparse matrix. Coerce to dense and pick the axis whose
        # length matches the cell count so labels always have length N.
        if sp.issparse(A):
            A = A.toarray()
        A = np.asarray(A)
        if A.shape[0] == N:           # (N, K)
            labels = A.argmax(axis=1)
        elif A.shape[1] == N:         # (K, N)
            labels = A.argmax(axis=0)
        else:
            raise RuntimeError(
                f"SEACells A_ shape {A.shape} doesn't contain N={N} on either axis"
            )
        labels = np.asarray(labels).ravel()
    except Exception as e:
        if __import__("os").environ.get("PRISM_REQUIRE_REAL"):
            raise RuntimeError(
                f"REAL SEACells required (PRISM_REQUIRE_REAL set) but it failed: {e}. "
                f"Fix the dependency (often pip install ipywidgets) and rerun; "
                f"refusing to silently fall back to a KMeans proxy.") from e
        warnings.warn(f"SEACells unavailable or failed ({e}); falling back "
                      f"to MiniBatchKMeans({K}) proxy.")
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.decomposition import PCA
        totals = np.maximum(X.sum(axis=1, keepdims=True), 1.0)
        cp10k = X * (1e4 / totals)
        logx = np.log1p(cp10k)
        pcs = PCA(n_components=min(n_pca, min(logx.shape) - 1),
                  random_state=random_state).fit_transform(logx)
        labels = MiniBatchKMeans(n_clusters=K, random_state=random_state,
                                 n_init=10, batch_size=256).fit_predict(pcs)

    rows = labels.astype(np.int64)
    cols = np.arange(N, dtype=np.int64)
    data = np.ones(N, dtype=np.int8)
    H = sp.csr_matrix((data, (rows, cols)), shape=(int(labels.max()) + 1, N))
    return H
