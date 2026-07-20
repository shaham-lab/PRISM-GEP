"""UMAP visualization of cells colored by GEP attribution.

Produces 3 plots per dataset:
  1. UMAP on the GEP-attribution vector (cells x K) — color by dominant GEP
  2. UMAP on log1p(expression) (sanity baseline) — color by dominant GEP
  3. UMAP on log1p(expression) — color by Leiden clustering on the same expression PCA

Also computes ARI/NMI between {dominant GEP} and {Leiden labels} as quantitative
agreement between PRISM-GEP-derived cell groups and standard scRNA-seq clustering.

Usage:
    python -m bio.cell_clustering --dataset breast_cancer --layout seed0
"""
from __future__ import annotations

import argparse
import json
import math as _math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# numba is used to JIT the on-the-fly Jensen-Shannon distance callable so it can
# be passed to umap.UMAP(metric=...) at O(N·k) memory (no N×N matrix). If numba
# is unavailable we fall back to a plain Python function — correct but slower;
# UMAP still accepts a non-jitted callable.
try:  # pragma: no cover - import guard
    from numba import njit as _njit

    def _numba_njit_fastmath(fn):
        return _njit(fastmath=True)(fn)
except Exception:  # pragma: no cover
    def _numba_njit_fastmath(fn):
        return fn

_math_log = _math.log
_math_sqrt = _math.sqrt

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS))



# Optional reference table; absent from a fresh clone, in which case the
# Leiden-comparison annotation is simply omitted (see _lookup_leiden_reference).
_LEIDEN_REF_CSV = WS / "results" / "leiden_vs_prism_ari.csv"


def _lookup_leiden_reference(dataset: str, *, target_k: int):
    """Return (ARI(Leiden at k = target_k, published), achieved_k) for a
    Leiden run at matched k, or (None, None) if the CSV / row is missing.
    """
    if not _LEIDEN_REF_CSV.exists():
        return None, None
    try:
        ref = pd.read_csv(_LEIDEN_REF_CSV)
    except Exception:
        return None, None
    sub = ref[(ref["dataset"] == dataset) & (ref["method"] == "Leiden")
              & (ref["target_k"] == target_k)]
    if sub.empty:
        return None, None
    row = sub.iloc[0]
    return float(row["ARI_vs_published"]), int(row["achieved_k"])

DATASET_FILES = {
    "breast_cancer":          "filtered_breast_cancer_cells_x_genes.csv",
    "pbmc3k":                 "filtered_pbmc3k_cells_x_genes.csv",
    "zeisel_brain":           "filtered_zeisel_brain_cells_x_genes.csv",
    "pancreas":               "filtered_pancreas_cells_x_genes.csv",
    "bonemarrow":             "filtered_bonemarrow_cells_x_genes.csv",
    "hemogenic_endothelium":  "filtered_hemogenic_endothelium_cells_x_genes.csv",
    "gastrulation":           "filtered_gastrulation_cells_x_genes.csv",
    "gastrulation_e75":       "filtered_gastrulation_e75_cells_x_genes.csv",
    "gastrulation_erythroid": "filtered_gastrulation_erythroid_cells_x_genes.csv",
}


def load_doc_topics(seed_dir: Path) -> np.ndarray | None:
    dt = seed_dir / "doc_topics.txt"
    if not dt.exists():
        return None
    df = pd.read_csv(dt, sep="\t", header=None)
    return df.iloc[:, 2:].values  # (n_cells, K)


def umap_2d(X: np.ndarray, *, random_state: int = 42, n_neighbors: int = 15,
            metric: str = "euclidean") -> np.ndarray:
    """Compute 2D UMAP embedding. `metric` can be any string UMAP accepts or
    'precomputed' if X is already a square distance matrix."""
    try:
        import umap
    except ImportError as e:
        raise ImportError("`pip install umap-learn` required") from e
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2,
                        random_state=random_state, metric=metric)
    return reducer.fit_transform(X)


def phate_2d(X: np.ndarray, *, random_state: int = 42, knn: int = 15) -> np.ndarray:
    """PHATE 2D embedding — designed for trajectory/simplex-like data."""
    import phate
    op = phate.PHATE(n_components=2, knn=knn, random_state=random_state, verbose=0,
                     n_jobs=1)
    return op.fit_transform(X)


def jsd_distance_matrix(P: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Pairwise Jensen-Shannon distance on rows of P (cells × K simplex).

    Fully vectorised via the closed form JSD = H(M) - 0.5(H(P)+H(Q)).
    Caller is responsible for subsampling when N is large — this function
    builds an (N, N, K) intermediate.

    DEPRECATED for large N: this O(N²K) materialised matrix OOMs on the
    ≥7k-cell datasets. Prefer ``jsd_metric`` (the numba callable below) passed
    to ``umap.UMAP(metric=jsd_metric)``, which is O(N·k) memory and scales.
    """
    Pn = np.clip(P, eps, None).astype(np.float64)
    Pn /= Pn.sum(axis=1, keepdims=True)
    HP = -(Pn * np.log2(Pn)).sum(axis=1)                       # (N,)
    M = 0.5 * (Pn[:, None, :] + Pn[None, :, :])                # (N, N, K)
    HM = -np.where(M > 0, M * np.log2(np.clip(M, eps, None)), 0.0).sum(axis=2)
    JSD = HM - 0.5 * (HP[:, None] + HP[None, :])
    JSD = np.clip(JSD, 0.0, 1.0)
    D = np.sqrt(JSD)
    np.fill_diagonal(D, 0.0)
    return D


# 1/ln(2): convert natural log -> log2. We use math.log (natural) inside the
# numba kernel because numba's np.log2 ufunc trips a float32/float64 type
# mismatch on the float32 arrays UMAP hands to a metric callable.
_INV_LN2 = 1.4426950408889634


@_numba_njit_fastmath
def jsd_metric(p, q):  # noqa: ANN001
    """Jensen-Shannon DISTANCE = sqrt(JSD) between two probability vectors
    (base-2 logs), computed termwise in O(k) memory with no N×N matrix.

    Pass directly to ``umap.UMAP(metric=jsd_metric)``. This is the scalable
    replacement for the precomputed ``jsd_distance_matrix`` path, which OOMs
    on the ≥7k-cell datasets. Closed form per pair:

        JSD = H(M) - 0.5(H(p)+H(q)),   M = 0.5(p+q)

    accumulated termwise as  -m·log2 m + 0.5 p·log2 p + 0.5 q·log2 q,
    then clamped to [0,1] and square-rooted.
    """
    s = 0.0
    for i in range(p.shape[0]):
        pi = p[i]
        qi = q[i]
        mi = 0.5 * (pi + qi)
        if mi > 0.0:
            s -= mi * _math_log(mi) * _INV_LN2
        if pi > 0.0:
            s += 0.5 * pi * _math_log(pi) * _INV_LN2
        if qi > 0.0:
            s += 0.5 * qi * _math_log(qi) * _INV_LN2
    if s < 0.0:
        s = 0.0
    if s > 1.0:
        s = 1.0
    return _math_sqrt(s)


def mahalanobis_VI(P: np.ndarray, *, reg: float = 1e-6) -> np.ndarray:
    """Regularized pseudo-inverse VI for GLOBAL Mahalanobis on the K-simplex.

    The K-component GEP simplex has rank K-1 (rows sum to 1), so its covariance
    is SINGULAR — a plain inverse blows up. We add ``reg · (tr Σ / K) · I`` and
    take the Moore-Penrose pseudo-inverse, giving a well-conditioned VI for
    ``umap.UMAP(metric="mahalanobis", metric_kwds={"VI": VI})``.

    NOTE (honesty): global Mahalanobis is exactly Euclidean distance after a
    fixed linear whitening (VI = LᵀL). It is "global-whitened euclidean", not a
    fundamentally new geometry — only the per-axis/correlation rescaling changes
    the UMAP neighbor graph.
    """
    cov = np.cov(np.asarray(P, dtype=np.float64), rowvar=False)
    K = cov.shape[0]
    eps = reg * (np.trace(cov) / K)
    return np.linalg.pinv(cov + eps * np.eye(K)).astype(np.float64)


def local_mahalanobis_whiten(P: np.ndarray, *, n_neighbors: int = 15,
                             reg: float = 1e-3):
    """Per-point LOCAL-Mahalanobis whitening (O(N·k) memory).

    A true per-pair local-Mahalanobis distance d(i,j) = sqrt((xᵢ-xⱼ)ᵀ Σ_i⁻¹
    (xᵢ-xⱼ)) is asymmetric and is NOT a metric, so it cannot be a UMAP metric
    callable. The standard manifold-metric workaround is to whiten each point
    by its own local-neighborhood covariance and run plain Euclidean UMAP on
    the whitened cloud. For each point i: estimate Σ_i over its kNN, regularize
    toward the global scale (``reg · tr Σ_global / K · I``) to tame the rank-4
    simplex deficiency, and apply Σ_i^{-1/2}.

    Returns ``(P_whitened, diag)`` where ``diag`` records honest stability
    failures: how many local covariances were singular, the worst condition
    number, and whether any whitened coords went non-finite.
    """
    from sklearn.neighbors import NearestNeighbors
    X = np.asarray(P, dtype=np.float64)
    N, K = X.shape
    k = min(n_neighbors, N - 1)
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    _, nbr = nn.kneighbors(X)
    glob_scale = np.trace(np.cov(X, rowvar=False)) / K
    Pw = np.empty_like(X)
    n_singular = 0
    max_cond = 0.0
    for i in range(N):
        c = np.cov(X[nbr[i]], rowvar=False) + reg * glob_scale * np.eye(K)
        w, V = np.linalg.eigh(c)
        if w[0] <= 1e-10:
            n_singular += 1
        cond = w[-1] / max(w[0], 1e-30)
        if cond > max_cond:
            max_cond = cond
        w = np.clip(w, 1e-12, None)
        Wmat = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
        Pw[i] = Wmat @ X[i]
    diag = {
        "n_points": int(N),
        "n_local_cov_singular": int(n_singular),
        "frac_local_cov_singular": float(n_singular) / N,
        "max_condition_number": float(max_cond),
        "any_nonfinite_whitened": bool(not np.isfinite(Pw).all()),
        "reg": float(reg),
        "n_neighbors": int(k),
    }
    return Pw, diag


def _tsne_2d(X: np.ndarray, *, random_state: int = 42) -> np.ndarray:
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, random_state=random_state, init="pca",
                learning_rate="auto", perplexity=min(30, max(5, X.shape[0] // 100))
                ).fit_transform(X)


def _expr_pca(expr: pd.DataFrame, *, n_pca: int = 50, random_state: int = 42,
              idx: np.ndarray | None = None) -> np.ndarray:
    """log1p(CP10K) + PCA. Returns (subset_)cells × n_pca matrix."""
    from sklearn.decomposition import PCA
    X = expr.values
    if idx is not None:
        X = X[idx]
    totals = np.maximum(X.sum(axis=1, keepdims=True), 1.0)
    cp10k = X * (1e4 / totals)
    logx = np.log1p(cp10k)
    n = min(n_pca, min(logx.shape) - 1)
    return PCA(n_components=n, random_state=random_state).fit_transform(logx)


def make_gep_embedding_figure(
    doc_topics: np.ndarray,
    out_path: Path,
    *,
    dataset: str,
    layout: str,
    expr: pd.DataFrame | None = None,
    published_labels: np.ndarray | None = None,
    published_label_names: list[str] | None = None,
    random_state: int = 42,
):
    """3-panel comparison of UMAP / PHATE / UMAP-on-JSD embeddings of the
    K-dim GEP attribution vectors. If published labels are provided, points
    are colored by them; else by dominant GEP."""
    K = doc_topics.shape[1]
    N = doc_topics.shape[0]
    # Subsample so JSD (N,N,K) stays in memory AND expression-PCA stays fast.
    SUBSAMPLE = 6000
    if N > SUBSAMPLE:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, SUBSAMPLE, replace=False)
        doc_topics_used = doc_topics[idx]
        if published_labels is not None:
            published_labels = published_labels[idx]
        print(f"  [embed] subsampled {N} -> {SUBSAMPLE} cells for viz")
    else:
        idx = None
        doc_topics_used = doc_topics
    dominant_gep = doc_topics_used.argmax(axis=1)

    # ----- Honest agreement score: ARI(dominant-GEP, published) on these cells.
    # PRISM-GEP is a soft decomposition, so this argmax projection is a lossy
    # view -- but we report it straight so the figure cannot oversell.
    ari_pub = None
    if published_labels is not None:
        try:
            from sklearn.metrics import adjusted_rand_score
            valid = published_labels >= 0
            if valid.sum() > 1:
                ari_pub = float(adjusted_rand_score(
                    dominant_gep[valid], published_labels[valid]))
        except Exception as e:
            print(f"    ARI(argmax, published) failed ({e})")

    print(f"  [embed] row1 UMAP on doc_topics ({doc_topics_used.shape}) ...")
    r1_umap = umap_2d(doc_topics_used, random_state=random_state)
    print(f"  [embed] row1 PHATE on doc_topics ...")
    try:
        r1_phate = phate_2d(doc_topics_used, random_state=random_state)
    except Exception as e:
        print(f"    PHATE-on-GEP failed ({e})")
        r1_phate = None
    print(f"  [embed] row1 UMAP on JSD distance matrix ...")
    try:
        D = jsd_distance_matrix(doc_topics_used)
        r1_jsd = umap_2d(D, random_state=random_state, metric="precomputed")
    except Exception as e:
        print(f"    UMAP-on-JSD failed ({e})")
        r1_jsd = None

    r2_umap = r2_phate = r2_tsne = None
    if expr is not None:
        try:
            print(f"  [embed] row2 PCA(50) on log1p(CP10K) expression ...")
            pcs = _expr_pca(expr, n_pca=50, random_state=random_state, idx=idx)
            print(f"  [embed] row2 UMAP on expression PCs ({pcs.shape}) ...")
            r2_umap = umap_2d(pcs, random_state=random_state)
            print(f"  [embed] row2 PHATE on expression PCs ...")
            try:
                r2_phate = phate_2d(pcs, random_state=random_state)
            except Exception as e:
                print(f"    PHATE-on-expr failed ({e})")
            print(f"  [embed] row2 t-SNE on expression PCs ...")
            try:
                r2_tsne = _tsne_2d(pcs, random_state=random_state)
            except Exception as e:
                print(f"    t-SNE-on-expr failed ({e})")
        except Exception as e:
            print(f"    Expression-space embeddings failed ({e})")

    if published_labels is not None and published_label_names is not None:
        n_lab = len(published_label_names)
        palette = [plt.get_cmap("tab20")(i % 20) for i in range(n_lab)]
        color_by = published_labels
        labels = published_label_names
        legend_title = "published label"
    else:
        palette = [plt.get_cmap("tab10")(i % 10) for i in range(K)]
        color_by = dominant_gep
        labels = [f"GEP {g}" for g in range(K)]
        legend_title = "dominant GEP"

    row1 = [("UMAP", r1_umap), ("PHATE", r1_phate), ("UMAP on JSD", r1_jsd)]
    row2 = [("UMAP", r2_umap), ("PHATE", r2_phate), ("t-SNE", r2_tsne)]

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 11))

    colour_word = ("published cell type" if published_labels is not None
                   else "dominant GEP")

    def _plot(ax, name, emb, row_label):
        if emb is None:
            ax.set_visible(False)
            return
        for li, lname in enumerate(labels):
            mask = (color_by == li)
            if not mask.any():
                continue
            ax.scatter(emb[mask, 0], emb[mask, 1], s=4, alpha=0.7,
                       color=palette[li], label=lname[:18])
        ax.set_title(f"{name} {row_label}\ncolour = {colour_word}", fontsize=9.5)
        ax.set_xlabel(f"{name}-1"); ax.set_ylabel(f"{name}-2")

    for j, (name, emb) in enumerate(row1):
        _plot(axes[0, j], name, emb, f"of {K}-dim GEP attribution")
    for j, (name, emb) in enumerate(row2):
        _plot(axes[1, j], name, emb, "of expression PCA(50) [reference]")

    # Left-margin band labels so it is unambiguous which row is which space.
    axes[0, 0].annotate(
        "TOP ROW\nPRISM-GEP\nattribution\nspace", xy=(-0.30, 0.5),
        xycoords="axes fraction", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#21618c", rotation=90)
    axes[1, 0].annotate(
        "BOTTOM ROW\nraw-expression\nreference", xy=(-0.30, 0.5),
        xycoords="axes fraction", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#7d6608", rotation=90)

    # One shared legend on the right side of the figure.
    handles, lbls = axes[0, 0].get_legend_handles_labels()
    if handles:
        leg_title = legend_title
        if ari_pub is not None:
            leg_title = f"{legend_title}\nARI(GEP argmax, published) = {ari_pub:.2f}"
        fig.legend(handles, lbls, loc="center right", fontsize=7,
                   markerscale=2, title=leg_title, title_fontsize=8,
                   bbox_to_anchor=(1.0, 0.5))

    fig.tight_layout(rect=(0.03, 0.0, 0.90, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def leiden_clusters(expr: pd.DataFrame, *, n_pca: int = 50, n_neighbors: int = 15,
                    resolution: float = 0.5) -> np.ndarray:
    """Run Leiden clustering on log1p(CP10K) expression via scanpy."""
    import scanpy as sc
    import anndata as ad
    adata = ad.AnnData(X=expr.values.astype(np.float32))
    adata.var_names = list(expr.columns)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=min(n_pca, min(adata.shape) - 1))
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    sc.tl.leiden(adata, resolution=resolution)
    return adata.obs["leiden"].astype(int).values


def make_umap_figure(
    expr: pd.DataFrame,
    doc_topics: np.ndarray,
    out_path: Path,
    *,
    dataset: str,
    layout: str,
    metrics_path: Path | None = None,
    random_state: int = 42,
):
    K = doc_topics.shape[1]
    dominant_gep = doc_topics.argmax(axis=1)
    print(f"  computing UMAP on doc_topics ({doc_topics.shape}) ...")
    umap_dt = umap_2d(doc_topics, random_state=random_state)

    print(f"  computing UMAP on log1p(CP10K) expression ...")
    cell_totals = expr.values.sum(axis=1, keepdims=True)
    cell_totals = np.maximum(cell_totals, 1.0)
    cp10k = expr.values * (1e4 / cell_totals)
    log_cp10k = np.log1p(cp10k)
    # PCA first to keep UMAP fast
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(50, min(log_cp10k.shape) - 1), random_state=random_state)
    pcs = pca.fit_transform(log_cp10k)
    umap_expr = umap_2d(pcs, random_state=random_state)

    print(f"  Leiden clustering ...")
    try:
        leiden = leiden_clusters(expr)
    except Exception as e:
        print(f"  (leiden failed: {e})")
        leiden = None

    # Compute ARI/NMI vs Leiden AND vs published cell-type labels (if available)
    metrics = {}
    if leiden is not None:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        metrics["dominant_gep_n_clusters"] = int(K)
        metrics["leiden_n_clusters"] = int(len(set(leiden)))
        metrics["ari_dominant_gep_vs_leiden"] = float(
            adjusted_rand_score(dominant_gep, leiden))
        metrics["nmi_dominant_gep_vs_leiden"] = float(
            normalized_mutual_info_score(dominant_gep, leiden))

    # Also compare against published cell-type labels if available
    pub_labels_csv = WS / "data" / dataset / "cell_type_labels.csv"
    pub_labels = None
    if pub_labels_csv.exists():
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        try:
            pub_df = pd.read_csv(pub_labels_csv)
            pub_id_to_type = dict(zip(pub_df["cell_id"].astype(str),
                                       pub_df["cell_type"].astype(str)))
            our_cell_ids = expr.index.astype(str).tolist()
            mapped = []
            mapped_dom = []
            for i, cid in enumerate(our_cell_ids):
                if cid in pub_id_to_type:
                    mapped.append(pub_id_to_type[cid])
                    mapped_dom.append(int(dominant_gep[i]))
            if mapped:
                # Encode label strings as integers
                label_to_int = {l: i for i, l in enumerate(sorted(set(mapped)))}
                int_labels = np.array([label_to_int[l] for l in mapped])
                int_dom = np.array(mapped_dom)
                pub_labels = (int_labels, int_dom, list(label_to_int.keys()), our_cell_ids)
                n_overlap = len(mapped)
                metrics["published_n_cells_overlap"] = int(n_overlap)
                metrics["published_n_cell_types"] = int(len(label_to_int))
                metrics["ari_dominant_gep_vs_published"] = float(
                    adjusted_rand_score(int_dom, int_labels))
                metrics["nmi_dominant_gep_vs_published"] = float(
                    normalized_mutual_info_score(int_dom, int_labels))
                print(f"  published-labels overlap: {n_overlap}/{len(our_cell_ids)} cells, "
                      f"{len(label_to_int)} cell types")
        except Exception as e:
            print(f"  published label comparison failed: {e}")
    print(f"  metrics: {metrics}")

    n_panels = 2
    if leiden is not None:
        n_panels += 1
    if pub_labels is not None:
        n_panels += 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    palette_gep = [cmap(i % 10) for i in range(K)]

    # Panel 1: UMAP on doc-topics, colored by dominant GEP
    ax = axes[0]
    for g in range(K):
        mask = (dominant_gep == g)
        ax.scatter(umap_dt[mask, 0], umap_dt[mask, 1], s=4, alpha=0.7,
                   color=palette_gep[g], label=f"GEP {g}")
    ax.set_title(f"UMAP on GEP attribution\n({dataset}/{layout})", fontsize=10)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize=7, markerscale=2)

    # Panel 2: UMAP on expression, colored by dominant GEP
    ax = axes[1]
    for g in range(K):
        mask = (dominant_gep == g)
        ax.scatter(umap_expr[mask, 0], umap_expr[mask, 1], s=4, alpha=0.7,
                   color=palette_gep[g], label=f"GEP {g}")
    ax.set_title(f"UMAP on log1p(CP10K) expression\ncolored by dominant GEP", fontsize=10)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize=7, markerscale=2)

    panel_idx = 2
    # Panel 3: UMAP on expression, colored by Leiden
    if leiden is not None:
        ax = axes[panel_idx]
        n_leiden = len(set(leiden))
        leiden_palette = [plt.get_cmap("tab20")(i % 20) for i in range(n_leiden)]
        for k in range(n_leiden):
            mask = (leiden == k)
            ax.scatter(umap_expr[mask, 0], umap_expr[mask, 1], s=4, alpha=0.7,
                       color=leiden_palette[k], label=f"L{k}")
        ax.set_title(f"UMAP on expression\ncolored by Leiden (k={n_leiden})\n"
                     f"ARI(GEP, Leiden) = {metrics['ari_dominant_gep_vs_leiden']:.3f}",
                     fontsize=10)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        ax.legend(loc="best", fontsize=6, markerscale=2, ncol=2)
        panel_idx += 1

    # Panel 4: UMAP on expression, colored by published cell-type labels
    if pub_labels is not None:
        ax = axes[panel_idx]
        int_labels, int_dom, label_names, our_cell_ids = pub_labels
        # Re-compute UMAP coords for the matched cells only
        cid_to_idx = {cid: i for i, cid in enumerate(our_cell_ids)}
        pub_df = pd.read_csv(WS / "data" / dataset / "cell_type_labels.csv")
        keep = []
        for i, cid in enumerate(our_cell_ids):
            if cid in dict(zip(pub_df["cell_id"].astype(str), pub_df["cell_type"].astype(str))):
                keep.append(i)
        keep = np.array(keep)
        sub_coords = umap_expr[keep]
        n_types = len(label_names)
        type_palette = [plt.get_cmap("tab20")(i % 20) for i in range(n_types)]
        for li, lname in enumerate(label_names):
            mask = (int_labels == li)
            ax.scatter(sub_coords[mask, 0], sub_coords[mask, 1], s=4, alpha=0.7,
                       color=type_palette[li], label=lname[:14])
        ax.set_title(f"UMAP on expression\ncolored by published labels (n={n_types})\n"
                     f"ARI(GEP, published) = {metrics['ari_dominant_gep_vs_published']:.3f}",
                     fontsize=10)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        ax.legend(loc="best", fontsize=6, markerscale=2, ncol=2)

    fig.suptitle(f"{dataset} ({layout}): cells colored by GEP attribution", fontsize=12)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    # Look up the matched-k Leiden reference ARI for this dataset, if computed.
    ref_ari, ref_k = _lookup_leiden_reference(
        dataset, target_k=K,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")

    if metrics_path is not None and metrics:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(f"  wrote {metrics_path}")


def _load_published_labels_for_cells(dataset: str, cell_ids: list[str]):
    """Returns (int_labels, label_names) aligned to cell_ids; or (None, None)."""
    pub = WS / "data" / dataset / "cell_type_labels.csv"
    if not pub.exists():
        return None, None
    df = pd.read_csv(pub)
    id2lab = dict(zip(df["cell_id"].astype(str), df["cell_type"].astype(str)))
    mapped = [id2lab.get(str(cid)) for cid in cell_ids]
    if not any(mapped):
        return None, None
    names = sorted({m for m in mapped if m is not None})
    name2int = {n: i for i, n in enumerate(names)}
    int_labels = np.array([name2int.get(m, -1) for m in mapped])
    return int_labels, names


def run_one(dataset: str, layout: str = "seed0", *, random_state: int = 42,
            skip_main: bool = False, skip_embeddings: bool = False):
    seed_dir = WS / "outputs" / dataset / layout
    if not seed_dir.exists():
        raise FileNotFoundError(seed_dir)
    expr = pd.read_csv(WS / "data" / dataset / DATASET_FILES[dataset], index_col=0)
    doc_topics = load_doc_topics(seed_dir)
    if doc_topics is None:
        raise RuntimeError(f"no doc_topics.txt in {seed_dir}")

    fig_dir = WS / "figures" / dataset
    main_out = fig_dir / f"{dataset}_{layout}_umap.pdf"
    embed_out = fig_dir / f"{dataset}_{layout}_gep_embeddings.pdf"
    metrics_path = WS / "outputs" / dataset / f"clustering_metrics_{layout}.json"

    if not skip_main:
        make_umap_figure(
            expr, doc_topics, main_out,
            dataset=dataset, layout=layout,
            metrics_path=metrics_path, random_state=random_state,
        )

    if not skip_embeddings:
        int_labels, names = _load_published_labels_for_cells(
            dataset, expr.index.astype(str).tolist())
        make_gep_embedding_figure(
            doc_topics, embed_out,
            dataset=dataset, layout=layout,
            expr=expr,
            published_labels=int_labels, published_label_names=names,
            random_state=random_state,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_FILES.keys()) + ["all"],
                        default="all")
    parser.add_argument("--layout", default="seed0")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--skip_main", action="store_true",
                        help="skip the 4-panel UMAP figure (already generated)")
    parser.add_argument("--skip_embeddings", action="store_true",
                        help="skip the 3-panel UMAP/PHATE/JSD figure")
    args = parser.parse_args()

    datasets = list(DATASET_FILES.keys()) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        print(f"\n=== {ds} / {args.layout} ===")
        try:
            run_one(ds, args.layout,
                    random_state=args.random_state,
                    skip_main=args.skip_main,
                    skip_embeddings=args.skip_embeddings)
        except Exception as e:
            print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()
