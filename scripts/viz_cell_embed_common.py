"""Shared 2-D cell-embedding loader for the GeneTrajectory-idiom cell figures (expression-over-embedding and
gene-bin-score). Priority:
  1. the dataset h5ad's precomputed obsm['X_umap'] (aligned by cell_id), when present;
  2. otherwise a PRISM-NATIVE embedding: PHATE on the K=5 cell-topic mixtures (doc_topics.txt), which is exactly
     how cells are represented by PRISM. doc_topics row order == filtered-expression row order (docid = index).

Returns a DataFrame with columns [x, y] indexed by cell_id, plus the aligned expression frame and a source tag.
"""
from __future__ import annotations
import glob, warnings
from pathlib import Path
import numpy as np, pandas as pd

WS = Path(__file__).resolve().parent.parent
K5 = 5


def _find_h5ad(ds: str):
    c = glob.glob(str(WS / "data" / ds / "*.h5ad"))
    return c[0] if c else None


def _expr(ds: str) -> pd.DataFrame:
    c = glob.glob(str(WS / "data" / ds / f"filtered_{ds}_cells_x_genes.csv"))
    e = pd.read_csv(c[0], index_col=0)
    e.index = e.index.astype(str)
    return e


def _parse_doc_topics(path: Path, K: int = K5):
    names, rows = [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 2 + K:
                continue
            names.append(p[1])
            rows.append([float(x) for x in p[2:2 + K]])
    return names, np.array(rows)


def load_cell_embedding(ds: str, seed_dir: Path | None = None):
    """-> (coords_df[cell_id -> (x,y)], expr_df aligned, source_str)."""
    expr = _expr(ds)
    # 0. explicit umap.csv sidecar (used for GeneTrajectory datasets prepped from their h5ad)
    ucsv = WS / "data" / ds / "umap.csv"
    if ucsv.exists():
        u = pd.read_csv(ucsv)
        u["cell_id"] = u["cell_id"].astype(str)
        u = u.set_index("cell_id")
        common = [c for c in u.index if c in expr.index]
        if len(common) >= 0.5 * len(expr):
            coords = u.loc[common][["x", "y"]]
            return coords, expr.loc[common], "h5ad:X_umap(sidecar)"
    h = _find_h5ad(ds)
    if h is not None:
        try:
            import anndata as ad
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a = ad.read_h5ad(h)
            if "X_umap" in a.obsm:
                ids = a.obs_names.astype(str)
                common = [c for c in ids if c in expr.index]
                if len(common) >= 0.5 * len(expr):
                    idx = {c: i for i, c in enumerate(ids)}
                    U = a.obsm["X_umap"][np.array([idx[c] for c in common])]
                    coords = pd.DataFrame(U[:, :2], columns=["x", "y"], index=common)
                    return coords, expr.loc[common], "h5ad:X_umap"
        except Exception as e:  # noqa: BLE001
            print(f"  [{ds}] h5ad X_umap failed ({e}); falling back to PHATE(doc_topics)")

    # PRISM-native fallback: PHATE on doc_topics (K=5 cell-topic mixtures); cache to disk (both cell-figure
    # scripts call this, so the second invocation reuses the embedding instead of recomputing PHATE).
    cache = WS / "outputs" / "trajectory" / ds / "cell_embed_phate.csv"
    if cache.exists():
        coords = pd.read_csv(cache, index_col=0)
        coords.index = coords.index.astype(str)
        common = [c for c in coords.index if c in expr.index]
        return coords.loc[common], expr.loc[common], "phate:doc_topics(K5,cached)"

    sd = seed_dir or (WS / "outputs" / "optimize_interval_grid" / "prism_opt10" / ds / "seed0")
    dt = sd / "doc_topics.txt"
    names, M = _parse_doc_topics(dt)
    if M.shape[0] != expr.shape[0]:
        # align by name if row counts differ
        name_to_row = {n: i for i, n in enumerate(names)}
        keep = [c for c in expr.index if c in name_to_row]
        M = M[[name_to_row[c] for c in keep]]
        expr = expr.loc[keep]
        names = keep
    else:
        names = list(expr.index)  # same order
    import phate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        op = phate.PHATE(n_components=2, verbose=False, random_state=0)
        Y = op.fit_transform(M)
    coords = pd.DataFrame(Y, columns=["x", "y"], index=names)
    try:
        coords.to_csv(cache)
    except Exception:  # noqa: BLE001
        pass
    return coords, expr.loc[coords.index], "phate:doc_topics(K5)"
