"""Cell-ordering pipelines built on "JS + a few extra steps".

The shipped headline orders cells by the FIRST PHATE COORDINATE of their K-dim
MALLET doc-topic (GEP-attribution) vectors under default EUCLIDEAN geometry.
An earlier geometry ablation already showed that simply
switching the PHATE geometry to the simplex-correct Jensen-Shannon metric
("plain JS-PHATE", candidate C5) robustly improves recovery on all 3 datasets
(Pancreas .871, Gast-Erythroid .813, Gastrulation .617; mean .767) but still
trails DPT (.795) / Slingshot (.825).

This module asks: starting from that same JS / simplex geometry of the
GEP-attribution vectors, do a FEW PRINCIPLED EXTRA STEPS recover the ordering
better still? Every extra step is geometry/simplex-motivated and chosen A
PRIORI (no parameter or metric is selected by maximizing the test score).

Candidates (all on the cells x K doc-topic matrix P, rows on the simplex):
  CTRL  plain JS-PHATE-1                -- reproduces ablation C5 (SANITY GATE).
  E1    JS-PHATE -> principal curve, order by arc-length projection.
          rationale: PHATE-1 projects the 2-D simplex embedding onto one axis
          and loses curvature; the trajectory is a 1-D *curve*, so fit a
          principal curve (Hastie-Stuetzle) in the 2-D JS-PHATE embedding and
          order cells by arc length. [extra-step family (4)]
  E2    JS-kNN denoise of P, then JS-PHATE-1.
          rationale: per-cell topic vectors are noisy; smoothing each cell's
          attribution over its JS-graph neighbours (MAGIC/kNN-smoothing) before
          manifold learning is a standard denoising step. Renormalise back to
          the simplex, then JS-PHATE-1. [extra-step family (2)]
  E3    JS diffusion-map leading component DC1.
          rationale: build a diffusion operator on the JS-metric kNN graph and
          take its leading non-trivial diffusion coordinate (the dominant axis
          of the simplex diffusion geometry) instead of PHATE-1. [family (3)]
  E4    JS diffusion pseudotime (DPT) from a SINGLE root anchor.
          rationale: diffusion pseudotime computed on the JS-metric diffusion
          operator of the GEP vectors, anchored by the SAME single root the
          DPT/Slingshot baselines use (no other label info). [family (1)]
  E5    Hellinger (sqrt-simplex) PHATE-1.
          rationale: alternative canonical simplex metric. Hellinger distance
          H(p,q)=||sqrt(p)-sqrt(q)||/sqrt2 is EXACTLY Euclidean in sqrt-prob
          coordinates (Fisher-Rao chordal geometry), so PHATE-1 on sqrt(P)
          realises a second simplex metric exactly. Discloses metric
          robustness vs JS (guards against cherry-picking JS). [family (5)]

E3 and E4 SHARE one diffusion operator built from the JS distance matrix, so a
single well-tested eigendecomposition feeds both.

SCORING: identical protocol to the published harness and to the earlier
cell-ordering ablation:
  score = |Spearman(ordering, published lineage rank)| per seed, then
  np.nanmean over seeds. Orientation is resolved by abs() exactly as the shipped
  PHATE-1 headline and the DPT/Slingshot baselines resolve it. We NEVER
  sign-flip-to-maximise (that is leakage). E4 (DPT) consumes ONLY the single
  root anchor (iroot = first finite cell whose lineage == spec.root_cluster) —
  identical to the baselines; no candidate consumes the ground-truth rank.
  `absrho` is byte-identical to recompute_cell_trajectory_honest.py:35-40 (and
  to the ablation harness). Dataset loaders/aligners are imported, not copied,
  from trajectory_baselines.

NO-LEAKAGE: every hyper-parameter (k=15 neighbours, n_evec=15, principal-curve
n_knots=10 / n_iter=12, denoise rounds=2) is fixed a priori in this file and is
NOT tuned against the Spearman score. The choice of which extra step to
"recommend" is made AFTER the run by the cross-dataset-robustness criterion
declared here, and ALL candidates are reported.

CRITICAL GATE: CTRL must reproduce plain JS-PHATE (ablation C5):
  Pancreas ~.871, Gast-Erythroid ~.813, Gastrulation ~.617. If it does not, the
  harness is wrong and candidate numbers are not trustworthy -> STOP.

Output:
  outputs/trajectory/cell_ordering_extra_steps.csv  (dataset, method, mean_rho, std_rho, n_seeds)
  outputs/trajectory/_search/cell_ordering_extra_steps_perseed.csv
  outputs/trajectory/_search/cell_ordering_extra_steps_cache.json  (resumable)

Run:
  python scripts/cell_ordering_extra_steps.py                      # seeds 0..4, all ds
  python scripts/cell_ordering_extra_steps.py --datasets pancreas --seeds 0 1
  python scripts/cell_ordering_extra_steps.py --seeds 0 1 2 3 4 5 6 7 8 9
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import spearmanr

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS / "scripts"))

from trajectory_baselines import (  # noqa: E402
    DATASET_SPECS, load_dataset, load_prism_doc_topics, phate_first_coord,
)

K_TYPES = {"pancreas": 8, "gastrulation": 9, "gastrulation_erythroid": 7}
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

TRAJ = WS / "outputs" / "trajectory"
SEARCH = TRAJ / "_search"
CACHE = SEARCH / "cell_ordering_extra_steps_cache.json"

# A-priori fixed hyper-parameters (NOT tuned on the score).
KNN = 15            # neighbours for graphs / diffusion / denoise
N_EVEC = 15         # diffusion eigenvectors
PCURVE_KNOTS = 10   # principal-curve spline interior knots
PCURVE_ITERS = 12   # principal-curve refinement iterations
DENOISE_ROUNDS = 2  # kNN-smoothing passes
MAHAL_REG_GLOBAL = 1e-3  # shrinkage on the pooled covariance (rank-deficient on simplex)
MAHAL_REG_LOCAL = 0.1    # shrinkage on each local covariance (k<dim+1 -> noisy)

# CTRL sanity target = ablation C5 (plain JS-PHATE), per dataset.
CTRL_TARGET = {"pancreas": 0.871, "gastrulation": 0.617,
               "gastrulation_erythroid": 0.813}
CTRL_TOL = 0.03     # |5-seed CTRL mean - 10-seed C5| must be <= this


# --- SCORER (byte-identical to recompute_cell_trajectory_honest.py:35-40) ---

def absrho(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return np.nan
    r = spearmanr(a[m], b[m]).correlation
    return abs(r) if r == r else np.nan


# --- shared geometry helpers ------------------------------------------------

def js_distance(P):
    """Dense pairwise Jensen-Shannon distance (a true simplex metric)."""
    D = squareform(pdist(P.astype(np.float64), metric="jensenshannon"))
    D[~np.isfinite(D)] = 0.0
    np.fill_diagonal(D, 0.0)
    return D


def phate_js_embed(P, n_components=2, random_state=42):
    """PHATE on precomputed JS distance; returns the n x n_components embedding.

    Identical PHATE config to the shipped headline (random_state=42); only the
    knn_dist (precomputed JS) differs. CTRL = column 0 of this embedding."""
    import phate
    D = js_distance(P)
    op = phate.PHATE(n_components=n_components, random_state=random_state,
                     verbose=0, n_jobs=1, knn_dist="precomputed")
    return op.fit_transform(D)


def js_diffusion(P, k=KNN, n_evec=N_EVEC):
    """Diffusion-map operator on the JS-metric kNN graph of P.

    Self-tuning Gaussian kernel (Zelnik-Manor & Perona) on a symmetric kNN
    graph, Coifman-Lafon anisotropic (alpha=1) normalisation, then the leading
    `n_evec` eigenpairs of the symmetric transition conjugate. Returns
    (eigvals lam[0..], right-eigvecs psi[:, 0..]) with lam descending and lam[0]
    the trivial stationary mode. Used by BOTH E3 (DC1 = psi[:,1]) and E4 (DPT)."""
    D = js_distance(P)
    n = D.shape[0]
    k = int(min(k, n - 1))

    # per-cell adaptive bandwidth = distance to k-th neighbour (excludes self)
    part = np.partition(D, k, axis=1)
    sigma = part[:, k]
    sigma[sigma <= 0] = np.median(sigma[sigma > 0]) if np.any(sigma > 0) else 1.0

    # symmetric kNN adjacency (union of mutual neighbours)
    idx = np.argsort(D, axis=1)[:, 1:k + 1]
    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1)
    A = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()
    A = A.maximum(A.T)  # symmetric union
    A = sp.triu(A, k=1).tocoo()  # unique undirected edges

    ri, ci = A.row, A.col
    w = np.exp(-(D[ri, ci] ** 2) / (sigma[ri] * sigma[ci]))
    K = sp.coo_matrix((w, (ri, ci)), shape=(n, n))
    K = (K + K.T).tocsr()  # symmetric kernel, zero diagonal

    # anisotropic (alpha=1) normalisation: K2 = K / (q_i q_j)
    q = np.asarray(K.sum(axis=1)).ravel()
    q[q == 0] = 1e-12
    Dinv = sp.diags(1.0 / q)
    K2 = Dinv @ K @ Dinv

    # symmetric transition conjugate S = d^-1/2 K2 d^-1/2
    d = np.asarray(K2.sum(axis=1)).ravel()
    d[d == 0] = 1e-12
    Dm = sp.diags(1.0 / np.sqrt(d))
    S = (Dm @ K2 @ Dm).tocsr()
    S = (S + S.T) * 0.5  # enforce numerical symmetry

    m = int(min(n_evec, n - 2))
    lam, v = eigsh(S, k=m, which="LA")
    order = np.argsort(lam)[::-1]
    lam = lam[order]
    v = v[:, order]
    psi = v / np.sqrt(d)[:, None]  # right eigenvectors of the transition matrix
    return lam, psi


def js_diffmap_dc1(P):
    """E3: leading non-trivial JS diffusion-map coordinate (psi_1)."""
    lam, psi = js_diffusion(P)
    return psi[:, 1]


def js_dpt(P, root_local):
    """E4: diffusion pseudotime from a single root on the JS diffusion operator.

    DPT distance (Haghverdi 2016): dpt(x) = || sum_{k>=1} (lam_k/(1-lam_k))
    (psi_k(x) - psi_k(root)) ||, exactly scanpy's accumulated-transition form.
    Uses ONLY the single root anchor; abs() resolves orientation."""
    if root_local is None:
        return np.full(P.shape[0], np.nan)
    lam, psi = js_diffusion(P)
    lam = lam[1:]
    psi = psi[:, 1:]
    w = lam / (1.0 - np.clip(lam, None, 1 - 1e-6))  # DPT eigen weights
    diff = (psi - psi[root_local][None, :]) * w[None, :]
    return np.sqrt((diff ** 2).sum(axis=1))


def js_denoise_phate1(P, k=KNN, rounds=DENOISE_ROUNDS, random_state=42):
    """E2: kNN-smooth P on its JS graph (renormalise to simplex), then JS-PHATE-1.

    Each round replaces a cell's topic vector by the mean over itself + its k JS
    neighbours (uniform weights), then renormalises rows to sum 1 so they stay on
    the simplex. Standard attribution denoising before manifold learning."""
    D = js_distance(P)
    n = D.shape[0]
    k = int(min(k, n - 1))
    idx = np.argsort(D, axis=1)[:, :k + 1]  # includes self (nearest is self)
    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = idx.reshape(-1)
    W = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()
    W = W.maximum(W.T)
    deg = np.asarray(W.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    Wn = sp.diags(1.0 / deg) @ W  # row-stochastic smoother
    Ps = P.astype(np.float64).copy()
    for _ in range(rounds):
        Ps = np.asarray(Wn @ Ps)
        s = Ps.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        Ps = Ps / s  # back onto the simplex
    return phate_first_coord_js(Ps, random_state=random_state)


def phate_first_coord_js(P, random_state=42):
    return phate_js_embed(P, n_components=2, random_state=random_state)[:, 0]


def hellinger_phate1(P, random_state=42):
    """E5: PHATE-1 in Hellinger (sqrt-simplex) geometry.

    Euclidean distance on sqrt(P) == sqrt(2) * Hellinger(P), so default-Euclidean
    PHATE on sqrt(P) realises the Hellinger simplex metric exactly."""
    return phate_first_coord(np.sqrt(np.clip(P, 0, None)), random_state=random_state)


# --- principal curve (Hastie-Stuetzle), unsupervised ------------------------

def principal_curve_arclength(Y, n_knots=PCURVE_KNOTS, n_iter=PCURVE_ITERS):
    """Fit a principal curve to 2-D points Y and return per-point arc length.

    Unsupervised: uses only Y (the JS-PHATE embedding). Iterates project ->
    spline-smooth, classic Hastie-Stuetzle. Spline flexibility is bounded a
    priori by n_knots interior knots (NOT tuned on the score)."""
    from scipy.interpolate import LSQUnivariateSpline
    from sklearn.decomposition import PCA

    Y = np.asarray(Y, float)
    n = Y.shape[0]
    # init arc length = projection on first PC
    lam = PCA(n_components=1, random_state=0).fit_transform(Y)[:, 0]

    def _fit_once(lam):
        order = np.argsort(lam)
        xs = lam[order].astype(float)
        # enforce strictly increasing x for the spline
        eps = (xs[-1] - xs[0]) / max(n, 1) * 1e-6 + 1e-9
        xs = xs + np.arange(n) * eps
        lo, hi = xs[0], xs[-1]
        if not np.isfinite(lo) or hi <= lo:
            return lam  # degenerate; leave unchanged
        qs = np.linspace(0, 1, n_knots + 2)[1:-1]
        knots = np.quantile(xs, qs)
        knots = np.unique(np.clip(knots, lo + (hi - lo) * 1e-3,
                                  hi - (hi - lo) * 1e-3))
        try:
            sx = LSQUnivariateSpline(xs, Y[order, 0], t=knots, k=3)
            sy = LSQUnivariateSpline(xs, Y[order, 1], t=knots, k=3)
        except Exception:
            return lam
        grid = np.linspace(lo, hi, 600)
        curve = np.column_stack([sx(grid), sy(grid)])
        seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0),
                                                  axis=1))]  # arc length on grid
        # project each point to nearest grid vertex -> its arc length
        nn = np.argmin(cdist(Y, curve), axis=1)
        return seg[nn]

    prev = lam
    for _ in range(n_iter):
        new = _fit_once(prev)
        if np.std(new) < 1e-12:
            break
        r = spearmanr(prev, new).correlation
        prev = new
        if r is not None and abs(r) > 0.9999:
            break
    return prev


def js_phate_pcurve(P, random_state=42):
    """E1: principal-curve arc length in the 2-D JS-PHATE embedding."""
    emb = phate_js_embed(P, n_components=2, random_state=random_state)
    return principal_curve_arclength(emb)


# --- Mahalanobis geometry (E6, E7) ------------------------------------------
# Standalone helpers so the verified E3/E4 js_diffusion path is untouched.

def _diffusion_eigs(K, n_evec=N_EVEC):
    """Coifman-Lafon (alpha=1) diffusion eigenpairs of a symmetric affinity K.

    Identical normalisation/conjugation to js_diffusion's tail, but taking an
    arbitrary precomputed kernel so the Mahalanobis candidates can supply their
    own (anisotropic) affinity. Returns (lam desc, psi right-eigvecs)."""
    K = sp.csr_matrix(K)
    K = (K + K.T) * 0.5
    n = K.shape[0]
    q = np.asarray(K.sum(axis=1)).ravel(); q[q == 0] = 1e-12
    Dinv = sp.diags(1.0 / q)
    K2 = Dinv @ K @ Dinv
    d = np.asarray(K2.sum(axis=1)).ravel(); d[d == 0] = 1e-12
    Dm = sp.diags(1.0 / np.sqrt(d))
    S = (Dm @ K2 @ Dm).tocsr(); S = (S + S.T) * 0.5
    m = int(min(n_evec, n - 2))
    lam, v = eigsh(S, k=m, which="LA")
    order = np.argsort(lam)[::-1]
    lam = lam[order]; v = v[:, order]
    psi = v / np.sqrt(d)[:, None]
    return lam, psi


def _knn_selftuning_kernel(D, k=KNN):
    """Self-tuning Gaussian affinity on the symmetric kNN graph of distance D.

    Same construction js_diffusion uses for the JS graph (Zelnik-Manor adaptive
    bandwidth = distance to k-th neighbour), factored out for E6."""
    n = D.shape[0]; k = int(min(k, n - 1))
    part = np.partition(D, k, axis=1); sigma = part[:, k]
    pos = sigma > 0
    sigma[~pos] = np.median(sigma[pos]) if pos.any() else 1.0
    idx = np.argsort(D, axis=1)[:, 1:k + 1]
    rows = np.repeat(np.arange(n), k); cols = idx.reshape(-1)
    A = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()
    A = A.maximum(A.T); A = sp.triu(A, k=1).tocoo()
    ri, ci = A.row, A.col
    w = np.exp(-(D[ri, ci] ** 2) / (sigma[ri] * sigma[ci]))
    K = sp.coo_matrix((w, (ri, ci)), shape=(n, n))
    return (K + K.T).tocsr()


def global_mahal_diffmap_dc1(P, reg=MAHAL_REG_GLOBAL):
    """E6: diffusion-map DC1 under the GLOBAL Mahalanobis metric of P.

    Whiten the topic-proportion space by the pooled covariance (pseudo-inverse,
    since the sum=1 constraint makes it rank K-1; small a-priori shrinkage), so
    Euclidean distance on the whitened coords IS the global Mahalanobis distance.
    Then the same self-tuning diffusion operator as E3, leading coordinate."""
    X = P.astype(np.float64)
    Xc = X - X.mean(0, keepdims=True)
    C = np.cov(Xc, rowvar=False)
    dim = C.shape[0]
    C = C + reg * (np.trace(C) / dim + 1e-12) * np.eye(dim)
    w, V = np.linalg.eigh(np.linalg.pinv(C))
    w = np.clip(w, 0, None)
    Wsqrt = (V * np.sqrt(w)) @ V.T          # symmetric Cinv^{1/2}
    Y = Xc @ Wsqrt
    D = squareform(pdist(Y, metric="euclidean"))
    D[~np.isfinite(D)] = 0.0; np.fill_diagonal(D, 0.0)
    lam, psi = _diffusion_eigs(_knn_selftuning_kernel(D))
    return psi[:, 1]


def local_mahal_diffmap_dc1(P, k=KNN, k_cov=None, reg=MAHAL_REG_LOCAL):
    """E7/E8: diffusion-map DC1 under a LOCAL Mahalanobis (anisotropic) kernel.

    Singer & Coifman (2008) anisotropic diffusion: each cell i gets a local
    covariance C_i from its `k_cov` nearest neighbours; the pairwise kernel uses
    M_ij=(C_i^+ + C_j^+)/2, stretching the metric along the local trajectory
    tangent. C_i is shrinkage-regularised a priori (k_cov<dim+1 makes it noisy)
    and pseudo-inverted. eps = median edge Mahalanobis distance (self-tuning).
    The diffusion graph always uses `k` neighbours; `k_cov` (default = k) lets a
    larger patch better-condition the covariance without densifying the graph."""
    X = P.astype(np.float64)
    n, dim = X.shape
    k_cov = int(k if k_cov is None else min(k_cov, n - 1))
    D0 = squareform(pdist(X, metric="euclidean"))
    order = np.argsort(D0, axis=1)
    cov_idx = order[:, 1:k_cov + 1]                    # patch for the covariance
    graph_idx = order[:, 1:k + 1]                      # edges for the diffusion graph
    Cinv = np.empty((n, dim, dim))
    for i in range(n):
        nb = X[cov_idx[i]] - X[i]                       # local patch, centred on i
        C = nb.T @ nb / max(cov_idx.shape[1], 1)
        C = C + reg * (np.trace(C) / dim + 1e-12) * np.eye(dim)
        Cinv[i] = np.linalg.pinv(C)
    rows = np.repeat(np.arange(n), graph_idx.shape[1]); cols = graph_idx.reshape(-1)
    A = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()
    A = A.maximum(A.T); A = sp.triu(A, k=1).tocoo()
    ri, ci = A.row, A.col
    diff = X[ri] - X[ci]
    M = 0.5 * (Cinv[ri] + Cinv[ci])
    d2 = np.einsum("ed,edf,ef->e", diff, M, diff)
    d2 = np.clip(d2, 0, None)
    pos = d2 > 0
    eps = np.median(d2[pos]) if pos.any() else 1.0
    w = np.exp(-d2 / eps)
    K = sp.coo_matrix((w, (ri, ci)), shape=(n, n))
    lam, psi = _diffusion_eigs((K + K.T).tocsr())
    return psi[:, 1]


# --- candidate registry -----------------------------------------------------

def ctrl_js_phate1(P, **_):
    return phate_js_embed(P, n_components=2, random_state=42)[:, 0]


CANDIDATES = {
    "CTRL_JS_PHATE1":       lambda P, **kw: ctrl_js_phate1(P),
    "E1_JS_PHATE_PCURVE":   lambda P, **kw: js_phate_pcurve(P),
    "E2_JS_DENOISE_PHATE1": lambda P, **kw: js_denoise_phate1(P),
    "E3_JS_DIFFMAP_DC1":    lambda P, **kw: js_diffmap_dc1(P),
    "E4_JS_DPT_root":       lambda P, root_local=None, **kw: js_dpt(P, root_local),
    "E5_HELLINGER_PHATE1":  lambda P, **kw: hellinger_phate1(P),
    "E6_GLOBAL_MAHAL_DC1":  lambda P, **kw: global_mahal_diffmap_dc1(P),
    "E7_LOCAL_MAHAL_DC1":   lambda P, **kw: local_mahal_diffmap_dc1(P),
    "E8_LOCAL_MAHAL_KCOV40": lambda P, **kw: local_mahal_diffmap_dc1(P, k_cov=40),
}


# --- cache helpers ----------------------------------------------------------

def load_cache():
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except Exception:
            return {}
    return {}


def save_cache(cache):
    SEARCH.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=2))


# --- per-dataset driver -----------------------------------------------------

def run_dataset(ds, seeds, cache):
    spec = DATASET_SPECS[ds]
    adata, rank = load_dataset(spec)
    truth = rank.astype(float)
    n = adata.n_obs
    ktype = K_TYPES[ds]
    lineage = adata.obs["lineage"].astype(str).to_numpy()
    print(f"\n===== {ds}: {n} cells; K{ktype}; root={spec.root_cluster} =====",
          flush=True)

    cache.setdefault(ds, {})
    for s in seeds:
        skey = f"seed{s}"
        layout = f"K{ktype}/seed{s}"
        dt = load_prism_doc_topics(spec, adata, layout=layout)
        if dt is None:
            print(f"  [seed{s}] no doc_topics at {layout} -- skip", flush=True)
            continue
        finite = ~np.isnan(dt).any(axis=1)
        P = dt[finite]

        # single root anchor (baselines' convention)
        root_mask = (lineage == str(spec.root_cluster)) & finite
        root_local = None
        if root_mask.any():
            root_global = int(np.where(root_mask)[0][0])
            root_local = int(finite[:root_global].sum())

        cache[ds].setdefault(skey, {})
        for cname, fn in CANDIDATES.items():
            if cname in cache[ds][skey] and cache[ds][skey][cname] is not None:
                continue
            t = time.time()
            try:
                axis = np.asarray(fn(P, root_local=root_local), float)
                pt = np.full(n, np.nan)
                pt[finite] = axis
                sc_val = absrho(pt, truth)
                sc_val = None if (sc_val != sc_val) else float(sc_val)
                cache[ds][skey][cname] = sc_val
                disp = "nan" if sc_val is None else f"{sc_val:.4f}"
                print(f"  [seed{s}] {cname:22s} |rho|={disp} "
                      f"({time.time()-t:.1f}s)", flush=True)
            except Exception as e:
                cache[ds][skey][cname] = None
                print(f"  [seed{s}] {cname:22s} FAILED: {repr(e)[:160]}",
                      flush=True)
            save_cache(cache)
    return cache


# --- aggregation ------------------------------------------------------------

def aggregate(cache, datasets):
    rows, long_rows = [], []
    for ds in datasets:
        for cname in CANDIDATES:
            vals = []
            for skey in sorted(cache.get(ds, {})):
                v = cache[ds][skey].get(cname, None)
                vals.append(np.nan if v is None else float(v))
                long_rows.append({"dataset": ds, "method": cname, "seed": skey,
                                  "rho": (np.nan if v is None else float(v))})
            arr = np.array(vals, float)
            finite = arr[np.isfinite(arr)]
            rows.append({
                "dataset": ds, "method": cname,
                "mean_rho": (float(np.nanmean(arr)) if finite.size else np.nan),
                "std_rho": (float(np.nanstd(arr)) if finite.size else np.nan),
                "n_seeds": int(finite.size),
            })
    return pd.DataFrame(rows), pd.DataFrame(long_rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["all"])
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = p.parse_args()
    datasets = (list(K_TYPES.keys())
                if args.datasets == ["all"] else args.datasets)

    cache = load_cache()
    for ds in datasets:
        if ds not in DATASET_SPECS:
            print(f"[{ds}] no spec -- SKIP", flush=True)
            continue
        run_dataset(ds, args.seeds, cache)

    summary, long_df = aggregate(cache, datasets)

    TRAJ.mkdir(parents=True, exist_ok=True)
    SEARCH.mkdir(parents=True, exist_ok=True)
    out_csv = TRAJ / "cell_ordering_extra_steps.csv"
    summary.to_csv(out_csv, index=False)
    long_df.to_csv(SEARCH / "cell_ordering_extra_steps_perseed.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n\n========== CELL-ORDERING EXTRA-STEPS (|Spearman rho|) ==========")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n----- CTRL sanity gate (must reproduce plain JS-PHATE / C5) -----")
    all_pass = True
    for ds in datasets:
        row = summary[(summary.dataset == ds) &
                      (summary.method == "CTRL_JS_PHATE1")]
        if row.empty or not np.isfinite(row.mean_rho.iloc[0]):
            print(f"  {ds:24s} CTRL=NA -- GATE FAIL")
            all_pass = False
            continue
        c = float(row.mean_rho.iloc[0])
        tgt = CTRL_TARGET[ds]
        ok = abs(c - tgt) <= CTRL_TOL
        all_pass &= ok
        print(f"  {ds:24s} CTRL={c:.4f}  C5~{tgt:.3f}  "
              f"|d|={abs(c-tgt):.3f}<={CTRL_TOL}  {'PASS' if ok else 'FAIL'}")
    print(f"\nCTRL SANITY: {'PASS' if all_pass else 'FAIL — harness suspect'}")
    print(f"\n[written] {out_csv}")


if __name__ == "__main__":
    main()
