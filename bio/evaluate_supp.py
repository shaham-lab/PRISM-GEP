"""Metric implementations used for the PRISM-GEP supplementary experiments.

  Coherence:
      coherence = (2/(K(K-1))) Σ_{i<j, i,j ∈ S} ρ_ij
      where ρ_ij = Spearman correlation between gene i and gene j expression.

  Strength:
      One-sided hypergeometric test:
        p = Σ_{i=x}^{min(K,M)} [C(M,i)·C(N-M,K-i)] / C(N,K)
      where:
        N = |U| = gene universe size
        K = |S| = top-N gene set size (typically 20)
        M = |P ∩ U| = pathway size restricted to gene universe
        x = |S ∩ P| = overlap between top-N and pathway
      Apply BH-FDR. Strength = -log10(q).

  Coverage:
      An earlier variant used a gene-hit interpretation:
      coverage = |{g in S: g belongs to at least one significant pathway}| /
                 |{g in S: g is annotated by the library}|
      where significant pathways are selected after FDR. This gives values on
      the same scale as the published table, unlike the literal printed
      cov(P|S)=|S ∩ P|/|P| formula.

  Step (ii) gene ordering (§D):
      gene rep = its distribution over PRISM-GEP topics
      distance = Hellinger between gene-topic distributions
      kernel  = Gaussian with bandwidth = median distance
      then diffusion maps within each program → first non-trivial
      eigenvector (EV2) as 1D ordering coordinate.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, hypergeom


_GENE_TO_PATHWAYS_CACHE: dict[int, dict[str, set[str]]] = {}


def coherence_supp(
    top_genes_per_gep: dict[int, list[str]],
    expression_df: pd.DataFrame,
) -> float:
    """coherence = mean over GEPs of mean-pairwise-Spearman within GEP.

    Exactly as written in supplementary §C.5.1.
    Case-insensitive gene lookup (MALLET's Csv2Vectors lowercases tokens
    by default, but expression matrices use original CamelCase mouse symbols).
    """
    # Build case-insensitive column lookup
    col_lookup = {c.upper(): c for c in expression_df.columns}
    per_gep = []
    for genes in top_genes_per_gep.values():
        present = [col_lookup[g.upper()] for g in genes if g.upper() in col_lookup]
        if len(present) < 2:
            continue
        sub = expression_df[present].values
        rho, _ = spearmanr(sub, axis=0)
        if np.ndim(rho) == 0:
            per_gep.append(float(rho))
            continue
        iu = np.triu_indices_from(rho, k=1)
        per_gep.append(float(np.nanmean(rho[iu])))
    if not per_gep:
        return float("nan")
    return float(np.mean(per_gep))


def _build_gene_to_pathways(library: dict[str, list[str]]) -> dict[str, set[str]]:
    """Invert library to gene → set of pathway names (uppercase symbols)."""
    cache_key = id(library)
    if cache_key in _GENE_TO_PATHWAYS_CACHE:
        return _GENE_TO_PATHWAYS_CACHE[cache_key]
    gene_to_paths = defaultdict(set)
    for term, members in library.items():
        for m in members:
            gene_to_paths[m.upper()].add(term)
    out = dict(gene_to_paths)
    _GENE_TO_PATHWAYS_CACHE[cache_key] = out
    return out


def _coverage_gene_hit(
    top_genes: list[str],
    library: dict[str, list[str]],
    significant_terms: set[str],
    gene_map: dict[str, str] | None = None,
) -> float:
    """Fraction of annotated top genes covered by any significant pathway.

    ``gene_map`` (optional) translates each top gene's UPPER symbol before
    matching the library — used for mouse->human ortholog mapping so mouse
    datasets match the human GO library. Default None = identity (legacy).
    """
    def _tr(g: str) -> str:
        gu = g.upper()
        return gene_map.get(gu, gu) if gene_map else gu
    top_upper = {_tr(g) for g in top_genes}
    gene_to_paths = _build_gene_to_pathways(library)
    annotated_top = {g for g in top_upper if g in gene_to_paths}
    hit_genes = {
        g
        for g in annotated_top
        if gene_to_paths[g] & significant_terms
    }
    den = len(annotated_top) if annotated_top else len(top_upper)
    return float(len(hit_genes) / max(den, 1))


def hypergeom_enrichment(
    top_genes: list[str],
    library: dict[str, list[str]],
    universe: list[str],
    *,
    q_threshold: float = 0.05,
    gene_map: dict[str, str] | None = None,
    fdr_full_family: bool = False,
) -> pd.DataFrame:
    """Per-pathway hypergeometric test + BH-FDR per supplementary §C.5.1.

    Parameters
    ----------
    top_genes : list[str]
        Top-K genes for one GEP (S).
    library : dict[str, list[str]]
        Pathway library: {pathway_name: list_of_member_genes}.
        Members are gene symbols.
    universe : list[str]
        Gene universe U. N = |U|.
    q_threshold : float
        FDR cutoff.

    Returns
    -------
    DataFrame with columns: term, M, M_in_U, x, k_top, p, q, sig
    """
    def _tr(g: str) -> str:
        gu = g.upper()
        return gene_map.get(gu, gu) if gene_map else gu

    universe_upper = {_tr(g) for g in universe}
    N = len(universe_upper)
    K = len(top_genes)
    top_upper = {_tr(g) for g in top_genes}

    n_tested = 0  # pathways actually tested (0 < |P∩U| < N), the true BH family
    rows = []
    for term, members in library.items():
        members_upper = {m.upper() for m in members}
        # M_in_U = pathway restricted to universe (matches hypergeom's "white balls")
        M_in_U = members_upper & universe_upper
        M = len(M_in_U)
        if M == 0 or M >= N:
            continue
        n_tested += 1
        x = len(top_upper & members_upper)
        if x == 0:
            continue
        # P(X >= x) = sum from i=x to min(K,M) of C(M,i)*C(N-M,K-i)/C(N,K)
        # scipy.stats.hypergeom.sf(x-1, N, M, K) = P(X > x-1) = P(X >= x)
        p = float(hypergeom.sf(x - 1, N, M, K))
        rows.append({
            "term": term,
            "M_full": len(members_upper),  # full pathway size (paper formula uses this)
            "M_in_U": M,                   # pathway ∩ universe (used in test)
            "x": x,
            "K": K,
            "p": p,
        })
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Benjamini-Hochberg FDR.
    #   fdr_full_family=False (legacy / Enrichr-style): family = overlapping terms.
    #   fdr_full_family=True  (strict ORA): family = all tested terms (0<|P∩U|<N).
    df = df.sort_values("p").reset_index(drop=True)
    n_overlap = len(df)
    n = n_tested if fdr_full_family else n_overlap
    df["rank"] = np.arange(1, n_overlap + 1)
    df["q_raw"] = df["p"] * n / df["rank"]
    # Enforce monotonicity from the bottom
    df["q"] = df["q_raw"][::-1].cummin()[::-1].clip(upper=1.0)
    df["sig"] = df["q"] < q_threshold
    df.attrs["n_tested"] = n_tested
    return df


def coverage_strength_supp(
    top_genes: list[str],
    library: dict[str, list[str]],
    universe: list[str],
    *,
    q_threshold: float = 0.05,
    use_M_full: bool = True,
    gene_map: dict[str, str] | None = None,
    fdr_full_family: bool = False,
) -> tuple[float | None, float | None, int]:
    """Compute gene-hit Coverage and Strength.

    Coverage uses the earlier high-scale interpretation:
        |top genes that hit any significant pathway| / |annotated top genes|

    ``use_M_full`` is retained for compatibility with older call sites; it no
    longer affects Coverage under this interpretation.
    ``gene_map`` / ``fdr_full_family`` are the ortholog-mapping and BH-family
    fixes (see hypergeom_enrichment); defaults preserve legacy behaviour.

    Returns (coverage, strength, n_significant).
    Both = None if no significant pathway.
    """
    df = hypergeom_enrichment(top_genes, library, universe, q_threshold=q_threshold,
                              gene_map=gene_map, fdr_full_family=fdr_full_family)
    if df.empty:
        return None, None, 0
    sig = df[df["sig"]].copy()
    if sig.empty:
        return None, None, 0
    n_sig = len(sig)

    cov = _coverage_gene_hit(top_genes, library, set(sig["term"]), gene_map=gene_map)
    strg = float((-np.log10(sig["q"].clip(lower=1e-300))).mean())
    return cov, strg, n_sig


def coverage_strength_top1_supp(
    top_genes: list[str],
    library: dict[str, list[str]],
    universe: list[str],
    *,
    q_threshold: float = 0.05,
    use_M_full: bool = True,
    gene_map: dict[str, str] | None = None,
    fdr_full_family: bool = False,
) -> tuple[float | None, float | None, int]:
    """Same as ``coverage_strength_supp`` but uses only the top-1 most-significant
    pathway (smallest q) for Strength."""
    df = hypergeom_enrichment(top_genes, library, universe, q_threshold=q_threshold,
                              gene_map=gene_map, fdr_full_family=fdr_full_family)
    if df.empty:
        return None, None, 0
    sig = df[df["sig"]].copy()
    if sig.empty:
        return None, None, 0
    n_sig = len(sig)

    cov = _coverage_gene_hit(top_genes, library, set(sig["term"]), gene_map=gene_map)
    top1 = sig.nsmallest(1, "q")
    strg = float(-np.log10(top1["q"].clip(lower=1e-300).iloc[0]))
    return cov, strg, n_sig


@dataclass
class Table1Result:
    coh_mean: float
    coh_std: float
    cov_mean: float | None
    cov_std: float | None
    str_mean: float | None
    str_std: float | None
    n_seeds: int


def aggregate_dataset(
    top_genes_per_gep_per_seed: dict[int, dict[int, list[str]]],
    expression_df: pd.DataFrame,
    library: dict[str, list[str]],
    *,
    q_threshold: float = 0.05,
    use_M_full: bool = True,
    use_top1_strength: bool = False,
) -> Table1Result:
    """Aggregate the gene-set quality metrics across seeds."""
    universe = list(expression_df.columns)
    coh_seeds, cov_seeds, str_seeds = [], [], []
    for seed, top_per_gep in top_genes_per_gep_per_seed.items():
        coh_seeds.append(coherence_supp(top_per_gep, expression_df))
        cov_per_gep, str_per_gep = [], []
        for gep, genes in top_per_gep.items():
            top20 = genes[:20]
            if use_top1_strength:
                cov, strg, _ = coverage_strength_top1_supp(
                    top20, library, universe, q_threshold=q_threshold,
                    use_M_full=use_M_full,
                )
            else:
                cov, strg, _ = coverage_strength_supp(
                    top20, library, universe, q_threshold=q_threshold,
                    use_M_full=use_M_full,
                )
            if cov is not None:
                cov_per_gep.append(cov)
            if strg is not None:
                str_per_gep.append(strg)
        if cov_per_gep:
            cov_seeds.append(float(np.mean(cov_per_gep)))
        if str_per_gep:
            str_seeds.append(float(np.mean(str_per_gep)))

    def _ms(xs: list[float]) -> tuple[float | None, float | None]:
        if not xs:
            return None, None
        m = float(np.mean(xs))
        s = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
        return m, s

    cm, cs = _ms(coh_seeds)
    cov_m, cov_s = _ms(cov_seeds)
    str_m, str_s = _ms(str_seeds)
    return Table1Result(cm or float("nan"), cs or 0.0, cov_m, cov_s,
                         str_m, str_s, len(coh_seeds))


def hellinger_distance_matrix(distributions: np.ndarray) -> np.ndarray:
    """Pairwise Hellinger distance between rows.

    Hellinger(p, q) = (1/sqrt(2)) · ||sqrt(p) - sqrt(q)||_2
    """
    sqrt_p = np.sqrt(np.maximum(distributions, 0))
    diff = sqrt_p[:, None, :] - sqrt_p[None, :, :]
    H2 = 0.5 * (diff ** 2).sum(axis=2)
    return np.sqrt(np.maximum(H2, 0))


def gene_ordering_supp(
    top_genes: list[str],
    gene_topic_distrib: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    """Step (ii) gene ordering per supplementary §D.

    Parameters
    ----------
    top_genes : list of gene names
    gene_topic_distrib : dict mapping gene → topic distribution vector

    Returns
    -------
    (ordered_genes, pseudotime_coords)
    """
    present = [g for g in top_genes if g in gene_topic_distrib]
    if len(present) < 3:
        raise ValueError(f"need >= 3 genes with topic distributions, got {len(present)}")
    distribs = np.array([gene_topic_distrib[g] for g in present])
    distribs = distribs / np.maximum(distribs.sum(axis=1, keepdims=True), 1e-12)

    H = hellinger_distance_matrix(distribs)

    # Gaussian kernel with bandwidth = median distance
    sigma = np.median(H[H > 0])
    if sigma == 0:
        raise ValueError("all distances are zero")
    K = np.exp(-(H ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(K, 0.0)  # standard for diffusion maps

    # Diffusion maps: P = D^{-1} K, take EV2 (first non-trivial eigenvector)
    row_sums = K.sum(axis=1)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    P = K / row_sums[:, None]
    eigvals, eigvecs = np.linalg.eig(P)
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    sort_idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, sort_idx]
    pseudotime = eigvecs[:, 1]  # EV2

    order = np.argsort(pseudotime)
    return [present[i] for i in order], pseudotime[order]
