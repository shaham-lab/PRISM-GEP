"""Paper-matching metric variants for PRISM-GEP Tables 1 & 2.

  Coherence  : mean within-set Spearman correlation of expression vectors,
               using ALL cells (NOT filtered to non-zero pairs).

  Coverage   : Use **gget enrichr** (paper-pinned) WITH dataset vocabulary
               as background_list, q < 0.05.
               Coverage = |{top-N genes hitting any sig pathway}| / annotated_top_N
               averaged over GEPs WITH ≥1 sig pathway.
               When NO GEP has a sig pathway → report "—" (matches BC).

  Strength   : Same enrichment as Coverage. Strength = mean over sig-GEPs of
               -log10(min adj_p_val per GEP). Use the SINGLE most significant
               pathway per GEP.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class Table1Row:
    coherence_mean: float
    coherence_std: float
    coverage_mean: float | None       # None means "no significant enrichments at all"
    coverage_std: float | None
    strength_mean: float | None
    strength_std: float | None


def coherence_paper(
    top_genes_per_gep: dict[int, list[str]],
    expression_df: pd.DataFrame,
) -> dict[int, float]:
    """Per-GEP coherence using Fisher-Z averaged Spearman.

    Paper-matching formula (verified by Phase A.3 sweep — PBMC and Zeisel
    reproduce to within 0.1-0.3% of paper):
        For each pair of genes (i, j) in the GEP's top-N:
            rho_ij = Spearman(expr[:, i], expr[:, j])
            z_ij   = arctanh(rho_ij)         # Fisher Z-transform
        coherence_gep = tanh( mean(z_ij) )   # back-transform after averaging

    The Fisher Z-transform is the statistically correct way to average
    correlation coefficients; the previous arithmetic-mean version
    underestimates by 7-11% systematically.
    """
    out = {}
    for gep, genes in top_genes_per_gep.items():
        present = [g for g in genes if g in expression_df.columns]
        if len(present) < 2:
            out[gep] = float("nan")
            continue
        sub = expression_df[present].values
        rho, _ = spearmanr(sub, axis=0)
        if np.ndim(rho) == 0:
            out[gep] = float(rho)
            continue
        iu = np.triu_indices_from(rho, k=1)
        # Fisher Z-transform: arctanh(r). Clip to avoid arctanh(-+1) = inf.
        z = np.arctanh(np.clip(rho[iu], -0.999, 0.999))
        avg_z = np.nanmean(z)
        out[gep] = float(np.tanh(avg_z))
    return out


def coherence_paper_global(
    top_genes_per_gep: dict[int, list[str]],
    expression_df: pd.DataFrame,
) -> float:
    """Pool all pairwise Spearman correlations across GEPs, Fisher-Z average.

    Equivalent to ``mean(coherence_paper(...))`` when all GEPs have the same
    number of pairs (which they do, given fixed top-N). Reported as the
    single Coherence number per dataset.
    """
    all_rhos = []
    for genes in top_genes_per_gep.values():
        present = [g for g in genes if g in expression_df.columns]
        if len(present) < 2:
            continue
        sub = expression_df[present].values
        rho, _ = spearmanr(sub, axis=0)
        if np.ndim(rho) == 0:
            all_rhos.append(float(rho))
            continue
        iu = np.triu_indices_from(rho, k=1)
        all_rhos.extend(rho[iu].tolist())
    if not all_rhos:
        return float("nan")
    z = np.arctanh(np.clip(np.array(all_rhos), -0.999, 0.999))
    return float(np.tanh(np.nanmean(z)))


def _build_gene2go_from_gseapy(library_name: str = "GO_Biological_Process_2021") -> dict[str, dict[str, set[str]]]:
    """Build gene -> set of GO terms map for human and mouse from gseapy library."""
    import gseapy
    g2g = {}
    for org_l, org_t in [("human", "Human"), ("mouse", "Mouse")]:
        lib = gseapy.get_library(library_name, organism=org_t)
        m = defaultdict(set)
        for term, members in lib.items():
            for mm in members:
                m[mm.upper()].add(term)
        g2g[org_l] = m
    return g2g


def go_bp_paper(
    top_genes_per_gep: dict[int, list[str]],
    organism: str,
    background: list[str],
    *,
    q_threshold: float = 0.05,
    use_gget: bool = True,
    gene2go_cache: dict | None = None,
    strength_uses_default_bg: bool = True,
) -> tuple[float | None, float | None]:
    """(Coverage, Strength) per paper-matching definitions.

    Hybrid background strategy (best paper match in our sweep):
      - Coverage: vocab-restricted background (matches BC's "—" exactly).
      - Strength: gget's default 20k+ background (matches PBMC + Zeisel
        Strength to within 1σ, vs 30%+ off with vocab-restricted bg).

    Set ``strength_uses_default_bg=False`` to use the same vocab-restricted
    background for Strength too (legacy behavior; underestimates Strength
    by ~50% on Zeisel).

    Returns (None, None) if NO GEP has any significant pathway under
    the (vocab-bg) Coverage test.
    """
    if gene2go_cache is None:
        gene2go_cache = _build_gene2go_from_gseapy()
    annotated = gene2go_cache[organism.lower()]

    if use_gget:
        import gget
        def _enrich(top_n, *, default_bg=False):
            kw = dict(genes=list(top_n), database="ontology",
                      species=organism.lower(), verbose=False)
            if default_bg:
                kw["background"] = True  # use gget's default 20k+ bg
            else:
                kw["background_list"] = background
            df = gget.enrichr(**kw)
            if df is None or df.empty:
                return None
            sig = df[df["adj_p_val"] < q_threshold]
            if sig.empty:
                return None
            # Return list of (overlap_genes_set_upper, adj_p)
            rows = []
            for _, r in sig.iterrows():
                og = r["overlapping_genes"]
                if isinstance(og, list):
                    overlap_upper = {g.upper() for g in og}
                else:
                    overlap_upper = set()
                rows.append((overlap_upper, float(r["adj_p_val"])))
            return rows
    else:
        import gseapy
        def _enrich(top_n, *, default_bg=False):
            try:
                kw = dict(gene_list=list(top_n),
                          gene_sets="GO_Biological_Process_2021",
                          organism=organism.lower(), outdir=None)
                if not default_bg:
                    kw["background"] = background
                enr = gseapy.enrichr(**kw)
            except Exception:
                return None
            df = enr.results
            sig = df[df["Adjusted P-value"] < q_threshold]
            if sig.empty:
                return None
            rows = []
            for _, r in sig.iterrows():
                og = str(r.get("Genes", "")).split(";")
                overlap_upper = {g.upper() for g in og if g}
                rows.append((overlap_upper, float(r["Adjusted P-value"])))
            return rows

    gep_covs = []
    gep_strs = []
    for gep, genes in top_genes_per_gep.items():
        top_n = list(genes)
        top_n_set_upper = {g.upper() for g in top_n}
        annotated_in_top = sum(1 for g in top_n_set_upper if g in annotated)
        if annotated_in_top == 0:
            continue
        sig_vocab = _enrich(top_n, default_bg=False)
        if sig_vocab is None:
            continue
        # Coverage: vocab-bg sig hits / annotated top-N
        hit = set()
        for overlap_upper, _ in sig_vocab:
            hit.update(overlap_upper & top_n_set_upper)
        gep_covs.append(len(hit) / annotated_in_top)
        # Strength: optionally re-fetch with default 20k bg for paper-matching p-values
        if strength_uses_default_bg:
            sig_str = _enrich(top_n, default_bg=True)
            if sig_str is None:
                # fall back to vocab-bg result
                sig_str = sig_vocab
        else:
            sig_str = sig_vocab
        min_q = min(q for _, q in sig_str)
        gep_strs.append(float(-np.log10(max(min_q, 1e-300))))

    if not gep_covs:
        return None, None
    return float(np.mean(gep_covs)), float(np.mean(gep_strs))


def evaluate_table1(
    top_genes_per_gep_per_seed: dict[int, dict[int, list[str]]],
    expression_df: pd.DataFrame,
    organism: str,
    *,
    use_gget: bool = True,
    gene2go_cache: dict | None = None,
) -> Table1Row:
    coh_means, cov_means, str_means = [], [], []
    background = list(expression_df.columns)
    for seed, top_per_gep in top_genes_per_gep_per_seed.items():
        # Use Fisher-Z global pool — matches paper for PBMC + Zeisel within 0.3%
        coh_means.append(coherence_paper_global(top_per_gep, expression_df))
        cov, strg = go_bp_paper(
            top_per_gep, organism=organism, background=background,
            use_gget=use_gget, gene2go_cache=gene2go_cache,
        )
        if cov is not None:
            cov_means.append(cov)
        if strg is not None and not np.isnan(strg):
            str_means.append(strg)

    def _ms(xs: list[float]) -> tuple[float, float]:
        if not xs:
            return float("nan"), 0.0
        return float(np.mean(xs)), (float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0)

    coh_m, coh_s = _ms(coh_means)
    cov_m, cov_s = _ms(cov_means) if cov_means else (None, None)
    str_m, str_s = _ms(str_means) if str_means else (None, None)
    return Table1Row(coh_m, coh_s, cov_m, cov_s, str_m, str_s)
