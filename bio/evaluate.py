"""Bio-domain evaluation metrics for PRISM-GEP.

Gene-set quality (GO BP, q < 0.05):
    Coherence  : mean within-set Spearman correlation of gene expression vectors
    Coverage   : fraction of GO BP pathway members recovered, averaged over
                 significant pathways
    Strength   : −log₁₀(q), FDR-adjusted, mean over significant pathways

LLM plausibility:
    GPT-4 coherence: confidence ∈ [0,1] that top genes co-participate in a
                     shared Biological Process, per Hu et al. 2025
                     (doi: 10.1038/s41592-024-02525-x)

Usage:
    from bio.evaluate import coherence_spearman, go_bp_enrichment, gpt4_coherence
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr



def coherence_spearman(
    top_genes: Sequence[str],
    expression_df: pd.DataFrame,
) -> float:
    """Mean within-set pairwise Spearman correlation of expression vectors.

    Parameters
    ----------
    top_genes : sequence of str
        Top-N genes for one GEP (typically N=20).
    expression_df : pd.DataFrame, shape (n_cells, n_genes)
        Cell × gene expression matrix. Column names must include all of
        ``top_genes``. Values are raw counts (or any monotone transform —
        Spearman is rank-invariant).

    Returns
    -------
    float
        Mean of upper-triangle entries of the Spearman correlation matrix
        among the top-N genes' expression vectors. NaN if N < 2.
    """
    present = [g for g in top_genes if g in expression_df.columns]
    if len(present) < 2:
        return float("nan")
    sub = expression_df[present].values  # (n_cells, N')
    rho, _ = spearmanr(sub, axis=0)
    if np.ndim(rho) == 0:
        # Only 2 genes → spearmanr returns scalar.
        return float(rho)
    iu = np.triu_indices_from(rho, k=1)
    return float(np.nanmean(rho[iu]))


def coherence_per_gep(
    top_genes_per_gep: dict[int, list[str]],
    expression_df: pd.DataFrame,
) -> dict[int, float]:
    """Apply ``coherence_spearman`` to each GEP."""
    return {
        gep_id: coherence_spearman(genes, expression_df)
        for gep_id, genes in top_genes_per_gep.items()
    }



@dataclass
class EnrichmentSummary:
    coverage: float       # mean fraction of pathway members recovered
    strength: float       # mean −log10(q)
    n_significant: int    # # pathways with q < 0.05
    raw: pd.DataFrame     # the full enrichr DataFrame for inspection


def go_bp_enrichment(
    top_genes: Sequence[str],
    organism: str = "human",
    *,
    q_threshold: float = 0.05,
    gene_sets: str = "GO_Biological_Process_2021",
    background: list[str] | None = None,
) -> EnrichmentSummary:
    """GO BP enrichment via gseapy.enrichr.

    Parameters
    ----------
    top_genes : sequence of str
        Top-N genes for one GEP.
    organism : {"human", "mouse"}
        Species of the dataset the GEP was fitted on.
    q_threshold : float
        FDR-adjusted significance cutoff. Paper uses 0.05.
    gene_sets : str
        gseapy enrichr library name.
    background : list[str] or None
        Background gene set (the dataset's full vocabulary). Recommended for
        small vocabularies like BreastCancer (V=297) where universe matters.

    Returns
    -------
    EnrichmentSummary
        coverage, strength, n_significant, and raw enrichr DataFrame.
        coverage = strength = 0.0 (and the cell shows "—" in the paper) when
        n_significant == 0.
    """
    try:
        import gseapy
    except ImportError as e:
        raise ImportError(
            "gseapy not installed — `pip install gseapy`. Required for GO BP "
            "enrichment."
        ) from e

    enr = gseapy.enrichr(
        gene_list=list(top_genes),
        gene_sets=gene_sets,
        organism=organism,
        background=background,
        outdir=None,  # don't write to disk
    )
    df = enr.results.copy()
    if df.empty:
        return EnrichmentSummary(0.0, 0.0, 0, df)

    sig = df[df["Adjusted P-value"] < q_threshold].copy()
    if sig.empty:
        return EnrichmentSummary(0.0, 0.0, 0, df)

    # Coverage: fraction of pathway members recovered, averaged over sig pathways.
    # Two output paths from gseapy.enrichr depending on whether custom background
    # was used:
    #   - No background:  has "Overlap" column = "k/N" string (k=overlap, N=pathway size)
    #   - With background: has "Genes" column = semicolon-separated overlap genes,
    #     but no pathway size. We derive pathway size from the GO library itself.
    if "Overlap" in sig.columns:
        def _coverage(overlap: str) -> float:
            k, n = overlap.split("/")
            return int(k) / int(n)
        sig["coverage"] = sig["Overlap"].map(_coverage)
    else:
        # Custom-background path. enrichr returns Genes (semicolon-separated
        # overlap) but not pathway size; look the pathway sizes up from the
        # GO library directly. Use FULL pathway size (no bg restriction) so
        # coverage = (top-N ∩ pathway) / |pathway| matches the paper's
        # "fraction of pathway members recovered" definition.
        # gseapy organism arg is title-case ("Human", "Mouse").
        org_titlecase = {"human": "Human", "mouse": "Mouse"}.get(
            organism.lower(), organism
        )
        try:
            library = gseapy.get_library(name=gene_sets, organism=org_titlecase)
        except Exception:
            library = None

        def _pathway_full_size(term: str) -> int | None:
            if library is None:
                return None
            for key in (term, term.split(" (")[0]):
                if key in library:
                    return max(len(library[key]), 1)
            return None

        def _overlap_count(genes_str: str) -> int:
            return len([g for g in genes_str.split(";") if g])

        sig["overlap_k"] = sig["Genes"].map(_overlap_count)
        sig["pathway_n"] = sig["Term"].map(_pathway_full_size)
        # Drop rows where pathway size lookup failed (rather than fabricate one)
        sig = sig[sig["pathway_n"].notna()].copy()
        if sig.empty:
            return EnrichmentSummary(0.0, 0.0, 0, df)
        sig["coverage"] = sig["overlap_k"] / sig["pathway_n"]
    coverage = float(sig["coverage"].mean())

    # Strength: mean −log10(q)
    sig["strength"] = -np.log10(sig["Adjusted P-value"].clip(lower=1e-300))
    strength = float(sig["strength"].mean())

    return EnrichmentSummary(coverage, strength, len(sig), df)


def go_bp_per_gep(
    top_genes_per_gep: dict[int, list[str]],
    organism: str,
    *,
    background: list[str] | None = None,
) -> dict[int, EnrichmentSummary]:
    return {
        gep_id: go_bp_enrichment(genes, organism, background=background)
        for gep_id, genes in top_genes_per_gep.items()
    }



# Prompt template adapted from Hu et al., Nature Methods 2025
# (doi: 10.1038/s41592-024-02525-x).
_GPT4_PROMPT_SYSTEM = (
    "You are a bioinformatics expert evaluating gene set coherence. "
    "Given a set of genes, judge whether they co-participate in a shared "
    "biological process and return a single confidence score between 0.0 "
    "(no shared process) and 1.0 (clearly shared process). Reply with the "
    "score only, no explanation."
)
_GPT4_PROMPT_USER = (
    "Genes: {genes}\n\n"
    "Confidence (0.0–1.0):"
)


def gpt4_coherence(
    top_genes: Sequence[str],
    *,
    model: str = "gpt-4-turbo",
    api_key: str | None = None,
    max_retries: int = 3,
) -> float:
    """Score one GEP's coherence with GPT-4 per Hu et al. 2025.

    Parameters
    ----------
    top_genes : sequence of str
    model : str
        OpenAI model name. Hu et al. used GPT-4; "gpt-4-turbo" is a reasonable
        default. Paper does not specify exact snapshot.
    api_key : str or None
        Defaults to OPENAI_API_KEY env var.
    max_retries : int
        Simple retry on transient errors.

    Returns
    -------
    float
        Confidence ∈ [0, 1]. NaN on persistent failure.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "openai package not installed — `pip install openai`."
        ) from e

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    user_msg = _GPT4_PROMPT_USER.format(genes=", ".join(top_genes))

    last_err = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _GPT4_PROMPT_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=10,
            )
            text = resp.choices[0].message.content.strip()
            score = _parse_confidence(text)
            if score is not None:
                return score
        except Exception as e:  # noqa: BLE001
            last_err = e
    if last_err is not None:
        print(f"[gpt4_coherence] giving up after {max_retries} retries: {last_err}")
    return float("nan")


def _parse_confidence(text: str) -> float | None:
    """Extract a float in [0,1] from a GPT response."""
    import re
    m = re.search(r"(?<![\w.])([01](?:\.\d+)?|0?\.\d+)", text)
    if not m:
        return None
    val = float(m.group(1))
    return val if 0.0 <= val <= 1.0 else None


def gpt4_per_gep(
    top_genes_per_gep: dict[int, list[str]],
    *, model: str = "gpt-4-turbo",
) -> dict[int, float]:
    return {
        gep_id: gpt4_coherence(genes, model=model)
        for gep_id, genes in top_genes_per_gep.items()
    }



@dataclass
class DatasetMetrics:
    coherence: float       # mean over GEPs
    coverage: float        # mean over GEPs (0 if no significant pathways anywhere)
    strength: float        # mean over GEPs (NaN if no significant pathways anywhere)
    llm_score: float       # mean over GEPs


def aggregate_per_dataset(
    top_genes_per_gep: dict[int, list[str]],
    expression_df: pd.DataFrame,
    organism: str,
    *,
    background: list[str] | None = None,
    use_llm: bool = False,
) -> DatasetMetrics:
    """Compute the gene-set quality metrics (and the LLM score if use_llm=True)
    for one model+dataset."""
    coh = coherence_per_gep(top_genes_per_gep, expression_df)
    enr = go_bp_per_gep(top_genes_per_gep, organism, background=background)

    coh_mean = float(np.nanmean(list(coh.values())))
    cov_mean = float(np.mean([s.coverage for s in enr.values()]))
    strengths = [s.strength for s in enr.values() if s.n_significant > 0]
    str_mean = float(np.mean(strengths)) if strengths else float("nan")

    if use_llm:
        llm = gpt4_per_gep(top_genes_per_gep)
        llm_mean = float(np.nanmean(list(llm.values())))
    else:
        llm_mean = float("nan")

    return DatasetMetrics(coh_mean, cov_mean, str_mean, llm_mean)
