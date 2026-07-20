"""LLM-based coherence per Hu et al. 2025.

Reference: Hu et al., Nature Methods 2025
(doi: 10.1038/s41592-024-02525-x), "Evaluation of large language models for
discovery of gene set function".

Approach:
    For each GEP, send its top-N genes to GPT-4 and ask:
        "Do these genes co-participate in a shared biological process?
         Return a confidence score from 0.0 (no shared process) to
         1.0 (clear shared process)."
    The mean confidence across GEPs (and seeds) is the LLM coherence.

Requires OpenAI API access (`pip install openai`, OPENAI_API_KEY env var).
Estimated cost: ~$0.03 per datasetXseed = $0.90 for full reproduction.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


# Prompt adapted from Hu et al. 2025 supplementary, modified for gene-set
# coherence specifically (not full functional naming).
HU_PROMPT_SYSTEM = (
    "You are an expert biologist evaluating whether a list of genes "
    "co-participates in a shared biological process. Respond with a "
    "single confidence score between 0.0 and 1.0:\n"
    "  - 0.0 = the genes have NO obvious shared biological function\n"
    "  - 0.5 = some genes share function, others do not\n"
    "  - 1.0 = ALL or nearly all genes participate in a clearly shared "
    "biological process\n"
    "Return ONLY the numeric score, nothing else."
)

HU_PROMPT_USER_TEMPLATE = (
    "Genes ({n}): {gene_list}\n\n"
    "Confidence (0.0-1.0):"
)


@dataclass
class LLMResult:
    score: float
    raw_response: str
    n_retries: int


def score_gep_with_gpt4(
    top_genes: Sequence[str],
    *,
    model: str = "gpt-4-turbo",
    api_key: str | None = None,
    max_retries: int = 3,
    temperature: float = 0.0,
) -> LLMResult:
    """Score one GEP's coherence with GPT-4. Returns confidence ∈ [0, 1]."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("`pip install openai` required for LLM coherence") from e

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    user_msg = HU_PROMPT_USER_TEMPLATE.format(
        n=len(top_genes), gene_list=", ".join(top_genes)
    )

    last_response = ""
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": HU_PROMPT_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temperature,
                max_tokens=10,
            )
            text = resp.choices[0].message.content.strip()
            last_response = text
            score = _parse_confidence(text)
            if score is not None:
                return LLMResult(score=score, raw_response=text, n_retries=attempt)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.0 * (attempt + 1))  # exponential backoff
    raise RuntimeError(
        f"failed after {max_retries} retries; last response: {last_response!r}; "
        f"last error: {last_err}"
    )


def _parse_confidence(text: str) -> float | None:
    """Extract a float in [0, 1] from a GPT response."""
    m = re.search(r"(?<![\w.])([01](?:\.\d+)?|0?\.\d+)(?![\w.])", text)
    if not m:
        return None
    val = float(m.group(1))
    return val if 0.0 <= val <= 1.0 else None


def score_dataset(
    top_genes_per_gep_per_seed: dict[int, dict[int, list[str]]],
    *,
    model: str = "gpt-4-turbo",
    api_key: str | None = None,
    cache_path: Path | None = None,
    verbose: bool = True,
) -> tuple[float, float]:
    """Score all (seed, GEP) pairs of a dataset, return (mean, std) over seeds.

    With ``cache_path``, scores are persisted as JSON so re-runs don't re-call
    the API.
    """
    cache: dict[str, float] = {}
    if cache_path is not None and cache_path.exists():
        cache = json.loads(cache_path.read_text())

    seed_means = []
    for seed, top_per_gep in top_genes_per_gep_per_seed.items():
        gep_scores = []
        for gep, genes in top_per_gep.items():
            cache_key = f"{seed}_{gep}_{','.join(genes[:20])}"
            if cache_key in cache:
                gep_scores.append(cache[cache_key])
                continue
            try:
                res = score_gep_with_gpt4(
                    genes[:20], model=model, api_key=api_key
                )
                gep_scores.append(res.score)
                cache[cache_key] = res.score
                if cache_path is not None:
                    cache_path.write_text(json.dumps(cache, indent=2))
                if verbose:
                    print(f"  seed{seed} GEP{gep}: {res.score:.3f}")
            except Exception as e:
                if verbose:
                    print(f"  seed{seed} GEP{gep}: FAILED ({e})")
        if gep_scores:
            seed_means.append(np.mean(gep_scores))

    if not seed_means:
        return float("nan"), float("nan")
    m = float(np.mean(seed_means))
    s = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
    return m, s


def main_cli():
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["breast_cancer", "pbmc3k", "zeisel_brain"])
    parser.add_argument("--model", default="gpt-4-turbo")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--cache", default=None,
                        help="JSON file to cache scores")
    args = parser.parse_args()

    WS = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(WS))
    from bio.extract_top_genes import (
        parse_topic_keys, parse_word_topic_counts, top_n_genes_from_counts,
    )

    per_seed = {}
    for seed in args.seeds:
        sd = WS / "outputs" / args.dataset / f"seed{seed}"
        keys = sd / "topic_keys.txt"
        if keys.exists():
            top = parse_topic_keys(keys)
            if top and all(len(v) >= args.top_n for v in top.values()):
                per_seed[seed] = {t: top[t][:args.top_n] for t in top}
                continue
        counts = sd / "word_topic_counts.txt"
        if counts.exists():
            per_seed[seed] = top_n_genes_from_counts(
                parse_word_topic_counts(counts), n=args.top_n
            )

    print(f"=== {args.dataset} LLM coherence ({args.model}, n={len(per_seed)} seeds) ===")
    cache = Path(args.cache) if args.cache else (
        WS / "outputs" / args.dataset / f"llm_coherence_{args.model}_cache.json"
    )
    cache.parent.mkdir(parents=True, exist_ok=True)

    m, s = score_dataset(per_seed, model=args.model, cache_path=cache)
    print(f"\nMean ± std over {len(per_seed)} seeds: {m:.4f} ± {s:.4f}")



if __name__ == "__main__":
    main_cli()
