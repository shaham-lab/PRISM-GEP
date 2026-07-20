"""Evaluate Coherence, Coverage, and Strength for completed new-dataset outputs.

This script is method-output driven: it discovers ``topic_keys.txt`` files for
PRISM plus baseline methods, computes the paper-style metrics, and reports
mean/std over requested seeds. It does not silently fabricate missing methods;
the companion coverage table lists how many seeds are available per method.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests


WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS))

from bio.evaluate_supp import coherence_supp, coverage_strength_supp  # noqa: E402
from bio.extract_top_genes import parse_topic_keys  # noqa: E402


METHODS = {
    "PRISM": "outputs/{dataset}/seed{seed}/topic_keys.txt",
    "PRISM_TAU3": "outputs/candidate_screen_tau3/{dataset}/seed{seed}/topic_keys.txt",
    "MALLET": "outputs/baselines/mallet_vanilla/{dataset}/seed{seed}/topic_keys.txt",
    "NMF": "outputs/baselines/nmf/{dataset}/seed{seed}/topic_keys.txt",
    "cNMF": "outputs/baselines/cnmf/{dataset}/seed{seed}/topic_keys.txt",
    "scHPF": "outputs/baselines/schpf/{dataset}/seed{seed}/topic_keys.txt",
    "ProdLDA": "outputs/baselines/prodlda/{dataset}/seed{seed}/topic_keys.txt",
}

FALLBACK_METHOD_PATHS = {
    "PRISM": [
        "outputs/candidate_param_sweep/{dataset}/baseline/seed{seed}/topic_keys.txt",
        "outputs/candidate_screen/{dataset}/seed{seed}/topic_keys.txt",
    ],
    "PRISM_TAU3": [],
    "MALLET": [
        "outputs/candidate_screen_baselines/mallet_vanilla/{dataset}/seed{seed}/topic_keys.txt",
    ],
    "NMF": [
        "outputs/candidate_screen_baselines/nmf/{dataset}/seed{seed}/topic_keys.txt",
    ],
}

DATASET_DIR = {
    "pancreas": "Pancreas",
    "bonemarrow": "BoneMarrow",
    "gastrulation_e75": "gastrulation_e75",
    "gastrulation_erythroid": "gastrulation_erythroid",
    "gastrulation": "gastrulation",
    "dentategyrus": "dentategyrus",
    "paul15": "paul15",
    "pbmc68k": "pbmc68k",
    "pbmc3k": "pbmc3k",
    "breast_cancer": "breast_cancer",
    "zeisel_brain": "zeisel_brain",
    "hemogenic_endothelium": "hemogenic_endothelium",
}

ORGANISM = {
    "pancreas": "mouse",
    "bonemarrow": "human",
    "gastrulation_e75": "mouse",
    "gastrulation_erythroid": "mouse",
    "gastrulation": "mouse",
    "dentategyrus": "mouse",
    "paul15": "mouse",
    "pbmc68k": "human",
    "pbmc3k": "human",
    "breast_cancer": "human",
    "zeisel_brain": "mouse",
}


def data_csv(dataset: str) -> Path:
    d = DATASET_DIR.get(dataset, dataset)
    return WS / "data" / d / f"filtered_{dataset}_cells_x_genes.csv"


def load_library(gene_sets: str) -> dict[str, list[str]]:
    """Download Enrichr gene-set library directly and cache it.

    gseapy.get_library_name currently fails in this environment, while the
    direct Enrichr library endpoint is reachable.
    """
    cache_dir = WS / "outputs" / "gene_set_libraries"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{gene_sets}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    url = f"https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={gene_sets}"
    print(f"[download] {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    library: dict[str, list[str]] = {}
    for line in response.text.splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        term = parts[0]
        genes = [g for g in parts[2:] if g]
        if genes:
            library[term] = genes
    cache_path.write_text(json.dumps(library), encoding="utf-8")
    return library


def top_key_path(dataset: str, method: str, seed: int) -> Path | None:
    templates = [METHODS[method]] + FALLBACK_METHOD_PATHS.get(method, [])
    for template in templates:
        path = WS / template.format(dataset=dataset, seed=seed)
        if path.exists():
            return path
    return None


def evaluate_seed(
    dataset: str,
    method: str,
    seed: int,
    expr: pd.DataFrame,
    library: dict[str, list[str]],
    *,
    top_n: int,
    q_threshold: float,
) -> dict[str, object] | None:
    path = top_key_path(dataset, method, seed)
    if path is None:
        return None
    top = {k: v[:top_n] for k, v in parse_topic_keys(path).items()}
    if not top:
        return None

    universe = list(expr.columns)
    coherence = coherence_supp(top, expr)
    covs, strengths, n_sig_total = [], [], 0
    for genes in top.values():
        cov, strength, n_sig = coverage_strength_supp(
            genes,
            library,
            universe,
            q_threshold=q_threshold,
            use_M_full=True,
        )
        n_sig_total += n_sig
        if cov is not None:
            covs.append(cov)
        if strength is not None:
            strengths.append(strength)

    return {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "coherence": coherence,
        "coverage": float(np.mean(covs)) if covs else np.nan,
        "strength": float(np.mean(strengths)) if strengths else np.nan,
        "n_sig": n_sig_total,
        "topic_keys": str(path.relative_to(WS)),
    }


def aggregate(df: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    expected = len(seeds)
    rows = []
    for keys, group in df.groupby(["dataset", "method"]):
        row = {
            "dataset": keys[0],
            "method": keys[1],
            "n_seeds": int(group["seed"].nunique()),
            "complete_10_seeds": int(group["seed"].nunique()) == expected,
            "n_sig_total": int(pd.to_numeric(group["n_sig"], errors="coerce").fillna(0).sum()),
        }
        for metric in ("coherence", "coverage", "strength"):
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_n"] = int(len(vals))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "method"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["pancreas", "bonemarrow"])
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--gene-sets", default="GO_Biological_Process_2021")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--q-threshold", type=float, default=0.05)
    parser.add_argument("--prism-root", default=None,
                        help="root for PRISM topic_keys (e.g. outputs_betafix); "
                             "overrides the default outputs/<ds>/ and disables "
                             "PRISM fallbacks so buggy runs are never picked up")
    parser.add_argument("--out", default=None,
                        help="also write the aggregate CSV to this path")
    args = parser.parse_args()

    if args.prism_root:
        root = args.prism_root.replace("\\", "/").rstrip("/")
        METHODS["PRISM"] = root + "/{dataset}/seed{seed}/topic_keys.txt"
        FALLBACK_METHOD_PATHS["PRISM"] = []
        print(f"[prism-root] PRISM topic_keys <- {METHODS['PRISM']}")

    library = load_library(args.gene_sets)
    rows = []
    coverage_rows = []
    for dataset in args.datasets:
        csv = data_csv(dataset)
        if not csv.exists():
            print(f"[missing dataset csv] {dataset}: {csv}")
            continue
        expr = pd.read_csv(csv, index_col=0)
        print(f"[dataset] {dataset}: {expr.shape[0]} cells x {expr.shape[1]} genes")
        for method in args.methods:
            n_present = 0
            for seed in args.seeds:
                result = evaluate_seed(
                    dataset,
                    method,
                    seed,
                    expr,
                    library,
                    top_n=args.top_n,
                    q_threshold=args.q_threshold,
                )
                if result is None:
                    continue
                n_present += 1
                rows.append(result)
                print(
                    f"  {method:>7s} seed{seed}: "
                    f"coh={result['coherence']:.4f} "
                    f"cov={result['coverage']:.4f} "
                    f"str={result['strength']:.4f} "
                    f"n_sig={result['n_sig']}"
                )
            coverage_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "available_seeds": n_present,
                    "expected_seeds": len(args.seeds),
                    "complete": n_present == len(args.seeds),
                }
            )

    out_dir = WS / "outputs" / "all_new_dataset_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(out_dir / "method_seed_coverage.csv", index=False)

    if not rows:
        print("No method outputs found.")
        return
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out_dir / "per_seed_metrics.csv", index=False)
    summary = aggregate(per_seed, args.seeds)
    summary.to_csv(out_dir / "aggregate_metrics.csv", index=False)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)
        print(f"wrote {out_path}")

    print("\n=== aggregate Coherence / Coverage / Strength ===")
    print(summary.to_string(index=False))
    print(f"\nwrote {out_dir / 'aggregate_metrics.csv'}")
    print(f"wrote {out_dir / 'per_seed_metrics.csv'}")
    print(f"wrote {out_dir / 'method_seed_coverage.csv'}")


if __name__ == "__main__":
    main()
