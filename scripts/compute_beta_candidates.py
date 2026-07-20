"""Compute PRISM-GEP beta priors for prepared candidate datasets.

Computes the Stage A-D beta prior for one or more datasets and writes it into the
layout that ``scripts/train_prism_standard.py`` reads:

  outputs/candidate_screen/<dataset>/beta_prism.csv

It also writes the Stage A-D intermediate artifacts through ``bio.pipeline``.
"""
from __future__ import annotations

import argparse
from pathlib import Path


WS = Path(__file__).resolve().parent.parent


def dataset_csv(dataset: str) -> Path:
    path = WS / "data" / dataset / f"filtered_{dataset}_cells_x_genes.csv"
    if path.exists():
        return path
    aliases = {"pancreas": "Pancreas", "bonemarrow": "BoneMarrow"}
    alias = aliases.get(dataset)
    if alias is not None:
        return WS / "data" / alias / f"filtered_{dataset}_cells_x_genes.csv"
    return path


def compute(dataset: str, args) -> None:
    import sys

    sys.path.insert(0, str(WS))
    from bio.pipeline import run_pipeline

    csv = dataset_csv(dataset)
    if not csv.exists():
        raise FileNotFoundError(csv)

    out_dir = WS / args.output_root / dataset
    print(f"\n=== {dataset} ===")
    run_pipeline(
        counts_csv=csv,
        output_dir=out_dir,
        K=args.topics,
        m=args.diffusion_components,
        n_neighbors=args.n_neighbors,
        n_pca=args.n_pca,
        expression_threshold=args.expression_threshold,
        neighborhood_min_support=args.neighborhood_min_support,
        binary_neighborhood_indicator=True,
        random_state=args.random_state,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--topics", type=int, default=5)
    parser.add_argument("--diffusion-components", type=int, default=20)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-pca", type=int, default=50)
    parser.add_argument("--expression-threshold", type=float, default=2.0,
                        help="Per-cell count cutoff (production value: 2.0)")
    parser.add_argument("--neighborhood-min-support", type=int, default=1)
    parser.add_argument("--output-root", default="outputs/candidate_screen")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    for dataset in args.datasets:
        compute(dataset, args)


if __name__ == "__main__":
    main()
