"""Gene-embedding ablation (Stage F swap point).

Swaps the gene-gene similarity used by Step (ii) within-GEP ordering and
re-scores against the canonical marker order (same metric as the gene
half of the trajectory evaluation).

Methods supported on CPU:
  - prism  : Stage-A PPMI-based gene-gene similarity (current PRISM-GEP default).
  - log1p  : log1p(mean expression) -> PCA(50) Euclidean similarity. Crude.
  - random : Random N(0,1) embedding. Chance floor.

Methods requiring GPU + checkpoint:
  - scgpt  : Pretrained scGPT gene embeddings (768-d). Skipped here unless
             --scgpt-checkpoint is provided; this scaffolding makes the
             integration trivial once GPU access is available.

Scoring: for each method we (i) embed each top-20 gene of each GEP, (ii)
build a gene-gene cosine-similarity matrix, (iii) run a 1-D diffusion-map
ordering on it (same routine as Stage F), (iv) score |Spearman ρ| against
the canonical marker order from gene_trajectory_baselines.py.

Outputs:
  outputs/gene_embedding_ablation/<ds>/aggregate_metrics.csv
  outputs/gene_embedding_ablation/aggregate_metrics.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS))

from bio.gene_embeddings import embed_genes  # noqa: E402

OUT_ROOT = WS / "outputs" / "gene_embedding_ablation"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def _cosine_sim(E: np.ndarray) -> np.ndarray:
    """Row-normalised cosine similarity."""
    norm = np.linalg.norm(E, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    Z = E / norm
    return Z @ Z.T


def _diffusion_order_1d(sim: np.ndarray) -> np.ndarray:
    """Return a 1-D ordering of rows of `sim` using diffusion EV2.

    Symmetrise sim, threshold negatives, row-stochastic, eigendecompose.
    """
    S = np.maximum(sim, 0.0)
    S = 0.5 * (S + S.T)
    row = S.sum(axis=1, keepdims=True)
    row = np.where(row == 0, 1.0, row)
    P = S / row
    # EV2 of the Markov chain is the diffusion-pseudotime coordinate.
    try:
        w, v = np.linalg.eig(P)
    except np.linalg.LinAlgError:
        return np.arange(P.shape[0])
    idx = np.argsort(-np.real(w))  # descending
    # EV1 ≈ constant; take EV2.
    return np.real(v[:, idx[1]]) if len(idx) > 1 else np.real(v[:, idx[0]])


def order_genes_with_embedding(gene_names, method, *, counts=None,
                                ppmi=None, gene_to_idx=None,
                                scgpt_checkpoint=None, scgpt_vocab=None,
                                scgpt_device="cpu"):
    """Embed and 1-D order a small gene list."""
    E, idx_map = embed_genes(
        gene_names, method=method,
        counts=counts, ppmi=ppmi, gene_to_idx=gene_to_idx,
        scgpt_checkpoint=scgpt_checkpoint, scgpt_vocab=scgpt_vocab,
        scgpt_device=scgpt_device,
    )
    # Keep only rows present in idx_map (scGPT may drop unknown genes).
    present = [g for g in gene_names if g in idx_map]
    if len(present) < 3:
        return None
    rows = [idx_map[g] for g in present]
    sub = E[rows]
    sim = _cosine_sim(sub)
    coord = _diffusion_order_1d(sim)
    order = np.argsort(coord)
    return [present[i] for i in order], coord


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["pancreas", "gastrulation_erythroid", "gastrulation"])
    p.add_argument("--methods", nargs="+",
                   default=["prism", "log1p", "random"])
    p.add_argument("--scgpt-checkpoint", default=None)
    p.add_argument("--scgpt-vocab", default=None)
    p.add_argument("--scgpt-device", default="cpu")
    p.add_argument("--out", default=None,
                   help="output CSV (default: outputs/gene_embedding_ablation/"
                        "aggregate_metrics.csv). Use this for partial runs so a "
                        "subset does not replace the full table.")
    p.add_argument("--append", action="store_true",
                   help="merge into the target CSV on (dataset, method) instead "
                        "of overwriting it.")
    args = p.parse_args()

    # Canonical marker orders. The four original datasets keep their in-code
    # lists so previously published numbers are reproduced byte-for-byte. Any
    # further dataset is read from the per-dataset orders CSV, which is the
    # same (gene, canonical_rank) source the gene-trajectory tables are built
    # from, so the two analyses cannot disagree about the ground-truth order.
    from scripts.gene_trajectory_baselines import (
        MARKERS_GASTRULATION,
        MARKERS_GASTRULATION_ERYTHROID,
        MARKERS_HEMOGENIC,
        MARKERS_PANCREAS,
    )
    MARKERS = {
        "pancreas": MARKERS_PANCREAS,
        "gastrulation": MARKERS_GASTRULATION,
        "gastrulation_erythroid": MARKERS_GASTRULATION_ERYTHROID,
        "hemogenic_endothelium": MARKERS_HEMOGENIC,
    }

    def markers_from_orders_csv(ds):
        """(gene, canonical_rank) pairs from outputs/trajectory/<ds>/..._orders.csv."""
        f = WS / "outputs" / "trajectory" / ds / f"gene_trajectory_{ds}_orders.csv"
        if not f.exists():
            return None
        d = pd.read_csv(f)
        if "gene" not in d.columns or "canonical_rank" not in d.columns:
            return None
        d = d.dropna(subset=["gene", "canonical_rank"])
        return [(str(g), float(r)) for g, r in zip(d["gene"], d["canonical_rank"])]

    for ds in args.datasets:
        if ds not in MARKERS:
            m = markers_from_orders_csv(ds)
            if m:
                MARKERS[ds] = m
                print(f"[{ds}] canonical order read from orders CSV ({len(m)} markers)")
    from scipy.stats import spearmanr

    rows = []
    for ds in args.datasets:
        if ds not in MARKERS:
            print(f"[{ds}] no canonical marker list -- SKIP")
            continue
        markers = MARKERS[ds]
        gene_names = [g for g, _ in markers]
        canon_full = {g: r for g, r in markers}

        # Load counts + PPMI (PPMI needed for prism method).
        csv = WS / "data" / ds / f"filtered_{ds}_cells_x_genes.csv"
        if not csv.exists():
            print(f"[{ds}] {csv} missing -- SKIP")
            continue
        df = pd.read_csv(csv, index_col=0)
        counts = df.values.astype(np.float32)
        gene_to_idx = {g: i for i, g in enumerate(df.columns)}
        # Subset counts to markers (case-insensitive).
        col_lookup = {g.lower(): g for g in df.columns}
        present_markers = [g for g in gene_names if g.lower() in col_lookup]
        if len(present_markers) < 3:
            print(f"[{ds}] <3 marker genes present -- SKIP")
            continue
        sub_cols = [col_lookup[g.lower()] for g in present_markers]
        counts_sub = df[sub_cols].values.astype(np.float32)
        gene_to_idx_sub = {g: i for i, g in enumerate(present_markers)}

        ppmi = None
        if "prism" in args.methods:
            from bio.ppmi_knn_over_cells import compute_gene_gene_ppmi
            print(f"\n[{ds}] computing PPMI for prism method ...")
            ppmi_result = compute_gene_gene_ppmi(counts, list(df.columns),
                                                  n_neighbors=15, n_pca=50)
            ppmi = ppmi_result.ppmi

        # canonical rank aligned to the markers actually present in the dataset
        canon_rank = np.array([canon_full[g] for g in present_markers],
                              dtype=float)
        for method in args.methods:
            kw = {}
            if method == "log1p":
                kw["counts"] = counts_sub
            if method == "prism":
                kw["ppmi"] = ppmi
                kw["gene_to_idx"] = gene_to_idx  # full vocab -> ppmi row
            if method == "scgpt":
                kw["scgpt_checkpoint"] = args.scgpt_checkpoint
                kw["scgpt_vocab"] = args.scgpt_vocab
                kw["scgpt_device"] = args.scgpt_device
            try:
                result = order_genes_with_embedding(present_markers,
                                                    method=method, **kw)
            except Exception as e:
                print(f"  {method}: FAILED -- {e}")
                rows.append({"dataset": ds, "method": method,
                             "spearman_abs": np.nan, "error": str(e)[:120]})
                continue
            if result is None:
                continue
            ordered, _ = result
            # Score: recovered position of each marker vs canonical rank.
            # Some embeddings (notably scGPT human-only vocab) may drop a few
            # markers; align both arrays to the genes actually embedded.
            recovered_pos = {g: i for i, g in enumerate(ordered)}
            scored = [g for g in present_markers if g in recovered_pos]
            if len(scored) < 3:
                print(f"  {method}: <3 markers after embedding -- SKIP")
                continue
            scored_canon = np.array([canon_full[g] for g in scored],
                                    dtype=float)
            recovered = np.array([recovered_pos[g] for g in scored],
                                 dtype=float)
            rho, _ = spearmanr(recovered, scored_canon)
            rho_abs = abs(rho)
            n_used = len(scored)
            n_drop = len(present_markers) - n_used
            print(f"  {method:8s}  |rho|={rho_abs:.3f}  "
                  f"n_used={n_used} dropped={n_drop}  "
                  f"order={ordered[:5]}...")
            rows.append({"dataset": ds, "method": method,
                         "spearman_abs": rho_abs,
                         "n_genes": n_used,
                         "n_dropped": n_drop})

    if rows:
        df_out = pd.DataFrame(rows)
        # Default target is the published aggregate. Writing it unconditionally
        # meant a partial run (say one dataset) silently replaced the full
        # published table, so --out lets a partial or exploratory run go
        # somewhere else, and --append merges instead of clobbering.
        out = Path(args.out) if args.out else OUT_ROOT / "aggregate_metrics.csv"
        if args.append and out.exists():
            prev = pd.read_csv(out)
            key = ["dataset", "method"]
            merged = pd.concat([prev, df_out], ignore_index=True)
            df_out = merged.drop_duplicates(subset=key, keep="last")
        out.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out, index=False)
        print(f"\nwrote {out}")
        print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
