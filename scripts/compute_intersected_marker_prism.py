"""Compute PRISM Step (ii) |Spearman rho| against canonical marker order on
the marker subset that scGPT can also embed (intersected-marker fairness
adjustment).

scGPT drops markers that are not in its public whole-human vocabulary
(`data/scgpt/vocab.json`). The matcher in
`scripts/scgpt_contextual_ablation.py` is `g.upper() in vocab`,
so mouse paralogs like Ins1/Ins2 (human has only INS) and the renamed
mouse T gene (human is TBXT) drop out, along with the mouse embryonic
globins Hba-x, Hbb-y, Hbb-bh1.

This script reads the per-marker gene-trajectory order CSVs and
recomputes Spearman vs canonical_rank on (i) the full marker set and
(ii) the intersected (= scGPT-embeddable) marker set.

Writes a CSV + prints a summary table for the gene-embedding comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

WS = Path(__file__).resolve().parent.parent
SCG_VOCAB = WS / "data" / "scgpt" / "vocab.json"
TRAJ = WS / "outputs" / "trajectory"
GEA = WS / "outputs" / "gene_embedding_ablation"
OUT = GEA / "intersected_marker_prism.csv"
AGG = GEA / "aggregate_metrics.csv"

# The four datasets of the published Table S11 plus the five further datasets
# that also carry a `canonical_rank` column in their gene-trajectory orders CSV.
PUBLISHED_DATASETS = [
    "pancreas",
    "gastrulation",
    "gastrulation_erythroid",
    "hemogenic_endothelium",
]
EXTRA_DATASETS = [
    "bonemarrow",
    "paul15",
    "dentategyrus",
    "endoderm_diff",
    "gastrulation_e75",
]
ALL_DATASETS = PUBLISHED_DATASETS + EXTRA_DATASETS


def orders_csv(ds: str) -> Path:
    return TRAJ / ds / f"gene_trajectory_{ds}_orders.csv"


def load_scgpt_scores(paths: list[Path]) -> pd.DataFrame:
    """Concatenate scGPT rows from every supplied metrics CSV (last wins)."""
    frames = [pd.read_csv(p) for p in paths if p.exists()]
    if not frames:
        return pd.DataFrame(columns=["dataset", "method", "spearman_abs"])
    df = pd.concat(frames, ignore_index=True)
    df = df[df["method"].isin(["scgpt", "scgpt_contextual"])]
    return df.drop_duplicates(subset=["dataset", "method"], keep="last")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument(
        "--scgpt-csv", nargs="+",
        default=[str(AGG), str(GEA / "extra5_scgpt.csv"),
                 str(GEA / "extra5_scgpt_contextual.csv")],
        help="metrics CSVs to pull the scGPT columns from, in increasing "
             "precedence order.",
    )
    args = ap.parse_args()

    vocab = set(json.load(open(SCG_VOCAB)).keys())
    scg = load_scgpt_scores([Path(p) for p in args.scgpt_csv])
    scg = scg.set_index(["dataset", "method"])["spearman_abs"]

    def lookup(ds: str, method: str) -> float:
        try:
            return float(scg.loc[(ds, method)])
        except KeyError:
            return float("nan")

    rows = []
    print(f"{'dataset':25s} {'n_full':>6s} {'PRISM_full':>10s} "
          f"{'n_int':>5s} {'PRISM_int':>10s} {'scGPT_static':>13s} {'scGPT_ctx':>10s} {'dropped'}")
    for ds in args.datasets:
        path = orders_csv(ds)
        if not path.exists():
            print(f"{ds:25s} orders CSV missing -- SKIP ({path})")
            continue
        df = pd.read_csv(path)
        if "canonical_rank" not in df.columns or "PRISM_K5_StepII" not in df.columns:
            print(f"{ds:25s} no canonical_rank/PRISM_K5_StepII column -- SKIP")
            continue
        df = df.dropna(subset=["gene", "canonical_rank", "PRISM_K5_StepII"])
        n_full = len(df)
        in_vocab = df["gene"].apply(lambda g: str(g).upper() in vocab)
        dropped = df.loc[~in_vocab, "gene"].tolist()
        sub = df[in_vocab]
        full_rho = abs(spearmanr(df["PRISM_K5_StepII"], df["canonical_rank"]).statistic)
        int_rho = (abs(spearmanr(sub["PRISM_K5_StepII"], sub["canonical_rank"]).statistic)
                   if len(sub) >= 3 else np.nan)
        scg_static = lookup(ds, "scgpt")
        scg_ctx = lookup(ds, "scgpt_contextual")
        print(f"{ds:25s} {n_full:>6d} {full_rho:>10.3f} {len(sub):>5d} {int_rho:>10.3f} "
              f"{scg_static:>13.3f} {scg_ctx:>10.3f}  dropped={dropped}")
        rows.append({
            "dataset": ds, "n_full": n_full, "PRISM_full": full_rho,
            "n_int": len(sub), "PRISM_int": int_rho,
            "scGPT_static": scg_static, "scGPT_ctx": scg_ctx,
            "dropped": ";".join(map(str, dropped)),
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWROTE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
