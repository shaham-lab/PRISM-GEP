"""Dedicated cell-trajectory baselines (Slingshot, DPT, PAGA-DPT, PCA-1) on every dataset that
has per-cell labels, so the PRISM column from traj_cell_all_datasets.py has something to be
compared against.

Baseline implementations are imported verbatim from trajectory_baselines.py (the published
harness) and the lineage definitions come from traj_cell_all_datasets.ORDERINGS, so each row is
scored against exactly the same rank vector and the same kept-cell subset as the PRISM column.

These methods are deterministic single runs, matching how the shipped table reports them (no
seed std), whereas PRISM is averaged over 10 seeds.

    python scripts/traj_cell_baselines_all.py --datasets pancreas      # validate vs shipped
    python scripts/traj_cell_baselines_all.py                          # all
Writes outputs/trajectory/cell_traj_baselines_all.csv
"""
from __future__ import annotations
import argparse, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS)); sys.path.insert(0, str(WS / "scripts"))
warnings.filterwarnings("ignore")
from traj_cell_all_datasets import ORDERINGS, load, paths          # noqa: E402
from trajectory_baselines import (                                  # noqa: E402  published harness
    dpt_pseudotime, paga_dpt_pseudotime, slingshot_pseudotime, pca_first_coord, spearman_abs,
)

# Shipped values to validate the harness against (supplementary tab:traj_cell).
SHIPPED = {"pancreas": {"Slingshot": .943, "DPT": .930},
           "gastrulation_erythroid": {"Slingshot": .778, "DPT": .849},
           "gastrulation": {"Slingshot": .755, "DPT": .607}}


def build_adata(ds):
    """AnnData of the kept cells, obs['lineage'] set, matching the PRISM column's subset."""
    import anndata as ad
    import scipy.sparse as sp
    _, cnt_p = paths(ds)
    ids, lab_of, rank, keep, order = load(ds)
    X = pd.read_csv(cnt_p, index_col=0)
    A = ad.AnnData(X=sp.csr_matrix(X.to_numpy(np.float32)[keep]))
    A.obs_names = [str(c) for c in np.asarray(ids)[keep]]
    A.var_names = X.columns.astype(str)
    A.obs["lineage"] = pd.Series(lab_of[keep], index=A.obs_names)
    return A, rank[keep], order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(ORDERINGS))
    a = ap.parse_args()
    rows = []
    for ds in a.datasets:
        t0 = time.time()
        try:
            A, rank, order = build_adata(ds)
        except Exception as e:  # noqa: BLE001
            print(f"[{ds}] build failed: {e!r}", flush=True); continue
        root = order[0]
        print(f"[{ds}] {A.n_obs} cells x {A.n_vars} genes, root={root!r}", flush=True)
        pts = {}
        import scanpy as sc
        a3 = A.copy(); sc.pp.normalize_total(a3, target_sum=1e4); sc.pp.log1p(a3)
        Xd = a3.X.toarray() if hasattr(a3.X, "toarray") else np.asarray(a3.X)
        pts["PCA_1"] = pca_first_coord(Xd)
        for name, fn in [("DPT", dpt_pseudotime), ("PAGA_DPT", paga_dpt_pseudotime),
                         ("Slingshot", slingshot_pseudotime)]:
            try:
                pts[name] = fn(A, root)
            except Exception as e:  # noqa: BLE001
                print(f"  {name} FAILED: {repr(e)[:150]}", flush=True)
        r = {"dataset": ds, "provenance": ORDERINGS[ds][0], "n_cells": int(A.n_obs)}
        for k, v in pts.items():
            r[k] = spearman_abs(np.asarray(v, float), rank)
        rows.append(r)
        chk = ""
        if ds in SHIPPED:
            chk = "  [vs shipped " + ", ".join(
                f"{m} {SHIPPED[ds][m]:.3f}->{r.get(m, float('nan')):.3f}" for m in SHIPPED[ds]) + "]"
        print(f"  " + "  ".join(f"{k}={r[k]:.3f}" for k in pts) + f"  ({time.time()-t0:.0f}s){chk}\n",
              flush=True)
        pd.DataFrame(rows).to_csv(WS / "outputs" / "trajectory" / "cell_traj_baselines_all.csv",
                                  index=False)

    df = pd.DataFrame(rows)
    if df.empty:
        print("nothing scored"); return
    print("\n=== dedicated cell-trajectory baselines, |Spearman rho| ===")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
