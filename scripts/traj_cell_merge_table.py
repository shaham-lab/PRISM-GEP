"""Merge the PRISM column (traj_cell_all_datasets.py) with the dedicated baselines
(traj_cell_baselines_all.py) into the full cell-trajectory table across every dataset that has
per-cell labels.

Adds the two columns that decide whether a row is worth shipping:
  spread   = max(method) - min(method). A row where every method scores the same is not
             discriminating between methods, whatever the absolute value.
  vs_pc1   = best method - PCA_1. Near zero means a plain first principal component of the
             expression matrix already does the job, so the row is not testing trajectory
             inference.

    python scripts/traj_cell_merge_table.py            # print table + write the CSVs

Outputs:
  results/tables/tab_traj_cell_all.csv              methods x datasets (+ mean), the
                                                    defensible rows only (3 decimals)
  results/tables/tab_traj_cell_all_diagnostics.csv  the full per-dataset table printed
                                                    below, spread / vs_pc1 included
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
# release layout: results/tables/, created on demand (scripts/paths.py)
from paths import tables_dir  # noqa: E402
TRAJ = WS / "outputs" / "trajectory"
PRETTY = {"pancreas": "Pancreas", "gastrulation_erythroid": "Gast. Eryth.",
          "gastrulation": "Gastrulation", "dentategyrus": "Dentate Gyrus",
          "ventral_neuron_diff": "Ventral neuron", "endoderm_diff": "Endoderm diff.",
          "gastrulation_e75": "Gastrulation E7.5", "bonemarrow": "Bonemarrow",
          "paul15": "Paul HSC", "mouse_hspc": "Mouse HSPC",
          "hemogenic_endothelium": "Hemogenic Endo.", "pbmc3k": "PBMC3k",
          "pbmc68k": "PBMC68k", "zeisel_brain": "Zeisel brain"}
METHODS = ["prism_mean", "Slingshot", "DPT", "PAGA_DPT", "PCA_1"]


def main():
    p = pd.read_csv(TRAJ / "cell_traj_all_datasets.csv")
    b = pd.read_csv(TRAJ / "cell_traj_baselines_all.csv").drop(columns=["provenance", "n_cells"],
                                                               errors="ignore")
    d = p.merge(b, on="dataset", how="left")
    have = [m for m in METHODS if m in d.columns]
    d["best"] = d[have].max(axis=1)
    d["spread"] = d[have].max(axis=1) - d[have].min(axis=1)
    d["vs_pc1"] = d["best"] - d["PCA_1"] if "PCA_1" in d else np.nan
    d["prism_rank"] = d[have].rank(axis=1, ascending=False)["prism_mean"]
    d = d.sort_values(["provenance", "prism_mean"], ascending=[True, False])

    cols = ["dataset", "provenance", "n_ranks", "pct_cells", "prism_mean", "prism_std",
            "Slingshot", "DPT", "PAGA_DPT", "PCA_1", "prism_rank", "spread", "vs_pc1",
            "perm_frac_ge", "libsize_rho"]
    print("=== FULL cell-trajectory table, all datasets with per-cell labels ===")
    print(d[[c for c in cols if c in d]].round(3).to_string(index=False))

    print("\n--- diagnostics ---")
    flat = d[d.spread < 0.05]
    if len(flat):
        print(f"no method separation (spread<0.05): {', '.join(flat.dataset)}")
    triv = d[d.vs_pc1 < 0.02]
    if len(triv):
        print(f"PCA-1 matches the best method (vs_pc1<0.02): {', '.join(triv.dataset)}")
    print(f"\nPRISM mean rank of {len(have)} methods: {d.prism_rank.mean():.2f}")
    for grp, sub in d.groupby("provenance"):
        print(f"  {grp:10} n={len(sub):2}  PRISM mean={sub.prism_mean.mean():.3f}  "
              f"rank={sub.prism_rank.mean():.2f}")

    diag_out = tables_dir() / "tab_traj_cell_all_diagnostics.csv"
    d[[c for c in cols if c in d]].round(3).to_csv(diag_out, index=False)
    print(f"\nfull table -> {diag_out}")

    # Same shape as the shipped table (methods as rows, datasets as columns)
    keep = d[d.provenance.isin(["shipped", "published", "inferred"])]
    label = {"prism_mean": "PRISM-GEP (JS diffusion-map)", "Slingshot": "Slingshot",
             "DPT": "DPT / PAGA-DPT", "PCA_1": "PCA-1 (no-model floor)"}
    rows = []
    for m in ["prism_mean", "Slingshot", "DPT", "PCA_1"]:
        if m not in keep:
            continue
        v = keep[m].to_numpy(float)
        row = {"method": label[m]}
        for ds, x in zip(keep.dataset, v):
            row[ds] = None if x != x else round(float(x), 3)
        row["mean"] = round(float(np.nanmean(v)), 3)
        rows.append(row)
    out = tables_dir() / "tab_traj_cell_all.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"table (defensible rows only, {len(keep)} datasets) -> {out}")


if __name__ == "__main__":
    main()
