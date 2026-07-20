"""Marker-bootstrap 95% CI on |Spearman rho| for the gene-trajectory
baselines.

Marker sets are small (8 / 11 / 15 markers per dataset) so a small-set
bootstrap is the right uncertainty axis — much more informative than the
point Spearman alone.

Method: resample the marker set with replacement (n_markers per draw),
recompute |Spearman ρ| between method order and canonical rank, repeat
B = 5000 times.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

WS = Path(__file__).resolve().parent.parent
TRAJ = WS / "outputs" / "trajectory"
OUT = TRAJ / "gene_trajectory_bootstrap_ci.csv"

ORDERS = {
    "pancreas": TRAJ / "pancreas" / "gene_trajectory_pancreas_orders.csv",
    "gastrulation": TRAJ / "gastrulation" / "gene_trajectory_gastrulation_orders.csv",
    "gastrulation_erythroid": TRAJ / "gastrulation_erythroid" / "gene_trajectory_gastrulation_erythroid_orders.csv",
    "hemogenic_endothelium": TRAJ / "hemogenic_endothelium" / "gene_trajectory_hemogenic_endothelium_orders.csv",
    "bonemarrow": TRAJ / "bonemarrow" / "gene_trajectory_bonemarrow_orders.csv",
    "gastrulation_e75": TRAJ / "gastrulation_e75" / "gene_trajectory_gastrulation_e75_orders.csv",
    "paul15": TRAJ / "paul15" / "gene_trajectory_paul15_orders.csv",
    "dentategyrus": TRAJ / "dentategyrus" / "gene_trajectory_dentategyrus_orders.csv",
    "endoderm_diff": TRAJ / "endoderm_diff" / "gene_trajectory_endoderm_diff_orders.csv",
}

B = 5000
RNG = np.random.default_rng(0)


def bootstrap(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    n = len(x)
    rhos = []
    for _ in range(B):
        idx = RNG.integers(0, n, size=n)
        # need at least 2 unique markers for spearman
        xi, yi = x[idx], y[idx]
        if len(set(idx.tolist())) < 2:
            continue
        try:
            r, _ = spearmanr(xi, yi)
            if np.isnan(r):
                continue
            rhos.append(abs(r))
        except Exception:
            continue
    rhos = np.array(rhos)
    return (float(rhos.mean()), float(np.percentile(rhos, 2.5)),
            float(np.percentile(rhos, 97.5)), float(rhos.std()))


def main() -> int:
    rows = []
    for ds, path in ORDERS.items():
        df = pd.read_csv(path)
        rank = df["canonical_rank"].to_numpy(dtype=float)
        n = len(df)
        methods = [c for c in df.columns if c not in ("gene", "canonical_rank")]
        for m in methods:
            x = df[m].to_numpy(dtype=float)
            if np.isnan(x).any():
                continue
            mean, lo, hi, std = bootstrap(x, rank)
            rows.append({"dataset": ds, "method": m, "n_markers": n,
                         "rho_boot_mean": mean, "rho_boot_std": std,
                         "rho_ci_lo": lo, "rho_ci_hi": hi})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"WROTE {OUT}\n")
    print(f"{'dataset':25s} {'method':20s} {'n':>3s} {'rho':>6s} {'CI95':>16s}")
    for _, r in out.iterrows():
        print(f"{r['dataset']:25s} {r['method']:20s} {int(r['n_markers']):>3d} "
              f"{r['rho_boot_mean']:>6.3f}  [{r['rho_ci_lo']:.3f}, {r['rho_ci_hi']:.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
