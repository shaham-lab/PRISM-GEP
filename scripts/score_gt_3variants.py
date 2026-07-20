"""Score GeneTrajectory three ways (ALL post-bugfix) so the author can decide which to report:
  (1) EV            -- leading eigenvector of GT's gene diffusion embedding (OUR single-axis re-scoring)
  (2) extract       -- GT's NATIVE multi-program output (assigned-trajectory Pseudoorder), scored
                       against the single canonical linear marker order
  (3) extract-best  -- GT's native output but scored on its BEST-matching extracted trajectory
                       (fair for a multi-program method: pick the one trajectory that best recovers
                       the canonical markers). Requires gt_full_trajectories.csv from run_genetrajectory.py.
Reference: PRISM K5 Step-(ii) and DPT (root-supervised) for context. Metric = |Spearman rho| vs canonical_rank.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

WS = Path(__file__).resolve().parent.parent
TRAJ = WS / "outputs" / "trajectory"
DS = ["pancreas", "gastrulation", "gastrulation_erythroid", "hemogenic_endothelium", "bonemarrow",
      "gastrulation_e75", "paul15", "dentategyrus", "endoderm_diff"]


def sp(a, b):
    a = pd.to_numeric(a, errors="coerce"); b = pd.to_numeric(b, errors="coerce")
    m = a.notna() & b.notna()
    if m.sum() < 3 or a[m].nunique() < 2:
        return np.nan, int(m.sum())
    return abs(spearmanr(a[m], b[m])[0]), int(m.sum())


def best_trajectory(ds, orders):
    gf = TRAJ / ds / "gt_full_trajectories.csv"
    if not gf.exists():
        return np.nan, None, 0
    gt = pd.read_csv(gf)
    lut = {str(g).upper(): g for g in gt["gene"]}
    mk = orders[["gene", "canonical_rank"]].copy()
    mk["gt_gene"] = mk["gene"].map(lambda g: lut.get(str(g).upper()))
    mg = mk.dropna(subset=["gt_gene"]).merge(gt, left_on="gt_gene", right_on="gene", suffixes=("", "_gt"))
    pcols = [c for c in gt.columns if c.lower().startswith("pseudoorder")]
    best, which, bn = np.nan, None, 0
    for pc in pcols:
        lab = f"Trajectory-{pc.split('-')[-1]}"
        sub = mg[mg["selected"] == lab]
        if len(sub) >= 3:
            s, n = sp(sub[pc], sub["canonical_rank"])
            if pd.notna(s) and (pd.isna(best) or s > best):
                best, which, bn = s, lab, n
    return best, which, bn


def main():
    rows = []
    for ds in DS:
        oc = TRAJ / ds / f"gene_trajectory_{ds}_orders.csv"
        if not oc.exists():
            continue
        o = pd.read_csv(oc)
        prism, _ = sp(o.get("PRISM_K5_StepII"), o["canonical_rank"])
        dpt, _ = sp(o["DPT_weighted_mean"], o["canonical_rank"]) if "DPT_weighted_mean" in o else (np.nan, 0)
        ev, n = sp(o["GeneTrajectory_EV"], o["canonical_rank"])
        ext, nx = sp(o["GeneTrajectory"], o["canonical_rank"])
        bt, which, bn = best_trajectory(ds, o)
        rows.append(dict(dataset=ds, n_markers=n, PRISM_K5=prism, DPT=dpt,
                         GT_EV=ev, GT_extract=ext, GT_ext_n=nx,
                         GT_best_traj=bt, best_which=which, best_n=bn))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))
    df.to_csv(TRAJ / "gt_3variants_scores.csv", index=False)
    print("\n=== summary (mean / median over datasets with a defined value) ===")
    for c in ["PRISM_K5", "GT_EV", "GT_extract", "GT_best_traj", "DPT"]:
        v = df[c].dropna()
        print(f"  {c:14} mean={v.mean():.3f}  median={v.median():.3f}  (n={len(v)} datasets)")
    print(f"\nWROTE {TRAJ / 'gt_3variants_scores.csv'}")


if __name__ == "__main__":
    main()
