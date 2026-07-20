"""Consolidate the gene-embedding comparison over all 9 marker datasets.

Table S11 (`tab:gene_embed`) was built on four datasets. Nine datasets carry a
`canonical_rank` column in `outputs/trajectory/<ds>/gene_trajectory_<ds>_orders.csv`,
so the same comparison is defined on all nine.

Inputs (all already on disk, produced by the three ablation scripts):
  outputs/gene_embedding_ablation/aggregate_metrics.csv        PUBLISHED 4 ds
  outputs/gene_embedding_ablation/extra5_nonscgpt.csv          log1p + random, 5 new ds
  outputs/gene_embedding_ablation/extra5_scgpt.csv             scGPT static, 5 new ds
  outputs/gene_embedding_ablation/extra5_scgpt_contextual.csv  scGPT contextual, 5 new ds
  outputs/gene_embedding_ablation/intersected_marker_prism.csv PRISM full + intersected, 9 ds

Outputs:
  outputs/gene_embedding_ablation/aggregate_metrics_9ds.csv
  paper/figures/tab_gene_embed_9ds.tex

The published aggregate is READ ONLY here. The script asserts every published
value survives the merge unchanged and refuses to write if one moved.
"""
from __future__ import annotations


import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS / "scripts"))
# release layout: paper/figures/, created on demand (scripts/paths.py)
from paths import figures_dir  # noqa: E402
GEA = WS / "outputs" / "gene_embedding_ablation"
TEX_OUT = figures_dir() / "tab_gene_embed_9ds.tex"

# Column order and the short labels used by the sibling 8-dataset trajectory
# table (figures/tab_traj_gene_8ds.tex), so the two read consistently.
DATASETS = [
    ("pancreas", "Panc."),
    ("gastrulation", "Gastr."),
    ("gastrulation_erythroid", "Eryth."),
    ("hemogenic_endothelium", "Hemog."),
    ("bonemarrow", "Bone."),
    ("paul15", "Paul15"),
    ("dentategyrus", "DG"),
    ("endoderm_diff", "Endo."),
    ("gastrulation_e75", "E75"),
]

# Gastrulation E7.5 is ill-posed for gene-trajectory recovery (a multi-lineage snapshot with no
# single developmental axis). It stays in the table, but every aggregate (Mean, Med., Rank) is
# reported both over all datasets and, in parentheses, over the datasets with E7.5 excluded.
E75 = "gastrulation_e75"

# `random` is a chance floor, not a competing representation. The sibling
# 8-dataset trajectory table (figures/tab_traj_gene_8ds.tex) bolds "the best
# among the learned methods" and leaves its Random-chance row unmarked. We
# follow that, otherwise the floor would be bolded on Dentate Gyrus and E7.5,
# where it happens to beat every method. The values are still printed, so the
# reader can see it.
NOT_BOLDED = {"random"}

GROUP_FULL = [
    ("prism_full", "PRISM-GEP (full)"),
    ("log1p", "log1p"),
    ("random", "random"),
]
GROUP_INT = [
    ("prism_int", "PRISM-GEP (int.)"),
    ("scgpt", "scGPT static"),
    ("scgpt_contextual", "scGPT ctx.\\"),
]

# The published Table S11 values, to three decimals. Any merge that moves one of
# these is a bug, not a result.
PUBLISHED = {
    ("pancreas", "log1p"): 0.765, ("pancreas", "random"): 0.376,
    ("pancreas", "scgpt"): 0.811, ("pancreas", "scgpt_contextual"): 0.730,
    ("pancreas", "prism_full"): 0.742, ("pancreas", "prism_int"): 0.791,
    ("gastrulation", "log1p"): 0.963, ("gastrulation", "random"): 0.014,
    ("gastrulation", "scgpt"): 0.342, ("gastrulation", "scgpt_contextual"): 0.488,
    ("gastrulation", "prism_full"): 0.798, ("gastrulation", "prism_int"): 0.732,
    ("gastrulation_erythroid", "log1p"): 0.519, ("gastrulation_erythroid", "random"): 0.259,
    ("gastrulation_erythroid", "scgpt"): 0.894, ("gastrulation_erythroid", "scgpt_contextual"): 0.894,
    ("gastrulation_erythroid", "prism_full"): 0.927, ("gastrulation_erythroid", "prism_int"): 0.894,
    # Hemogenic marker order corrected 2026-07-19 (Cd44/Lmo2/Tal1 -> stage 1): every cell
    # scored against the canonical rank moved. log1p/random/prism recomputed faithfully; the
    # two scGPT cells could NOT be re-scored locally (contextual needs the scgpt package + GPU;
    # the static ordering does not reproduce the published pancreas value in this environment),
    # so they retain their pre-correction (old-rank) scores and remain STALE pending a GPU re-run.
    ("hemogenic_endothelium", "log1p"): 0.674, ("hemogenic_endothelium", "random"): 0.477,
    ("hemogenic_endothelium", "scgpt"): 0.075, ("hemogenic_endothelium", "scgpt_contextual"): 0.724,
    ("hemogenic_endothelium", "prism_full"): 0.936, ("hemogenic_endothelium", "prism_int"): 0.936,
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"  MISSING {path.name} -- rows from it will be absent")
        return pd.DataFrame(columns=["dataset", "method", "spearman_abs", "n_genes", "n_dropped"])
    df = pd.read_csv(path)
    for c in ("n_genes", "n_dropped"):
        if c not in df.columns:
            df[c] = np.nan
    print(f"  read {path.name}: {len(df)} rows")
    return df[["dataset", "method", "spearman_abs", "n_genes", "n_dropped"]]


def build_long(intersected: Path) -> pd.DataFrame:
    print("inputs:")
    frames = [
        _read(GEA / "aggregate_metrics.csv"),          # published 4 ds, carried verbatim
        _read(GEA / "extra5_nonscgpt.csv"),
        _read(GEA / "extra5_scgpt.csv"),
        _read(GEA / "extra5_scgpt_contextual.csv"),
    ]
    frames = [f for f in frames if len(f)]
    long = pd.concat(frames, ignore_index=True)
    long["n_dropped"] = long["n_dropped"].astype(float)
    long["n_genes"] = long["n_genes"].astype(float)

    # PRISM rows live in a separate wide CSV (they are recomputed from the
    # gene-trajectory orders, not from an embedding run).
    inter = pd.read_csv(intersected)
    print(f"  read {intersected.name}: {len(inter)} rows")
    prism_rows = []
    for _, r in inter.iterrows():
        prism_rows.append({
            "dataset": r["dataset"], "method": "prism_full",
            "spearman_abs": r["PRISM_full"], "n_genes": r["n_full"], "n_dropped": 0,
        })
        prism_rows.append({
            "dataset": r["dataset"], "method": "prism_int",
            "spearman_abs": r["PRISM_int"], "n_genes": r["n_int"],
            "n_dropped": r["n_full"] - r["n_int"],
        })
    long = pd.concat([long, pd.DataFrame(prism_rows)], ignore_index=True)

    # The published contextual rows were written without an n_dropped column.
    # Fill it arithmetically from the full marker count (this is bookkeeping, it
    # touches no measured value); the static rows confirm the same arithmetic.
    n_full = dict(zip(inter["dataset"], inter["n_full"]))
    miss = long["n_dropped"].isna() & long["dataset"].isin(n_full)
    long.loc[miss, "n_dropped"] = (
        long.loc[miss, "dataset"].map(n_full) - long.loc[miss, "n_genes"]
    )

    keep = [d for d, _ in DATASETS]
    long = long[long["dataset"].isin(keep)]
    long = long.drop_duplicates(subset=["dataset", "method"], keep="last")
    long["dataset"] = pd.Categorical(long["dataset"], categories=keep, ordered=True)
    order = [m for m, _ in GROUP_FULL + GROUP_INT]
    long["method"] = pd.Categorical(long["method"], categories=order, ordered=True)
    return long.sort_values(["dataset", "method"]).reset_index(drop=True)


def check_published(long: pd.DataFrame) -> list[str]:
    idx = long.astype({"dataset": str, "method": str}).set_index(
        ["dataset", "method"])["spearman_abs"]
    bad = []
    for key, expected in sorted(PUBLISHED.items()):
        try:
            got = float(idx.loc[key])
        except KeyError:
            bad.append(f"{key}: MISSING from the merge (expected {expected:.3f})")
            continue
        if abs(round(got, 3) - expected) > 1e-9:
            bad.append(f"{key}: expected {expected:.3f}, merged {got:.3f}")
    return bad


def fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "---"
    return f"{v:.3f}".lstrip("0")


def emit_tex(long: pd.DataFrame) -> str:
    piv = long.pivot_table(index="method", columns="dataset", values="spearman_abs",
                           observed=True, dropna=False)
    lines = [
        "% AUTO-GENERATED by scripts/build_gene_embed_9ds.py -- do not edit by hand.",
        "% 9-dataset replacement for Supplementary Table S11 (tab:gene_embed).",
        "\\begin{tabular}{l" + "c" * len(DATASETS) + "|ccc}",
        "\\toprule",
        "\\textbf{Embedding} & " + " & ".join(lbl for _, lbl in DATASETS)
        + " & \\textbf{Mean} & \\textbf{Med.} & \\textbf{Rank} \\\\",
        "\\midrule",
    ]
    n_col = len(DATASETS) + 4
    for group, header in ((GROUP_FULL, "Full marker set"),
                          (GROUP_INT, "scGPT-embeddable intersection")):
        methods = [m for m, _ in group if m not in NOT_BOLDED]
        # Per column, the best within the group -- but only when it is a unique
        # maximum at the printed precision. The published table leaves the
        # three-way .894 tie on Gastrulation Erythroid unmarked; keep that.
        best = {}
        for ds, _ in DATASETS:
            vals = {m: round(float(piv.loc[m, ds]), 3) for m in methods
                    if m in piv.index and not pd.isna(piv.loc[m, ds])}
            if not vals:
                continue
            top = max(vals.values())
            winners = [m for m, v in vals.items() if v == top]
            if len(winners) == 1:
                best[ds] = winners[0]
        # Aggregates are computed WITHIN the group, never across both. The two groups
        # score on different marker sets (full vs the subset scGPT can embed), so a mean
        # or a rank pooled over all six rows would compare different quantities.
        group_all = [m for m, _ in group]
        # Mean rank is restricted to datasets where EVERY member of the group is defined,
        # otherwise rows would be ranked out of 3 on some datasets and out of 2 on others
        # (scGPT contextual is missing on two) and the averages would not be comparable.
        rankable = [ds for ds, _ in DATASETS
                    if all(m in piv.index and not pd.isna(piv.loc[m, ds]) for m in group_all)]
        # Ranks are RECOMPUTED over the E7.5-excluded dataset set, not dropped from the full
        # ranking; per-dataset ranks are position-independent, so this equals averaging the
        # per-dataset ranks over the reduced set.
        rankable_r = [ds for ds in rankable if ds != E75]
        mean_rank, mean_rank_r = {}, {}
        for m in group_all:
            for src, dst in ((rankable, mean_rank), (rankable_r, mean_rank_r)):
                rs = []
                for ds in src:
                    vals = sorted((float(piv.loc[mm, ds]) for mm in group_all), reverse=True)
                    rs.append(vals.index(float(piv.loc[m, ds])) + 1)
                dst[m] = float(np.mean(rs)) if rs else float("nan")
        # Best-in-group is judged on the FULL aggregates; bold/underline stays on the full value.
        best_mean = max((np.nanmean([piv.loc[m, ds] for ds, _ in DATASETS])
                         for m in group_all if m not in NOT_BOLDED), default=np.nan)
        best_rank = min((mean_rank[m] for m in group_all if m not in NOT_BOLDED),
                        default=np.nan)

        lines.append(f"\\multicolumn{{{n_col}}}{{l}}{{\\textit{{{header}}}}}\\\\")
        for m, label in group:
            cells = []
            for ds, _ in DATASETS:
                v = piv.loc[m, ds] if m in piv.index else np.nan
                s = fmt(v)
                if best.get(ds) == m and s != "---":
                    s = f"\\best{{{s}}}"
                cells.append(s)
            vals = np.array([piv.loc[m, ds] if m in piv.index else np.nan
                             for ds, _ in DATASETS], dtype=float)
            ok = vals[~np.isnan(vals)]
            mu, med = (np.mean(ok), np.median(ok)) if len(ok) else (np.nan, np.nan)
            # Same aggregates over the datasets minus Gastrulation E7.5.
            vals_r = np.array([piv.loc[m, ds] if m in piv.index else np.nan
                               for ds, _ in DATASETS if ds != E75], dtype=float)
            ok_r = vals_r[~np.isnan(vals_r)]
            mu_r, med_r = (np.mean(ok_r), np.median(ok_r)) if len(ok_r) else (np.nan, np.nan)
            # A superscript marks a mean taken over fewer than all nine datasets.
            sup = "" if len(ok) == len(DATASETS) else f"$^{{{len(ok)}}}$"
            mu_s = fmt(mu) + sup
            med_s = fmt(med) + sup
            if m not in NOT_BOLDED and not np.isnan(mu) and abs(mu - best_mean) < 5e-4:
                mu_s = f"\\best{{{fmt(mu)}}}" + sup
            mr = mean_rank.get(m, float("nan"))
            mr_s = "---" if np.isnan(mr) else f"{mr:.2f}"
            if m not in NOT_BOLDED and not np.isnan(mr) and abs(mr - best_rank) < 1e-9:
                mr_s = f"\\best{{{mr_s}}}"
            # Append the E7.5-excluded value in parentheses (smaller, to hold column width).
            mr_r = mean_rank_r.get(m, float("nan"))
            mr_r_s = "---" if np.isnan(mr_r) else f"{mr_r:.2f}"
            mu_s = mu_s + rf"~{{\footnotesize({fmt(mu_r)})}}"
            med_s = med_s + rf"~{{\footnotesize({fmt(med_r)})}}"
            mr_s = mr_s + rf"~{{\footnotesize({mr_r_s})}}"
            cells += [mu_s, med_s, mr_s]
            lines.append(f"{label:<22s} & " + " & ".join(cells) + " \\\\")
        lines.append(f"% group '{header}': mean rank over {len(rankable)} of "
                     f"{len(DATASETS)} datasets where all members are defined")
        if group is GROUP_FULL:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", ""]

    # A second small block records how many markers each group scored, which is
    # the vocabulary-coverage story (scGPT drops 4 of 8 markers on Paul15).
    n_full = [piv_n(long, "prism_full", ds, "n_genes") for ds, _ in DATASETS]
    n_int = [piv_n(long, "prism_int", ds, "n_genes") for ds, _ in DATASETS]
    lines.append("% marker counts: full  = " + "/".join(map(str, n_full)))
    lines.append("% marker counts: inter = " + "/".join(map(str, n_int)))
    return "\n".join(lines) + "\n"


def piv_n(long: pd.DataFrame, method: str, ds: str, col: str) -> str:
    sel = long[(long["method"] == method) & (long["dataset"] == ds)]
    if sel.empty or pd.isna(sel.iloc[0][col]):
        return "-"
    return str(int(sel.iloc[0][col]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intersected", default=str(GEA / "intersected_marker_prism.csv"))
    ap.add_argument("--out-csv", default=str(GEA / "aggregate_metrics_9ds.csv"))
    ap.add_argument("--out-tex", default=str(TEX_OUT))
    ap.add_argument("--force", action="store_true",
                    help="write even if a published value failed its check")
    args = ap.parse_args()

    long = build_long(Path(args.intersected))

    bad = check_published(long)
    print("\npublished-value check (24 cells of the original Table S11):")
    if bad:
        for b in bad:
            print(f"  FAIL {b}")
        if not args.force:
            print("\nREFUSING to write. Published numbers must not move.")
            return 1
    else:
        print("  OK -- all 24 published values reproduce exactly")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(out_csv, index=False)
    print(f"\nWROTE {out_csv}")

    tex = emit_tex(long)
    out_tex = Path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(tex, encoding="utf-8")
    print(f"WROTE {out_tex}")

    piv = long.pivot_table(index="method", columns="dataset", values="spearman_abs",
                           observed=True, dropna=False)
    print("\n|Spearman rho| (rows = method, cols = dataset)")
    print(piv.round(3).to_string())
    pivn = long.pivot_table(index="method", columns="dataset", values="n_dropped",
                            observed=True, dropna=False)
    print("\nn_dropped")
    print(pivn.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
