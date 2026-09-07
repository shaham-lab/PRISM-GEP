r"""Emit the prompt-(B) LLM-plausibility panel over all nine LLM-scored datasets (CSV).

Data sources (single source of truth):
  * 6 additional datasets : results/llm_coherence_all_v2.csv (on disk)
  * 3 original datasets   : the main-paper LLM table, mirrored in MAIN below
Both are the faithful naming-then-confidence protocol (prompt B, Hu et al. 2025).

Self-check: before emitting, the script recomputes the published six-dataset panel and the
published nine-dataset aggregate and refuses to write if either disagrees. This guarantees
the new rows are produced by the same convention as the numbers already in the paper.

Run: python scripts/build_llm_table_all9.py
Out: results/tables/tab_llm_all9.csv
     One row per dataset with the six methods' GPT-4 plausibility (3 decimals), then three
     summary rows keyed "mean_rank" (2 decimals), "mean" and "std" (3 decimals) in the
     `dataset` column.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
from paths import results_dir, tables_dir  # noqa: E402

CSV = os.path.join(str(results_dir()), "llm_coherence_all_v2.csv")
OUT = Path(tables_dir()) / "tab_llm_all9.csv"

METHODS = ["PRISM", "MALLET", "scHPF", "NMF", "cNMF", "ProdLDA"]

# The three original datasets: the main-paper LLM table (prompt B), reproduced here.
MAIN = {
    "breast_cancer": {"PRISM": .8320, "MALLET": .7920, "NMF": .7880, "cNMF": .8000, "scHPF": .7712, "ProdLDA": .7600},
    "pbmc3k":        {"PRISM": .8997, "MALLET": .8392, "NMF": .6590, "cNMF": .9021, "scHPF": .8326, "ProdLDA": .4950},
    "zeisel_brain":  {"PRISM": .9350, "MALLET": .8920, "NMF": .6670, "cNMF": .9330, "scHPF": .8980, "ProdLDA": .5950},
}


def load_matrix():
    """datasets x methods matrix of GPT-4 plausibility (prompt B), nine datasets."""
    df = pd.read_csv(CSV)
    add = df[df.dataset.isin(SIX)].pivot(index="dataset", columns="method", values="llm_mean")
    rows = {ds: d for ds, d in MAIN.items()}
    for ds in SIX:
        rows[ds] = {m: float(add.loc[ds, m]) for m in METHODS}
    return pd.DataFrame(rows).T[METHODS]

# column order used in the prompt-(B) panel
COLS = ["PRISM", "MALLET", "NMF", "cNMF", "scHPF", "ProdLDA"]
SIX = ["hemogenic_endothelium", "pancreas", "gastrulation_e75",
       "gastrulation", "gastrulation_erythroid", "bonemarrow"]
NINE = ["breast_cancer", "pbmc3k", "zeisel_brain"] + SIX
DISPLAY = {
    "breast_cancer": "BreastCancer", "pbmc3k": "PBMC3k", "zeisel_brain": "Zeisel brain",
    "hemogenic_endothelium": "Hemogenic Endo.", "pancreas": "Pancreas",
    "gastrulation_e75": "Gastrulation E7.5", "gastrulation": "Gastrulation (full)",
    "gastrulation_erythroid": "Gastrulation Eryth.", "bonemarrow": "Bonemarrow",
}


def f3(v):
    """three decimals, plain number"""
    return round(float(v), 3)


def ranks(M):
    """per-dataset ranks, 1 = best (highest score); ties share the average rank."""
    return M.rank(axis=1, ascending=False, method="average")


def summarize(M):
    R = ranks(M)
    # ddof=1 (sample std) is the published convention: it reproduces the six-dataset
    # panel's stds exactly, whereas ddof=0 comes out sqrt(5/6) too low.
    return M.mean(), M.median(), M.std(ddof=1), R.mean()


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'OK ' if ok else 'BAD'} {label:38} got {got:.3f}  published {want:.3f}")
    return ok


M = load_matrix()[COLS]
M9 = M.loc[NINE]
M6 = M.loc[SIX]

print("== consistency check vs the six-dataset panel ==")
m6, _, s6, r6 = summarize(M6)
ok = True
for meth, wr, wm, ws in [("PRISM", 1.92, .875, .054), ("MALLET", 2.25, .871, .057),
                         ("NMF", 3.33, .846, .041), ("cNMF", 4.33, .773, .104),
                         ("scHPF", 3.17, .858, .051), ("ProdLDA", 6.00, .338, .194)]:
    ok &= check(f"{meth} mean rank (6ds)", r6[meth], wr, 0.02)
    ok &= check(f"{meth} mean (6ds)", m6[meth], wm, 0.002)
    ok &= check(f"{meth} std (6ds)", s6[meth], ws, 0.002)

print("\n== consistency check vs the nine-dataset aggregate ==")
m9, med9, s9, r9 = summarize(M9)
for meth, wm, wmed, wr in [("PRISM", .879, .886, 1.72), ("MALLET", .861, .882, 2.61)]:
    ok &= check(f"{meth} mean (9ds)", m9[meth], wm, 0.002)
    ok &= check(f"{meth} median (9ds)", med9[meth], wmed, 0.002)
    ok &= check(f"{meth} mean rank (9ds)", r9[meth], wr, 0.02)

if not ok:
    sys.exit("\nSELF-CHECK FAILED: recomputation disagrees with the published numbers. "
             "Not writing the table.")
print("\nAll self-checks passed. Emitting all-nine panel.\n")

# Prompt-(B) naming-then-confidence panel over ALL NINE LLM-scored datasets.
COLNAME = {m: ("PRISM-GEP" if m == "PRISM" else m) for m in COLS}
rows = []
for ds in NINE:
    row = {"dataset": ds, "dataset_label": DISPLAY[ds]}
    row.update({COLNAME[m]: f3(M9.loc[ds, m]) for m in COLS})
    rows.append(row)
# Summary rows: mean rank (lower is better, 2 decimals), mean and std (ddof=1) over the nine.
rows.append({"dataset": "mean_rank", "dataset_label": "mean rank",
             **{COLNAME[m]: round(float(r9[m]), 2) for m in COLS}})
rows.append({"dataset": "mean", "dataset_label": "mean", **{COLNAME[m]: f3(m9[m]) for m in COLS}})
rows.append({"dataset": "std", "dataset_label": "std", **{COLNAME[m]: f3(s9[m]) for m in COLS}})
tab = pd.DataFrame(rows)
tab.to_csv(OUT, index=False)
print(tab.to_string(index=False))
print(f"\nwrote {OUT}")
