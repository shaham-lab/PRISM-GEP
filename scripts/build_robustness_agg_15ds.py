#!/usr/bin/env python
r"""Emit supplementary Table S6 (aggregate robustness) over ALL FIFTEEN datasets.

Yanir 2026-07-19: move Table S6 from the 9-dataset panel to the full 15-dataset benchmark, so it
matches main-text Figure 3 and Supplementary Table S7. Method set follows Uri's 2026-07-19 framing
decision: the FOUR specialized methods plus PRISM-GEP, with MALLET excluded (the isolate-the-prior
comparison lives in the vs-MALLET section, exactly as in Figure 3).

Why the 9-dataset panel was worth replacing: on 9 datasets scHPF, NMF and MALLET are ALSO never
worst, so "never worst" does not distinguish PRISM-GEP there. On 15 datasets PRISM-GEP is the only
method that never finishes last (scHPF 5, NMF 4, cNMF 14, ProdLDA 20).

All definitions are imported from build_unified_table1 so this cannot drift from the published
convention. Before emitting, the script REPRODUCES the published 9-dataset/6-method table and
refuses to write if any value disagrees.

Run: python scripts/build_robustness_agg_15ds.py
Out: paper/figures/tab_robustness_agg_15ds.tex
"""
import os
from pathlib import Path
import sys
import random

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
# release layout: paper/figures/, created on demand (scripts/paths.py)
from paths import figures_dir, results_dir  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "scripts"))
import build_unified_table1 as B  # noqa: E402  (shared definitions)

# NOTE: deliberately NOT "tab_robustness_agg_15ds.tex". build_unified_table1.py writes that
# filename with its own SIX-method convention (MALLET included), and running it silently
# overwrote this table once already. The distinct name keeps the two from colliding.
OUT = os.path.join(str(figures_dir()), "tab_robustness_agg_15ds_vs_specialized.tex")
AGG = str(results_dir() / "full_metrics_combined_4config.csv")
PS = str(results_dir() / "full_metrics_perseed.csv")

SIX = ["PRISM", "MALLET", "NMF", "cNMF", "scHPF", "ProdLDA"]
FIVE = ["PRISM", "NMF", "cNMF", "scHPF", "ProdLDA"]
DISP = {"PRISM": r"\textbf{PRISM-GEP}", "MALLET": "MALLET", "NMF": "NMF",
        "cNMF": "cNMF", "scHPF": "scHPF", "ProdLDA": "ProdLDA"}

df = pd.read_csv(AGG).set_index("dataset")
ALL15 = list(df.index)
NINE = B.NINE


def agg(methods, datasets, n_boot=2000):
    """Mean rank + bootstrap CI over datasets, geometric composite, worst-axis, last-place count.
    Mirrors build_unified_table1.agg_table exactly, including the seed."""
    random.seed(0)
    B.METHODS = methods
    normcell = {m: {} for m in methods}
    rankcell = {m: {} for m in methods}
    ds_cells = {d: [] for d in datasets}
    for ds in datasets:
        for me in B.METR:
            vals = {m: B.val(df, ds, m, me) for m in methods}
            if sum(pd.notna(v) for v in vals.values()) < 3:
                continue
            defined = [v for v in vals.values() if pd.notna(v)]
            lo, hi = min(defined), max(defined)
            ser = pd.Series({m: (v if pd.notna(v) else -1e9) for m, v in vals.items()})
            rk = ser.rank(ascending=False, method="min")
            ds_cells[ds].append((ds, me))
            for m in methods:
                v = vals[m]
                normcell[m][(ds, me)] = ((v - lo) / (hi - lo) if (pd.notna(v) and hi > lo)
                                         else (0.0 if pd.isna(v) else 1.0))
                rankcell[m][(ds, me)] = rk[m]
    cells = [c for d in datasets for c in ds_cells[d]]

    def mean_rank(m, cs):
        return np.mean([rankcell[m][c] for c in cs]) if cs else np.nan

    rows = {}
    for m in methods:
        boots = []
        for _ in range(n_boot):
            samp = [c for d in [random.choice(datasets) for _ in datasets] for c in ds_cells[d]]
            if samp:
                boots.append(mean_rank(m, samp))
        lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
        geom = float(np.exp(np.mean(np.log(np.clip([normcell[m][c] for c in cells], 1e-6, 1)))))
        per = [np.mean([normcell[m][(d, me)] for d in datasets if (d, me) in normcell[m]])
               for me in B.METR]
        worst_ax = min(per)
        last = sum(1 for c in cells if rankcell[m][c] == max(rankcell[x][c] for x in methods))
        rows[m] = dict(mr=mean_rank(m, cells), lo=lo_ci, hi=hi_ci,
                       geom=geom, worst=worst_ax, last=last)
    return rows, len(cells)


def welch(methods, datasets):
    """PRISM-GEP vs each baseline, cell by cell, two-sample Welch over seeds."""
    ps = pd.read_csv(PS)
    wins = tie = loss = 0
    for ds in datasets:
        for me in ["coherence", "coverage", "strength"]:
            pr = ps[(ps.dataset == ds) & (ps.variant == B.PS_VAR["PRISM"])][me].dropna()
            if len(pr) < 2:
                continue
            for m in [x for x in methods if x != "PRISM"]:
                bl = ps[(ps.dataset == ds) & (ps.variant == B.PS_VAR[m])][me].dropna()
                if len(bl) < 2:
                    continue
                _, p = stats.ttest_ind(pr, bl, equal_var=False)
                if p < 0.05 and pr.mean() > bl.mean():
                    wins += 1
                elif p < 0.05 and pr.mean() < bl.mean():
                    loss += 1
                else:
                    tie += 1
    return wins, tie, loss


# ---- self-check: reproduce the PUBLISHED 9-dataset / 6-method table ----
print("== self-check vs published Table S6 (9 datasets, 6 methods) ==")
pub = {"scHPF": (2.80, 0.643, 0.711), "PRISM": (2.84, 0.694, 0.644),
       "MALLET": (3.00, 0.666, 0.589), "NMF": (3.16, 0.690, 0.745),
       "cNMF": (4.44, 0.003, 0.000), "ProdLDA": (4.72, 0.000, 0.000)}
r9, n9 = agg(SIX, NINE)
ok = (n9 == 25)
print(f"  {'OK ' if ok else 'BAD'} rankable cells: got {n9}, published 25")
for m, (mr, g, w) in pub.items():
    a = abs(r9[m]["mr"] - mr) <= 0.005
    b = abs(r9[m]["geom"] - g) <= 0.0015
    c = abs(r9[m]["worst"] - w) <= 0.0015
    ok &= a and b and c
    print(f"  {'OK ' if (a and b and c) else 'BAD'} {m:8s} rank {r9[m]['mr']:.2f}/{mr}  "
          f"geom {r9[m]['geom']:.3f}/{g}  worst {r9[m]['worst']:.3f}/{w}")
w9 = welch(SIX, NINE)
tot9 = sum(w9)
ok_w = (w9[0] + w9[1] == 100 and tot9 == 125)
ok &= ok_w
print(f"  {'OK ' if ok_w else 'BAD'} Welch best-or-tied {w9[0] + w9[1]} of {tot9}, published 100 of 125")

if not ok:
    sys.exit("\nSELF-CHECK FAILED: cannot reproduce the published table. Not writing.")
print("\nSelf-check passed.\n")

# ---- emit the 15-dataset / 5-method table ----
r15, n15 = agg(FIVE, ALL15)
w15 = welch(FIVE, ALL15)
order = sorted(FIVE, key=lambda m: r15[m]["mr"])

lines = [r"% AUTO-GENERATED by scripts/build_robustness_agg_15ds.py -- do not hand-edit numbers.",
         rf"% 15 datasets, {n15} rankable entries, PRISM-GEP vs the four specialized methods "
         r"(MALLET excluded, as in main-text Figure 3).",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"Method & Mean rank\,$\downarrow$ (95\% CI) & Geom.\ composite\,$\uparrow$ & Worst-axis\,$\uparrow$ \\",
         r"\midrule"]
for m in order:
    d = r15[m]
    lines.append(f"{DISP[m]} & {d['mr']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}] & "
                 f"{d['geom']:.3f} & {d['worst']:.3f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("\n".join(lines))
print(f"\nwrote {OUT}")
print(f"\n--- prose numbers for S6.1 ---")
print(f"rankable entries: {n15}")
for m in order:
    d = r15[m]
    print(f"  {m:8s} rank {d['mr']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]  geom {d['geom']:.3f}  "
          f"worst {d['worst']:.3f}  last {d['last']}")
print(f"  Welch: best-or-tied {w15[0] + w15[1]} of {sum(w15)}  (win {w15[0]}, tie {w15[1]}, loss {w15[2]})")
