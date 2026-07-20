#!/usr/bin/env python
"""
Supplementary figure: GPT-4 plausibility (LLM-as-judge) across the nine LLM-scored
datasets under the faithful naming-then-confidence protocol (Hu et al. 2025).

Shows the aggregate WIN explicitly: PRISM-GEP has the best mean, the best median AND
the best mean rank of the six methods, in a tight top cluster with MALLET/scHPF --
disclosed honestly by drawing every per-dataset point, not just the summary.

Data sources (faithful / naming-then-confidence protocol = prompt B = the protocol
used by main Table 2):
  * 6 additional datasets : outputs/rebuttal_summary/llm_coherence_all_v2.csv  (on disk)
  * 3 main-table datasets : main.tex Table 2 (tab:bio-llm), reproduced below.

Output: paper/figures/llm_plausibility_9ds.pdf
Run:    python scripts/make_llm_win_fig.py [--orient horizontal|vertical] [--out PATH]
"""
import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
# release layout: inputs from results/, figures to paper/figures/
# (scripts/paths.py); the authoring tree uses different roots.
from paths import figures_dir, results_dir  # noqa: E402
CSV = os.path.join(str(results_dir()), "llm_coherence_all_v2.csv")
OUT = os.path.join(str(figures_dir()), "llm_plausibility_9ds.pdf")

METHODS = ["PRISM", "MALLET", "scHPF", "NMF", "cNMF", "ProdLDA"]
DISP = {"PRISM": "PRISM-GEP", "MALLET": "MALLET", "scHPF": "scHPF",
        "NMF": "NMF", "cNMF": "cNMF", "ProdLDA": "ProdLDA"}
Y0 = 0.4   # score-axis lower bound (a few ProdLDA points fall below and are off-scale)
INK, MUTE, TEAL = "#222222", "#6b6b6b", "#0b7d73"

MAIN = {
    "breast_cancer": {"PRISM": .8320, "MALLET": .7920, "NMF": .7880, "cNMF": .8000, "scHPF": .7712, "ProdLDA": .7600},
    "pbmc3k":        {"PRISM": .8997, "MALLET": .8392, "NMF": .6590, "cNMF": .9021, "scHPF": .8326, "ProdLDA": .4950},
    "zeisel_brain":  {"PRISM": .9350, "MALLET": .8920, "NMF": .6670, "cNMF": .9330, "scHPF": .8980, "ProdLDA": .5950},
}
SIX = ["hemogenic_endothelium", "pancreas", "gastrulation_e75",
       "gastrulation", "gastrulation_erythroid", "bonemarrow"]


def load_matrix():
    df = pd.read_csv(CSV)
    add = df[df.dataset.isin(SIX)].pivot(index="dataset", columns="method", values="llm_mean")
    rows = {ds: d for ds, d in MAIN.items()}
    for ds in SIX:
        rows[ds] = {m: float(add.loc[ds, m]) for m in METHODS}
    return pd.DataFrame(rows).T[METHODS]


def mc(m):  return TEAL if m == "PRISM" else "#8f8f8f"
def dc(m):  return "#8fcfc8" if m == "PRISM" else "#d2d2d2"


def render_vertical(M, means, medians, ranks, order, out):
    n = len(order)
    x = np.arange(n)
    xp = order.index("PRISM")
    rng = np.random.default_rng(0)
    fig, (axa, axb) = plt.subplots(
        2, 1, figsize=(4.7, 3.7), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.15], "hspace": 0.14})

    # (a) score distribution
    axa.axvspan(xp - 0.45, xp + 0.45, color=TEAL, alpha=0.07, zorder=0)
    for xi, m in zip(x, order):
        vals = M[m].values
        jit = (rng.random(len(vals)) - 0.5) * 0.42
        axa.scatter(np.full_like(vals, xi) + jit, vals, s=10, color=dc(m),
                    edgecolor="none", zorder=2, clip_on=True)
        axa.plot([xi - 0.27, xi + 0.27], [medians[m], medians[m]], color=mc(m), lw=1.8, zorder=3)
        axa.scatter([xi], [means[m]], marker="D", s=34, color=mc(m), edgecolor="white",
                    linewidth=0.6, zorder=4)
        axa.text(xi, 0.977, f"{means[m]:.3f}", ha="center", va="top", fontsize=7,
                 color=mc(m), fontweight="bold" if m == "PRISM" else "normal")
    axa.set_ylim(Y0, 1.0)
    axa.set_xlim(-0.6, n - 0.4)
    axa.set_ylabel("GPT-4 plausibility", fontsize=8.5)
    axa.set_yticks([0.4, 0.6, 0.8, 1.0])
    axa.set_title("(a)", fontsize=8.5, loc="left")   # marker reading-key -> LaTeX caption
    for s in ("top", "right"):
        axa.spines[s].set_visible(False)
    axa.tick_params(length=0, labelsize=8)

    # (b) mean rank -- lollipop, axis inverted so rank 1 (best) is at the top
    base = 6.35
    axb.axvspan(xp - 0.45, xp + 0.45, color=TEAL, alpha=0.07, zorder=0)
    for xi, m in zip(x, order):
        axb.vlines(xi, base, ranks[m], color=mc(m), lw=1.6, zorder=2)
        axb.scatter([xi], [ranks[m]], s=30, color=mc(m), zorder=3)
        axb.text(xi + 0.12, ranks[m], f"{ranks[m]:.2f}", ha="left", va="center", fontsize=7,
                 color=(TEAL if m == "PRISM" else INK), fontweight="bold" if m == "PRISM" else "normal")
    axb.set_ylim(base, 0.5)
    axb.set_ylabel("mean rank", fontsize=8.5)
    axb.set_yticks([1, 3, 5])
    axb.set_title("(b)", fontsize=8.5, loc="left")   # reading-key -> LaTeX caption
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)
    axb.set_xticks(x)
    axb.set_xticklabels([DISP[m] for m in order], fontsize=8)
    for lab, m in zip(axb.get_xticklabels(), order):
        lab.set_fontweight("bold" if m == "PRISM" else "normal")
        lab.set_color(TEAL if m == "PRISM" else INK)
    axb.tick_params(length=0, labelsize=8)
    fig.savefig(out, bbox_inches="tight")


def render_horizontal(M, means, medians, ranks, order, out):
    n = len(order)
    y = np.arange(n)[::-1]
    rng = np.random.default_rng(0)
    fig, (axm, axr) = plt.subplots(
        1, 2, figsize=(6.6, 2.35), gridspec_kw={"width_ratios": [2.45, 1.0], "wspace": 0.28})
    for yi, m in zip(y, order):
        vals = M[m].values
        jit = (rng.random(len(vals)) - 0.5) * 0.22
        axm.scatter(vals, np.full_like(vals, yi) + jit, s=10, color=dc(m), edgecolor="none", zorder=2)
        axm.plot([medians[m], medians[m]], [yi - 0.27, yi + 0.27], color=mc(m), lw=1.7, zorder=3)
        axm.scatter([means[m]], [yi], marker="D", s=34, color=mc(m), edgecolor="white", linewidth=0.6, zorder=4)
        axm.text(max(means[m], medians[m]) + 0.012, yi, f"{means[m]:.3f}", ha="left", va="center",
                 fontsize=7, color=mc(m), fontweight="bold" if m == "PRISM" else "normal")
    axm.set_yticks(y); axm.set_yticklabels([DISP[m] for m in order])
    for lab, m in zip(axm.get_yticklabels(), order):
        lab.set_fontweight("bold" if m == "PRISM" else "normal"); lab.set_color(TEAL if m == "PRISM" else INK)
    axm.set_xlim(Y0, 1.0); axm.set_ylim(-0.7, n - 0.3)
    axm.set_xlabel("GPT-4 plausibility", fontsize=8.5)
    axm.set_title("(a)", fontsize=8.5, loc="left")   # marker reading-key -> LaTeX caption
    for s in ("top", "right"):
        axm.spines[s].set_visible(False)
    axm.tick_params(length=0, labelsize=8.5); axm.set_xticks([0.4, 0.6, 0.8, 1.0])
    for yi, m in zip(y, order):
        axr.barh(yi, ranks[m], height=0.6, color=mc(m), edgecolor="white", zorder=2)
        axr.text(ranks[m] + 0.16, yi, f"{ranks[m]:.2f}", ha="right", va="center", fontsize=7.6,
                 color=(TEAL if m == "PRISM" else INK), fontweight="bold" if m == "PRISM" else "normal", zorder=5)
    axr.set_yticks(y); axr.set_yticklabels([]); axr.set_xlim(0, 6.4); axr.set_ylim(-0.7, n - 0.3)
    axr.invert_xaxis(); axr.set_xlabel("mean rank  (1 = best)", fontsize=8.5)
    axr.set_title("(b)", fontsize=8.5, loc="left")   # reading-key -> LaTeX caption
    for s in ("top", "left"):
        axr.spines[s].set_visible(False)
    axr.tick_params(length=0, labelsize=8); axr.set_xticks([1, 3, 5])
    fig.savefig(out, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orient", choices=["horizontal", "vertical"], default="horizontal")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    M = load_matrix()
    means, medians = M.mean(), M.median()
    ranks = M.rank(axis=1, ascending=False, method="average").mean()
    order = means.sort_values(ascending=False).index.tolist()
    print("mean  :", {m: round(means[m], 3) for m in order})
    print("median:", {m: round(medians[m], 3) for m in order})
    print("mrank :", {m: round(ranks[m], 2) for m in order})
    assert means.idxmax() == "PRISM" and medians.idxmax() == "PRISM" and ranks.idxmin() == "PRISM"

    (render_vertical if a.orient == "vertical" else render_horizontal)(M, means, medians, ranks, order, a.out)
    print("saved", a.out, f"({a.orient})")


if __name__ == "__main__":
    main()
