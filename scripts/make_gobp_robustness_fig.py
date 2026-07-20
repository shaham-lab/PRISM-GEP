"""Main-paper GO-BP robustness headline figure (design v1, editorial rank bars).

Shows, across the 43 rankable (dataset x metric) GO-BP evaluations over the 15-dataset benchmark,
how often each method ranks 1st..5th among PRISM-GEP and the four specialized single-cell
factorizations (cNMF, scHPF, NMF, ProdLDA). MALLET is deliberately excluded here: the vs-MALLET
comparison is the isolate-the-prior control and lives in the appendix.

Two configs:
  opt0  (default; the beta-faithful config, hyperparameter re-estimation disabled) -> the MAIN figure.
        PRISM-GEP never finishes last (0/43): the robustness headline.
  opt10 (optimize-interval 10; the prior is re-estimated during training)          -> an APPENDIX figure.
        More first-place finishes but NO LONGER never-worst -- shown honestly for the trade-off.

A (dataset,metric) cell is rankable iff PRISM is defined there (excludes BreastCancer Cov/Str)
-> 43 cells, matching the supplement. Numbers on the bars and the never-worst subtitle are computed
from the data, so the opt10 figure truthfully shows PRISM's last-place cells.

Out: figures/gobp_robustness_headline.pdf / .png            (opt0)
     figures/gobp_robustness_headline_opt10.pdf / .png      (opt10)
Run: python scripts/make_gobp_robustness_fig.py [opt0|opt10]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

CFG = sys.argv[1] if len(sys.argv) > 1 else "opt0"
assert CFG in ("opt0", "opt10"), CFG
PRISM = "PRISMo0" if CFG == "opt0" else "PRISMo10"

WS = Path(__file__).resolve().parent.parent
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
# release layout: inputs from results/, figures to paper/figures/
# (scripts/paths.py); the authoring tree uses different roots.
from paths import figures_dir, results_dir  # noqa: E402
sys.path.insert(0, str(WS / "scripts"))
from figsafe import save_and_deploy  # noqa: E402

OUT = figures_dir()    # the paper's figure dir (where main.tex reads figures/)

# ---------------------------------------------------------------------------
# Print legibility (2026-07-20)
# ---------------------------------------------------------------------------
# This panel appears twice: main.tex at 0.68\linewidth of a figure* (357pt) and,
# the binding case, supplementary.tex at 0.52\linewidth (273pt). Drawn 421pt
# wide it was shrunk to 0.65x in the supplement, putting the rank-key labels at
# 4.2pt and the bar counts at 5.2pt. Drawing it near the supplement's placed
# width fixes both placements at once; the mean/worst annotations stack instead
# of sitting side by side so the bars keep their room.
FIG_W_IN, FIG_H_IN = 5.15, 2.03   # was 7.1 x 2.55
KEY_X = 12.0                      # rank-key offset, data units past the bars (was 17.5)
F_KEY = 7.0                       # rank-key digits and the best/worst captions (was 6.6 / 6.4)
DF = pd.read_csv(results_dir() / "full_metrics_combined_4config.csv")
METHODS = [PRISM, "scHPF", "cNMF", "NMF", "ProdLDA"]     # MALLET excluded (vs-MALLET -> appendix)
LABEL = {PRISM: "PRISM-GEP", "scHPF": "scHPF", "cNMF": "cNMF", "NMF": "NMF", "ProdLDA": "ProdLDA"}

GRAY = ["#ececec", "#cdcdcd", "#a2a2a2", "#6f6f6f"]           # ranks 1..4 for specialists
CRIM = "#b0495b"                                              # rank 5 (worst)
TEAL = ["#cfe6e2", "#8ec9c1", "#3ea298", "#0f716a"]          # ranks 1..4 for PRISM-GEP
TEAL_DARK, INK, MUTE = "#0c5e57", "#2b2b2b", "#8a8a8a"


def rank_table():
    recs = []
    for _, r in DF.iterrows():
        for mk in ("coh", "cov", "str"):
            vals = [r[f"{m}_{mk}"] for m in METHODS]
            if pd.isna(vals[0]):
                continue
            arr = np.where(np.isnan(vals), -np.inf, vals)
            recs.append(dict(zip(METHODS, rankdata(-arr, method="min"))))
    return pd.DataFrame(recs)


def txt_on(hexc):
    r, g, b = matplotlib.colors.to_rgb(hexc)
    return "#ffffff" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else "#3a3a3a"


def main():
    RK = rank_table()
    N = len(RK)
    p_best = int((RK[PRISM] == 1).sum())
    p_worst = int((RK[PRISM] == 5).sum())
    seg_colors = lambda m: (TEAL if m == PRISM else GRAY) + [CRIM]
    counts = lambda m: [int((RK[m] == p).sum()) for p in range(1, 6)]
    order = sorted(METHODS, key=lambda m: RK[m].mean())      # best mean rank on top

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 9, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    BAR_H, n = 0.46, len(order)
    for i, m in enumerate(order):
        y = (n - 1) - i
        is_p = m == PRISM
        cs, c, left = seg_colors(m), counts(m), 0
        for p in range(5):
            w = c[p]
            if w == 0:
                continue
            ax.barh(y, w, left=left, height=BAR_H, color=cs[p], edgecolor="white",
                    linewidth=1.1, zorder=2)
            if w >= 2:
                ax.text(left + w / 2.0, y, str(w), ha="center", va="center", color=txt_on(cs[p]),
                        fontsize=8, fontweight="bold" if p == 4 else "normal", zorder=4)
            left += w
        ax.text(-0.8, y, LABEL[m], ha="right", va="center", fontsize=10.5 if is_p else 9.5,
                fontweight="bold" if is_p else "normal", color=TEAL_DARK if is_p else INK)
        # Stacked, not side by side: side by side these two labels needed 115pt of
        # a figure that has to print 273pt wide, which is what forced the drawing
        # to be so much wider than its placement (2026-07-20 legibility pass).
        ax.text(N + 1.0, y + 0.24, f"mean {RK[m].mean():.2f}", ha="left", va="center", fontsize=8,
                color=TEAL_DARK if is_p else MUTE, fontweight="bold" if is_p else "normal")
        worst = c[4]
        ax.text(N + 1.0, y - 0.24, "0 worst" if worst == 0 else f"{worst}× worst", ha="left",
                va="center", fontsize=8, color=(TEAL_DARK if worst == 0 else CRIM),
                fontweight="bold" if (is_p or worst == 0) else "normal")

    ax.set_xlim(0, N)
    ax.set_ylim(-0.75, n - 0.30)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    if CFG == "opt0":
        # Caption owns the description; the best...worst legend at the bottom already
        # states the rank key, so the panel carries no title or top reading key.
        title = ""
        sub = ""
    else:
        # opt10 title + subtitle removed 2026-07-17 -> moved to LaTeX caption (Principle 3),
        # so the opt10 branch now matches the opt0 branch (blank title/sub).
        title = ""
        sub = ""
    if title:
        ax.text(-0.8, n - 0.05, title, ha="left", va="bottom", fontsize=11, fontweight="bold", color=INK)
        ax.text(-0.8, n - 0.34, sub, ha="left", va="bottom", fontsize=8.3, color="#5c5c5c")
    elif sub:
        ax.text(-0.8, n - 0.05, sub, ha="left", va="bottom", fontsize=8.3, color="#5c5c5c")

    # Rank key, moved to the RIGHT MARGIN 2026-07-19 (Yanir). It used to sit in a band under
    # the bars, which added height to the figure and therefore cost vertical space on the
    # page. The right side is already occupied only by short "mean"/"worst" labels and has
    # room to spare, so the key now stacks vertically there and the float loses that band.
    # The words "best"/"worst" and the "= PRISM-GEP (our method)" swatch stay deleted: that
    # is caption text, and the caption carries it.
    kx, kw, kh = N + KEY_X, 1.6, 0.34
    ytop = ybot = None
    for j, col in enumerate(GRAY + [CRIM]):
        y = (n - 1) - j * (kh + 0.10) - 0.55
        ax.add_patch(Rectangle((kx, y), kw, kh, color=col, ec="white", lw=0.6,
                               clip_on=False, zorder=5))
        ax.text(kx + kw + 0.7, y + kh / 2, str(j + 1), ha="left", va="center",
                fontsize=F_KEY, color=MUTE, clip_on=False)
        ytop = y + kh if ytop is None else ytop
        ybot = y
    # "best" above rank 1 and "worst" below rank 5. These sit in the right margin, which is
    # otherwise empty, so they cost no page height, and they let the caption drop its own
    # "from 1 (best) to 5 (worst)" sentence, which did cost a line.
    ax.text(kx + kw / 2, ytop + 0.10, "best", ha="center", va="bottom",
            fontsize=F_KEY, color=MUTE, clip_on=False)
    ax.text(kx + kw / 2, ybot - 0.10, "worst", ha="center", va="top",
            fontsize=F_KEY, color=MUTE, clip_on=False)

    plt.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    stem = "gobp_robustness_headline" if CFG == "opt0" else "gobp_robustness_headline_opt10"
    save_and_deploy(fig, figures_dir() / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT / stem}.pdf  ({CFG}: PRISM #best={p_best}, #worst={p_worst})")


if __name__ == "__main__":
    main()
