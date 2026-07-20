"""Held-out perplexity model selection for the topic count K, via sklearn LDA.

Why sklearn and not MALLET: the lab-patched MALLET build REQUIRES a beta-file and
its held-out estimator (MarginalProbEstimator.leftToRight) crashes
("Index .. out of bounds for length 0") on this build, so MALLET held-out LL is
not computable here. sklearn's LatentDirichletAllocation gives the standard
held-out perplexity model-selection curve (symmetric Dirichlet prior, variational
inference). This answers "does a held-out criterion pick a parsimonious K, or
favour more topics?" — orthogonal to the gene-set metrics. It is NOT the exact
MALLET+corpus-beta model; the perplexity-vs-K SHAPE is what selects K.

Output: outputs/k_selection/heldout_sklearn.csv  +  per-dataset summary + plot.

Run with --plot-only to redraw the figure from the cached CSV without refitting.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation as LDA

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS / "scripts"))
from figsafe import save_and_deploy  # noqa: E402

OUT = WS / "outputs" / "k_selection"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Print legibility (2026-07-20)
# ---------------------------------------------------------------------------
# supplementary.tex places this at 0.82\linewidth in a single column, i.e. 208pt
# wide. This script never set a font size, so everything inherited matplotlib's
# 10pt default, was drawn 487pt wide and then shrunk to 0.43x -- 3.4pt on paper.
# Drawing at very nearly the placed width instead means the source point sizes
# below are, to within a few percent, the sizes the reader gets.
FIG_W_IN, FIG_H_IN = 2.92, 1.90
F_TICK, F_LABEL, F_LEGEND = 6.2, 6.8, 5.8


def plot(out: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    for ds in out.dataset.unique():
        sub = out[out.dataset == ds].sort_values("K")
        ax.plot(sub.K, sub.perplexity, marker="o", ms=2.6, lw=1.0,
                label=ds.replace("_", " "))
    ax.axvline(5, color="k", ls="--", lw=0.9, label="paper K=5")
    ax.set_xlabel("number of topics K", fontsize=F_LABEL)
    ax.set_ylabel("held-out perplexity", fontsize=F_LABEL)
    ax.tick_params(labelsize=F_TICK, length=2.2, pad=1.5)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    # in-figure descriptive title removed 2026-07-10 -> moved to LaTeX caption
    # Two columns, parked in the empty band between the pancreas/Zeisel curves and
    # bonemarrow. A single column at this font is taller than that band, and
    # matplotlib's "best" then drops it on top of the bonemarrow curve.
    ax.legend(fontsize=F_LEGEND, ncol=2, loc="center", labelspacing=0.28,
              columnspacing=0.9, handlelength=1.3, handletextpad=0.4,
              borderpad=0.30, framealpha=0.88)
    ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout(pad=0.35)
    dst = OUT / "heldout_perplexity_vs_K.pdf"
    save_and_deploy(fig, dst, bbox_inches="tight")
    print(f"\nWROTE {dst}")

DATASETS = ["pbmc3k", "zeisel_brain", "pancreas", "bonemarrow",
            "hemogenic_endothelium", "gastrulation_erythroid"]
K_GRID = [2, 3, 5, 8, 10, 15, 20, 30]
TEST_FRAC = 0.15
SEED = 0
MAX_CELLS = 4000   # subsample larger datasets for tractability
MAX_ITER = 25


def data_csv(ds: str) -> Path:
    return WS / "data" / ds / f"filtered_{ds}_cells_x_genes.csv"


def main() -> int:
    rows = []
    for ds in DATASETS:
        csv = data_csv(ds)
        if not csv.exists():
            print(f"[{ds}] no csv -- SKIP", flush=True)
            continue
        df = pd.read_csv(csv, index_col=0)
        X = df.values.astype(float)
        rng = np.random.default_rng(SEED)
        if X.shape[0] > MAX_CELLS:
            keep = rng.choice(X.shape[0], MAX_CELLS, replace=False)
            X = X[keep]
        idx = rng.permutation(X.shape[0])
        nt = max(1, int(TEST_FRAC * X.shape[0]))
        Xte, Xtr = X[idx[:nt]], X[idx[nt:]]
        print(f"[{ds}] train {Xtr.shape} test {Xte.shape}", flush=True)
        for K in K_GRID:
            t = time.time()
            m = LDA(n_components=K, learning_method="batch", max_iter=MAX_ITER,
                    random_state=SEED, n_jobs=1).fit(Xtr)
            perp = float(m.perplexity(Xte))
            rows.append({"dataset": ds, "K": K, "perplexity": perp,
                         "ll_per_token": -np.log(perp)})
            print(f"  K={K:2d}  perplexity={perp:9.1f}  ({time.time()-t:.0f}s)",
                  flush=True)
            pd.DataFrame(rows).to_csv(OUT / "heldout_sklearn.csv", index=False)

    out = pd.DataFrame(rows)
    print("\n" + "=" * 64)
    print("HELD-OUT PERPLEXITY K SELECTION (min perplexity = best fit)")
    print("=" * 64)
    for ds in out.dataset.unique():
        sub = out[out.dataset == ds].sort_values("K")
        best = sub.loc[sub.perplexity.idxmin()]
        p5 = sub[sub.K == 5]["perplexity"]
        p5v = float(p5.iloc[0]) if len(p5) else float("nan")
        curve = ", ".join(f"K{int(r.K)}={r.perplexity:.0f}" for _, r in sub.iterrows())
        print(f"  {ds:>22s}: held-out best K={int(best.K):2d} "
              f"(perpl {best.perplexity:.0f}); K=5 perpl {p5v:.0f} | {curve}")

    try:
        plot(out)
    except Exception as e:
        print("plot skipped:", e)
    print(f"WROTE {OUT/'heldout_sklearn.csv'}")
    return 0


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot(pd.read_csv(OUT / "heldout_sklearn.csv"))
        raise SystemExit(0)
    raise SystemExit(main())
