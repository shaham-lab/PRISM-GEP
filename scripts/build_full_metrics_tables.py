"""Full-benchmark REPRODUCTION as two tables (15 datasets x 6 methods x 3 metrics).

Everything recomputed by us (no published-imported numbers), one consistent
convention:
  * PRISM  = prism_opt0   (optimize-interval OFF -> precomputed beta used as-is)
  * MALLET = uniform_opt0  in Table A  /  uniform_opt10 in Table B   <-- the ONLY
             difference between the two tables
  * NMF / cNMF / scHPF / ProdLDA = outputs/baselines/<m> (config-independent)
Metrics via the canonical evaluator (bio.evaluate_supp): Coherence (mean pairwise
Spearman of top-20), Coverage (variant-D per-gene hit), Strength (mean -log10 q),
GO_Biological_Process_2021, top_n=20, q<0.05, mean over 10 seeds.

Resumable: per-(dataset, method_variant, seed) metrics are cached to
results/full_metrics_perseed.csv; re-runs skip cached rows. That file is shipped, so
a clean clone (which has no raw data/) rebuilds the tables straight from the cache.
Use COH_DATASETS=a,b to restrict datasets (memory: pbmc68k is 68k cells).
Output (all under results/): full_metrics_combined_4config.csv +
full_metrics_MALLETopt0.csv / full_metrics_MALLETopt10.csv + printed.
"""
from __future__ import annotations
import os, sys, glob
from pathlib import Path
import numpy as np, pandas as pd

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS)); sys.path.insert(0, str(WS / "scripts"))
from bio.evaluate_supp import coherence_supp, coverage_strength_supp   # noqa: E402
from bio.extract_top_genes import parse_topic_keys                     # noqa: E402
from evaluate_all_new_dataset_table import load_library               # noqa: E402
from paths import data_dir, outputs_dir, results_dir                  # noqa: E402

TOP_N, QTHR = 20, 0.05
ORIGINALS = ["breast_cancer", "pbmc3k", "zeisel_brain"]
EXTENSIONS = ["hemogenic_endothelium", "pancreas", "gastrulation",
              "gastrulation_e75", "gastrulation_erythroid", "bonemarrow"]
NEW = ["paul15", "dentategyrus", "pbmc68k", "endoderm_diff",
       "ventral_neuron_diff", "mouse_hspc"]
GROUP = ({d: "original" for d in ORIGINALS} | {d: "extension" for d in EXTENSIONS}
         | {d: "new" for d in NEW})
DATASETS = ORIGINALS + EXTENSIONS + NEW
_sel = os.environ.get("COH_DATASETS")
if _sel:
    DATASETS = [d for d in DATASETS if d in set(_sel.split(","))]

# 4 LDA configs (PRISM/MALLET x opt0/opt10) + config-independent baselines
VARIANTS = ["PRISM_opt0", "PRISM_opt10", "MALLET_opt0", "MALLET_opt10",
            "NMF", "cNMF", "scHPF", "ProdLDA"]
RESULTS = results_dir()
CACHE = RESULTS / "full_metrics_perseed.csv"


def _seeds(base: Path) -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(str(base / "seed[0-9]/topic_keys.txt")))]


def topic_paths(ds: str, variant: str) -> list[Path]:
    OUT = outputs_dir()
    G = OUT / "optimize_interval_grid"
    if variant == "PRISM_opt0":
        return _seeds(G / "prism_opt0" / ds)
    if variant == "PRISM_opt10":
        return _seeds(G / "prism_opt10" / ds)
    if variant == "MALLET_opt0":
        return _seeds(G / "uniform_opt0" / ds)
    if variant == "MALLET_opt10":
        return _seeds(G / "uniform_opt10" / ds)
    if variant == "cNMF":
        s = _seeds(OUT / "baselines/cnmf" / ds)
        if s:
            return s
        f = OUT / "extended_benchmark/cnmf_new" / f"{ds}_cnmf_topic_keys.txt"
        return [f] if (ds in NEW and f.exists()) else []
    mdir = {"NMF": "nmf", "scHPF": "schpf", "ProdLDA": "prodlda"}[variant]
    return _seeds(OUT / "baselines" / mdir / ds)


def eval_seed(path: Path, expr: pd.DataFrame, library: dict) -> tuple[float, float, float, int]:
    top = {k: v[:TOP_N] for k, v in parse_topic_keys(path).items()}
    if not top:
        return np.nan, np.nan, np.nan, 0
    universe = list(expr.columns)
    coh = coherence_supp(top, expr)
    covs, strs, nsig = [], [], 0
    for genes in top.values():
        c, s, n = coverage_strength_supp(genes, library, universe, q_threshold=QTHR, use_M_full=True)
        nsig += n
        if c is not None:
            covs.append(c)
        if s is not None:
            strs.append(s)
    return (coh, float(np.mean(covs)) if covs else np.nan,
            float(np.mean(strs)) if strs else np.nan, nsig)


def load_cache() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_csv(CACHE)
    return pd.DataFrame(columns=["dataset", "variant", "seed", "coherence", "coverage", "strength", "n_sig"])


def main():
    library = load_library("GO_Biological_Process_2021")
    cache = load_cache()
    done = {(r.dataset, r.variant, int(r.seed)) for r in cache.itertuples()}
    new_rows = []
    for ds in DATASETS:
        csv = data_dir() / ds / f"filtered_{ds}_cells_x_genes.csv"
        if not csv.exists():
            print(f"[skip] {ds}: no expression"); continue
        need = [(v, p) for v in VARIANTS for i, p in enumerate(topic_paths(ds, v))
                if (ds, v, i) not in done]
        if not need:
            print(f"[{ds}] all cached"); continue
        print(f"[{ds}] loading expression ({len(need)} to compute) ...", flush=True)
        expr = pd.read_csv(csv, index_col=0)
        for v in VARIANTS:
            for i, p in enumerate(topic_paths(ds, v)):
                if (ds, v, i) in done:
                    continue
                coh, cov, strg, nsig = eval_seed(p, expr, library)
                new_rows.append(dict(dataset=ds, variant=v, seed=i, coherence=coh,
                                     coverage=cov, strength=strg, n_sig=nsig))
            # flush after each variant so long runs are crash-safe
            if new_rows:
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True).to_csv(CACHE, index=False)
        del expr
        cache = load_cache(); done = {(r.dataset, r.variant, int(r.seed)) for r in cache.itertuples()}
        new_rows = []
        print(f"[{ds}] done", flush=True)

    # ---- assemble the two tables ----
    cache = load_cache()
    agg = cache.groupby(["dataset", "variant"]).agg(
        coherence=("coherence", "mean"), coverage=("coverage", "mean"),
        strength=("strength", "mean"), n=("seed", "nunique")).reset_index()

    def table(cols, disp, tag):
        rows = []
        for ds in DATASETS:
            row = {"dataset": ds, "group": GROUP[ds]}
            for c in cols:
                sub = agg[(agg.dataset == ds) & (agg.variant == c)]
                for m in ("coherence", "coverage", "strength"):
                    row[f"{disp[c]}_{m[:3]}"] = round(float(sub[m].iloc[0]), 3) if len(sub) else np.nan
                row[f"{disp[c]}_n"] = int(sub["n"].iloc[0]) if len(sub) else 0
            rows.append(row)
        df = pd.DataFrame(rows)
        out = RESULTS / f"full_metrics_{tag}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\n===== FULL METRICS — {tag} =====")
        with pd.option_context("display.width", 400, "display.max_columns", 70):
            print(df.to_string(index=False))
        print(f"WROTE {out}")

    ALL4 = ["PRISM_opt0", "PRISM_opt10", "MALLET_opt0", "MALLET_opt10", "NMF", "cNMF", "scHPF", "ProdLDA"]
    # Combined 4-config table: PRISM opt0/opt10 + MALLET opt0/opt10 + baselines.
    table(ALL4, {"PRISM_opt0": "PRISMo0", "PRISM_opt10": "PRISMo10", "MALLET_opt0": "MALLETo0",
                 "MALLET_opt10": "MALLETo10", "NMF": "NMF", "cNMF": "cNMF", "scHPF": "scHPF",
                 "ProdLDA": "ProdLDA"}, "combined_4config")
    # Focused views: PRISM(opt0) vs MALLET at each config (the two-table request).
    base = {"NMF": "NMF", "cNMF": "cNMF", "scHPF": "scHPF", "ProdLDA": "ProdLDA"}
    table(["PRISM_opt0", "MALLET_opt0"] + list(base),
          {"PRISM_opt0": "PRISM", "MALLET_opt0": "MALLET", **base}, "MALLETopt0")
    table(["PRISM_opt0", "MALLET_opt10"] + list(base),
          {"PRISM_opt0": "PRISM", "MALLET_opt10": "MALLET", **base}, "MALLETopt10")


if __name__ == "__main__":
    main()
