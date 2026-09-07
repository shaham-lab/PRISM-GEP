r"""M2 + m3: split the unified GO-BP table into PRISM-vs-SOTA and PRISM-vs-MALLET, WITH stds.

Emits (config in {o0=opt0 primary, o10=opt10 alternative}; datasets in {9,14,15}):
  results/tables/tab_prism_vs_sota_<n>ds_<cfg>.csv     (PRISM, cNMF, scHPF, NMF, ProdLDA)
      tidy rows: dataset, method, metric, mean, sd, mark
  results/tables/tab_prism_vs_mallet_<n>ds_<cfg>.csv   (PRISM, MALLET)
      one row per (dataset, metric): prism_mean, prism_sd, mallet_mean, mallet_sd, welch_winner

Means come from full_metrics_combined_4config.csv (source of truth); stds from
full_metrics_perseed.csv. Coherence and coverage to 3 decimals, strength to 2. Numbers are
computed, never hand-entered.

Marks:
  - SOTA table: mark = "best" for the per-(dataset,metric) max mean, "second" for the 2nd.
  - MALLET table: welch_winner names ONLY a Welch-significant winner (p<0.05, 10 seeds);
    statistical ties are "tie", so the 23/25 ties read as ties (not a spurious PRISM loss).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

WS = Path(__file__).resolve().parent.parent
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
# release layout: results/tables/, created on demand (scripts/paths.py)
from paths import tables_dir, results_dir  # noqa: E402
AGG = results_dir() / "full_metrics_combined_4config.csv"
PS = results_dir() / "full_metrics_perseed.csv"
OUT = tables_dir()

NINE = ["breast_cancer", "pbmc3k", "zeisel_brain", "hemogenic_endothelium", "pancreas",
        "gastrulation_e75", "gastrulation", "gastrulation_erythroid", "bonemarrow"]
FOURTEEN = NINE[:5] + ["gastrulation", "gastrulation_e75", "gastrulation_erythroid", "bonemarrow",
                       "paul15", "dentategyrus", "pbmc68k", "endoderm_diff", "ventral_neuron_diff"]
FOURTEEN = list(dict.fromkeys(NINE + ["paul15", "dentategyrus", "pbmc68k", "endoderm_diff", "ventral_neuron_diff"]))
FIFTEEN = FOURTEEN + ["mouse_hspc"]
PRETTY = {"breast_cancer": "BreastCancer", "pbmc3k": "PBMC3k", "zeisel_brain": "Zeisel brain",
          "hemogenic_endothelium": "Hemogenic Endo.", "pancreas": "Pancreas",
          "gastrulation_e75": "Gastrulation E7.5", "gastrulation": "Gastrulation (full)",
          "gastrulation_erythroid": "Gastrulation Eryth.", "bonemarrow": "Bonemarrow",
          "paul15": "Paul HSC", "dentategyrus": "Dentate Gyrus", "pbmc68k": "PBMC68k",
          "endoderm_diff": "Endoderm diff.", "ventral_neuron_diff": "Ventral neuron",
          "mouse_hspc": "Mouse HSPC"}
METR = ["coh", "cov", "str"]
LONG = {"coh": "coherence", "cov": "coverage", "str": "strength"}
# All three metrics: higher is better.

df = pd.read_csv(AGG).set_index("dataset")
ps = pd.read_csv(PS)
STD = ps.groupby(["dataset", "variant"])[["coherence", "coverage", "strength"]].std(ddof=1)


def cfg_long(cfg):  # o0 -> opt0
    return "opt0" if cfg == "o0" else "opt10"


def comb_col(method, cfg, me):
    if method in ("PRISM", "MALLET"):
        return f"{method}{cfg}_{me}"
    return f"{method}_{me}"


def variant(method, cfg):
    if method in ("PRISM", "MALLET"):
        return f"{method}_{cfg_long(cfg)}"
    return method


def mean_of(ds, method, cfg, me):
    c = comb_col(method, cfg, me)
    return df.loc[ds, c] if c in df.columns else np.nan


def std_of(ds, method, cfg, me):
    v = variant(method, cfg)
    try:
        return STD.loc[(ds, v), LONG[me]]
    except KeyError:
        return np.nan


def fmt_mean(x, me):
    """Strength to 2 decimals, coherence/coverage to 3; undefined stays blank (None)."""
    if pd.isna(x):
        return None
    return round(float(x), 2) if me == "str" else round(float(x), 3)


def fmt_ms(ds, method, cfg, me):
    """(mean, sd) for one cell, rounded as fmt_mean; sd is None where no per-seed rows exist."""
    m = mean_of(ds, method, cfg, me)
    if pd.isna(m):
        return None, None
    s = std_of(ds, method, cfg, me)
    return fmt_mean(m, me), (None if pd.isna(s) else fmt_mean(s, me))


def welch_sig(ds, me, cfg):
    """Return (winner or None) among PRISM/MALLET if Welch p<0.05 else None (tie)."""
    a = ps[(ps.dataset == ds) & (ps.variant == f"PRISM_{cfg_long(cfg)}")][LONG[me]].dropna().values
    b = ps[(ps.dataset == ds) & (ps.variant == f"MALLET_{cfg_long(cfg)}")][LONG[me]].dropna().values
    if len(a) < 2 or len(b) < 2:
        if len(a) and len(b):
            return "PRISM" if a.mean() > b.mean() else ("MALLET" if a.mean() < b.mean() else None)
        return None
    if np.std(a) == 0 and np.std(b) == 0:
        return "PRISM" if a.mean() > b.mean() else ("MALLET" if a.mean() < b.mean() else None)
    p = stats.ttest_ind(a, b, equal_var=False).pvalue
    if p >= 0.05:
        return None
    return "PRISM" if a.mean() > b.mean() else "MALLET"


def build_sota(datasets, methods, cfg, label):
    rows = []
    for ds in datasets:
        for me in METR:
            means = {m: mean_of(ds, m, cfg, me) for m in methods}
            defined = [v for v in means.values() if pd.notna(v)]
            best = max(defined) if defined else None
            second = sorted(set(defined), reverse=True)[1] if len(set(defined)) > 1 else None
            for m in methods:
                mean, sd = fmt_ms(ds, m, cfg, me)
                v = means[m]
                mark = ""
                if pd.notna(v) and best is not None and abs(v - best) < 1e-9:
                    mark = "best"
                elif pd.notna(v) and second is not None and abs(v - second) < 1e-9:
                    mark = "second"
                rows.append(dict(dataset=ds, dataset_label=PRETTY.get(ds, ds),
                                 method="PRISM-GEP" if m == "PRISM" else m,
                                 metric=LONG[me], mean=mean, sd=sd, mark=mark))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / f"tab_prism_vs_sota_{label}.csv", index=False)
    return tab


def build_mallet(datasets, cfg, label):
    rows = []
    tie = win = loss = 0
    for ds in datasets:
        for me in METR:
            pm, psd = fmt_ms(ds, "PRISM", cfg, me)
            mm, msd = fmt_ms(ds, "MALLET", cfg, me)
            wsig = welch_sig(ds, me, cfg)
            row = dict(dataset=ds, dataset_label=PRETTY.get(ds, ds), metric=LONG[me],
                       prism_mean=pm, prism_sd=psd, mallet_mean=mm, mallet_sd=msd,
                       welch_winner=None)
            if pd.notna(mean_of(ds, "PRISM", cfg, me)) and pd.notna(mean_of(ds, "MALLET", cfg, me)):
                # tally (once per (ds,me))
                if wsig is None:
                    tie += 1
                    row["welch_winner"] = "tie"
                elif wsig == "PRISM":
                    win += 1
                    row["welch_winner"] = "PRISM-GEP"
                else:
                    loss += 1
                    row["welch_winner"] = "MALLET"
            rows.append(row)
    # PRISM vs MALLET Welch: win/tie/loss counts are printed by main(); welch_winner names
    # only a Welch-significant winner (p<0.05, 10 seeds), "tie" otherwise.
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / f"tab_prism_vs_mallet_{label}.csv", index=False)
    return win, tie, loss


def main():
    SOTA = ["PRISM", "cNMF", "scHPF", "NMF", "ProdLDA"]
    for cfg in ("o0", "o10"):
        for datasets, tag in ((NINE, "9ds"), (FOURTEEN, "14ds"), (FIFTEEN, "15ds")):
            lbl = f"{tag}_{cfg}"
            build_sota(datasets, SOTA, cfg, lbl)
            w, t, l = build_mallet(datasets, cfg, lbl)
            print(f"[{lbl}] wrote SOTA + MALLET tables; PRISM vs MALLET Welch: win {w} / tie {t} / loss {l}")
    print("OUT:", OUT)


if __name__ == "__main__":
    main()
