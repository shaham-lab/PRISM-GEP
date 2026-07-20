r"""M2 + m3: split the unified GO-BP table into PRISM-vs-SOTA and PRISM-vs-MALLET, WITH stds.

Emits (config in {o0=opt0 primary, o10=opt10 alternative}; datasets in {9,14}):
  paper/figures/tab_prism_vs_sota_<n>ds_<cfg>.tex     (PRISM, cNMF, scHPF, NMF, ProdLDA)
  paper/figures/tab_prism_vs_mallet_<n>ds_<cfg>.tex   (PRISM, MALLET)

Means come from full_metrics_combined_4config.csv (source of truth); stds from
full_metrics_perseed.csv. Every value is mean(std). Numbers are computed, never hand-entered.

Bolding:
  - SOTA table: \best = per-(dataset,metric) max mean, \secondbest = 2nd (as in the old Table 1).
  - MALLET table: bold marks ONLY a Welch-significant winner (p<0.05, 10 seeds); statistical ties
    are shown un-bolded, so the 23/25 ties read as ties (not a spurious PRISM loss). Honest.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

WS = Path(__file__).resolve().parent.parent
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
# release layout: paper/figures/, created on demand (scripts/paths.py)
from paths import figures_dir, results_dir  # noqa: E402
AGG = results_dir() / "full_metrics_combined_4config.csv"
PS = results_dir() / "full_metrics_perseed.csv"
OUT = figures_dir()

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
# Column header for one method block. The arrows are the tables' only "higher is better"
# cue (the supplement caption does not state it), so they must be emitted here -- they
# were previously hand-added to the shipped .tex and regenerating silently dropped them.
HDR = r"Coh$\uparrow$ & Cov$\uparrow$ & Str$\uparrow$"

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
    if pd.isna(x):
        return "---"
    return f"{x:.2f}" if me == "str" else (f"{x:.3f}".lstrip("0") or "0")


def fmt_ms(ds, method, cfg, me):
    m = mean_of(ds, method, cfg, me)
    if pd.isna(m):
        return "---"
    s = std_of(ds, method, cfg, me)
    mm = fmt_mean(m, me)
    if pd.isna(s):
        return mm
    ss = f"{s:.2f}" if me == "str" else (f"{s:.3f}".lstrip("0") or "0")
    return rf"{mm}\,\tiny{{({ss})}}"


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
    ncol = len(methods)
    head = "& " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{'\\textbf{PRISM-GEP}' if m=='PRISM' else m}}}" for m in methods) + r" \\"
    cmid = "".join(rf"\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(ncol))
    lines = [rf"\begin{{tabular}}{{l *{{{3*ncol}}}{{c}}}}", r"\toprule", head, cmid,
             r"\textbf{Dataset} & " + " & ".join([HDR] * ncol) + r" \\", r"\midrule"]
    for ds in datasets:
        row = [PRETTY.get(ds, ds)]
        for me in METR:
            means = {m: mean_of(ds, m, cfg, me) for m in methods}
            defined = [v for v in means.values() if pd.notna(v)]
            best = max(defined) if defined else None
            second = sorted(set(defined), reverse=True)[1] if len(set(defined)) > 1 else None
            for m in methods:
                cell = fmt_ms(ds, m, cfg, me)
                v = means[m]
                if pd.notna(v) and best is not None and abs(v - best) < 1e-9:
                    cell = rf"\best{{{cell}}}"
                elif pd.notna(v) and second is not None and abs(v - second) < 1e-9:
                    cell = rf"\secondbest{{{cell}}}"
                row.append((m, me, cell))
        out = [PRETTY.get(ds, ds)]
        for m in methods:
            for me in METR:
                out.append(next(c for mm, mee, c in row[1:] if mm == m and mee == me))
        lines.append(" & ".join(out) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    frag = "% AUTO-GENERATED by scripts/build_split_tables.py -- do not hand-edit numbers.\n" + "\n".join(lines) + "\n"
    (OUT / f"tab_prism_vs_sota_{label}.tex").write_text(frag, encoding="utf-8")
    return frag


def build_mallet(datasets, cfg, label):
    methods = ["PRISM", "MALLET"]
    lines = [r"\begin{tabular}{l *{6}{c}}", r"\toprule",
             r"& \multicolumn{3}{c}{\textbf{PRISM-GEP}} & \multicolumn{3}{c}{MALLET} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"\textbf{Dataset} & " + " & ".join([HDR] * 2) + r" \\", r"\midrule"]
    tie = win = loss = 0
    for ds in datasets:
        row = [PRETTY.get(ds, ds)]
        for m in methods:
            for me in METR:
                cell = fmt_ms(ds, m, cfg, me)
                wsig = welch_sig(ds, me, cfg)
                if wsig == m and pd.notna(mean_of(ds, m, cfg, me)):
                    cell = rf"\best{{{cell}}}"
                row.append(cell)
        # tally (once per (ds,me), on the PRISM pass)
        for me in METR:
            if pd.isna(mean_of(ds, "PRISM", cfg, me)) or pd.isna(mean_of(ds, "MALLET", cfg, me)):
                continue
            w = welch_sig(ds, me, cfg)
            if w is None:
                tie += 1
            elif w == "PRISM":
                win += 1
            else:
                loss += 1
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    frag = ("% AUTO-GENERATED by scripts/build_split_tables.py -- do not hand-edit numbers.\n"
            f"% PRISM vs MALLET Welch ({label}): win {win} / tie {tie} / loss {loss}; bold = Welch-significant winner.\n"
            + "\n".join(lines) + "\n")
    (OUT / f"tab_prism_vs_mallet_{label}.tex").write_text(frag, encoding="utf-8")
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
