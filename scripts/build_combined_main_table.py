"""Build the combined main-paper table: gene ordering and cell ordering, one table.

The two tables it replaces scored the SAME nine datasets in the SAME order, so they shared a
column axis and forced the reader to look in two places. They do NOT, however, measure the
same thing, and that is the trap this layout has to avoid:

  groups 1-2  gene ordering, |rho| against the canonical MARKER order
  group 3     cell ordering, |rho| against the published LINEAGE rank of each cell

A number in group 3 is therefore not comparable with a number in group 1 or 2. Every group
header states its own metric for that reason, and the Mean/Median/Rank aggregates and the
bold/underline marks are computed WITHIN a group, never across.

Second trap: the "two label ranks only" caveat (an ordering and its reverse score alike, so
the column cannot separate methods) applies to Hemogenic and Endoderm in the CELL-ordering
sense only. In the gene-ordering groups those same datasets carry 15 and 7 markers and are
perfectly discriminating. The dagger is therefore attached to the cell-ordering group header,
not to the shared column headings.

=== 2026-07-20: the gene-ordering groups now come from the ten-seed run ===

Every row of groups 1 and 2 used to be one seed, which was not the same claim in any two rows.
The ten-seed run (outputs/gene_embedding_ablation/tenseed_2026-07-20/) established what a seed
actually selects per method, and the rows are built accordingly. They are NOT uniform, and the
caption has to say so:

  PRISM full / int.   10 seeds. A seed picks which MALLET fit Step (ii) reads. mean (sd).
  scGPT contextual    10 seeds. Near-deterministic in practice (sd = 0 on six of nine), but the
                      RNG exists (scgpt.preprocess._digitize), so it is reported as mean (sd)
                      with the zeros shown rather than hidden.
  scGPT static        DETERMINISTIC. Pure lookup in a frozen embedding matrix, no RNG on the
                      path. One value, NO sd. Faking a seed dimension here would be a lie.
  log1p               DETERMINISTIC, and seeds are the wrong instrument. The instability is
                      ill-conditioning, not sampling. See the bracket convention below.
  random              NOT a method and never was a floor. The published row was ONE draw from
                      default_rng(42), which on Dentate Gyrus happened to land at .616, the
                      null's own 95th percentile. It is now the chance level: mean (sd) over
                      10,000 random orderings per dataset.

Four value corrections are applied here, sourced from the 2026-07-20 audit artifacts rather
than hardcoded, and asserted so the script fails loudly if an artifact moves:

  log1p  gastrulation  .963 -> .160   the .963 float is not reachable by the log1p code path;
                                      it is labelled `gastrulation,prism` in the June-12 rerun
                                      CSV (an unpublished PPMI-row PRISM variant, not our
                                      published PRISM column) and leaked into the log1p row.
  log1p  erythroid     .519 -> .618   second corruption from the same commit family; .519 is
                                      exactly the MINIMUM of the attainable perturbation set.
  static hemogenic     .075 -> .064   pre-correction marker order. We fixed the hemogenic
                                      markers on 2026-07-19 and propagated it to our own
                                      column but not to scGPT's.
  static pancreas      .811 -> .831   .811 is sourced only from the June-12 file whose PRISM
                                      column is off by 0.42 on the same dataset; every current
                                      code path returns .831.

Two of the four help us (log1p gastrulation, static hemogenic) and two hurt us (log1p
erythroid, static pancreas). They ship together on purpose.

=== log1p bracket convention ===

On six of the nine datasets the gene-gene cosine graph is disconnected, the second diffusion
eigenvalue is exactly 1, and the ordering eigenvector is an arbitrary basis choice inside a
degenerate eigenspace. Printing a bare point value there implies a determinacy the cell does
not have, and printing "---" implies the pipeline produces nothing, which is also false: the
shipped pipeline returns a specific reproducible number. So those cells print the point value
followed by the interval the value spans under a 1e-7 input perturbation, in SQUARE brackets.
This is deliberately the least self-serving option available: the intervals reach ABOVE
PRISM-GEP on several datasets (erythroid to .964, Dentate Gyrus to .975, Endoderm to .945), so
the convention shows log1p's best case, not just its instability.

Brackets = perturbation range (log1p only). Round \tiny parentheses = sd over seeds. Two
parenthetical meanings, both spelled out in the caption.

The aggregate columns used to carry a third: the same aggregate with Gastrulation E7.5
excluded. Dropped 2026-07-20 (Yanir). They were introduced when E7.5 read .062, a collapsed
outlier worth setting aside. On ten seeds it reads .345, a merely low score, so excluding it
now reads as cherry-picking rather than as removing a broken measurement. The distortion it
causes also fell from .081 to .051.

Inputs : outputs/gene_embedding_ablation/tenseed_2026-07-20/seed_summary.csv
         outputs/gene_embedding_ablation/tenseed_2026-07-20/perseed_prism_and_random.csv
         outputs/gene_embedding_ablation/audit_2026-07-20/log1p_recompute.csv
         outputs/gene_embedding_ablation/audit_2026-07-20/log1p_stability.csv
         outputs/gene_embedding_ablation/audit_2026-07-20/lens5_static_recompute.csv
         outputs/gene_embedding_ablation/aggregate_metrics_9ds.csv   (published, for the asserts)
         outputs/trajectory/cell_traj_all_datasets.csv
         outputs/trajectory/cell_traj_baselines_all.csv
Output : paper/figures/tab_combined_main.tex
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS / "scripts"))
# release layout: paper/figures/, created on demand (scripts/paths.py)
from paths import figures_dir  # noqa: E402
DEFAULT_OUT = figures_dir() / "tab_combined_main.tex"
TENSEED = WS / "outputs" / "gene_embedding_ablation" / "tenseed_2026-07-20"
AUDIT = WS / "outputs" / "gene_embedding_ablation" / "audit_2026-07-20"

DS = [
    ("pancreas", "Panc."), ("gastrulation", "Gastr."),
    ("gastrulation_erythroid", "Eryth."), ("hemogenic_endothelium", "Hemog."),
    ("bonemarrow", "Bone."), ("paul15", "Paul15"),
    ("dentategyrus", "DG"), ("endoderm_diff", "Endo."),
    ("gastrulation_e75", "E75"),
]

# A second diffusion eigenvalue this close to 1 means the eigenspace is degenerate and the
# ordering eigenvector is an arbitrary choice within it.
ILLPOSED_EV2 = 1.0 - 1e-6

# The four corrections, asserted against the artifacts. (method, dataset, published, corrected).
EXPECTED_CORRECTIONS = {
    ("log1p", "gastrulation"): (0.9629500128629352, 0.16049166881048924),
    ("log1p", "gastrulation_erythroid"): (0.5188745216627709, 0.617707763884251),
    ("scgpt", "hemogenic_endothelium"): (0.0748481188565119, 0.06362090102803518),
    ("scgpt", "pancreas"): (0.8107718435854702, 0.8309690069488095),
}


def f3(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "---"
    return f"{v:.3f}".lstrip("0") if 0 <= v < 1 else f"{v:.3f}"


def cell_mean_sd(v, sd):
    """`.844\\,\\tiny{(.057)}` -- the convention already used by tab_prism_vs_sota_15ds_o0.tex."""
    if sd is None or (isinstance(sd, float) and np.isnan(sd)):
        return f3(v)
    return rf"{f3(v)}\,\tiny{{({f3(sd)})}}"


def cell_range(v, lo, hi):
    """`.160\\,\\tiny{[.009, .821]}` -- point value plus its perturbation interval.

    SQUARE brackets, so it cannot be confused with the round-parenthesis sd.
    """
    return rf"{f3(v)}\,\tiny{{[{f3(lo)}, {f3(hi)}]}}"


def crank(vals, live, higher_better=True):
    """Competition (min) rank over the live entries: tied values share the best rank.

    Ordinal ranking (argsort of an argsort) would break a tie by ROW ORDER, which hands
    the first-listed method a free rank-1 it did not earn. The sibling generator
    scripts/build_gene_embed_9ds.py already ranks by competition, so the two scripts
    disagreed on identical data until this was fixed.
    """
    ordered = sorted((vals[j] for j in live), reverse=higher_better)
    return {j: ordered.index(vals[j]) + 1 for j in live}


def mark(vals, i, fmt=f3, higher_better=True):
    """Bold the best and underline the second best in a column.

    The floors (chance level, PCA-1) COMPETE here rather than being excluded. Excluding them
    looked tidier but asserted something false. Note that the case this docstring used to cite,
    the random row taking Dentate Gyrus at .616, was an artifact of a single lucky draw: the
    true chance level there is .417, below PRISM-GEP. The principle stands anyway, and PCA-1
    still takes Endoderm in the cell-ordering group.

    A mark is only printed when the rank is held by exactly ONE method. On an exact tie
    every tied cell stays plain, because bolding one of them asserts a separation the
    numbers do not contain. The three-way .894 tie on Gastrulation Erythroid is the case
    this guard exists for, and build_gene_embed_9ds.py leaves it unmarked too.

    The guard also covers ties AT PRINTED PRECISION, not just exact ones. Ranking still runs on
    the raw values, so the mean-rank aggregate is untouched, but the mark is suppressed when
    another live cell prints the same string. Bonemarrow in group 2 is why: scGPT contextual
    .90349 leads PRISM-GEP .90262, both print .903, and Welch gives p = 0.59. A bold .903
    sitting next to a plain .903 reads as a typesetting error and claims a separation that is
    neither visible nor significant.
    """
    live = [j for j, v in enumerate(vals) if not np.isnan(v)]
    if np.isnan(vals[i]) or not live:
        return fmt(vals[i])
    ranks = crank(vals, live, higher_better=higher_better)
    s = fmt(vals[i])
    holders = sum(1 for j in live if ranks[j] == ranks[i])
    prints_same = any(j != i and fmt(vals[j]) == s for j in live)
    if holders == 1 and not prints_same:
        if ranks[i] == 1:
            return f"\\best{{{s}}}"
        if ranks[i] == 2:
            return f"\\secondbest{{{s}}}"
    return s


def _colranks(sub):
    """Competition rank down each column of `sub` (higher value = better = rank 1)."""
    out = np.empty(sub.shape, dtype=float)
    for j in range(sub.shape[1]):
        col = sub[:, j]
        live = list(range(sub.shape[0]))
        r = crank(col, live, higher_better=True)
        for i in live:
            out[i, j] = r[i]
    return out


def group_block(title, rows):
    """rows = list of (label, values, decorations). Returns latex lines.

    `values` drives every rank, aggregate and mark. `decorations` is a per-dataset list of
    already-rendered cell strings (mean+sd, or point+perturbation range) or None to print the
    bare value. Keeping the two apart is the point: the sd and the perturbation interval are
    presentation, never inputs to a comparison.

    Every aggregate (Mean, Med., Rank) is computed over all nine datasets.
    """
    M = np.array([r[1] for r in rows], dtype=float)
    means = np.nanmean(M, axis=1)
    meds = np.nanmedian(M, axis=1)
    # mean rank within the group, over datasets where every member is defined
    ok = ~np.isnan(M).any(axis=0)
    if ok.sum():
        sub = M[:, ok]
        ranks = _colranks(sub)
        mranks = ranks.mean(axis=1).astype(float)
    else:
        mranks = np.full(len(rows), np.nan)

    def agg_cell(full_vec, i, fmt=f3, higher_better=True):
        return mark(full_vec, i, fmt=fmt, higher_better=higher_better)

    # Group headers are BOLD, not italic (Yanir 2026-07-19). Set here rather than by hand in
    # the fragment: this file is regenerated, so a hand edit there is silently reverted the
    # next time anyone runs the script.
    out = [f"\\multicolumn{{13}}{{l}}{{\\textbf{{{title}}}}}\\\\"]
    for i, (label, vals, deco) in enumerate(rows):
        cells = []
        for j in range(M.shape[1]):
            plain = mark(M[:, j], i)
            d = None if deco is None else deco[j]
            if d is None:
                cells.append(plain)
            elif plain.startswith("\\best{"):
                cells.append(f"\\best{{{d}}}")
            elif plain.startswith("\\secondbest{"):
                cells.append(f"\\secondbest{{{d}}}")
            else:
                cells.append(d)
        cells.append(agg_cell(means, i))
        cells.append(agg_cell(meds, i))
        # Mean rank: two decimals, and LOWER is better, so the comparison direction flips.
        rank_fmt = lambda v: f"{v:.2f}"
        cells.append(agg_cell(mranks, i, fmt=rank_fmt, higher_better=False))
        out.append(f"{label} & " + " & ".join(cells) + " \\\\")
    return out


def load_gene_ordering(keys):
    """Return {method: (values, decorations)} for the two gene-ordering groups.

    Everything is read from the ten-seed run and the audit artifacts. The published
    aggregate_metrics_9ds.csv is loaded only to ASSERT which cells changed, so that a silent
    drift in an artifact fails the build instead of quietly reshaping the table.
    """
    pub = (pd.read_csv(WS / "outputs" / "gene_embedding_ablation" / "aggregate_metrics_9ds.csv")
           .pivot_table(index="method", columns="dataset", values="spearman_abs"))

    # ---- seeded rows: PRISM full, PRISM int., scGPT contextual (n = 10) ----
    ss = pd.read_csv(TENSEED / "seed_summary.csv").set_index(["dataset", "method"])
    seeded = {}
    for m in ("prism_full", "prism_int", "scgpt_contextual"):
        vals, deco = [], []
        for k in keys:
            row = ss.loc[(k, m)]
            assert int(row["count"]) == 10, f"{m}/{k}: expected 10 seeds, got {row['count']}"
            vals.append(float(row["mean"]))
            deco.append(cell_mean_sd(float(row["mean"]), float(row["std"])))
        seeded[m] = (vals, deco)

    # ---- chance level: 10,000 random orderings per dataset, on the FULL marker set ----
    rnd = pd.read_csv(TENSEED / "perseed_prism_and_random.csv")
    rnd = rnd[rnd["method"] == "random_full"].set_index("dataset")
    rv, rd = [], []
    for k in keys:
        row = rnd.loc[k]
        assert int(row["n_draws"]) == 10000, f"random/{k}: expected 10000 draws"
        rv.append(float(row["spearman_abs"]))
        rd.append(cell_mean_sd(float(row["spearman_abs"]), float(row["sd"])))

    # ---- log1p: deterministic, two corrections, six ill-conditioned cells ----
    lg = pd.read_csv(AUDIT / "log1p_recompute.csv")
    lg = lg[lg["variant"] == "pipeline"].set_index("dataset")
    st = pd.read_csv(AUDIT / "log1p_stability.csv").set_index("dataset")
    lv, ld, n_ill = [], [], 0
    for k in keys:
        v = float(lg.loc[k, "spearman_abs"])
        lv.append(v)
        if float(st.loc[k, "ev2"]) > ILLPOSED_EV2:
            n_ill += 1
            lo, hi = float(st.loc[k, "rho_min"]), float(st.loc[k, "rho_max"])
            assert lo <= v <= hi, f"log1p/{k}: point {v} outside perturbation range [{lo},{hi}]"
            ld.append(cell_range(v, lo, hi))
        else:
            ld.append(None)
    assert n_ill == 6, f"expected 6 ill-conditioned log1p cells, found {n_ill}"

    # ---- scGPT static: deterministic, no sd, two corrections ----
    sc = pd.read_csv(AUDIT / "lens5_static_recompute.csv").set_index("dataset")
    sv = [float(sc.loc[k, "rho_new_markers"]) for k in keys]

    # ---- assert exactly the four expected corrections, and nothing else ----
    changed = {}
    for m, series in (("log1p", lv), ("scgpt", sv)):
        for k, v in zip(keys, series):
            p = float(pub.loc[m, k])
            if abs(v - p) > 1e-9:
                changed[(m, k)] = (p, v)
    assert set(changed) == set(EXPECTED_CORRECTIONS), (
        f"correction set drifted.\n  found:    {sorted(changed)}\n"
        f"  expected: {sorted(EXPECTED_CORRECTIONS)}")
    for key, (p, v) in changed.items():
        ep, ev = EXPECTED_CORRECTIONS[key]
        assert abs(p - ep) < 1e-9 and abs(v - ev) < 1e-9, f"{key}: {p}->{v}, expected {ep}->{ev}"

    print("  corrections applied (published -> corrected):")
    for (m, k), (p, v) in sorted(changed.items()):
        print(f"    {m:8s} {k:24s} {f3(p)} -> {f3(v)}")

    return {"prism_full": seeded["prism_full"], "prism_int": seeded["prism_int"],
            "scgpt_contextual": seeded["scgpt_contextual"],
            "random": (rv, rd), "log1p": (lv, ld), "scgpt": (sv, None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    keys = [k for k, _ in DS]
    ge = load_gene_ordering(keys)

    ct = pd.read_csv(WS / "outputs" / "trajectory" / "cell_traj_all_datasets.csv").set_index("dataset")
    cb = pd.read_csv(WS / "outputs" / "trajectory" / "cell_traj_baselines_all.csv").set_index("dataset")

    def ctrow(df, col):
        return [df.loc[k, col] if k in df.index else np.nan for k in keys]

    G1 = [("PRISM-GEP (full)", *ge["prism_full"]),
          ("log1p", *ge["log1p"]),
          ("random \\emph{(chance level)}", *ge["random"])]
    G2 = [("PRISM-GEP (int.)", *ge["prism_int"]),
          ("scGPT static", *ge["scgpt"]),
          ("scGPT contextual", *ge["scgpt_contextual"])]
    # Cell ordering is outside the ten-seed run and is carried through unchanged.
    G3 = [("PRISM-GEP (JS diff.-map)", ctrow(ct, "prism_mean"), None),
          ("Slingshot", ctrow(cb, "Slingshot"), None),
          ("DPT / PAGA-DPT", ctrow(cb, "DPT"), None),
          ("PCA-1 \\emph{(floor)}", ctrow(cb, "PCA_1"), None)]

    lines = []
    lines += group_block("Gene ordering, $|\\rho|$ vs canonical marker order: full marker set",
                         G1)
    lines.append("\\midrule")
    lines += group_block("Gene ordering, $|\\rho|$ vs canonical marker order: "
                         "scGPT-embeddable intersection", G2)
    lines.append("\\midrule")
    lines += group_block("Cell ordering, $|\\rho|$ vs published lineage rank "
                         "($^{\\dagger}$Hemog.\\ and Endo.\\ have two label ranks only, "
                         "so those columns cannot separate methods)", G3)

    hdr = ("\\textbf{Method} & " + " & ".join(lbl for _, lbl in DS)
           + " & \\textbf{Mean} & \\textbf{Med.} & \\textbf{Rank} \\\\")
    frag = "\n".join([
        "% AUTO-GENERATED by scripts/build_combined_main_table.py -- do not hand-edit.",
        "% Groups measure DIFFERENT quantities. Aggregates and bold/underline are within-group.",
        "% Gene-ordering groups: ten-seed run outputs/gene_embedding_ablation/tenseed_2026-07-20/.",
        "% \\tiny(round) = sd over seeds; \\tiny[square] = log1p perturbation range.",
        "\\begin{tabular}{l" + "c" * len(DS) + "|ccc}",
        "\\toprule", hdr, "\\midrule",
        *lines,
        "\\bottomrule", "\\end{tabular}", "",
    ])
    out.write_text(frag, encoding="utf-8")
    print(f"wrote {out}")
    print(f"  3 groups, {len(G1)+len(G2)+len(G3)} method rows, {len(DS)} datasets")
    print("  bold = best in group, underline = second in group, floors compete for bold and underline")


if __name__ == "__main__":
    main()
