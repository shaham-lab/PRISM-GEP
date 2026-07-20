"""Cell-trajectory recovery on EVERY benchmark dataset that has per-cell labels.

Answers "what happens if we add all the datasets to the cell-ordering table?" by actually
computing it, instead of pre-filtering on judgement. 14 of 15 datasets run (breast_cancer has
zero per-cell annotation on disk, so nothing can be correlated against).

The PRISM ordering is the SHIPPED rule, imported verbatim from cell_ordering_extra_steps so the geometry
is identical to the published row: E3_JS_DIFFMAP_DC1 = leading non-trivial diffusion coordinate
of the Jensen-Shannon kNN graph over the GEP-attribution vectors.

TWO DIFFERENCES FROM THE SHIPPED TABLE, both deliberate and both disclosed in the output:
  1. K_topics=5 for every dataset (from outputs/optimize_interval_grid/prism_opt0/), whereas the
     three shipped rows use K=#published-cell-types (8/9/7). K=5 is the only K available for all
     14. The three shipped datasets are re-run at K=5 here too, so the panel is internally
     consistent and the K effect is directly readable against the published values.
  2. Datasets whose labels have no published linear ordering are still scored, using an ordering
     recorded in ORDERINGS with provenance='arbitrary'. That is what the permutation floor is for.

PERMUTATION FLOOR (the point of this script). For each dataset we shuffle the label->rank MAP
(preserving class sizes) B times and re-score the SAME PRISM ordering. p95 of that null is the
score an ordering of this dataset achieves by chance. A real rho at or below the floor means the
row measures nothing, which converts "steady-state tissue, no lineage" from an opinion into a
number. Trivial-predictor floors (library size, PC1) are reported for the same reason: a ground
truth that library size alone recovers is not testing trajectory inference.

    python scripts/traj_cell_all_datasets.py --seeds 0 1 2      # quick look
    python scripts/traj_cell_all_datasets.py                    # all 10 seeds
Writes outputs/trajectory/cell_traj_all_datasets.csv (+ _perseed.csv) and prints the table.
"""
from __future__ import annotations
import argparse, glob, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS)); sys.path.insert(0, str(WS / "scripts"))
from cell_ordering_extra_steps import js_diffmap_dc1, absrho  # noqa: E402  (shipped geometry, verbatim)

GRID = "outputs/optimize_interval_grid/prism_opt0/{ds}/seed{s}/doc_topics.txt"
B_PERM = 500

# provenance: 'shipped'   = already in the supplementary table
#             'published' = the dataset's own source paper states this ordering
#             'inferred'  = pairwise relations are published, this linearisation is ours
#             'arbitrary' = no developmental axis exists; ordering chosen to have something to score
ORDERINGS = {
    "pancreas": ("shipped", "Ductal|Ngn3 low EP|Ngn3 high EP|Pre-endocrine|Endocrine"),
    "gastrulation": ("shipped", "E6.5|E6.75|E7.0|E7.25|E7.5|E7.75|E8.0|E8.25|E8.5"),
    "gastrulation_erythroid": ("shipped", "E7.0|E7.25|E7.5|E7.75|E8.0|E8.25|E8.5"),
    "dentategyrus": ("published", "Radial Glia-like|nIPC|Neuroblast|Granule immature|Granule mature"),
    "ventral_neuron_diff": ("published", "26 day|54 day|100 day|125 day"),
    "endoderm_diff": ("published", "0 hour|72 hour"),
    "gastrulation_e75": ("inferred", "Epiblast|Primitive Streak|Anterior Primitive Streak|Def. endoderm|Gut"),
    "bonemarrow": ("inferred", "HSC_1|Ery_1|Ery_2"),
    "paul15": ("inferred", "7MEP|6Ery|5Ery|4Ery|3Ery|2Ery|1Ery"),
    "mouse_hspc": ("inferred", "long term hematopoietic stem cell|"
                   "hematopoietic stem cell and hematopoietic multipotent progenitor cell|"
                   "megakaryocyte-erythroid progenitor cell, common myeloid progenitor and "
                   "granulocyte monocyte progenitor cell"),
    "hemogenic_endothelium": ("inferred", "endothelial cell|hematopoietic precursor cell"),
    # No developmental axis. Scored anyway so the floor can speak for itself.
    "pbmc3k": ("arbitrary", "Megakaryocytes|Dendritic cells|CD14+ Monocytes|FCGR3A+ Monocytes|"
               "B cells|NK cells|CD8 T cells|CD4 T cells"),
    "pbmc68k": ("arbitrary", "CD34+|Dendritic|CD14+ Monocyte|CD19+ B|CD56+ NK|"
                "CD4+/CD45RA+/CD25- Naive T|CD4+/CD25 T Reg|CD4+ T Helper2|CD4+/CD45RO+ Memory|"
                "CD8+/CD45RA+ Naive Cytotoxic|CD8+ Cytotoxic T"),
    "zeisel_brain": ("arbitrary", "astrocytes_ependymal|oligodendrocytes|microglia|endothelial-mural|"
                     "interneurons|pyramidal SS|pyramidal CA1"),
}


def paths(ds):
    lab = [p for p in glob.glob(str(WS / "data" / "*" / "cell_type_labels.csv"))
           if os.path.basename(os.path.dirname(p)).lower() == ds.lower()]
    cnt = glob.glob(str(WS / "data" / "*" / f"filtered_{ds}_cells_x_genes.csv"))
    return (lab[0] if lab else None), (cnt[0] if cnt else None)


# Datasets whose cell_type_labels.csv is NOT the vocabulary the ordering is written in.
# pancreas: the csv holds the FINE 8-label set, the shipped spec uses h5ad clusters_coarse.
# Reading the csv silently drops every Endocrine cell (coverage 65% instead of 100%).
H5AD_LABELS = {"pancreas": ("data/pancreas/endocrinogenesis_day15.h5ad", "clusters_coarse")}


def label_map(ds, lab_p):
    if ds in H5AD_LABELS:
        import anndata as ad
        rel, col = H5AD_LABELS[ds]
        with __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            A = ad.read_h5ad(WS / rel)
        return dict(zip(A.obs_names.astype(str), A.obs[col].astype(str)))
    lab = pd.read_csv(lab_p)
    return dict(zip(lab["cell_id"].astype(str), lab["cell_type"].astype(str)))


def load(ds):
    """-> (cell_ids aligned to doc_topics rows, rank array, keep mask, order list)."""
    lab_p, cnt_p = paths(ds)
    ids = pd.read_csv(cnt_p, index_col=0, usecols=[0]).index.astype(str).to_numpy()
    m = label_map(ds, lab_p)
    order = ORDERINGS[ds][1].split("|")
    rmap = {n: i for i, n in enumerate(order)}
    lab_of = np.array([m.get(c, "\0") for c in ids])
    keep = np.array([l in rmap for l in lab_of])
    rank = np.array([rmap.get(l, -1) for l in lab_of], float)
    return ids, lab_of, rank, keep, order


def doc_topics(ds, s, n_expected):
    p = WS / GRID.format(ds=ds, s=s)
    if not p.exists():
        return None
    P = pd.read_csv(p, sep="\t", header=None).iloc[:, 2:].to_numpy(float)
    if P.shape[0] != n_expected:
        print(f"  [{ds}/seed{s}] doc_topics rows {P.shape[0]} != {n_expected} -- skip", flush=True)
        return None
    P = np.clip(P, 1e-12, None)
    return P / P.sum(axis=1, keepdims=True)


def trivial_floors(ds, keep, rank_k):
    """|rho| of library size and PC1 of log1p counts -- predictors that use no model at all."""
    _, cnt_p = paths(ds)
    X = pd.read_csv(cnt_p, index_col=0).to_numpy(float)[keep]
    lib = absrho(X.sum(axis=1), rank_k)
    L = np.log1p(X / np.maximum(X.sum(axis=1, keepdims=True), 1) * 1e4)
    L -= L.mean(axis=0)
    # PC1 via randomized SVD on the centred matrix
    from sklearn.utils.extmath import randomized_svd
    _, _, Vt = randomized_svd(L, n_components=1, random_state=0)
    return lib, absrho(L @ Vt[0], rank_k)


def perm_floor(order_vec, lab_k, order, true_rho, rng):
    """How special is the TRUE ordering among all orderings of the same labels?

    Enumerate (or sample) permutations of the label->rank map, class sizes preserved, and rescore
    the same PRISM ordering. Returns (frac_ge, best, mean): frac_ge = fraction of orderings that do
    at least as well as the published one. Small frac_ge means the published order is genuinely
    what the ordering recovers. frac_ge near 1 means any ordering scores about the same, so the row
    measures nothing.

    p95 is NOT usable here: abs() makes an ordering and its exact reversal score identically, and
    for k<=5 there are only k! distinct orderings, so the upper tail collides with the true value.
    """
    from itertools import permutations as iperm
    from math import factorial
    k = len(order)
    codes = np.array([order.index(l) for l in lab_k])
    exact = factorial(k) <= 5040
    perms = list(iperm(range(k))) if exact else [tuple(rng.permutation(k)) for _ in range(B_PERM)]
    vals = []
    for p in perms:
        vals.append(absrho(order_vec, np.asarray(p, float)[codes]))
    v = np.array([x for x in vals if x == x])
    tol = 1e-9
    return float((v >= true_rho - tol).sum() / v.size), float(v.max()), float(v.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(ORDERINGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows, per = [], []

    for ds in a.datasets:
        t0 = time.time()
        prov, _ = ORDERINGS[ds]
        try:
            ids, lab_of, rank, keep, order = load(ds)
        except Exception as e:  # noqa: BLE001
            print(f"[{ds}] load failed: {e!r}", flush=True); continue
        rank_k, lab_k = rank[keep], lab_of[keep]
        cov = 100.0 * keep.sum() / keep.size
        print(f"[{ds}] {prov}, {len(order)} ranks, {keep.sum()}/{keep.size} cells ({cov:.1f}%)", flush=True)

        scores, floor95, floormean = [], [], []
        for s in a.seeds:
            P = doc_topics(ds, s, keep.size)
            if P is None:
                continue
            try:
                o = js_diffmap_dc1(P[keep])
            except Exception as e:  # noqa: BLE001
                print(f"  seed{s} FAILED {e!r}", flush=True); continue
            r = absrho(o, rank_k)
            scores.append(r); per.append(dict(dataset=ds, seed=s, rho=r))
            if s == a.seeds[0]:
                fr, bst, fm = perm_floor(o, lab_k, order, r, rng)
                floor95.append((fr, bst)); floormean.append(fm)
            print(f"  seed{s} |rho|={r:.3f}", flush=True)
        if not scores:
            print(f"[{ds}] no seeds scored", flush=True); continue
        try:
            lib, pc1 = trivial_floors(ds, keep, rank_k)
        except Exception as e:  # noqa: BLE001
            print(f"  trivial floors failed: {e!r}", flush=True); lib = pc1 = np.nan
        rows.append(dict(dataset=ds, provenance=prov, n_ranks=len(order), n_cells=int(keep.sum()),
                         pct_cells=round(cov, 1), prism_mean=float(np.mean(scores)),
                         prism_std=float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                         n_seeds=len(scores),
                         perm_frac_ge=floor95[0][0] if floor95 else np.nan,
                         perm_best=floor95[0][1] if floor95 else np.nan,
                         perm_mean=floormean[0] if floormean else np.nan,
                         libsize_rho=lib, pc1_rho=pc1))
        r = rows[-1]
        print(f"[{ds}] PRISM {r['prism_mean']:.3f}+/-{r['prism_std']:.3f} | orderings >= true:"
              f" {100*r['perm_frac_ge']:.1f}% (best any order {r['perm_best']:.3f})"
              f" | lib {lib:.3f} | PC1 {pc1:.3f}  ({time.time()-t0:.0f}s)\n", flush=True)
        out = WS / "outputs" / "trajectory"
        pd.DataFrame(rows).to_csv(out / "cell_traj_all_datasets.csv", index=False)
        pd.DataFrame(per).to_csv(out / "cell_traj_all_datasets_perseed.csv", index=False)

    df = pd.DataFrame(rows)
    if df.empty:
        print("nothing scored"); return
    df["order_special"] = df.perm_frac_ge <= 0.05
    df["beats_trivial"] = df.prism_mean > df[["libsize_rho", "pc1_rho"]].max(axis=1)
    print("\n=== cell-trajectory recovery, all datasets, PRISM E3_JS_DIFFMAP_DC1 @ K_topics=5 ===")
    print(df.sort_values(["provenance", "prism_mean"], ascending=[True, False]).round(3).to_string(index=False))
    print("\nperm_frac_ge   = fraction of ALL orderings of these labels scoring >= the published one.")
    print("order_special  = that fraction is <=5%, i.e. the published order is what PRISM recovers,")
    print("                 not just any ordering. Near 1.0 means the row measures nothing.")
    print("beats_trivial  = PRISM above BOTH library size and PC1 (tests more than depth/variance).")


if __name__ == "__main__":
    main()
