"""Score PRISM Step (ii) gene ordering against the canonical marker progression on
nine developmental datasets: Pancreas, Gastrulation, Gastrulation Erythroid,
Hemogenic Endothelium, Bone Marrow, Gastrulation E7.5, Paul15, Dentate Gyrus and
Endoderm-diff.

Datasets with a curated cell-lineage annotation (lineage_col set) additionally get
a DPT cell-pseudotime reference baseline; datasets without one (lineage_col=None)
are scored on PRISM Step (ii) + expression-magnitude only.

Also provides the scoring primitives (safe_spearman_abs, random_baseline_mean_rho)
used by build_traj_gene_8ds_supp.py.

For each dataset we use a small, well-known set of marker genes whose
biological order along the developmental trajectory is canonical (e.g.
Pancreas: Sox9 -> Hes1 -> Neurog3 -> Neurod1 -> Pax6 -> Ins1 -> Gcg).
We then ask: when each method is given this gene set and asked to order
it, how close is the recovered order to the canonical order?

Methods:
  1. PRISM K=5 Step (ii) — Hellinger on topic distributions + diffusion EV2
  2. PRISM K=#types Step (ii)
  3. Expression-magnitude (rank by mean expression across all cells)
  4. Cell-pseudotime-weighted mean (rank genes by their expression-weighted
     mean DPT pseudotime — a sensible baseline using a state-of-the-art
     cell-trajectory tool)
  5. Random (1000-shuffle mean rho — chance floor)

Score: |Spearman rho| between recovered order and canonical order.

Outputs:
  outputs/trajectory/gene_trajectory_scores.csv
  outputs/trajectory/<ds>/gene_trajectory_<ds>.pdf
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS))

from bio.evaluate_supp import gene_ordering_supp  # noqa: E402


# Pancreas (Bastidas-Ponce et al. 2019). Differentiation: ductal/progenitor
# -> transient Ngn3+ EP -> pre-endocrine -> mature endocrine.
MARKERS_PANCREAS = [
    ("Sox9",     0),   # Ductal/progenitor TF
    ("Hes1",     0),   # Ductal/progenitor (Notch downstream)
    ("Spp1",     0),   # Ductal
    ("Krt8",     0),   # Ductal/early
    ("Krt18",    0),   # Ductal/early
    ("Neurog3",  1),   # Ngn3+ transient EP master switch
    ("Neurod1",  2),   # Pre-endocrine TF
    ("Pax6",     2),   # Pan-endocrine TF
    ("Isl1",     2),   # Pan-endocrine TF
    ("Ins1",     3),   # Beta mature
    ("Ins2",     3),   # Beta mature
    ("Gcg",      3),   # Alpha mature
    ("Sst",      3),   # Delta mature
    ("Ghrl",     3),   # Epsilon mature
    ("Pyy",      3),   # Mature endocrine peptide
]

# Gastrulation full atlas (Pijuan-Sala et al. 2019). Stages E6.5 -> E8.5,
# from epiblast/pluripotency -> primitive streak -> mesoderm/ectoderm.
MARKERS_GASTRULATION = [
    ("Pou5f1",  0),    # Epiblast / pluripotency
    ("Nanog",   0),    # Epiblast / pluripotency
    ("Sox2",    0),    # Epiblast
    ("Eomes",   1),    # Primitive streak (early)
    ("T",       1),    # Brachyury, primitive streak
    ("Mixl1",   1),    # Primitive streak
    ("Mesp1",   2),    # Nascent mesoderm
    ("Mesp2",   2),    # Mesoderm
    ("Foxa2",   3),    # Definitive endoderm
    ("Sox17",   3),    # Definitive endoderm
    ("Hba-x",   3),    # Erythroid (late)
    ("Hbb-y",   3),    # Erythroid (late)
]

# Erythroid sub-trajectory (within gastrulation). Stages E7.0 -> E8.5,
# blood-progenitor -> primitive erythroid.
MARKERS_GASTRULATION_ERYTHROID = [
    ("Sox2",    0),    # Earlier (ectoderm contaminant) / epiblast
    ("T",       0),    # Primitive streak
    ("Tal1",    1),    # Hematoendothelial progenitor (early)
    ("Lmo2",    1),    # Heme TF
    ("Gata1",   2),    # Heme / erythroid commitment
    ("Klf1",    2),    # Erythroid TF
    ("Hbb-bh1", 3),    # Embryonic hemoglobin (early erythroid)
    ("Hba-x",   3),    # Embryonic hemoglobin
    ("Hbb-y",   3),    # Embryonic hemoglobin
]

# Hemogenic endothelium / endothelial-to-haematopoietic transition (EHT)
# (EBI E-MTAB-8271). Lineage: endothelial -> hemogenic (endocardium
# progenitor) -> haematopoietic precursor -> erythroid-committed.
# Ambiguous arterial/Notch genes (Hey1, Mycn) deliberately
# omitted. All methods are scored against this same order, so curation
# imprecision affects absolute level, not the method-vs-method ranking.
MARKERS_HEMOGENIC = [
    ("Sox17",   0),    # Arterial / endothelial
    ("Cldn5",   0),    # Endothelial tight junction
    ("Emcn",    0),    # Endomucin, endothelial
    ("Egfl7",   0),    # Endothelial
    ("Runx1",   1),    # Hemogenic endothelium master TF
    ("Gata2",   1),    # Hemogenic / endothelial-to-blood
    ("Gfi1b",   1),    # Hemogenic transition
    ("Cd44",    1),    # EHT surface marker, hemogenic transition
    ("Lmo2",    1),    # Haematoendothelial TF, hemogenic stage
    ("Tal1",    1),    # SCL, haematoendothelial TF, hemogenic stage
    ("Myb",     2),    # Definitive haematopoietic precursor
    ("Spi1",    2),    # PU.1, HSPC / myeloid
    ("Itga2b",  2),    # CD41, early haematopoietic
    ("Gypa",    3),    # Erythroid-committed (latest)
    ("Klf1",    3),    # Erythroid TF (latest)
]

# Bone marrow (human CD34+ HSPC -> erythroid maturation). HUMAN uppercase symbols.
# Lineage: HSC/multipotent -> committed progenitor -> erythroid commitment TFs ->
# erythroid-committed -> terminal globin.
MARKERS_BONEMARROW = [
    ("CD34",  0), ("HLF",   0), ("AVP",  0), ("MEIS1", 0), ("MLLT3", 0),  # HSC / multipotent
    ("GATA2", 1), ("KIT",   1), ("TAL1", 1),                              # committed progenitor
    ("GATA1", 2), ("KLF1",  2), ("NFE2", 2), ("TFRC",  2),                # erythroid commitment TFs
    ("GYPA",  3), ("ALAS2", 3), ("EPOR", 3), ("BLVRB", 3),                # erythroid-committed
    ("HBB",   4), ("HBA1",  4), ("HBA2", 4), ("AHSP",  4),                # terminal globin
]

# Gastrulation E7.5 snapshot (Pijuan-Sala et al. 2019, E7.5 subset). Pluripotency
# -> primitive streak -> early lineage specification -> differentiated lineages.
MARKERS_GASTRULATION_E75 = [
    ("Pou5f1", 0), ("Nanog", 0),                                          # pluripotent epiblast
    ("T",      1), ("Mixl1", 1), ("Fgf8",  1),                            # primitive streak
    ("Mesp1",  2), ("Snai1", 2), ("Foxa2", 2), ("Cer1",  2),             # early lineage specification
    ("Gata4",  3), ("Hand1", 3), ("Tbx5",  3), ("Sox17", 3), ("Pax6", 3),  # differentiated lineages
]
# Every E7.5 cell type is admitted (all mapped to root stage 0); DPT is seeded from
# the Epiblast root below. Used to build gastrulation_e75's lineage_map.
E75_CELLTYPES = [
    "Rostral neurectoderm", "Epiblast", "Nascent mesoderm", "Mesenchyme",
    "Mixed mesoderm", "Primitive Streak", "Caudal epiblast", "Blood progenitors 2",
    "Haematoendothelial progenitors", "Def. endoderm", "Visceral endoderm",
    "Intermediate mesoderm", "Blood progenitors 1", "Gut", "Surface ectoderm",
    "ExE mesoderm", "Pharyngeal mesoderm", "Paraxial mesoderm",
    "Anterior Primitive Streak", "Caudal neurectoderm", "Notochord", "PGC",
    "Somitic mesoderm", "Caudal Mesoderm", "Allantois", "Erythroid1", "Endothelium",
]

# Paul15 (mouse myeloid progenitor differentiation, Paul et al. 2015). Erythroid
# maturation cascade: early progenitor -> erythroid commitment -> terminal globin.
# NF-E2 (Nfe2) is deliberately excluded: it is the megakaryocyte-lineage TF
# (Shivdasani et al. 1995), i.e. the one divergent-lineage marker, not erythroid.
MARKERS_PAUL15 = [
    ("Gata2", 0),                                            # early / multipotent progenitor
    ("Gata1", 1), ("Zfpm1", 1),                             # erythroid commitment TFs
    ("Klf1",  2),                                            # erythroid maturation TF
    ("Car1",  3), ("Car2", 3), ("Hba-a2", 3), ("Hbb-b1", 3),  # terminal erythroid
]

# Dentate gyrus neurogenesis: radial glia / neural stem cell -> intermediate
# progenitor -> immature neuron -> mature granule neuron. Snap25 is ranked latest
# as a late synaptic-maturation marker, later than the immature-neuron bHLH Neurod6
# (Hodge et al. 2012).
MARKERS_DENTATEGYRUS = [
    ("Sox2",    0), ("Hes5", 0),   # radial glia / neural stem cell
    ("Eomes",   1),                # intermediate progenitor (Tbr2)
    ("Neurod6", 2),                # immature neuron
    ("Snap25",  3),                # mature granule neuron (synaptic)
]

# Definitive-endoderm differentiation: pluripotent -> primitive streak ->
# definitive endoderm.
MARKERS_ENDODERM_DIFF = [
    ("Pou5f1", 0), ("Nanog", 0),                   # pluripotent
    ("Mixl1",  1), ("Gsc",   1),                   # primitive streak
    ("Sox17",  2), ("Foxa2", 2), ("Cxcr4", 2),     # definitive endoderm
]


def load_word_topic_counts(layout_dir: Path, K: int):
    """Returns dict gene_lower -> normalized (K,) distribution."""
    f = layout_dir / "word_topic_counts.txt"
    if not f.exists():
        return None
    out = {}
    with open(f) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            gene = parts[1].lower()
            counts = np.zeros(K, dtype=float)
            for tok in parts[2:]:
                if ":" not in tok:
                    continue
                t, c = tok.split(":")
                ti = int(t)
                if ti < K:
                    counts[ti] = float(c)
            s = counts.sum()
            if s > 0:
                out[gene] = counts / s
    return out


def load_expression(dataset_csv: Path) -> pd.DataFrame:
    return pd.read_csv(dataset_csv, index_col=0)


def load_h5ad(h5ad_path: Path):
    import anndata as ad
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return ad.read_h5ad(h5ad_path)


def make_adata_from_csv(csv: Path, labels_csv: Path, label_colname: str):
    """Build an AnnData from the filtered CSV + cell_type_labels.csv,
    putting the label into adata.obs[label_colname]."""
    import anndata as ad
    expr = pd.read_csv(csv, index_col=0)
    labs = pd.read_csv(labels_csv).set_index("cell_id")
    aligned = labs.reindex(expr.index.astype(str))
    adata = ad.AnnData(X=expr.values.astype(np.float32),
                       obs=pd.DataFrame({label_colname: aligned["cell_type"].values},
                                        index=expr.index.astype(str)),
                       var=pd.DataFrame(index=expr.columns))
    return adata


def prism_step_ii_order(markers_present: list[str],
                        gene_dist: dict[str, np.ndarray]) -> np.ndarray:
    """Returns pseudotime (one value per gene in markers_present, same order)."""
    lower_markers = [g.lower() for g in markers_present]
    sub_dist = {g: gene_dist[g] for g in lower_markers if g in gene_dist}
    ordered_genes, pseudotime_in_order = gene_ordering_supp(lower_markers, sub_dist)
    name_to_pt = dict(zip(ordered_genes, pseudotime_in_order))
    return np.array([name_to_pt.get(g, np.nan) for g in lower_markers])


def expression_magnitude_order(markers_present, expr_df) -> np.ndarray:
    """Rank genes by mean expression across cells (typically the latest-stage
    genes have highest expression for high-abundance hormones; reasonable
    but naive baseline)."""
    cols_lower = {c.lower(): c for c in expr_df.columns}
    means = []
    for g in markers_present:
        col = cols_lower.get(g.lower())
        means.append(expr_df[col].mean() if col else np.nan)
    return np.array(means)


def pseudotime_weighted_mean_order(markers_present, expr_df, cell_pseudotime
                                   ) -> np.ndarray:
    """For each gene, compute the expression-weighted mean cell-pseudotime.
    A gene mostly expressed at late times will get a high value.
    cell_pseudotime: (n_cells,) aligned to expr_df rows."""
    cols_lower = {c.lower(): c for c in expr_df.columns}
    pt = np.asarray(cell_pseudotime, dtype=float)
    valid = np.isfinite(pt)
    pt_safe = np.where(valid, pt, 0.0)  # avoid NaN*0 poisoning the sum
    results = []
    for g in markers_present:
        col = cols_lower.get(g.lower())
        if col is None:
            results.append(np.nan); continue
        e = expr_df[col].values.astype(float)
        w = e * valid
        denom = w.sum()
        if denom <= 0:
            results.append(np.nan)
        else:
            results.append((pt_safe * w).sum() / denom)
    return np.array(results)


def random_baseline_mean_rho(n_markers: int, canonical_rank: np.ndarray,
                             n_iter: int = 1000) -> float:
    rng = np.random.default_rng(42)
    rhos = []
    for _ in range(n_iter):
        perm = rng.permutation(n_markers)
        rho, _ = spearmanr(perm, canonical_rank)
        rhos.append(abs(rho))
    return float(np.mean(rhos))


def safe_spearman_abs(x, y) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    rho, _ = spearmanr(x[valid], y[valid])
    return float(abs(rho))


def run_dataset(name: str, markers: list[tuple[str, int]],
                csv: Path, h5ad: Path, root_cluster: str,
                lineage_col: str, lineage_map: dict[str, int],
                prism_layouts: list[tuple[str, str, int]]):
    """prism_layouts: list of (layout, tag, K) e.g. [("seed0", "K5", 5),
                                                     ("K8/seed0", "K8", 8)]."""
    print(f"\n========== {name} ==========")
    expr = load_expression(csv)
    # Build canonical-marker presence set (gene must exist in HVG vocab)
    cols_lower = {c.lower(): c for c in expr.columns}
    markers_present = []
    canonical_rank = []
    for g, rank in markers:
        if g.lower() in cols_lower:
            markers_present.append(g)
            canonical_rank.append(rank)
    canonical_rank = np.array(canonical_rank)
    print(f"  {len(markers_present)}/{len(markers)} markers present in HVG vocab")
    print(f"  markers present: {markers_present}")
    if len(markers_present) < 4:
        print(f"  not enough markers; SKIP")
        return None

    # Cell pseudotime from DPT for the weighted-mean reference baseline. Only for
    # datasets with a curated cell-lineage annotation; when lineage_col is None
    # (paul15 / dentategyrus / endoderm_diff) this baseline is skipped and the
    # dataset is scored on PRISM Step (ii) + expression-magnitude only.
    cell_dpt_aligned = None
    if lineage_col is not None:
        print(f"  computing DPT cell pseudotime (root={root_cluster}) ...")
        import scanpy as sc
        if h5ad is not None and h5ad.exists():
            adata = load_h5ad(h5ad)
        else:
            labels_csv = csv.parent / "cell_type_labels.csv"
            adata = make_adata_from_csv(csv, labels_csv, lineage_col)
        if lineage_col not in adata.obs.columns:
            raise ValueError(f"lineage_col '{lineage_col}' not in obs (have {list(adata.obs.columns)})")
        keep = adata.obs[lineage_col].astype(str).isin(lineage_map).values
        adata = adata[keep].copy()
        a = adata.copy()
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat", subset=True)
        sc.pp.pca(a, n_comps=30)
        sc.pp.neighbors(a, n_neighbors=15)
        sc.tl.diffmap(a)
        root_idx = int(np.where(adata.obs[lineage_col] == root_cluster)[0][0])
        a.uns["iroot"] = root_idx
        sc.tl.dpt(a)
        cell_dpt = a.obs["dpt_pseudotime"].values

        # The expression CSV cell ids and the h5ad cell ids may not align by index.
        # Build a per-cell DPT vector aligned to expr.index.
        h5_ids = adata.obs.index.astype(str).tolist()
        expr_ids = expr.index.astype(str).tolist()
        h5_to_dpt = dict(zip(h5_ids, cell_dpt))
        cell_dpt_aligned = np.array([h5_to_dpt.get(cid, np.nan) for cid in expr_ids])
        print(f"  DPT-aligned to expr: {np.isfinite(cell_dpt_aligned).sum()}/{len(expr_ids)} cells")
    else:
        print("  no lineage annotation (lineage_col=None) -- skipping DPT baseline")

    scores = {}
    raw_orders = {}

    # PRISM Step (ii) at each layout
    for layout, tag, K in prism_layouts:
        layout_dir = WS / "outputs" / name / layout
        gd = load_word_topic_counts(layout_dir, K)
        if gd is None:
            print(f"  [PRISM {tag}] no word_topic_counts at {layout} -- SKIP")
            continue
        try:
            pt = prism_step_ii_order(markers_present, gd)
            scores[f"PRISM_{tag}_StepII"] = safe_spearman_abs(pt, canonical_rank)
            raw_orders[f"PRISM_{tag}_StepII"] = pt
        except Exception as e:
            print(f"  [PRISM {tag}] FAILED: {e}")

    # Baselines
    em = expression_magnitude_order(markers_present, expr)
    scores["Expression_magnitude"] = safe_spearman_abs(em, canonical_rank)
    raw_orders["Expression_magnitude"] = em

    if cell_dpt_aligned is not None:
        pwm = pseudotime_weighted_mean_order(markers_present, expr, cell_dpt_aligned)
        scores["DPT_weighted_mean"] = safe_spearman_abs(pwm, canonical_rank)
        raw_orders["DPT_weighted_mean"] = pwm

    scores["Random_chance"] = random_baseline_mean_rho(len(markers_present),
                                                       canonical_rank)
    print(f"\n  === {name}: |Spearman rho| vs canonical marker order ===")
    for n, s in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"    {n:30s}  rho={s:.4f}")

    # Persist
    out = WS / "outputs" / "trajectory" / name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for m_idx, (g, can_rank) in enumerate(zip(markers_present, canonical_rank)):
        row = {"gene": g, "canonical_rank": int(can_rank)}
        for k, vec in raw_orders.items():
            row[k] = float(vec[m_idx])
        rows.append(row)
    new_df = pd.DataFrame(rows)
    # Preserve a previously-computed GeneTrajectory column (added post-hoc by
    # scripts/run_genetrajectory.py) so re-running baselines does not wipe it.
    orders_path = out / f"gene_trajectory_{name}_orders.csv"
    if orders_path.exists():
        old = pd.read_csv(orders_path)
        if "GeneTrajectory" in old.columns:
            gt_map = dict(zip(old["gene"].astype(str), old["GeneTrajectory"]))
            new_df["GeneTrajectory"] = [gt_map.get(str(g), float("nan"))
                                        for g in new_df["gene"]]
            print(f"  preserved existing GeneTrajectory column "
                  f"({new_df['GeneTrajectory'].notna().sum()} markers)")
    new_df.to_csv(orders_path, index=False)
    (out / f"gene_trajectory_{name}_scores.json").write_text(
        json.dumps({"scores": scores,
                    "n_markers": len(markers_present),
                    "markers": markers_present,
                    "canonical_rank": canonical_rank.tolist()}, indent=2))

    # Plot
    plot_gene_panel(name, markers_present, canonical_rank, raw_orders, scores,
                    out / f"gene_trajectory_{name}.pdf")
    return scores


def plot_gene_panel(name, markers, canonical_rank, raw_orders, scores, out_path):
    """One row of subplots: each = method's predicted gene pseudotime vs
    canonical rank, with rho annotation. Last = bar chart of |rho|."""
    methods = list(raw_orders.keys()) + ["Random_chance"]
    n_methods = len(methods)
    ncols = min(4, n_methods)
    nrows = (n_methods + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.0))
    axes_flat = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

    for i, m in enumerate(methods[:-1]):
        ax = axes_flat[i]
        y = raw_orders[m]
        ax.scatter(canonical_rank, y, c="#1f77b4" if m.startswith("PRISM")
                                                   else "#ff7f0e", s=60)
        for g, x, yy in zip(markers, canonical_rank, y):
            if np.isfinite(yy):
                ax.annotate(g, (x, yy), fontsize=7, xytext=(2, 2),
                            textcoords="offset points")
        ax.set_xlabel("canonical rank")
        ax.set_ylabel("recovered pseudotime")
        ax.set_title(f"{m}\n|ρ|={scores.get(m, float('nan')):.3f}",
                     fontsize=10)
        ax.grid(alpha=0.3)

    # Last panel: bar chart
    ax = axes_flat[-1]
    ordered = sorted(scores.items(), key=lambda x: -x[1])
    names = [n for n, _ in ordered]
    vals = [s for _, s in ordered]
    colors = ["#1f77b4" if n.startswith("PRISM")
              else "#999999" if n == "Random_chance"
              else "#ff7f0e" for n in names]
    bars = ax.barh(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(v + 0.005, b.get_y() + b.get_height()/2, f"{v:.3f}",
                va="center", fontsize=9)
    ax.set_xlabel("|Spearman ρ| vs canonical marker order")
    ax.set_title(f"Gene-trajectory recovery — {name}")
    ax.set_xlim(0, max(vals) * 1.15 if vals else 1.0)

    for j in range(n_methods, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Gene-trajectory recovery — {name}\n"
                 f"|Spearman ρ| between recovered gene order "
                 f"and {len(markers)}-marker canonical order",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


DATASET_SPECS = {
    "pancreas": dict(
        markers=MARKERS_PANCREAS,
        csv=WS / "data" / "pancreas" / "filtered_pancreas_cells_x_genes.csv",
        h5ad=WS / "data" / "pancreas" / "endocrinogenesis_day15.h5ad",
        root_cluster="Ductal",
        lineage_col="clusters_coarse",
        lineage_map={"Ductal": 0, "Ngn3 low EP": 1, "Ngn3 high EP": 2,
                     "Pre-endocrine": 3, "Endocrine": 4},
        prism_layouts=[("seed0", "K5", 5), ("K8/seed0", "K8", 8)],
    ),
    "gastrulation": dict(
        markers=MARKERS_GASTRULATION,
        csv=WS / "data" / "gastrulation" / "filtered_gastrulation_cells_x_genes.csv",
        h5ad=None,
        root_cluster="E6.5",
        lineage_col="stage",
        lineage_map={"E6.5": 0, "E6.75": 1, "E7.0": 2, "E7.25": 3,
                     "E7.5": 4, "E7.75": 5, "E8.0": 6, "E8.25": 7, "E8.5": 8},
        prism_layouts=[("seed0", "K5", 5), ("K9/seed0", "K9", 9)],
    ),
    "gastrulation_erythroid": dict(
        markers=MARKERS_GASTRULATION_ERYTHROID,
        csv=WS / "data" / "gastrulation_erythroid" / "filtered_gastrulation_erythroid_cells_x_genes.csv",
        h5ad=None,
        root_cluster="E7.0",
        lineage_col="stage",
        lineage_map={"E7.0": 0, "E7.25": 1, "E7.5": 2, "E7.75": 3,
                     "E8.0": 4, "E8.25": 5, "E8.5": 6},
        prism_layouts=[("seed0", "K5", 5), ("K7/seed0", "K7", 7)],
    ),
    "hemogenic_endothelium": dict(
        markers=MARKERS_HEMOGENIC,
        csv=WS / "data" / "hemogenic_endothelium" / "filtered_hemogenic_endothelium_cells_x_genes.csv",
        h5ad=None,
        root_cluster="endothelial cell",
        lineage_col="cell_type",
        lineage_map={"endothelial cell": 0, "endocardium cell progenitor": 1,
                     "hematopoietic precursor cell": 2},
        prism_layouts=[("seed0", "K5", 5), ("K4/seed0", "K4", 4)],
    ),
    # --- datasets with a curated lineage annotation (full harness + DPT) ---
    "bonemarrow": dict(
        markers=MARKERS_BONEMARROW,
        csv=WS / "data" / "BoneMarrow" / "filtered_bonemarrow_cells_x_genes.csv",
        h5ad=WS / "data" / "BoneMarrow" / "human_cd34_bone_marrow.h5ad",
        root_cluster="HSC_1",
        lineage_col="clusters",
        lineage_map={"HSC_1": 0, "HSC_2": 0, "Precursors": 1, "Ery_1": 2, "Ery_2": 3},
        prism_layouts=[("seed0", "K5", 5), ("K10/seed0", "K10", 10)],
    ),
    "gastrulation_e75": dict(
        markers=MARKERS_GASTRULATION_E75,
        csv=WS / "data" / "gastrulation_e75" / "filtered_gastrulation_e75_cells_x_genes.csv",
        h5ad=None,
        root_cluster="Epiblast",
        lineage_col="celltype",
        lineage_map={ct: 0 for ct in E75_CELLTYPES},
        prism_layouts=[("seed0", "K5", 5)],
    ),
    # --- datasets without a curated lineage annotation (PRISM Step (ii) +
    #     expression-magnitude only; no DPT reference baseline) ---
    "paul15": dict(
        markers=MARKERS_PAUL15,
        csv=WS / "data" / "paul15" / "filtered_paul15_cells_x_genes.csv",
        h5ad=None,
        root_cluster=None,
        lineage_col=None,
        lineage_map=None,
        prism_layouts=[("seed0", "K5", 5)],
    ),
    "dentategyrus": dict(
        markers=MARKERS_DENTATEGYRUS,
        csv=WS / "data" / "dentategyrus" / "filtered_dentategyrus_cells_x_genes.csv",
        h5ad=None,
        root_cluster=None,
        lineage_col=None,
        lineage_map=None,
        prism_layouts=[("seed0", "K5", 5)],
    ),
    "endoderm_diff": dict(
        markers=MARKERS_ENDODERM_DIFF,
        csv=WS / "data" / "endoderm_diff" / "filtered_endoderm_diff_cells_x_genes.csv",
        h5ad=None,
        root_cluster=None,
        lineage_col=None,
        lineage_map=None,
        prism_layouts=[("seed0", "K5", 5)],
    ),
}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DATASET_SPECS.keys()))
    args = p.parse_args()

    summary = {}
    for ds in args.datasets:
        spec = DATASET_SPECS.get(ds)
        if spec is None:
            print(f"[{ds}] no spec -- SKIP")
            continue
        try:
            summary[ds] = run_dataset(ds, **spec)
        except Exception as e:
            print(f"[{ds}] FAILED: {e}")
            import traceback; traceback.print_exc()

    # Cross-dataset wide table
    if summary:
        all_methods = set()
        for s in summary.values():
            if s is None: continue
            all_methods.update(s.keys())
        rows = []
        for m in sorted(all_methods):
            row = {"method": m}
            ss = []
            for ds, s in summary.items():
                v = (s or {}).get(m, float("nan"))
                row[ds] = v
                if np.isfinite(v): ss.append(v)
            row["mean"] = float(np.mean(ss)) if ss else float("nan")
            rows.append(row)
        df = pd.DataFrame(rows).sort_values("mean", ascending=False)
        out = WS / "outputs" / "trajectory" / "gene_trajectory_scores.csv"
        df.to_csv(out, index=False)
        print(f"\n=== CROSS-DATASET ===")
        print(df.to_string(index=False))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
