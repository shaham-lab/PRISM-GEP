"""PRISM's OWN gene trajectory — native Step-(ii) gene ordering, NO GeneTrajectory.

This figure shows PRISM-GEP's *own* gene-ordering method (supplementary §D,
Step (ii): Hellinger distance on gene-topic distributions + a Gaussian-kernel
diffusion map, leading gene-diffusion coordinate EV2) applied to a clean
developmental lineage. There is NO GeneTrajectory machinery here — no
extract_gene_trajectory, no add_gene_bin_score. Every step below uses only
PRISM's GEPs (word_topic_counts.txt) and PRISM's cell representation
(doc_topics.txt -> PHATE, or the dataset's own UMAP when present).

Idiom (borrowed from the GeneTrajectory *figure style* only, computed 100% by
PRISM):
  1. Cell embedding coloured by developmental stage / lineage cell type.
  2. Identify the lineage PROGRAM: the GEP(s) whose top genes carry the lineage
     markers. Take that program's top genes; order them by PRISM Step-(ii)
     (early -> late). Axis orientation is fixed from the canonical markers
     themselves (progenitor markers early, terminal markers late).
  3. Gene-bin sweep: split the ordered program genes into 5 bins by Step-(ii)
     pseudo-order; per cell, bin score = proportion of that bin's genes
     expressed (>0) in the cell. Draw the 5 bin scores over the cell embedding.
  4. A single directional sweep arrow through the top-quartile-most-active
     centroid of each bin (plain centroids collapse because bin scores are high
     nearly everywhere on a purified lineage). Drawn identically on every panel.
     Honesty gate: the arrow is only drawn if the bins advance monotonically
     (Spearman(bin index, projection on sweep axis) >= 0.55 AND the bin1->bin5
     centroid span is >= 10% of the embedding diagonal); otherwise the panels
     are annotated "no clean sweep".
  5. Marker-recovery validation panel: |rho| between PRISM's recovered order of
     the canonical markers and their canonical biological order.

Gene-ordering methods (the 1-D ordering of a GEP's genes).  Every method still
operates ONLY on PRISM's gene-topic distributions (word_topic_counts.txt); this
is purely a "how do we lay the program's genes on one axis" choice.  Pick with
``--method``:

  hellinger_diffusion  Hellinger metric + Gaussian-kernel diffusion EV2  (paper's
                       Step-(ii); the DEFAULT).
  js_diffusion         Jensen-Shannon metric + diffusion EV2.
  euclidean_diffusion  Euclidean distance on the topic simplex + diffusion EV2.
  cosine_diffusion     Cosine distance + diffusion EV2.
  correlation_diffusion  1-Pearson distance + diffusion EV2.
  phate1               PHATE 1-D on the gene-topic distributions (precomputed
                       Hellinger metric fed to PHATE).
  dpt                  Diffusion pseudotime from an anchor gene (the earliest
                       canonical marker), on the Hellinger diffusion operator.

Every method reports the SAME honest marker-recovery |rho| and the SAME arrow
verdict, so they are directly comparable.  A ``--sweep-methods`` mode measures
all methods on a dataset, prints a ranked table, and writes it to the scratchpad.

Output (one PDF per dataset, using that dataset's chosen/best method):
  figures/prism_gene_trajectory_erythroid.pdf   (gastrulation_erythroid)
  figures/prism_gene_trajectory_pancreas.pdf
  figures/prism_gene_trajectory_<ds>.pdf        (gastrulation, gastrulation_e75,
                                                 bonemarrow, hemogenic_endothelium)

Regenerate:
  python scripts/viz_prism_native_trajectory.py --datasets gastrulation_erythroid
  python scripts/viz_prism_native_trajectory.py --datasets pancreas
  python scripts/viz_prism_native_trajectory.py   # all viable datasets, best method each
  python scripts/viz_prism_native_trajectory.py --datasets pancreas --method js_diffusion
  python scripts/viz_prism_native_trajectory.py --sweep-methods   # rank methods, no PDFs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

WS = Path(__file__).resolve().parent.parent
import sys as _sys; _sys.path.insert(0, str(Path(__file__).resolve().parent))
# release layout: paper/figures/, created on demand (scripts/paths.py)
from paths import figures_dir  # noqa: E402
sys.path.insert(0, str(WS))

# Read-only imports from shared modules (NOT edited).
from bio.evaluate_supp import gene_ordering_supp          # PRISM Step-(ii)  # noqa: E402
from scripts.viz_cell_embed_common import load_cell_embedding  # noqa: E402

K5 = 5
N_BINS = 5

# All gene-ordering methods this script supports (see module docstring).
METHODS = [
    "hellinger_diffusion",   # paper's Step-(ii)  = DEFAULT
    "js_diffusion",
    "euclidean_diffusion",
    "cosine_diffusion",
    "correlation_diffusion",
    "phate1",
    "dpt",
]
DEFAULT_METHOD = "hellinger_diffusion"

# Human-readable label for figure titles/captions per method.
METHOD_LABEL = {
    "hellinger_diffusion": "Hellinger + diffusion (paper Step-(ii))",
    "js_diffusion": "Jensen-Shannon + diffusion",
    "euclidean_diffusion": "Euclidean + diffusion",
    "cosine_diffusion": "cosine + diffusion",
    "correlation_diffusion": "correlation + diffusion",
    "phate1": "PHATE 1-D (on Hellinger metric)",
    "dpt": "diffusion pseudotime (anchored)",
}


#   lineage_topics : the GEP indices (0-based, K=5) whose union of top genes
#                    forms the lineage program.
#   markers        : canonical (gene, biological_rank) order, early -> late.
#   early_markers  : genes that anchor the EARLY end (used to orient EV2).
#   stage_col_csv  : which per-cell label file/column to colour the embedding by
#                    and the biological order of those categories (early->late).
DATASET_SPECS = {
    "gastrulation_erythroid": dict(
        title="Gastrulation erythroid",
        program_label="erythroid maturation",
        lineage_topics=[0, 1, 2],   # globin / mid-erythroid / progenitor GEPs
        n_program_genes=48,
        # Purified erythroid population: nearly every gene (incl. globins) is
        # detected in nearly every cell, so the raw >0 bin-score saturates on
        # housekeeping genes. Standard trajectory-gene dynamic-range filter:
        # keep only program genes detected in < this fraction of cells before
        # the bin-score sweep. Ordering/marker-recovery use the FULL program.
        bin_dynamic_range_max=0.92,
        markers=[
            ("T", 0), ("Tal1", 1), ("Lmo2", 1), ("Gata1", 2),
            ("Klf1", 2), ("Hbb-bh1", 3), ("Hba-x", 3), ("Hbb-y", 3),
        ],
        early_markers=["T", "Tal1", "Lmo2"],
        late_markers=["Hbb-bh1", "Hba-x", "Hbb-y"],
        # colour by developmental stage (obs_metadata 'stage' == cell_type_labels)
        stage_source=("obs_metadata.csv", "celltype"),
        stage_order=["Blood progenitors 1", "Blood progenitors 2",
                     "Erythroid1", "Erythroid2", "Erythroid3"],
        stage_short={"Blood progenitors 1": "BP1", "Blood progenitors 2": "BP2",
                     "Erythroid1": "Ery1", "Erythroid2": "Ery2",
                     "Erythroid3": "Ery3"},
    ),
    "pancreas": dict(
        title="Pancreas endocrinogenesis",
        program_label="endocrine differentiation",
        lineage_topics=[3, 4],      # mature-endocrine GEPs (Ins/Gcg/Pyy/Ghrl)
        n_program_genes=48,
        # Pancreas spans ductal->endocrine with genuine ON/OFF gene dynamics,
        # so the raw >0 bin-score already sweeps; a lenient filter drops only
        # fully-ubiquitous (100%) genes, leaving the biology intact.
        bin_dynamic_range_max=0.995,
        markers=[
            ("Sox9", 0), ("Hes1", 0), ("Spp1", 0), ("Krt8", 0), ("Krt18", 0),
            ("Neurog3", 1), ("Neurod1", 2), ("Pax6", 2), ("Isl1", 2),
            ("Ins1", 3), ("Ins2", 3), ("Gcg", 3), ("Sst", 3),
            ("Ghrl", 3), ("Pyy", 3),
        ],
        early_markers=["Sox9", "Hes1", "Spp1"],
        late_markers=["Ins1", "Ins2", "Gcg"],
        stage_source=("cell_type_labels.csv", "cell_type"),
        stage_order=["Ductal", "Ngn3 low EP", "Ngn3 high EP", "Pre-endocrine",
                     "Beta", "Alpha", "Delta", "Epsilon"],
        stage_short={"Ductal": "Duct", "Ngn3 low EP": "Ngn3lo",
                     "Ngn3 high EP": "Ngn3hi", "Pre-endocrine": "PreEndo",
                     "Beta": "Beta", "Alpha": "Alpha", "Delta": "Delta",
                     "Epsilon": "Eps"},
    ),
    # Four MORE datasets with PRISM outputs on disk (outputs/<ds>/seed0/).
    # Markers + canonical ranks taken from
    #   outputs/trajectory/<ds>/gene_trajectory_<ds>_orders.csv (the same marker
    #   sets used by the gene-trajectory benchmark).  lineage_topics = GEPs whose gene-topic argmax
    #   carries the program's markers (verified against word_topic_counts.txt).
    "gastrulation": dict(
        title="Gastrulation (blood emergence)",
        program_label="epiblast → mesoderm → erythroid",
        # T1 = epiblast/PS TFs (Pou5f1/Nanog/Eomes/T/Mixl1/Mesp*);
        # T0 = endoderm (Foxa2/Sox17); T4 = erythroid globins (Hba-x/Hbb-y).
        lineage_topics=[1, 0, 4],
        n_program_genes=48,
        bin_dynamic_range_max=0.92,
        markers=[
            ("Pou5f1", 0), ("Nanog", 0), ("Eomes", 1), ("T", 1), ("Mixl1", 1),
            ("Mesp1", 2), ("Mesp2", 2), ("Foxa2", 3), ("Sox17", 3),
            ("Hba-x", 3), ("Hbb-y", 3),
        ],
        early_markers=["Pou5f1", "Nanog"],
        late_markers=["Hba-x", "Hbb-y"],
        stage_source=("cell_type_labels.csv", "cell_type"),   # developmental stage E6.5..E8.5
        stage_order=["E6.5", "E6.75", "E7.0", "E7.25", "E7.5", "E7.75",
                     "E8.0", "E8.25", "E8.5"],
        stage_short={s: s.replace("E", "") for s in
                     ["E6.5", "E6.75", "E7.0", "E7.25", "E7.5", "E7.75",
                      "E8.0", "E8.25", "E8.5"]},
    ),
    "gastrulation_e75": dict(
        title="Gastrulation E7.5 (single snapshot)",
        program_label="epiblast → primitive-streak → mesendoderm",
        # T2 = epiblast (Pou5f1/Nanog/Pax6); T3 = PS/mesoderm (T/Mixl1/Mesp1);
        # T4 = endoderm (Foxa2/Cer1/Sox17); T1 = cardiac (Hand1/Tbx5).
        lineage_topics=[2, 3, 4, 1],
        n_program_genes=48,
        bin_dynamic_range_max=0.92,
        markers=[
            ("Pou5f1", 0), ("Nanog", 0), ("T", 1), ("Mixl1", 1), ("Fgf8", 1),
            ("Mesp1", 2), ("Snai1", 2), ("Foxa2", 2), ("Cer1", 2),
            ("Gata4", 3), ("Hand1", 3), ("Tbx5", 3), ("Sox17", 3), ("Pax6", 3),
        ],
        early_markers=["Pou5f1", "Nanog"],
        late_markers=["Gata4", "Hand1", "Tbx5"],
        stage_source=("cell_type_labels.csv", "cell_type"),
        # A single-timepoint snapshot: cell types are a lineage FAN, not a line;
        # ordered epiblast -> streak -> mesoderm/endoderm for the colour legend.
        stage_order=["Epiblast", "Caudal epiblast", "Primitive Streak",
                     "Anterior Primitive Streak", "Nascent mesoderm",
                     "Mixed mesoderm", "Intermediate mesoderm",
                     "Paraxial mesoderm", "Somitic mesoderm", "Pharyngeal mesoderm",
                     "ExE mesoderm", "Caudal Mesoderm", "Mesenchyme",
                     "Def. endoderm", "Visceral endoderm", "Gut",
                     "Blood progenitors 1", "Blood progenitors 2",
                     "Haematoendothelial progenitors", "Endothelium",
                     "Rostral neurectoderm", "Caudal neurectoderm",
                     "Surface ectoderm", "Notochord", "Allantois", "PGC",
                     "Erythroid1"],
        stage_short={},   # names are long; legend uses full names, ncol handles it
    ),
    "bonemarrow": dict(
        title="Bone marrow (HSC → erythroid)",
        program_label="HSC → erythroid maturation",
        # T2 = HSC/progenitor (HLF/AVP/MEIS1/CD34); T0 = erythroid
        # (GATA1/KLF1/globins/AHSP).
        lineage_topics=[2, 0],
        n_program_genes=48,
        bin_dynamic_range_max=0.92,
        markers=[
            ("CD34", 0), ("HLF", 0), ("AVP", 0), ("MEIS1", 0), ("MLLT3", 0),
            ("GATA2", 1), ("KIT", 1), ("TAL1", 1),
            ("GATA1", 2), ("KLF1", 2), ("NFE2", 2), ("TFRC", 2),
            ("GYPA", 3), ("ALAS2", 3), ("EPOR", 3), ("BLVRB", 3),
            ("HBB", 4), ("HBA1", 4), ("HBA2", 4), ("AHSP", 4),
        ],
        early_markers=["CD34", "HLF", "AVP", "MEIS1"],
        late_markers=["HBB", "HBA1", "HBA2", "AHSP"],
        stage_source=("cell_type_labels.csv", "cell_type"),
        stage_order=["HSC_1", "HSC_2", "Precursors", "CLP", "Mega",
                     "Ery_1", "Ery_2", "Mono_1", "Mono_2", "DCs"],
        stage_short={"HSC_1": "HSC1", "HSC_2": "HSC2", "Precursors": "Prec",
                     "Ery_1": "Ery1", "Ery_2": "Ery2", "Mono_1": "Mono1",
                     "Mono_2": "Mono2"},
    ),
    "hemogenic_endothelium": dict(
        title="Hemogenic endothelium (EHT)",
        program_label="endothelial → hematopoietic transition",
        # T2 = endothelial (Cldn5/Emcn); T3 = hematopoietic TFs
        # (Runx1/Gata2/Myb/Spi1) + erythroid (Gypa/Klf1); T4 = early (Sox17).
        lineage_topics=[2, 3, 4],
        n_program_genes=48,
        bin_dynamic_range_max=0.92,
        markers=[
            ("Sox17", 0), ("Cldn5", 0), ("Emcn", 0), ("Egfl7", 0),
            ("Runx1", 1), ("Gata2", 1), ("Gfi1b", 1),
            ("Myb", 2), ("Spi1", 2), ("Itga2b", 2), ("Cd44", 2),
            ("Lmo2", 2), ("Tal1", 2),
            ("Gypa", 3), ("Klf1", 3),
        ],
        early_markers=["Sox17", "Cldn5", "Emcn", "Egfl7"],
        late_markers=["Gypa", "Klf1"],
        stage_source=("cell_type_labels.csv", "cell_type"),
        stage_order=["endothelial cell", "endocardium cell progenitor",
                     "hematopoietic precursor cell", "cardiac muscle myoblast"],
        stage_short={"endothelial cell": "Endo",
                     "endocardium cell progenitor": "Endocard",
                     "hematopoietic precursor cell": "HPC",
                     "cardiac muscle myoblast": "CardMyo"},
    ),
    # GeneTrajectory comparison datasets (PRISM-native, trained 2026-07-03).
    # lineage_topics + markers verified against outputs/<ds>/seed0/
    # word_topic_counts.txt (per-gene argmax topic), NOT guessed:
    #   gt_myeloid: T1=classical CD14+ mono (S100A8/9/12,VCAN,FCN1);
    #               T4=non-classical CD16+ (FCGR3A=1.00,CDKN1C,C1QA,MS4A7,LST1);
    #               T0=DC branch (FCER1A/CD1C/CLEC10A) -> EXCLUDED from lineage.
    #   gt_dermal:  T2=dermis/fibroblast (Col1a1/Col3a1/Dcn/Lum,+Corin/Alx4);
    #               T4=dermal condensate/early DP (Sox2=1.00,Ptch1,Trps1,Dkk1,
    #               Bmp4); T3=cell-cycle (Mki67/Top2a) -> EXCLUDED.
    "gt_myeloid": dict(
        title="Human myeloid (monocyte maturation)",
        program_label="classical → non-classical monocyte",
        # T3 dropped (bio review 2026-07-03): T3 is a stress/ribosomal junk topic
        # (FTH1/NEAT1/JUNB/JUN/FOSB/TXNIP), not a monocyte-maturation intermediate.
        lineage_topics=[1, 4],
        n_program_genes=48,
        bin_dynamic_range_max=0.92,
        markers=[
            # CD14 -> rank 0 (defining classical marker); LYZ dropped (pan-myeloid, non-directional).
            ("S100A8", 0), ("S100A9", 0), ("S100A12", 0), ("VCAN", 0), ("FCN1", 0), ("CD14", 0),
            ("MNDA", 1),
            ("MS4A7", 2), ("LST1", 2), ("AIF1", 2), ("COTL1", 2),
            ("FCGR3A", 3), ("CDKN1C", 3), ("C1QA", 3), ("RHOC", 3),
        ],
        early_markers=["S100A8", "S100A9", "S100A12", "VCAN"],
        late_markers=["FCGR3A", "CDKN1C", "C1QA", "RHOC"],
        stage_source=("cell_type_labels.csv", "cell_type"),
        # cluster IDs mapped to state empirically (cluster x dominant PRISM topic):
        # "1"->classical/intermediate mono, "10"->non-classical CD16, "11"->DC.
        stage_order=["1", "10", "11"],
        stage_short={"1": "Classical mono", "10": "Non-classical CD16",
                     "11": "DC (branch)"},
    ),
    "gt_dermal": dict(
        title="Mouse dermal condensate (skin)",
        program_label="fibroblast → dermal condensate",
        # T1 (Mfap5/Ptn/Dlk1) = a redundant fibroblast sub-state kept as an early rung;
        # T3 (cell-cycle) excluded.
        lineage_topics=[2, 1, 4],
        n_program_genes=48,
        bin_dynamic_range_max=0.92,
        markers=[
            # Corrected ranks (bio review 2026-07-03): Vim -> early (pan-fibroblast, not late);
            # Alx4/Corin -> rank 3 (specific dermal-papilla/condensate inducers, ~as late as Sox2).
            ("Col1a1", 0), ("Col3a1", 0), ("Col1a2", 0), ("Dcn", 0), ("Lum", 0), ("Vim", 0),
            ("Mfap5", 1), ("Ptn", 1), ("Dlk1", 1),
            ("Crabp1", 2),
            ("Alx4", 3), ("Corin", 3), ("Bmp4", 3),
            ("Sox2", 4), ("Ptch1", 4), ("Trps1", 4), ("Dkk1", 4), ("Sox18", 4),
        ],
        early_markers=["Col1a1", "Col3a1", "Col1a2", "Dcn", "Lum"],
        late_markers=["Sox2", "Ptch1", "Trps1", "Dkk1", "Sox18"],
        stage_source=("cell_type_labels.csv", "cell_type"),
        stage_order=["UD", "LD", "DC"],
        stage_short={"UD": "Upper dermis", "LD": "Lower dermis",
                     "DC": "Dermal condensate"},
    ),
}


def load_gene_topic_dist(ds: str, K: int = K5) -> dict[str, np.ndarray]:
    """gene(lower) -> row-normalized (K,) topic distribution from
    word_topic_counts.txt."""
    f = WS / "outputs" / ds / "seed0" / "word_topic_counts.txt"
    gd: dict[str, np.ndarray] = {}
    with open(f) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 2:
                continue
            g = p[1].lower()
            c = np.zeros(K, dtype=float)
            for tok in p[2:]:
                if ":" not in tok:
                    continue
                t, cc = tok.split(":")
                ti = int(t)
                if ti < K:
                    c[ti] = float(cc)
            s = c.sum()
            if s > 0:
                gd[g] = c / s
    return gd


def load_topic_keys(ds: str) -> dict[int, list[str]]:
    """topic id -> list of top gene names (as printed, original case)."""
    out: dict[int, list[str]] = {}
    with open(WS / "outputs" / ds / "seed0" / "topic_keys.txt") as fh:
        for line in fh:
            p = line.split()
            if len(p) < 4:
                continue
            out[int(p[0])] = p[3:]
    return out


def load_stage_labels(ds: str, spec: dict) -> pd.Series:
    fname, col = spec["stage_source"]
    df = pd.read_csv(WS / "data" / ds / fname)
    df["cell_id"] = df["cell_id"].astype(str)
    # Coerce labels to str: some datasets (e.g. gt_myeloid) store cell_type as
    # numeric cluster IDs that read_csv parses as int, which then fails the
    # string match against spec["stage_order"]. Idempotent for string labels.
    return df.set_index("cell_id")[col].astype(str)


# Every "*_diffusion" method reuses the paper's diffusion-EV2 machinery
# (Gaussian kernel with median-distance bandwidth, P = D^-1 K, EV2) but swaps
# the DISTANCE between gene-topic distributions.  hellinger_diffusion is exactly
# gene_ordering_supp() (the paper's Step-(ii)); we re-implement the generic form
# here so the distance is pluggable, and delegate to the imported function for
# the Hellinger case so the DEFAULT path is byte-identical to the shared module.

def _js_distance_matrix(P: np.ndarray) -> np.ndarray:
    """Pairwise Jensen-Shannon DISTANCE (sqrt of JS divergence, base-2) between
    rows of the row-stochastic matrix P."""
    n = P.shape[0]
    eps = 1e-12
    Pe = np.clip(P, eps, None)
    logP = np.log2(Pe)
    D = np.zeros((n, n))
    for i in range(n):
        M = 0.5 * (P[i][None, :] + P)              # (n, K) mixtures
        Me = np.clip(M, eps, None)
        logM = np.log2(Me)
        kl_pm = (Pe[i][None, :] * (logP[i][None, :] - logM)).sum(axis=1)
        kl_qm = (Pe * (logP - logM)).sum(axis=1)
        jsd = 0.5 * kl_pm + 0.5 * kl_qm
        D[i] = np.sqrt(np.clip(jsd, 0, None))
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D


def _distance_matrix(P: np.ndarray, metric: str) -> np.ndarray:
    """Pairwise distance between gene-topic distributions (rows of P)."""
    if metric == "hellinger":
        sqrt_p = np.sqrt(np.maximum(P, 0))
        diff = sqrt_p[:, None, :] - sqrt_p[None, :, :]
        return np.sqrt(np.maximum(0.5 * (diff ** 2).sum(axis=2), 0))
    if metric == "js":
        return _js_distance_matrix(P)
    if metric in ("euclidean", "cosine", "correlation"):
        from scipy.spatial.distance import squareform, pdist
        return squareform(pdist(P, metric=metric))
    raise ValueError(f"unknown metric {metric}")


def _diffusion_ev2(D: np.ndarray) -> np.ndarray:
    """Gaussian kernel (median-distance bandwidth) + P=D^-1 K + EV2. Returns the
    per-gene 1-D coordinate (row order preserved)."""
    pos = D[D > 0]
    sigma = np.median(pos) if pos.size else 1.0
    if sigma == 0:
        sigma = 1.0
    K = np.exp(-(D ** 2) / (2 * sigma ** 2))
    np.fill_diagonal(K, 0.0)
    rs = K.sum(axis=1)
    rs = np.where(rs > 0, rs, 1.0)
    Pmat = K / rs[:, None]
    evals, evecs = np.linalg.eig(Pmat)
    evals, evecs = evals.real, evecs.real
    idx = np.argsort(evals)[::-1]
    return evecs[:, idx][:, 1]   # EV2


def _order_by_coord(present: list[str], coord: np.ndarray):
    order = np.argsort(coord)
    return [present[i] for i in order], coord[order]


def gene_ordering_method(top_genes: list[str],
                         gd: dict[str, np.ndarray],
                         method: str = DEFAULT_METHOD,
                         early_markers: list[str] | None = None):
    """Order a program's genes on one axis by ``method``.  Returns
    (ordered_genes, pseudotime) exactly like gene_ordering_supp, in ascending
    pseudotime.  Operates ONLY on PRISM gene-topic distributions."""
    present = [g for g in top_genes if g in gd]
    if len(present) < 3:
        raise ValueError(f"need >= 3 genes with topic distributions, got {len(present)}")

    if method == "hellinger_diffusion":
        # delegate to the shared module so the DEFAULT is byte-identical
        return gene_ordering_supp(present, gd)

    P = np.array([gd[g] for g in present], dtype=float)
    P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12)

    if method.endswith("_diffusion"):
        metric = method[: -len("_diffusion")]
        D = _distance_matrix(P, metric)
        coord = _diffusion_ev2(D)
        return _order_by_coord(present, coord)

    if method == "phate1":
        import phate
        import warnings
        D = _distance_matrix(P, "hellinger")   # PHATE on the paper's metric
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            op = phate.PHATE(n_components=1, knn_dist="precomputed",
                             knn=min(5, len(present) - 1), verbose=False,
                             random_state=0)
            Y = op.fit_transform(D).ravel()
        return _order_by_coord(present, Y)

    if method == "dpt":
        # diffusion pseudotime from an anchor gene: distance in the diffusion
        # embedding from the earliest canonical marker present in the program.
        D = _distance_matrix(P, "hellinger")
        pos = D[D > 0]
        sigma = np.median(pos) if pos.size else 1.0
        K = np.exp(-(D ** 2) / (2 * sigma ** 2))
        np.fill_diagonal(K, 0.0)
        rs = K.sum(axis=1); rs = np.where(rs > 0, rs, 1.0)
        Pmat = K / rs[:, None]
        evals, evecs = np.linalg.eig(Pmat)
        evals, evecs = evals.real, evecs.real
        idx = np.argsort(evals)[::-1]
        evals, evecs = evals[idx], evecs[:, idx]
        # multi-scale diffusion coords (Haghverdi et al.): phi_k * eval_k/(1-eval_k)
        ncomp = min(len(present) - 1, 10)
        w = np.zeros(ncomp)
        for k in range(1, ncomp + 1):
            lam = np.clip(evals[k], -0.999999, 0.999999)
            w[k - 1] = lam / (1.0 - lam)
        emb = evecs[:, 1:ncomp + 1] * w[None, :]
        # anchor = earliest canonical early marker present, else global argmin EV2
        anchor = None
        if early_markers:
            for g in early_markers:
                if g.lower() in present:
                    anchor = present.index(g.lower()); break
        if anchor is None:
            anchor = int(np.argmin(evecs[:, 1]))
        dpt = np.linalg.norm(emb - emb[anchor][None, :], axis=1)
        return _order_by_coord(present, dpt)

    raise ValueError(f"unknown method {method}")


def build_program_genes(ds: str, spec: dict, gd: dict[str, np.ndarray]) -> list[str]:
    """Union of top genes across the lineage GEP(s), plus canonical markers,
    truncated to n_program_genes, all lowercased and present in gd."""
    keys = load_topic_keys(ds)
    prog: list[str] = []
    seen: set[str] = set()
    # interleave the topics' top genes so the program isn't dominated by one GEP
    per_topic = [keys[t] for t in spec["lineage_topics"] if t in keys]
    maxlen = max((len(x) for x in per_topic), default=0)
    for i in range(maxlen):
        for genes in per_topic:
            if i < len(genes):
                gl = genes[i].lower()
                if gl not in seen and gl in gd:
                    seen.add(gl)
                    prog.append(gl)
    # ensure canonical markers are included
    for g, _ in spec["markers"]:
        gl = g.lower()
        if gl not in seen and gl in gd:
            seen.add(gl)
            prog.append(gl)
    return prog[: spec["n_program_genes"]]


def prism_step_ii(program: list[str], gd: dict[str, np.ndarray],
                  method: str = DEFAULT_METHOD,
                  early_markers: list[str] | None = None):
    """Returns dict gene(lower) -> pseudotime, oriented so that mean(early
    markers) < mean(late markers) is NOT enforced here — caller orients.
    ``method`` selects the gene-ordering method (see gene_ordering_method)."""
    ordered, pt = gene_ordering_method(program, gd, method=method,
                                       early_markers=early_markers)
    return dict(zip(ordered, pt))


def orient_pseudotime(pt_map: dict[str, float], spec: dict) -> dict[str, float]:
    """Flip EV2 sign (arbitrary) so canonical EARLY markers get low pseudotime
    and LATE markers high. Orientation is derived from the markers, not chosen
    by hand."""
    early = [pt_map[g.lower()] for g in spec["early_markers"] if g.lower() in pt_map]
    late = [pt_map[g.lower()] for g in spec["late_markers"] if g.lower() in pt_map]
    flip = False
    if early and late:
        flip = np.mean(early) > np.mean(late)
    if flip:
        pt_map = {g: -v for g, v in pt_map.items()}
    # rescale to [0,1] for readability
    vals = np.array(list(pt_map.values()))
    lo, hi = vals.min(), vals.max()
    rng = (hi - lo) if hi > lo else 1.0
    return {g: (v - lo) / rng for g, v in pt_map.items()}


def marker_recovery_rho(spec: dict, gd: dict[str, np.ndarray],
                        method: str = DEFAULT_METHOD) -> tuple[float, list, np.ndarray, np.ndarray]:
    """|Spearman| between PRISM's recovered order of the canonical markers and
    their canonical biological rank. Computed on the markers ALONE, exactly the
    gene-trajectory benchmark quantity.  ``method`` selects the gene-ordering method."""
    markers = [g for g, _ in spec["markers"]]
    ranks = np.array([r for _, r in spec["markers"]], dtype=float)
    present = [(g, r) for (g, r) in zip(markers, ranks) if g.lower() in gd]
    genes = [g for g, _ in present]
    ranks_p = np.array([r for _, r in present])
    ordered, pt = gene_ordering_method([g.lower() for g in genes], gd,
                                       method=method,
                                       early_markers=spec.get("early_markers"))
    name_to_pt = dict(zip(ordered, pt))
    recovered = np.array([name_to_pt.get(g.lower(), np.nan) for g in genes])
    valid = np.isfinite(recovered)
    rho, _ = spearmanr(recovered[valid], ranks_p[valid])
    return abs(float(rho)), genes, recovered, ranks_p


def gene_bin_scores(program_ordered: list[str], pt: np.ndarray,
                    expr: pd.DataFrame, n_bins: int = N_BINS) -> np.ndarray:
    """For each cell, bin_score[b] = fraction of bin-b program genes with
    expression > 0 in that cell. Genes split into equal-count bins along the
    Step-(ii) pseudo-order. Returns (n_cells, n_bins)."""
    cols_lower = {c.lower(): c for c in expr.columns}
    order = np.argsort(pt)
    genes_sorted = [program_ordered[i] for i in order]
    n = len(genes_sorted)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    scores = np.zeros((expr.shape[0], n_bins))
    for b in range(n_bins):
        bin_genes = genes_sorted[edges[b]:edges[b + 1]]
        cols = [cols_lower[g] for g in bin_genes if g in cols_lower]
        if not cols:
            continue
        sub = expr[cols].values
        scores[:, b] = (sub > 0).mean(axis=1)
    return scores


def sweep_centroids(coords: np.ndarray, scores: np.ndarray, top_q: float = 0.75):
    """Centroid of each bin's TOP-QUARTILE most-active cells. Returns
    (n_bins, 2)."""
    n_bins = scores.shape[1]
    cents = np.full((n_bins, 2), np.nan)
    for b in range(n_bins):
        s = scores[:, b]
        thr = np.quantile(s, top_q)
        sel = s >= thr
        if sel.sum() >= 3:
            cents[b] = coords[sel].mean(axis=0)
    return cents


def arrow_verdict(cents: np.ndarray, coords: np.ndarray):
    """Honesty gate for the directional sweep.

    Monotonicity is judged on the principal sweep axis fit through the bin
    top-quartile centroids: require |Spearman(bin index, projection)| >= 0.55
    AND the bin1->bin5 span >= 10% of the embedding diagonal.

    Returns (ok, rho, span_frac, path) where ``path`` is the ordered
    (n_valid_bins, 2) polyline through the bin centroids (early -> late). The
    arrow is drawn as this polyline, so it follows the ACTUAL bin progression
    rather than an idealised straight line."""
    valid = np.isfinite(cents).all(axis=1)
    if valid.sum() < 3:
        return False, np.nan, np.nan, None
    C = cents[valid]
    bins = np.where(valid)[0]
    center = C.mean(axis=0)
    Cc = C - center
    _, _, Vt = np.linalg.svd(Cc, full_matrices=False)
    axis = Vt[0]
    proj = Cc @ axis
    rho, _ = spearmanr(bins, proj)
    if rho < 0:
        rho = -rho
    span = proj.max() - proj.min()
    diag = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))
    span_frac = span / diag if diag > 0 else 0.0
    ok = (abs(rho) >= 0.55) and (span_frac >= 0.10)
    # path = the bin centroids in bin order (early -> late)
    path = C  # already in ascending bin order because cents rows are bin order
    return ok, abs(float(rho)), float(span_frac), path


def _stage_palette(order):
    cmap = plt.get_cmap("viridis")
    return {s: cmap(i / max(len(order) - 1, 1)) for i, s in enumerate(order)}


def make_figure(ds: str, spec: dict, out_path: Path, method: str = DEFAULT_METHOD):
    print(f"\n========== {ds}  (method={method}) ==========")
    gd = load_gene_topic_dist(ds)
    print(f"  gene-topic distributions: {len(gd)} genes")

    # ---- cell embedding (PRISM-native or dataset UMAP) ----
    coords_df, expr, source = load_cell_embedding(ds)
    coords = coords_df[["x", "y"]].values
    print(f"  cell embedding source: {source}  ({coords.shape[0]} cells)")

    # ---- stage / cell-type colouring ----
    stage = load_stage_labels(ds, spec).reindex(coords_df.index.astype(str))
    order = [s for s in spec["stage_order"] if s in set(stage.dropna())]
    pal = _stage_palette(order)

    # ---- PRISM native gene ordering (Step ii) — FULL program ----
    program = build_program_genes(ds, spec, gd)
    pt_map = orient_pseudotime(
        prism_step_ii(program, gd, method=method,
                      early_markers=spec.get("early_markers")), spec)
    program = [g for g in program if g in pt_map]
    pt = np.array([pt_map[g] for g in program])
    print(f"  program: {len(program)} genes from GEPs {spec['lineage_topics']}")
    ordered_show = [g for g in sorted(program, key=lambda g: pt_map[g])]
    print(f"    early: {ordered_show[:6]}")
    print(f"    late : {ordered_show[-6:]}")

    # ---- dynamic-range filter for the bin-score sweep ----
    # Standard trajectory-gene practice: drop genes detected in ~all cells,
    # whose >0 indicator carries no cell-to-cell signal. Re-order the surviving
    # genes by PRISM Step-(ii) (their relative order is unchanged from the full
    # program). Ordering strip + marker recovery still use the FULL program.
    dr_max = spec.get("bin_dynamic_range_max", 1.01)
    cols_lower = {c.lower(): c for c in expr.columns}
    frac = {g: (expr[cols_lower[g]].values > 0).mean()
            for g in program if g in cols_lower}
    bin_program = [g for g in program if frac.get(g, 1.0) < dr_max]
    n_dropped = len(program) - len(bin_program)
    bin_pt = np.array([pt_map[g] for g in bin_program])
    print(f"  bin-sweep program: {len(bin_program)} genes "
          f"(dropped {n_dropped} ubiquitous, detected >= {dr_max:.0%} of cells)")

    # ---- gene-bin scores + sweep ----
    scores = gene_bin_scores(bin_program, bin_pt, expr)
    cents = sweep_centroids(coords, scores)
    ok, rho, span_frac, path = arrow_verdict(cents, coords)
    verdict = (f"clean sweep (rho={rho:.2f}, span={span_frac:.0%})" if ok
               else f"NO clean sweep (rho={rho:.2f}, span={span_frac:.0%})")
    print(f"  arrow verdict: {verdict}")

    # ---- marker recovery ----
    mrho, mgenes, mrec, mrank = marker_recovery_rho(spec, gd, method=method)
    print(f"  marker recovery |rho| = {mrho:.3f}  ({len(mgenes)} markers)")

    # -------------------------------------------------------------------
    # Layout: row 1 = [cell-type embedding | marker-recovery scatter];
    #         row 2 = 5 gene-bin panels (bin1..bin5), each with the same arrow.
    # -------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, N_BINS, height_ratios=[1.15, 1.0],
                          hspace=0.32, wspace=0.18)

    # (row1, col0-1 merged) cell embedding coloured by stage/cell type
    ax0 = fig.add_subplot(gs[0, 0:2])
    for s in order:
        m = (stage == s).values
        ax0.scatter(coords[m, 0], coords[m, 1], s=6, color=pal[s],
                    label=spec["stage_short"].get(s, s), linewidths=0)
    na = stage.isna().values
    if na.any():
        ax0.scatter(coords[na, 0], coords[na, 1], s=5, color="#dddddd",
                    linewidths=0, zorder=0)
    ax0.set_title(f"PRISM cell embedding — coloured by cell type\n"
                  f"({source})", fontsize=10)
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.legend(markerscale=2.2, fontsize=8, loc="best", framealpha=0.85,
               ncol=2)

    # (row1, col2-3 merged) marker recovery scatter — oriented early-low/late-high
    # (consistent across datasets) and de-collided so co-ranked markers don't overprint.
    axm = fig.add_subplot(gs[0, 2:4])
    mrec_o = np.array(mrec, dtype=float)
    v = np.isfinite(mrec_o)
    # |rho| is sign-invariant; force the plotted trend POSITIVE so direction never flips.
    if v.sum() >= 2 and spearmanr(mrank[v], mrec_o[v]).correlation < 0:
        mrec_o = -mrec_o
    # spread co-ranked markers along x so their labels stop stacking
    from collections import defaultdict as _dd
    _grp = _dd(list)
    for i, r in enumerate(mrank):
        _grp[int(r)].append(i)
    xj = mrank.astype(float).copy()
    for r, idxs in _grp.items():
        if len(idxs) > 1:
            order_in = sorted(idxs, key=lambda k: (mrec_o[k] if np.isfinite(mrec_o[k]) else 0.0))
            for off, k in zip(np.linspace(-0.30, 0.30, len(idxs)), order_in):
                xj[k] = r + off
    axm.scatter(xj, mrec_o, c="#1f77b4", s=52, zorder=3, edgecolor="white", linewidth=0.6)
    _texts = [axm.text(x, y, g, fontsize=6.5, zorder=4)
              for g, x, y in zip(mgenes, xj, mrec_o) if np.isfinite(y)]
    try:   # proper non-overlapping label placement with leader lines
        from adjustText import adjust_text
        adjust_text(_texts, ax=axm, arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.4),
                    expand_points=(1.4, 1.6), force_text=(0.4, 0.6))
    except Exception:
        pass
    if v.sum() >= 2:
        z = np.polyfit(mrank[v], mrec_o[v], 1)
        xs = np.linspace(mrank.min(), mrank.max(), 20)
        axm.plot(xs, np.polyval(z, xs), color="#aaaaaa", ls="--", lw=1.2, zorder=1)
    axm.set_xlabel("canonical marker rank (early → late)")
    axm.set_ylabel("PRISM gene pseudotime (early → late)")
    axm.set_title(f"Marker recovery — PRISM ordered the known markers\n"
                  f"|ρ| = {mrho:.3f}  (chance ≈ 0.31)", fontsize=10)
    axm.grid(alpha=0.3)

    # (row1, col4) program-gene ordering strip (early->late colour bar of genes)
    axg = fig.add_subplot(gs[0, 4])
    o = np.argsort(pt)
    genes_sorted = [program[i] for i in o]
    pt_sorted = pt[o]
    y = np.arange(len(genes_sorted))
    axg.scatter(pt_sorted, y, c=pt_sorted, cmap="plasma", s=18)
    # label a subset (markers + endpoints) to keep it readable
    marker_lower = {g.lower() for g, _ in spec["markers"]}
    for yi, (g, x) in enumerate(zip(genes_sorted, pt_sorted)):
        if g in marker_lower or yi < 2 or yi > len(genes_sorted) - 3:
            axg.annotate(g, (x, yi), fontsize=6.5, xytext=(3, 0),
                         textcoords="offset points", va="center")
    axg.set_title(f"Program genes ordered by PRISM\n"
                  f"{METHOD_LABEL.get(method, method)}\n"
                  f"(n={len(program)})", fontsize=8.5)
    axg.set_xlabel("pseudotime (early→late)")
    axg.set_yticks([])
    axg.set_xlim(-0.05, 1.35)

    # (row2) 5 gene-bin score panels with the identical sweep arrow.
    # Bins are cut on the dynamic-range-filtered, Step-(ii)-ordered program.
    bo = np.argsort(bin_pt)
    bin_genes_sorted = [bin_program[i] for i in bo]
    edges = np.linspace(0, len(bin_program), N_BINS + 1).astype(int)
    for b in range(N_BINS):
        ax = fig.add_subplot(gs[1, b])
        s = scores[:, b]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=s, s=6, cmap="magma",
                        vmin=0, vmax=np.quantile(scores, 0.99), linewidths=0)
        # identical sweep drawn on every panel: a polyline through the bin
        # centroids (early -> late), so it follows the ACTUAL bin progression.
        if path is not None and len(path) >= 2:
            col = "#00e5ff" if ok else "#888888"
            # draw the connecting segments up to the last, then an arrowhead
            ax.plot(path[:, 0], path[:, 1], color=col,
                    lw=2.4 if ok else 1.4, ls="-" if ok else ":",
                    zorder=5, solid_capstyle="round")
            arr = FancyArrowPatch(tuple(path[-2]), tuple(path[-1]),
                                  arrowstyle="-|>", mutation_scale=22,
                                  lw=2.4 if ok else 1.4, color=col, zorder=6)
            ax.add_patch(arr)
            # small dots at each bin centroid (waypoints)
            ax.scatter(path[:, 0], path[:, 1], s=18, color=col,
                       edgecolors="k", linewidths=0.4, zorder=7)
        bin_genes = bin_genes_sorted[edges[b]:edges[b + 1]]
        head = ", ".join(bin_genes[:3])
        ax.set_title(f"bin {b+1} score\n{head}…", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    return dict(source=source, method=method, n_program=len(program),
                n_bin_program=len(bin_program), n_dropped=n_dropped,
                marker_rho=mrho, arrow_ok=ok, sweep_rho=rho,
                span_frac=span_frac, lineage_topics=spec["lineage_topics"])


def measure_dataset_method(ds: str, spec: dict, method: str,
                           _cache: dict | None = None) -> dict:
    """Measure marker |rho|, sweep Spearman, span, and the arrow verdict for one
    (dataset, method) WITHOUT drawing a figure.  ``_cache`` (optional) memoizes
    the per-dataset embedding/expression/gene-topic loads across methods."""
    _cache = _cache if _cache is not None else {}
    if "gd" not in _cache:
        _cache["gd"] = load_gene_topic_dist(ds)
        coords_df, expr, source = load_cell_embedding(ds)
        _cache["coords"] = coords_df[["x", "y"]].values
        _cache["expr"] = expr
        _cache["source"] = source
    gd = _cache["gd"]; coords = _cache["coords"]; expr = _cache["expr"]
    source = _cache["source"]

    # marker recovery on the markers alone
    try:
        mrho, mgenes, _, _ = marker_recovery_rho(spec, gd, method=method)
    except Exception as e:  # noqa: BLE001
        return dict(dataset=ds, method=method, ok=False, error=str(e),
                    marker_rho=np.nan, sweep_rho=np.nan, span_frac=np.nan,
                    arrow_ok=False, source=source, n_program=0)

    # program ordering + bin sweep + arrow
    program = build_program_genes(ds, spec, gd)
    pt_map = orient_pseudotime(
        prism_step_ii(program, gd, method=method,
                      early_markers=spec.get("early_markers")), spec)
    program = [g for g in program if g in pt_map]
    dr_max = spec.get("bin_dynamic_range_max", 1.01)
    cols_lower = {c.lower(): c for c in expr.columns}
    frac = {g: (expr[cols_lower[g]].values > 0).mean()
            for g in program if g in cols_lower}
    bin_program = [g for g in program if frac.get(g, 1.0) < dr_max]
    bin_pt = np.array([pt_map[g] for g in bin_program])
    scores = gene_bin_scores(bin_program, bin_pt, expr)
    cents = sweep_centroids(coords, scores)
    ok, rho, span_frac, _ = arrow_verdict(cents, coords)
    return dict(dataset=ds, method=method, marker_rho=float(mrho),
                sweep_rho=float(rho), span_frac=float(span_frac),
                arrow_ok=bool(ok), n_program=len(program),
                n_markers=len(mgenes), source=source, error="")


def viable_datasets() -> list[str]:
    """Datasets that have BOTH a spec AND PRISM outputs on disk
    (outputs/<ds>/seed0/{word_topic_counts,topic_keys,doc_topics}.txt).  Missing
    ones (e.g. gt_myeloid/gt_dermal/dentategyrus — no PRISM run) are skipped."""
    out = []
    for ds in DATASET_SPECS:
        seed0 = WS / "outputs" / ds / "seed0"
        need = ["word_topic_counts.txt", "topic_keys.txt", "doc_topics.txt"]
        if all((seed0 / f).exists() for f in need):
            out.append(ds)
        else:
            print(f"[{ds}] missing PRISM outputs in {seed0} -- SKIP")
    return out


def out_path_for(ds: str) -> Path:
    """Keep the original pancreas/erythroid filenames; add the rest."""
    special = {
        "gastrulation_erythroid": "prism_gene_trajectory_erythroid.pdf",
        "pancreas": "prism_gene_trajectory_pancreas.pdf",
    }
    return figures_dir() / special.get(ds, f"prism_gene_trajectory_{ds}.pdf")


def _sweep_score(r: dict) -> float:
    """Rank key for a (dataset, method) row: reward a clean directional sweep,
    then span, then sweep Spearman, then marker |rho|.  Higher = better."""
    if not np.isfinite(r.get("marker_rho", np.nan)):
        return -1e9
    return (100.0 * float(r.get("arrow_ok", False))
            + 20.0 * (r.get("span_frac") or 0.0)
            + 10.0 * (r.get("sweep_rho") or 0.0)
            + 5.0 * (r.get("marker_rho") or 0.0))


def run_sweep(datasets: list[str], methods: list[str], csv_out: Path):
    """Measure every (dataset, method), print a ranked table, write CSV."""
    rows = []
    for ds in datasets:
        spec = DATASET_SPECS[ds]
        cache: dict = {}
        print(f"\n===== sweeping methods on {ds} =====")
        for m in methods:
            r = measure_dataset_method(ds, spec, m, _cache=cache)
            r["rank_score"] = _sweep_score(r)
            rows.append(r)
            print(f"  {m:22s} marker|rho|={r['marker_rho']:.3f} "
                  f"arrow_ok={str(r['arrow_ok']):5s} sweep_rho={r['sweep_rho']:.2f} "
                  f"span={r['span_frac']:.0%}"
                  + (f"  ERROR:{r['error']}" if r.get('error') else ""))
    df = pd.DataFrame(rows).sort_values("rank_score", ascending=False)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)
    print(f"\n=== RANKED (dataset, method) — best directional trajectory first ===")
    print(f"    written to {csv_out}")
    show = df[["dataset", "method", "marker_rho", "arrow_ok", "sweep_rho",
               "span_frac", "rank_score"]].copy()
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(show.to_string(index=False,
              formatters={"marker_rho": "{:.3f}".format,
                          "sweep_rho": "{:.2f}".format,
                          "span_frac": "{:.0%}".format,
                          "rank_score": "{:.1f}".format}))
    return df


def best_method_per_dataset(df: pd.DataFrame) -> dict[str, str]:
    best = {}
    for ds, sub in df.groupby("dataset"):
        best[ds] = sub.sort_values("rank_score", ascending=False).iloc[0]["method"]
    return best


# Default scratch location for the method-sweep CSV: a repo-relative outputs dir.
# Override with --csv-out. (No absolute/user-specific path is baked in.)
SCRATCH = WS / "outputs" / "native_trajectory"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=None,
                   help="datasets to render (default = all viable)")
    p.add_argument("--method", default=None, choices=METHODS,
                   help="force this gene-ordering method for every dataset")
    p.add_argument("--sweep-methods", action="store_true",
                   help="measure ALL methods on each dataset, rank, write CSV, "
                        "then render each dataset with its BEST method")
    p.add_argument("--no-figures", action="store_true",
                   help="with --sweep-methods, only rank; do not render PDFs")
    p.add_argument("--csv-out", default=str(SCRATCH / "method_sweep_ranked.csv"))
    args = p.parse_args()

    datasets = args.datasets or viable_datasets()
    datasets = [d for d in datasets if d in DATASET_SPECS]

    if args.sweep_methods:
        df = run_sweep(datasets, METHODS, Path(args.csv_out))
        if args.no_figures:
            return
        best = best_method_per_dataset(df)
        print("\n=== rendering each dataset with its BEST method ===")
        summary = {}
        for ds in datasets:
            m = best.get(ds, DEFAULT_METHOD)
            summary[ds] = make_figure(ds, DATASET_SPECS[ds], out_path_for(ds), method=m)
        _print_summary(summary)
        return

    # non-sweep path: one method per dataset (forced or default)
    summary = {}
    for ds in datasets:
        m = args.method or DEFAULT_METHOD
        summary[ds] = make_figure(ds, DATASET_SPECS[ds], out_path_for(ds), method=m)
    _print_summary(summary)


def _print_summary(summary: dict):
    print("\n=== SUMMARY ===")
    for ds, s in summary.items():
        print(f"  {ds}: method={s['method']} GEPs={s['lineage_topics']} "
              f"n_genes={s['n_program']} (bin={s['n_bin_program']}, "
              f"-{s['n_dropped']} ubiq) marker|rho|={s['marker_rho']:.3f} "
              f"arrow_ok={s['arrow_ok']} sweep_rho={s['sweep_rho']:.2f} "
              f"span={s['span_frac']:.0%} [{s['source']}]")


if __name__ == "__main__":
    main()
