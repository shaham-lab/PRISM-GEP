"""Per-GEP expression and topic-word heatmaps.

Three heatmap families, nine orderings in total.

Top-genes × cells (one panel per GEP, SAME cell ordering across panels):
    byGEP         — cells sorted by dominant-GEP (argmax doc_topics) then by
                    descending attribution within block. Vertical separators
                    between dominant-GEP blocks. Gene rows: descending
                    p(gene|GEP) from word_topic_counts.
    byTrajectory  — cells sorted ascending by `lineage_rank`. The rank comes
                    from outputs/trajectory/<ds>/<ds>_pseudotimes.csv for
                    pancreas, gastrulation and gastrulation_erythroid, and is
                    derived from the cell-type labels in lineage order for
                    bonemarrow and hemogenic_endothelium (LABEL_LINEAGE_ORDERS).
                    Vertical separators between lineage stages. Gene rows:
                    step (ii) diffusion pseudotime within each GEP
                    (bio.evaluate_supp.gene_ordering_supp).
                    Falls back to byGEP for the remaining datasets.
    byCellType    — cells sorted by published cell_type_labels.csv label.
                    Vertical separators between label blocks. Gene rows:
                    hierarchical clustering (average linkage, euclidean) on
                    the 20 genes' expression vectors across the displayed
                    cells. Falls back to byGEP for breast_cancer.

Topic-word distribution (gene × GEP matrix, p(gene|GEP)):
    byDominant    — genes grouped by argmax-GEP (descending max-p within
                    group). Horizontal separators between groups.
    byCluster     — hierarchical clustering on gene × GEP matrix
                    (average linkage, euclidean).
    union         — first-appearance ordering across the per-GEP top-N lists.

Cell-type cross-tabulations (require published labels):
    gep-x-celltype        — fraction of each cell type dominated by each GEP.
    gene-x-celltype       — union of top genes × cell types, log-mean expression.
    topic-byCellType-dist — p(gene|GEP) with gene rows grouped by argmax cell type.

Run:
    python -m bio.heatmaps --dataset pancreas                  # all 9 variants
    python -m bio.heatmaps --dataset pancreas --variant byGEP  # one only
    python -m bio.heatmaps --all-datasets                      # every dataset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
from scipy.cluster.hierarchy import linkage, leaves_list

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS))

from bio.extract_top_genes import (  # noqa: E402
    parse_topic_keys, parse_word_topic_counts, top_n_genes_from_counts,
)
from bio.evaluate_supp import gene_ordering_supp  # noqa: E402


DATASET_FILES = {
    "breast_cancer":          "filtered_breast_cancer_cells_x_genes.csv",
    "pbmc3k":                 "filtered_pbmc3k_cells_x_genes.csv",
    "zeisel_brain":           "filtered_zeisel_brain_cells_x_genes.csv",
    "pancreas":               "filtered_pancreas_cells_x_genes.csv",
    "bonemarrow":             "filtered_bonemarrow_cells_x_genes.csv",
    "hemogenic_endothelium":  "filtered_hemogenic_endothelium_cells_x_genes.csv",
    "gastrulation":           "filtered_gastrulation_cells_x_genes.csv",
    "gastrulation_e75":       "filtered_gastrulation_e75_cells_x_genes.csv",
    "gastrulation_erythroid": "filtered_gastrulation_erythroid_cells_x_genes.csv",
}

ALL_DATASETS = list(DATASET_FILES.keys())
CELL_VARIANTS = ("byGEP", "byTrajectory", "byCellType")
TOPIC_VARIANTS = ("byDominant", "byCluster", "union")
CELLTYPE_VARIANTS = ("gep-x-celltype", "gene-x-celltype", "topic-byCellType-dist")
ALL_VARIANTS = CELL_VARIANTS + TOPIC_VARIANTS + CELLTYPE_VARIANTS

# Datasets that have published cell-trajectories (on-disk pseudotimes CSV)
HAS_TRAJECTORY = {"pancreas", "gastrulation", "gastrulation_erythroid"}
# Datasets that lack published cell-type labels
NO_CELL_TYPES = {"breast_cancer"}

# Hardcoded cell-type -> lineage_rank fallback for datasets without
# on-disk pseudotimes. Cells whose label is not present in these lists
# (incl. "nan") are dropped from byTrajectory plots.
BONEMARROW_LINEAGE_ORDER = [
    "HSC_1", "HSC_2", "Precursors", "CLP",
    "Mega", "Ery_1", "Ery_2",
    "Mono_1", "Mono_2", "DCs",
]

HEMOGENIC_LINEAGE_ORDER = [
    "endothelial cell",
    "endocardium cell progenitor",
    "hematopoietic precursor cell",
    "cardiac muscle myoblast",  # off-trajectory branch, place at end
]

LABEL_LINEAGE_ORDERS = {
    "bonemarrow": BONEMARROW_LINEAGE_ORDER,
    "hemogenic_endothelium": HEMOGENIC_LINEAGE_ORDER,
}

# Datasets where byTrajectory uses a label-derived lineage_rank
HAS_LABEL_LINEAGE = set(LABEL_LINEAGE_ORDERS.keys())

PLOT_CELL_CAP = 600  # max cells to render per figure (stratified subsample)


#
# For the three datasets shown in the paper we colour each gene row-label by
# the canonical cell type / biological program it is a known marker of. A gene
# that is NOT a canonical marker of any program (housekeeping, ribosomal,
# mitochondrial, generic, or simply not in the curated list) stays grey/black.
#
# This is intentionally a SMALL hand-curated dictionary of textbook markers,
# NOT an automated lookup -- the point is to give a non-biologist reader a
# fixed, checkable colour key so they can SEE whether a GEP's top genes are
# the markers of ONE coherent cell type (programme is real) or a MIX of
# several / mostly housekeeping (programme is weak). Sources: standard
# scRNA-seq marker panels (PanglaoDB / CellMarker / the original dataset
# papers). Gene names are matched case-insensitively.
#
# Each value is a (category_key) string; CATEGORY_COLORS below maps category
# -> colour and human-readable legend label. "housekeeping" is given its own
# muted colour because flagging ribosomal/mito dominance is part of the
# honest answer.

MARKERS: dict[str, dict[str, str]] = {
    "pancreas": {
        # beta cells (insulin / amylin)
        "INS1": "beta", "INS2": "beta", "IAPP": "beta", "NNAT": "beta",
        # alpha cells (glucagon)
        "GCG": "alpha", "PYY": "alpha", "PPY": "alpha",
        # delta cells (somatostatin)
        "SST": "delta",
        # epsilon cells (ghrelin)
        "GHRL": "epsilon",
        # pan-endocrine hormone-processing / secretory granule
        "CHGA": "endocrine", "CHGB": "endocrine", "PCSK1N": "endocrine",
        "CPE": "endocrine", "SCG2": "endocrine", "ISL1": "endocrine",
        "TTR": "endocrine", "PCSK2": "endocrine",
        # endocrine progenitor (Ngn3 transient)
        "NEUROG3": "endo_progenitor", "SOX4": "endo_progenitor",
        "BTG2": "endo_progenitor", "CCK": "endo_progenitor",
        # acinar / exocrine
        "CTRB1": "acinar", "PRSS1": "acinar", "CPA1": "acinar",
        # ductal
        "KRT8": "ductal", "KRT18": "ductal", "SPP1": "ductal", "CLU": "ductal",
    },
    "zeisel_brain": {
        # oligodendrocyte (myelin)
        "PLP1": "oligo", "MBP": "oligo", "MOG": "oligo", "MOBP": "oligo",
        "MAG": "oligo", "CNP": "oligo", "MAL": "oligo", "TRF": "oligo",
        "UGT8A": "oligo", "ERMN": "oligo", "CRYAB": "oligo", "QK": "oligo",
        "TSPAN2": "oligo", "APOD": "oligo",
        # astrocyte
        "AQP4": "astro", "SLC1A2": "astro", "SLC1A3": "astro", "GJA1": "astro",
        "ATP1A2": "astro", "APOE": "astro", "CLU": "astro", "SPARCL1": "astro",
        "GFAP": "astro",
        # GABAergic interneuron
        "GAD1": "gaba", "GAD2": "gaba", "VIP": "gaba", "SST": "gaba",
        "NPY": "gaba", "CNR1": "gaba", "HTR3A": "gaba",
        # pan-neuronal / synaptic (shared across excitatory + inhibitory)
        "SNAP25": "neuron", "SYT1": "neuron", "MEG3": "neuron",
        "NRGN": "neuron", "STMN3": "neuron", "STMN2": "neuron",
        "RTN1": "neuron", "NRXN3": "neuron", "PCP4": "neuron",
        "CAMK2N1": "neuron", "GRIA1": "neuron", "SNCA": "neuron",
        # microglia / immune
        "CST3": "microglia", "C1QA": "microglia", "CTSS": "microglia",
        # endothelial / vascular
        "CLDN5": "endothelial", "ACTA2": "endothelial",
    },
    "breast_cancer": {
        # epithelial / tumour (cytokeratins, ERBB2, luminal)
        "KRT8": "epithelial", "KRT18": "epithelial", "KRT19": "epithelial",
        "ERBB2": "epithelial", "TFF1": "epithelial", "TFF3": "epithelial",
        "CD24": "epithelial", "XBP1": "epithelial", "CDH3": "epithelial",
        "GATA3": "epithelial", "MLPH": "epithelial", "CCND1": "epithelial",
        "FASN": "epithelial", "GRB7": "epithelial", "SLC39A6": "epithelial",
        # plasma / B cells (immunoglobulin)
        "IGHG1": "plasma", "IGHG4": "plasma", "IGKC": "plasma",
        "IGHM": "plasma", "DERL3": "plasma", "CD19": "plasma",
        "POU2AF1": "plasma",
        # myeloid / macrophage
        "C1QA": "myeloid", "APOE": "myeloid", "LYZ": "myeloid",
        "FCN1": "myeloid", "S100A8": "myeloid", "LST1": "myeloid",
        "CD36": "myeloid", "CTSL": "myeloid",
        # T cells (incl. Treg)
        "FOXP3": "tcell", "CD3G": "tcell", "TRAC": "tcell", "ICOS": "tcell",
        "IL7R": "tcell", "XCL1": "tcell", "CD69": "tcell",
        # fibroblast / stromal / ECM
        "COL1A2": "stromal", "COL3A1": "stromal", "COL4A1": "stromal",
        "FN1": "stromal", "ACTA2": "stromal", "SULF1": "stromal",
        "HSPG2": "stromal", "LAMA1": "stromal", "MYL9": "stromal",
        "SNAI2": "stromal",
        # endothelial / vascular
        "PECAM1": "endothelial", "PLVAP": "endothelial", "GNG11": "endothelial",
    },
    # ---- PBMC 3k (10x peripheral blood mononuclear cells) -----------------
    # Canonical PBMC panel (PanglaoDB / Seurat pbmc3k tutorial). Many of the
    # actual top genes here are ribosomal (RPL/RPS), mitochondrial (MT-CO*),
    # cytoskeletal (ACTB/TMSB*) or MALAT1 -> left black as housekeeping.
    "pbmc3k": {
        # T cells (the task's CD3D/CD3E/IL7R are the textbook T markers; of
        # these only the more general T/activation markers appear in top-20)
        "CD3D": "tcell_pbmc", "CD3E": "tcell_pbmc", "IL7R": "tcell_pbmc",
        "CCL5": "tcell_pbmc", "LTB": "tcell_pbmc",
        # B cells
        "MS4A1": "bcell_pbmc", "CD79A": "bcell_pbmc", "CD79B": "bcell_pbmc",
        # NK / cytotoxic
        "NKG7": "nk_pbmc", "GNLY": "nk_pbmc", "GZMB": "nk_pbmc",
        "GZMA": "nk_pbmc", "PRF1": "nk_pbmc", "CTSW": "nk_pbmc",
        "CST7": "nk_pbmc",
        # CD14+ monocytes
        "CD14": "mono_pbmc", "LYZ": "mono_pbmc", "S100A8": "mono_pbmc",
        "S100A9": "mono_pbmc", "FCN1": "mono_pbmc", "CST3": "mono_pbmc",
        "TYROBP": "mono_pbmc",
        # FCGR3A+ (CD16) monocytes
        "FCGR3A": "fcgr3a_pbmc", "MS4A7": "fcgr3a_pbmc",
        # dendritic cells / antigen presentation (HLA-II, CD74)
        "FCER1A": "dc_pbmc", "CD74": "dc_pbmc", "HLA-DRA": "dc_pbmc",
        "HLA-DPB1": "dc_pbmc", "HLA-DPA1": "dc_pbmc", "HLA-DRB1": "dc_pbmc",
        # platelets / megakaryocyte
        "PPBP": "platelet_pbmc",
    },
    # ---- bonemarrow (Paul et al. mouse haematopoiesis; lowercase symbols) --
    "bonemarrow": {
        # erythroid
        "HBB": "ery_bm", "HBA-A1": "ery_bm", "HBA-A2": "ery_bm",
        "BLVRB": "ery_bm", "AHSP": "ery_bm", "CA1": "ery_bm",
        "APOC1": "ery_bm", "GYPA": "ery_bm", "GYPC": "ery_bm",
        # myeloid / granulocyte (the task's MPO/ELANE/LYZ)
        "MPO": "myeloid_bm", "ELANE": "myeloid_bm", "PRTN3": "myeloid_bm",
        "AZU1": "myeloid_bm", "CTSG": "myeloid_bm", "LYZ": "myeloid_bm",
        "CLEC11A": "myeloid_bm", "PLAC8": "myeloid_bm", "SRGN": "myeloid_bm",
        # lymphoid (the task's CD3/CD79)
        "CD3D": "lymph_bm", "CD3E": "lymph_bm", "CD79A": "lymph_bm",
        "CD79B": "lymph_bm", "CD52": "lymph_bm", "IGLL1": "lymph_bm",
        "CD74": "lymph_bm",
        # HSC / progenitor
        "CD34": "hsc_bm", "SOX4": "hsc_bm", "SPINK2": "hsc_bm",
        "PRSS57": "hsc_bm", "FAM30A": "hsc_bm", "HOPX": "hsc_bm",
        # megakaryocyte
        "PF4": "mega_bm", "PPBP": "mega_bm",
    },
    # ---- hemogenic_endothelium (EHT; human symbols, ribosomal-dominated) ---
    # Most top genes here are ribosomal (RPS*/RPL*), ferritin (FTL/FTH1),
    # histone (H4C3/H2AZ1) or generic (PTMA/TUBA1B/HMGB1/2) -> black.
    "hemogenic_endothelium": {
        # endothelial (the task's CDH5/PECAM1/KDR)
        "CDH5": "endo_eht", "PECAM1": "endo_eht", "KDR": "endo_eht",
        "GNG11": "endo_eht", "HAPLN1": "endo_eht",
        # hemogenic / HSC (the task's RUNX1/GFI1/MYB)
        "RUNX1": "hemo_eht", "GFI1": "hemo_eht", "MYB": "hemo_eht",
        "MDK": "hemo_eht",
        # blood / erythro-myeloid commitment (the task's GATA1/SPI1)
        "GATA1": "blood_eht", "SPI1": "blood_eht", "LGALS1": "blood_eht",
    },
    # ---- gastrulation (mouse E6.5-E8.5 atlas; Title-case symbols) ----------
    # Most top genes are ribosomal/glycolytic/heat-shock housekeeping -> black.
    "gastrulation": {
        # epiblast / pluripotency
        "POU5F1": "epiblast_gas", "NANOG": "epiblast_gas", "UTF1": "epiblast_gas",
        # primitive streak
        "T": "pstreak_gas", "MIXL1": "pstreak_gas", "FST": "pstreak_gas",
        # mesoderm
        "MESP1": "meso_gas", "HAND1": "meso_gas", "MYL7": "meso_gas",
        "GATA4": "meso_gas",
        # endoderm
        "SOX17": "endo_gas", "FOXA2": "endo_gas", "TTR": "endo_gas",
        # blood / erythroid
        "GATA1": "blood_gas", "HBB-BH1": "blood_gas", "HBB-Y": "blood_gas",
        "HBA-X": "blood_gas", "HBA-A1": "blood_gas", "HBA-A2": "blood_gas",
        "HBB-BS": "blood_gas",
        # ectoderm / neural
        "SOX2": "ecto_gas", "SOX1": "ecto_gas", "PAX6": "ecto_gas",
        # visceral endoderm / epithelial (Krt8/18 strongly expressed here)
        "KRT8": "vendo_gas", "KRT18": "vendo_gas", "SPINK1": "vendo_gas",
        "TRH": "vendo_gas", "EMB": "vendo_gas",
    },
    "gastrulation_e75": {
        # epiblast / pluripotency
        "POU5F1": "epiblast_gas", "NANOG": "epiblast_gas", "UTF1": "epiblast_gas",
        # primitive streak
        "T": "pstreak_gas", "MIXL1": "pstreak_gas", "FST": "pstreak_gas",
        # mesoderm
        "MESP1": "meso_gas", "HAND1": "meso_gas", "MYL7": "meso_gas",
        "GATA4": "meso_gas",
        # endoderm
        "SOX17": "endo_gas", "FOXA2": "endo_gas", "TTR": "endo_gas",
        # blood / erythroid
        "GATA1": "blood_gas", "HBB-BH1": "blood_gas", "HBB-Y": "blood_gas",
        "HBA-X": "blood_gas", "HBA-A1": "blood_gas", "HBA-A2": "blood_gas",
        "HBB-BS": "blood_gas",
        # ectoderm / neural
        "SOX2": "ecto_gas", "SOX1": "ecto_gas", "PAX6": "ecto_gas",
        # visceral endoderm / epithelial
        "KRT8": "vendo_gas", "KRT18": "vendo_gas", "SPINK1": "vendo_gas",
        "TRH": "vendo_gas", "EMB": "vendo_gas",
    },
    "gastrulation_erythroid": {
        # epiblast / pluripotency
        "POU5F1": "epiblast_gas", "NANOG": "epiblast_gas", "UTF1": "epiblast_gas",
        # primitive streak
        "T": "pstreak_gas", "MIXL1": "pstreak_gas", "FST": "pstreak_gas",
        # mesoderm
        "MESP1": "meso_gas", "HAND1": "meso_gas", "MYL7": "meso_gas",
        "GATA4": "meso_gas",
        # endoderm
        "SOX17": "endo_gas", "FOXA2": "endo_gas", "TTR": "endo_gas",
        # blood / erythroid (this is an erythroid-trajectory subset, so the
        # haemoglobin genes dominate the top lists)
        "GATA1": "blood_gas", "HBB-BH1": "blood_gas", "HBB-Y": "blood_gas",
        "HBA-X": "blood_gas", "HBA-A1": "blood_gas", "HBA-A2": "blood_gas",
        "HBB-BS": "blood_gas", "BLVRB": "blood_gas", "CAR2": "blood_gas",
        "CITED2": "blood_gas", "GYPA": "blood_gas",
        # ectoderm / neural
        "SOX2": "ecto_gas", "SOX1": "ecto_gas", "PAX6": "ecto_gas",
        # visceral endoderm / epithelial
        "KRT8": "vendo_gas", "KRT18": "vendo_gas", "SPINK1": "vendo_gas",
        "TRH": "vendo_gas", "EMB": "vendo_gas",
    },
}

# category -> (colour, human-readable legend label)
CATEGORY_COLORS: dict[str, tuple[str, str]] = {
    # pancreas
    "beta":            ("#d62728", "beta (Ins1/2, Iapp)"),
    "alpha":           ("#1f77b4", "alpha (Gcg, Pyy)"),
    "delta":           ("#9467bd", "delta (Sst)"),
    "epsilon":         ("#e377c2", "epsilon (Ghrl)"),
    "endocrine":       ("#ff7f0e", "endocrine (Chga, Cpe, Ttr)"),
    "endo_progenitor": ("#2ca02c", "endocrine progenitor (Neurog3, Sox4)"),
    "acinar":          ("#8c564b", "acinar (Ctrb1)"),
    "ductal":          ("#17becf", "ductal (Krt8/18, Spp1)"),
    # zeisel_brain
    "oligo":           ("#d62728", "oligodendrocyte (Plp1, Mbp, Mog)"),
    "astro":           ("#1f77b4", "astrocyte (Aqp4, Slc1a2, Apoe)"),
    "gaba":            ("#9467bd", "GABAergic (Gad1/2, Sst, Vip)"),
    "neuron":          ("#2ca02c", "pan-neuronal (Snap25, Syt1, Meg3)"),
    "microglia":       ("#ff7f0e", "microglia/immune (Cst3, C1qa)"),
    # breast_cancer
    "epithelial":      ("#d62728", "epithelial/tumour (KRT8/18/19, ERBB2, TFF1)"),
    "plasma":          ("#1f77b4", "plasma/B (IGHG1, IGKC)"),
    "myeloid":         ("#ff7f0e", "myeloid (C1QA, APOE, LYZ)"),
    "tcell":           ("#9467bd", "T cell (FOXP3, CD3G)"),
    "stromal":         ("#8c564b", "fibroblast/stromal (COL1A2, FN1)"),
    # pbmc3k
    "tcell_pbmc":      ("#1f77b4", "T cell (CD3D/E, IL7R, CCL5)"),
    "bcell_pbmc":      ("#17becf", "B cell (MS4A1, CD79A/B)"),
    "nk_pbmc":         ("#9467bd", "NK/cytotoxic (NKG7, GNLY, GZMA/B)"),
    "mono_pbmc":       ("#ff7f0e", "CD14 mono (CD14, LYZ, S100A8/9)"),
    "fcgr3a_pbmc":     ("#bcbd22", "FCGR3A mono (FCGR3A, MS4A7)"),
    "dc_pbmc":         ("#2ca02c", "DC/antigen-pres. (FCER1A, CD74, HLA-II)"),
    "platelet_pbmc":   ("#e377c2", "platelet (PPBP)"),
    # bonemarrow
    "ery_bm":          ("#d62728", "erythroid (Hbb, Blvrb, Ca1, Ahsp)"),
    "myeloid_bm":      ("#ff7f0e", "myeloid/granulocyte (Mpo, Elane, Lyz)"),
    "lymph_bm":        ("#1f77b4", "lymphoid (Cd3, Cd79, Cd74, Igll1)"),
    "hsc_bm":          ("#2ca02c", "HSC/progenitor (Sox4, Spink2, Prss57)"),
    "mega_bm":         ("#e377c2", "megakaryocyte (Pf4, Ppbp)"),
    # hemogenic_endothelium
    "endo_eht":        ("#1f77b4", "endothelial (Cdh5, Pecam1, Kdr, Gng11)"),
    "hemo_eht":        ("#d62728", "hemogenic/HSC (Runx1, Gfi1, Myb)"),
    "blood_eht":       ("#ff7f0e", "blood commitment (Gata1, Spi1)"),
    # gastrulation family (germ-layer / lineage)
    "epiblast_gas":    ("#9467bd", "epiblast (Pou5f1, Nanog)"),
    "pstreak_gas":     ("#8c564b", "primitive streak (T, Mixl1)"),
    "meso_gas":        ("#2ca02c", "mesoderm (Mesp1, Hand1, Myl7)"),
    "endo_gas":        ("#17becf", "endoderm (Sox17, Foxa2)"),
    "blood_gas":       ("#d62728", "blood/erythroid (Gata1, Hba-x, Hbb-bh1)"),
    "ecto_gas":        ("#bcbd22", "ectoderm/neural (Sox2)"),
    "vendo_gas":       ("#ff7f0e", "visceral endoderm/epithelial (Krt8/18, Spink1)"),
    # shared
    "endothelial":     ("#7f7f7f", "endothelial (PECAM1, Cldn5)"),
}

# colour for genes with no canonical marker assignment
NONMARKER_COLOR = "#000000"   # plain black
HOUSEKEEPING_COLOR = "#000000"  # (kept for clarity; non-markers are black)


def gene_category(dataset: str, gene: str) -> str | None:
    """Return the marker category for a gene in a dataset, or None if the
    gene is not a curated canonical marker (case-insensitive)."""
    table = MARKERS.get(dataset)
    if table is None:
        return None
    return table.get(gene.upper())


def gene_label_color(dataset: str, gene: str) -> str:
    """Colour for a gene's row-label tick."""
    cat = gene_category(dataset, gene)
    if cat is None:
        return NONMARKER_COLOR
    return CATEGORY_COLORS[cat][0]


def color_ytick_labels(ax, genes: list[str], dataset: str) -> None:
    """Recolour the y-tick labels of `ax` (already set to `genes`, in order)
    by their marker category. Non-markers stay black."""
    for tick, g in zip(ax.get_yticklabels(), genes):
        tick.set_color(gene_label_color(dataset, g))


def marker_legend_handles(dataset: str, genes_shown: set[str]):
    """Build matplotlib legend handles for the marker categories that
    actually appear among `genes_shown` for this dataset. Always appends a
    'not a canonical marker' grey/black entry."""
    from matplotlib.patches import Patch
    table = MARKERS.get(dataset, {})
    cats_present: list[str] = []
    for g in genes_shown:
        cat = table.get(g.upper())
        if cat is not None and cat not in cats_present:
            cats_present.append(cat)
    # Keep a stable, biologically grouped order: follow CATEGORY_COLORS order
    ordered = [c for c in CATEGORY_COLORS if c in cats_present]
    handles = [Patch(facecolor=CATEGORY_COLORS[c][0], edgecolor="none",
                     label=CATEGORY_COLORS[c][1]) for c in ordered]
    handles.append(Patch(facecolor=NONMARKER_COLOR, edgecolor="none",
                         label="not a canonical marker"))
    return handles



def load_top_per_gep(seed_dir: Path, top_n: int = 20) -> dict[int, list[str]]:
    keys = seed_dir / "topic_keys.txt"
    if keys.exists():
        per = parse_topic_keys(keys)
        if per and all(len(v) >= top_n for v in per.values()):
            return {t: per[t][:top_n] for t in per}
    counts = seed_dir / "word_topic_counts.txt"
    return top_n_genes_from_counts(parse_word_topic_counts(counts), n=top_n)


def load_doc_topics(seed_dir: Path) -> np.ndarray | None:
    dt = seed_dir / "doc_topics.txt"
    if not dt.exists():
        return None
    df = pd.read_csv(dt, sep="\t", header=None)
    return df.iloc[:, 2:].values


def load_doc_topic_cell_ids(seed_dir: Path) -> list[str] | None:
    dt = seed_dir / "doc_topics.txt"
    if not dt.exists():
        return None
    df = pd.read_csv(dt, sep="\t", header=None)
    return df.iloc[:, 1].astype(str).tolist()


def load_lineage_rank(dataset: str) -> pd.Series | None:
    path = WS / "outputs" / "trajectory" / dataset / f"{dataset}_pseudotimes.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "lineage_rank" not in df.columns or "cell_id" not in df.columns:
        return None
    return pd.Series(df["lineage_rank"].values, index=df["cell_id"].astype(str))


def load_lineage_stage_names(dataset: str) -> pd.Series | None:
    """Return a Series cell_id -> stage NAME (e.g. 'Ductal') for datasets
    whose pseudotimes CSV has a `lineage` column. Returns None otherwise."""
    path = WS / "outputs" / "trajectory" / dataset / f"{dataset}_pseudotimes.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "lineage" not in df.columns or "cell_id" not in df.columns:
        return None
    return pd.Series(df["lineage"].astype(str).values,
                     index=df["cell_id"].astype(str))


def load_lineage_rank_from_labels(dataset: str) -> tuple[pd.Series, pd.Series] | None:
    """Fallback lineage_rank built from cell_type_labels using the hardcoded
    ordering for `bonemarrow` / `hemogenic_endothelium`.

    Returns (rank_series, stage_name_series) keyed by cell_id, with cells
    whose label is missing or unknown DROPPED (not just NaN-ed).
    """
    if dataset not in LABEL_LINEAGE_ORDERS:
        return None
    order = LABEL_LINEAGE_ORDERS[dataset]
    label_to_rank = {lab: i for i, lab in enumerate(order)}
    labels = load_cell_type_labels(dataset)
    if labels is None:
        return None
    # Drop cells whose label is not in the ordering (incl. literal "nan")
    keep = labels.isin(order)
    labels = labels[keep]
    ranks = labels.map(label_to_rank).astype(int)
    return ranks, labels.astype(str)


def load_cell_type_labels(dataset: str) -> pd.Series | None:
    path = WS / "data" / dataset / "cell_type_labels.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "cell_id" not in df.columns:
        return None
    label_col = [c for c in df.columns if c != "cell_id"][0]
    return pd.Series(df[label_col].astype(str).values, index=df["cell_id"].astype(str))


def case_insensitive_select(expr: pd.DataFrame, genes: list[str]) -> list[str]:
    lut = {c.upper(): c for c in expr.columns}
    return [lut[g.upper()] for g in genes if g.upper() in lut]


def build_gene_topic_distrib(
    word_topic_counts: dict[int, dict[str, int]],
) -> dict[str, np.ndarray]:
    """gene -> normalised topic distribution (sums to 1 across topics)."""
    K = max(word_topic_counts.keys()) + 1
    all_genes: set[str] = set()
    for gd in word_topic_counts.values():
        all_genes.update(gd.keys())
    out: dict[str, np.ndarray] = {}
    for g in all_genes:
        v = np.array([word_topic_counts.get(t, {}).get(g, 0) for t in range(K)],
                     dtype=float)
        s = v.sum()
        if s > 0:
            out[g] = v / s
    return out



def _stratified_subsample(
    indices: np.ndarray, blocks: np.ndarray, cap: int, rng: np.random.RandomState,
) -> np.ndarray:
    """Subsample `indices` (length == len(blocks)) so total <= cap, stratified
    by `blocks`. Preserves order within each block."""
    if len(indices) <= cap:
        return indices
    unique_blocks, counts = np.unique(blocks, return_counts=True)
    # proportional allocation, min 1 per non-empty block
    alloc = np.maximum(1, np.round(counts / counts.sum() * cap).astype(int))
    # Trim excess if we over-allocated
    while alloc.sum() > cap:
        i = np.argmax(alloc)
        alloc[i] -= 1
    keep_mask = np.zeros(len(indices), dtype=bool)
    for b, n in zip(unique_blocks, alloc):
        block_pos = np.where(blocks == b)[0]
        if len(block_pos) <= n:
            keep_mask[block_pos] = True
        else:
            # Evenly spaced sample preserves visual ordering
            sel = np.linspace(0, len(block_pos) - 1, n).astype(int)
            keep_mask[block_pos[sel]] = True
    return indices[keep_mask]


def cell_order_byGEP(
    doc_topics: np.ndarray, rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (cell_order, block_ids, block_boundaries).
    block_ids[i] = dominant GEP for the cell at position i in cell_order.
    block_boundaries = positions where a new GEP block starts (excluding 0)."""
    dom = np.argmax(doc_topics, axis=1)
    n = len(dom)
    full_order = np.arange(n)
    # Sort by dominant GEP ascending, then by descending attribution
    sort_key = np.lexsort((-doc_topics[full_order, dom[full_order]], dom[full_order]))
    full_order = full_order[sort_key]
    block_seq = dom[full_order]
    # Now subsample to PLOT_CELL_CAP, stratified by dominant GEP
    full_order = _stratified_subsample(full_order, block_seq, PLOT_CELL_CAP, rng)
    block_seq = dom[full_order]
    boundaries = np.where(np.diff(block_seq) != 0)[0] + 1
    return full_order, block_seq, boundaries


def cell_order_byTrajectory(
    doc_topics: np.ndarray,
    cell_ids: list[str],
    lineage: pd.Series,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Returns (cell_order, stage_ids, boundaries) or None if no overlap."""
    n = len(cell_ids)
    # Align: pull lineage_rank for each cell id; cells without rank get NaN
    rank_per_cell = np.array(
        [lineage.get(cid, np.nan) for cid in cell_ids], dtype=float
    )
    valid = ~np.isnan(rank_per_cell)
    if valid.sum() < 10:
        return None
    full_order = np.where(valid)[0]
    # sort by rank ascending
    sort_key = np.argsort(rank_per_cell[full_order], kind="stable")
    full_order = full_order[sort_key]
    block_seq = rank_per_cell[full_order].astype(int)
    full_order = _stratified_subsample(full_order, block_seq, PLOT_CELL_CAP, rng)
    block_seq = rank_per_cell[full_order].astype(int)
    boundaries = np.where(np.diff(block_seq) != 0)[0] + 1
    return full_order, block_seq, boundaries


def cell_order_byCellType(
    cell_ids: list[str],
    labels: pd.Series,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n = len(cell_ids)
    lab_per_cell = np.array(
        [labels.get(cid, None) for cid in cell_ids], dtype=object
    )
    valid = np.array([x is not None for x in lab_per_cell])
    if valid.sum() < 10:
        return None
    full_order = np.where(valid)[0]
    # encode labels to ints, sorted alphabetically
    uniq = sorted({lab_per_cell[i] for i in full_order})
    label_to_int = {lab: i for i, lab in enumerate(uniq)}
    int_lab = np.array([label_to_int[lab_per_cell[i]] for i in full_order])
    sort_key = np.argsort(int_lab, kind="stable")
    full_order = full_order[sort_key]
    block_seq = int_lab[sort_key]
    full_order = _stratified_subsample(full_order, block_seq, PLOT_CELL_CAP, rng)
    # recompute block_seq after subsampling
    pos_map = {old: i for i, old in enumerate(np.where(valid)[0][sort_key])}
    block_seq = np.array(
        [label_to_int[lab_per_cell[i]] for i in full_order]
    )
    boundaries = np.where(np.diff(block_seq) != 0)[0] + 1
    return full_order, block_seq, boundaries, uniq



def gene_order_byGEP_default(genes: list[str]) -> list[str]:
    """Genes already arrive in descending p(gene|GEP) order (topic_keys/wtc).
    """
    return list(genes)


def gene_order_byTrajectory(
    genes: list[str], gene_topic_distrib: dict[str, np.ndarray],
) -> list[str]:
    """Step (ii) diffusion pseudotime within the GEP."""
    try:
        ordered, _ = gene_ordering_supp(genes, gene_topic_distrib)
    except Exception:
        return list(genes)
    # gene_ordering_supp drops genes lacking a distribution; pad with the rest
    remainder = [g for g in genes if g not in ordered]
    return list(ordered) + remainder


def gene_order_byCluster_on_expr(
    sub_expr: np.ndarray, genes: list[str],
) -> list[str]:
    """Hierarchical clustering on the 20 genes' expression vectors."""
    if sub_expr.shape[1] < 2 or len(genes) < 2:
        return list(genes)
    X = sub_expr.T  # genes x cells
    if not np.isfinite(X).all() or X.std(axis=1).sum() == 0:
        return list(genes)
    try:
        Z = linkage(X, method="average", metric="euclidean")
        order = leaves_list(Z)
        return [genes[i] for i in order]
    except Exception:
        return list(genes)



def heatmap_top_genes_x_cells(
    expr: pd.DataFrame,
    top_genes_per_gep: dict[int, list[str]],
    doc_topics: np.ndarray,
    out_path: Path,
    *,
    dataset: str,
    cell_variant: str,
    cell_ids: list[str],
    gene_topic_distrib: dict[str, np.ndarray],
    log_transform: bool = True,
    rng: np.random.RandomState | None = None,
    title: str | None = None,
    legible: bool = False,
):
    # `legible=True` renders a full-page supplement-quality figure: larger gene
    # row-labels, a less-squashed per-panel aspect, no embedded matplotlib
    # caption (the LaTeX \caption carries the prose), so that one figure per
    # page reads clearly at 100% zoom. The marker-colour legend is kept.
    rng = rng or np.random.RandomState(0)
    K = len(top_genes_per_gep)

    fallback_note = None
    # Panel order over GEPs: defaults to ascending GEP id; byTrajectory
    # overrides with ascending mean lineage_rank of dominant-GEP cells.
    panel_gep_order: list[int] = sorted(top_genes_per_gep.keys())
    panel_annot: dict[int, str] = {}  # gep -> annotation suffix

    if cell_variant == "byTrajectory":
        # Resolve lineage_rank: prefer on-disk pseudotimes, else label fallback.
        lineage = None
        stage_names = None
        label_derived = False
        if dataset in HAS_TRAJECTORY:
            lineage = load_lineage_rank(dataset)
            stage_names = load_lineage_stage_names(dataset)
        elif dataset in HAS_LABEL_LINEAGE:
            lab = load_lineage_rank_from_labels(dataset)
            if lab is not None:
                lineage, stage_names = lab
                label_derived = True

        tr = (cell_order_byTrajectory(doc_topics, cell_ids, lineage, rng)
              if lineage is not None else None)
        if tr is None:
            fallback_note = "(fallback: by dominant GEP — no usable lineage)"
            order_info = cell_order_byGEP(doc_topics, rng)
            block_labels = None
        else:
            cell_order, block_seq, boundaries = tr
            order_info = (cell_order, block_seq, boundaries)
            uniq_stages = sorted(np.unique(block_seq).tolist())
            # Build block labels: prefer stage NAMES when available
            if stage_names is not None:
                # For each integer rank, take modal stage name among cells
                # at that rank in the *plotted* order.
                cid_arr = np.asarray(cell_ids)
                rank_to_name: dict[int, str] = {}
                for s in uniq_stages:
                    pos = np.where(block_seq == s)[0]
                    names_here = [stage_names.get(cid_arr[cell_order[p]])
                                  for p in pos
                                  if stage_names.get(cid_arr[cell_order[p]]) is not None]
                    if names_here:
                        # modal label
                        vals, cnts = np.unique(np.asarray(names_here),
                                               return_counts=True)
                        rank_to_name[s] = str(vals[np.argmax(cnts)])
                    else:
                        rank_to_name[s] = f"stage {s}"
                block_labels = [f"{s}:{rank_to_name[s]}" for s in uniq_stages]
            else:
                block_labels = [f"stage {s}" for s in uniq_stages]
            if label_derived:
                fallback_note = ("(lineage_rank derived from cell-type "
                                 "labels in lineage order)")

            # --- Panel reorder by mean lineage_rank of each GEP's
            # dominant cells (computed over the PLOTTED cells) ---
            dom_plot = np.argmax(doc_topics[cell_order], axis=1)
            rank_plot = block_seq.astype(float)
            gep_mean_rank: dict[int, float] = {}
            for g in sorted(top_genes_per_gep.keys()):
                mask = (dom_plot == g)
                if mask.any():
                    gep_mean_rank[g] = float(rank_plot[mask].mean())
                else:
                    gep_mean_rank[g] = float("inf")  # push empty GEPs to end
            panel_gep_order = sorted(top_genes_per_gep.keys(),
                                     key=lambda g: (gep_mean_rank[g], g))

            # Annotations: "GEP g (mean rank X.X[ — StageName])"
            # Map mean rank -> modal stage label (if names exist).
            for g in panel_gep_order:
                mr = gep_mean_rank[g]
                if not np.isfinite(mr):
                    panel_annot[g] = "(no dominant cells)"
                    continue
                ann = f"(mean rank {mr:.1f}"
                if stage_names is not None:
                    mask = (dom_plot == g)
                    cid_arr = np.asarray(cell_ids)
                    names_here = [stage_names.get(cid_arr[cell_order[p]])
                                  for p in np.where(mask)[0]
                                  if stage_names.get(cid_arr[cell_order[p]]) is not None]
                    if names_here:
                        vals, cnts = np.unique(np.asarray(names_here),
                                               return_counts=True)
                        ann += f" — {vals[np.argmax(cnts)]}"
                ann += ")"
                panel_annot[g] = ann
    elif cell_variant == "byCellType":
        if dataset in NO_CELL_TYPES:
            fallback_note = "(fallback: by dominant GEP — no cell-type labels)"
            order_info = cell_order_byGEP(doc_topics, rng)
            block_labels = None
        else:
            labels = load_cell_type_labels(dataset)
            ct = cell_order_byCellType(cell_ids, labels, rng) if labels is not None else None
            if ct is None:
                fallback_note = "(fallback: by dominant GEP — cell-type join failed)"
                order_info = cell_order_byGEP(doc_topics, rng)
                block_labels = None
            else:
                cell_order, block_seq, boundaries, uniq_labs = ct
                order_info = (cell_order, block_seq, boundaries)
                block_labels = uniq_labs
    else:  # byGEP
        order_info = cell_order_byGEP(doc_topics, rng)
        block_labels = [f"GEP {g}" for g in sorted(top_genes_per_gep.keys())]

    cell_order, block_seq, boundaries = order_info
    n_cells = len(cell_order)

    # X-axis caption per variant
    if cell_variant == "byTrajectory" and fallback_note is None:
        xcap = "cells sorted by lineage_rank (left to right: early to late)"
    elif cell_variant == "byCellType" and fallback_note is None:
        xcap = "cells sorted by published cell-type label"
    else:
        xcap = "cells sorted by dominant GEP"

    # --- Figure layout: horizontal 1 x K panels, side-by-side ---
    # Default: de-squashed aspect (3.0*K wide) for the small inline stub.
    # legible: narrower per-panel width (so the whole landscape image is less
    # extreme in aspect and survives width-scaling to \textwidth on a full
    # page) and a taller canvas, so the 20 gene rows per panel render large.
    panel_w = 2.4 if legible else 3.0
    canvas_h = 8.6 if legible else 6.4
    fig, axes = plt.subplots(
        1, K,
        figsize=(panel_w * K, canvas_h),
        sharey=False, sharex=True,
    )
    if K == 1:
        axes = np.array([axes])
    ytick_fs = 12 if legible else 8

    gene_cmap = plt.get_cmap("viridis")
    genes_shown: set[str] = set()  # collect for the marker legend

    for i, gep in enumerate(panel_gep_order):
        ax = axes[i]
        gene_list = top_genes_per_gep[gep]

        if cell_variant == "byTrajectory":
            ordered_genes = gene_order_byTrajectory(gene_list, gene_topic_distrib)
        elif cell_variant == "byCellType":
            tmp = case_insensitive_select(expr, gene_list)
            if tmp:
                sub_tmp = expr.iloc[cell_order, :][tmp].values
                if log_transform:
                    sub_tmp = np.log1p(sub_tmp)
                ordered_genes_tmp = gene_order_byCluster_on_expr(sub_tmp, tmp)
                up_to_orig = {g.upper(): g for g in gene_list}
                ordered_genes = [up_to_orig[g.upper()] for g in ordered_genes_tmp
                                 if g.upper() in up_to_orig]
                missing = [g for g in gene_list if g not in ordered_genes]
                ordered_genes = ordered_genes + missing
            else:
                ordered_genes = list(gene_list)
        else:
            ordered_genes = gene_order_byGEP_default(gene_list)

        present = case_insensitive_select(expr, ordered_genes)
        if not present:
            ax.set_visible(False)
            continue
        sub = expr.iloc[cell_order, :][present].values
        if log_transform:
            sub = np.log1p(sub)
        # Enforce shared cell-axis: every panel MUST be n_cells wide
        assert sub.shape[0] == n_cells, (
            f"panel cell count mismatch: {sub.shape[0]} vs {n_cells}"
        )
        sub = sub.T  # genes x cells
        assert sub.shape[1] == n_cells, (
            f"panel cell count mismatch: {sub.shape[1]} vs {n_cells}"
        )

        im = ax.imshow(
            sub, aspect="auto", cmap=gene_cmap, interpolation="nearest",
        )
        ax.set_yticks(range(len(present)))
        ax.set_yticklabels(present, fontsize=ytick_fs)
        # colour each gene row-label by the canonical cell-type/program
        # it marks (non-markers stay black). Lets a non-biologist SEE whether
        # the panel's top genes are markers of ONE coherent program.
        color_ytick_labels(ax, present, dataset)
        genes_shown.update(present)
        if gep in panel_annot:
            ax.set_title(f"GEP {gep} {panel_annot[gep]}",
                         fontsize=11 if legible else 9)
        else:
            ax.set_title(f"GEP {gep}", fontsize=12 if legible else 10)
        for b in boundaries:
            ax.axvline(b - 0.5, color="white", linewidth=0.7, alpha=0.85)
        ax.set_xticks([])
        ax.set_xlabel(xcap, fontsize=10 if legible else 8)
        # Per-panel colourbar (original behaviour)
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                            label="log(1+count)" if log_transform else "count")
        cbar.ax.tick_params(labelsize=9 if legible else 6)

    subtitle = (
        f"{dataset} — top-20 genes x cells (cells {cell_variant})"
        f"  [n={n_cells}, cap={PLOT_CELL_CAP}]"
    )
    if fallback_note:
        subtitle += f"  {fallback_note}"
    if not legible:
        fig.suptitle(title or subtitle, fontsize=10)
    # legible: reclaim the vertical space the embedded caption used (the LaTeX
    # \caption carries the prose) -- leave only a thin band for the legend.
    rect_bottom = 0.10 if legible else 0.30
    try:
        fig.tight_layout(rect=(0, rect_bottom, 1, 0.96))
    except Exception:
        pass
    # marker-colour legend across the bottom, mapping each gene-label
    # colour to the canonical cell type / program it marks. Sits just below
    # the panels (and, in the inline stub, above the wrapped caption).
    handles = marker_legend_handles(dataset, genes_shown)
    if handles:
        fig.legend(
            handles=handles, title="gene-label colour = canonical marker of",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015 if legible else 0.20),
            ncol=min(len(handles), 5),
            fontsize=9 if legible else 6, title_fontsize=10 if legible else 7,
            frameon=False, handlelength=1.0, columnspacing=1.2,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # legible: extra pad so the tight bbox does not crop the last panel's
    # x-axis label ("...dominant GEP") that overhangs the rightmost axes.
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                pad_inches=0.25 if legible else 0.1)
    plt.close(fig)


def _seed_from_layout(out_path: Path) -> int | None:
    """Best-effort extract 'seedN' from the output filename."""
    import re
    m = re.search(r"seed(\d+)", out_path.name)
    return int(m.group(1)) if m else None



def _compute_gep_mean_lineage_rank(
    dataset: str,
    doc_topics: np.ndarray | None,
    cell_ids: list[str] | None,
    K: int,
) -> dict[int, float] | None:
    """Return {gep -> mean lineage_rank of cells where it is dominant}, or
    None if dataset has no lineage_rank source. Cells without rank are skipped.
    GEPs with no dominant cells get +inf so they sort to the end."""
    if doc_topics is None or cell_ids is None:
        return None
    if dataset in HAS_TRAJECTORY:
        lineage = load_lineage_rank(dataset)
    elif dataset in HAS_LABEL_LINEAGE:
        lab = load_lineage_rank_from_labels(dataset)
        lineage = lab[0] if lab is not None else None
    else:
        return None
    if lineage is None:
        return None
    rank_per_cell = np.array(
        [lineage.get(cid, np.nan) for cid in cell_ids], dtype=float
    )
    dom = np.argmax(doc_topics, axis=1)
    out: dict[int, float] = {}
    for g in range(K):
        mask = (dom == g) & (~np.isnan(rank_per_cell))
        out[g] = float(rank_per_cell[mask].mean()) if mask.any() else float("inf")
    return out


def heatmap_topic_word_distribution(
    word_topic_counts: dict[int, dict[str, int]],
    top_genes_per_gep: dict[int, list[str]],
    out_path: Path,
    *,
    dataset: str,
    topic_variant: str,
    title: str | None = None,
    doc_topics: np.ndarray | None = None,
    cell_ids: list[str] | None = None,
    legible: bool = False,
):
    # `legible=True` renders a full-page supplement figure: larger gene
    # row-labels and taller rows, no embedded matplotlib caption (the LaTeX
    # \caption carries the prose), so one figure per page reads at 100% zoom.
    # Union of top genes (first-appearance order)
    all_top: list[str] = []
    seen: set[str] = set()
    for gep in sorted(top_genes_per_gep.keys()):
        for g in top_genes_per_gep[gep]:
            if g not in seen:
                all_top.append(g)
                seen.add(g)

    K = max(word_topic_counts.keys()) + 1
    topic_totals = np.zeros(K)
    for t, gd in word_topic_counts.items():
        topic_totals[t] = sum(gd.values())

    matrix = np.zeros((len(all_top), K))
    for i, g in enumerate(all_top):
        for t in range(K):
            c = word_topic_counts.get(t, {}).get(g, 0)
            if topic_totals[t] > 0 and c > 0:
                matrix[i, t] = c / topic_totals[t]

    sep_rows: list[int] = []  # horizontal separator positions
    # Column ordering over GEPs. Default identity; byDominant may reorder
    # by mean lineage_rank when the dataset has one.
    col_order: list[int] = list(range(K))
    col_labels: list[str] = [f"GEP {t}" for t in range(K)]
    reorder_note = ""

    if topic_variant == "byDominant":
        gep_mr = _compute_gep_mean_lineage_rank(dataset, doc_topics, cell_ids, K)
        if gep_mr is not None:
            col_order = sorted(range(K), key=lambda g: (gep_mr[g], g))
            matrix = matrix[:, col_order]
            col_labels = []
            for g in col_order:
                mr = gep_mr[g]
                if np.isfinite(mr):
                    col_labels.append(f"GEP {g}\n(rank {mr:.1f})")
                else:
                    col_labels.append(f"GEP {g}\n(no cells)")
            reorder_note = " (cols sorted by mean lineage_rank)"

        argmax = np.argmax(matrix, axis=1)
        max_p = matrix.max(axis=1)
        order_keys = list(zip(argmax.tolist(), (-max_p).tolist(),
                              range(len(all_top))))
        order = [i for _, _, i in sorted(order_keys)]
        all_top = [all_top[i] for i in order]
        matrix = matrix[order, :]
        # Recompute group boundaries
        new_argmax = np.argmax(matrix, axis=1)
        sep_rows = (np.where(np.diff(new_argmax) != 0)[0] + 1).tolist()
    elif topic_variant == "byCluster":
        if len(all_top) >= 2:
            X = matrix.copy()
            # Avoid zero-variance crash
            if X.std() > 0:
                try:
                    Z = linkage(X, method="average", metric="euclidean")
                    order = leaves_list(Z)
                    all_top = [all_top[i] for i in order]
                    matrix = matrix[order, :]
                except Exception:
                    pass
    # 'union' leaves the original first-appearance ordering

    matrix = np.maximum(matrix, 1e-8)

    row_h = 0.26 if legible else 0.18
    col_w = 2.6 if legible else 0.6
    fig, ax = plt.subplots(
        figsize=(col_w * K + 2.2, row_h * len(all_top) + 1.6))
    im = ax.imshow(matrix, aspect="auto", cmap="magma",
                   norm=LogNorm(vmin=max(matrix.min(), 1e-6), vmax=matrix.max()))
    ax.set_xticks(range(K))
    ax.set_xticklabels(col_labels, rotation=45, ha="right",
                       fontsize=11 if legible else 8)
    ax.set_yticks(range(len(all_top)))
    ax.set_yticklabels(all_top, fontsize=9 if legible else 6)
    # colour each gene row-label by the canonical cell-type/program it
    # marks (non-markers stay black).
    color_ytick_labels(ax, all_top, dataset)
    for r in sep_rows:
        ax.axhline(r - 0.5, color="white", linewidth=0.5, alpha=0.8)
    subtitle = title or (
        f"{dataset} — p(gene|GEP) (rows {topic_variant}){reorder_note}"
    )
    if not legible:
        ax.set_title(subtitle, fontsize=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="p(gene|GEP)")
    if legible:
        cbar.ax.tick_params(labelsize=9)
    # marker-colour legend below the heatmap.
    handles = marker_legend_handles(dataset, set(all_top))
    if handles:
        fig.legend(
            handles=handles, title="gene-label colour = marker of",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01 if legible else 0.135),
            ncol=2, fontsize=8 if legible else 5,
            title_fontsize=9 if legible else 6,
            frameon=False, handlelength=1.0, columnspacing=1.0,
        )
    # legible: reclaim the caption band (LaTeX \caption carries the prose);
    # keep just enough bottom margin for the marker legend.
    fig.tight_layout(rect=(0, 0.06 if legible else 0.30, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                pad_inches=0.25 if legible else 0.1)
    plt.close(fig)



def _resolve_celltype_inputs(
    dataset: str,
    doc_topics: np.ndarray,
    cell_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, str], dict[str, float] | None,
           dict[str, str] | None] | None:
    """Build per-cell aligned arrays of (kept_cell_indices, kept_labels, ordered_types,
    type_to_count_str, type_mean_rank_or_None, type_modal_stage_or_None).

    Drops cells whose label is missing or equal to literal 'nan'.
    """
    labels = load_cell_type_labels(dataset)
    if labels is None:
        return None
    n = len(cell_ids)
    lab_per_cell = np.array(
        [labels.get(cid, None) for cid in cell_ids], dtype=object
    )
    keep_mask = np.array(
        [(x is not None) and (str(x).lower() != "nan") for x in lab_per_cell]
    )
    if keep_mask.sum() < 10:
        return None
    kept_idx = np.where(keep_mask)[0]
    kept_labels = np.array([str(lab_per_cell[i]) for i in kept_idx])

    # Cell-type ordering:
    # - trajectory datasets: by mean lineage_rank of cells of that type
    # - non-trajectory: alphabetical
    rank_per_cell = None
    stage_names = None
    if dataset in HAS_TRAJECTORY:
        lineage = load_lineage_rank(dataset)
        stage_names = load_lineage_stage_names(dataset)
        if lineage is not None:
            rank_per_cell = np.array(
                [lineage.get(cid, np.nan) for cid in cell_ids], dtype=float
            )
    elif dataset in HAS_LABEL_LINEAGE:
        lab = load_lineage_rank_from_labels(dataset)
        if lab is not None:
            lin, sn = lab
            rank_per_cell = np.array(
                [lin.get(cid, np.nan) for cid in cell_ids], dtype=float
            )
            stage_names = sn

    uniq_types = sorted(set(kept_labels.tolist()))
    type_mean_rank: dict[str, float] | None = None
    if rank_per_cell is not None:
        type_mean_rank = {}
        for t in uniq_types:
            mask = (kept_labels == t)
            rk = rank_per_cell[kept_idx][mask]
            rk = rk[~np.isnan(rk)]
            type_mean_rank[t] = float(rk.mean()) if len(rk) else float("inf")
        ordered_types = sorted(uniq_types, key=lambda t: (type_mean_rank[t], t))
    else:
        ordered_types = uniq_types

    type_count_str = {t: f"{t} (n={int((kept_labels == t).sum())})"
                      for t in ordered_types}

    # modal stage name per type if stage_names available (for row labels)
    type_modal_stage = None
    if stage_names is not None:
        type_modal_stage = {}
        for t in ordered_types:
            mask = (kept_labels == t)
            cids_here = [cell_ids[i] for i in kept_idx[mask]]
            names = [stage_names.get(c) for c in cids_here]
            names = [n for n in names if n is not None]
            if names:
                vals, cnts = np.unique(np.asarray(names), return_counts=True)
                type_modal_stage[t] = str(vals[np.argmax(cnts)])

    return (kept_idx, kept_labels, ordered_types, type_count_str,
            type_mean_rank, type_modal_stage)


def _gep_panel_order_and_annot(
    dataset: str, doc_topics: np.ndarray, cell_ids: list[str], K: int,
) -> tuple[list[int], dict[int, str]]:
    """For trajectory datasets, reorder GEPs by mean lineage rank of dominant
    cells and build 'mean rank X.X — StageName' annotations.
    For non-trajectory datasets, return identity order and empty annotation."""
    gep_mr = _compute_gep_mean_lineage_rank(dataset, doc_topics, cell_ids, K)
    if gep_mr is None:
        return list(range(K)), {}
    panel_order = sorted(range(K), key=lambda g: (gep_mr[g], g))
    # modal stage per GEP (when stage_names available)
    stage_names = None
    if dataset in HAS_TRAJECTORY:
        stage_names = load_lineage_stage_names(dataset)
    elif dataset in HAS_LABEL_LINEAGE:
        lab = load_lineage_rank_from_labels(dataset)
        stage_names = lab[1] if lab is not None else None
    dom = np.argmax(doc_topics, axis=1)
    annot: dict[int, str] = {}
    for g in panel_order:
        mr = gep_mr[g]
        if not np.isfinite(mr):
            annot[g] = "(no dominant cells)"
            continue
        s = f"(mean rank {mr:.1f}"
        if stage_names is not None:
            mask = (dom == g)
            cids_here = [cell_ids[i] for i in np.where(mask)[0]]
            names = [stage_names.get(c) for c in cids_here]
            names = [n for n in names if n is not None]
            if names:
                vals, cnts = np.unique(np.asarray(names), return_counts=True)
                s += f" — {vals[np.argmax(cnts)]}"
        s += ")"
        annot[g] = s
    return panel_order, annot


def figure_gep_x_celltype(
    doc_topics: np.ndarray,
    cell_ids: list[str],
    dataset: str,
    out_path: Path,
    *,
    return_matrix: bool = False,
):
    """GEP x cell-type dominance overlap: K x N matrix where cell[g, t] = fraction of cells of type t
    where GEP g is the argmax dominant topic.
    """
    res = _resolve_celltype_inputs(dataset, doc_topics, cell_ids)
    if res is None:
        print(f"  [{dataset}] skipping gep-x-celltype: no usable labels")
        return None
    (kept_idx, kept_labels, ordered_types, type_count_str,
     type_mean_rank, type_modal_stage) = res
    K = doc_topics.shape[1]
    dom = np.argmax(doc_topics, axis=1)
    dom_kept = dom[kept_idx]

    panel_order, annot = _gep_panel_order_and_annot(
        dataset, doc_topics, cell_ids, K
    )

    matrix = np.zeros((K, len(ordered_types)))
    for j, t in enumerate(ordered_types):
        mask = (kept_labels == t)
        n_t = mask.sum()
        if n_t == 0:
            continue
        for g in range(K):
            matrix[g, j] = ((dom_kept[mask] == g).sum()) / n_t
    matrix = matrix[panel_order, :]

    # Row labels
    row_labels = []
    for g in panel_order:
        if g in annot and "mean rank" in annot.get(g, ""):
            row_labels.append(f"GEP {g} {annot[g]}")
        else:
            row_labels.append(f"GEP {g}")
    col_labels = [type_count_str[t] for t in ordered_types]

    h = max(2.0, 0.55 * K + 1.4)
    w = max(4.0, 0.95 * len(ordered_types) + 2.4)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis",
                   vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks(range(len(ordered_types)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(K))
    ax.set_yticklabels(row_labels, fontsize=8)
    # annotate fractions
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            color = "white" if v < 0.55 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=7, color=color)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="fraction")
    title = f"{dataset} — GEP x cell-type dominance overlap (K={K})"
    subtitle = ("Cell (g,t) = fraction of cells of type t where GEP g "
                "is dominant.")
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if return_matrix:
        return matrix, [f"GEP {g}" for g in panel_order], ordered_types
    return None


def _union_top_genes_in_expr(
    top_genes_per_gep: dict[int, list[str]], expr: pd.DataFrame,
) -> tuple[list[str], dict[str, int]]:
    """Union of top genes across GEPs (first-appearance order), keeping only
    those present in `expr`. Also returns gene -> argmax-GEP map (first GEP
    where the gene appears wins for argmax-GEP-from-topics — we recompute
    below from word_topic_counts for accuracy)."""
    all_top: list[str] = []
    seen: set[str] = set()
    for gep in sorted(top_genes_per_gep.keys()):
        for g in top_genes_per_gep[gep]:
            if g not in seen:
                all_top.append(g)
                seen.add(g)
    present = case_insensitive_select(expr, all_top)
    return present, {}


def figure_gene_x_celltype(
    expr: pd.DataFrame,
    top_genes_per_gep: dict[int, list[str]],
    word_topic_counts: dict[int, dict[str, int]],
    doc_topics: np.ndarray,
    cell_ids: list[str],
    dataset: str,
    out_path: Path,
    *,
    return_stats: bool = False,
):
    """Union-of-top-genes x cell-types, log-mean expression.
    Rows grouped by argmax cell-type. Left strip shows argmax-GEP per gene.
    """
    res = _resolve_celltype_inputs(dataset, doc_topics, cell_ids)
    if res is None:
        print(f"  [{dataset}] skipping gene-x-celltype: no usable labels")
        return None
    (kept_idx, kept_labels, ordered_types, type_count_str,
     _type_mean_rank, _type_modal_stage) = res

    present, _ = _union_top_genes_in_expr(top_genes_per_gep, expr)
    if not present:
        print(f"  [{dataset}] skipping gene-x-celltype: no top genes in expr")
        return None

    sub_expr = expr.iloc[kept_idx, :][present].values  # cells x genes
    log_expr = np.log1p(sub_expr)
    # mean log expression per (gene, type)
    G = len(present)
    T = len(ordered_types)
    mat = np.zeros((G, T))
    for j, t in enumerate(ordered_types):
        mask = (kept_labels == t)
        if mask.any():
            mat[:, j] = log_expr[mask, :].mean(axis=0)

    # each gene's argmax cell-type (for grouping)
    gene_argmax_type_idx = np.argmax(mat, axis=1)
    gene_argmax_type = [ordered_types[j] for j in gene_argmax_type_idx]

    # each gene's argmax GEP from word_topic_counts (independent of expr)
    K = max(word_topic_counts.keys()) + 1
    topic_totals = np.array(
        [sum(word_topic_counts.get(t, {}).values()) for t in range(K)], dtype=float
    )
    gene_argmax_gep: dict[str, int] = {}
    # case-insensitive lookup of wtc gene names
    wtc_genes_lut: dict[str, str] = {}
    for t in range(K):
        for g in word_topic_counts.get(t, {}).keys():
            wtc_genes_lut.setdefault(g.upper(), g)
    for g in present:
        key = wtc_genes_lut.get(g.upper())
        if key is None:
            gene_argmax_gep[g] = -1
            continue
        p = np.zeros(K)
        for t in range(K):
            c = word_topic_counts.get(t, {}).get(key, 0)
            if topic_totals[t] > 0:
                p[t] = c / topic_totals[t]
        gene_argmax_gep[g] = int(np.argmax(p)) if p.sum() > 0 else -1

    # group rows by argmax cell-type (in column order); within group sort by
    # descending mean expression in that dominant cell-type.
    row_order: list[int] = []
    sep_rows: list[int] = []
    cumulative = 0
    for j, t in enumerate(ordered_types):
        idxs = [i for i, gj in enumerate(gene_argmax_type_idx) if gj == j]
        if not idxs:
            continue
        idxs_sorted = sorted(idxs, key=lambda i: -mat[i, j])
        row_order.extend(idxs_sorted)
        cumulative += len(idxs_sorted)
        sep_rows.append(cumulative)
    sep_rows = sep_rows[:-1]  # last boundary is at bottom edge

    mat_ord = mat[row_order, :]
    genes_ord = [present[i] for i in row_order]
    gep_strip = np.array([gene_argmax_gep[g] for g in genes_ord], dtype=int)

    # Plot: left thin strip (GEP categorical) + main heatmap
    h = max(4.0, 0.13 * G + 1.6)
    w = max(5.0, 0.85 * T + 3.0)
    fig, (ax_strip, ax) = plt.subplots(
        1, 2, figsize=(w, h),
        gridspec_kw={"width_ratios": [0.15, w - 0.15], "wspace": 0.02},
    )

    # Build categorical palette (tab10 truncated to K + 1 for missing -1)
    base_palette = plt.get_cmap("tab10").colors
    # map gep id -> color index; -1 (missing) -> gray
    cat_values = sorted(set(gep_strip.tolist()))
    color_list = []
    color_lookup: dict[int, tuple] = {}
    for v in cat_values:
        if v == -1:
            c = (0.7, 0.7, 0.7)
        else:
            c = base_palette[v % len(base_palette)]
        color_lookup[v] = c
        color_list.append(c)
    # map gep ids to consecutive ints for imshow
    value_to_idx = {v: i for i, v in enumerate(cat_values)}
    strip_idx = np.array([value_to_idx[v] for v in gep_strip])[:, None]
    cmap_strip = ListedColormap(color_list)
    bounds = np.arange(len(cat_values) + 1) - 0.5
    norm_strip = BoundaryNorm(bounds, cmap_strip.N)
    ax_strip.imshow(strip_idx, aspect="auto", cmap=cmap_strip, norm=norm_strip,
                    interpolation="nearest")
    ax_strip.set_xticks([])
    ax_strip.set_yticks(range(G))
    ax_strip.set_yticklabels(genes_ord, fontsize=5)
    ax_strip.set_title("GEP", fontsize=7)
    for r in sep_rows:
        ax_strip.axhline(r - 0.5, color="white", linewidth=0.7)

    im = ax.imshow(mat_ord, aspect="auto", cmap="viridis",
                   interpolation="nearest")
    ax.set_xticks(range(T))
    ax.set_xticklabels([type_count_str[t] for t in ordered_types],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks([])
    for r in sep_rows:
        ax.axhline(r - 0.5, color="white", linewidth=0.7)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                 label="mean log(1+count)")

    # GEP legend (one patch per unique GEP)
    handles = []
    from matplotlib.patches import Patch
    for v in cat_values:
        lab = f"GEP {v}" if v != -1 else "no GEP data"
        handles.append(Patch(facecolor=color_lookup[v], label=lab))
    ax.legend(handles=handles, title="dominant GEP",
              loc="center left", bbox_to_anchor=(1.18, 0.5),
              fontsize=7, title_fontsize=8, frameon=False)

    title = (f"{dataset} — top-gene x cell-type expression "
             f"(rows grouped by dominant cell-type)")
    subtitle = ("Left strip = each gene's dominant GEP. Block colour-"
                "alignment with cell-type columns is the non-circular evidence.")
    fig.suptitle(f"{title}\n{subtitle}", fontsize=10)
    try:
        fig.tight_layout(rect=(0, 0.18, 1, 0.95))
    except Exception:
        pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if return_stats:
        return {
            "genes_ord": genes_ord,
            "gene_argmax_type": [gene_argmax_type[i] for i in row_order],
            "gene_argmax_gep": gep_strip,
            "ordered_types": ordered_types,
            "kept_idx": kept_idx,
            "kept_labels": kept_labels,
        }
    return None


def figure_topic_byCellType_dist(
    word_topic_counts: dict[int, dict[str, int]],
    top_genes_per_gep: dict[int, list[str]],
    expr: pd.DataFrame,
    doc_topics: np.ndarray,
    cell_ids: list[str],
    dataset: str,
    out_path: Path,
):
    """Union-of-top-genes x K GEPs of p(gene|GEP), with rows grouped
    by dominant cell-type (computed from raw expression), and within each
    cell-type group ordered by descending p(gene | argmax-GEP from MALLET).
    """
    res = _resolve_celltype_inputs(dataset, doc_topics, cell_ids)
    if res is None:
        print(f"  [{dataset}] skipping topic-byCellType-dist: no labels")
        return None
    (kept_idx, kept_labels, ordered_types, type_count_str,
     _type_mean_rank, _type_modal_stage) = res

    # union of top genes
    all_top: list[str] = []
    seen: set[str] = set()
    for gep in sorted(top_genes_per_gep.keys()):
        for g in top_genes_per_gep[gep]:
            if g not in seen:
                all_top.append(g)
                seen.add(g)

    # restrict to genes present in expr (to compute cell-type assignment)
    present_in_expr = case_insensitive_select(expr, all_top)
    expr_lut = {c.upper(): c for c in expr.columns}
    kept_genes = [g for g in all_top if g.upper() in expr_lut]
    if not kept_genes:
        print(f"  [{dataset}] skipping topic-byCellType-dist: no genes")
        return None

    # compute mean log expression per (gene, type) using only kept genes
    expr_cols = [expr_lut[g.upper()] for g in kept_genes]
    sub_expr = expr.iloc[kept_idx, :][expr_cols].values
    log_expr = np.log1p(sub_expr)
    T = len(ordered_types)
    G = len(kept_genes)
    expr_mat = np.zeros((G, T))
    for j, t in enumerate(ordered_types):
        mask = (kept_labels == t)
        if mask.any():
            expr_mat[:, j] = log_expr[mask, :].mean(axis=0)
    gene_argmax_type_idx = np.argmax(expr_mat, axis=1)

    # build p(gene|GEP) matrix
    K = max(word_topic_counts.keys()) + 1
    topic_totals = np.array(
        [sum(word_topic_counts.get(t, {}).values()) for t in range(K)], dtype=float
    )
    wtc_lut: dict[str, str] = {}
    for t in range(K):
        for g in word_topic_counts.get(t, {}).keys():
            wtc_lut.setdefault(g.upper(), g)
    pmat = np.zeros((G, K))
    for i, g in enumerate(kept_genes):
        key = wtc_lut.get(g.upper())
        if key is None:
            continue
        for t in range(K):
            c = word_topic_counts.get(t, {}).get(key, 0)
            if topic_totals[t] > 0:
                pmat[i, t] = c / topic_totals[t]

    # Column reorder by mean lineage rank if available
    col_order, col_annot = _gep_panel_order_and_annot(
        dataset, doc_topics, cell_ids, K
    )
    pmat_cols = pmat[:, col_order]
    col_labels = []
    for g in col_order:
        if g in col_annot and "mean rank" in col_annot.get(g, ""):
            col_labels.append(f"GEP {g}\n{col_annot[g]}")
        else:
            col_labels.append(f"GEP {g}")

    # each gene's argmax GEP from pmat (independent of expr)
    gene_argmax_gep = np.argmax(pmat, axis=1)

    # group rows by argmax cell-type; within group, sort by descending
    # p(gene | argmax-GEP-of-gene)
    row_order: list[int] = []
    sep_rows: list[int] = []
    cumulative = 0
    for j, t in enumerate(ordered_types):
        idxs = [i for i, gj in enumerate(gene_argmax_type_idx) if gj == j]
        if not idxs:
            continue
        idxs_sorted = sorted(
            idxs, key=lambda i: -pmat[i, gene_argmax_gep[i]]
        )
        row_order.extend(idxs_sorted)
        cumulative += len(idxs_sorted)
        sep_rows.append(cumulative)
    sep_rows = sep_rows[:-1]

    pmat_ord = pmat_cols[row_order, :]
    genes_ord = [kept_genes[i] for i in row_order]
    pmat_ord = np.maximum(pmat_ord, 1e-8)

    h = max(4.0, 0.16 * G + 1.6)
    w = max(4.0, 0.7 * K + 2.4)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(
        pmat_ord, aspect="auto", cmap="magma",
        norm=LogNorm(vmin=max(pmat_ord.min(), 1e-6), vmax=pmat_ord.max()),
        interpolation="nearest",
    )
    ax.set_xticks(range(K))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(G))
    ax.set_yticklabels(genes_ord, fontsize=5)
    for r in sep_rows:
        ax.axhline(r - 0.5, color="white", linewidth=0.7)

    # annotate the cell-type group with text on the right margin
    cumulative = 0
    for j, t in enumerate(ordered_types):
        n_here = int(np.sum(gene_argmax_type_idx == j))
        if n_here == 0:
            continue
        mid = cumulative + n_here / 2 - 0.5
        ax.text(K - 0.3, mid, t, va="center", ha="left",
                fontsize=7, rotation=0)
        cumulative += n_here

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.12, label="p(gene|GEP)")
    title = (f"{dataset} — topic-word p(w|z), rows grouped by dominant "
             f"cell-type")
    ax.set_title(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return None



def run_dataset(
    dataset: str,
    layout: str = "seed0",
    variants: tuple[str, ...] = ALL_VARIANTS,
    log_transform: bool = True,
    legible: bool = False,
) -> list[Path]:
    seed_dir = WS / "outputs" / dataset / layout
    if not seed_dir.exists():
        raise FileNotFoundError(seed_dir)
    expr = pd.read_csv(WS / "data" / dataset / DATASET_FILES[dataset], index_col=0)
    top = load_top_per_gep(seed_dir)
    doc_topics = load_doc_topics(seed_dir)
    cell_ids = load_doc_topic_cell_ids(seed_dir)
    word_topic_counts = parse_word_topic_counts(seed_dir / "word_topic_counts.txt")
    gene_topic_distrib = build_gene_topic_distrib(word_topic_counts)

    fig_dir = WS / "figures" / dataset
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    rng = np.random.RandomState(0)

    for v in variants:
        if v in CELL_VARIANTS:
            if doc_topics is None or cell_ids is None:
                print(f"  [{dataset}] skipping {v}: no doc_topics")
                continue
            out = fig_dir / f"{dataset}_{layout}_top_genes_x_cells__{v}.pdf"
            heatmap_top_genes_x_cells(
                expr, top, doc_topics, out,
                dataset=dataset,
                cell_variant=v,
                cell_ids=cell_ids,
                gene_topic_distrib=gene_topic_distrib,
                log_transform=log_transform,
                rng=np.random.RandomState(0),
                legible=legible,
            )
            written.append(out)
            print(f"  wrote {out}")
        elif v in TOPIC_VARIANTS:
            out = fig_dir / f"{dataset}_{layout}_topic_word_dist__{v}.pdf"
            heatmap_topic_word_distribution(
                word_topic_counts, top, out,
                dataset=dataset, topic_variant=v,
                doc_topics=doc_topics, cell_ids=cell_ids,
                legible=legible,
            )
            written.append(out)
            print(f"  wrote {out}")
        elif v in CELLTYPE_VARIANTS:
            if dataset in NO_CELL_TYPES:
                print(f"  skipping {dataset}: no cell-type labels")
                continue
            if doc_topics is None or cell_ids is None:
                print(f"  [{dataset}] skipping {v}: no doc_topics")
                continue
            if v == "gep-x-celltype":
                out = fig_dir / f"{dataset}_{layout}_gep_x_celltype.pdf"
                figure_gep_x_celltype(
                    doc_topics, cell_ids, dataset, out,
                )
                written.append(out)
                print(f"  wrote {out}")
            elif v == "gene-x-celltype":
                out = fig_dir / f"{dataset}_{layout}_gene_x_celltype.pdf"
                figure_gene_x_celltype(
                    expr, top, word_topic_counts, doc_topics, cell_ids,
                    dataset, out,
                )
                written.append(out)
                print(f"  wrote {out}")
            elif v == "topic-byCellType-dist":
                out = fig_dir / f"{dataset}_{layout}_topic_word_dist__byCellType.pdf"
                figure_topic_byCellType_dist(
                    word_topic_counts, top, expr, doc_topics, cell_ids,
                    dataset, out,
                )
                written.append(out)
                print(f"  wrote {out}")
        else:
            print(f"  [{dataset}] unknown variant {v!r}; skip")
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--layout", default="seed0")
    parser.add_argument("--variant", choices=list(ALL_VARIANTS))
    parser.add_argument("--gep-x-celltype", action="store_true",
                        dest="gep_x_celltype")
    parser.add_argument("--gene-x-celltype", action="store_true",
                        dest="gene_x_celltype")
    parser.add_argument("--topic-byCellType-dist", action="store_true",
                        dest="topic_bycelltype_dist")
    parser.add_argument("--all-celltype", action="store_true",
                        dest="all_celltype",
                        help="Run the three non-circular cell-type figures.")
    parser.add_argument("--no_log", action="store_true")
    parser.add_argument("--legible", action="store_true",
                        help="Full-page supplement sizing: larger gene row "
                             "labels and a taller canvas. This is the sizing "
                             "used for the supplementary heatmaps.")
    args = parser.parse_args()

    if not args.dataset and not args.all_datasets:
        parser.error("specify --dataset or --all-datasets")

    chosen: list[str] = []
    if args.variant:
        chosen.append(args.variant)
    if args.gep_x_celltype:
        chosen.append("gep-x-celltype")
    if args.gene_x_celltype:
        chosen.append("gene-x-celltype")
    if args.topic_bycelltype_dist:
        chosen.append("topic-byCellType-dist")
    if args.all_celltype:
        chosen = list(CELLTYPE_VARIANTS)
    variants = tuple(chosen) if chosen else ALL_VARIANTS
    datasets = ALL_DATASETS if args.all_datasets else [args.dataset]

    total: list[Path] = []
    for ds in datasets:
        print(f"=== {ds} ===")
        try:
            written = run_dataset(
                ds, layout=args.layout, variants=variants,
                log_transform=(not args.no_log),
                legible=args.legible,
            )
            total.extend(written)
        except Exception as e:
            print(f"  ERROR on {ds}: {e!r}")
    print(f"\nTotal PDFs written: {len(total)}")


if __name__ == "__main__":
    main()
