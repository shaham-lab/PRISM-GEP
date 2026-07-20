# Data

Raw single-cell datasets are **not** committed to this repository (they total several GB).
This file lists the public sources and the deterministic preprocessing that regenerates the
inputs the pipeline reads.

## Expected on-disk layout

Each dataset lives under `data/<dataset>/` with a counts matrix and (for cell-type recovery
experiments) a labels file:

```
data/<dataset>/filtered_<dataset>_cells_x_genes.csv   # cells × HVGs (up to 5000), raw counts
data/<dataset>/cell_type_labels.csv                   # one label per cell (optional)
```

## Preprocessing recipe (Appendix §B.1)

Preprocessing is **gene-filter + HVG selection only, with no cell-level QC**:

1. Load the raw counts matrix (genes × cells) from the source below and keep a copy of the
   raw integer counts.
2. Drop genes detected in fewer than 3 cells (`scanpy.pp.filter_genes(adata, min_cells=3)`).
3. Select the top **5000 highly variable genes** with Seurat v3's HVG method,
   `scanpy.pp.highly_variable_genes(adata, n_top_genes=5000, flavor="seurat_v3", layer="counts")`.
   `flavor="seurat_v3"` fits its variance model on the **raw integer counts** (here the `counts`
   layer), so the selection is computed on counts, not on log-normalized values. The producer
   scripts also `normalize_total` + `log1p` a throwaway copy, but because `layer="counts"` is
   passed, that transform is never read by the HVG step and does not change which genes are
   chosen. For datasets whose post-detectability-filter vocabulary is already below 5000 (for
   example `paul15`, V=3451, or `hemogenic_endothelium`, V=4894), `n_top_genes` is capped at the
   vocabulary size and all surviving genes are kept. (`breast_cancer` is a separate case: it is a
   curated 297-gene panel that does not pass through this HVG recipe at all, described under
   "BreastCancer data availability" below.)
4. Subset to the selected genes and write their **raw integer counts** (cells × HVGs) to
   `filtered_<dataset>_cells_x_genes.csv`. The pipeline consumes integer counts, not normalized
   values.

This procedure is deterministic, so the `filtered_*_cells_x_genes.csv` inputs are fully
reproducible from the public sources. The sole exception is `breast_cancer`, a laboratory-generated
panel used as released that has no public source (see "BreastCancer data availability" below).

## Sources

The benchmark is 15 scRNA-seq datasets. The `<dataset>` code name in the first column is the
exact directory name the code expects (`scripts/build_full_metrics_tables.py` resolves the
roster as `ORIGINALS + EXTENSIONS + NEW`). "Source" is the reproducible anchor: either a Python
loader call that downloads the data deterministically, or a public accession. The last two
columns are the cell and gene counts **after** the preprocessing above, i.e. the exact shape of
`filtered_<dataset>_cells_x_genes.csv`. They match Table 1 of the supplement.

| `<dataset>` | Source (loader call or accession) | Organism | Cells | Genes |
|---|---|---|---|---|
| `breast_cancer` | Laboratory-generated patient breast-tumor scRNA-seq, used as released (see "BreastCancer data availability" below) | human | 2,748 | 297 |
| `pbmc3k` | `scanpy.datasets.pbmc3k()` (10x Genomics healthy-donor PBMCs) | human | 2,700 | 5,000 |
| `zeisel_brain` | GEO `GSE60361` (Zeisel et al. 2015, mouse cortex/hippocampus) | mouse | 3,005 | 5,000 |
| `hemogenic_endothelium` | EBI Expression Atlas `E-MTAB-8271` (human PSC differentiation toward haemogenic endothelium) | human | 2,679 | 4,894 |
| `pancreas` | `scvelo.datasets.pancreas()` (Bastidas-Ponce et al. 2019, endocrinogenesis) | mouse | 3,696 | 5,000 |
| `gastrulation` | `scvelo.datasets.gastrulation()` (Pijuan-Sala et al. 2019 atlas), random 8,000-cell subsample, `numpy.random.default_rng(42)` | mouse | 8,000 | 5,000 |
| `gastrulation_e75` | `scvelo.datasets.gastrulation_e75()` (E7.5 stage of the same atlas) | mouse | 7,202 | 5,000 |
| `gastrulation_erythroid` | `scvelo.datasets.gastrulation_erythroid()` (erythroid sub-stream of the same atlas) | mouse | 9,815 | 5,000 |
| `bonemarrow` | `scvelo.datasets.bonemarrow()` (Setty et al. 2019, human bone marrow) | human | 5,780 | 5,000 |
| `paul15` | `scanpy.datasets.paul15()` (Paul et al. 2015, myeloid progenitor differentiation) | mouse | 2,730 | 3,451 |
| `dentategyrus` | `scvelo.datasets.dentategyrus()` (mouse hippocampal dentate-gyrus neurogenesis) | mouse | 2,930 | 5,000 |
| `pbmc68k` | `scvelo.datasets.pbmc68k()` (10x Genomics 68k PBMCs), random 8,000-cell subsample, `numpy.random.default_rng(42)` | human | 8,000 | 5,000 |
| `endoderm_diff` | EBI Expression Atlas `E-MTAB-7008` (human iPSC endoderm differentiation) | human | 1,012 | 4,638 |
| `ventral_neuron_diff` | EBI Expression Atlas `E-GEOD-93593` (hESC differentiation to ventral neural cell types) | human | 1,733 | 4,951 |
| `mouse_hspc` | EBI Expression Atlas `E-GEOD-81682` (mouse hematopoietic stem and progenitor cells) | mouse | 1,919 | 4,823 |

### Source notes

- **`scvelo` / `scanpy` builtins.** The loader call is the reproducible source: it downloads a
  fixed, versioned matrix. For `gastrulation` and `pbmc68k` the full matrix is randomly
  subsampled to 8,000 cells with `numpy.random.default_rng(42)` before preprocessing, which is
  why their cell counts are exactly 8,000. No other dataset is subsampled.
- **EBI Single Cell Expression Atlas datasets** (`hemogenic_endothelium`, `endoderm_diff`,
  `ventral_neuron_diff`, `mouse_hspc`) are fetched with
  `scanpy.datasets.ebi_expression_atlas("<accession>", filter_boring=True)` followed by
  `var_names_make_unique()`. These matrices arrive with Ensembl gene IDs, which are mapped to
  gene symbols through mygene.info (`ensembl.gene` scope) so the vocabulary is symbol-based for
  the GO-BP metrics. When the per-cell count total is unusually high (median row sum above
  20,000) the rows are rescaled to a pseudo-count total of 5,000 before rounding to integers.
- **`zeisel_brain`** is fetched from GEO `GSE60361` and passed through the same 5,000-HVG
  preprocessing as the rest, as is `pbmc3k`. `breast_cancer` is the one exception (see below).

### Screened but not part of the benchmark

Candidate screening during dataset selection also touched EBI accessions `E-HCAD-18`
(`heart_valves`) and `E-MTAB-7324` (`mouse_embryo_chimera`). Neither is one of the 15 benchmark
datasets and neither is scored anywhere in this repository. They are listed here only so a reader
who encounters those accessions in a candidate directory knows they were dropped.

## BreastCancer data availability

`breast_cancer` is the one dataset in the panel that does not originate from a public accession.
It is a laboratory-generated single-cell matrix from a single patient's breast tumor, a curated
panel of 297 genes (uppercase human symbols) across 2,748 cells, used exactly as released. There
is no upstream raw matrix, no GEO / ArrayExpress / ENA accession, and no producer or preprocessing
script for it anywhere in this repository. Unlike every other dataset it therefore does **not** pass
through the HVG recipe above: its 297-gene vocabulary *is* the curated panel, not a top-5000 HVG
selection, which is why it is the smallest vocabulary in the study. It also ships without a
`cell_type_labels.csv`, so cell-type-recovery experiments skip it and its trajectory and heatmap
figures are colored by dominant GEP rather than by published labels. The curated matrix is available
from the corresponding author (Uri Shaham, Bar-Ilan University) upon reasonable request.

## Other reference files (download, not shipped)

- **MGI mouse–human homology** (`HOM_MouseHumanSequence.rpt`, ~15 MB): download from
  [MGI](https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt) if you
  need cross-species gene mapping. Not required for the core scRNA GO-BP pipeline.
- **GO Biological Process**: not needed locally — GO-BP enrichment is computed via the online
  Enrichr API (`GO_Biological_Process_2021` / `2023`) through `gseapy.enrichr`.
