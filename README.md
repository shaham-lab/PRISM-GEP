# PRISM-GEP

**Viewing Single-Cell Expression through the Lens of Topic Modeling.**

PRISM-GEP (**PRI**or from scRNA-seq **S**tatistics for **M**odeling **G**ene-**E**xpression
**P**rograms) recovers gene-expression programs (GEPs) from single-cell data by replacing
LDA's flat symmetric prior with a **data-derived Dirichlet prior** estimated from the
dataset's own gene–gene co-expression. The prior needs no external database or reference
dataset; it initializes LDA without altering the generative process, so the fitted model
stays a standard, interpretable topic model.

This repository contains the PRISM-GEP method, its evaluation code, and the generators for
the paper's benchmark tables and the bulk of its figures. It is the PRISM-GEP implementation:
the comparison methods the paper benchmarks against are third-party tools with their own
repositories, and their per-seed scores are supplied directly in `results/` so the benchmark
tables regenerate without them.

> This release ships what is needed to reproduce the results, together with
> [`supplementary.pdf`](supplementary.pdf) — the paper's Supplementary Material, which documents
> the method stages, the metric definitions and the tables these scripts regenerate. The table
> generators emit **CSV files of numbers**, not LaTeX. The rendered manuscript itself and the
> figure PDFs are **not** committed. Raw datasets are **not** committed either — see
> [Data](#data) for accessions and the preprocessing recipe.

---

## Repository layout

```
bio/            The method, Stages A–F: PPMI, diffusion embedding, GMM, method-of-moments β,
                gene ordering, and evaluation (GO-BP, LLM plausibility)
prism_lib/      Vendored LDA-MALLET wrapper + graph utils + method-of-moments
config/         Production run configs (JSON)
mallet_patch/   Eight-file topic-model fork of MALLET 2.0.8 (vector-beta --beta-file prior + 2 bug fixes)
scripts/        Reproduction scripts (pipeline runners, figure/table generators)
results/        Metrics CSVs that back the paper's headline tables (inputs to the generators)
data/           README only — dataset accessions + preprocessing (no raw data)

supplementary.pdf   The paper's Supplementary Material
```

Generated at run time (git-ignored, created on demand):

```
outputs/        Pipeline intermediates: β priors, per-seed MALLET output, baselines
results/tables/ Where the table generators (scripts/build_*.py) write their CSV tables
paper/figures/  Where the figure generators write their PDFs
mallet/         Your patched MALLET checkout (see below)
```

`bio/` and `prism_lib/` are importable top-level packages (the method uses absolute
`from bio.…` / `from prism_lib.…` imports); run scripts from the repository root so both
resolve.

## How PRISM-GEP works (Stages A–F)

The pipeline produces a Dirichlet β prior CSV that is handed to a (patched) MALLET LDA via
`--beta-file`:

| Stage | What | Code |
|---|---|---|
| A | kNN-over-cells PPMI gene–gene affinity | `bio/ppmi_knn_over_cells.py` |
| B | Diffusion-map embedding of genes | inline in `bio/pipeline.py` |
| C | GMM soft clustering + Bayes inversion → per-topic gene distributions | `bio/pipeline.py` |
| D | Method-of-moments Dirichlet β̂ | `prism_lib/methods_of_moments.py` |
| E | LDA inference with the informed prior (patched MALLET) | `prism_lib/ldamallet.py` + `mallet_patch/` |
| F | Within-GEP diffusion-based gene ordering | `bio/gene_ordering.py` |

Evaluation (GO-BP coherence / coverage / strength; LLM plausibility) lives in
`bio/evaluate.py`, `bio/evaluate_paper.py`, `bio/evaluate_supp.py`, `bio/llm_coherence.py`.

---

## Requirements

- Python 3.10+ (tested with the pins in `requirements.txt`: numpy 1.26, scipy 1.13,
  pandas 2.2, scikit-learn 1.6, matplotlib 3.10, seaborn 0.13, gseapy ≥ 1.1.3).
- Java (JDK 8+) and Apache Ant, to build the patched MALLET.
- Optional: an `OPENAI_API_KEY` in the environment for the LLM-plausibility metric
  (`bio/llm_coherence.py`); all other metrics run offline except GO-BP enrichment,
  which calls the online Enrichr API via `gseapy`.

```bash
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -r requirements.txt
```

### Build the patched MALLET

PRISM-GEP runs LDA with [MALLET](https://mimno.github.io/Mallet/) 2.0.8 modified into a small
eight-file topic-model fork. The central change is Stage E's informed prior: `beta` becomes a
length-`V` vector loaded from `--beta-file`, which touches `ParallelTopicModel.java`,
`WorkerRunnable.java`, `TopicInferencer.java`, `MarginalProbEstimator.java`,
`PolylingualTopicModel.java`, `TopicModelDiagnostics.java`, `WeightedTopicModel.java` and
`tui/TopicTrainer.java`. It also carries two upstreamable bug fixes:

1. `printTopicWordWeights` (wrote array references, not weights), and
2. `optimizeBeta`'s `int[V][maxCount]` allocation (~4 GB at V=5000, OOM), replaced with a
   ragged `int[V][maxCountForType+1]` (~2 MB).

See [`mallet_patch/README.md`](mallet_patch/README.md) for the full build steps and the
per-file list.

Check MALLET out **at the repository root as `mallet/`** — `config/*.json` and
`scripts/train_prism_standard.py` use the relative classpath `mallet/class` +
`mallet/lib/mallet-deps.jar`, so any other location will not be found.

```bash
# Obtain MALLET 2.0.8 source (e.g. https://mimno.github.io/Mallet/) into ./mallet
# Overlay the eight modified topic-model files (see mallet_patch/README.md for the per-file list):
cp mallet_patch/*.java                mallet/src/cc/mallet/topics/
cp mallet_patch/tui/TopicTrainer.java mallet/src/cc/mallet/topics/tui/
cd mallet && ant jar && cd ..
# classpath used by the configs: mallet/class : mallet/lib/mallet-deps.jar
```

---

## Reproduce the paper

### 1. Get + preprocess a dataset

See [`data/README.md`](data/README.md) for accessions and the exact recipe. `scripts/preprocess.py`
regenerates each dataset to a counts matrix at
`data/<dataset>/filtered_<dataset>_cells_x_genes.csv` — up to 5000 highly variable genes chosen
by Seurat v3's HVG method fit on raw counts (`flavor="seurat_v3"`, `layer="counts"`), kept as
raw integer counts (Appendix §B.1). Cell-type labels go to `data/<dataset>/cell_type_labels.csv`.

```bash
python scripts/preprocess.py --all           # 12 regenerable datasets
python scripts/preprocess.py pancreas         # a single dataset
python scripts/preprocess.py --list           # regenerable + inherited names
```

`pbmc3k`, `zeisel_brain` and `breast_cancer` were inherited as pre-filtered matrices and have
no producer — ship their `filtered_*_cells_x_genes.csv` as data. Preprocessing needs the
`scanpy` / `scvelo` / `scikit-misc` block in `requirements.txt`.

### 2. Compute the PRISM-GEP β prior (Stages A–D)

Run from the repository root so `bio` / `prism_lib` are importable. **Pass the production
`--expression_threshold 2.0` explicitly** — the script's own default is `0.0`, which yields a
different prior from the published one:

```bash
python -m bio.pipeline --dataset <dataset> --K 5 --m 20 \
    --n_neighbors 15 --n_pca 50 --expression_threshold 2.0 --neighborhood_min_support 1
# → outputs/<dataset>/beta_prism.csv   (one row of V comma-separated floats)
```

All production settings are in `config/prism_pg_expr_prod.json` (knn15, pca50,
expr-threshold 2.0, support 1, m20, K5, GMM n_init 1, method-of-moments, α50, opt-int 10).

### 3. Train LDA with the informed prior (patched MALLET)

`train_prism_standard.py` reads its β from `outputs/candidate_screen/<dataset>/beta_prism.csv`,
so either run `scripts/compute_beta_candidates.py` (which writes that layout directly) or copy
step 2's output into place:

```bash
mkdir -p outputs/candidate_screen/<dataset>
cp outputs/<dataset>/beta_prism.csv outputs/candidate_screen/<dataset>/

python scripts/train_prism_standard.py <dataset> --seeds 0 1 2 3 4 5 6 7 8 9
# α=50, --optimize-interval 10, K=5, 1000 iterations, 10 seeds
# → outputs/<dataset>/seed<N>/{topic_keys,doc_topics,topic_word_weights}.txt
```

### 3b. Train the optimize-interval grid (opt0 — the headline benchmark config)

The two benchmark tables score PRISM-GEP at **`--optimize-interval 0`** ("opt0": the β prior
frozen and used exactly as computed), with the production `--optimize-interval 10` ("opt10",
step 3) reported alongside. `scripts/optimize_interval_grid.py` trains every arm the scorer
reads — PRISM-β and uniform-β MALLET, each at opt0 and opt10 — for the nine core datasets;
`scripts/run_grid_newds_opt0.py` adds the opt0 arms for the six later datasets:

```bash
python scripts/optimize_interval_grid.py     # 9 core datasets x 4 arms x 10 seeds
python scripts/run_grid_newds_opt0.py        # 6 later datasets, opt0 arms x 10 seeds
```

Both read the column-order β from step 2/3 (`outputs/candidate_screen/<dataset>/beta_prism.csv`
or `outputs/<dataset>/beta_prism.csv`), re-align it to MALLET's type-id order themselves, and
write

```
outputs/optimize_interval_grid/{prism_opt0,prism_opt10,uniform_opt0,uniform_opt10}/<ds>/seed<N>/
```

which `build_full_metrics_tables.py` reads directly — no copy step. Resumable: any
(dataset, arm, seed) whose `topic_keys.txt` already exists is skipped. (Both scripts also run
per-arm GO-BP evaluation via `evaluate_all_new_dataset_table.py`, which uses the online
Enrichr API.)

**Flat-layout fallback.** To train a single arm without the full grid, point
`train_prism_standard.py` at a distinct per-arm root (never the bare `outputs/<dataset>/`,
which is the opt10 production slot from step 3):

```bash
python scripts/train_prism_standard.py <dataset> --optimize-interval 0 \
    --out-root outputs/prism_opt0 --seeds 0 1 2 3 4 5 6 7 8 9
# → outputs/prism_opt0/<dataset>/seed<N>/ ; build_full_metrics_tables.py falls back to this layout
```

### 4. Evaluate → tables

The per-seed GO-BP scores for every method in the paper — PRISM-GEP at both optimization
intervals, MALLET, and the comparison methods — ship in `results/full_metrics_perseed.csv`.
The benchmark table generators read that file (and the aggregate it produces,
`results/full_metrics_combined_4config.csv`), so every benchmark table regenerates from a
fresh clone with no training run at all. Each generator writes one or more **CSV** files to
`results/tables/` (created on demand, git-ignored); the exceptions are noted in the table
below.

```bash
python scripts/build_full_metrics_tables.py       # GO-BP coherence / coverage / strength
python scripts/build_split_tables.py              # main benchmark tables (opt0 / opt10)
python scripts/build_unified_table1.py            # combined-table variant + aggregate stats
python scripts/build_robustness_agg_15ds.py       # aggregate robustness table (15 datasets)
python scripts/build_llm_table_all9.py            # LLM-plausibility panel (9 datasets)
```

| Script | Writes |
|---|---|
| `build_full_metrics_tables.py` | `results/full_metrics_combined_4config.csv`, `results/full_metrics_MALLETopt0.csv`, `results/full_metrics_MALLETopt10.csv` (15 datasets x 6 methods x 3 GO-BP metrics, 10-seed means; written to `results/`, not `results/tables/`) |
| `build_split_tables.py` | `results/tables/tab_prism_vs_sota_{9,14,15}ds_{o0,o10}.csv` (tidy rows: dataset, method, metric, mean, sd, mark) and `tab_prism_vs_mallet_{9,14,15}ds_{o0,o10}.csv` (one row per dataset x metric: PRISM-GEP and MALLET mean, sd, and the Welch-significant winner or `tie`) |
| `build_unified_table1.py` | `results/tables/tab_unified_{9,15}ds_opt0.csv` (all six methods in one tidy table) and `results/robustness_aggregates_{9ds,15ds}.csv` (mean rank with bootstrap CI, geometric composite, worst-axis) |
| `build_robustness_agg_15ds.py` | `results/tables/tab_robustness_agg_15ds_vs_specialized.csv` (PRISM-GEP vs the four specialized methods: mean rank with 95% bootstrap CI, geometric composite, worst-axis, last-place count) |
| `build_llm_table_all9.py` | `results/tables/tab_llm_all9.csv` (GPT-4 plausibility per dataset and method, with mean-rank, mean and std summary rows) |

Steps 1–3b above regenerate PRISM-GEP's own numbers end to end. `build_full_metrics_tables.py`
reads the opt0 arms (the headline PRISM and MALLET columns) from the step-3b grid layout and
the opt10 PRISM column from step 3's flat `outputs/<dataset>/seed<N>/`, caching each newly
scored seed back into `results/full_metrics_perseed.csv`.

### 5. Trajectory and gene-embedding tables

The gene-ordering and cell-ordering evaluations read the per-dataset CSVs shipped under
`outputs/trajectory/` and `outputs/gene_embedding_ablation/` (marker orders, ten-seed
summaries, baseline scores), so these tables also regenerate from a fresh clone:

```bash
python scripts/build_combined_main_table.py       # main-text gene + cell ordering table
python scripts/build_traj_gene_9ds_ci.py          # per-dataset gene ordering, seed sd + marker-bootstrap CI
python scripts/build_traj_cell_9ds.py             # cell ordering, 9 datasets
python scripts/traj_cell_merge_table.py           # cell ordering, all labelled datasets + diagnostics
python scripts/build_gene_embed_9ds.py            # gene-embedding comparison, 9 datasets
```

| Script | Writes (under `results/tables/`) |
|---|---|
| `build_combined_main_table.py` | `tab_combined_main.csv` (tidy rows: group, method, dataset, value, sd, range_lo, range_hi, mark; the aggregates appear as the pseudo-datasets `mean`, `median`, `mean_rank`) |
| `build_traj_gene_9ds_ci.py` | `tab_traj_gene_9ds_ci.csv` (dataset, method, rho, s_seed, boot_mean, ci_lo, ci_hi, n_panel) |
| `build_traj_cell_9ds.py` | `tab_traj_cell_9ds.csv` (per-dataset rho per method, mean, median, mean rank) |
| `traj_cell_merge_table.py` | `tab_traj_cell_all.csv` and `tab_traj_cell_all_diagnostics.csv` |
| `build_gene_embed_9ds.py` | `tab_gene_embed_9ds.csv`, plus the long-format `outputs/gene_embedding_ablation/aggregate_metrics_9ds.csv` |

`scripts/build_traj_gene_8ds_supp.py` belongs to the same family but is a **figure**
generator: it draws the main-text gene-trajectory mean/median panel and prints its aggregates
to stdout. It writes no table file.

Conventions shared by every generated CSV: plain numbers only (scores to 3 decimals, GO-BP
strength and mean ranks to 2), no typographic markers; where the paper marks a best or
second-best entry the CSV carries it as a `mark` column (`best` / `second`, empty otherwise);
an empty cell means the value is undefined (no significant enrichment, or no bootstrap
interval), never zero.

### 6. Figures

Figures are still generated here — only the LaTeX table fragments are gone.

| Figure (paper) | Script |
|---|---|
| GO-BP robustness headline (main) | `scripts/make_gobp_robustness_fig.py` |
| LLM plausibility (main) | `scripts/make_llm_win_fig.py` |
| Gene-trajectory mean/median panel (main) | `scripts/build_traj_gene_8ds_supp.py` |
| Native gene trajectories (supp) | `scripts/viz_prism_native_trajectory.py` |
| Cascade heatmaps (supp) | `scripts/viz_cascade_heatmap.py` |
| GEP embedding panels (supp) | `bio/cell_clustering.py` |
| Top-genes / topic-word heatmaps (supp) | `bio/heatmaps.py --legible` |

All figure generators write PDFs to `paper/figures/`, which is created on demand and
git-ignored. No script writes a LaTeX fragment: the table generators of steps 4 and 5 write
CSV to `results/tables/` instead.

The benchmark-summary generators (`make_gobp_robustness_fig.py`, `make_llm_win_fig.py`,
`build_llm_table_all9.py`, `build_robustness_agg_15ds.py`, `build_split_tables.py`,
`build_unified_table1.py`) read only the CSVs in `results/`, so they reproduce the paper's
headline numbers from a fresh clone with no training run. The remaining per-dataset figure
scripts additionally need `outputs/…` from steps 2–3.

The supplementary heatmaps use the full-page sizing, e.g.

```bash
python -m bio.heatmaps --dataset pancreas --variant byGEP --legible
python -m bio.heatmaps --dataset pancreas --variant union --legible
```

**Not generated by this repository.** The following appear in the paper but have no generator
here. Figure 1 and the two reduction diagrams are hand-drawn schematics; the rest were produced
by analysis code kept outside this release:

- the developmental-lineage trajectory panels (`traj_dev_erythroid`, `traj_dev_bonemarrow`)
- the simulation benchmark, activation-cascade and myeloid-marker supplementary panels
- the topic-count sensitivity plot and the held-out perplexity curve
- the four qualitative gene-ordering illustrations

Everything else — every benchmark and trajectory table, the GO-BP robustness and
LLM-plausibility figures, the gene-trajectory panels, the cascade heatmaps, the GEP embedding
panels and the per-program heatmaps — regenerates from this repository.

**Typesetting the tables.** The generators here stop at the numbers. Turning those numbers
into the manuscript's typeset tables is a separate step that happens outside this repository,
in the authors' manuscript project, and produces no result of its own. A reader reproducing
the work wants the values, and those are the CSV files in `results/tables/`.

### Gene-embedding ablation (Stage F swap)

`scripts/gene_embedding_ablation.py` swaps the gene-gene similarity that drives the Step (ii)
within-GEP ordering and re-scores the recovered order against the canonical marker order (the
same `|Spearman rho|` metric as the gene half of the trajectory evaluation). Three of the four
methods run CPU-only from a fresh clone:

```bash
# PRISM (Stage-A PPMI) vs a log1p-PCA baseline vs a random-embedding floor
python scripts/gene_embedding_ablation.py \
    --datasets pancreas gastrulation gastrulation_erythroid hemogenic_endothelium \
    --methods prism log1p random
# → outputs/gene_embedding_ablation/aggregate_metrics.csv
```

The two scGPT arms need the pretrained scGPT whole-human checkpoint, which is **not shipped**
(a large external dependency). Place `best_model.pt` (~207 MB), `vocab.json` and `args.json`
under `data/scgpt/`; all three are downloadable from the HuggingFace mirror `MohamedMabrouk/scGPT`:

```python
from huggingface_hub import hf_hub_download
for f in ("best_model.pt", "vocab.json", "args.json"):
    hf_hub_download("MohamedMabrouk/scGPT", f, local_dir="data/scgpt")
```

- **Static token embeddings** (CPU; reads the checkpoint directly, no `scgpt` package needed):

  ```bash
  python scripts/gene_embedding_ablation.py --methods scgpt \
      --scgpt-checkpoint data/scgpt/best_model.pt --scgpt-vocab data/scgpt/vocab.json
  ```

- **Contextual embeddings** (needs the `scgpt` package and, in practice, a GPU): runs the
  transformer over every cell and averages each marker's contextual token embedding.

  ```bash
  python scripts/scgpt_contextual_ablation.py --model-dir data/scgpt --device cuda
  # → outputs/gene_embedding_ablation/scgpt_contextual_metrics.csv
  ```

  On machines where `torchtext` cannot load its C++ extension, the scorer auto-installs a
  pure-Python vocab shim (`scripts/_torchtext_shim.py`) so `import scgpt` still succeeds.

All four arms require `torch`; the two scGPT arms additionally require the checkpoint above (and,
for the contextual arm, the `scgpt` package). Every arm writes its `|Spearman rho|` per dataset
into `outputs/gene_embedding_ablation/`.

---

## Data

Raw single-cell datasets are **not** included (they total several GB). `data/README.md` lists
each dataset's public source (GEO / EBI ArrayExpress / `scanpy`/`scVelo` builtins) and the
preprocessing command that regenerates the `filtered_*_cells_x_genes.csv` inputs. Datasets
used: pbmc3k, zeisel_brain, breast_cancer, hemogenic_endothelium, pancreas, bonemarrow,
dentategyrus, endoderm_diff, three gastrulation subsets, and the EBI accessions
E-GEOD-81682, E-GEOD-93593, E-HCAD-18, E-MTAB-7008, E-MTAB-7324.

GO-BP enrichment uses the online Enrichr API (`GO_Biological_Process_2021` throughout), so no
local GO database download is required.

---

## Citation

Buznah, Ishon & Shaham, *PRISM-GEP: Viewing Single-Cell Expression through the Lens of Topic
Modeling*.

## License

See [`LICENSE`](LICENSE). The LDA-MALLET wrapper and the patched `ParallelTopicModel.java`
derive from Apache MALLET and are used under its license; see `mallet_patch/README.md`.
