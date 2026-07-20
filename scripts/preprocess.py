"""Regenerate the per-dataset counts matrices the PRISM-GEP pipeline reads.

For each benchmark dataset this writes

    data/<dataset>/filtered_<dataset>_cells_x_genes.csv   # cells x <=5000 HVGs, raw integer counts
    data/<dataset>/cell_type_labels.csv                   # cell_id, cell_type (when a label column exists)
    data/<dataset>/dataset_summary.txt                    # provenance snapshot

from the public source, using one deterministic 5000-HVG recipe (Appendix B.1).

The 5000-HVG recipe (identical for every dataset)
-------------------------------------------------
1. Load the raw counts matrix from the public source and stash it in
   ``adata.layers["counts"]``.
2. ``sc.pp.filter_genes(adata, min_cells=3)`` -- drop genes seen in < 3 cells.
3. Select the top ``min(5000, n_genes)`` highly variable genes with Seurat v3's
   HVG method. Selection is computed on the RAW COUNTS: scanpy's
   ``flavor="seurat_v3"`` reads the ``layer="counts"`` matrix, not ``.X``. The
   ``normalize_total`` + ``log1p`` on the working copy's ``.X`` is a no-op with
   respect to gene selection when ``layer=`` is passed -- it is kept only so the
   working AnnData carries a log-normalized ``.X`` for any downstream inspection.
   (This is the correction to the older "Seurat v3 on log1p" wording: the HVG
   model is fit on raw counts; the selection is then applied to raw counts.)
4. Subset to the selected genes, round the raw counts to integers, and write
   ``cells x genes`` to ``filtered_<dataset>_cells_x_genes.csv`` with a leading
   ``"Cell number"`` column of cell ids.

The recipe requires ``scanpy`` with the ``scikit-misc`` (``skmisc.loess``) backend
that ``flavor="seurat_v3"`` needs, plus ``scvelo`` for the builtin loaders; see
``requirements.txt``.

Dataset coverage
----------------
This regenerates 12 of the 15 benchmark datasets from code. The three ORIGINAL
datasets ``pbmc3k``, ``zeisel_brain`` and ``breast_cancer`` were inherited as
pre-filtered counts matrices and have no producer script -- ship their
``filtered_*_cells_x_genes.csv`` as data (see ``data/README.md``).

Usage
-----
    python scripts/preprocess.py --all
    python scripts/preprocess.py pancreas hemogenic_endothelium
    python scripts/preprocess.py --list

Run from the repository root. Outputs land under ``data/`` (git-ignored). EBI
Atlas datasets fetch Ensembl-ID matrices and map them to gene symbols via
mygene.info (needed because the GO-BP evaluator uses symbol-based Enrichr
libraries); the mapping is cached under ``outputs/gene_symbol_cache/``.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import data_dir, outputs_dir  # noqa: E402

N_HVG = 5000

# The three ORIGINAL datasets have no producer: they were inherited pre-filtered
# and are shipped as data. Asking to regenerate them is an error, not a silent skip.
INHERITED = ("pbmc3k", "zeisel_brain", "breast_cancer")


# --------------------------------------------------------------------------- #
# Dataset registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Spec:
    name: str
    recipe: str                       # "builtin" | "ebi"
    source: str                       # loader dotted-path, or EBI accession
    label_candidates: tuple[str, ...] = ()
    max_cells: int | None = None      # deterministic downsample (rng seed 42)
    organism: str = "human"           # EBI: species for Ensembl->symbol mapping
    var_names_make_unique: bool = True
    counts_source: str = "layers"     # "layers" (counts/spliced/raw/X) | "X" (adata.X only)
    round_mode: str = "rint"          # "rint" (round-half-to-even) | "truncate"
    ebi_row_rescale: bool = False     # EBI: rescale rows with huge library sizes
    ebi_drop_zero_genes: bool = False # EBI: drop HVGs that round to all-zero


SPECS: dict[str, Spec] = {
    # --- scvelo / scanpy builtins (prepare_candidate_datasets.py) ------------
    "dentategyrus": Spec(
        "dentategyrus", "builtin", "scvelo.datasets.dentategyrus",
        label_candidates=("clusters", "cell_type", "celltype", "CellType"),
    ),
    "gastrulation": Spec(
        "gastrulation", "builtin", "scvelo.datasets.gastrulation",
        label_candidates=("stage", "celltype", "cell_type", "clusters"),
        max_cells=8000,
    ),
    "gastrulation_e75": Spec(
        "gastrulation_e75", "builtin", "scvelo.datasets.gastrulation_e75",
        label_candidates=("celltype", "cluster", "stage", "clusters"),
    ),
    "gastrulation_erythroid": Spec(
        "gastrulation_erythroid", "builtin", "scvelo.datasets.gastrulation_erythroid",
        label_candidates=("stage", "celltype", "cell_type", "clusters"),
    ),
    "pbmc68k": Spec(
        "pbmc68k", "builtin", "scvelo.datasets.pbmc68k",
        label_candidates=("celltype", "cell_type", "clusters", "bulk_labels"),
        max_cells=8000,
    ),
    "paul15": Spec(
        "paul15", "builtin", "scanpy.datasets.paul15",
        label_candidates=("paul15_clusters", "clusters", "cell_type"),
    ),
    # --- scvelo builtins, "fetch" variant (fetch_new_datasets.py) ------------
    # Same HVG core; these two were produced without var-name uniquing and with
    # truncating int cast rather than round-half-to-even.
    "pancreas": Spec(
        "pancreas", "builtin", "scvelo.datasets.pancreas",
        label_candidates=("clusters",),
        var_names_make_unique=False, counts_source="X", round_mode="truncate",
    ),
    "bonemarrow": Spec(
        "bonemarrow", "builtin", "scvelo.datasets.bonemarrow",
        label_candidates=("clusters", "cell_type", "celltype", "celltypes", "Cell_Type"),
        var_names_make_unique=False, counts_source="X", round_mode="truncate",
    ),
    # --- EBI Single Cell Expression Atlas (prepare_ebi_candidate_datasets.py) -
    "endoderm_diff": Spec(
        "endoderm_diff", "ebi", "E-MTAB-7008",
        label_candidates=("cell type", "developmental stage", "time", "day", "protocol", "cluster"),
        organism="human", ebi_row_rescale=True, ebi_drop_zero_genes=True,
    ),
    "ventral_neuron_diff": Spec(
        "ventral_neuron_diff", "ebi", "E-GEOD-93593",
        label_candidates=("cell type", "day", "time", "developmental stage", "cluster"),
        organism="human", ebi_row_rescale=True, ebi_drop_zero_genes=True,
    ),
    "hemogenic_endothelium": Spec(
        "hemogenic_endothelium", "ebi", "E-MTAB-8271",
        label_candidates=("cell type", "day", "time", "cluster", "developmental stage"),
        organism="human", ebi_row_rescale=True, ebi_drop_zero_genes=True,
    ),
    "mouse_hspc": Spec(
        "mouse_hspc", "ebi", "E-GEOD-81682",
        label_candidates=("cell type", "cell population", "phenotype", "cluster", "age"),
        organism="mouse", ebi_row_rescale=True, ebi_drop_zero_genes=True,
    ),
}


assert all(name == spec.name for name, spec in SPECS.items()), \
    "SPECS key must equal Spec.name (it sets the data/<name>/ output path)"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _dense(x) -> np.ndarray:
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def _resolve_loader(path: str) -> Callable:
    module_name, func_name = path.rsplit(".", 1)
    if module_name == "scvelo.datasets":
        import scvelo as scv
        return getattr(scv.datasets, func_name)
    if module_name == "scanpy.datasets":
        import scanpy as sc
        return getattr(sc.datasets, func_name)
    raise ValueError(f"Unsupported loader module: {module_name}")


def _counts_from_layers(adata) -> np.ndarray:
    for key in ("counts", "raw_counts", "spliced"):
        if key in adata.layers:
            return adata.layers[key]
    if adata.raw is not None:
        return adata.raw.X
    return adata.X


def _clean_obs_name(col: str) -> str:
    return col.lower().replace("_", " ").replace("-", " ")


def _pick_label_builtin(adata, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in adata.obs.columns and adata.obs[col].nunique(dropna=True) > 1:
            return col
    for col in adata.obs.columns:
        if 1 < adata.obs[col].nunique(dropna=True) <= 50:
            return col
    return None


def _pick_label_ebi(adata, keywords: tuple[str, ...]) -> str | None:
    ranked = []
    for col in adata.obs.columns:
        nunique = adata.obs[col].nunique(dropna=True)
        if not (1 < nunique <= 80):
            continue
        clean = _clean_obs_name(col)
        score = sum(10 for kw in keywords if kw in clean)
        if "ontology term" in clean:
            score -= 3
        if "organism" in clean or "individual" in clean or "single cell identifier" in clean:
            score -= 5
        ranked.append((score, -nunique, col))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][2]


def _hvg_mask(adata, n_hvg: int) -> np.ndarray:
    """Seurat-v3 HVG selection on raw counts (layer='counts'). Returns a boolean mask."""
    import scanpy as sc
    adata_norm = adata.copy()
    sc.pp.normalize_total(adata_norm, target_sum=1e4)
    sc.pp.log1p(adata_norm)
    n_top = min(n_hvg, adata_norm.n_vars)
    sc.pp.highly_variable_genes(
        adata_norm, n_top_genes=n_top, flavor="seurat_v3", layer="counts",
    )
    return adata_norm.var["highly_variable"].values


def _map_ensembl_to_symbols(genes: list[str], organism: str) -> list[str]:
    """Map Ensembl gene IDs to symbols via mygene.info (cached). GO-BP Enrichr
    libraries are symbol-based, so EBI Atlas Ensembl-ID matrices must be mapped."""
    import requests
    if not genes or not all(
        g.startswith(("ENSG", "ENSMUSG")) for g in genes[: min(50, len(genes))]
    ):
        return genes

    cache_dir = outputs_dir() / "gene_symbol_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha1("\n".join(genes).encode("utf-8")).hexdigest()[:16]
    cache_path = cache_dir / f"{organism}_{len(genes)}_{cache_key}.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        symbols = cached["symbol"].astype(str).tolist()
        unmapped = sum(s.startswith(("ENSG", "ENSMUSG")) for s in symbols)
        if unmapped < len(symbols) * 0.5:
            return symbols
        print(f"  ignoring stale symbol cache with {unmapped}/{len(symbols)} unmapped IDs")

    species = "human" if organism == "human" else "mouse"
    mapping: dict[str, str] = {}
    print(f"  mapping {len(genes)} Ensembl IDs to {species} symbols")
    for i in range(0, len(genes), 50):
        chunk = genes[i : i + 50]
        try:
            response = requests.get(
                "https://mygene.info/v3/query",
                params={
                    "q": " OR ".join(chunk),
                    "scopes": "ensembl.gene",
                    "fields": "symbol,ensembl.gene",
                    "species": species,
                    "size": len(chunk),
                },
                timeout=60,
            )
            response.raise_for_status()
            for item in response.json().get("hits", []):
                symbol = item.get("symbol")
                ensembl = item.get("ensembl")
                if not symbol or not ensembl:
                    continue
                ids = [ensembl.get("gene")] if isinstance(ensembl, dict) else [
                    entry.get("gene") for entry in ensembl if isinstance(entry, dict)
                ]
                for query in ids:
                    if query:
                        mapping[str(query)] = str(symbol)
        except Exception as exc:
            print(f"  symbol mapping failed for chunk {i // 50}: {type(exc).__name__}: {exc}")
        time.sleep(0.2)
    print(f"  mapped {len(mapping)}/{len(genes)} Ensembl IDs")

    out, seen = [], {}
    for gene in genes:
        symbol = mapping.get(gene, gene)
        count = seen.get(symbol, 0)
        seen[symbol] = count + 1
        out.append(symbol if count == 0 else f"{symbol}_{count + 1}")
    pd.DataFrame({"ensembl": genes, "symbol": out}).to_csv(cache_path, index=False)
    return out


def _round_counts(counts: np.ndarray, mode: str) -> np.ndarray:
    counts = _dense(counts).astype(np.float64)
    counts = np.maximum(counts, 0)
    if mode == "truncate":
        return counts.astype(np.int64)
    return np.rint(counts).astype(np.int64)


def _write_outputs(spec: Spec, adata, counts: np.ndarray,
                   gene_names: list[str], label_col: str | None) -> None:
    out_dir = data_dir() / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"filtered_{spec.name}_cells_x_genes.csv"

    df = pd.DataFrame(counts, columns=gene_names)
    df.insert(0, "Cell number", list(adata.obs_names))
    df.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}  ({df.shape[0]} cells x {df.shape[1] - 1} genes)")

    if label_col is not None:
        labels = pd.DataFrame({
            "cell_id": list(adata.obs_names),
            "cell_type": [str(x) for x in adata.obs[label_col].values],
        })
        labels.to_csv(out_dir / "cell_type_labels.csv", index=False)
        print(f"  wrote labels from '{label_col}': {labels['cell_type'].nunique()} classes")
    else:
        print("  no label column found")

    with (out_dir / "dataset_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"name: {spec.name}\n")
        f.write(f"source: {spec.source}\n")
        f.write(f"recipe: {spec.recipe}\n")
        f.write(f"cells: {adata.n_obs}\n")
        f.write(f"genes: {counts.shape[1]}\n")
        f.write(f"label_col: {label_col}\n")
        if label_col is not None:
            f.write("label_counts:\n")
            for label, count in adata.obs[label_col].astype(str).value_counts().items():
                f.write(f"  {label}: {count}\n")


# --------------------------------------------------------------------------- #
# Recipes
# --------------------------------------------------------------------------- #
def _run_builtin(spec: Spec) -> None:
    import scanpy as sc
    loader = _resolve_loader(spec.source)
    adata = loader()
    if spec.var_names_make_unique:
        adata.var_names_make_unique()
    print(f"  loaded {adata.shape}  obs={list(adata.obs.columns)}")

    if spec.max_cells is not None and adata.n_obs > spec.max_cells:
        rng = np.random.default_rng(42)
        idx = rng.choice(adata.n_obs, size=spec.max_cells, replace=False)
        adata = adata[idx].copy()
        print(f"  downsampled to {adata.shape} (rng seed 42)")

    if "counts" not in adata.layers:
        src = adata.X if spec.counts_source == "X" else _counts_from_layers(adata)
        adata.layers["counts"] = src.copy()
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"  after filter_genes(min_cells=3): {adata.shape}")

    adata = adata[:, _hvg_mask(adata, N_HVG)].copy()
    print(f"  after HVG: {adata.shape}")

    counts = _round_counts(adata.layers["counts"], spec.round_mode)
    label_col = _pick_label_builtin(adata, spec.label_candidates)
    _write_outputs(spec, adata, counts, list(adata.var_names), label_col)


def _run_ebi(spec: Spec) -> None:
    import scanpy as sc
    adata = sc.datasets.ebi_expression_atlas(spec.source, filter_boring=True)
    if spec.var_names_make_unique:
        adata.var_names_make_unique()
    print(f"  loaded {adata.shape}  obs={list(adata.obs.columns)}")

    if spec.max_cells is not None and adata.n_obs > spec.max_cells:
        rng = np.random.default_rng(42)
        idx = rng.choice(adata.n_obs, size=spec.max_cells, replace=False)
        adata = adata[idx].copy()
        print(f"  downsampled to {adata.shape} (rng seed 42)")

    adata.layers["counts"] = _counts_from_layers(adata).copy()
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"  after filter_genes(min_cells=3): {adata.shape}")

    adata = adata[:, _hvg_mask(adata, N_HVG)].copy()
    print(f"  after HVG: {adata.shape}")

    counts = _dense(adata.layers["counts"]).astype(np.float64)
    counts = np.maximum(counts, 0)
    if spec.ebi_row_rescale:
        row_sums = counts.sum(axis=1)
        if float(np.nanmedian(row_sums)) > 20000:
            print("  high magnitudes; rescaling rows to pseudo-count total 5000")
            counts = counts * (5000.0 / np.maximum(row_sums, 1.0))[:, None]
    counts = np.rint(counts).astype(np.int64)

    if spec.ebi_drop_zero_genes:
        nonzero = counts.sum(axis=0) > 0
        if not np.all(nonzero):
            print(f"  dropping {int((~nonzero).sum())} zero-count HVGs after rounding")
            counts = counts[:, nonzero]
            adata = adata[:, nonzero].copy()

    gene_names = _map_ensembl_to_symbols(list(adata.var_names), spec.organism)
    label_col = _pick_label_ebi(adata, spec.label_candidates)
    _write_outputs(spec, adata, counts, gene_names, label_col)


def preprocess(name: str, *, force: bool) -> None:
    if name in INHERITED:
        raise ValueError(
            f"{name!r} is an inherited pre-filtered dataset with no producer; "
            f"ship its filtered_{name}_cells_x_genes.csv as data (see data/README.md)."
        )
    if name not in SPECS:
        raise KeyError(f"unknown dataset {name!r}; --list to see the {len(SPECS)} regenerable names")
    spec = SPECS[name]
    out_csv = data_dir() / spec.name / f"filtered_{spec.name}_cells_x_genes.csv"
    if out_csv.exists() and not force:
        print(f"[skip] {spec.name}: {out_csv} exists (use --force to overwrite)")
        return
    print(f"\n=== {spec.name}  ({spec.recipe}: {spec.source}) ===")
    if spec.recipe == "ebi":
        _run_ebi(spec)
    else:
        _run_builtin(spec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="*", help="dataset names (default: all regenerable)")
    parser.add_argument("--all", action="store_true", help="preprocess every regenerable dataset")
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    parser.add_argument("--list", action="store_true", help="list regenerable + inherited datasets and exit")
    args = parser.parse_args()

    if args.list:
        print(f"Regenerable ({len(SPECS)}):")
        for name, spec in SPECS.items():
            print(f"  {name:24s} {spec.recipe:8s} {spec.source}")
        print(f"\nInherited (ship as data, no producer): {', '.join(INHERITED)}")
        return

    names = list(SPECS) if (args.all or not args.datasets) else args.datasets
    for name in names:
        try:
            preprocess(name, force=args.force)
        except Exception as exc:
            print(f"[error] {name}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
