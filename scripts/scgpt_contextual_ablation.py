"""Contextual scGPT gene-embedding ablation.

Runs the scGPT transformer over each cell, averages contextual token
embeddings for each canonical marker gene across cells, then reuses the
same diffusion-map ordering metric as gene_embedding_ablation.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset, SequentialSampler

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS))

# torchtext 0.18 cannot load its C++ extension against torch 2.6, and
# `scgpt/__init__.py` imports it transitively, so any scgpt import dies with
# WinError 127. Install a pure-Python stand-in for the two vocab symbols scgpt
# actually uses. No-op when the real torchtext loads.
from scripts import _torchtext_shim  # noqa: E402

_torchtext_shim.install()

from scripts.gene_embedding_ablation import _cosine_sim, _diffusion_order_1d  # noqa: E402
from scripts.gene_trajectory_baselines import (  # noqa: E402
    MARKERS_GASTRULATION,
    MARKERS_GASTRULATION_ERYTHROID,
    MARKERS_HEMOGENIC,
    MARKERS_PANCREAS,
)

OUT_ROOT = WS / "outputs" / "gene_embedding_ablation"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

TRAJ_ROOT = WS / "outputs" / "trajectory"

DATASETS = {
    "pancreas": WS / "data" / "Pancreas" / "filtered_pancreas_cells_x_genes.csv",
    "gastrulation_erythroid": WS / "data" / "gastrulation_erythroid" / "filtered_gastrulation_erythroid_cells_x_genes.csv",
    "gastrulation": WS / "data" / "gastrulation" / "filtered_gastrulation_cells_x_genes.csv",
    "hemogenic_endothelium": WS / "data" / "hemogenic_endothelium" / "filtered_hemogenic_endothelium_cells_x_genes.csv",
}

MARKERS = {
    "pancreas": MARKERS_PANCREAS,
    "gastrulation": MARKERS_GASTRULATION,
    "gastrulation_erythroid": MARKERS_GASTRULATION_ERYTHROID,
    "hemogenic_endothelium": MARKERS_HEMOGENIC,
}


def counts_path_for(ds: str) -> Path | None:
    """`data/<Ds>/filtered_<ds>_cells_x_genes.csv`, matching the directory
    case-insensitively (some live in `BoneMarrow`, `DentateGyrus`, `Pancreas`)."""
    if ds in DATASETS:
        return DATASETS[ds]
    fname = f"filtered_{ds}_cells_x_genes.csv"
    direct = WS / "data" / ds / fname
    if direct.exists():
        return direct
    for d in (WS / "data").iterdir():
        if d.is_dir() and d.name.lower() == ds.lower():
            cand = d / fname
            if cand.exists():
                return cand
    return None


def markers_from_orders_csv(ds: str):
    """(gene, canonical_rank) pairs from the gene-trajectory orders CSV.

    Same source `gene_embedding_ablation.py` falls back to, so the static and
    contextual passes cannot disagree about the ground-truth order.
    """
    f = TRAJ_ROOT / ds / f"gene_trajectory_{ds}_orders.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    if "gene" not in d.columns or "canonical_rank" not in d.columns:
        return None
    d = d.dropna(subset=["gene", "canonical_rank"])
    return [(str(g), float(r)) for g, r in zip(d["gene"], d["canonical_rank"])]


class CountDataset(Dataset):
    def __init__(self, counts: np.ndarray, gene_ids: np.ndarray, cls_id: int, pad_value: float):
        self.counts = counts
        self.gene_ids = gene_ids
        self.cls_id = cls_id
        self.pad_value = pad_value

    def __len__(self) -> int:
        return self.counts.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.counts[idx]
        nz = np.nonzero(row)[0]
        genes = np.insert(self.gene_ids[nz], 0, self.cls_id)
        values = np.insert(row[nz], 0, self.pad_value)
        return {
            "id": idx,
            "genes": torch.from_numpy(genes).long(),
            "expressions": torch.from_numpy(values.astype(np.float32)),
        }


def load_scgpt_model(model_dir: Path, device: torch.device, use_fast_transformer: bool):
    from scgpt.model import TransformerModel
    from scgpt.tasks.cell_emb import load_pretrained
    from scgpt.tokenizer.gene_tokenizer import GeneVocab

    vocab = GeneVocab.from_file(model_dir / "vocab.json")
    for token in ("<pad>", "<cls>", "<eoc>"):
        if token not in vocab:
            vocab.append_token(token)
    vocab.set_default_index(vocab["<pad>"])

    with open(model_dir / "args.json", "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=cfg["embsize"],
        nhead=cfg["nheads"],
        d_hid=cfg["d_hid"],
        nlayers=cfg["nlayers"],
        nlayers_cls=cfg["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=cfg["dropout"],
        pad_token=cfg["pad_token"],
        pad_value=cfg["pad_value"],
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=use_fast_transformer,
        fast_transformer_backend="flash",
        pre_norm=False,
    )
    state = torch.load(model_dir / "best_model.pt", map_location=device)
    load_pretrained(model, state, verbose=False)
    model.to(device)
    model.eval()
    return model, vocab, cfg


def contextual_marker_embeddings(
    csv: Path,
    marker_genes: list[str],
    model,
    vocab,
    cfg: dict,
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    from scgpt.data_collator import DataCollator

    df = pd.read_csv(csv, index_col=0)
    col_lookup = {c.lower(): c for c in df.columns}
    present = [g for g in marker_genes if g.lower() in col_lookup and g.upper() in vocab]
    if len(present) < 3:
        return np.zeros((0, cfg["embsize"]), dtype=np.float32), present

    genes_in_vocab = [g for g in df.columns if g.upper() in vocab]
    counts = df[genes_in_vocab].values.astype(np.float32, copy=False)
    gene_ids = np.array([vocab[g.upper()] for g in genes_in_vocab], dtype=np.int64)
    target_ids = {vocab[g.upper()]: g for g in present}

    sums = {g: np.zeros(cfg["embsize"], dtype=np.float64) for g in present}
    counts_seen = {g: 0 for g in present}

    dataset = CountDataset(counts, gene_ids, vocab["<cls>"], cfg["pad_value"])
    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab[cfg["pad_token"]],
        pad_value=cfg["pad_value"],
        do_mlm=False,
        do_binning=True,
        max_length=max_length,
        sampling=True,
        keep_first_n_tokens=1,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=collator,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    pad_id = vocab[cfg["pad_token"]]
    use_amp = device.type == "cuda"
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for batch_idx, batch in enumerate(loader, start=1):
            input_gene_ids = batch["gene"].to(device)
            src_key_padding_mask = input_gene_ids.eq(pad_id)
            encoded = model._encode(
                input_gene_ids,
                batch["expr"].to(device),
                src_key_padding_mask=src_key_padding_mask,
            )
            encoded_cpu = encoded.detach().float().cpu().numpy()
            ids_cpu = input_gene_ids.detach().cpu().numpy()
            for token_id, gene in target_ids.items():
                where = ids_cpu == token_id
                if not where.any():
                    continue
                vals = encoded_cpu[where]
                sums[gene] += vals.sum(axis=0)
                counts_seen[gene] += vals.shape[0]
            if batch_idx % 25 == 0:
                print(f"    processed {min(batch_idx * batch_size, len(dataset))}/{len(dataset)} cells")

    embedded = [g for g in present if counts_seen[g] > 0]
    E = np.vstack([(sums[g] / counts_seen[g]).astype(np.float32) for g in embedded])
    missing = [g for g in present if counts_seen[g] == 0]
    if missing:
        print(f"    contextual scGPT saw zero expressed tokens for {missing}")
    return E, embedded


def score_embeddings(E: np.ndarray, genes: list[str], canon_full: dict[str, int]) -> tuple[float, list[str]]:
    sim = _cosine_sim(E)
    coord = _diffusion_order_1d(sim)
    ordered = [genes[i] for i in np.argsort(coord)]
    recovered = {g: i for i, g in enumerate(ordered)}
    x = np.array([recovered[g] for g in genes], dtype=float)
    y = np.array([canon_full[g] for g in genes], dtype=float)
    rho, _ = spearmanr(x, y)
    return abs(float(rho)), ordered


def merge_rows(rows: list[dict], out: Path, append: bool = True) -> None:
    """Write `rows` to `out`.

    NOTE: this used to hard-code `outputs/gene_embedding_ablation/aggregate_metrics.csv`,
    which is the PUBLISHED 4-dataset table. A partial run therefore silently
    replaced published results. The target is now an explicit argument and the
    default (see `main`) is a scratch file, never the published aggregate.
    """
    df_out = pd.DataFrame(rows)
    if append and out.exists():
        old = pd.read_csv(out)
        key_cols = ["dataset", "method"]
        new_keys = set(map(tuple, df_out[key_cols].astype(str).values))
        keep_old = old[
            ~old[key_cols].astype(str).apply(tuple, axis=1).isin(new_keys)
        ]
        df_out = pd.concat([keep_old, df_out], ignore_index=True, sort=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(df_out.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--model-dir", default=str(WS / "data" / "scgpt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1200)
    parser.add_argument("--use-fast-transformer", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip datasets already present in --out (resume).")
    parser.add_argument(
        "--out",
        default=str(OUT_ROOT / "scgpt_contextual_metrics.csv"),
        help="output CSV. Deliberately NOT aggregate_metrics.csv, which holds "
             "the published 4-dataset table.",
    )
    parser.add_argument(
        "--no-append", action="store_true",
        help="overwrite --out instead of merging on (dataset, method).",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    published = (OUT_ROOT / "aggregate_metrics.csv").resolve()
    if out_path.resolve() == published:
        parser.error(
            "refusing to write aggregate_metrics.csv (the published 4-dataset "
            "table). Pass a different --out and consolidate deliberately."
        )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"loading scGPT from {args.model_dir} on {device}")
    if device.type == "cuda":
        print(f"cuda device: {torch.cuda.get_device_name(0)}")
    model, vocab, cfg = load_scgpt_model(Path(args.model_dir), device, args.use_fast_transformer)

    done = set()
    if args.skip_existing and out_path.exists():
        prev = pd.read_csv(out_path)
        done = set(prev.loc[prev["method"] == "scgpt_contextual", "dataset"])
        if done:
            print(f"resuming: {sorted(done)} already in {out_path.name}")

    rows = []
    for ds in args.datasets:
        if ds in done:
            print(f"\n[{ds}] SKIP -- already in {out_path.name}")
            continue
        print(f"\n[{ds}] contextual scGPT")
        # Re-seed per dataset. DataCollator runs with sampling=True, so cells
        # with more expressed genes than --max-length get a RANDOM token
        # subset. Seeding once per run would make every dataset's result depend
        # on how much RNG the preceding datasets consumed, so a partial re-run
        # could not reproduce it. Per-dataset seeding makes each row
        # independently reproducible.
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        marker_pairs = MARKERS.get(ds)
        if marker_pairs is None:
            marker_pairs = markers_from_orders_csv(ds)
            if marker_pairs is None:
                print(f"  no canonical marker order for {ds} -- SKIP")
                continue
            print(f"  canonical order read from orders CSV ({len(marker_pairs)} markers)")
        csv_path = counts_path_for(ds)
        if csv_path is None:
            print(f"  counts CSV for {ds} not found under data/ -- SKIP")
            continue
        marker_genes = [g for g, _ in marker_pairs]
        canon_full = {g: r for g, r in marker_pairs}
        E, genes = contextual_marker_embeddings(
            csv_path,
            marker_genes,
            model,
            vocab,
            cfg,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
        )
        if len(genes) < 3:
            print(f"  <3 embedded marker genes -- SKIP")
            continue
        rho_abs, ordered = score_embeddings(E, genes, canon_full)
        n_drop = len(marker_genes) - len(genes)
        print(f"  scgpt_contextual |rho|={rho_abs:.3f} n={len(genes)} "
              f"dropped={n_drop} order={ordered[:5]}...")
        rows.append({
            "dataset": ds,
            "method": "scgpt_contextual",
            "spearman_abs": rho_abs,
            "n_genes": len(genes),
            "n_dropped": n_drop,
        })
        # Checkpoint after every dataset. These runs take hours on this GPU and
        # a kill during the last dataset used to discard everything.
        merge_rows(rows, out_path, append=not args.no_append)


if __name__ == "__main__":
    main()
