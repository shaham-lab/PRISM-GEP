"""Gene-embedding alternatives for Stage F (within-GEP gene ordering).

The PRISM-GEP Stage F orders the top-N genes of each GEP via a diffusion
map on a gene-gene similarity kernel derived from the Stage-A PPMI matrix.
This module provides a pretrained gene-foundation embedding (scGPT) as an
alternative source of that ordering, for comparison against the default.

This module exposes a single interface:

    embed_genes(gene_names, method, **kwargs) -> (E, idx_map)

returning an embedding matrix `E` of shape (G, D) (one row per gene) and a
dict mapping gene_name -> row index, so any downstream consumer can swap
embeddings without changing its own code.

Methods
-------
- "prism"   : the existing Stage-F diffusion ordering on the PRISM PPMI
              (returns a 2-d coordinate per gene -- the existing Step (ii)
              output).
- "scgpt"   : pretrained scGPT gene embeddings (768-d). Requires
              `scgpt` package + a model checkpoint (e.g. the human
              foundation model at ~1.5GB). Deferred to GPU box; will load
              the checkpoint into CPU memory if `device="cpu"` is forced
              but it is much slower.
- "random"  : sanity baseline — N(0, 1) embedding. Should perform at
              chance on the Step (ii) gene-ordering benchmark.
- "log1p"   : crude expression-based baseline — log1p(mean expression
              per cell type) projected to D dimensions via PCA. Useful as
              a "non-learned but biology-aware" lower bound.

This module is intentionally self-contained: the scGPT path imports lazily
so the rest of the pipeline runs even when scGPT is not installed.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import numpy as np

METHODS = Literal["prism", "scgpt", "random", "log1p"]


def embed_genes(
    gene_names: list[str] | tuple[str, ...],
    method: METHODS = "prism",
    *,
    counts: np.ndarray | None = None,        # (N_cells, G); for "log1p"
    ppmi=None,                                # sp.csr_matrix; for "prism"
    gene_to_idx: dict[str, int] | None = None,  # for "prism"
    scgpt_checkpoint: str | Path | None = None,
    scgpt_vocab: str | Path | None = None,
    scgpt_device: str = "cpu",
    pca_dim: int = 50,
    random_state: int = 42,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return (E, idx_map) where E has shape (len(gene_names), D)."""
    if method == "prism":
        return _embed_prism(gene_names, ppmi, gene_to_idx)
    if method == "scgpt":
        return _embed_scgpt(gene_names, scgpt_checkpoint, scgpt_vocab,
                            scgpt_device)
    if method == "random":
        return _embed_random(gene_names, pca_dim, random_state)
    if method == "log1p":
        if counts is None:
            raise ValueError("'log1p' requires counts=(N_cells, G)")
        return _embed_log1p(gene_names, counts, pca_dim, random_state)
    raise ValueError(f"unknown method {method!r}; choose from {METHODS.__args__}")


def _embed_prism(gene_names, ppmi, gene_to_idx):
    """Stage-A PPMI rows restricted to gene_names. The downstream Step (ii)
    consumer (`bio.gene_ordering.order_genes_in_gep`) takes the PPMI matrix
    plus the gene list and computes the diffusion embedding internally, so
    here we simply return the gene_names-aligned PPMI sub-matrix and a
    self-consistent idx_map for callers that just want the embedding.
    """
    if ppmi is None or gene_to_idx is None:
        raise ValueError("'prism' requires ppmi + gene_to_idx")
    import scipy.sparse as sp
    rows = [gene_to_idx[g] for g in gene_names if g in gene_to_idx]
    if len(rows) != len(gene_names):
        missing = [g for g in gene_names if g not in gene_to_idx]
        warnings.warn(f"gene_to_idx missing {len(missing)} genes; first 5: {missing[:5]}")
    sub = ppmi[rows][:, rows]
    E = sub.toarray() if sp.issparse(sub) else np.asarray(sub)
    idx_map = {g: i for i, g in enumerate(gene_names) if g in gene_to_idx}
    return E.astype(np.float32), idx_map


def _embed_random(gene_names, dim, random_state):
    rng = np.random.default_rng(random_state)
    E = rng.standard_normal((len(gene_names), dim)).astype(np.float32)
    idx_map = {g: i for i, g in enumerate(gene_names)}
    return E, idx_map


def _embed_log1p(gene_names, counts, dim, random_state):
    """log1p mean expression -> PCA(dim)."""
    from sklearn.decomposition import PCA
    X = np.log1p(np.asarray(counts, dtype=np.float32))  # (N, G)
    # Gene representation = its log1p expression column averaged per cell;
    # then project to dim. (Gene-as-row matrix.)
    G = X.T  # (genes, cells)
    G_centered = G - G.mean(axis=0, keepdims=True)
    pca = PCA(n_components=min(dim, min(G_centered.shape) - 1),
              random_state=random_state)
    E = pca.fit_transform(G_centered).astype(np.float32)
    idx_map = {g: i for i, g in enumerate(gene_names)}
    return E, idx_map


def _embed_scgpt(gene_names, checkpoint, vocab, device):
    """Pretrained scGPT gene embeddings (Scope A — static token embeddings).

    Bypasses the upstream ``scgpt`` package entirely so this runs on Windows
    where ``import scgpt`` fails on ``torchtext._extension`` (OSError
    WinError 127). We just need three things from the scGPT release:

      1. ``best_model.pt`` (PyTorch state-dict).
      2. ``vocab.json`` (gene-symbol -> token-id mapping; plain JSON dict).
      3. The location of the gene-embedding matrix inside the state-dict
         (``encoder.embedding.weight`` in the public ``MohamedMabrouk/scGPT``
         mirror of the whole-human foundation model).

    Both files are downloadable from HuggingFace:
        from huggingface_hub import hf_hub_download
        hf_hub_download('MohamedMabrouk/scGPT', 'best_model.pt', local_dir=...)
        hf_hub_download('MohamedMabrouk/scGPT', 'vocab.json',    local_dir=...)
    """
    import json
    import torch
    if checkpoint is None:
        raise ValueError("scgpt requires --scgpt-checkpoint path/to/best_model.pt")
    if vocab is None:
        vocab = Path(checkpoint).parent / "vocab.json"

    # 1. Vocab — scGPT's GeneVocab is just a {symbol: id} dict on disk.
    with open(vocab, "r", encoding="utf-8") as fh:
        gv = json.load(fh)
    # Special tokens; the scGPT vocab uses "<pad>" as the unknown fallback.
    unk_token = "<pad>"
    unk_id = gv.get(unk_token)

    # 2. State-dict + locate the gene-embedding matrix.
    state = torch.load(str(checkpoint), map_location=device,
                       weights_only=False)
    keymap = state.get("model_state_dict", state)
    # Prefer the explicit gene-encoder key; fall back to any *embedding*.weight.
    preferred_keys = [
        "encoder.embedding.weight",
        "gene_encoder.embedding.weight",
        "value_encoder.embedding.weight",
    ]
    key = next((k for k in preferred_keys if k in keymap), None)
    if key is None:
        key = next((k for k in keymap if k.endswith("embedding.weight")), None)
    if key is None:
        raise RuntimeError("could not locate gene-embedding matrix in scgpt "
                            f"checkpoint; keys present: {list(keymap)[:10]}")
    W = keymap[key].cpu().numpy()  # (V_vocab, D)

    D = W.shape[1]
    E = np.zeros((len(gene_names), D), dtype=np.float32)
    missing = []
    idx_map = {}
    for i, g in enumerate(gene_names):
        # scGPT canonicalises gene symbols upper-case.
        sym = g.upper()
        token_id = gv.get(sym, unk_id)
        if token_id == unk_id or token_id is None:
            missing.append(g)
            continue
        E[i] = W[token_id]
        idx_map[g] = i
    if missing:
        warnings.warn(f"scGPT vocab missing {len(missing)} genes "
                      f"(first 5: {missing[:5]}); their rows are zero.")
    return E, idx_map
