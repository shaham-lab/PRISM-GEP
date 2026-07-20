"""Train PRISM-GEP with the production config (alpha=50, optimize-interval=10).

Writes to the standard per-dataset layout used by the plotting scripts:

  outputs/<dataset>/seed<N>/{topic_keys, word_topic_counts, doc_topics,
                              topic_word_weights}.txt

The beta prior is taken from whichever prior-building entry point was used:
outputs/candidate_screen/<dataset>/ (scripts/compute_beta_candidates.py) or
outputs/<dataset>/ (python -m bio.pipeline). Build one of them first.

The .ser InstanceList is built into ground_truth_ser/<dataset>.ser on first
run and reused.

Resumable: SKIPs any (dataset, seed) whose topic_keys.txt is already present.

Usage:
    python scripts/train_prism_standard.py hemogenic_endothelium gastrulation \
        --seeds 0 1 2 3 4 5 6 7 8 9
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path(__file__).resolve().parent.parent


def csv_path(dataset: str) -> Path:
    return WS / "data" / dataset / f"filtered_{dataset}_cells_x_genes.csv"


def beta_path(dataset: str) -> Path:
    # Prefer the alphabet-aligned beta (first-encounter / MALLET type-id order, with
    # a .genes.tsv sidecar). The bare beta_prism.csv files written before the reorder
    # fix are in data-CSV COLUMN order, which MALLET (reading --beta-file positionally)
    # maps to the WRONG genes (~all of them on the 5000-HVG datasets).
    #
    # Both prior-building entry points are accepted, in this order:
    #   scripts/compute_beta_candidates.py -> outputs/candidate_screen/<ds>/
    #   python -m bio.pipeline             -> outputs/<ds>/
    screen = WS / "outputs" / "candidate_screen" / dataset
    direct = WS / "outputs" / dataset
    for cand in (screen / "beta_prism_alpha.csv", screen / "beta_prism.csv",
                 direct / "beta_prism_alpha.csv", direct / "beta_prism.csv"):
        if cand.exists():
            return cand
    return screen / "beta_prism.csv"      # reported by the not-found error in train()


def assert_beta_aligned(beta: Path, dataset: str) -> None:
    """Refuse a non-uniform beta that is not provably in MALLET first-encounter order.

    MALLET consumes ``--beta-file`` POSITIONALLY by type-id (= first-encounter order of
    the token stream), not by data-CSV column order. A non-uniform beta written in
    column order silently attaches ~every gene's prior weight to the wrong gene, turning
    the informative prior into noise. ``bio.pipeline.write_beta_csv`` always emits a
    ``<beta>.genes.tsv`` sidecar when it reorders correctly, so a non-uniform beta
    lacking that sidecar is a stale column-ordered file and is rejected. A constant
    (uniform) beta is order-invariant and always allowed.
    """
    vals = np.atleast_1d(np.loadtxt(beta, delimiter=",").ravel())
    if vals.size == 0 or np.allclose(vals, vals.flat[0]):
        return  # uniform / constant beta: order does not matter
    sidecar = beta.with_suffix(beta.suffix + ".genes.tsv")
    if not sidecar.exists():
        raise RuntimeError(
            f"[{dataset}] refusing non-uniform beta without an alphabet-aligned "
            f".genes.tsv sidecar:\n  {beta}\n"
            "MALLET reads --beta-file positionally (first-encounter / type-id order), "
            "NOT data-CSV column order. Files without the sidecar predate the reorder "
            "fix and are column-ordered -> the prior lands on the wrong genes. "
            "Regenerate with scripts/compute_beta_candidates.py (which writes the "
            "sidecar via bio.pipeline.write_beta_csv) or pass the *_alpha.csv variant.")


def ser_path(dataset: str) -> Path:
    return WS / "ground_truth_ser" / f"{dataset}.ser"


def tsv_path(dataset: str) -> Path:
    return WS / "ground_truth_ser" / f"{dataset}.tsv"


def csv_to_tsv(dataset: str, *, force: bool) -> Path:
    csv = csv_path(dataset)
    if not csv.exists():
        raise FileNotFoundError(csv)
    tsv = tsv_path(dataset)
    if tsv.exists() and not force:
        return tsv
    tsv.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{dataset}] reading {csv}")
    df = pd.read_csv(csv, index_col=0)
    cells = df.index.astype(str).tolist()
    genes = [re.sub(r"\s+", "_", str(g).strip()) for g in df.columns]
    counts = np.rint(np.maximum(df.values, 0)).astype(np.int64)
    print(f"[{dataset}] writing {tsv}  ({counts.shape[0]} cells x {counts.shape[1]} genes)")
    with tsv.open("w", encoding="utf-8") as fh:
        for i, cell in enumerate(cells):
            nz = np.nonzero(counts[i])[0]
            tokens = []
            for j in nz:
                c = int(counts[i, j])
                if c > 0:
                    tokens.extend([genes[j]] * c)
            fh.write(f"{cell}\tcell\t{' '.join(tokens)}\n")
    return tsv


def run_csv2vectors(dataset: str, tsv: Path, *, force: bool) -> Path:
    ser = ser_path(dataset)
    if ser.exists() and not force:
        return ser
    cp = os.pathsep.join(["mallet/class", "mallet/lib/mallet-deps.jar"])
    cmd = [
        "java", "-Xmx4G", "-cp", cp,
        "cc.mallet.classify.tui.Csv2Vectors",
        "--input", str(tsv),
        "--output", str(ser),
        "--keep-sequence",
        "--token-regex", r"[^\s]+",
        "--preserve-case", "true",
    ]
    print(f"[{dataset}] Csv2Vectors -> {ser}")
    proc = subprocess.run(cmd, cwd=WS, text=True, capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"Csv2Vectors failed for {dataset}")
    return ser


def train(dataset: str, seed: int, *, iterations: int, alpha: float,
          optimize_interval: int, num_topics: int = 5,
          beta_override: Path | None = None,
          out_root: Path | None = None) -> bool:
    ser = ser_path(dataset)
    beta = beta_override if beta_override is not None else beta_path(dataset)
    if not ser.exists():
        raise FileNotFoundError(ser)
    if not beta.exists():
        raise FileNotFoundError(
            f"No beta prior found for '{dataset}'. Build one first with either\n"
            f"  python -m bio.pipeline --dataset {dataset}\n"
            f"  python scripts/compute_beta_candidates.py {dataset}\n"
            f"(searched outputs/candidate_screen/{dataset}/ and outputs/{dataset}/)")
    assert_beta_aligned(beta, dataset)

    if out_root is None:
        out_root = WS / "outputs" / dataset
    out_dir = out_root / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = out_dir / "topic_keys.txt"
    if keys.exists():
        print(f"[{dataset}] seed={seed} SKIP (topic_keys.txt exists)")
        return False

    cp = os.pathsep.join(["mallet/class", "mallet/lib/mallet-deps.jar"])
    cmd = [
        "java", "-Xmx4G", "-cp", cp,
        "cc.mallet.topics.tui.TopicTrainer",
        "--input", str(ser),
        "--num-topics", str(num_topics),
        "--num-iterations", str(iterations),
        "--random-seed", str(seed),
        "--alpha", str(alpha),
        "--optimize-interval", str(optimize_interval),
        "--topic-word-weights-file", str(out_dir / "topic_word_weights.txt"),
        "--output-topic-keys", str(keys),
        "--word-topic-counts-file", str(out_dir / "word_topic_counts.txt"),
        "--output-doc-topics", str(out_dir / "doc_topics.txt"),
        "--beta-file", str(beta),
    ]
    print(f"[{dataset}] training seed={seed}  (alpha={alpha}, opt={optimize_interval}, iters={iterations})")
    proc = subprocess.run(cmd, cwd=WS, text=True, capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"TopicTrainer failed for {dataset}, seed={seed}")
    tail = "\n".join((proc.stdout or "").splitlines()[-3:])
    if tail:
        print(tail)
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("datasets", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=50.0)
    p.add_argument("--optimize-interval", type=int, default=10)
    p.add_argument("--force-ser", action="store_true")
    p.add_argument("--force-tsv", action="store_true")
    p.add_argument("--num-topics", type=int, default=5,
                   help="K. Default 5. If != 5, outputs go to outputs/<ds>/K<n>/ "
                        "and the matching beta file outputs/candidate_screen/<ds>/"
                        "beta_prism_K<n>.csv must exist (build it with "
                        "scripts/compute_beta_candidates.py --topics <n>).")
    p.add_argument("--out-root", default=None,
                   help="Write per-seed output under <out-root>/<ds>/seed<N>/ instead of "
                        "the default outputs/<ds>/. Use a distinct per-arm root (e.g. "
                        "outputs/prism_opt0) when training a single optimize-interval arm, "
                        "so opt0 and opt10 runs never share the bare outputs/<ds>/ slot. "
                        "Relative paths are resolved against the repository root. Cannot be "
                        "combined with --num-topics != 5.")
    args = p.parse_args()

    if args.out_root is not None and args.num_topics != 5:
        p.error("--out-root cannot be combined with --num-topics != 5")

    for ds in args.datasets:
        tsv = csv_to_tsv(ds, force=args.force_tsv)
        run_csv2vectors(ds, tsv, force=args.force_ser)

        beta_override = None
        out_root = None
        if args.num_topics != 5:
            beta_override = (WS / "outputs" / "candidate_screen" / ds
                             / f"beta_prism_K{args.num_topics}.csv")
            out_root = WS / "outputs" / ds / f"K{args.num_topics}"
        elif args.out_root is not None:
            base = Path(args.out_root)
            out_root = (base if base.is_absolute() else WS / base) / ds

        for seed in args.seeds:
            train(ds, seed,
                  iterations=args.iterations,
                  alpha=args.alpha,
                  optimize_interval=args.optimize_interval,
                  num_topics=args.num_topics,
                  beta_override=beta_override,
                  out_root=out_root)


if __name__ == "__main__":
    main()
