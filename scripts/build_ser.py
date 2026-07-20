"""Build MALLET InstanceList .ser file from a cell x gene CSV.

Workflow:
  1. Read CSV (rows=cells, columns=genes, values=counts; first col = cell ID)
  2. For each cell, emit a MALLET TSV line:
       <doc_id>\t<label>\t<gene1 gene1 gene2 gene2 gene2 ...>
     where each gene is repeated by its count.
  3. Run cc.mallet.classify.tui.Csv2Vectors --keep-sequence to build the .ser.

The resulting .ser is the InstanceList consumed by TopicTrainer.

Usage:
    python scripts/build_ser.py --dataset pancreas
    python scripts/build_ser.py --dataset bonemarrow
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path(__file__).resolve().parent.parent

DATASET_FILES = {
    "pancreas":   "filtered_pancreas_cells_x_genes.csv",
    "bonemarrow": "filtered_bonemarrow_cells_x_genes.csv",
}


def csv_to_mallet_tsv(csv_path: Path, tsv_path: Path) -> None:
    """Convert cell x gene CSV to MALLET-format TSV (doc_id, label, text)."""
    print(f"  reading {csv_path} ...")
    df = pd.read_csv(csv_path, index_col=0)
    cells = df.index.astype(str).tolist()
    gene_names = list(df.columns)
    counts = df.values.astype(int)
    n_cells, n_genes = counts.shape
    print(f"  shape: {n_cells} cells x {n_genes} genes")

    print(f"  writing {tsv_path} ...")
    with tsv_path.open("w", encoding="utf-8") as fh:
        for i in range(n_cells):
            row = counts[i]
            nz = np.nonzero(row)[0]
            tokens = []
            for j in nz:
                tokens.extend([gene_names[j]] * int(row[j]))
            line = f"{cells[i]}\tcell\t{' '.join(tokens)}"
            fh.write(line + "\n")
    size_mb = tsv_path.stat().st_size / (1024 * 1024)
    print(f"  TSV size: {size_mb:.1f} MB")


def run_csv2vectors(tsv_path: Path, ser_path: Path) -> None:
    """Use MALLET's Csv2Vectors --keep-sequence to build the InstanceList."""
    print(f"  running Csv2Vectors -> {ser_path} ...")
    java = os.environ.get("JAVA", "java")
    cp = "mallet/class" + os.pathsep + "mallet/lib/mallet-deps.jar"
    cmd = [
        java, "-Xmx4G", "-cp", cp,
        "cc.mallet.classify.tui.Csv2Vectors",
        "--input", str(tsv_path),
        "--output", str(ser_path),
        "--keep-sequence",
        "--token-regex", r"[^\s]+",
    ]
    proc = subprocess.run(cmd, cwd=WS, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout); print(proc.stderr)
        raise RuntimeError(f"Csv2Vectors failed (exit {proc.returncode})")
    # show last lines of stdout
    print(proc.stdout.strip().splitlines()[-3:] if proc.stdout else "(no output)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(DATASET_FILES.keys()))
    args = p.parse_args()

    csv = WS / "data" / args.dataset / DATASET_FILES[args.dataset]
    if not csv.exists():
        raise FileNotFoundError(csv)

    out_dir = WS / "ground_truth_ser"
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / f"{args.dataset}.tsv"
    ser = out_dir / f"{args.dataset}.ser"

    csv_to_mallet_tsv(csv, tsv)
    run_csv2vectors(tsv, ser)

    # Cleanup intermediate TSV (keep .ser only)
    if ser.exists():
        size_kb = ser.stat().st_size / 1024
        print(f"  .ser size: {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
