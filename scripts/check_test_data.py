"""Quick installation check on the bundled test dataset (Pancreas).

Recomputes the PRISM-GEP informed prior (Stages A-D) on data/pancreas/ with the production
settings and compares it with data/pancreas/expected_beta_prism.csv. The computation is
deterministic, so on a correct installation the two agree to floating-point precision.
Takes under a minute and needs no Java: MALLET training is not part of this check.

    python scripts/check_test_data.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "pancreas"
OUT = ROOT / "outputs" / "pancreas" / "beta_prism.csv"
EXPECTED = DATA / "expected_beta_prism.csv"
TOL = 1e-6

if not (DATA / "filtered_pancreas_cells_x_genes.csv").exists():
    sys.exit("missing data/pancreas/filtered_pancreas_cells_x_genes.csv")

cmd = [sys.executable, "-m", "bio.pipeline", "--dataset", "pancreas", "--K", "5", "--m", "20",
       "--n_neighbors", "15", "--n_pca", "50", "--expression_threshold", "2.0",
       "--neighborhood_min_support", "1"]
print("running:", " ".join(cmd[1:]))
subprocess.run(cmd, cwd=ROOT, check=True)

got = np.loadtxt(OUT, delimiter=",")
want = np.loadtxt(EXPECTED, delimiter=",")
if got.shape != want.shape:
    sys.exit(f"FAIL: prior has {got.size} entries, expected {want.size}")
diff = float(np.max(np.abs(got - want)))
print(f"prior entries: {got.size}, max absolute difference from the reference: {diff:.2e}")
if diff > TOL:
    sys.exit(f"FAIL: difference above {TOL:g}")
print("PASS: the installation reproduces the reference prior")
