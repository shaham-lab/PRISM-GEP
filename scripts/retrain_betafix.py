"""Probe -> reorder -> retrain PRISM with the corrected (alphabet-order) beta.

Self-consistent regardless of how the .ser was built:
  1. Build/reuse ground_truth_ser/<ds>.ser (via train_prism_standard helpers).
  2. PROBE train (1 iteration) to emit word_topic_counts.txt = the EXACT alphabet
     of THIS .ser.
  3. Reorder the dataset's column-order beta into that alphabet order
     -> outputs_betafix/<ds>/beta_prism_alpha.csv (+ .genes.tsv sidecar).
  4. Full train seeds with the corrected beta into outputs_betafix/<ds>/seedN.

Never overwrites the buggy runs: everything lands under outputs_betafix/.
Resumable: every step SKIPs if its output already exists.

Source (column-order) beta resolution per dataset, first existing wins:
    outputs/candidate_screen/<ds>/beta_prism.csv
    ground_truth_beta/<ds>_beta.csv

Usage:
    python scripts/retrain_betafix.py pancreas bonemarrow hemogenic_endothelium \
        gastrulation gastrulation_e75 gastrulation_erythroid \
        --seeds 0 1 2 3 4 5 6 7 8 9 --iterations 1000
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS))
sys.path.insert(0, str(WS / "scripts"))

import train_prism_standard as T  # noqa: E402
from bio.pipeline import first_encounter_order  # noqa: E402


def norm(g: str) -> str:
    return re.sub(r"\s+", "_", str(g).strip()).casefold()


def source_beta(ds: str) -> Path:
    # The column-order (data-CSV order) beta written by either prior-building entry
    # point; reorder_beta() re-aligns it to MALLET type-id order downstream. Accept
    # both documented locations: scripts/compute_beta_candidates.py writes
    # outputs/candidate_screen/<ds>/, `python -m bio.pipeline` writes outputs/<ds>/.
    cands = [
        WS / "outputs" / "candidate_screen" / ds / "beta_prism.csv",
        WS / "outputs" / ds / "beta_prism.csv",
        WS / "ground_truth_beta" / f"{ds}_beta.csv",
    ]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"no source beta for {ds}; tried {cands}")


def alphabet_from_wtc(wtc: Path) -> list[str]:
    id_to_gene: dict[int, str] = {}
    with wtc.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2:
                id_to_gene[int(parts[0])] = parts[1]
    n = max(id_to_gene) + 1
    if any(i not in id_to_gene for i in range(n)):
        raise ValueError(f"{wtc}: gaps in type ids")
    return [id_to_gene[i] for i in range(n)]


def reorder_beta(src_beta: Path, ds: str, alphabet: list[str], out: Path) -> Path:
    cols = list(pd.read_csv(T.csv_path(ds), index_col=0, nrows=0).columns)
    beta = np.atleast_1d(np.loadtxt(src_beta, delimiter=",")).ravel()
    if len(beta) != len(cols):
        raise ValueError(f"{ds}: beta len {len(beta)} != #cols {len(cols)}")
    g2b = {norm(g): float(b) for g, b in zip(cols, beta)}
    missing = [g for g in alphabet if norm(g) not in g2b]
    if missing:
        raise ValueError(f"{ds}: {len(missing)} alphabet genes absent from beta")
    beta_re = np.array([g2b[norm(g)] for g in alphabet], dtype=float)
    assert np.allclose(np.sort(beta_re), np.sort(beta)), "reorder not a permutation"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, [beta_re], delimiter=",", fmt="%.6f")
    with out.with_suffix(out.suffix + ".genes.tsv").open("w", encoding="utf-8") as fh:
        fh.write("type_id\tgene\tbeta\n")
        for i, g in enumerate(alphabet):
            fh.write(f"{i}\t{g}\t{beta_re[i]:.6f}\n")
    return out


def run_dataset(ds: str, seeds: list[int], iterations: int,
                alpha: float, optimize_interval: int) -> None:
    print(f"\n========== {ds} ==========")
    betafix_root = WS / "outputs_betafix" / ds

    # 1. ensure .ser
    tsv = T.csv_to_tsv(ds, force=False)
    T.run_csv2vectors(ds, tsv, force=False)

    src = source_beta(ds)
    print(f"[{ds}] source (column-order) beta: {src.relative_to(WS)}")

    # 2. probe train (1 iter) for the alphabet
    probe_root = betafix_root / "_probe"
    probe_wtc = probe_root / "seed0" / "word_topic_counts.txt"
    if not probe_wtc.exists():
        print(f"[{ds}] probe train (1 iteration) for alphabet ...")
        T.train(ds, 0, iterations=1, alpha=alpha,
                optimize_interval=optimize_interval,
                beta_override=src, out_root=probe_root)
    else:
        print(f"[{ds}] probe alphabet present, reuse")
    alphabet = alphabet_from_wtc(probe_wtc)

    # 3. reorder
    beta_alpha = betafix_root / "beta_prism_alpha.csv"
    if not beta_alpha.exists():
        reorder_beta(src, ds, alphabet, beta_alpha)
        print(f"[{ds}] wrote corrected beta {beta_alpha.relative_to(WS)}")
    else:
        print(f"[{ds}] corrected beta present, reuse")

    # 4. full train with corrected beta. Per-seed try/except so a transient
    #    MALLET/JVM failure (e.g. memory pressure) skips that seed instead of
    #    aborting the whole queue; re-running retries the skipped seed.
    failures = []
    for s in seeds:
        try:
            T.train(ds, s, iterations=iterations, alpha=alpha,
                    optimize_interval=optimize_interval,
                    beta_override=beta_alpha, out_root=betafix_root)
        except Exception as e:  # noqa: BLE001
            print(f"[{ds}] seed={s} FAILED (continuing): {e}")
            # remove any empty/partial seed dir so a later resume retries it
            sd = betafix_root / f"seed{s}"
            if sd.exists() and not (sd / "topic_keys.txt").exists():
                for p in sorted(sd.glob("*")):
                    try:
                        p.unlink()
                    except OSError:
                        pass
            failures.append(s)
    if failures:
        print(f"[{ds}] done WITH FAILURES seeds={failures} -> {betafix_root.relative_to(WS)}")
    else:
        print(f"[{ds}] done -> {betafix_root.relative_to(WS)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("datasets", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=50.0)
    p.add_argument("--optimize-interval", type=int, default=10)
    args = p.parse_args()
    for ds in args.datasets:
        try:
            run_dataset(ds, args.seeds, args.iterations, args.alpha,
                        args.optimize_interval)
        except Exception as e:  # noqa: BLE001
            print(f"[{ds}] DATASET FAILED (continuing to next): {e}")


if __name__ == "__main__":
    main()
