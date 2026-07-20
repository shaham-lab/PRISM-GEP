"""Run the 6 NEW datasets at OPT0 (optimize-interval OFF) so PRISM uses the
PRECOMPUTED beta prior as-is (no MALLET re-optimization of the prior) -- the
methodologically correct config, matching the 3 originals and the 9-core opt0
runs. Mirrors run_grid_newds_opt10.py but with optimize-interval=0.

Output: outputs/optimize_interval_grid/{prism_opt0,uniform_opt0}/<ds>/seed*/
Resumable: train_config SKIPs (ds,config,seed) whose topic_keys.txt exists.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS)); sys.path.insert(0, str(WS / "scripts"))
import optimize_interval_grid as G   # noqa: E402

NEW = ["paul15", "dentategyrus", "pbmc68k", "endoderm_diff",
       "ventral_neuron_diff", "mouse_hspc"]
CONFIGS = [("prism_opt0", "prism", 0), ("uniform_opt0", "uniform", 0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=NEW)
    datasets = ap.parse_args().datasets
    for ds in datasets:
        try:
            pb, ub = G.prep_betas(ds)
        except Exception as e:
            print(f"[{ds}] prep_betas ERROR: {type(e).__name__}: {e}", flush=True)
            continue
        for name, prior, opt in CONFIGS:
            try:
                n = G.train_config(ds, name, prior, opt, pb, ub)
                G.eval_config(ds, name)
                print(f"[{ds}] {name}: {n}/10 trained + evaled", flush=True)
            except Exception as e:
                print(f"[{ds}] {name} ERROR: {type(e).__name__}: {e}", flush=True)
    print("GRID_NEWDS_OPT0_DONE", flush=True)


if __name__ == "__main__":
    main()
