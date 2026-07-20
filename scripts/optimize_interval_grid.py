"""Fresh optimize-interval grid across all 9 datasets x 10 seeds.

For every dataset, train BOTH priors (PRISM corpus-beta and uniform-beta) at
optimize-interval 10 AND 0, 10 seeds each, then build a comparison table.
4 configs x 9 datasets x 10 seeds = 360 fresh trainings (the 9x10x2 PRISM grid, plus
the matching uniform-beta baseline so the table is an actual PRISM-vs-MALLET comparison,
with vs without optimization).

  prism_opt10   : corpus beta (alphabet-aligned), --optimize-interval 10   (= production PRISM)
  prism_opt0    : corpus beta (alphabet-aligned), --optimize-interval 0    (beta FROZEN)
  uniform_opt10 : uniform 0.01 beta,              --optimize-interval 10   (= vanilla MALLET)
  uniform_opt0  : uniform 0.01 beta,              --optimize-interval 0    (frozen uniform)

Everything is NEW (own output root), resumable (SKIP-if-exists per seed), and
processed fast->slow so small datasets finish first. After each dataset's 4 configs
train+eval, the comparison table + README are rebuilt, so partial progress is usable.

Caveat (documented in the README): the three original datasets (breast_cancer/pbmc3k/
zeisel) use a corpus beta reconstructed by this pipeline rather than a precomputed beta
file; on breast_cancer that beta is near-uniform, so PRISM ~ uniform there is expected.

Run (background):  python scripts/optimize_interval_grid.py
Rebuild table only: python scripts/optimize_interval_grid.py --table-only
"""
from __future__ import annotations
import csv
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS))
sys.path.insert(0, str(WS / "scripts"))
import train_prism_standard as T                                   # noqa: E402
from retrain_betafix import source_beta, alphabet_from_wtc, reorder_beta  # noqa: E402

# fast -> slow (by cell count) so small datasets land first
DATASETS = ["pbmc3k", "hemogenic_endothelium", "breast_cancer", "zeisel_brain",
            "pancreas", "bonemarrow", "gastrulation_e75", "gastrulation",
            "gastrulation_erythroid"]
SEEDS = list(range(10))
ALPHA, ITERS = 50.0, 1000
ROOT = WS / "outputs" / "optimize_interval_grid"
EVAL = WS / "scripts" / "evaluate_all_new_dataset_table.py"
README = ROOT / "README.md"
# (config name, prior, optimize_interval)
CONFIGS = [("prism_opt10", "prism", 10), ("prism_opt0", "prism", 0),
           ("uniform_opt10", "uniform", 10), ("uniform_opt0", "uniform", 0)]
LABEL = {"prism_opt10": "PRISM opt10", "prism_opt0": "PRISM opt0",
         "uniform_opt10": "MALLET opt10", "uniform_opt0": "MALLET opt0"}


def log(m):
    print(m, flush=True)


def prep_betas(ds):
    """Probe the .ser for its alphabet, cache aligned PRISM beta + matching uniform beta."""
    tsv = T.csv_to_tsv(ds, force=False)
    T.run_csv2vectors(ds, tsv, force=False)
    src = source_beta(ds)
    probe_wtc = ROOT / "_probe" / ds / "seed0" / "word_topic_counts.txt"
    if not probe_wtc.exists():
        log(f"[{ds}] probe (1 iter) for alphabet")
        T.train(ds, 0, iterations=1, alpha=ALPHA, optimize_interval=0,
                beta_override=src, out_root=ROOT / "_probe" / ds)
    alphabet = alphabet_from_wtc(probe_wtc)
    prism_beta = ROOT / "_betas" / f"{ds}_prism_alpha.csv"
    if not prism_beta.exists():
        reorder_beta(src, ds, alphabet, prism_beta)
    uni_beta = ROOT / "_betas" / f"{ds}_uniform.csv"
    if not uni_beta.exists():
        arr = np.atleast_1d(np.loadtxt(prism_beta, delimiter=",")).ravel()
        uni_beta.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(uni_beta, [np.full_like(arr, 0.01)], delimiter=",", fmt="%.6f")
    return prism_beta, uni_beta


def train_config(ds, name, prior, opt, prism_beta, uni_beta):
    beta = prism_beta if prior == "prism" else uni_beta
    out_root = ROOT / name / ds
    done = 0
    for s in SEEDS:
        if (out_root / f"seed{s}" / "topic_keys.txt").exists():
            done += 1
            continue
        T.train(ds, s, iterations=ITERS, alpha=ALPHA, optimize_interval=opt,
                beta_override=beta, out_root=out_root)
        done += 1
    return done


def eval_config(ds, name):
    out = ROOT / "_eval" / f"{name}_{ds}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    # always re-eval after training (cheap vs train); overwrite to stay fresh
    subprocess.run([sys.executable, str(EVAL), "--methods", "PRISM", "--datasets", ds,
                    "--seeds", *map(str, SEEDS), "--prism-root", str(ROOT / name),
                    "--out", str(out)], cwd=str(WS), check=False)
    return out


def read_eval(name, ds):
    f = ROOT / "_eval" / f"{name}_{ds}.csv"
    if not f.exists():
        return None
    with open(f, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["dataset"] == ds:
                def g(k):
                    v = r.get(k)
                    return float(v) if v not in (None, "", "nan") else None
                return {"coh": g("coherence_mean"), "cov": g("coverage_mean"),
                        "str": g("strength_mean"), "n": r.get("n_seeds", "?")}
    return None


def build_table():
    lines = ["# Optimize-interval grid -- fresh, all 9 datasets x 10 seeds\n",
             "All runs NEW (`outputs/optimize_interval_grid/`). 4 configs per dataset:\n",
             "- **PRISM opt10** = corpus beta, `--optimize-interval 10` (production PRISM)",
             "- **PRISM opt0**  = corpus beta, **frozen** (`--optimize-interval 0`)",
             "- **MALLET opt10**= uniform beta, `--optimize-interval 10` (vanilla MALLET)",
             "- **MALLET opt0** = uniform beta, **frozen**\n",
             "Metrics: mean over available seeds; Coverage = per-gene, Strength = mean -log10 q "
             "(as `evaluate_all_new_dataset_table.py` emits). Same eval for all 4 configs.\n",
             "**Key reads:** PRISM-vs-MALLET *within* a setting = the beta-prior effect; "
             "opt10-vs-opt0 = the optimization effect. If PRISM>>MALLET at opt0 but PRISM~MALLET "
             "at opt10, optimize-interval is washing out the prior.\n",
             "_Caveat: the three originals (breast_cancer/pbmc3k/zeisel) use a corpus beta "
             "reconstructed by this pipeline (near-uniform on breast_cancer), not a "
             "precomputed beta file._\n"]
    # main comparison table
    lines.append("## Comparison (Coherence / Coverage / Strength)\n")
    hdr = "| Dataset | Metric | PRISM opt10 | MALLET opt10 | Δ(P−M) opt10 | PRISM opt0 | MALLET opt0 | Δ(P−M) opt0 |"
    lines.append(hdr)
    lines.append("|" + "---|" * 8)
    metrics = [("Coherence", "coh"), ("Coverage", "cov"), ("Strength", "str")]
    for ds in DATASETS:
        p10, m10 = read_eval("prism_opt10", ds), read_eval("uniform_opt10", ds)
        p0, m0 = read_eval("prism_opt0", ds), read_eval("uniform_opt0", ds)
        if not any([p10, m10, p0, m0]):
            continue
        for mlabel, mk in metrics:
            def fmt(d):
                return f"{d[mk]:.3f}" if d and d.get(mk) is not None else "--"
            def dd(a, b):
                if a and b and a.get(mk) is not None and b.get(mk) is not None:
                    return f"{a[mk]-b[mk]:+.3f}"
                return "--"
            lines.append(f"| {ds} | {mlabel} | {fmt(p10)} | {fmt(m10)} | {dd(p10,m10)} "
                         f"| {fmt(p0)} | {fmt(m0)} | {dd(p0,m0)} |")
        lines.append("| | | | | | | | |")
    # progress / status
    lines.append("\n## Seed-completion status\n")
    lines.append("| Dataset | PRISM opt10 | PRISM opt0 | MALLET opt10 | MALLET opt0 |")
    lines.append("|" + "---|" * 5)
    for ds in DATASETS:
        cells = []
        for name, _, _ in CONFIGS:
            d = ROOT / name / ds
            n = len(list(d.glob("seed*/topic_keys.txt"))) if d.exists() else 0
            cells.append(f"{n}/10")
        lines.append(f"| {ds} | " + " | ".join(cells) + " |")
    README.parent.mkdir(parents=True, exist_ok=True)
    README.write_text("\n".join(lines), encoding="utf-8")
    log(f"[table] wrote {README.relative_to(WS)}")


def main():
    if "--table-only" in sys.argv:
        build_table()
        return
    # Keep Windows awake while the grid runs (transient; lapses when this process
    # exits). Note: closing the laptop lid still forces sleep regardless.
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        log("[power] requested system stay-awake (ES_SYSTEM_REQUIRED) for grid lifetime")
    except Exception as e:  # noqa: BLE001
        log(f"[power] stay-awake request failed (continuing): {e}")
    workers = 4
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
    ROOT.mkdir(parents=True, exist_ok=True)
    n_train = len(DATASETS) * len(SEEDS) * len(CONFIGS)
    log(f"=== optimize-interval grid: {len(DATASETS)} datasets x {len(SEEDS)} seeds x "
        f"{len(CONFIGS)} configs = {n_train} trainings, {workers} parallel workers ===")

    # Phase 0: prep all betas SERIALLY (probe+reorder) so concurrent units of the
    # same dataset never race on the cached beta files.
    prepped = {}
    for ds in DATASETS:
        try:
            prepped[ds] = prep_betas(ds)
            log(f"[prep] {ds} ok")
        except Exception as e:  # noqa: BLE001
            log(f"[prep] {ds} FAILED (skipping dataset): {e}")

    # Phase 1: parallel over (dataset, config) units. Each trains its 10 seeds
    # (skip-if-exists), evals, and rebuilds the README under a lock.
    lock = threading.Lock()

    def do_unit(ds, name, prior, opt):
        try:
            pb, ub = prepped[ds]
            n = train_config(ds, name, prior, opt, pb, ub)
            eval_config(ds, name)
            with lock:
                log(f"[done] {ds}/{name} {n}/10 seeds")
                build_table()
        except Exception as e:  # noqa: BLE001
            log(f"[unit FAIL] {ds}/{name}: {e}")

    units = [(ds, n, p, o) for ds in prepped for (n, p, o) in CONFIGS]
    log(f"=== launching {len(units)} (dataset x config) units across {workers} workers ===")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(do_unit, *u) for u in units]
        for f in as_completed(futs):
            f.result()
    build_table()
    log("=== GRID DONE ===")


if __name__ == "__main__":
    main()
