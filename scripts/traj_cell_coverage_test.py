"""Can the sub-lineage datasets be scored at 100% coverage instead of dropping off-lineage cells?

For each dataset whose lineage_order covers only part of the annotation, we keep ALL cells and
assign the off-lineage cell types a rank, under three placements that are each as defensible as
the others:

  END    off-lineage types appended after the last lineage stage
  START  off-lineage types placed before the first stage
  MID    off-lineage types placed at the midpoint of the lineage
  TIED   all off-lineage types share ONE rank at the end (they are not ordered among themselves)

If those placements give similar scores, the off-lineage cells carry no weight and we may as well
keep 100% of cells. If they disagree, the score is a function of an arbitrary choice, and dropping
the off-lineage cells is the only defensible option. The spread across placements IS the answer.

    python scripts/traj_cell_coverage_test.py --seeds 0 1 2
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS)); sys.path.insert(0, str(WS / "scripts"))
from cell_ordering_extra_steps import js_diffmap_dc1, absrho          # noqa: E402
from traj_cell_all_datasets import ORDERINGS, paths, label_map, doc_topics  # noqa: E402

# datasets where the shipped ordering covers only part of the annotation
SUBLINEAGE = ["paul15", "bonemarrow", "gastrulation_e75", "dentategyrus", "hemogenic_endothelium"]


def placements(order, extra):
    """rank map variants over the FULL label set (lineage `order` + off-lineage `extra`)."""
    n = len(order)
    base = {l: float(i) for i, l in enumerate(order)}
    out = {}
    v = dict(base); v.update({l: float(n + j) for j, l in enumerate(extra)}); out["END"] = v
    v = {l: float(len(extra) + i) for i, l in enumerate(order)}
    v.update({l: float(j) for j, l in enumerate(extra)}); out["START"] = v
    v = {l: float(i if i < n / 2 else i + len(extra)) for i, l in enumerate(order)}
    v.update({l: float(n // 2 + j) for j, l in enumerate(extra)}); out["MID"] = v
    v = dict(base); v.update({l: float(n) for l in extra}); out["TIED"] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=SUBLINEAGE)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    rows = []
    for ds in a.datasets:
        lab_p, cnt_p = paths(ds)
        ids = pd.read_csv(cnt_p, index_col=0, usecols=[0]).index.astype(str).to_numpy()
        m = label_map(ds, lab_p)
        lab_of = np.array([m.get(c, "\0") for c in ids])
        order = ORDERINGS[ds][1].split("|")
        extra = sorted(set(lab_of) - set(order) - {"\0"})
        has = lab_of != "\0"
        sub = np.isin(lab_of, order)
        print(f"[{ds}] lineage {len(order)} labels ({100*sub.sum()/sub.size:.1f}% cells), "
              f"off-lineage {len(extra)} labels ({100*(has & ~sub).sum()/has.size:.1f}%)", flush=True)
        maps = placements(order, extra)
        acc = {k: [] for k in maps}
        acc["SUBLINEAGE_ONLY"] = []
        for s in a.seeds:
            P = doc_topics(ds, s, ids.size)
            if P is None:
                continue
            o_full = js_diffmap_dc1(P[has])
            for k, rm in maps.items():
                acc[k].append(absrho(o_full, np.array([rm[l] for l in lab_of[has]], float)))
            o_sub = js_diffmap_dc1(P[sub])
            acc["SUBLINEAGE_ONLY"].append(absrho(o_sub, np.array(
                [order.index(l) for l in lab_of[sub]], float)))
        r = {"dataset": ds, "n_lineage_labels": len(order), "n_offlineage_labels": len(extra),
             "pct_sublineage": round(100 * sub.sum() / sub.size, 1)}
        for k, v in acc.items():
            r[k] = float(np.mean(v)) if v else np.nan
        pl = [r[k] for k in ["END", "START", "MID", "TIED"]]
        r["placement_spread"] = float(np.nanmax(pl) - np.nanmin(pl))
        rows.append(r)
        print(f"   sub-only={r['SUBLINEAGE_ONLY']:.3f} | END={r['END']:.3f} START={r['START']:.3f} "
              f"MID={r['MID']:.3f} TIED={r['TIED']:.3f} | placement spread={r['placement_spread']:.3f}\n",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(WS / "outputs" / "trajectory" / "cell_traj_coverage_test.csv", index=False)
    print("=== can we score at 100% coverage? ===")
    print(df.round(3).to_string(index=False))
    print("\nplacement_spread = how much PRISM's score moves purely from WHERE we put the")
    print("off-lineage cells. Large spread means the 100%-coverage number is a choice, not a result.")


if __name__ == "__main__":
    main()
