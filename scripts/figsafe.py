"""Atomic figure writes + deploy, shared by the figure generators.

LaTeX includes these PDFs directly and another process may be running pdflatex
at any moment. matplotlib's savefig and shutil.copyfile both write in place over
many syscalls, so a compile that lands mid-write reads a truncated PDF and dies.
Write to a temp file in the SAME directory (same filesystem, so the rename is
atomic) and os.replace() onto the final name.

Extracted from scripts/viz_prism_native_trajectory.py (2026-07-20) so every
generator can use the same safe path.
"""
from __future__ import annotations


import sys
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

__all__ = ["replace_with_retry", "atomic_savefig", "atomic_copy", "save_and_deploy",
           "pdfcrop"]

WS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WS / "scripts"))
# release layout: paper/figures/, created on demand (scripts/paths.py)
from paths import figures_dir  # noqa: E402
DEPLOY = figures_dir()

# Margin kept on each side by scripts/crop_paper_figures.py. Cropping to the
# same value here means the deployed copy is already tight, so that script's
# later passes report "already tight" and leave the file alone -- and, more
# importantly, the natural width we measure is the width LaTeX will scale.
CROP_MARGIN_PT = "2"


def replace_with_retry(tmp, dst: Path, tries: int = 12, delay: float = 0.75) -> None:
    """os.replace(), retried while the destination is locked.

    On Windows a reader holding the file without FILE_SHARE_DELETE makes the
    rename fail with PermissionError, and pdflatex reading these figures is
    exactly such a reader. That is a transient collision, not an error: the
    destination is still the old, complete PDF, so we simply wait and retry.
    """
    dst = Path(dst)
    for i in range(tries):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            print(f"  [locked] {dst.name} is open elsewhere, retrying ({i + 1}/{tries})")
            time.sleep(delay)


def atomic_savefig(fig, out_path, **kw) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(out_path.parent),
                               prefix=f"._{out_path.stem}.", suffix=out_path.suffix or ".pdf")
    os.close(fd)
    try:
        fig.savefig(tmp, **kw)
        replace_with_retry(tmp, out_path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_copy(src, dst) -> None:
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dst.parent),
                               prefix=f"._{dst.stem}.", suffix=dst.suffix or ".pdf")
    os.close(fd)
    try:
        shutil.copyfile(src, tmp)
        replace_with_retry(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def pdfcrop(src: Path, dst: Path) -> bool:
    """pdfcrop src -> dst with the project's standard margin. False if unavailable."""
    try:
        r = subprocess.run(["pdfcrop", "--margins", CROP_MARGIN_PT, str(src), str(dst)],
                           capture_output=True, text=True)
    except FileNotFoundError:
        return False
    return r.returncode == 0 and Path(dst).exists()


def save_and_deploy(fig, out_path, deploy_dir=None, crop=True, **kw) -> None:
    """Write the figure atomically, crop it to ink, and mirror it to the LaTeX tree.

    The workspace copy and the deployed copy are written from the same bytes, so
    they stay hash-identical -- the two have silently diverged before.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(out_path.parent),
                               prefix=f"._{out_path.stem}.", suffix=".pdf")
    os.close(fd)
    cropped = None
    try:
        fig.savefig(tmp, **kw)
        final = tmp
        if crop and out_path.suffix.lower() == ".pdf":
            cropped = tmp[:-4] + ".crop.pdf"
            if pdfcrop(Path(tmp), Path(cropped)):
                final = cropped
            else:
                print("  [warn] pdfcrop unavailable, shipping uncropped")
                cropped = None
        if final is not tmp:
            os.unlink(tmp)
            tmp = None
        replace_with_retry(final, out_path)
        cropped = None
    except BaseException:
        for f in (tmp, cropped):
            if f and os.path.exists(f):
                os.unlink(f)
        raise
    dd = Path(deploy_dir) if deploy_dir is not None else DEPLOY
    if dd.is_dir():
        tgt = dd / out_path.name
        if tgt.resolve() != out_path.resolve():
            atomic_copy(out_path, tgt)
            print(f"  deployed -> {tgt}")
