"""Parse MALLET --topic-word-weights-file output → top-N genes per GEP.

Consumes a weights file path and returns a dict, with no assumptions about the
surrounding directory layout.

MALLET writes one line per (topic, word, weight) triple, separated by tabs.

Usage:
    from bio.extract_top_genes import top_genes_per_topic
    top = top_genes_per_topic("outputs/breast_cancer/prism_topic_word_weights.txt", n=20)
    # top: {0: ['GENE1', 'GENE2', ...], 1: [...], ..., 4: [...]}
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_topic_word_weights(path: str | Path) -> dict[int, list[tuple[str, float]]]:
    """Parse MALLET --topic-word-weights-file output.

    Returns {topic_id: [(word, weight), ...]} unsorted.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[int, list[tuple[str, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 3:
                parts = s.split(",")
            if len(parts) < 3:
                continue
            try:
                topic = int(parts[0])
                word = parts[1]
                weight = float(parts[2])
            except ValueError:
                continue
            out[topic].append((word, weight))
    return dict(out)


def top_genes_per_topic(
    path: str | Path, n: int = 20
) -> dict[int, list[str]]:
    """Return top-n genes per topic, sorted by weight descending."""
    raw = parse_topic_word_weights(path)
    return {
        t: [w for w, _ in sorted(wl, key=lambda x: x[1], reverse=True)[:n]]
        for t, wl in raw.items()
    }


def parse_topic_keys(path: str | Path) -> dict[int, list[str]]:
    """Parse MALLET --output-topic-keys format.

    File format: one line per topic:
        <topic_id>\t<alpha>\t<word1> <word2> ... <wordN>

    Returns {topic_id: [word1, word2, ..., wordN]}, in MALLET's order
    (descending by topic-word weight).
    """
    path = Path(path)
    out: dict[int, list[str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            parts = s.split("\t")
            if len(parts) < 3:
                continue
            try:
                topic = int(parts[0])
            except ValueError:
                continue
            words = parts[2].split()
            out[topic] = words
    return out


def parse_word_topic_counts(path: str | Path) -> dict[int, dict[str, int]]:
    """Parse MALLET --word-topic-counts-file format.

    File format: one line per word in vocabulary:
        <word_idx> <word_text> <topic1>:<count1> <topic2>:<count2> ...

    Returns {topic_id: {word: count}} for all words seen.
    Use ``top_n_genes_from_counts`` to get top-N genes per topic.
    """
    from collections import defaultdict
    path = Path(path)
    per_topic: dict[int, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            word = parts[1]
            for tok in parts[2:]:
                if ":" not in tok:
                    continue
                t_str, c_str = tok.split(":", 1)
                try:
                    t = int(t_str)
                    c = int(c_str)
                except ValueError:
                    continue
                per_topic[t][word] = c
    return dict(per_topic)


def top_n_genes_from_counts(
    word_topic_counts: dict[int, dict[str, int]], n: int = 20
) -> dict[int, list[str]]:
    """Top-n genes per topic ranked by raw assignment count."""
    return {
        t: [w for w, _ in sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]]
        for t, d in word_topic_counts.items()
    }


def topic_word_probabilities(
    path: str | Path,
) -> dict[int, dict[str, float]]:
    """Per topic, normalize weights to probabilities (sum=1 per topic)."""
    raw = parse_topic_word_weights(path)
    out = {}
    for t, wl in raw.items():
        words, weights = zip(*wl)
        w = np.array(weights, dtype=float)
        s = w.sum()
        probs = (w / s) if s > 0 else np.zeros_like(w)
        out[t] = dict(zip(words, probs))
    return out
