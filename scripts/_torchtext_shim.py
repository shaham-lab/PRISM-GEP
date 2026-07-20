"""Minimal pure-Python stand-in for ``torchtext.vocab`` so ``import scgpt`` works.

Why this exists
---------------
``scgpt`` 0.2.4 imports ``torchtext.vocab`` in ``scgpt/tokenizer/gene_tokenizer.py``,
and ``scgpt/__init__.py`` imports that tokenizer, so *any* ``from scgpt... import ...``
pulls torchtext in. On this machine torchtext 0.18 cannot load its C++ extension
against torch 2.6.0+cu124::

    OSError: [WinError 127] The specified procedure could not be found  (libtorchtext)

``bio/gene_embeddings.py`` already sidesteps this for the *static* scGPT
embeddings by reading ``best_model.pt``/``vocab.json`` directly. The
*contextual* ablation needs the real ``TransformerModel``, so instead we
install a tiny replacement for the only two torchtext names scgpt touches:
``torchtext.vocab.Vocab`` and ``torchtext.vocab.vocab``.

``GeneVocab`` subclasses ``Vocab`` and uses only ``__len__``/``__contains__``/
``__getitem__``/``insert_token``/``append_token``/``set_default_index``/
``get_stoi``/``get_itos``, plus the ``.vocab`` handle it passes to ``super()``.
All of that is a dict and a list, so the shim is exact for our use: the vocab is
loaded from ``vocab.json`` via ``GeneVocab.from_file``, which only inserts
tokens at consecutive indices.

Import this module *before* anything imports ``scgpt``. It is a no-op if the
real torchtext loads fine.
"""
from __future__ import annotations

import sys
import types
from collections import OrderedDict


def _real_torchtext_works() -> bool:
    try:
        import torchtext.vocab  # noqa: F401
        return True
    except Exception:
        return False


class _VocabImpl:
    """The object torchtext calls ``vocab.vocab`` (its C++ handle)."""

    def __init__(self, itos=None):
        self.itos = list(itos or [])
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.default_index = None


class Vocab:
    def __init__(self, vocab_impl):
        if isinstance(vocab_impl, Vocab):
            vocab_impl = vocab_impl.vocab
        if not isinstance(vocab_impl, _VocabImpl):
            vocab_impl = _VocabImpl(vocab_impl)
        self._impl = vocab_impl

    # torchtext exposes the backing handle as `.vocab`
    @property
    def vocab(self) -> _VocabImpl:
        return self._impl

    def __len__(self) -> int:
        return len(self._impl.itos)

    def __contains__(self, token: str) -> bool:
        return token in self._impl.stoi

    def __getitem__(self, token: str) -> int:
        idx = self._impl.stoi.get(token)
        if idx is None:
            if self._impl.default_index is None:
                raise RuntimeError(f"token {token!r} not in vocab and no default index set")
            return self._impl.default_index
        return idx

    def insert_token(self, token: str, index: int) -> None:
        if token in self._impl.stoi:
            raise RuntimeError(f"token {token!r} already exists in the vocab")
        n = len(self._impl.itos)
        if not 0 <= index <= n:
            raise RuntimeError(f"insert index {index} out of range")
        if index == n:
            # Append: the only case GeneVocab.from_dict hits, and it must be
            # O(1). Rebuilding stoi here made loading the 60k-token scGPT vocab
            # quadratic (~3.7e9 dict writes) and looked like a hang.
            self._impl.itos.append(token)
            self._impl.stoi[token] = index
            return
        self._impl.itos.insert(index, token)
        for i in range(index, len(self._impl.itos)):
            self._impl.stoi[self._impl.itos[i]] = i

    def append_token(self, token: str) -> None:
        self.insert_token(token, len(self._impl.itos))

    def set_default_index(self, index) -> None:
        self._impl.default_index = index

    def get_default_index(self):
        return self._impl.default_index

    def get_stoi(self) -> dict:
        return dict(self._impl.stoi)

    def get_itos(self) -> list:
        return list(self._impl.itos)

    def lookup_token(self, index: int) -> str:
        return self._impl.itos[index]

    def lookup_tokens(self, indices) -> list:
        return [self._impl.itos[i] for i in indices]

    def lookup_indices(self, tokens) -> list:
        return [self[t] for t in tokens]

    def forward(self, tokens) -> list:
        return self.lookup_indices(tokens)


def vocab(ordered_dict, min_freq: int = 1, specials=None, special_first: bool = True) -> Vocab:
    """torchtext.vocab.vocab: build a Vocab from an ordered token -> freq dict."""
    if not isinstance(ordered_dict, OrderedDict):
        ordered_dict = OrderedDict(ordered_dict)
    tokens = [t for t, f in ordered_dict.items() if f >= min_freq]
    if specials:
        tokens = [t for t in tokens if t not in set(specials)]
        tokens = list(specials) + tokens if special_first else tokens + list(specials)
    return Vocab(_VocabImpl(tokens))


def build_vocab_from_iterator(iterator, min_freq: int = 1, specials=None,
                              special_first: bool = True, max_tokens=None) -> Vocab:
    from collections import Counter
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)
    ordered = OrderedDict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))
    return vocab(ordered, min_freq=min_freq, specials=specials, special_first=special_first)


def install(verbose: bool = True) -> bool:
    """Install the shim if (and only if) the real torchtext cannot import.

    Returns True when the shim was installed.
    """
    if "scgpt" in sys.modules:
        raise RuntimeError("install() must be called before scgpt is imported")
    if _real_torchtext_works():
        if verbose:
            print("[torchtext shim] real torchtext imports fine -- not shimming")
        return False

    # Drop the half-initialised real package so our stub is what scgpt sees.
    for name in [m for m in sys.modules if m == "torchtext" or m.startswith("torchtext.")]:
        del sys.modules[name]

    tt = types.ModuleType("torchtext")
    tt.__version__ = "0.0.0+shim"
    tt_vocab = types.ModuleType("torchtext.vocab")
    tt_vocab.Vocab = Vocab
    tt_vocab.vocab = vocab
    tt_vocab.build_vocab_from_iterator = build_vocab_from_iterator
    tt.vocab = tt_vocab
    sys.modules["torchtext"] = tt
    sys.modules["torchtext.vocab"] = tt_vocab
    if verbose:
        print("[torchtext shim] real torchtext failed to load -- using pure-Python vocab shim")
    return True
