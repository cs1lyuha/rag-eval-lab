"""Două „modele de embedding” care rulează offline, plus un adaptor opțional.

- `TfidfEmbedder`: vectori TF-IDF pe stem-uri, normalizați L2. Lexical,
  dar ponderat: cuvintele rare contează mai mult.
- `CharNgramEmbedder`: n-grame de caractere (3 și 4) trecute prin feature
  hashing într-un vector dens de dimensiune fixă. Prinde forme flexionare
  și greșeli de tastare pe care stemming-ul prin prefix le ratează.
- `SentenceTransformerEmbedder`: model neural real (de ex.
  `intfloat/multilingual-e5-small`), doar dacă e instalat
  `sentence-transformers`. Nu rulează în CI.

Toate întorc matrice numpy cu rânduri normalizate, deci similaritatea
cosinus este un simplu produs scalar.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Protocol

import numpy as np

from rag_lab.text import tokens, words


class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> "Embedder": ...

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0)

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        df: Counter[str] = Counter()
        for t in texts:
            df.update(set(tokens(t)))
        self.vocab = {term: i for i, term in enumerate(sorted(df))}
        n = len(texts)
        self.idf = np.array([math.log((1 + n) / (1 + df[t])) + 1 for t in sorted(df)])
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), len(self.vocab)))
        for row, t in enumerate(texts):
            for term, count in Counter(tokens(t)).items():
                col = self.vocab.get(term)
                if col is not None:
                    m[row, col] = 1 + math.log(count)
        return _normalize(m * self.idf)


class CharNgramEmbedder:
    name = "char-ngram"

    def __init__(self, dim: int = 2048, ns: tuple[int, ...] = (3, 4)) -> None:
        self.dim = dim
        self.ns = ns
        self.idf = np.ones(dim)

    def _bucket(self, gram: str) -> int:
        return int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=4).digest(), "little") % self.dim

    def _grams(self, text: str) -> Counter[int]:
        out: Counter[int] = Counter()
        for w in words(text):
            padded = f"<{w}>"
            for n in self.ns:
                for i in range(max(1, len(padded) - n + 1)):
                    out[self._bucket(padded[i:i + n])] += 1
        return out

    def fit(self, texts: list[str]) -> "CharNgramEmbedder":
        df = np.zeros(self.dim)
        for t in texts:
            for b in self._grams(t):
                df[b] += 1
        self.idf = np.log((1 + len(texts)) / (1 + df)) + 1
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim))
        for row, t in enumerate(texts):
            for b, count in self._grams(t).items():
                m[row, b] = 1 + math.log(count)
        return _normalize(m * self.idf)


class SentenceTransformerEmbedder:  # pragma: no cover - dependență opțională, grea
    def __init__(self, model: str = "intfloat/multilingual-e5-small") -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model.split("/")[-1]
        self.model = SentenceTransformer(model)
        self.e5 = "e5" in model

    def fit(self, texts: list[str]) -> "SentenceTransformerEmbedder":
        return self

    def embed(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        if self.e5:
            texts = [("query: " if query else "passage: ") + t for t in texts]
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))


EMBEDDERS = {"tfidf": TfidfEmbedder, "char-ngram": CharNgramEmbedder}
