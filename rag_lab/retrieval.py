"""Retrieval: dens (cosinus), BM25, hibrid (Reciprocal Rank Fusion) și reranking.

Toate retriever-ele au aceeași interfață: `search(query, k)` întoarce o
listă de `Hit(chunk, score)` ordonată descrescător.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rag_lab.chunking import Chunk
from rag_lab.embeddings import Embedder
from rag_lab.text import tokens


@dataclass
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[Hit]: ...


class DenseRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        self.chunks = chunks
        self.embedder = embedder.fit([c.text for c in chunks])
        self.matrix = self.embedder.embed([c.text for c in chunks])
        self.name = f"dense:{embedder.name}"

    def search(self, query: str, k: int) -> list[Hit]:
        q = self.embedder.embed([query])[0]
        scores = self.matrix @ q
        top = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in top]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokens(c.text)) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(1, len(self.lengths))
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query_terms: list[str], i: int) -> float:
        doc, length, s = self.docs[i], self.lengths[i], 0.0
        for t in query_terms:
            tf = doc.get(t, 0)
            if tf:
                s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg_len))
        return s

    def search(self, query: str, k: int) -> list[Hit]:
        terms = tokens(query)
        scored = sorted(((self.score(terms, i), i) for i in range(len(self.chunks))), key=lambda x: (-x[0], x[1]))
        return [Hit(self.chunks[i], s) for s, i in scored[:k]]


class HybridRetriever:
    """Reciprocal Rank Fusion: scor = Σ 1 / (rrf_k + rang). Nu cere scoruri pe aceeași scară."""

    def __init__(self, retrievers: list[Retriever], rrf_k: int = 60, depth: int = 50) -> None:
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.depth = depth
        self.name = "hybrid(" + "+".join(r.name for r in retrievers) + ")"

    def search(self, query: str, k: int) -> list[Hit]:
        scores: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for r in self.retrievers:
            for rank, hit in enumerate(r.search(query, self.depth), start=1):
                scores[hit.chunk.id] = scores.get(hit.chunk.id, 0.0) + 1 / (self.rrf_k + rank)
                by_id[hit.chunk.id] = hit.chunk
        ordered = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]
        return [Hit(by_id[cid], s) for cid, s in ordered]


class LexicalReranker:
    """Reranker ieftin, fără model: reordonează candidații după cât de bine
    acoperă termenii întrebării (ponderat cu IDF), cu bonus pentru perechi
    de termeni consecutivi și pentru termeni apropiați în text.

    Joacă rolul unui cross-encoder: se uită la întrebare și la chunk
    *împreună*, nu la vectori calculați separat. Un cross-encoder real
    (de ex. `BAAI/bge-reranker-v2-m3`) se poate pune în loc, cu aceeași interfață.
    """

    name = "lexical-rerank"

    def __init__(self, idf: dict[str, float], window: int = 12) -> None:
        self.idf = idf
        self.window = window

    def score(self, query: str, chunk: Chunk) -> float:
        q = list(dict.fromkeys(tokens(query)))
        if not q:
            return 0.0
        text = tokens(chunk.text)
        present = set(text)
        total = sum(self.idf.get(t, 1.0) for t in q)
        coverage = sum(self.idf.get(t, 1.0) for t in q if t in present) / total
        bigrams = set(zip(text, text[1:]))
        bigram_hits = sum(1 for pair in zip(q, q[1:]) if pair in bigrams) / max(1, len(q) - 1)
        positions = [i for i, t in enumerate(text) if t in q]
        best = 0
        for i, p in enumerate(positions):
            window_terms = {text[j] for j in positions[i:] if j - p < self.window}
            best = max(best, len(window_terms))
        proximity = best / len(q)
        return coverage + 0.3 * bigram_hits + 0.3 * proximity


class RerankedRetriever:
    def __init__(self, base: Retriever, reranker: LexicalReranker, candidates: int = 20) -> None:
        self.base = base
        self.reranker = reranker
        self.candidates = candidates
        self.name = f"{base.name}+rerank"

    def search(self, query: str, k: int) -> list[Hit]:
        pool = self.base.search(query, self.candidates)
        # rangul inițial rămâne doar ca departajare la scoruri egale
        rescored = [Hit(h.chunk, self.reranker.score(query, h.chunk) - 1e-4 * rank) for rank, h in enumerate(pool)]
        rescored.sort(key=lambda h: -h.score)
        return rescored[:k]
