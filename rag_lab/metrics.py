"""Metrici de retrieval și de răspuns.

Retrieval (pe intervale de caractere, independent de chunking):
- un chunk e *relevant* dacă acoperă cel puțin jumătate dintr-o dovadă
  (sau dovada e mai lungă decât chunk-ul și chunk-ul e în întregime în ea);
- hit@k: există măcar un chunk relevant în top k;
- MRR: 1 / rangul primului chunk relevant;
- context precision: ce fracție din chunk-urile trimise e relevantă;
- context recall: ce fracție din caracterele dovezilor e acoperită.

Răspuns:
- token F1 față de răspunsul de referință (ca la SQuAD);
- faithfulness (aproximare fără LLM): ce fracție din cuvintele de conținut
  ale răspunsului apar în contextul primit; un răspuns inventat are scor mic.
"""

from __future__ import annotations

from collections import Counter

from rag_lab.chunking import Chunk
from rag_lab.dataset import Question, Span
from rag_lab.text import tokens, words


def overlap(chunk: Chunk, span: Span) -> int:
    if chunk.doc != span.doc:
        return 0
    return max(0, min(chunk.end, span.end) - max(chunk.start, span.start))


def is_relevant(chunk: Chunk, q: Question) -> bool:
    for span in q.spans:
        ov = overlap(chunk, span)
        if ov >= len(span) / 2 or (ov > 0 and ov == chunk.end - chunk.start):
            return True
    return False


def hit_at_k(chunks: list[Chunk], q: Question, k: int) -> float:
    return float(any(is_relevant(c, q) for c in chunks[:k]))


def reciprocal_rank(chunks: list[Chunk], q: Question) -> float:
    for rank, c in enumerate(chunks, start=1):
        if is_relevant(c, q):
            return 1 / rank
    return 0.0


def context_precision(chunks: list[Chunk], q: Question) -> float:
    if not chunks:
        return 0.0
    return sum(any(overlap(c, s) > 0 for s in q.spans) for c in chunks) / len(chunks)


def context_recall(chunks: list[Chunk], q: Question) -> float:
    total = sum(len(s) for s in q.spans)
    if total == 0:
        return 1.0
    covered = 0
    for span in q.spans:
        marks = bytearray(len(span))
        for c in chunks:
            if c.doc != span.doc:
                continue
            a, b = max(c.start, span.start), min(c.end, span.end)
            for i in range(a - span.start, b - span.start):
                marks[i] = 1
        covered += sum(marks)
    return covered / total


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = words(prediction), words(reference)
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def faithfulness(answer: str, context: str) -> float:
    content = [t for t in tokens(answer) if len(t) > 2 or t.isdigit()]
    if not content:
        return 1.0
    ctx = set(tokens(context))
    return sum(t in ctx for t in content) / len(content)
