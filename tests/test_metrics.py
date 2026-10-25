import pytest

from rag_lab.chunking import Chunk
from rag_lab.dataset import Question, Span
from rag_lab.metrics import (context_precision, context_recall, faithfulness, hit_at_k, is_relevant,
                             reciprocal_rank, token_f1)

Q = Question("q", "direct", "?", "răspuns", ("x",), (Span("d", 100, 140),))
Q2 = Question("q", "multi", "?", "", ("x", "y"), (Span("d", 100, 140), Span("e", 0, 20)))


def ch(doc, a, b, cid=None):
    return Chunk(cid or f"{doc}:{a}", doc, a, b, "t")


def test_relevance_needs_half_of_the_evidence():
    assert is_relevant(ch("d", 90, 125), Q)       # 25/40 caractere
    assert not is_relevant(ch("d", 130, 200), Q)  # 10/40
    assert not is_relevant(ch("x", 100, 140), Q)  # alt document


def test_small_chunk_inside_long_evidence_counts():
    assert is_relevant(ch("d", 105, 115), Q)


def test_hit_and_mrr():
    ranked = [ch("d", 0, 50), ch("d", 300, 400), ch("d", 95, 150)]
    assert hit_at_k(ranked, Q, 2) == 0.0
    assert hit_at_k(ranked, Q, 3) == 1.0
    assert reciprocal_rank(ranked, Q) == pytest.approx(1 / 3)


def test_context_precision_counts_overlapping_chunks():
    assert context_precision([ch("d", 95, 150), ch("d", 0, 50)], Q) == 0.5
    assert context_precision([], Q) == 0.0


def test_context_recall_partial_and_union():
    assert context_recall([ch("d", 100, 120)], Q) == pytest.approx(0.5)
    assert context_recall([ch("d", 100, 120), ch("d", 110, 140)], Q) == pytest.approx(1.0)
    assert context_recall([ch("d", 100, 140)], Q2) == pytest.approx(40 / 60)


def test_recall_for_unanswerable_is_one():
    assert context_recall([], Question("u", "unanswerable", "?", "", (), ())) == 1.0


def test_token_f1():
    assert token_f1("50 de lei", "Livrarea costă 50 de lei") == pytest.approx(0.75)
    assert token_f1("nimic comun", "altceva") == 0.0
    assert token_f1("", "") == 1.0


def test_faithfulness_penalizes_invented_facts():
    ctx = "Livrarea standard costă 50 de lei."
    assert faithfulness("Livrarea standard costă 50 de lei.", ctx) == 1.0
    assert faithfulness("Livrarea expres costă 120 de lei prin DHL.", ctx) <= 0.5
