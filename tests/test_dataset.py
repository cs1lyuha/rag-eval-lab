from collections import Counter

import pytest

from rag_lab.corpus import load_corpus
from rag_lab.dataset import EvidenceNotFound, load_questions, locate

DOCS = load_corpus()
QUESTIONS = load_questions(DOCS)


def test_at_least_fifty_answerable_questions():
    assert sum(q.answerable for q in QUESTIONS) >= 50


def test_ids_unique():
    assert len({q.id for q in QUESTIONS}) == len(QUESTIONS)


def test_every_evidence_points_to_exact_text():
    for q in QUESTIONS:
        for evidence, span in zip(q.evidence, q.spans):
            doc = next(d for d in DOCS if d.name == span.doc)
            assert doc.text[span.start:span.end] == evidence


def test_question_mix():
    types = Counter(q.type for q in QUESTIONS)
    assert types["paraphrase"] >= 15
    assert types["unanswerable"] >= 5
    assert types["multi"] >= 3


def test_every_document_is_covered():
    assert {s.doc for q in QUESTIONS for s in q.spans} == {d.name for d in DOCS}


def test_locate_rejects_missing_evidence():
    with pytest.raises(EvidenceNotFound):
        locate("această frază nu există", DOCS)
