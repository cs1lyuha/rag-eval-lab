"""Setul de evaluare: întrebări, răspunsuri de referință și pasaje-dovadă.

Fiecare întrebare are `evidence` = una sau mai multe fraze copiate exact
din kb/. La încărcare fiecare frază e localizată în corpus (document +
poziție), ca metricile să lucreze pe intervale de caractere, nu pe
„chunk-ul corect”. Altfel setul ar trebui refăcut la fiecare strategie
de chunking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag_lab.corpus import Document

DATA = Path(__file__).resolve().parent.parent / "data" / "questions.jsonl"


@dataclass(frozen=True)
class Span:
    doc: str
    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    question: str
    answer: str
    evidence: tuple[str, ...]
    spans: tuple[Span, ...]

    @property
    def answerable(self) -> bool:
        return bool(self.spans)


class EvidenceNotFound(ValueError):
    pass


def locate(evidence: str, docs: list[Document]) -> Span:
    matches = [(d.name, d.text.find(evidence)) for d in docs if evidence in d.text]
    if len(matches) != 1:
        raise EvidenceNotFound(f"dovada apare de {len(matches)} ori în corpus: {evidence[:60]!r}")
    name, pos = matches[0]
    return Span(name, pos, pos + len(evidence))


def load_questions(docs: list[Document], path: Path = DATA) -> list[Question]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        spans = tuple(locate(e, docs) for e in row["evidence"])
        out.append(Question(row["id"], row["type"], row["question"], row["answer"], tuple(row["evidence"]), spans))
    return out
