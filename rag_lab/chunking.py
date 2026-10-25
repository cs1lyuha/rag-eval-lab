"""Strategii de chunking.

Fiecare chunk ține minte pozițiile (caracter de început și de sfârșit) în
documentul original. Așa evaluarea poate verifica exact dacă un chunk
conține pasajul-dovadă al unei întrebări, indiferent de strategie.

Strategii:
- `fixed`: fereastră glisantă de N cuvinte cu overlap, ignoră structura;
- `section`: o secțiune `##` = un chunk; secțiunile lungi se taie în ferestre.

Opțional, `headers=True` pune în fața textului „Document › Secțiune”
(contextual chunk headers), ca un chunk scurt să nu-și piardă contextul.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_lab.corpus import Document

WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    start: int
    end: int
    text: str            # textul trimis la indexare și la model
    section: str = ""

    @property
    def words(self) -> int:
        return len(self.text.split())


def _section_at(doc: Document, pos: int) -> str:
    title = ""
    for s in doc.sections:
        if s.heading_start <= pos:
            title = s.title
    return title


def _windows(doc: Document, lo: int, hi: int, size: int, overlap: int) -> list[tuple[int, int]]:
    words = [m.span() for m in WORD.finditer(doc.text, lo, hi)]
    if not words:
        return []
    step = max(1, size - overlap)
    spans = []
    for i in range(0, len(words), step):
        window = words[i:i + size]
        spans.append((window[0][0], window[-1][1]))
        if i + size >= len(words):
            break
    return spans


def _make(doc: Document, spans: list[tuple[int, int]], prefix: str, headers: bool) -> list[Chunk]:
    chunks = []
    for n, (a, b) in enumerate(spans):
        section = _section_at(doc, a)
        body = doc.text[a:b]
        text = f"{doc.title} › {section}\n{body}" if headers and section else body
        chunks.append(Chunk(f"{doc.name}:{prefix}{n}", doc.name, a, b, text, section))
    return chunks


def fixed(docs: list[Document], size: int, overlap: int = 0, headers: bool = False) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap trebuie să fie mai mic decât size")
    out: list[Chunk] = []
    for doc in docs:
        body_start = doc.sections[0].heading_start if doc.sections else 0
        out += _make(doc, _windows(doc, body_start, len(doc.text), size, overlap), "f", headers)
    return out


def by_section(docs: list[Document], max_words: int = 400, overlap: int = 20,
               headers: bool = False) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        spans = []
        for s in doc.sections:
            spans += _windows(doc, s.start, s.end, max_words, min(overlap, max_words - 1))
        out += _make(doc, spans, "s", headers)
    return out


def build(docs: list[Document], strategy: str, headers: bool = False) -> list[Chunk]:
    """Strategii după nume, folosite în grila de experimente: `section`, `fixed-100`, ..."""
    if strategy == "section":
        return by_section(docs, headers=headers)
    if strategy.startswith("fixed-"):
        size = int(strategy.split("-")[1])
        return fixed(docs, size, overlap=size // 5, headers=headers)
    raise ValueError(f"strategie necunoscută: {strategy}")
