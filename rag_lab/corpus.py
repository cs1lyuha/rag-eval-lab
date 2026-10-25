"""Încărcarea documentelor din kb/*.md și împărțirea lor pe secțiuni.

Fiecare secțiune `## Titlu` primește un id stabil `document#titlu-slug`,
folosit apoi în citări și în setul de evaluare.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent / "kb"


@dataclass(frozen=True)
class Section:
    doc: str          # numele fișierului fără .md
    title: str        # titlul secțiunii
    text: str         # textul secțiunii, fără titlu
    start: int        # poziția în textul complet al documentului

    @property
    def id(self) -> str:
        return f"{self.doc}#{slugify(self.title)}"


@dataclass(frozen=True)
class Document:
    name: str
    title: str
    text: str
    sections: tuple[Section, ...]


def slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.lower())
    value = "".join(c for c in value if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")


def parse_document(name: str, raw: str) -> Document:
    title_match = re.search(r"^# (.+)$", raw, flags=re.M)
    title = title_match.group(1).strip() if title_match else name
    sections = []
    headings = list(re.finditer(r"^## (.+)$", raw, flags=re.M))
    for i, h in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(raw)
        body = raw[h.end():end].strip()
        sections.append(Section(name, h.group(1).strip(), body, h.end()))
    return Document(name, title, raw, tuple(sections))


def load_corpus(kb_dir: Path = KB_DIR) -> list[Document]:
    return [parse_document(p.stem, p.read_text(encoding="utf-8")) for p in sorted(kb_dir.glob("*.md"))]
