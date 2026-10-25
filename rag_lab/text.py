"""Normalizare de text pentru română: litere mici, fără diacritice, stemming prin prefix.

Stemming-ul e intenționat simplu (primele 6 litere): „livrarea”, „livrare”,
„livrării” → „livrar”. Pentru un corpus mic merge surprinzător de bine și e
ușor de explicat. Un stemmer Snowball ar fi pasul următor.
"""

from __future__ import annotations

import re
import unicodedata

STOPWORDS = set("""
a ai al ale am ar are as asa au ca cat cate cati ce cea cei cel cele cand care cat catre cu
cum da daca de din dintre doar dupa e ea ei el este eu fara fi fie fost iar il in inainte
intr intre isi la le li lor lui ma mai mi ne nici nu o ori pe pentru prin sa sau se si sub
sunt te tot toate toti un una unei unor unui va vor
""".split())

PREFIX = 6


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", strip_accents(text))


def tokens(text: str) -> list[str]:
    """Cuvinte normalizate, fără stopwords, tăiate la prefix."""
    return [w[:PREFIX] for w in words(text) if w not in STOPWORDS]
