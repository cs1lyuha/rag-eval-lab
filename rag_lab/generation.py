"""Generarea răspunsului din contextul recuperat.

- `ExtractiveGenerator` (offline, determinist): alege din context propoziția
  care acoperă cel mai bine întrebarea. Dacă nicio propoziție nu trece de
  prag, refuză („nu știu”), exact ce vrem la întrebările fără răspuns.
  Servește ca bază de comparație și rulează în CI.
- `ClaudeGenerator`: răspuns generat de Claude doar din context, cu output
  structurat `{answer, abstain}`. Rulează doar cu ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from rag_lab.chunking import Chunk
from rag_lab.retrieval import LexicalReranker
from rag_lab.text import tokens

ABSTAIN = "Nu am găsit informația în documentație."


@dataclass
class Answer:
    text: str
    abstained: bool
    input_tokens: int = 0
    output_tokens: int = 0


class Generator(Protocol):
    name: str

    def answer(self, question: str, chunks: list[Chunk]) -> Answer: ...


def estimate_tokens(text: str) -> int:
    """Aproximare grosieră: ~4 caractere pe token (română, tokenizer BPE)."""
    return max(1, len(text) // 4)


def sentences(text: str) -> list[str]:
    text = re.sub(r"^#+ .*$", "", text, flags=re.M)
    text = re.sub(r"^.+ › .+$", "", text, flags=re.M)  # contextual header
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.split()) >= 3]


class ExtractiveGenerator:
    name = "extractive"

    def __init__(self, idf: dict[str, float], threshold: float = 0.4, max_sentences: int = 2,
                 keep_ratio: float = 0.8) -> None:
        self.scorer = LexicalReranker(idf)
        self.threshold = threshold
        self.max_sentences = max_sentences
        self.keep_ratio = keep_ratio

    def answer(self, question: str, chunks: list[Chunk]) -> Answer:
        context = "\n\n".join(c.text for c in chunks)
        scored = sorted(((self.scorer.score(question, Chunk("s", "", 0, 0, s)), i, s)
                         for i, s in enumerate(dict.fromkeys(sentences(context)))), reverse=True)
        best_score = scored[0][0] if scored else 0.0
        abstained = best_score < self.threshold or not tokens(question)
        if abstained:
            text = ABSTAIN
        else:
            # a doua propoziție intră doar dacă e aproape la fel de relevantă; ordinea din text se păstrează
            picked = [(i, s) for score, i, s in scored[:self.max_sentences] if score >= self.keep_ratio * best_score]
            text = " ".join(s for _, s in sorted(picked))
        return Answer(text, abstained, estimate_tokens(question + context), estimate_tokens(text))


SYSTEM = (
    "Ești asistentul de suport TechNova. Răspunde scurt, în română, DOAR pe baza fragmentelor primite. "
    "Dacă fragmentele nu conțin răspunsul, setează abstain=true și nu ghici."
)

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "abstain": {"type": "boolean"}},
    "required": ["answer", "abstain"],
    "additionalProperties": False,
}


class ClaudeGenerator:
    def __init__(self, model: str = "claude-opus-5-5", client: Any = None) -> None:
        if client is None:  # pragma: no cover - doar live
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.name = model

    def answer(self, question: str, chunks: list[Chunk]) -> Answer:
        context = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, start=1))
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "low"},
            messages=[{"role": "user", "content": f"Fragmente:\n{context}\n\nÎntrebare: {question}"}],
        )
        if message.stop_reason == "refusal":
            return Answer(ABSTAIN, True, message.usage.input_tokens, message.usage.output_tokens)
        data = json.loads(next(b.text for b in message.content if b.type == "text"))
        abstained = bool(data["abstain"])
        return Answer(ABSTAIN if abstained else data["answer"], abstained,
                      message.usage.input_tokens, message.usage.output_tokens)
