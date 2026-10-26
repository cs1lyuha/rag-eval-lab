"""Grila de experimente: chunking × retriever × reranking × top-k.

    python -m rag_lab.experiment                 # grila completă, offline (~10 s)
    python -m rag_lab.experiment --generator claude --config "section|hybrid|rerank|5"

Pentru fiecare configurație măsurăm calitatea retrieval-ului, calitatea
răspunsului (generator extractiv), costul estimat al generării cu un LLM
și latența retrieval-ului. Rezultatele ajung în results/.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_lab import metrics
from rag_lab.chunking import Chunk, build
from rag_lab.corpus import load_corpus
from rag_lab.dataset import Question, load_questions
from rag_lab.embeddings import CharNgramEmbedder, TfidfEmbedder
from rag_lab.generation import ExtractiveGenerator, Generator, estimate_tokens
from rag_lab.retrieval import (BM25Retriever, DenseRetriever, HybridRetriever, LexicalReranker,
                               RerankedRetriever, Retriever)

RESULTS = Path(__file__).resolve().parent.parent / "results"
CHUNKINGS = ["section", "fixed-50", "fixed-100", "fixed-200"]
RETRIEVERS = ["bm25", "tfidf", "char-ngram", "hybrid"]
KS = [1, 3, 5, 8]
# prețul de referință pentru cost: Claude Sonnet 5.5, USD / 1M tokeni
PRICE_IN, PRICE_OUT = 2.00, 10.00
SYSTEM_TOKENS = 60


@dataclass
class Row:
    chunking: str
    retriever: str
    rerank: bool
    k: int
    chunks_total: int
    hit_at_k: float
    mrr: float
    ctx_precision: float
    ctx_recall: float
    answer_recall: float
    answer_f1: float
    faithfulness: float
    abstain_ok: float      # întrebări fără răspuns refuzate corect
    false_abstain: float   # întrebări cu răspuns refuzate greșit
    ctx_tokens: float
    usd_per_1k: float
    latency_ms: float

    @property
    def key(self) -> str:
        return f"{self.chunking}|{self.retriever}|{'rerank' if self.rerank else 'no-rerank'}|{self.k}"


def make_retriever(name: str, chunks: list[Chunk], rerank: bool) -> tuple[Retriever, BM25Retriever]:
    bm25 = BM25Retriever(chunks)
    if name == "bm25":
        base: Retriever = bm25
    elif name == "tfidf":
        base = DenseRetriever(chunks, TfidfEmbedder())
    elif name == "char-ngram":
        base = DenseRetriever(chunks, CharNgramEmbedder())
    elif name == "hybrid":
        base = HybridRetriever([bm25, DenseRetriever(chunks, CharNgramEmbedder())])
    else:
        raise ValueError(name)
    if rerank:
        base = RerankedRetriever(base, LexicalReranker(bm25.idf), candidates=20)
    return base, bm25


def evaluate(chunking: str, retriever_name: str, rerank: bool, questions: list[Question],
             generator: Generator | None = None, ks: list[int] = KS) -> list[Row]:
    docs = load_corpus()
    chunks = build(docs, chunking)
    retriever, bm25 = make_retriever(retriever_name, chunks, rerank)
    generator = generator or ExtractiveGenerator(bm25.idf)

    ranked: dict[str, list[Chunk]] = {}
    timings = []
    for q in questions:
        t0 = time.perf_counter()
        ranked[q.id] = [h.chunk for h in retriever.search(q.question, max(ks))]
        timings.append((time.perf_counter() - t0) * 1000)
    latency = statistics.median(timings)

    rows = []
    answerable = [q for q in questions if q.answerable]
    unanswerable = [q for q in questions if not q.answerable]
    for k in ks:
        f1, recall, faith, abstain_ok, false_abstain, ctx_tok, out_tok = [], [], [], [], [], [], []
        for q in questions:
            top = ranked[q.id][:k]
            ans = generator.answer(q.question, top)
            ctx_tok.append(ans.input_tokens + SYSTEM_TOKENS)
            out_tok.append(ans.output_tokens)
            if q.answerable:
                f1.append(0.0 if ans.abstained else metrics.token_f1(ans.text, q.answer))
                recall.append(0.0 if ans.abstained else metrics.answer_recall(ans.text, q.answer))
                false_abstain.append(float(ans.abstained))
                if not ans.abstained:
                    faith.append(metrics.faithfulness(ans.text, "\n".join(c.text for c in top)))
            else:
                abstain_ok.append(float(ans.abstained))
        mean_in, mean_out = statistics.mean(ctx_tok), statistics.mean(out_tok)
        rows.append(Row(
            chunking, retriever_name, rerank, k, len(chunks),
            hit_at_k=statistics.mean(metrics.hit_at_k(ranked[q.id], q, k) for q in answerable),
            mrr=statistics.mean(metrics.reciprocal_rank(ranked[q.id][:k], q) for q in answerable),
            ctx_precision=statistics.mean(metrics.context_precision(ranked[q.id][:k], q) for q in answerable),
            ctx_recall=statistics.mean(metrics.context_recall(ranked[q.id][:k], q) for q in answerable),
            answer_recall=statistics.mean(recall),
            answer_f1=statistics.mean(f1),
            faithfulness=statistics.mean(faith) if faith else 0.0,
            abstain_ok=statistics.mean(abstain_ok) if abstain_ok else 0.0,
            false_abstain=statistics.mean(false_abstain),
            ctx_tokens=mean_in,
            usd_per_1k=1000 * (mean_in * PRICE_IN + mean_out * PRICE_OUT) / 1_000_000,
            latency_ms=latency,
        ))
    return rows


def by_type(questions: list[Question], chunking: str = "section", k: int = 3) -> dict[str, dict[str, float]]:
    """hit@k pe tipuri de întrebări, pentru fiecare retriever (fără reranking)."""
    docs = load_corpus()
    chunks = build(docs, chunking)
    types = sorted({q.type for q in questions if q.answerable})
    out: dict[str, dict[str, float]] = {}
    for name in RETRIEVERS:
        retriever, _ = make_retriever(name, chunks, rerank=False)
        out[name] = {}
        for t in types:
            qs = [q for q in questions if q.type == t]
            out[name][t] = statistics.mean(
                metrics.hit_at_k([h.chunk for h in retriever.search(q.question, k)], q, k) for q in qs)
    return out


def threshold_sweep(questions: list[Question], config: str,
                    thresholds=(0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8)) -> list[dict[str, float]]:
    """Cum se schimbă refuzurile când mutăm pragul generatorului extractiv."""
    chunking, retriever_name, rerank, k = config.split("|")
    chunks = build(load_corpus(), chunking)
    retriever, bm25 = make_retriever(retriever_name, chunks, rerank == "rerank")
    ranked = {q.id: [h.chunk for h in retriever.search(q.question, int(k))] for q in questions}
    out = []
    for th in thresholds:
        gen = ExtractiveGenerator(bm25.idf, threshold=th)
        answers = {q.id: gen.answer(q.question, ranked[q.id]) for q in questions}
        ans_q = [q for q in questions if q.answerable]
        un_q = [q for q in questions if not q.answerable]
        out.append({
            "threshold": th,
            "abstain_ok": statistics.mean(float(answers[q.id].abstained) for q in un_q),
            "false_abstain": statistics.mean(float(answers[q.id].abstained) for q in ans_q),
            "answer_recall": statistics.mean(
                0.0 if answers[q.id].abstained else metrics.answer_recall(answers[q.id].text, q.answer)
                for q in ans_q),
        })
    return out


def run_grid(questions: list[Question]) -> list[Row]:
    rows = []
    for chunking in CHUNKINGS:
        for retriever in RETRIEVERS:
            for rerank in (False, True):
                rows += evaluate(chunking, retriever, rerank, questions)
    return rows


def write_csv(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(r).items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", choices=["extractive", "claude"], default="extractive")
    parser.add_argument("--config", help='o singură configurație, de ex. "section|hybrid|rerank|5"')
    args = parser.parse_args()

    questions = load_questions(load_corpus())
    if args.config:
        chunking, retriever, rerank, k = args.config.split("|")
        generator = None
        if args.generator == "claude":
            from rag_lab.generation import ClaudeGenerator
            generator = ClaudeGenerator()
        rows = evaluate(chunking, retriever, rerank == "rerank", questions, generator, ks=[int(k)])
        for key, value in asdict(rows[0]).items():
            print(f"{key:>15}: {value:.3f}" if isinstance(value, float) else f"{key:>15}: {value}")
        return

    t0 = time.perf_counter()
    rows = run_grid(questions)
    write_csv(rows, RESULTS / "grid.csv")
    from rag_lab.report import write_report
    best = max(rows, key=lambda r: (r.answer_recall, r.hit_at_k, -r.ctx_tokens))
    write_report(rows, RESULTS / "summary.md", types=by_type(questions),
                 sweep=threshold_sweep(questions, best.key), sweep_config=best.key)
    print(f"{len(rows)} configurații evaluate în {time.perf_counter() - t0:.1f} s → results/grid.csv, results/summary.md")


if __name__ == "__main__":
    main()
