from rag_lab.corpus import load_corpus
from rag_lab.dataset import load_questions
from rag_lab.experiment import by_type, evaluate, threshold_sweep
from rag_lab.report import pareto, write_report

QUESTIONS = load_questions(load_corpus())


def test_evaluate_returns_one_row_per_k_with_sane_values():
    rows = evaluate("section", "bm25", False, QUESTIONS, ks=[1, 5])
    assert [r.k for r in rows] == [1, 5]
    for r in rows:
        for attr in ("hit_at_k", "mrr", "ctx_precision", "ctx_recall", "answer_recall", "abstain_ok"):
            assert 0.0 <= getattr(r, attr) <= 1.0
    assert rows[1].hit_at_k >= rows[0].hit_at_k
    assert rows[1].ctx_tokens > rows[0].ctx_tokens
    assert rows[1].usd_per_1k > rows[0].usd_per_1k


def test_retrieval_quality_floor():
    """Regresie: dacă o schimbare strică retrieval-ul, CI-ul pică."""
    row = evaluate("section", "hybrid", False, QUESTIONS, ks=[5])[0]
    assert row.hit_at_k >= 0.95
    assert row.mrr >= 0.85


def test_by_type_and_sweep_shapes():
    types = by_type(QUESTIONS)
    assert set(types) == {"bm25", "tfidf", "char-ngram", "hybrid"}
    sweep = threshold_sweep(QUESTIONS, "section|bm25|no-rerank|3", thresholds=(0.2, 0.9))
    assert sweep[0]["false_abstain"] <= sweep[1]["false_abstain"]


def test_report_and_pareto(tmp_path):
    rows = evaluate("section", "bm25", False, QUESTIONS, ks=[1, 3])
    front = pareto(rows)
    assert front and all(r in rows for r in front)
    out = tmp_path / "summary.md"
    write_report(rows, out)
    assert "Frontiera Pareto" in out.read_text(encoding="utf-8")
