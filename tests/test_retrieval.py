import numpy as np
import pytest

from rag_lab.chunking import build
from rag_lab.corpus import load_corpus
from rag_lab.embeddings import CharNgramEmbedder, TfidfEmbedder
from rag_lab.retrieval import (BM25Retriever, DenseRetriever, HybridRetriever, LexicalReranker,
                               RerankedRetriever)
from rag_lab.text import strip_accents, tokens

CHUNKS = build(load_corpus(), "section")


def test_tokens_normalize_romanian():
    assert strip_accents("Știință") == "stiinta"
    assert tokens("Livrarea și livrării") == ["livrar", "livrar"]


@pytest.mark.parametrize("cls", [TfidfEmbedder, CharNgramEmbedder])
def test_embeddings_are_unit_vectors(cls):
    emb = cls().fit([c.text for c in CHUNKS])
    m = emb.embed([c.text for c in CHUNKS[:5]] + ["xyz"])
    assert np.allclose(np.linalg.norm(m[:5], axis=1), 1.0)


def test_char_ngrams_tolerate_typos():
    emb = CharNgramEmbedder().fit(["garanția produsului", "livrarea comenzii"])
    a, b, c = emb.embed(["garantia produsului", "garantzia produsuluii", "livrarea comenzii"])
    assert a @ b > a @ c


def top_doc(retriever, query):
    return retriever.search(query, 1)[0].chunk.doc


QUERIES = [
    ("Cât costă livrarea standard?", "livrare"),
    ("În câte zile pot returna un produs cumpărat online?", "retur"),
    ("Cât costă abonamentul TechNova Plus pentru studenți?", "tehnova-plus"),
    ("Ce taxă de depozitare se aplică pentru produsele nerevendicate din service?", "service"),
]


@pytest.mark.parametrize("query,doc", QUERIES)
def test_bm25_finds_obvious_documents(query, doc):
    assert top_doc(BM25Retriever(CHUNKS), query) == doc


@pytest.mark.parametrize("query,doc", QUERIES)
def test_dense_tfidf_finds_obvious_documents(query, doc):
    assert top_doc(DenseRetriever(CHUNKS, TfidfEmbedder()), query) == doc


def test_hybrid_merges_without_duplicates():
    hybrid = HybridRetriever([BM25Retriever(CHUNKS), DenseRetriever(CHUNKS, CharNgramEmbedder())])
    hits = hybrid.search("garanție baterie laptop", 10)
    ids = [h.chunk.id for h in hits]
    assert len(ids) == len(set(ids)) == 10
    assert hits[0].score >= hits[-1].score


def test_rrf_rewards_agreement():
    bm25 = BM25Retriever(CHUNKS)
    hybrid = HybridRetriever([bm25, bm25])
    assert hybrid.search("plata în rate", 1)[0].chunk.id == bm25.search("plata în rate", 1)[0].chunk.id


def test_reranker_prefers_full_coverage():
    bm25 = BM25Retriever(CHUNKS)
    rr = LexicalReranker(bm25.idf)
    good = next(c for c in CHUNKS if "Apple Pay" in c.text)
    other = next(c for c in CHUNKS if c.doc == "garantie")
    assert rr.score("limita pentru Apple Pay", good) > rr.score("limita pentru Apple Pay", other)


def test_reranked_retriever_returns_k_from_pool():
    bm25 = BM25Retriever(CHUNKS)
    rr = RerankedRetriever(bm25, LexicalReranker(bm25.idf), candidates=15)
    hits = rr.search("cum se resetează parola contului", 3)
    assert len(hits) == 3
    assert hits[0].chunk.doc == "cont-client"
