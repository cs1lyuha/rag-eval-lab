import pytest

from rag_lab.chunking import build, by_section, fixed
from rag_lab.corpus import load_corpus, parse_document

DOCS = load_corpus()
SMALL = parse_document("d", "# Doc\n\n## A\n" + " ".join(f"w{i}" for i in range(25)) + "\n\n## B\nunu doi trei\n")


def test_chunk_offsets_point_back_to_original_text():
    for chunk in fixed(DOCS, 50, 10) + by_section(DOCS):
        doc = next(d for d in DOCS if d.name == chunk.doc)
        assert doc.text[chunk.start:chunk.end] == chunk.text


def test_fixed_window_sizes_and_overlap():
    chunks = fixed([SMALL], size=10, overlap=2)
    assert all(c.words <= 10 for c in chunks)
    # cuvintele de la finalul unui chunk apar la începutul următorului
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-2:] == second[:2]


def test_fixed_covers_every_word():
    chunks = fixed([SMALL], size=7, overlap=0)
    covered = " ".join(c.text for c in chunks).split()
    assert "w24" in covered and "trei" in covered


def test_section_chunks_do_not_cross_sections():
    chunks = by_section([SMALL], max_words=100)
    assert [c.section for c in chunks] == ["A", "B"]
    assert "unu" not in chunks[0].text


def test_long_section_is_split():
    chunks = by_section([SMALL], max_words=10, overlap=0)
    assert len([c for c in chunks if c.section == "A"]) == 3


def test_headers_prefix_document_and_section():
    chunk = by_section([SMALL], headers=True)[1]
    assert chunk.text.startswith("Doc › B\n")


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        fixed(DOCS, 10, 10)


@pytest.mark.parametrize("strategy", ["section", "fixed-50", "fixed-100", "fixed-200"])
def test_named_strategies(strategy):
    chunks = build(DOCS, strategy)
    assert len(chunks) > 12
    assert len({c.id for c in chunks}) == len(chunks)


def test_bigger_chunks_mean_fewer_chunks():
    assert len(build(DOCS, "fixed-50")) > len(build(DOCS, "fixed-100")) > len(build(DOCS, "fixed-200"))
