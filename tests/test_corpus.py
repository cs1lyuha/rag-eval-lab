from rag_lab.corpus import load_corpus, parse_document, slugify


def test_corpus_has_twelve_documents_with_sections():
    docs = load_corpus()
    assert len(docs) == 12
    assert all(len(d.sections) >= 4 for d in docs)


def test_section_ids_are_unique_and_stable():
    ids = [s.id for d in load_corpus() for s in d.sections]
    assert len(ids) == len(set(ids))
    assert "livrare#costul-livrarii" in ids


def test_slugify_strips_diacritics():
    assert slugify("Ștergerea contului") == "stergerea-contului"


def test_parse_document_splits_on_h2():
    doc = parse_document("x", "# T\n\n## A\nunu\n\n## B\ndoi\n")
    assert [s.title for s in doc.sections] == ["A", "B"]
    assert doc.sections[1].text == "doi"
