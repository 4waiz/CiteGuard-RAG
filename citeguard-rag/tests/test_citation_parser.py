from citeguard.preprocess.citation_parser import CitationParser, parse_citations_in_sentence


def test_named_citation():
    ids, uncertain = parse_citations_in_sentence("Foo bar [doc1].")
    assert ids == ["doc1"]
    assert uncertain is False


def test_numeric_citation_normalized_to_doc_prefix():
    ids, _ = parse_citations_in_sentence("Foo bar [1] and [2].")
    assert ids == ["doc1", "doc2"]


def test_multi_citation_inside_brackets():
    ids, _ = parse_citations_in_sentence("Foo [doc1, doc2] bar.")
    assert ids == ["doc1", "doc2"]


def test_no_citation_is_not_uncertain():
    ids, uncertain = parse_citations_in_sentence("Plain sentence with no brackets.")
    assert ids == []
    assert uncertain is False


def test_unparseable_bracket_marks_uncertain():
    ids, uncertain = parse_citations_in_sentence("Confusing [see Fig 2].")
    assert ids == []
    assert uncertain is True


def test_empty_bracket_marks_uncertain():
    ids, uncertain = parse_citations_in_sentence("Trailing bracket [].")
    assert ids == []
    assert uncertain is True


def test_dedup_in_one_sentence():
    ids, _ = parse_citations_in_sentence("Mix [doc1] and [doc1].")
    assert ids == ["doc1"]


def test_custom_numeric_prefix():
    p = CitationParser(numeric_prefix="ref")
    ids, _ = p.parse("Cited as [3].")
    assert ids == ["ref3"]
