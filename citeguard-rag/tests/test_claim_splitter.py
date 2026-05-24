from citeguard.preprocess.claim_splitter import ClaimSplitter, split_into_claims


def test_splits_on_sentence_boundary():
    answer = "Water boils at 100C [doc1]. It freezes at 0C [doc1]."
    claims = split_into_claims("ex_x", answer)
    assert len(claims) == 2
    assert claims[0].claim_id == "ex_x_c001"
    assert claims[1].claim_id == "ex_x_c002"
    assert "boils" in claims[0].text
    assert claims[0].cited_doc_ids == ["doc1"]


def test_strips_citation_markers_from_text_for_embedding():
    answer = "Foo bar [doc2]."
    claims = split_into_claims("ex_y", answer)
    assert claims[0].text == "Foo bar."


def test_keeps_raw_text_with_citations():
    answer = "Foo bar [doc2]."
    claims = split_into_claims("ex_y", answer)
    assert "[doc2]" in claims[0].raw_text


def test_keeps_citations_when_disabled():
    s = ClaimSplitter(strip_citations_for_embedding=False)
    claims = s.split("ex_z", "Foo [doc1].")
    assert "[doc1]" in claims[0].text


def test_min_chars_filters_tiny_fragments():
    s = ClaimSplitter(min_chars=20)
    claims = s.split("ex_w", "Short. " * 5)
    assert claims == []


def test_handles_empty_answer():
    assert split_into_claims("ex_e", "") == []


def test_claim_ids_are_stable_and_sequential():
    text = "Sentence one is here. Sentence two follows. Sentence three concludes."
    claims = split_into_claims("ex_stable", text)
    assert [c.claim_id for c in claims] == ["ex_stable_c001", "ex_stable_c002", "ex_stable_c003"]
