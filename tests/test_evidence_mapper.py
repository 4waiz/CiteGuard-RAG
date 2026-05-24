from citeguard.preprocess.claim_splitter import split_into_claims
from citeguard.retrieval.evidence_mapper import EvidenceMapper
from citeguard.retrieval.vector_store import VectorStore
from citeguard.schemas import Context


def _contexts():
    return [
        Context(doc_id="doc1", title="Water", text="Water boils at 100 degrees Celsius at sea level."),
        Context(doc_id="doc2", title="Ice", text="Ice melts at zero degrees Celsius at standard pressure."),
        Context(doc_id="doc3", title="Steam", text="Steam is the gaseous phase of water."),
    ]


def test_mapper_collects_cited_doc_text(fake_sbert):
    ctx = _contexts()
    claims = split_into_claims("ex_m", "Water boils at 100C [doc1].")
    vs = VectorStore(ctx, sbert=fake_sbert)
    mapper = EvidenceMapper(ctx, top_k=3, use_bm25=True, use_dense=True, vector_store=vs)
    mapping = mapper.map(claims[0])
    cited = mapping.cited_chunks
    assert len(cited) == 1
    assert cited[0].doc_id == "doc1"
    assert "boils" in cited[0].text


def test_mapper_returns_retrieved_chunks_for_uncited_claim(fake_sbert):
    ctx = _contexts()
    claims = split_into_claims("ex_m2", "Water boils at 100 degrees Celsius.")
    vs = VectorStore(ctx, sbert=fake_sbert)
    mapper = EvidenceMapper(ctx, top_k=3, use_bm25=True, use_dense=True, vector_store=vs)
    mapping = mapper.map(claims[0])
    assert mapping.cited_chunks == []
    retrieved_ids = mapping.top_evidence_doc_ids()
    assert "doc1" in retrieved_ids


def test_mapper_handles_dangling_citation(fake_sbert):
    ctx = _contexts()
    claims = split_into_claims("ex_m3", "Some unrelated claim [doc999].")
    vs = VectorStore(ctx, sbert=fake_sbert)
    mapper = EvidenceMapper(ctx, top_k=3, use_bm25=True, use_dense=True, vector_store=vs)
    mapping = mapper.map(claims[0])
    # Dangling citation produces a placeholder chunk with empty text.
    assert len(mapping.cited_chunks) == 1
    assert mapping.cited_chunks[0].doc_id == "doc999"
    assert mapping.cited_chunks[0].text == ""
