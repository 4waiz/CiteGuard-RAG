from citeguard.metrics.citation import citation_metrics
from citeguard.metrics.classification import classification_report_dict, confusion_matrix
from citeguard.metrics.retrieval import retrieval_recall_at_k
from citeguard.schemas import LABELS, SUPPORTED, UNSUPPORTED


def test_classification_report_perfect():
    rep = classification_report_dict([SUPPORTED, UNSUPPORTED], [SUPPORTED, UNSUPPORTED])
    assert rep["macro_f1"] > 0.0
    assert rep["per_label"][SUPPORTED]["f1"] == 1.0


def test_classification_report_empty():
    rep = classification_report_dict([], [])
    assert rep["macro_f1"] == 0.0
    for label in LABELS:
        assert rep["per_label"][label]["f1"] == 0.0


def test_confusion_matrix_shape():
    cm, labels = confusion_matrix([SUPPORTED], [SUPPORTED])
    assert cm.shape == (len(LABELS), len(LABELS))
    assert labels == LABELS


def test_citation_metrics_perfect():
    cited = [["doc1"], ["doc2"]]
    gold = [["doc1"], ["doc2"]]
    m = citation_metrics(cited, gold)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.missing_citation_rate == 0.0
    assert m.citation_mismatch_rate == 0.0


def test_citation_metrics_missing_and_mismatch():
    cited = [[], ["doc3"]]   # one missing, one wrong
    gold = [["doc1"], ["doc2"]]
    m = citation_metrics(cited, gold)
    assert m.missing_citation_rate == 0.5
    assert m.citation_mismatch_rate == 0.5
    assert m.precision == 0.0  # only the second claim contributes, and it's wrong


def test_retrieval_recall_at_k():
    retrieved = [["doc1", "doc2"], ["doc3", "doc4"]]
    gold = [["doc1"], ["doc9"]]
    r = retrieval_recall_at_k(retrieved, gold, k=2)
    assert r == 0.5


def test_retrieval_recall_ignores_empty_gold():
    retrieved = [["doc1"], ["doc2"]]
    gold = [["doc1"], []]
    r = retrieval_recall_at_k(retrieved, gold, k=1)
    assert r == 1.0
