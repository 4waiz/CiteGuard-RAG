"""End-to-end smoke test.

We patch the sentence-transformer cache with a fake encoder and disable NLI
so the test never touches the network. With those two pieces stubbed, the
pipeline still runs every other component end-to-end on real sample data.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from citeguard.config import CiteGuardConfig
from citeguard.data_loaders.custom_jsonl import load_jsonl
from citeguard.pipeline import CiteGuardPipeline
from citeguard.retrieval import vector_store as vs_module


SAMPLE = Path(__file__).resolve().parent.parent / "data" / "samples" / "custom_rag_examples.jsonl"


@pytest.fixture
def stub_sbert(monkeypatch, fake_sbert):
    # Make every VectorStore use the fake encoder regardless of model name.
    monkeypatch.setattr(vs_module, "get_sbert", lambda *a, **k: fake_sbert)
    yield


def _config_no_nli(tmp_path: Path) -> CiteGuardConfig:
    cfg = CiteGuardConfig()
    cfg.evaluators.nli.enabled = False
    cfg.evaluators.llm_judge.enabled = False
    cfg.output.tables_dir = str(tmp_path / "tables")
    cfg.output.figures_dir = str(tmp_path / "figures")
    cfg.output.cards_dir = str(tmp_path / "cards")
    cfg.output.reports_dir = str(tmp_path / "reports")
    return cfg


def test_pipeline_runs_end_to_end(stub_sbert, tmp_path):
    examples = load_jsonl(SAMPLE)
    assert examples, "sample data should contain examples"
    cfg = _config_no_nli(tmp_path)
    pipeline = CiteGuardPipeline(cfg)
    results = pipeline.run(examples, show_progress=False)

    assert results["evaluations"], "pipeline should produce at least one evaluation"
    assert results["example_summaries"]
    assert results["evidence_cards"]
    # Every evaluation must carry a predicted label from the canonical set.
    labels = {e.predicted_label for e in results["evaluations"]}
    from citeguard.schemas import LABELS

    assert labels.issubset(set(LABELS))


def test_pipeline_writes_csvs_and_cards(stub_sbert, tmp_path):
    examples = load_jsonl(SAMPLE)
    cfg = _config_no_nli(tmp_path)
    pipeline = CiteGuardPipeline(cfg)
    results = pipeline.run(examples, show_progress=False)
    paths = pipeline.write_outputs(results)

    assert paths["claim_eval"].exists()
    assert paths["example_summary"].exists()
    assert paths["evidence_cards"].exists()

    df = pd.read_csv(paths["claim_eval"])
    expected_cols = {
        "example_id", "claim_id", "claim_text", "predicted_label",
        "support_score", "latency_ms", "rationale",
    }
    assert expected_cols.issubset(df.columns)

    with open(paths["evidence_cards"], "r", encoding="utf-8") as f:
        cards = [json.loads(l) for l in f if l.strip()]
    assert cards
    for c in cards:
        assert "claim_id" in c
        assert "predicted_label" in c
