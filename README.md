---
title: CiteGuard-RAG
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.39.0"
app_file: app.py
pinned: false
license: mit
short_description: Claim-level diagnosis for retrieval-augmented generation outputs.
---

# CiteGuard-RAG

A lightweight, reproducible toolkit for **claim-level diagnosis** of
retrieval-augmented generation (RAG) outputs. CiteGuard-RAG looks at a
generated answer plus its retrieved context and labels each claim with one
of:

| Label                | Meaning                                                                 |
|----------------------|-------------------------------------------------------------------------|
| `SUPPORTED`          | A cited document entails the claim.                                     |
| `UNSUPPORTED`        | Neither cited nor retrieved evidence supports the claim.                |
| `CONTRADICTED`       | Cited or retrieved evidence contradicts the claim.                      |
| `CITATION_MISMATCH`  | Citation exists, but a *different* retrieved doc actually supports it.  |
| `SUPPORTED_UNCITED`  | No citation, but retrieved evidence does support the claim.             |
| `NOT_ENOUGH_EVIDENCE`| Related evidence exists but does not clearly entail or contradict.      |
| `PARSER_UNCERTAIN`   | Citation parser could not interpret the citation marker.                |

> CiteGuard-RAG is an **audit aid**, not a truth oracle. Outputs reflect what
> the configured retrieval, semantic similarity, and NLI models believe —
> not ground truth.

## Project layout

```
citeguard-rag/
  configs/                 # default + ablation YAML configs
  data/samples/            # synthetic JSONL examples (clearly labeled)
  outputs/                 # CSV, JSONL, PNG, HTML report outputs
  src/citeguard/           # library code
  app/streamlit_app.py     # dashboard
  scripts/                 # CLI helpers (benchmark, perturbations, plots)
  tests/                   # pytest suite
  notebooks/               # error analysis notebook
```

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
# editable install for the CLI entry point
pip install -e .
```

The first call to the sentence-transformer and NLI models downloads weights
from the Hugging Face Hub. After that, runs are fully local.

## Quick start

```bash
# 1) inspect the sample data
python scripts/download_sample_data.py

# 2) run the full pipeline on sample data
python -m citeguard.cli evaluate \
  --input data/samples/custom_rag_examples.jsonl \
  --config configs/default.yaml \
  --output outputs/

# 3) generate the six paper figures
python scripts/make_figures.py

# 4) launch the dashboard
streamlit run app/streamlit_app.py
```

## Input format

Each line of the JSONL is one example:

```json
{
  "example_id": "ex_001",
  "dataset": "custom",
  "question": "What is the boiling point of water at sea level?",
  "answer": "Water boils at 100 degrees Celsius [doc1].",
  "contexts": [
    {"doc_id": "doc1", "title": "Water properties", "text": "Water boils at 100C..."}
  ],
  "gold_claim_labels": [
    {"claim_text": "Water boils at 100 degrees Celsius.", "label": "SUPPORTED", "support_doc_ids": ["doc1"]}
  ]
}
```

`gold_claim_labels` is optional — without it, the pipeline still produces
predicted labels but classification metrics are zero-valued.

Citation markers `[doc1]`, `[1]`, `[doc1, doc2]` are all recognized. Numeric
citations like `[1]` are normalized to `doc1` so they line up with `doc_id`.

## Running on other datasets

`src/citeguard/data_loaders/{ragtruth,alce,hotpotqa,qasper}.py` are skeleton
loaders. They expect each dataset to be converted to the universal JSONL
schema above. We do **not** redistribute the source datasets — download
them from the original authors and convert before loading.

## Running the dashboard

```bash
streamlit run app/streamlit_app.py
```

Pages:

- **Overview** — totals, predicted-label distribution, per-example summary.
- **Error distribution** — stacked bar of correct/incorrect per label.
- **Claim inspection** — filterable table with download.
- **Evidence cards** — full per-claim view: question, answer, citations,
  best evidence, NLI label, semantic score, rationale.

## Outputs

After `citeguard evaluate`:

| Path                                         | Contents                                |
|----------------------------------------------|------------------------------------------|
| `outputs/tables/claim_eval.csv`              | One row per claim with all signals.      |
| `outputs/tables/example_summary.csv`         | Per-example rates and recall@5.          |
| `outputs/tables/aggregate_metrics.json`      | Macro/weighted F1, confusion matrix.     |
| `outputs/evidence_cards/evidence_cards.jsonl`| Full per-claim evidence object.          |

After `scripts/make_figures.py`:

- `outputs/figures/fig_error_distribution.png`
- `outputs/figures/fig_confusion_matrix.png`
- `outputs/figures/fig_baseline_comparison.png`
- `outputs/figures/fig_ablation.png`
- `outputs/figures/fig_retrieval_vs_support.png`
- `outputs/figures/fig_runtime.png`

After `scripts/run_benchmark.py`:

- `outputs/tables/benchmark_summary.csv` and `.json`
- Per-run outputs under `outputs/default/`, `outputs/ablation_*/`

## Metrics

- **Macro-F1, weighted-F1, per-label P/R/F1** — computed only over claims
  with a `gold_label`.
- **Citation precision / recall / mismatch / missing rate** — computed
  per example and aggregated.
- **Retrieval recall@k (default k=5)** — fraction of gold support doc ids
  found in the top-k retrieved doc ids.
- **Runtime** — per-claim latency in ms, captured at evaluation time.

## Configuration

`configs/default.yaml` controls models, retrieval, evaluator toggles, and
diagnostic thresholds. Two ablations are provided:

- `ablation_semantic_only.yaml` — disables NLI.
- `ablation_nli_only.yaml` — disables dense retrieval.

`scripts/run_benchmark.py` runs all three and writes a benchmark summary.

## Optional LLM judge

`src/citeguard/evaluators/llm_judge.py` is a **stub**, intentionally
disabled by default. Enabling it requires both a config flag and an
explicit model name. The shipped stub raises rather than fabricating a
label — implement a real client before using it.

## Tests

```bash
pytest
```

The smoke test stubs out the sentence-transformer and disables NLI so it
runs offline.

## Limitations

- The rule-based claim splitter is naive: it splits on `.!?` and may miss
  intra-sentence claim boundaries.
- The hybrid rules use fixed thresholds. They are *defaults*, not tuned
  numbers — tune on held-out data before reporting in a paper.
- All shipped sample data is **synthetic** and clearly marked
  (`is_synthetic: true`, dataset = `custom`).
- We do not report benchmark numbers in this README. Run the pipeline on
  your own data to generate real results.

## Ethical note

CiteGuard-RAG is intended as an audit and triage aid for RAG developers
and researchers. It is **not** a truth oracle, and its predicted labels
reflect noisy model signals. Do not use it as the sole basis for
downstream decisions about whether a model output is factually correct.

## License

MIT — see `LICENSE`.
