"""CiteGuard-RAG Streamlit dashboard.

Loads ``outputs/tables/claim_eval.csv`` and
``outputs/evidence_cards/evidence_cards.jsonl`` and lets the user explore
predicted labels, filter by label, and inspect per-claim evidence cards.

Run with:
    streamlit run dashboard/streamlit_app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


def _pick(*candidates: str) -> Path:
    """Return the first candidate that exists, else the first candidate.

    Lets us prefer fresh ``outputs/`` produced by a local run, and fall back
    to the committed ``sample_outputs/`` so the Hugging Face Space dashboard
    has data the moment it boots.
    """
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return Path(candidates[0])


DEFAULT_CLAIM_EVAL = _pick("outputs/tables/claim_eval.csv", "sample_outputs/tables/claim_eval.csv")
DEFAULT_CARDS = _pick("outputs/evidence_cards/evidence_cards.jsonl", "sample_outputs/evidence_cards/evidence_cards.jsonl")
DEFAULT_SUMMARY = _pick("outputs/tables/example_summary.csv", "sample_outputs/tables/example_summary.csv")


@st.cache_data
def load_claim_eval(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_cards(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


@st.cache_data
def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def overview(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    st.header("Overview")
    if df.empty:
        st.warning("claim_eval.csv not found. Run `citeguard evaluate` first.")
        return
    cols = st.columns(4)
    cols[0].metric("Total claims", len(df))
    cols[1].metric("Examples", df["example_id"].nunique() if "example_id" in df else 0)
    has_gold = df["gold_label"].astype(str).str.len() > 0 if "gold_label" in df else pd.Series(False)
    n_gold = int(has_gold.sum())
    cols[2].metric("Claims with gold label", n_gold)
    if n_gold:
        acc = float((df.loc[has_gold, "is_correct"].astype(bool)).mean())
        cols[3].metric("Accuracy (vs gold)", f"{acc:.2%}")
    else:
        cols[3].metric("Accuracy (vs gold)", "n/a")

    st.subheader("Predicted label distribution")
    dist = df["predicted_label"].value_counts().rename_axis("label").reset_index(name="count")
    st.bar_chart(dist.set_index("label"))

    if not summary.empty:
        st.subheader("Per-example summary")
        st.dataframe(summary, use_container_width=True)


def error_distribution(df: pd.DataFrame) -> None:
    st.header("Error distribution")
    if df.empty:
        st.info("No data.")
        return
    has_gold = df["gold_label"].astype(str).str.len() > 0
    sub = df.loc[has_gold].copy()
    if sub.empty:
        st.info("No gold labels available; cannot show errors. Showing prediction counts instead.")
        st.bar_chart(df["predicted_label"].value_counts())
        return
    sub["correct"] = sub["is_correct"].astype(bool)
    grouped = (
        sub.groupby(["predicted_label", "correct"]).size().unstack(fill_value=0)
    )
    st.bar_chart(grouped)
    st.caption("Stacked: True = predicted matched gold; False = misclassification.")


def claim_inspection(df: pd.DataFrame) -> None:
    st.header("Claim inspection table")
    if df.empty:
        st.info("No data.")
        return
    labels = sorted(df["predicted_label"].dropna().unique().tolist())
    chosen = st.multiselect("Filter by predicted label", labels, default=labels)
    examples = sorted(df["example_id"].dropna().unique().tolist())
    ex_chosen = st.multiselect("Filter by example_id", examples, default=examples)

    view = df[df["predicted_label"].isin(chosen) & df["example_id"].isin(ex_chosen)].copy()
    show_cols = [
        "example_id", "claim_id", "claim_text", "cited_doc_ids",
        "predicted_label", "gold_label", "support_score",
        "semantic_score", "nli_label", "entail_prob", "contradict_prob",
        "best_evidence_doc_id", "latency_ms",
    ]
    show_cols = [c for c in show_cols if c in view.columns]
    st.dataframe(view[show_cols], use_container_width=True)

    st.download_button(
        "Download filtered CSV",
        view.to_csv(index=False).encode("utf-8"),
        file_name="claim_eval_filtered.csv",
        mime="text/csv",
    )


def card_viewer(cards: list[dict]) -> None:
    st.header("Evidence card viewer")
    if not cards:
        st.info("No evidence cards found.")
        return
    labels = sorted({c.get("predicted_label", "") for c in cards if c.get("predicted_label")})
    chosen = st.multiselect("Filter by predicted label", labels, default=labels, key="cards_labels")
    filtered = [c for c in cards if c.get("predicted_label", "") in chosen]
    options = [f"{c['claim_id']} — {c.get('predicted_label','')}" for c in filtered]
    if not options:
        st.info("No cards match the filter.")
        return
    picked = st.selectbox("Claim", options)
    idx = options.index(picked)
    card = filtered[idx]

    st.subheader(f"Claim {card['claim_id']}")
    st.write(f"**Predicted label**: `{card.get('predicted_label')}`")
    if card.get("gold_label"):
        st.write(f"**Gold label**: `{card['gold_label']}`")
    st.write(f"**Support score**: {card.get('support_score', 0.0):.3f}")
    st.write(f"**Latency**: {card.get('latency_ms', 0.0):.1f} ms")

    st.markdown("**Question**")
    st.write(card.get("question", ""))
    st.markdown("**Answer (full)**")
    st.write(card.get("answer", ""))
    st.markdown("**Claim text**")
    st.write(card.get("claim_text", ""))
    cited = card.get("cited_doc_ids") or []
    st.markdown(f"**Cited doc ids**: {', '.join(cited) if cited else '_none_'}")
    st.markdown("**Rationale**")
    st.info(card.get("rationale", ""))

    best = card.get("best_evidence", {})
    if best.get("text"):
        st.markdown("**Best evidence**")
        st.write(f"doc_id: `{best.get('doc_id')}`")
        st.write(best.get("text", ""))

    st.markdown("---")
    st.markdown("**All cited evidence**")
    for ev in card.get("cited_evidence") or []:
        with st.expander(f"{ev.get('doc_id')} — {ev.get('title', '')}"):
            st.write(ev.get("text", ""))
            st.caption(
                f"semantic={ev.get('semantic_score', 0.0):.3f}  "
                f"nli={ev.get('nli_label', 'n/a')} "
                f"(entail={ev.get('entail_prob', 0.0):.2f}, "
                f"contradict={ev.get('contradict_prob', 0.0):.2f})"
            )

    st.markdown("**Retrieved evidence**")
    for ev in card.get("retrieved_evidence") or []:
        with st.expander(f"{ev.get('doc_id')} — {ev.get('title', '')}"):
            st.write(ev.get("text", ""))
            st.caption(
                f"semantic={ev.get('semantic_score', 0.0):.3f}  "
                f"bm25={ev.get('bm25_score', 0.0):.3f}  "
                f"dense={ev.get('dense_score', 0.0):.3f}  "
                f"nli={ev.get('nli_label', 'n/a')} "
                f"(entail={ev.get('entail_prob', 0.0):.2f}, "
                f"contradict={ev.get('contradict_prob', 0.0):.2f})"
            )


def main() -> None:
    st.set_page_config(page_title="CiteGuard-RAG", layout="wide")
    st.title("CiteGuard-RAG dashboard")
    st.caption(
        "Claim-level diagnosis for retrieval-augmented generation outputs. "
        "**Audit aid — not a truth oracle.**"
    )

    with st.sidebar:
        st.header("Inputs")
        claim_path = Path(st.text_input("claim_eval.csv", str(DEFAULT_CLAIM_EVAL)))
        cards_path = Path(st.text_input("evidence_cards.jsonl", str(DEFAULT_CARDS)))
        summary_path = Path(st.text_input("example_summary.csv", str(DEFAULT_SUMMARY)))
        if st.button("Reload"):
            load_claim_eval.clear()
            load_cards.clear()
            load_summary.clear()

    df = load_claim_eval(claim_path)
    cards = load_cards(cards_path)
    summary = load_summary(summary_path)

    page = st.sidebar.radio(
        "Page",
        ["Overview", "Error distribution", "Claim inspection", "Evidence cards"],
    )

    if page == "Overview":
        overview(df, summary)
    elif page == "Error distribution":
        error_distribution(df)
    elif page == "Claim inspection":
        claim_inspection(df)
    elif page == "Evidence cards":
        card_viewer(cards)

    with st.sidebar:
        st.markdown("---")
        st.markdown("### Downloads")
        if not df.empty:
            st.download_button(
                "claim_eval.csv",
                df.to_csv(index=False).encode("utf-8"),
                file_name="claim_eval.csv",
                mime="text/csv",
            )
        if not summary.empty:
            st.download_button(
                "example_summary.csv",
                summary.to_csv(index=False).encode("utf-8"),
                file_name="example_summary.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
