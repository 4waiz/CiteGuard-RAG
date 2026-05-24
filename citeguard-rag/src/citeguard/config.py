"""Configuration loading. Thin wrapper around YAML + env overrides."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelsConfig:
    sbert: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli: str = "cross-encoder/nli-deberta-v3-base"
    device: str = "auto"


@dataclass
class RetrievalConfig:
    top_k: int = 5
    use_bm25: bool = True
    use_dense: bool = True


@dataclass
class EvaluatorToggle:
    enabled: bool = True
    model: str | None = None
    max_evidence_chars: int = 1500


@dataclass
class EvaluatorsConfig:
    semantic: EvaluatorToggle = field(default_factory=lambda: EvaluatorToggle(enabled=True))
    nli: EvaluatorToggle = field(default_factory=lambda: EvaluatorToggle(enabled=True))
    bm25: EvaluatorToggle = field(default_factory=lambda: EvaluatorToggle(enabled=True))
    llm_judge: EvaluatorToggle = field(default_factory=lambda: EvaluatorToggle(enabled=False))


@dataclass
class ThresholdsConfig:
    entail_support: float = 0.55
    contradict: float = 0.55
    semantic_related: float = 0.55
    semantic_strong: float = 0.70
    unsupported_floor: float = 0.30


@dataclass
class ClaimSplitterConfig:
    min_chars: int = 5
    strip_citations_for_embedding: bool = True


@dataclass
class CitationParserConfig:
    patterns: list[str] = field(default_factory=lambda: [r"\[([^\]]+)\]"])
    numeric_prefix: str = "doc"


@dataclass
class OutputConfig:
    tables_dir: str = "outputs/tables"
    figures_dir: str = "outputs/figures"
    cards_dir: str = "outputs/evidence_cards"
    reports_dir: str = "outputs/reports"


@dataclass
class CiteGuardConfig:
    run_name: str = "default"
    models: ModelsConfig = field(default_factory=ModelsConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    evaluators: EvaluatorsConfig = field(default_factory=EvaluatorsConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    claim_splitter: ClaimSplitterConfig = field(default_factory=ClaimSplitterConfig)
    citation_parser: CitationParserConfig = field(default_factory=CitationParserConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _apply_env_overrides(cfg: CiteGuardConfig) -> CiteGuardConfig:
    if os.environ.get("CITEGUARD_SBERT_MODEL"):
        cfg.models.sbert = os.environ["CITEGUARD_SBERT_MODEL"]
    if os.environ.get("CITEGUARD_NLI_MODEL"):
        cfg.models.nli = os.environ["CITEGUARD_NLI_MODEL"]
    if os.environ.get("CITEGUARD_FORCE_CPU"):
        cfg.models.device = "cpu"
    if os.environ.get("CITEGUARD_LLM_JUDGE_ENABLED", "").lower() in {"1", "true", "yes"}:
        cfg.evaluators.llm_judge.enabled = True
    if os.environ.get("CITEGUARD_LLM_JUDGE_MODEL"):
        cfg.evaluators.llm_judge.model = os.environ["CITEGUARD_LLM_JUDGE_MODEL"]
    return cfg


def _toggle_from_dict(d: dict[str, Any] | None, default: EvaluatorToggle) -> EvaluatorToggle:
    if not d:
        return default
    return EvaluatorToggle(
        enabled=bool(d.get("enabled", default.enabled)),
        model=d.get("model", default.model),
        max_evidence_chars=int(d.get("max_evidence_chars", default.max_evidence_chars)),
    )


def load_config(path: str | Path | None) -> CiteGuardConfig:
    """Load a YAML config and apply environment overrides.

    Passing ``None`` returns the in-code defaults with env overrides applied.
    """
    cfg = CiteGuardConfig()
    if path is None:
        return _apply_env_overrides(cfg)

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg.run_name = str(raw.get("run_name", cfg.run_name))

    m = raw.get("models", {}) or {}
    cfg.models = ModelsConfig(
        sbert=str(m.get("sbert", cfg.models.sbert)),
        nli=str(m.get("nli", cfg.models.nli)),
        device=str(m.get("device", cfg.models.device)),
    )

    r = raw.get("retrieval", {}) or {}
    cfg.retrieval = RetrievalConfig(
        top_k=int(r.get("top_k", cfg.retrieval.top_k)),
        use_bm25=bool(r.get("use_bm25", cfg.retrieval.use_bm25)),
        use_dense=bool(r.get("use_dense", cfg.retrieval.use_dense)),
    )

    e = raw.get("evaluators", {}) or {}
    cfg.evaluators = EvaluatorsConfig(
        semantic=_toggle_from_dict(e.get("semantic"), cfg.evaluators.semantic),
        nli=_toggle_from_dict(e.get("nli"), cfg.evaluators.nli),
        bm25=_toggle_from_dict(e.get("bm25"), cfg.evaluators.bm25),
        llm_judge=_toggle_from_dict(e.get("llm_judge"), cfg.evaluators.llm_judge),
    )

    t = raw.get("thresholds", {}) or {}
    cfg.thresholds = ThresholdsConfig(
        entail_support=float(t.get("entail_support", cfg.thresholds.entail_support)),
        contradict=float(t.get("contradict", cfg.thresholds.contradict)),
        semantic_related=float(t.get("semantic_related", cfg.thresholds.semantic_related)),
        semantic_strong=float(t.get("semantic_strong", cfg.thresholds.semantic_strong)),
        unsupported_floor=float(t.get("unsupported_floor", cfg.thresholds.unsupported_floor)),
    )

    cs = raw.get("claim_splitter", {}) or {}
    cfg.claim_splitter = ClaimSplitterConfig(
        min_chars=int(cs.get("min_chars", cfg.claim_splitter.min_chars)),
        strip_citations_for_embedding=bool(
            cs.get("strip_citations_for_embedding", cfg.claim_splitter.strip_citations_for_embedding)
        ),
    )

    cp = raw.get("citation_parser", {}) or {}
    cfg.citation_parser = CitationParserConfig(
        patterns=list(cp.get("patterns", cfg.citation_parser.patterns)),
        numeric_prefix=str(cp.get("numeric_prefix", cfg.citation_parser.numeric_prefix)),
    )

    o = raw.get("output", {}) or {}
    cfg.output = OutputConfig(
        tables_dir=str(o.get("tables_dir", cfg.output.tables_dir)),
        figures_dir=str(o.get("figures_dir", cfg.output.figures_dir)),
        cards_dir=str(o.get("cards_dir", cfg.output.cards_dir)),
        reports_dir=str(o.get("reports_dir", cfg.output.reports_dir)),
    )

    return _apply_env_overrides(cfg)
