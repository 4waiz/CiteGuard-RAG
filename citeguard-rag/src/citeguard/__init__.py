"""CiteGuard-RAG: claim-level diagnosis toolkit for retrieval-augmented generation."""

__version__ = "0.1.0"

from .schemas import (
    LABELS,
    PARSER_UNCERTAIN,
    Claim,
    ClaimEvaluation,
    Context,
    RAGExample,
)

__all__ = [
    "__version__",
    "LABELS",
    "PARSER_UNCERTAIN",
    "Claim",
    "ClaimEvaluation",
    "Context",
    "RAGExample",
]
