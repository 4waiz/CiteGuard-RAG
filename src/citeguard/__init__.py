"""CiteGuard-RAG: claim-level diagnosis toolkit for retrieval-augmented generation."""

import os as _os

# Keep transformers torch-only — some local envs ship a TensorFlow build that
# is incompatible with NumPy 2.x and crashes on import.
_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("USE_FLAX", "0")
_os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

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
