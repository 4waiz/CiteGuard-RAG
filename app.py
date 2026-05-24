"""Hugging Face Spaces entrypoint.

The Spaces Streamlit SDK runs ``streamlit run app.py`` from the repo root.
We delegate to the real dashboard module in ``app/streamlit_app.py`` so
there's a single source of truth.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Keep transformers torch-only — Spaces images sometimes carry a tensorflow
# build that pins NumPy 1.x and crashes on import.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

# Make the src/ layout importable without an editable install.
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

# Streamlit imports the script at module level. Importing the dashboard
# module is enough — it calls st.* at import time via the trailing main().
from app.streamlit_app import main  # noqa: E402

main()
