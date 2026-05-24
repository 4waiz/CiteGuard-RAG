"""Command-line interface."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Force transformers to be torch-only. Some Windows environments ship a
# TensorFlow build pinned to NumPy 1.x; importing it under NumPy 2.x crashes
# the whole process. We never use TF or Flax, so disable them up front.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import click

from .config import load_config
from .data_loaders.custom_jsonl import load_jsonl
from .pipeline import CiteGuardPipeline


@click.group()
def main() -> None:
    """CiteGuard-RAG command-line entry point."""


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--config", "config_path", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "output_dir", default="outputs", type=click.Path(file_okay=False))
@click.option("--no-progress", is_flag=True, default=False, help="Disable progress bar.")
def evaluate(input_path: str, config_path: str | None, output_dir: str, no_progress: bool) -> None:
    """Run the full pipeline on a JSONL of RAG examples."""
    cfg = load_config(config_path)
    # Re-root output dirs under the requested --output if provided.
    out_root = Path(output_dir)
    cfg.output.tables_dir = str(out_root / "tables")
    cfg.output.figures_dir = str(out_root / "figures")
    cfg.output.cards_dir = str(out_root / "evidence_cards")
    cfg.output.reports_dir = str(out_root / "reports")

    examples = load_jsonl(input_path)
    if not examples:
        click.echo(f"No examples loaded from {input_path}.", err=True)
        sys.exit(1)

    pipeline = CiteGuardPipeline(cfg)
    results = pipeline.run(examples, show_progress=not no_progress)
    paths = pipeline.write_outputs(results)

    click.echo(f"Wrote {len(results['evaluations'])} claim evaluations.")
    for k, v in paths.items():
        click.echo(f"  {k}: {v}")
    agg = results["aggregate"]
    click.echo(
        f"macro-F1={agg['macro_f1']:.4f} weighted-F1={agg['weighted_f1']:.4f} "
        f"(n_with_gold={agg['n_with_gold_label']}/{agg['n_claims']})"
    )


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False))
def split(input_path: str) -> None:
    """Print claim splits for each example. Useful for debugging the splitter."""
    from .preprocess.claim_splitter import ClaimSplitter

    splitter = ClaimSplitter()
    for ex in load_jsonl(input_path):
        claims = splitter.split(ex.example_id, ex.answer)
        click.echo(f"=== {ex.example_id} ({len(claims)} claims) ===")
        for c in claims:
            click.echo(f"  [{c.claim_id}] cited={c.cited_doc_ids} :: {c.text}")


if __name__ == "__main__":  # pragma: no cover
    main()
