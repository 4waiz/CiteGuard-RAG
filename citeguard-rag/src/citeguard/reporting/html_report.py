"""Minimal HTML summary report (no external deps)."""
from __future__ import annotations

import html
from pathlib import Path


def _table(rows: list[list[str]], header: list[str]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def write_html_report(
    out_path: str | Path,
    run_name: str,
    overall_metrics: dict,
    per_label_rows: list[list[str]],
    figures: list[Path],
) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig_html = ""
    for fig in figures:
        try:
            rel = Path(fig).resolve().relative_to(out.resolve().parent)
            src = rel.as_posix()
        except Exception:
            src = Path(fig).as_posix()
        fig_html += f'<figure><img src="{html.escape(src)}" alt="{html.escape(fig.name)}"><figcaption>{html.escape(fig.name)}</figcaption></figure>'

    overall_rows = [[k, f"{v:.4f}" if isinstance(v, float) else str(v)] for k, v in overall_metrics.items()]
    overall_table = _table(overall_rows, ["Metric", "Value"])
    per_label_table = _table(per_label_rows, ["Label", "Precision", "Recall", "F1", "Support"])

    body = f"""
    <!doctype html>
    <html><head>
    <meta charset="utf-8">
    <title>CiteGuard-RAG report — {html.escape(run_name)}</title>
    <style>
      body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2em; max-width: 1000px; }}
      table {{ border-collapse: collapse; margin: 1em 0; }}
      td, th {{ border: 1px solid #ccc; padding: 0.4em 0.8em; }}
      th {{ background: #f3f3f3; text-align: left; }}
      figure {{ margin: 1em 0; }}
      img {{ max-width: 100%; height: auto; }}
      .note {{ background: #fff8dc; padding: 1em; border-left: 4px solid #d4af37; }}
    </style>
    </head><body>
    <h1>CiteGuard-RAG report</h1>
    <p>Run: <strong>{html.escape(run_name)}</strong></p>
    <p class="note">CiteGuard-RAG is an audit aid, not a truth oracle. Outputs reflect what the configured models believe, not ground truth.</p>
    <h2>Overall metrics</h2>
    {overall_table}
    <h2>Per-label metrics</h2>
    {per_label_table}
    <h2>Figures</h2>
    {fig_html}
    </body></html>
    """
    out.write_text(body, encoding="utf-8")
    return out
