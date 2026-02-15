from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv

from .config import get_settings
from .indexer import build_index
from .llm import LLMClient
from .answer import answer_question
from .analytics import extract_revenue_series, plot_series

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def index(
    docs: str = typer.Option(..., help="Path to documents folder"),
    out: str = typer.Option(..., help="Output folder for the index"),
    embed_model: str = typer.Option(None, help="Embedding model (SentenceTransformers name)"),
    chunk_size: int = typer.Option(None, help="Chunk size in characters"),
    chunk_overlap: int = typer.Option(None, help="Chunk overlap in characters"),
):
    """Build an index from documents."""
    load_dotenv()
    s = get_settings()
    out_dir = build_index(
        docs_dir=docs,
        out_dir=out,
        embed_model_name=embed_model or s.embed_model,
        chunk_size=chunk_size or s.chunk_size,
        chunk_overlap=chunk_overlap or s.chunk_overlap,
    )
    console.print(Panel.fit(f"Index created at: [bold]{out_dir}[/bold]"))


@app.command()
def ask(
    index: str = typer.Option(..., help="Path to the built index"),
    question: str = typer.Option(..., help="Leadership question"),
    top_k: int = typer.Option(None, help="Number of chunks to retrieve"),
    model: str = typer.Option(None, help="LLM model name"),
    plot: bool = typer.Option(False, help="Attempt to create a plot for time-series questions"),
):
    """Ask a question and get a grounded answer."""
    load_dotenv()
    s = get_settings()

    llm = LLMClient.from_env(model=model or s.openai_model, api_key=s.openai_api_key, base_url=s.openai_base_url)
    result = answer_question(index_dir=index, question=question, llm=llm, top_k=top_k or s.top_k)

    console.print(Panel.fit(f"[bold]Question[/bold]\n{result.question}"))
    console.print(result.answer)
    console.print(f"\n[bold]Confidence (heuristic):[/bold] {result.confidence:.2f}")

    # Show retrieved chunks table (short)
    table = Table(title="Retrieved Evidence")
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Chunk ID")
    table.add_column("Doc")
    for i, ch in enumerate(result.retrieved, start=1):
        table.add_row(str(i), f"{ch.score:.3f}", ch.chunk_id, ch.doc_id)
    console.print(table)

    if plot:
        points = extract_revenue_series(result.retrieved)
        out_path = str(Path(index) / "plot_revenue.png")
        saved = plot_series(points, title="Extracted Revenue Trend (heuristic)", out_path=out_path)
        if saved:
            console.print(Panel.fit(f"Plot saved to: [bold]{saved}[/bold]"))
        else:
            console.print(Panel.fit("Not enough numeric time-series points found to plot."))


if __name__ == "__main__":
    app()
