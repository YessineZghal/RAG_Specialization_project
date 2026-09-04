"""Command-line entry point for the Level 1 pipeline.

Run from inside `01-naive-rag/` (so `python -m src.cli` resolves the local
`src` package):

    uv run python -m src.cli ingest              # build + persist the index
    uv run python -m src.cli ask "your question"  # ask against it
    uv run python -m src.cli evaluate              # Recall@K on shared/evaluation

Or via the root Makefile: `make ask Q="..."`, `make eval`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from rich.console import Console
from rich.table import Table

from .config import settings
from .ingest import load_from_directory, load_from_hf_dataset
from .pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")
console = Console()


def cmd_ingest(args: argparse.Namespace) -> None:
    pipeline = RAGPipeline()
    if args.source == "sample-docs":
        documents = load_from_directory(settings.sample_docs_dir)
    else:
        documents = load_from_hf_dataset(limit=args.limit)

    if not documents:
        console.print(
            f"[red]No documents loaded from '{args.source}'.[/] "
            "Nothing to index."
        )
        sys.exit(1)

    n_chunks = pipeline.build_index(documents)
    pipeline.save_index()
    console.print(
        f"[green]Indexed[/] {len(documents)} documents -> {n_chunks} chunks. "
        f"Saved to {settings.index_dir}"
    )


def cmd_ask(args: argparse.Namespace) -> None:
    pipeline = _load_or_build_pipeline(args.source, args.limit)
    answer = pipeline.ask(args.question, top_k=args.top_k)

    console.print(f"\n[bold]Q:[/] {answer.question}")
    console.print(f"[bold]A:[/] {answer.answer}\n")

    table = Table(title="Retrieved sources")
    table.add_column("#")
    table.add_column("Score")
    table.add_column("Document")
    table.add_column("Chunk preview")
    for i, retrieved in enumerate(answer.sources, start=1):
        preview = retrieved.chunk.text[:100].replace("\n", " ") + "..."
        table.add_row(str(i), f"{retrieved.score:.3f}", retrieved.chunk.document_id, preview)
    console.print(table)


def cmd_evaluate(args: argparse.Namespace) -> None:
    questions_path = settings.shared_eval_dir / "questions.jsonl"
    sources_path = settings.shared_eval_dir / "expected_sources.jsonl"
    if not questions_path.exists() or not sources_path.exists():
        console.print(
            "[red]No evaluation set found.[/] Build one first with:\n"
            "  uv run python -m src.build_eval_set"
        )
        sys.exit(1)

    questions = _read_jsonl(questions_path)
    expected_sources = {row["id"]: set(row["document_ids"]) for row in _read_jsonl(sources_path)}

    pipeline = _load_or_build_pipeline(args.source, args.limit)

    hits, evaluable = 0, 0
    for row in questions:
        expected = expected_sources.get(row["id"], set())
        if not expected:
            continue  # heuristic in build_eval_set.py found no confident match — skip
        evaluable += 1
        answer = pipeline.ask(row["question"], top_k=args.top_k)
        retrieved_ids = {r.chunk.document_id for r in answer.sources}
        if retrieved_ids & expected:
            hits += 1

    if evaluable == 0:
        console.print("[yellow]No evaluable questions (all had empty expected_sources).[/]")
        return

    recall_at_k = hits / evaluable
    console.print(
        f"\n[bold]Recall@{args.top_k}[/]: {recall_at_k:.2%} "
        f"({hits}/{evaluable} questions had a relevant source in the Top-{args.top_k})"
    )


def _load_or_build_pipeline(source: str, limit: int | None) -> RAGPipeline:
    try:
        return RAGPipeline.load_index()
    except FileNotFoundError:
        console.print(
            f"[yellow]No saved index found at {settings.index_dir} — building one from "
            f"'{source}' first (use `ingest` to persist it for next time).[/]"
        )
        pipeline = RAGPipeline()
        documents = (
            load_from_directory(settings.sample_docs_dir)
            if source == "sample-docs"
            else load_from_hf_dataset(limit=limit)
        )
        pipeline.build_index(documents)
        return pipeline


def _read_jsonl(path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--source",
        choices=["hf-dataset", "sample-docs"],
        default="hf-dataset",
        help="Where to load documents from if no saved index exists.",
    )
    common.add_argument("--limit", type=int, default=500, help="Max documents from the HF dataset.")
    common.add_argument("--top-k", type=int, default=settings.top_k)

    ingest_parser = subparsers.add_parser("ingest", parents=[common], help="Build and save the index.")
    ingest_parser.set_defaults(func=cmd_ingest)

    ask_parser = subparsers.add_parser("ask", parents=[common], help="Ask a question.")
    ask_parser.add_argument("question")
    ask_parser.set_defaults(func=cmd_ask)

    eval_parser = subparsers.add_parser(
        "evaluate", parents=[common], help="Report Recall@K on shared/evaluation."
    )
    eval_parser.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
