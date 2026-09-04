#!/usr/bin/env python3
"""Full modular RAG demo: route a question to the right backend, retrieve
from it, and answer — documents (PDF), SQL (Chinook), graph (extracted
from the PDF), web (live search), or API (arXiv).

Usage:
    cd 03-modular-rag
    uv run python examples/modular_rag.py "How many tracks are in the database?"
    uv run python examples/modular_rag.py "What does the Transformer architecture look like?"
    uv run python examples/modular_rag.py "Who is affiliated with Google Brain?"
    uv run python examples/modular_rag.py "What is the latest news on large language models?"
    uv run python examples/modular_rag.py "When was the Attention Is All You Need paper published?"
"""

from __future__ import annotations

import sys
from pathlib import Path

LEVEL_DIR = Path(__file__).resolve().parent.parent
for sub in ["", "routing", "multi-retriever", "sql-rag", "graph-rag", "web-rag", "api-rag"]:
    sys.path.insert(0, str(LEVEL_DIR / sub) if sub else str(LEVEL_DIR))

from modular_common.llm import OllamaLLM  # noqa: E402
from modular_common.pdf import load_chunks  # noqa: E402
from rule_router import rule_route  # noqa: E402
from vector_retriever import VectorRetriever  # noqa: E402
from text_to_sql import answer_from_sql  # noqa: E402
from entity_extraction import extract_triples  # noqa: E402
from graph_builder import build_graph  # noqa: E402
from graph_retrieval import graph_search  # noqa: E402
from page_extraction import search_and_extract  # noqa: E402
from tool_api import search_arxiv  # noqa: E402


def answer_documents(question: str, llm: OllamaLLM) -> str:
    chunks = load_chunks()
    retriever = VectorRetriever.from_texts(
        {cid: c["text"] for cid, c in chunks.items()}, cache_name="pdf_chunks"
    )
    hits = retriever.search(question, top_k=3)
    context = "\n\n".join(chunks[cid]["text"] for cid, _ in hits)
    prompt = f"Context from the paper:\n{context}\n\nQuestion: {question}\nAnswer using only the context:"
    return llm.complete(prompt)


def answer_sql(question: str, llm: OllamaLLM) -> str:
    result = answer_from_sql(question, llm=llm)
    return f"SQL: {result['sql']}\nResult: {result['rows'][:10]}"


def answer_graph(question: str, llm: OllamaLLM, graph_triples: list[dict]) -> str:
    graph = build_graph(graph_triples)
    facts = graph_search(graph, question)
    if not facts:
        return "No relevant facts found in the knowledge graph."
    context = "\n".join(facts)
    prompt = f"Known facts:\n{context}\n\nQuestion: {question}\nAnswer using only these facts:"
    return llm.complete(prompt)


def answer_web(question: str, llm: OllamaLLM) -> str:
    results = search_and_extract(question, max_results=3)
    context = "\n\n".join(f"[{r['title']}] {r['text'][:500]}" for r in results if r.get("text"))
    prompt = f"Web search results:\n{context}\n\nQuestion: {question}\nAnswer:"
    return llm.complete(prompt)


def answer_api(question: str, llm: OllamaLLM) -> str:
    papers = search_arxiv("id:1706.03762", max_results=1)
    if not papers:
        return "Could not reach the arXiv API."
    paper = papers[0]
    context = (
        f"Title: {paper['title']}\nAuthors: {', '.join(paper['authors'])}\n"
        f"Published: {paper['published']}\nSummary: {paper['summary']}"
    )
    prompt = f"Paper metadata:\n{context}\n\nQuestion: {question}\nAnswer:"
    return llm.complete(prompt)


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "What is the Transformer architecture?"
    llm = OllamaLLM()

    route = rule_route(question)
    print(f"Q: {question}")
    print(f"Routed to: {route}\n")

    if route == "documents":
        answer = answer_documents(question, llm)
    elif route == "sql":
        answer = answer_sql(question, llm)
    elif route == "graph":
        print("Extracting entities from the paper's intro to build the graph (first run only)...")
        chunks = load_chunks()
        first_chunks = dict(list(chunks.items())[:4])
        triples = [t for c in first_chunks.values() for t in extract_triples(c["text"])]
        answer = answer_graph(question, llm, triples)
    elif route == "web":
        answer = answer_web(question, llm)
    else:  # api
        answer = answer_api(question, llm)

    print(f"A: {answer}")


if __name__ == "__main__":
    main()
