#!/usr/bin/env python3
"""Full autonomous research agent: plans, chooses among 5 real tools
(vector search, document fetch, SQL, live web, knowledge graph),
iterates until it has enough evidence, then answers with source
verification — on real, open data (TriviaQA + Northwind).

Usage:
    cd 05-agentic-rag
    uv run python examples/research_agent.py "A sophomore is a student in which year of a US college?"
    uv run python examples/research_agent.py "How many products are in the Northwind database?"
"""

from __future__ import annotations

import sys
from pathlib import Path

LEVEL_DIR = Path(__file__).resolve().parent.parent
for sub in ["", "tools"]:
    sys.path.insert(0, str(LEVEL_DIR / sub) if sub else str(LEVEL_DIR))

from agentic_common.dataset import prepare  # noqa: E402
from agentic_common.llm import OllamaLLM  # noqa: E402
from agentic_common.retrieval import DenseRetriever  # noqa: E402
from agents.rag_agent import RAGAgent  # noqa: E402
from graph_tool import GraphTool, build_graph, extract_triples  # noqa: E402
from sql_tool import SqlTool  # noqa: E402
from vector_tool import GetDocumentTool, VectorTool  # noqa: E402
from web_tool import WebTool  # noqa: E402


def main() -> None:
    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "A sophomore is a student in which year of a US college?"
    )
    llm = OllamaLLM()

    print("Loading TriviaQA subset (cached after first run)...")
    data = prepare()
    print(f"Corpus: {len(data.corpus)} chunks · {len(data.questions)} labeled questions available.\n")

    corpus_texts = {cid: c["text"] for cid, c in data.corpus.items()}
    retriever = DenseRetriever.from_corpus(corpus_texts)

    # Build a small knowledge graph from a handful of corpus chunks (first
    # run only pays the extraction cost; kept tiny on purpose for this demo).
    sample_chunks = list(data.corpus.values())[:8]
    triples = [t for chunk in sample_chunks for t in extract_triples(chunk["text"], llm=llm)]
    graph = build_graph(triples)

    tools = {
        "vector_search": VectorTool(retriever, data.corpus),
        "get_document": GetDocumentTool(data),
        "sql_query": SqlTool(llm=llm),
        "web_search": WebTool(),
        "graph_search": GraphTool(graph),
    }

    agent = RAGAgent(tools=tools, llm=llm)
    state = agent.run(question)

    print(f"Q: {question}")
    print(f"Plan: {state.plan}\n")
    print("Steps taken:")
    for call in state.tool_history:
        print(f"  [{call.step}] {call.tool}({call.tool_input!r}) -> {str(call.result)[:150]}")
    print(f"\nStop reason: {state.stop_reason}")
    print(f"Sources verified: {state.verified}")
    print(f"\nA: {state.answer}")


if __name__ == "__main__":
    main()
