#!/usr/bin/env python3
"""Full multi-agent business research system: a supervisor routes a task
to specialized agents (retrieval, SQL, web, graph, research), runs them
in parallel, verifies each finding, and synthesizes one combined answer
— on real, open data (financial-qa-10K + Sakila).

Usage:
    cd 06-multi-agent-rag
    uv run python examples/company_research_system.py "How much were the company's debt obligations as of December 31, 2023?"
    uv run python examples/company_research_system.py "How many films are in the Sakila database?"
"""

from __future__ import annotations

import sys
from pathlib import Path

LEVEL_DIR = Path(__file__).resolve().parent.parent
for sub in ["", "supervisor", "workflows"]:
    sys.path.insert(0, str(LEVEL_DIR / sub) if sub else str(LEVEL_DIR))

from multiagent_common.dataset import prepare  # noqa: E402
from multiagent_common.llm import OllamaLLM  # noqa: E402
from multiagent_common.loader import load_agent_class  # noqa: E402
from multiagent_common.retrieval import DenseRetriever  # noqa: E402
from supervisor import Supervisor  # noqa: E402
from supervisor_graph import run_supervisor_graph  # noqa: E402

RetrievalAgent = load_agent_class("retrieval-agent", "RetrievalAgent")
SqlAgent = load_agent_class("sql-agent", "SqlAgent")
WebAgent = load_agent_class("web-agent", "WebAgent")
GraphAgent = load_agent_class("graph-agent", "GraphAgent")


def main() -> None:
    task = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "How much were the company's debt obligations as of December 31, 2023?"
    )
    llm = OllamaLLM()

    print("Loading financial-qa-10K subset (cached after first run)...")
    data = prepare()
    print(f"Corpus: {len(data.corpus)} filing excerpts across {data.metadata['n_tickers']} companies.\n")

    corpus_texts = {cid: c["text"] for cid, c in data.corpus.items()}
    retriever = DenseRetriever.from_corpus(corpus_texts)
    retrieval_agent = RetrievalAgent(retriever, data.corpus, llm=llm)
    sql_agent = SqlAgent(llm=llm)
    web_agent = WebAgent(llm=llm)

    # Build a small company knowledge graph from a handful of corpus docs
    # (first run only pays the extraction cost).
    import importlib.util

    spec = importlib.util.spec_from_file_location("_graph_agent_impl", LEVEL_DIR / "graph-agent" / "agent.py")
    graph_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graph_module)

    sample_docs = list(data.corpus.values())[:8]
    triples = [t for d in sample_docs for t in graph_module.extract_triples(d["text"], llm=llm)]
    graph = graph_module.build_graph(triples)
    graph_agent = graph_module.GraphAgent(graph, llm=llm)

    research_agent = load_agent_class("research-agent", "ResearchAgent")(retrieval_agent, graph_agent, llm=llm)
    verification_agent = load_agent_class("verification-agent", "VerificationAgent")(llm=llm)
    synthesis_agent = load_agent_class("synthesis-agent", "SynthesisAgent")(llm=llm)

    agents = {
        "retrieval-agent": retrieval_agent,
        "sql-agent": sql_agent,
        "web-agent": web_agent,
        "graph-agent": graph_agent,
        "research-agent": research_agent,
    }
    supervisor = Supervisor(agents, llm=llm)

    result = run_supervisor_graph(task, supervisor, verification_agent, synthesis_agent, use_parallel=True)

    print(f"Task: {task}")
    print(f"Routed to: {result['routed_to']}")
    print(f"Elapsed: {result['elapsed_seconds']:.2f}s\n")
    for name, agent_result in result["results"].items():
        print(f"[{name}] verified={result['verified'][name]}")
        print(f"  {agent_result.output[:200]}")
    print(f"\nFinal synthesis:\n{result['synthesis'].output}")


if __name__ == "__main__":
    main()
