from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from indexing.community_summary import build_community_summaries, detect_communities  # noqa: E402


def _build_graph_with_two_real_clusters_and_one_isolated_node():
    graph = nx.MultiDiGraph()
    # Cluster A: a tightly connected trio
    graph.add_node("study-1", name="Study-1", type="Study")
    graph.add_node("diabetes", name="Diabetes", type="Condition")
    graph.add_node("metformin", name="Metformin", type="Intervention")
    graph.add_edge("study-1", "diabetes", relation="STUDIES", doc_id="d1")
    graph.add_edge("study-1", "metformin", relation="USES_INTERVENTION", doc_id="d1")
    # Cluster B: a separate, unrelated pair
    graph.add_node("study-2", name="Study-2", type="Study")
    graph.add_node("asthma", name="Asthma", type="Condition")
    graph.add_edge("study-2", "asthma", relation="STUDIES", doc_id="d2")
    # An isolated singleton -- nothing connects to it at all
    graph.add_node("outcome-lonely", name="Lonely Outcome", type="Outcome")
    return graph


class FakeLLM:
    def __init__(self, response: str = "A summary.") -> None:
        self.response = response
        self.calls: list[str] = []

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.2) -> str:
        self.calls.append(prompt)
        return self.response


def test_detect_communities_finds_the_real_clusters_and_drops_the_singleton():
    graph = _build_graph_with_two_real_clusters_and_one_isolated_node()

    communities = detect_communities(graph)

    all_nodes_in_communities = {n for c in communities for n in c}
    assert "outcome-lonely" not in all_nodes_in_communities  # singleton dropped
    assert all(len(c) >= 2 for c in communities)
    # the two real clusters must not be merged into one
    assert any({"study-1", "diabetes", "metformin"} <= c for c in communities)
    assert any({"study-2", "asthma"} <= c for c in communities)


def test_detect_communities_on_an_empty_graph_returns_no_communities():
    assert detect_communities(nx.MultiDiGraph()) == []


def test_build_community_summaries_makes_one_llm_call_per_community():
    graph = _build_graph_with_two_real_clusters_and_one_isolated_node()
    llm = FakeLLM(response="These entities concern a biomedical intervention study.")

    summaries = build_community_summaries(graph, llm)

    assert len(summaries) == 2
    assert len(llm.calls) == 2
    for community in summaries.values():
        assert community["summary"] == "These entities concern a biomedical intervention study."
        assert len(community["nodes"]) >= 2


def test_build_community_summaries_prompt_includes_the_communitys_own_facts():
    graph = _build_graph_with_two_real_clusters_and_one_isolated_node()
    llm = FakeLLM()

    build_community_summaries(graph, llm)

    combined_prompts = " ".join(llm.calls)
    assert "Study-1" in combined_prompts
    assert "STUDIES" in combined_prompts


def test_build_community_summaries_on_a_graph_with_no_real_communities_makes_no_llm_calls():
    graph = nx.MultiDiGraph()
    graph.add_node("only-node", name="Only Node", type="Condition")  # a lone singleton
    llm = FakeLLM()

    summaries = build_community_summaries(graph, llm)

    assert summaries == {}
    assert llm.calls == []
