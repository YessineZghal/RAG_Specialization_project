from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.rag_agent import RAGAgent


def test_parse_decision_extracts_action_and_input():
    raw = "ACTION: vector_search\nINPUT: sophomore year college"
    action, tool_input = RAGAgent._parse_decision(raw)
    assert action == "vector_search"
    assert tool_input == "sophomore year college"


def test_parse_decision_is_case_insensitive_on_action_label():
    raw = "action: FINISH\ninput: The answer is 42."
    action, tool_input = RAGAgent._parse_decision(raw)
    assert action == "finish"
    assert tool_input == "The answer is 42."


def test_parse_decision_falls_back_to_finish_on_unparseable_output():
    action, _ = RAGAgent._parse_decision("I'm not sure what to do.")
    assert action == "finish"


def test_agent_finishes_immediately_when_decided(fake_llm):
    llm = fake_llm(responses=[
        "Search the corpus",  # plan
        "ACTION: finish\nINPUT: The answer is 42.",  # decide
    ])
    agent = RAGAgent(tools={}, llm=llm, max_steps=3)
    state = agent.run("some question")

    assert state.stop_reason == "answered"
    assert state.answer == "The answer is 42."
    assert state.tool_history == []
    assert state.verified is False  # no tools were called -> never checked


def test_agent_calls_tool_then_answers_when_evidence_sufficient(fake_llm, fake_vector_tool):
    tool = fake_vector_tool()
    llm = fake_llm(responses=[
        "Search the corpus",  # plan
        "ACTION: vector_search\nINPUT: some query",  # decide
        "yes",  # is_evidence_sufficient
        "Sophomores are second-year students.",  # _generate_answer
        "supported",  # check_sources
    ])
    agent = RAGAgent(tools={"vector_search": tool}, llm=llm, max_steps=3)
    state = agent.run("some question")

    assert state.stop_reason == "sufficient_evidence"
    assert len(state.tool_history) == 1
    assert tool.calls == ["some query"]
    assert state.answer == "Sophomores are second-year students."
    assert state.verified is True


def test_agent_handles_hallucinated_tool_name_without_crashing(fake_llm):
    llm = fake_llm(responses=[
        "Search the corpus",  # plan
        "ACTION: nonexistent_tool\nINPUT: q1",  # decide -> invalid tool
        "ACTION: finish\nINPUT: Final answer.",  # decide again
        "supported",  # check_sources (tool_history has the invalid-action entry)
    ])
    agent = RAGAgent(tools={}, llm=llm, max_steps=3)
    state = agent.run("some question")

    assert state.tool_history[0].tool == "invalid_action"
    assert state.answer == "Final answer."


def test_agent_stops_at_max_steps_and_falls_back(fake_llm, fake_vector_tool):
    tool = fake_vector_tool()
    llm = fake_llm(responses=[
        "Search the corpus",  # plan
        "ACTION: vector_search\nINPUT: q1",  # decide, step 1
        "no",  # insufficient
        "ACTION: vector_search\nINPUT: q2",  # decide, step 2
        "no",  # insufficient (loop exhausted, max_steps=2)
        "Fallback answer from evidence.",  # _generate_answer
        "supported",  # check_sources
    ])
    agent = RAGAgent(tools={"vector_search": tool}, llm=llm, max_steps=2)
    state = agent.run("some question")

    assert state.stop_reason == "max_steps"
    assert len(state.tool_history) == 2
    assert state.answer == "Fallback answer from evidence."


def test_agent_tool_exception_is_recorded_not_raised(fake_llm):
    def broken_tool(query):
        raise RuntimeError("simulated failure")

    llm = fake_llm(responses=[
        "Search the corpus",  # plan
        "ACTION: broken\nINPUT: q1",  # decide
        "yes",  # is_evidence_sufficient -- even a failed call counts as "evidence" text here
        "Best-effort answer.",  # _generate_answer
        "supported",  # check_sources
    ])
    agent = RAGAgent(tools={"broken": broken_tool}, llm=llm, max_steps=2)
    state = agent.run("some question")

    assert "Tool error" in str(state.tool_history[0].result)
    assert state.answer == "Best-effort answer."
