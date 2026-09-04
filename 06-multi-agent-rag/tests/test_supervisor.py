from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "supervisor"))
from supervisor import Supervisor


def test_route_selects_named_agents(fake_llm):
    llm = fake_llm(response="retrieval-agent, graph-agent")
    supervisor = Supervisor(agents={}, llm=llm)
    assert supervisor.route("some task") == ["retrieval-agent", "graph-agent"]


def test_route_falls_back_to_research_agent_on_unparseable_output(fake_llm):
    llm = fake_llm(response="I'm not sure which agent to use.")
    supervisor = Supervisor(agents={}, llm=llm)
    assert supervisor.route("some task") == ["research-agent"]


def test_route_ignores_unknown_agent_names(fake_llm):
    llm = fake_llm(response="retrieval-agent, made-up-agent")
    supervisor = Supervisor(agents={}, llm=llm)
    assert supervisor.route("some task") == ["retrieval-agent"]


def test_delegate_only_calls_agents_that_exist(fake_llm, fake_agent):
    llm = fake_llm(response="retrieval-agent, sql-agent")
    retrieval = fake_agent("retrieval-agent")
    # sql-agent is routed to but not registered -- must be silently skipped
    supervisor = Supervisor(agents={"retrieval-agent": retrieval}, llm=llm)

    results = supervisor.delegate("some task")

    assert set(results.keys()) == {"retrieval-agent"}
    assert retrieval.calls == ["some task"]
