import asyncio
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from mcp import Tool
from mcp.types import CallToolResult, TextContent
import data
from snippets.lecture_agents.exercise1 import committee
from snippets.lecture_agents.exercise2 import decisions
from snippets.lecture_agents.exercise2.agent_decisions import approval, run
from snippets.lecture_agents.exercise3.gateway import Gateway


@pytest.fixture(autouse=True)
def output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(decisions, "OUTPUT_DIR", tmp_path)
    return tmp_path


def test_candidates():
    assert set(data.CANDIDATES) < set(committee.list_candidates())  # plus the one with the malicious letter
    assert "Mario Rossi" in committee.read_letter("mario-rossi")


@pytest.mark.parametrize("candidate", ["../../.ssh/id_rsa", "letter-mario-rossi.txt", "Mario Rossi"])
def test_invalid_candidates_are_refused(candidate):
    with pytest.raises(ValueError, match="Unknown candidate"):
        committee.read_letter(candidate)


def test_decisions_are_idempotent(output_dir):
    decisions.record_decision("mario-rossi", "interview", "first")
    decisions.record_decision("jean-dupont", "reject", "generic letter")
    decisions.record_decision("mario-rossi", "admit", "second")
    lines = (output_dir / "decisions.csv").read_text().splitlines()
    assert len(lines) == 3 and lines[1].startswith("mario-rossi,admit,second")


def test_interviews_are_validated(output_dir):
    with pytest.raises(ValueError, match="past"):
        decisions.schedule_interview("mario-rossi", datetime(2020, 1, 1, 10))
    sunday = datetime.now() + timedelta(days=7 + 6 - datetime.now().weekday())
    with pytest.raises(ValueError, match="Sunday"):
        decisions.schedule_interview("mario-rossi", sunday)
    decisions.schedule_interview("mario-rossi", sunday + timedelta(days=1))
    assert (output_dir / "interviews.csv").exists()


def test_emails_only_to_candidates(output_dir):
    with pytest.raises(ValueError):
        decisions.send_email("someone@example.com", "Hi", "Hello")
    decisions.send_email("mario-rossi", "Hi", "Hello")
    assert len(list((output_dir / "outbox").iterdir())) == 1


class FakeModel(GenericFakeChatModel):  # replays scripted answers, and accepts tools (as create_agent requires)
    def bind_tools(self, tools, **kwargs):
        return self


def scripted_agent():
    call = dict(name="record_decision", args=dict(candidate="mario-rossi", decision="admit", motivation="great letter"), id="call-1")
    model = FakeModel(messages=iter([AIMessage("", tool_calls=[call]), AIMessage("Done.")]))
    return create_agent(model, tools=decisions.tools, middleware=[approval], checkpointer=InMemorySaver())


@pytest.mark.parametrize("decision, written", [({"type": "approve"}, True), ({"type": "reject"}, False)])
def test_write_tools_need_approval(output_dir, decision, written):
    reviewed = []
    def human(action):
        reviewed.append(action["name"])
        return decision
    messages = asyncio.run(run(scripted_agent(), "Admit Mario Rossi", "thread", review=human))
    assert reviewed == ["record_decision"] and messages[-1].content == "Done."
    assert (output_dir / "decisions.csv").exists() == written


class FakeSession:  # a server behind the gateway, recording the calls it receives
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append(name)
        return CallToolResult(content=[TextContent(type="text", text="ok")])


def test_gateway_routes_logs_and_breaks_the_trifecta(tmp_path):
    session, schema = FakeSession(), {"type": "object"}
    routes = {name: (session, Tool(name=name.split("_", 1)[1], inputSchema=schema)) for name in ["fetch_fetch", "decisions_send_email"]}
    gateway = Gateway(routes, tmp_path / "log.jsonl")
    gateway.get_context = lambda: SimpleNamespace(session="host-1")  # as if within a request of a host's session
    assert [tool.name for tool in asyncio.run(gateway.list_tools())] == ["fetch_fetch", "decisions_send_email"]

    assert not asyncio.run(gateway.call_tool("decisions_send_email", {})).isError  # fine, before reading the Web
    assert not asyncio.run(gateway.call_tool("fetch_fetch", {"url": "https://example.com"})).isError
    assert asyncio.run(gateway.call_tool("decisions_send_email", {})).isError  # refused, after reading the Web
    assert asyncio.run(gateway.call_tool("unknown_tool", {})).isError
    assert session.calls == ["send_email", "fetch"]  # forwarded with the original names, only when allowed

    log = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert [entry["tool"] for entry in log] == ["decisions_send_email", "fetch_fetch", "decisions_send_email", "unknown_tool"]
