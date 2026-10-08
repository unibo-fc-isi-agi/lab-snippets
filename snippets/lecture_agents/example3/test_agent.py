"""
Tests the trajectories of the agent of Example 1 (bis): which tools it calls, with which arguments, and what it answers.

Run with: poetry run python -m snippets -l agents -e 3 [PYTEST OPTIONS, e.g. -v]
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (must support tool calling).
"""
import functools
import sys
import pytest
from snippets.lecture_agents.example1bis.agent_langchain import agent


@functools.cache  # one agent run per question, shared by the tests
def trajectory(question: str) -> tuple[list[dict], str]:
    """Runs the agent on a question, returning the tool calls it made (in order) and its final answer."""
    messages = agent.invoke({"messages": [("user", question)]}, {"recursion_limit": 20})["messages"]
    tool_calls = [call for message in messages for call in getattr(message, "tool_calls", [])]
    return tool_calls, messages[-1].content


def test_time_in_tokyo():
    tool_calls, _ = trajectory("What time is it in Tokyo?")
    assert [(call["name"], call["args"]) for call in tool_calls] == [("get_current_time", {"timezone": "Asia/Tokyo"})]


def test_weather_in_bologna():
    tool_calls, _ = trajectory("Is it raining in Bologna right now?")
    assert [call["name"] for call in tool_calls] == ["get_weather"]  # exactly one call, no wasted steps
    assert "bologna" in tool_calls[0]["args"]["location"].lower()


def test_no_tools_when_not_needed():
    tool_calls, _ = trajectory("What is the capital of France?")
    assert tool_calls == []  # well-known facts need no tools


def test_unknown_location():
    _, answer = trajectory("What's the weather like in Xyzzyville?")
    assert not any(char.isdigit() for char in answer), "the agent must not invent temperatures"


if __name__ == "__main__":  # let pytest run the tests in this file (plus any option given on the command line)
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
