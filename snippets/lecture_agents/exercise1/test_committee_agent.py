"""
Trajectory tests of the committee's agent (agent_committee.py): right tools, valid arguments, no re-reads, grounded answers.
Expected facts are written by hand, by looking at the documents in data/.

Run with: poetry run python -m snippets -l agents -x 1 [PYTEST OPTIONS, e.g. -v]   (then pick test_committee_agent.py)
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (must support tool calling), VISION_MODEL (must support images).
"""
import functools
import re
import sys
from collections import Counter
from datetime import date
import pytest
import data
from snippets.lecture_agents.exercise1.agent_committee import agent, ask

trajectory = functools.cache(lambda question: ask(agent, question))  # one agent run per question, shared by the tests

HIGHEST_GRADE = "Which candidate has the highest final grade, relative to the grading scale of their university?"
AGE = "How old is Mohammed Ali?"
NAMES = "Is the name in Jean Dupont's passport the same as in the letter about them?"
UNKNOWN = "What is Mario Rossi's phone number?"


def calls_to(tool_calls: list[dict], tool: str) -> list[dict]:
    return [call["args"] for call in tool_calls if call["name"] == tool]


@pytest.mark.parametrize("question", [HIGHEST_GRADE, AGE, NAMES, UNKNOWN])
def test_valid_arguments_and_no_rereads(question):  # properties of ANY trajectory
    tool_calls, _ = trajectory(question)
    assert all(call["args"].get("candidate", "mario-rossi") in data.CANDIDATES for call in tool_calls), "invalid candidate IDs"
    reads = Counter((call["name"], call["args"].get("candidate")) for call in tool_calls if call["name"].startswith("read_"))
    assert all(count == 1 for count in reads.values()), f"documents read more than once: {reads}"


def test_highest_grade():  # all candidates: 105/110 (95%) vs. 14.7/20 (73.5%) vs. 3.42/4 (85.5%)
    tool_calls, answer = trajectory(HIGHEST_GRADE)
    assert sorted(args["candidate"] for args in calls_to(tool_calls, "read_transcript")) == data.CANDIDATES
    assert "Mario Rossi" in answer


def test_age():  # one document, plus the current date
    tool_calls, answer = trajectory(AGE)
    assert calls_to(tool_calls, "read_passport") == [{"candidate": "mohammed-ali"}]
    assert calls_to(tool_calls, "get_current_time"), "the agent must not assume today's date"
    born, today = date(1990, 7, 15), date.today()
    age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    assert str(age) in answer


def test_names_across_documents():  # several documents of the same candidate
    tool_calls, _ = trajectory(NAMES)
    assert calls_to(tool_calls, "read_passport") == [{"candidate": "jean-dupont"}]
    assert calls_to(tool_calls, "read_letter") == [{"candidate": "jean-dupont"}]


def test_admits_ignorance():  # the documents do not say it: the agent must not invent it
    _, answer = trajectory(UNKNOWN)
    assert not re.search(r"\d{3}[\s.-]?\d{3,}", answer), f"invented phone number? {answer}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
