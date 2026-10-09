"""
Tests of the thesis-precheck skill: static checks, triggering (from evals/evals.json), and the outcome on the teacher's thesis.

Run with: poetry run python -m snippets -l skills -x 1 [PYTEST OPTIONS]   (then pick test_precheck.py)
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (must support tool calling).
"""
import json
import sys
import pytest
from snippets.lecture_skills.example3.test_skills import first_skill
from snippets.lecture_skills.exercise1 import precheck  # i.e. the agent of Example 2, with the skill of this exercise
from snippets.lecture_skills.skills import validate

SKILL = precheck.agent.skills["thesis-precheck"]
EVALS = json.loads((SKILL.path / "evals" / "evals.json").read_text())


def test_skill_is_valid():
    assert validate(SKILL) == []


@pytest.mark.parametrize("case", EVALS["triggers"], ids=lambda case: case["query"][:40])
def test_triggering(case):
    runs = [first_skill(case["query"]) for _ in range(3)]
    expected = "thesis-precheck" if case["should_trigger"] else None
    assert runs.count(expected) >= 2, f"expected {expected}, got {runs}"


def test_outcome_on_teachers_thesis(monkeypatch):
    monkeypatch.setattr(precheck.agent, "approve", lambda command: True)  # no human in the loop, in tests
    request = f"Pre-check my thesis, {precheck.thesis([])}, before submission. Write sections to output/thesis-sections/."
    messages = [dict(role="system", content=precheck.agent.instructions()), dict(role="user", content=request)]
    report = precheck.agent.react(messages, max_steps=25)
    loaded = [call["function"]["name"] for message in messages for call in message.get("tool_calls", [])]
    assert loaded[0] == "load_skill"  # the skill is loaded before acting
    assert "| Severity |" in report or "| # |" in report  # the report follows the template
    assert "314" in report  # page with "the the": a fact found by the script, which must be reported


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
