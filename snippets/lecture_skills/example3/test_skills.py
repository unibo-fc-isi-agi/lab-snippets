"""
Tests the skills of Example 1: static checks (they are valid), and triggering tests (the agent of Example 2 loads the right one, or none).

Run with: poetry run python -m snippets -l skills -e 3 [PYTEST OPTIONS, e.g. -v, or -k valid for the static checks only]
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (must support tool calling).
"""
import json
import subprocess
import sys
import pytest
from snippets.lecture_skills.example2 import agent_skills as agent
from snippets.lecture_skills.skills import validate


@pytest.mark.parametrize("name", sorted(agent.skills))
def test_skill_is_valid(name):  # static: front matter, referenced files
    assert validate(agent.skills[name]) == []


def test_script_runs():  # static: the skill's script works, and refuses unknown candidates
    script = agent.skills["phd-application-review"].path / "scripts" / "check_application.py"
    output = subprocess.run([sys.executable, script, "mario-rossi"], capture_output=True, text=True, check=True).stdout
    assert json.loads(output)["missing"] == []
    assert subprocess.run([sys.executable, script, "../../etc/passwd"], capture_output=True).returncode != 0


# requests, and the skill which should be loaded (None: no skill). Near misses are the most informative cases
TRIGGERS = [
    ("Review the application of Mario Rossi.", "phd-application-review"),
    ("Is Jean Dupont's recommendation letter convincing? Assess it.", "phd-application-review"),
    ("Write the letter telling Mohammed Ali that he is admitted.", "admission-letter"),
    ("Invite Jean Dupont to the interview on 7 July at 10:00: prepare the e-mail.", "admission-letter"),
    ("What time is it in Tokyo?", None),
    ("Translate 'admission committee' into Italian.", None),
    ("Write a recommendation letter for my student Anna Bianchi.", None),  # near miss: a letter, but not to a candidate
]


def first_skill(request: str) -> str | None:
    """One step of the agent: the skill it loads first, if any (only the first action matters for triggering)."""
    messages = [dict(role="system", content=agent.instructions()), dict(role="user", content=request)]
    response = agent.client.chat.completions.create(model=agent.model, messages=messages, tools=agent.tool_definitions)
    for call in response.choices[0].message.tool_calls or []:
        if call.function.name == "load_skill":
            return json.loads(call.function.arguments)["name"]
    return None


@pytest.mark.parametrize("request_, expected", TRIGGERS)
def test_triggering(request_, expected):
    runs = [first_skill(request_) for _ in range(3)]  # LLMs are not deterministic: majority of 3 runs
    assert runs.count(expected) >= 2, f"expected {expected}, got {runs}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
