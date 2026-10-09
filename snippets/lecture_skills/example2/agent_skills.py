"""
An agent from scratch, with skills: the ReAct loop of the agents lecture (example 1), plus progressive disclosure of skills.

Run with: poetry run python -m snippets -l skills -e 2 [SKILLS_FOLDER ...]   (default: the skills of example 1)
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (must support tool calling).
"""
import json
import os
import subprocess
import sys
from openai import OpenAI
from pydantic import TypeAdapter
from snippets.lecture_skills.example1 import FOLDER
from snippets.lecture_skills.skills import Skill, catalogue, discover

base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/")
api_key = os.environ.get("OPENAI_API_KEY") or input(f"Enter your API key for {base_url}: ")
model = os.environ.get("OPENAI_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

client = OpenAI(base_url=base_url, api_key=api_key)

skills = discover(*(sys.argv[1:] if __name__ == "__main__" else []) or [FOLDER])  # 1. discover (once, at startup)
role = "You are the assistant of the admission committee of a PhD programme."


def instructions() -> str:  # 2. advertise: names and descriptions only, in the system prompt
    return f"""{role}
You have SKILLS: procedures telling you how to do some tasks.
Before doing a task matching the description of a skill, load the skill with `load_skill`, then follow its instructions.
Do not load skills that are not relevant to the request. Paths in a skill's instructions are relative to the skill's folder.

Available skills:
{catalogue(skills)}"""


def file_of(skill: Skill, path: str):
    file = (skill.path / path).resolve()
    if not file.is_relative_to(skill.path.resolve()):  # never trust the LLM's arguments: e.g. "../../.ssh/id_rsa"
        raise ValueError(f"{path} is outside the skill's folder")
    return file


def load_skill(name: str) -> str:  # 3. load: level 2 of progressive disclosure
    """Loads the instructions of a skill, given its name. Use it before doing a task matching the skill's description."""
    return skills[name].body


def read_skill_file(skill: str, path: str) -> str:  # level 3: resources, read on demand
    """Reads a file of a skill (e.g. references/rubric.md), given the skill's name and a path relative to the skill's folder."""
    return file_of(skills[skill], path).read_text()


def approve(command: list[str]) -> bool:  # the human in the loop, in the controller (replaceable, e.g. in tests)
    return input(f"    [approve] run {' '.join(command[1:])}? [y/N] ").strip().lower() == "y"


def run_skill_script(skill: str, script: str, args: list[str] | None = None) -> str:  # 4. execute, upon the user's approval
    """Runs a Python script of a skill (e.g. scripts/check.py) with the given command-line arguments, and returns its output."""
    if not isinstance(args or [], list):  # LLMs may pass e.g. {"candidate": "mario-rossi"}: explain the error, they'll retry
        raise TypeError("args must be a list of strings, e.g. [\"mario-rossi\"]")
    command = [sys.executable, os.path.relpath(file_of(skills[skill], script)), *map(str, args or [])]
    if not approve(command):
        return "The user did not approve running this script."
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    return result.stdout + result.stderr


tools = [load_skill, read_skill_file, run_skill_script]


def tool_definition(function) -> dict:  # as in the agents lecture: name, docstring, and JSON Schema of the parameters
    return dict(type="function", function=dict(
        name=function.__name__,
        description=function.__doc__,
        parameters=TypeAdapter(function).json_schema(),
    ))


tool_definitions = [tool_definition(f) for f in tools]
tools_by_name = {f.__name__: f for f in tools}


def call_tool(tool_call) -> str:
    try:
        function = tools_by_name[tool_call.function.name]
        return str(function(**json.loads(tool_call.function.arguments or "{}")))
    except Exception as e:  # errors are fed back to the LLM, which may recover
        return f"Error: {e!r}"


def react(messages: list, max_steps: int = 15) -> str:  # the ReAct loop of the agents lecture, unchanged
    for _ in range(max_steps):
        response = client.chat.completions.create(model=model, messages=messages, tools=tool_definitions)
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            return message.content
        for tool_call in message.tool_calls:
            print(f"    [tool] {tool_call.function.name}({tool_call.function.arguments})")
            messages.append(dict(role="tool", tool_call_id=tool_call.id, content=call_tool(tool_call)))
    return f"Sorry, I could not answer in {max_steps} steps."


if __name__ == "__main__":
    print("# Skills:", ", ".join(skills) or "none")
    messages = [dict(role="system", content=instructions())]
    while True:
        try:
            messages.append(dict(role="user", content=input("You: ")))
        except (EOFError, KeyboardInterrupt):
            break
        print(f"AI: {react(messages)}")
