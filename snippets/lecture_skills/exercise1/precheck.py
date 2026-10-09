"""
Pre-checks a thesis with the thesis-precheck skill, via the agent of Example 2 (plus a tool to read the thesis' sections).
Without arguments, it downloads the teacher's PhD thesis.

Run with: poetry run python -m snippets -l skills -x 1 [path/to/thesis.pdf]   (then pick precheck.py)
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (must support tool calling).
"""
import sys
import urllib.request
from pathlib import Path
from snippets.lecture_skills.example2 import agent_skills as agent
from snippets.lecture_skills.skills import discover

FOLDER = Path(__file__).parent  # contains the thesis-precheck/ skill
OUTPUT = Path("output")  # where the thesis is downloaded, and its sections are written
DEFAULT_THESIS = ("https://github.com/gciatto/phd-thesis/releases/download/"
                  "1.1.0%2B2022-07-16-14-34/phd-thesis-1.1.0%2B2022-07-16-14-34.pdf")


def read_output_file(path: str) -> str:
    """Reads a text file written in the output/ folder (e.g. a section of the thesis), given its path."""
    file = Path(path).resolve()
    if not file.is_relative_to(OUTPUT.resolve()):  # the skill's scripts write there; nothing else is readable
        raise ValueError(f"{path} is outside {OUTPUT}/")
    return file.read_text()


def thesis(argv: list[str]) -> Path:
    if argv:
        return Path(argv[0])
    file = OUTPUT / "phd-thesis.pdf"
    if not file.exists():
        print(f"# Downloading {DEFAULT_THESIS} to {file}")
        OUTPUT.mkdir(exist_ok=True)
        urllib.request.urlretrieve(DEFAULT_THESIS, file)
    return file


# the agent of Example 2, with the skill of this exercise, and one more (generic, yet confined) tool
agent.skills = discover(FOLDER)
agent.role = "You assist students in finalising their theses."
agent.tools_by_name[read_output_file.__name__] = read_output_file
agent.tool_definitions.append(agent.tool_definition(read_output_file))

if __name__ == "__main__":
    request = f"Pre-check my thesis, {thesis(sys.argv[1:])}, before submission. Write sections to {OUTPUT}/thesis-sections/."
    messages = [dict(role="system", content=agent.instructions()), dict(role="user", content=request)]
    print(agent.react(messages, max_steps=25))
