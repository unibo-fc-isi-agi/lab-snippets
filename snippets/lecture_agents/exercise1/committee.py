"""
Read-only tools to inspect the candidates' applications: plain Python functions, documented for the LLM (as in simple_tools.py).
Information is extracted from pictures INSIDE the tools (via LLM-based workflows), so the agent only sees (small) structured data.
LLM-based workflows are imported lazily (i.e. upon first use), so that importing this module requires no API key (cf. Exercise 2 and 3).
"""
import functools
from datetime import date
from pathlib import Path
from typing import Annotated, Optional
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
import data

# where each candidate's letter is: candidates are the ones with a letter (Exercise 2 adds one, by extending this dictionary)
LETTERS: dict[str, Path] = {candidate: data.letter(candidate) for candidate in data.CANDIDATES}

Candidate = Annotated[str, Field(description="ID of a candidate, as returned by list_candidates, e.g. 'mario-rossi'")]


def check(candidate: str) -> str:  # not a tool: never trust the LLM's arguments (e.g. candidate "../../.ssh/id_rsa")
    if candidate not in LETTERS:  # an allow-list: only known IDs, hence only known files, can be read
        raise ValueError(f"Unknown candidate: {candidate!r}. Valid candidates are: {', '.join(sorted(LETTERS))}.")
    return candidate


def list_candidates() -> list[str]:
    """List the IDs of all candidates, to be used as arguments of the other tools."""
    return sorted(LETTERS)


def read_letter(candidate: Candidate) -> str:
    """Read the full text of the recommendation letter of a candidate."""
    text = LETTERS[check(candidate)].read_text()
    # the letter is wrapped in tags, so that the LLM can tell (untrusted) documents from instructions
    return f"<letter candidate={candidate!r}>\n{text}\n</letter>"


# extraction is slow and costly, and files do not change: results are cached (in memory, for the lifetime of the process).
# the cache is on helper functions, not on tools, so that tools keep their plain signatures (from which their JSON schemas are derived)
@functools.cache
def _passport(candidate: str) -> dict:
    from snippets.lecture_prompting.exercise2.id_extraction import extract
    info, to_be_reviewed = extract(data.passport(candidate), samples=3)  # the workflow of Exercise 2 of the prompting lecture
    return dict(info.model_dump(mode="json"), fields_to_be_reviewed_by_humans=to_be_reviewed)


def read_passport(candidate: Candidate) -> dict:
    """Read the passport of a candidate: name, nationality, date of birth, document number, and expiry date.
    Fields whose extraction is uncertain are listed in 'fields_to_be_reviewed_by_humans'."""
    if not data.passport(check(candidate)).exists():
        raise ValueError(f"No passport on file for {candidate!r}.")
    return _passport(candidate)


class Course(BaseModel):
    name: str = Field(description="Name of the course, as written in the transcript.")
    credits: float = Field(description="Credits (e.g. ECTS) of the course.")
    grade: str = Field(description="Grade obtained, as written in the transcript (e.g. '28', '30 e lode', '17,5', 'A-').")

class TranscriptInfo(BaseModel):
    """Structured information extracted from a transcript of records."""

    name: str = Field(description="Full name of the student, given names first.")
    degree: str = Field(description="Degree and programme, e.g. 'Master in Computer Science'.")
    graduation_date: Optional[date] = Field(description="Date of graduation, if any.")
    final_grade: str = Field(description="Final grade of the degree, as written in the transcript, including its scale (e.g. '100/110').")
    final_grade_value: float = Field(description="Numeric value of the final grade (e.g. 100 for '100/110').")
    final_grade_max: float = Field(description="Maximum value of the final grade's scale (e.g. 110 for '100/110').")
    courses: list[Course] = Field(description="All the courses in the transcript, excluding the final exam or thesis.")

@functools.cache
def _transcript(candidate: str) -> dict:
    from snippets.lecture_prompting.exercise2.id_extraction import message_with_image, vision_llm
    info = vision_llm.with_structured_output(TranscriptInfo).invoke([
        SystemMessage("You extract structured information from the pictures of transcripts of records. Report values exactly as written."),
        message_with_image("Please extract the information from this transcript.", data.transcript(candidate)),
    ])
    # grades are on different scales (110, 20, 4.0, ...): comparing them is a computation, so it is done by code, not by the LLM
    return dict(info.model_dump(mode="json"), final_grade_percentage=round(100 * info.final_grade_value / info.final_grade_max, 1))


def read_transcript(candidate: Candidate) -> dict:
    """Read the transcript of records of a candidate: degree, graduation date, final grade (also as a percentage
    of its scale, to compare candidates graduated under different grading systems), and courses with credits and grades."""
    if not data.transcript(check(candidate)).exists():
        raise ValueError(f"No transcript on file for {candidate!r}.")
    return _transcript(candidate)


@functools.cache
def _score(candidate: str) -> dict:
    from snippets.lecture_prompting.exercise1.letter_scoring_checklist import score_letter as score_letter_text
    info = score_letter_text(LETTERS[candidate].read_text())  # the checklist-based scoring of Exercise 1 of the prompting lecture
    return dict(score=info.score, max_score=5, penalties=info.penalties)


def score_letter(candidate: Candidate) -> dict:
    """Score the recommendation letter of a candidate, from 0 (worst) to 5 (best), with the reasons for each lost point."""
    return _score(check(candidate))


tools = [list_candidates, read_letter, read_passport, read_transcript, score_letter]  # the tools made available to the agent


if __name__ == "__main__":  # shows what the LLM sees of each tool, i.e. its definition (as a JSON Schema), e.g. to debug docstrings
    import json
    from langchain_core.utils.function_calling import convert_to_openai_tool
    for function in tools:
        print(json.dumps(convert_to_openai_tool(function), indent=2))
