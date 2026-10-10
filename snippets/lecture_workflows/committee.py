"""
The committee's documents and actions, shared by the examples of this lecture: plain functions, documented for the LLM,
usable both as workflow steps and as tools. LLMs are created lazily, so importing this module requires no API key.

- read_transcript: a vision LLM extracts the degree and final grade from the picture of a transcript of records;
- score_letter: the letter-scoring system of the prompting lecture (Example 1 bis);
- schedule_interview, send_email: actions with side effects, simulated by writing files (under WORKFLOWS_OUTPUT).

Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL, VISION_MODEL (must support images), WORKFLOWS_OUTPUT.
"""
import base64
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated
from pydantic import BaseModel, Field
import data

OUTPUT_DIR = Path(os.environ.get("WORKFLOWS_OUTPUT", "output/workflows"))  # outbox, interviews (tests redirect it elsewhere)

Candidate = Annotated[str, Field(description=f"ID of a candidate, one of: {', '.join(data.CANDIDATES)}")]


def check(candidate: str) -> str:  # never trust the LLM's arguments: an allow-list of candidates
    if candidate not in data.CANDIDATES:
        raise ValueError(f"Unknown candidate: {candidate!r}. Valid candidates are: {', '.join(data.CANDIDATES)}.")
    return candidate


def list_candidates() -> list[str]:
    """List the IDs of all candidates."""
    return data.CANDIDATES


def read_letter(candidate: Candidate) -> str:
    """Read the full text of the recommendation letter of a candidate."""
    return f"<letter candidate={candidate!r}>\n{data.letter(check(candidate)).read_text()}\n</letter>"


class Transcript(BaseModel):
    degree: str = Field(description="Degree and programme, as written, e.g. 'Laurea Magistrale in Informatica'.")
    final_grade_value: float = Field(description="Numeric value of the final grade (e.g. 105 for '105/110').")
    final_grade_max: float = Field(description="Maximum value of the final grade's scale (e.g. 110 for '105/110').")


def read_transcript(candidate: Candidate) -> dict:
    """Read the transcript of records of a candidate: degree, and final grade (also as a percentage of its scale)."""
    from langchain_openai import ChatOpenAI
    picture = base64.b64encode(data.transcript(check(candidate)).read_bytes()).decode()
    vision_llm = ChatOpenAI(base_url=os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/"), api_key=os.environ.get("OPENAI_API_KEY"),
                            model=os.environ.get("VISION_MODEL", "google/gemma-4-26b-a4b-it:free"))
    info = vision_llm.with_structured_output(Transcript).invoke([("user", [
        {"type": "text", "text": "Extract the degree and the final grade from this transcript of records, exactly as written."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{picture}"}},
    ])])
    # grades are on different scales (110, 20, 4.0): comparing them is a computation, done by code
    return dict(info.model_dump(), final_grade_percentage=round(100 * info.final_grade_value / info.final_grade_max, 1))


def score_letter(candidate: Candidate) -> dict:
    """Score the recommendation letter of a candidate, from 0 (worst) to 5 (best)."""
    from snippets.lecture_prompting.example1bis.letter_scoring_langchain import score_letter as score_letter_text
    for _ in range(3):  # small LLMs may return scores out of range (e.g. 70): retry, then give up
        score = score_letter_text(data.letter(check(candidate)).read_text()).score
        if 0 <= score <= 5:
            return {"score": score, "max_score": 5}
    raise ValueError(f"Implausible letter score ({score}) for {candidate}: check the letter by hand.")


def schedule_interview(
    candidate: Candidate,
    when: Annotated[datetime, Field(description="Date and time of the interview, in ISO 8601 format, e.g. '2026-11-03T10:30'")],
) -> str:
    """Schedule the interview of a candidate, in the future. Scheduling it again re-schedules it."""
    if when.replace(tzinfo=None) <= datetime.now():
        raise ValueError(f"{when} is in the past: interviews must be scheduled in the future.")
    file = OUTPUT_DIR / "interviews" / f"{check(candidate)}.txt"  # one file per candidate: idempotent
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(when.isoformat(timespec="minutes"))
    return f"Interview of {candidate} scheduled on {when:%A %Y-%m-%d at %H:%M}."


def send_email(
    candidate: Candidate,
    subject: Annotated[str, Field(description="Subject of the e-mail")],
    body: Annotated[str, Field(description="Body of the e-mail, in plain text")],
) -> str:
    """Send an e-mail to a candidate. Beware: e-mails cannot be unsent."""
    file = OUTPUT_DIR / "outbox" / f"{datetime.now():%Y%m%d-%H%M%S-%f}-{check(candidate)}.txt"  # simulated
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(f"To: {candidate}\nSubject: {subject}\n\n{body}\n")
    return f"E-mail sent to {candidate}."


read_only_tools = [list_candidates, read_letter, read_transcript, score_letter]
write_tools = [schedule_interview, send_email]  # side effects: a human must approve them
