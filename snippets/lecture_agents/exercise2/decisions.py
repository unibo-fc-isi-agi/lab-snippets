"""
Write-enabled tools, acting on behalf of the committee: they have SIDE EFFECTS (files under COMMITTEE_OUTPUT, default: output/),
so they must never be executed without the approval of a committee member (cf. agent_decisions.py).
Tools validate whatever the LLM may get wrong, and are idempotent where possible.
"""
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal
from pydantic import Field
from snippets.lecture_agents.exercise1 import committee
from snippets.lecture_agents.exercise1.committee import Candidate, check

OUTPUT_DIR = Path(os.environ.get("COMMITTEE_OUTPUT", "output"))  # decisions.csv, interviews.csv, outbox/ (tests redirect it elsewhere)

# a 4th candidate, whose letter contains a prompt injection (cf. the letter itself): part of this exercise's scenario
committee.LETTERS["eve-mallory"] = Path(__file__).parent / "letter-eve-mallory.txt"


def _upsert(file: Path, candidate: str, **row) -> None:
    """Writes the row of a candidate into a CSV file, REPLACING any previous row of theirs: so, doing it twice is the same as once."""
    rows = {}
    if file.exists():
        with file.open(newline="") as f:
            rows = {r["candidate"]: r for r in csv.DictReader(f)}
    rows[candidate] = dict(candidate=candidate, **row, recorded_at=datetime.now().isoformat(timespec="seconds"))
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[candidate]))
        writer.writeheader()
        writer.writerows(rows.values())


def record_decision(
    candidate: Candidate,
    decision: Annotated[Literal["admit", "reject", "interview"], Field(description="The committee's decision")],
    motivation: Annotated[str, Field(description="Why the committee took this decision, grounded on the application's documents")],
) -> str:
    """Record the committee's decision about a candidate. A new decision about the same candidate replaces the previous one."""
    _upsert(OUTPUT_DIR / "decisions.csv", check(candidate), decision=decision, motivation=motivation)
    return f"Decision recorded: {decision} for {candidate}."


def schedule_interview(
    candidate: Candidate,
    when: Annotated[datetime, Field(description="Date and time of the interview, in ISO 8601 format, e.g. '2026-11-03T10:30'")],
) -> str:
    """Schedule the interview of a candidate, on a working day in the future. Scheduling it again re-schedules it."""
    when = when.astimezone().replace(tzinfo=None) if when.tzinfo else when  # local time, so that it can be compared with now()
    if when <= datetime.now():  # validation in the tool: the LLM may get dates wrong, whatever the prompt says
        raise ValueError(f"{when} is in the past: interviews must be scheduled in the future.")
    if when.weekday() == 6:
        raise ValueError(f"{when:%Y-%m-%d} is a Sunday: pick a working day.")
    _upsert(OUTPUT_DIR / "interviews.csv", check(candidate), when=when.isoformat(timespec="minutes"))
    return f"Interview of {candidate} scheduled on {when:%A %Y-%m-%d at %H:%M}."


def send_email(
    candidate: Candidate,
    subject: Annotated[str, Field(description="Subject of the e-mail")],
    body: Annotated[str, Field(description="Body of the e-mail, in plain text")],
) -> str:
    """Send an e-mail to a candidate. Beware: e-mails cannot be unsent."""
    # simulated: the e-mail is written into the outbox, one file per e-mail (NOT idempotent: as real e-mails)
    file = OUTPUT_DIR / "outbox" / f"{datetime.now():%Y%m%d-%H%M%S-%f}-{check(candidate)}.txt"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(f"To: {candidate}\nSubject: {subject}\n\n{body}\n")
    return f"E-mail sent to {candidate}."


tools = [record_decision, schedule_interview, send_email]


if __name__ == "__main__":  # shows what the LLM sees of each tool (as in committee.py)
    import json
    from langchain_core.utils.function_calling import convert_to_openai_tool
    for function in tools:
        print(json.dumps(convert_to_openai_tool(function), indent=2))
