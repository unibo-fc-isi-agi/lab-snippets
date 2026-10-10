"""
The admission workflow of the running example, as STEPS: plain functions over a shared state, reused by
Example 1 (a state machine written from scratch) and Example 1 bis (the same workflow, with LangGraph).
Each step reads the state, and returns a PARTIAL update: only the fields it changes.

LLM steps reuse the extraction and scoring of committee.py, which import their LLMs lazily:
importing this module requires no API key.
"""
import os
from pathlib import Path
from typing import TypedDict
from snippets.lecture_workflows import committee

OUTPUT_DIR = Path(os.environ.get("WORKFLOWS_OUTPUT", "output/workflows"))  # checkpoints, outbox (tests redirect it elsewhere)
INTERVIEW_SLOTS = 2  # how many candidates are invited to the interview


class State(TypedDict, total=False):  # small, and JSON-friendly: it is saved after each step
    run: str                         # ID of the run (e.g. "2026-10-12T10:30"), used to name the e-mails it sends
    candidates: list[str]            # IDs of the candidates, e.g. ["jean-dupont", "mario-rossi", "mohammed-ali"]
    transcripts: dict[str, dict]     # candidate -> information extracted from their transcript of records
    problems: dict[str, list[str]]   # candidate -> why they are NOT eligible (an empty list: eligible)
    scores: dict[str, dict]          # candidate -> score of their recommendation letter
    ranking: list[dict]              # eligible candidates, best first: [{"candidate": ..., "points": ..., "invited": ...}]
    approved: bool                   # the committee's decision on the proposal
    notified: list[str]              # candidates e-mailed so far


# --- LLM steps (the only non-deterministic ones) ---

def extract_transcript(candidate: str) -> dict:  # a vision LLM reads the picture of the transcript
    return committee.read_transcript(candidate)


def score_letter(candidate: str) -> dict:  # the letter scoring of the prompting lecture (Example 1 bis)
    return committee.score_letter(candidate)


def read_transcripts(state: State) -> State:
    return {"transcripts": {candidate: extract_transcript(candidate) for candidate in state["candidates"]}}


def score_letters(state: State) -> State:  # only for ELIGIBLE candidates: no LLM calls are wasted on the others
    return {"scores": {candidate: score_letter(candidate) for candidate in eligible(state)}}


# --- data steps (pure code: deterministic, testable without LLMs) ---

MASTER_DEGREES = ("master", "magistrale", "msc", "m.sc")  # how Master's degrees are named, in the transcripts we know of


def check_eligibility(state: State) -> State:
    """Article 3 of the regulations: a Master's degree is required. Plus a GATE on what the LLM extracted."""
    problems = {}
    for candidate, transcript in state["transcripts"].items():
        problems[candidate] = []
        if not any(word in transcript["degree"].lower() for word in MASTER_DEGREES):
            problems[candidate].append(f"a Master's degree is required (Art. 3), found: {transcript['degree']!r}")
        if not 0 < transcript["final_grade_value"] <= transcript["final_grade_max"]:  # e.g. '105/110' misread as 105510
            problems[candidate].append(f"implausible grade {transcript['final_grade_value']}/{transcript['final_grade_max']}: check it")
    return {"problems": problems}


def eligible(state: State) -> list[str]:
    return [candidate for candidate, problems in state["problems"].items() if not problems]


def rank(state: State) -> State:
    """Article 8 of the regulations, for the documents we have: academic record (up to 30 points) + letter (up to 10)."""
    rows = []
    for candidate in eligible(state):
        grade, letter = state["transcripts"][candidate], state["scores"][candidate]
        points = 30 * grade["final_grade_percentage"] / 100 + 10 * letter["score"] / letter["max_score"]
        rows.append({"candidate": candidate, "points": round(points, 1)})
    rows.sort(key=lambda row: row["points"], reverse=True)
    for position, row in enumerate(rows):
        row["invited"] = position < INTERVIEW_SLOTS
    return {"ranking": rows}


# --- user-input step: what the committee is asked (the engines decide HOW to ask) ---

def approval_request(state: State) -> str:
    lines = [f"  {i}. {row['candidate']}: {row['points']} points" + (" -> interview" if row["invited"] else "")
             for i, row in enumerate(state.get("ranking", []), start=1)]
    lines += [f"  -  {candidate}: NOT eligible, {'; '.join(problems)}" for candidate, problems in state["problems"].items() if problems]
    return "Proposal of the workflow (candidates will be notified accordingly):\n" + "\n".join(lines) + "\nDo you approve it?"


def is_yes(answer: str) -> bool:
    return answer.strip().lower() in ("y", "yes")


# --- action step (side effects!) ---

def send_email(run: str, candidate: str, subject: str, body: str) -> bool:
    """Simulated: writes the e-mail into the outbox. IDEMPOTENT: one e-mail per run and candidate, even if called twice."""
    file = OUTPUT_DIR / "outbox" / f"{run}-{candidate}.txt".replace(":", "")
    if file.exists():
        return False  # already sent, e.g. by a run which crashed (or was resumed) after sending it: do not send it twice
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(f"To: {candidate}\nSubject: {subject}\n\n{body}\n")
    return True


def notify(state: State) -> State:
    outcomes = {row["candidate"]: "You are invited to the interview." if row["invited"] else "You are not among the candidates invited to the interview."
                for row in state.get("ranking", [])}
    outcomes |= {candidate: "Your application is not eligible: " + "; ".join(problems) + "."
                 for candidate, problems in state["problems"].items() if problems}
    for candidate, outcome in outcomes.items():
        send_email(state["run"], candidate, "Your application to the PhD programme", f"Dear candidate,\n{outcome}\nBest regards,\nThe committee")
    return {"notified": sorted(outcomes)}
