"""
Tests of the admission workflow (Example 1 bis), at three levels:
- STEPS: data and action steps are plain functions, tested with no LLM;
- PATHS: LLM steps are replaced by FAKES (fixed outputs, written by hand), to test the graph cheaply and deterministically;
- END-TO-END: real LLMs, real documents, and the expectations of the validating lecture's dataset (test_data.yml).

Run with: poetry run python -m snippets -l workflows -e 6 [PYTEST OPTIONS, e.g. -v, or -k "not end_to_end" to skip LLM calls]
Configure via env vars (end-to-end test only): OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL, VISION_MODEL (must support images).
"""
import sys
import pytest
from langgraph.types import Command
import data
from snippets.lecture_validating.dataset import TEST_CASES
from snippets.lecture_workflows import admission
from snippets.lecture_workflows.example1bis.admission_langgraph import build, checkpointer

# what a CORRECT extraction and scoring return (by looking at the documents): the fakes' outputs
TRANSCRIPTS = {
    "jean-dupont": {"degree": "Master Informatique", "final_grade_value": 14.7, "final_grade_max": 20, "final_grade_percentage": 73.5},
    "mario-rossi": {"degree": "Laurea Magistrale in Informatica", "final_grade_value": 105, "final_grade_max": 110, "final_grade_percentage": 95.5},
    "mohammed-ali": {"degree": "Bachelor of Science in Software Engineering", "final_grade_value": 3.42, "final_grade_max": 4, "final_grade_percentage": 85.5},
}
SCORES = {"jean-dupont": 4, "mario-rossi": 5, "mohammed-ali": 3}


@pytest.fixture(autouse=True)
def output_dir(tmp_path, monkeypatch):  # each test writes checkpoints and e-mails into a fresh directory
    monkeypatch.setattr(admission, "OUTPUT_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def fake_llm(monkeypatch):  # replaces the LLM steps, recording the calls
    calls = []
    monkeypatch.setattr(admission, "extract_transcript", lambda c: calls.append(("transcript", c)) or TRANSCRIPTS[c])
    monkeypatch.setattr(admission, "score_letter", lambda c: calls.append(("letter", c)) or {"score": SCORES[c], "max_score": 5})
    return calls


def run(workflow, thread: str, answer: str | None) -> dict:
    """Runs a thread until the approval, answers it (if an answer is given), and returns the final state."""
    config = {"configurable": {"thread_id": thread}}
    result = workflow.invoke({"run": thread, "candidates": data.CANDIDATES}, config)
    assert "__interrupt__" in result, "the workflow must ask for approval"
    return result if answer is None else workflow.invoke(Command(resume=answer), config)


# --- steps ---

def test_eligibility():
    problems = admission.check_eligibility({"transcripts": TRANSCRIPTS})["problems"]
    assert problems["mario-rossi"] == problems["jean-dupont"] == [] and "Art. 3" in problems["mohammed-ali"][0]
    assert "implausible" in admission.check_eligibility({"transcripts": {"x": TRANSCRIPTS["mario-rossi"] | {"final_grade_value": 105510}}})["problems"]["x"][0]


def test_ranking():
    state = {"transcripts": TRANSCRIPTS, "problems": {"mario-rossi": [], "jean-dupont": [], "mohammed-ali": ["Art. 3"]},
             "scores": {c: {"score": s, "max_score": 5} for c, s in SCORES.items()}}
    ranking = admission.rank(state)["ranking"]
    assert [row["candidate"] for row in ranking] == ["mario-rossi", "jean-dupont"]  # mohammed-ali is not eligible
    assert ranking[0]["points"] == pytest.approx(30 * 0.955 + 10, abs=0.1)  # Article 8: grade (up to 30) + letter (up to 10)
    assert sum(row["invited"] for row in ranking) <= admission.INTERVIEW_SLOTS


def test_notify_is_idempotent(output_dir):
    state = {"run": "r1", "problems": {"mohammed-ali": ["Art. 3"]}, "ranking": [{"candidate": "mario-rossi", "invited": True}]}
    admission.notify(state)
    admission.notify(state)  # e.g. a resumed run, re-executing the step
    assert len(list((output_dir / "outbox").iterdir())) == 2  # one e-mail per candidate, not two


# --- paths (fake LLMs) ---

def test_approved_proposal_notifies_everybody(fake_llm, output_dir):
    final = run(build(checkpointer()), "approved", "yes")
    assert final["approved"] and final["notified"] == sorted(data.CANDIDATES)
    assert len(list((output_dir / "outbox").iterdir())) == len(data.CANDIDATES)


def test_rejected_proposal_notifies_nobody(fake_llm, output_dir):
    final = run(build(checkpointer()), "rejected", "no")
    assert not final["approved"] and "notified" not in final and not (output_dir / "outbox").exists()


def test_no_letters_scored_for_ineligible_candidates(fake_llm):
    run(build(checkpointer()), "scoring", "no")
    assert ("letter", "mohammed-ali") not in fake_llm and ("letter", "mario-rossi") in fake_llm


def test_resume_from_another_process(fake_llm, output_dir):
    run(build(checkpointer()), "durable", answer=None)  # stops at the approval...
    fake_llm.clear()
    another = build(checkpointer())  # ... and is resumed by ANOTHER instance of the workflow (e.g. after a restart)
    final = another.invoke(Command(resume="yes"), {"configurable": {"thread_id": "durable"}})
    assert final["notified"] == sorted(data.CANDIDATES)
    assert fake_llm == []  # nothing was re-extracted, nor re-scored: the state came from the checkpoint


# --- end-to-end (real LLMs: slow, costly, non-deterministic) ---

def test_end_to_end():
    final = run(build(checkpointer()), "end-to-end", "yes")
    assert admission.eligible(final) == ["jean-dupont", "mario-rossi"]  # Mohammed Ali has a Bachelor's degree only
    assert final["ranking"][0]["candidate"] == "mario-rossi"
    for case in TEST_CASES:  # the letters' scores, within the ranges of the validating lecture's dataset
        candidate, expected = case["input"].removeprefix("letter-").removesuffix(".txt"), case["expectations"]
        if candidate in final["scores"]:
            assert expected.get("min_score", 0) <= final["scores"][candidate]["score"] <= expected.get("max_score", 5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
