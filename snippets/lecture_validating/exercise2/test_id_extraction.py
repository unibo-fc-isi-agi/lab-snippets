"""
Tests the ID-document extractor of Exercise 2 of the prompting lecture (with pytest): accuracy of single and voted extractions,
invariants, consistency across samples, and an LLM-as-a-judge (validated against human judgement) for what code cannot check.

Run with: poetry run python -m snippets -l validating -x 2 [PYTEST OPTIONS, e.g. -v -s]
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, VISION_MODEL (system under test), JUDGE_MODEL (judge, must support images, default: VISION_MODEL),
SAMPLES (samples per picture, default: 3). Run the suite once per model (e.g. VISION_MODEL=... poetry run ...), and compare the reports.
"""
import functools
import os
import sys
import unicodedata
from datetime import date
from pathlib import Path
import pytest
import yaml
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import data
from snippets.lecture_prompting.exercise2.id_extraction import (
    IDDocumentInfo, sample, vote, message_with_image, vision_model, api_key, base_url,
)

# step 1: the test dataset (a file, not code: cf. passports.yml)
TEST_CASES = yaml.safe_load((Path(__file__).parent / "passports.yml").read_text())
for_each_case = pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c["input"])
SAMPLES = int(os.environ.get("SAMPLES", 3))  # cost: SAMPLES x 3 pictures requests (+ 3 for the judge): keep it small while developing
FIELDS = list(IDDocumentInfo.model_fields)


@functools.cache  # NOT a cache of LLM responses across runs (that would hide variance), just one batch of samples per picture per run
def samples_of(picture: str) -> list[IDDocumentInfo]:
    return sample(data.DIR / picture, SAMPLES)


# step 2: deterministic scorers, i.e. exact match for IDs and dates, normalised match for names (case, accents, order of names)
def normalise_name(name: str) -> set[str]:
    """e.g. 'DUPONT Jean' -> {'jean', 'dupont'}"""
    return set(unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().replace(",", " ").split())

def correct(info: IDDocumentInfo, expected: dict, field: str) -> bool:
    actual = getattr(info, field)
    if field in {"name", "nationality"}:
        return normalise_name(actual) == normalise_name(expected[field])
    if field == "id_number":
        return actual.strip().upper() == expected[field]
    return actual == expected[field]  # dates: `date` objects on both sides, so the format does not matter

def accuracy(results: list[IDDocumentInfo], expected: dict) -> float:
    """Fraction of correct fields, over all fields of all results."""
    return sum(correct(r, expected, f) for r in results for f in FIELDS) / (len(results) * len(FIELDS))


@for_each_case
@pytest.mark.parametrize("field", FIELDS)
def test_voted_field_is_correct(case, field):  # the system, as the committee would use it, i.e. with voting
    voted, agreement = vote(samples_of(case["input"]))
    assert correct(voted, case["expected"], field), f"{field}: {getattr(voted, field)!r} (agreement: {agreement[field]:.0%})"


# invariants: checked on EVERY sample, as they must hold whatever the expected values are
@for_each_case
def test_invariants(case):
    for info in samples_of(case["input"]):
        assert info.date_of_birth < info.expiration_date
        assert (date.today() - info.date_of_birth).days >= 18 * 365.25, "candidates must be of age"
        assert info.id_number.strip(), "ID number must not be empty"


# step 3: consistency, i.e. agreement rate per field, asserted on RATES (not on single runs), to avoid flaky tests
@for_each_case
def test_consistency(case):
    _, agreement = vote(samples_of(case["input"]))
    print(f"\n{case['input']} agreement: {agreement}")  # visible with -s
    inconsistent = {field: rate for field, rate in agreement.items() if rate < 2 / 3}
    assert not inconsistent, f"fields with less than 2/3 agreement: {inconsistent}"


# ... and does voting improve accuracy w.r.t. single queries? (a measure, rather than a requirement: it fails only if voting makes things WORSE)
def test_voting_vs_single_queries():
    single = sum(accuracy(samples_of(c["input"]), c["expected"]) for c in TEST_CASES) / len(TEST_CASES)
    voted = sum(accuracy([vote(samples_of(c["input"]))[0]], c["expected"]) for c in TEST_CASES) / len(TEST_CASES)
    print(f"\n[{vision_model}] accuracy of single queries: {single:.0%}, of voted ones: {voted:.0%}")
    assert voted >= single


# step 4: LLM-as-a-judge, only for what code cannot check (legibility), with a (different) multimodal model
class Legibility(BaseModel):
    reasoning: str = Field(description="Which parts of the document are hard to read, if any, and why (e.g. stamps, glare, blur).")
    legible: bool = Field(description="Whether name, date of birth, document number, and expiry date are legible enough to trust an automatic extraction.")

# (better a DIFFERENT model than the system under test, set via JUDGE_MODEL: by default, the same is used, as few free models support images)
judge = ChatOpenAI(base_url=base_url, api_key=api_key, model=os.environ.get("JUDGE_MODEL", vision_model), temperature=0)
legibility_judge = judge.with_structured_output(Legibility)

@for_each_case
def test_legibility_judge_agrees_with_humans(case):  # validates the judge against my own judgement (in passports.yml)
    verdict = legibility_judge.invoke([
        SystemMessage("You assess the quality of pictures of ID documents, before any information is extracted from them."),
        message_with_image("Is this document legible enough?", data.DIR / case["input"]),
    ])
    assert verdict.legible == case["expected"]["legible"], verdict.reasoning


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
