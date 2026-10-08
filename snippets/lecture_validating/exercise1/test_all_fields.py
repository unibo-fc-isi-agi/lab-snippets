"""
Completes the test suite of Example 1 (pytest + DeepEval), so that EVERY field of LetterInfo is checked:
the tests of Example 1 are re-used as they are, and new ones check the remaining fields, against the expectations in test_data.yml.

Run with: poetry run python -m snippets -l validating -x 1 [PYTEST OPTIONS, e.g. -v]
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (system under test), JUDGE_MODEL (judge).
"""
import string
import sys
import unicodedata
from pathlib import Path
import pytest
import yaml
from snippets.lecture_validating.dataset import read_letter
# the tests of Example 1 (names, email, score, relationship, skills/strengths/weaknesses): imported, hence collected and run here too.
# `score` is the same cached scoring function, so each letter is still scored only ONCE, across old and new tests
from snippets.lecture_validating.example1.test_letter_scoring import (
    score, test_names_and_email, test_score_in_range, test_best_letter_is_mario_rossi, test_relationship, test_skills_strengths_weaknesses,
)

# step 3: the new expectations (same inputs as Example 1's test data)
TEST_CASES = yaml.safe_load((Path(__file__).parent / "test_data.yml").read_text())
for_each_case = pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c["input"])


# step 2: tolerant matching, i.e. normalisation + containment (cf. the conventions in test_data.yml)
def normalise(text: str | None) -> str:
    """e.g. 'Maîtresse de Conférences.' -> 'maitresse de conferences'"""
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()  # strips accents
    return " ".join(text.lower().translate(str.maketrans(string.punctuation, " " * len(string.punctuation))).split())

def matches(actual: str | None, expected: str | list | None) -> bool:
    """Whether the actual value matches the expected one (or any of them, if a list), as per the conventions in test_data.yml."""
    if isinstance(expected, list):
        return any(matches(actual, e) for e in expected)
    if expected is None:  # the letter does not say it: nothing (or an explicit "unknown", which some models write) is expected
        return not actual or normalise(actual) in {"none", "null", "unknown", "not specified", "not mentioned", "n a"}
    return actual is not None and normalise(expected) in normalise(actual)


# step 4: one test per group of fields, with the most appropriate kind of check
@for_each_case
@pytest.mark.parametrize("field", ["degree", "alma_mater"])
def test_applicant_studies(case, field):  # tolerant match
    actual = getattr(score(read_letter(case)).applicant, field)
    assert matches(actual, case["expectations"][field]), f"{field}: {actual!r}"

@for_each_case
def test_attended_courses(case):  # set inclusion: every expected course is among the extracted ones (order, and extra ones, do not matter)
    actual = score(read_letter(case)).applicant.attended
    missing = [course for course in case["expectations"]["attended"] if not any(matches(a, course) for a in actual)]
    assert not missing, f"missing courses: {missing}, extracted: {actual}"

@for_each_case
@pytest.mark.parametrize("field", ["affiliation", "position", "nationality"])
def test_author_info(case, field):  # tolerant match
    actual = getattr(score(read_letter(case)).author, field)
    assert matches(actual, case["expectations"][field]), f"{field}: {actual!r}"

@for_each_case
def test_author_seniority(case):  # exact match, among admissible values (as the scale is ambiguous for some positions)
    actual = score(read_letter(case)).author.seniority
    assert actual in case["expectations"]["seniority"]

@for_each_case
def test_application_for(case):  # tolerant match
    actual = score(read_letter(case)).application_for
    assert matches(actual, case["expectations"]["application_for"]), f"application_for: {actual!r}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
