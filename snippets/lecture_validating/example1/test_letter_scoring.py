"""
Tests the letter-scoring system (Example 1 bis of the prompting lecture) with pytest and DeepEval:
deterministic assertions, a relational property, and LLM-as-a-judge metrics.

Run with: poetry run python -m snippets -l validating -e 1 [PYTEST OPTIONS, e.g. -v]
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (system under test), JUDGE_MODEL (judge).
"""
import functools
import os
import sys
import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.models import OpenRouterModel
from deepeval.test_case import LLMTestCase, SingleTurnParams
from snippets.lecture_validating.dataset import TEST_CASES, read_letter
from snippets.lecture_prompting.example1bis.letter_scoring_langchain import score_letter, api_key, base_url

score = functools.cache(score_letter)  # this is the scoring function under test! (cached: one LLM call per letter)
for_each_case = pytest.mark.parametrize("case", TEST_CASES, ids=lambda c: c["expectations"]["applicant_name"])


# 1. deterministic scorers: plain assertions on some fields of the structured output
@for_each_case
def test_names_and_email(case):  # exact match
    info, expected = score(read_letter(case)), case["expectations"]
    assert info.applicant.name == expected["applicant_name"]
    assert info.author.name == expected["author_name"]
    assert info.author.email == expected["author_email"]

@for_each_case
def test_score_in_range(case):  # tolerant match
    info, expected = score(read_letter(case)), case["expectations"]
    assert expected.get("min_score", 0) <= info.score <= expected.get("max_score", 5)


# 2. relational property: holds across inputs, even if single scores may vary
def test_best_letter_is_mario_rossi():
    scores = {c["expectations"]["applicant_name"]: score(read_letter(c)).score for c in TEST_CASES}
    assert scores["Mario Rossi"] == max(scores.values())


# 3. LLM-as-a-judge: a (different) model grades what is hard to check with code
judge = OpenRouterModel(model=os.environ.get("JUDGE_MODEL", "google/gemma-4-31b-it:free"), api_key=api_key, base_url=base_url)

relationship = GEval(  # reference-based: the judge compares the actual output with the expected one
    name="Relationship",
    criteria="The actual output describes the relationship between the author of a recommendation letter and the applicant. "
             "It must be consistent with the expected output: penalise contradictions, omitted key facts, and invented facts.",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    model=judge,
    threshold=0.7,  # GEval scores are in [0, 1]: below the threshold, the test fails
)

groundedness = GEval(  # reference-free: the judge compares the actual output with the input letter
    name="Groundedness",
    criteria="Every skill, strength, and weakness listed in the actual output must be explicitly supported by the letter in the input. "
             "Penalise any invented, exaggerated, or misattributed item.",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    model=judge,
    threshold=0.7,
)

@for_each_case
def test_relationship(case):
    letter = read_letter(case)
    actual, expected = score(letter).author.relationship_with_applicant, case["expectations"]["relationship_with_applicant"]
    assert_test(LLMTestCase(input=letter, actual_output=str(actual), expected_output=expected), [relationship])

@for_each_case
def test_skills_strengths_weaknesses(case):
    letter = read_letter(case)
    actual = score(letter).applicant.model_dump_json(include={"skills", "strengths", "weaknesses"})
    assert_test(LLMTestCase(input=letter, actual_output=actual), [groundedness])


if __name__ == "__main__":  # let pytest run the tests in this file (plus any option given on the command line)
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
