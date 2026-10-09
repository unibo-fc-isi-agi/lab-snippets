"""
Evaluates the letter-scoring system (Example 1 bis of the prompting lecture) with MLflow:
deterministic scorers, plus LLM-as-a-judge ones, with results tracked as an MLflow run.

Run with: poetry run python -m snippets -l validating -e 1bis
then browse the results with: poetry run mlflow ui
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (system under test), JUDGE_MODEL (judge).
"""
import os
import sys
from datetime import datetime
from typing import Literal
import mlflow
from mlflow.genai import scorer
from mlflow.genai.judges import make_judge
from snippets.lecture_validating.dataset import TEST_CASES, read_letter
from snippets.lecture_prompting.example1bis.letter_scoring_langchain import score_letter, base_url

# 1. the dataset: "inputs" are passed to predict_fn, "expectations" to the scorers
data = [dict(inputs=dict(letter_text=read_letter(c)), expectations=c["expectations"]) for c in TEST_CASES]

# 2. the system under test (outputs must be serialisable)
def predict_fn(letter_text: str) -> dict:
    return score_letter(letter_text).model_dump()

# 3. deterministic scorers, on some fields of the structured output
@scorer
def names_and_email(outputs, expectations) -> bool:  # exact match
    return (outputs["applicant"]["name"] == expectations["applicant_name"]
            and outputs["author"]["name"] == expectations["author_name"]
            and outputs["author"]["email"] == expectations["author_email"])

@scorer
def score_in_range(outputs, expectations) -> bool:  # tolerant match
    return expectations.get("min_score", 0) <= outputs["score"] <= expectations.get("max_score", 5)

# 4. LLM-as-a-judge (the API key is read from OPENAI_API_KEY)
judge_options = dict(model=f"openai:/{os.environ.get('JUDGE_MODEL', 'google/gemma-4-31b-it:free')}", base_url=base_url,
                     feedback_value_type=Literal["yes", "no"], generate_rationale_first=True)

relationship = make_judge(  # reference-based: the judge compares outputs with expectations
    name="relationship",
    instructions="The {{ outputs }} contain the information extracted from a recommendation letter, and the {{ expectations }} what a human expects. "
                 "Answer 'yes' if the author's relationship_with_applicant in the outputs is consistent with the relationship_with_applicant "
                 "in the expectations (no contradictions, no omitted key facts, no invented facts), 'no' otherwise.",
    **judge_options,
)

groundedness = make_judge(  # reference-free: the judge compares outputs with inputs
    name="groundedness",
    instructions="The {{ inputs }} contain a recommendation letter, and the {{ outputs }} the information extracted from it. "
                 "Answer 'yes' if every skill, strength, and weakness of the applicant in the outputs is explicitly supported by the letter, 'no' otherwise.",
    **judge_options,
)


if __name__ == "__main__":
    mlflow.set_experiment("letter-scoring")  # runs are grouped into experiments
    with mlflow.start_run(run_name=datetime.now().isoformat(timespec="seconds")):  # e.g. "2026-10-01T17:36:26"
        results = mlflow.genai.evaluate(data=data, predict_fn=predict_fn, scorers=[names_and_email, score_in_range, relationship, groundedness])
    print(results.metrics)  # fraction of test cases passing each scorer, e.g. {'score_in_range/mean': 0.67, ...}
    failed = [name for name, value in results.metrics.items() if value != 1]  # also catches NaN (scorer errors)
    if failed:
        sys.exit(f"Failed scorers: {failed}")  # non-zero exit code, as a failing test suite (e.g. to make CI fail)
