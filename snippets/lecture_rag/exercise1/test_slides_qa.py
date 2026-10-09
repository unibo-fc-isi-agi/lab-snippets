"""
Evaluation of the Q/A system (qa.py), against the gold set in gold.yml (pinned to a release of the slides, for reproducibility):
- retrieval: recall@k (how often a right slide is among the first k) and MRR (how high the first right slide is), embeddings only;
- generation: answers are grounded in the retrieved slides (DeepEval's FaithfulnessMetric, i.e. LLM-as-a-judge),
  cite a right slide, and admit when the slides do not cover the question.

Run with: poetry run python -m snippets -l rag -x 1 [PYTEST OPTIONS, e.g. -s to see the metrics, -k retrieval for retrieval only]   (then pick test_slides_qa.py)
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (system under test), JUDGE_MODEL (judge),
EMBEDDINGS_BASE_URL, EMBEDDINGS_API_KEY, EMBEDDINGS_MODEL, RAG_CACHE_DIR.
"""
import os
import sys
from pathlib import Path
import pytest
import yaml
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric
from deepeval.models import OpenRouterModel
from deepeval.test_case import LLMTestCase
from snippets.lecture_prompting.example1bis.letter_scoring_langchain import api_key, base_url
from snippets.lecture_rag.exercise1 import index, qa
from snippets.lecture_rag.exercise1.slides import Slide, release

GOLD = yaml.safe_load((Path(__file__).parent / "gold.yml").read_text())
KS = (1, 3, 5, 10)


# retrieval metrics: a question's rank is the position of the first right slide among the retrieved ones (None if missing)
def rank(retrieved: list[Slide], case: dict) -> int | None:
    for position, slide in enumerate(retrieved, start=1):
        if slide.lecture == case["lecture"] and slide.title in case["titles"]:  # keyed on titles, as pages shift across releases
            return position
    return None

def recall_at_k(ranks: list[int | None], k: int) -> float:
    return sum(r is not None and r <= k for r in ranks) / len(ranks)

def mrr(ranks: list[int | None]) -> float:  # mean reciprocal rank: 1 if always first, 0.5 if always second, ...
    return sum(1 / r for r in ranks if r is not None) / len(ranks)


@pytest.fixture(scope="module")
def db():  # the same database as qa.py, (re-)indexed incrementally to the release of the gold set
    db = index.open_db()
    index.update(db, release(GOLD["tag"]))
    return db


def test_retrieval(db):
    ranks = [rank(index.search(db, case["question"], max(KS)), case) for case in GOLD["questions"]]
    for case, r in zip(GOLD["questions"], ranks):
        print(f"rank {r}: {case['question']}")
    print(*(f"recall@{k} = {recall_at_k(ranks, k):.2f}" for k in KS), f"MRR = {mrr(ranks):.2f}", sep="; ")
    assert recall_at_k(ranks, 5) >= 0.7  # a regression threshold (0.75 with nomic-embed-text): raise it as retrieval improves


judge = OpenRouterModel(model=os.environ.get("JUDGE_MODEL", "google/gemma-4-31b-it:free"), api_key=api_key, base_url=base_url)
faithfulness = FaithfulnessMetric(model=judge, threshold=0.7)  # claims in the answer that the retrieved slides support / all claims

@pytest.mark.parametrize("case", GOLD["questions"], ids=lambda case: case["question"])
def test_answer(db, case):
    answer, retrieved = qa.ask(db, case["question"])
    assert answer.covered
    assert any(rank([slide], case) for slide in qa.cited(answer, retrieved)), "no right slide is cited"
    test_case = LLMTestCase(input=case["question"], actual_output=answer.answer, retrieval_context=[s.text for s in retrieved])
    assert_test(test_case, [faithfulness])


@pytest.mark.parametrize("question", ["What is the capital of Australia?", "Who won the 2006 FIFA World Cup?"])
def test_not_covered(db, question):
    answer, retrieved = qa.ask(db, question)
    assert not answer.covered and not qa.cited(answer, retrieved)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
