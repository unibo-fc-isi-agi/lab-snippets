"""
Evaluates the whole RAG pipeline of Example 4 with LLM-as-a-judge metrics (DeepEval), on a few questions:
- faithfulness: is every claim in the answer supported by the retrieved chunks? (i.e. no hallucinations)
- answer relevancy: does the answer address the question?
- contextual recall: do the retrieved chunks contain everything needed for the expected answer?

Run with: poetry run python -m snippets -l rag -e 5 [PYTEST OPTIONS, e.g. -v] (pick test_generation.py)
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (system under test), JUDGE_MODEL (judge),
plus EMBEDDINGS_BASE_URL, EMBEDDINGS_API_KEY, EMBEDDINGS_MODEL.
With slow (e.g. local) judges, raise DeepEval's timeout, e.g.: DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=1200
"""
import os
import sys
import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, ContextualRecallMetric, FaithfulnessMetric
from deepeval.models import OpenRouterModel
from deepeval.test_case import LLMTestCase
from snippets.lecture_rag.example4.rag_langchain import api_key, base_url, generate, retrieve

TEST_CASES = {  # question -> expected answer (only needed by contextual recall)
    "Which English certificates are accepted, and with which minimum scores?":
        "IELTS Academic with at least 6.0, TOEFL iBT with at least 80, Cambridge B2 First or higher, "
        "or any other certificate stating B2 level or higher; certificates must be obtained no earlier than 1 January 2023.",
    "Who wrote Mario Rossi's recommendation letter?":
        "Prof. Alessandro Bianchi, Associate Professor of Computer Science at the University of Bologna.",
    "How are ties in the final ranking broken?":
        "Priority goes to the candidate with the higher score in the research proposal; if still tied, to the younger candidate.",
}

judge = OpenRouterModel(model=os.environ.get("JUDGE_MODEL", "openai/gpt-oss-120b"), api_key=api_key, base_url=base_url)
metrics = [FaithfulnessMetric(model=judge, threshold=0.7),
           AnswerRelevancyMetric(model=judge, threshold=0.7),
           ContextualRecallMetric(model=judge, threshold=0.7)]


@pytest.mark.parametrize("question", TEST_CASES)
def test_rag(question):
    documents = retrieve(question)
    answer = generate(question, documents)
    assert answer.answerable  # a deterministic check first: all these questions are answerable
    assert_test(LLMTestCase(
        input=question,
        actual_output=answer.answer,
        expected_output=TEST_CASES[question],
        retrieval_context=[d.page_content for d in documents],  # what the answer must be faithful to
    ), metrics)


if __name__ == "__main__":  # let pytest run the tests in this file (plus any option given on the command line)
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
