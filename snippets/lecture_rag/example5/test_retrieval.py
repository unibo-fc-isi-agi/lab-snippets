"""
Evaluates the retrieval step alone, against a small "gold set" of questions, each with the chunk(s) that answer it:
recall@k (are the right chunks in the top k?) and MRR (how high is the first right chunk?), for vector and hybrid search.
Deterministic, given the embedding model: no LLM involved.

Run with: poetry run python -m snippets -l rag -e 5 [PYTEST OPTIONS, e.g. -v -s] (pick test_retrieval.py)
Configure via env vars: EMBEDDINGS_BASE_URL, EMBEDDINGS_API_KEY, EMBEDDINGS_MODEL.
"""
import functools
import sys
import pytest
from snippets.lecture_rag.example2bis.vector_store_sqlite_vec import search
from snippets.lecture_rag.example3.hybrid_search import create_index, hybrid_search
from snippets.lecture_rag.vec import connect

GOLD_SET = {  # question -> IDs of the chunks that answer it
    "Is there an age limit to apply?": {"regulations-phd#art3"},
    "What is form PHD-07?": {"regulations-phd#art13"},
    "What happens if two candidates get the same total score?": {"regulations-phd#art12"},
    "How much is a scholarship worth per year?": {"regulations-phd#art11"},
    "Which English certificates are accepted?": {"regulations-phd#art5"},
    "When is the application deadline?": {"regulations-phd#art6"},
    "Can the committee use AI tools to evaluate applications?": {"regulations-phd#art14"},
    "How can a candidate appeal against the ranking?": {"regulations-phd#art15"},
    "What did Mario Rossi's thesis deal with?": {"letter-mario-rossi#p7"},
    "What are Jean Dupont's weaknesses?": {"letter-jean-dupont#p11"},
}
K = 3


@functools.cache  # indexing happens once, in memory
def index():
    conn = connect()
    create_index(conn)  # both the vector and the full-text indexes, as in Example 3
    return conn


RETRIEVERS = {  # name -> function from (question, k) to the IDs of the retrieved chunks, best first
    "vectors": lambda question, k: [id for _, id, _ in search(index(), question, k)],
    "hybrid": lambda question, k: hybrid_search(index(), question, k)[2],
}


def recall_at_k(retrieved: list[str], relevant: set[str]) -> float:  # fraction of the relevant chunks that were retrieved
    return len(relevant.intersection(retrieved)) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:  # 1 if the 1st is relevant, 1/2 if the 2nd, ..., 0 if none
    return next((1 / rank for rank, id in enumerate(retrieved, start=1) if id in relevant), 0)


@pytest.mark.parametrize("name", RETRIEVERS)
def test_retrieval_quality(name):
    retrieve = RETRIEVERS[name]
    results = {question: retrieve(question, K) for question in GOLD_SET}
    recall = sum(recall_at_k(results[q], GOLD_SET[q]) for q in GOLD_SET) / len(GOLD_SET)
    mrr = sum(reciprocal_rank(results[q], GOLD_SET[q]) for q in GOLD_SET) / len(GOLD_SET)
    print(f"\n{name}: recall@{K} = {recall:.2f}, MRR = {mrr:.2f}")
    for question, retrieved in results.items():
        print(f"    {'OK  ' if GOLD_SET[question] & set(retrieved) else 'MISS'} {question} -> {retrieved}")
    assert recall >= 0.7  # thresholds: regressions (e.g. a worse chunking or embedding model) make the test fail
    assert mrr >= 0.6


if __name__ == "__main__":  # let pytest run the tests in this file (plus any option given on the command line)
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
