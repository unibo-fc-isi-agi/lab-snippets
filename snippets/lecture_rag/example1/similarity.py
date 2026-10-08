"""
Embeddings and similarity: a few sentences are turned into vectors, then compared pairwise via cosine similarity.
Paraphrases score high even with no words in common, unrelated sentences score low.

Run with: poetry run python -m snippets -l rag -e 1 ["another sentence" ...]
Configure via env vars: EMBEDDINGS_BASE_URL, EMBEDDINGS_API_KEY, EMBEDDINGS_MODEL.
"""
import math
import sys
from snippets.lecture_rag.embeddings import embed, model

sentences = [
    "The candidate must certify an English level of at least B2.",                # 1
    "Applicants need an upper-intermediate knowledge of the English language.",   # 2: paraphrase of 1
    "Mario Rossi wrote his thesis on predictive analytics for web data.",         # 3
    "Mr. Rossi's dissertation dealt with forecasting models on online datasets.", # 4: paraphrase of 3
    "The recipe calls for two eggs and a cup of flour.",                          # 5: unrelated
    "It will probably rain in Bologna tomorrow.",                                 # 6: unrelated
] + sys.argv[1:]


def cosine(u: list[float], v: list[float]) -> float:  # 1 = same direction, 0 = orthogonal, -1 = opposite
    dot = sum(a * b for a, b in zip(u, v))
    return dot / (math.sqrt(sum(a * a for a in u)) * math.sqrt(sum(b * b for b in v)))


if __name__ == "__main__":
    vectors = embed(sentences)  # one vector per sentence, all with the same dimension
    print(f"Model {model}: {len(vectors)} vectors of dimension {len(vectors[0])}, e.g. {[round(x, 3) for x in vectors[0][:4]]}...\n")
    for i, sentence in enumerate(sentences, start=1):
        print(f"{i}) {sentence}")
    print("\n    " + "".join(f"{j:>6}" for j in range(1, len(sentences) + 1)))
    for i, u in enumerate(vectors, start=1):
        print(f"{i:>4}" + "".join(f"{cosine(u, v):6.2f}" for v in vectors))
