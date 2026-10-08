"""
Hybrid search in a single SQLite file: keyword search (BM25, via the built-in FTS5 extension) plus vector search
(via sqlite-vec, as in Example 2 bis), fused via Reciprocal Rank Fusion (RRF). The three rankings are shown side by side.
Keywords win on exact codes and names, vectors win on paraphrases, RRF gets (most of) the best of both, e.g. try:

    poetry run python -m snippets -l rag -e 3 "What is form PHD-07?"
    # keyword wins: Article 13 (the only one mentioning PHD-07) is 1st for BM25 and RRF, while vectors prefer Article 7
    poetry run python -m snippets -l rag -e 3 "What happens if two candidates get the same total score?"
    # semantic wins: Article 12 ("Ranking and ties") is 1st for vectors and RRF, while BM25 prefers Article 5 (scores)

Run with: poetry run python -m snippets -l rag -e 3 "QUESTION" [--k 5] [--reindex]
Configure via env vars: EMBEDDINGS_BASE_URL, EMBEDDINGS_API_KEY, EMBEDDINGS_MODEL.
"""
import re
import sqlite3
from argparse import ArgumentParser
from pathlib import Path
from snippets.lecture_rag.corpus import chunks
from snippets.lecture_rag.example2bis.vector_store_sqlite_vec import create_index as create_vector_index, search as vector_search
from snippets.lecture_rag.vec import connect

DB_FILE = Path("output/rag-example3.db")


def create_index(conn: sqlite3.Connection) -> None:
    create_vector_index(conn)  # the vec0 table "chunks", as in Example 2 bis
    conn.execute("DROP TABLE IF EXISTS chunks_fts")
    conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(id UNINDEXED, text)")  # full-text index of the same chunks
    conn.executemany("INSERT INTO chunks_fts(id, text) VALUES (?, ?)", [(c.id, c.text) for c in chunks()])
    conn.commit()


def keyword_search(conn: sqlite3.Connection, query: str, k: int) -> list[str]:
    """IDs of the k chunks best matching the query's words, according to BM25."""
    words = re.findall(r"\w+", query)  # e.g. "form PHD-07" -> ["form", "PHD", "07"]
    fts_query = " OR ".join(f'"{word}"' for word in words)  # any word may match; quoted, so no FTS5 syntax errors
    rows = conn.execute("SELECT id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?", (fts_query, k))
    return [id for (id,) in rows]  # bm25() is lower for better matches, hence ascending order


def rrf(*rankings: list[str], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion: each ranking gives 1 / (k + rank) to each of its items; items are sorted by their total."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, id in enumerate(ranking, start=1):
            scores[id] = scores.get(id, 0) + 1 / (k + rank)  # only ranks matter, not the (incomparable) scores
    return sorted(scores, key=scores.get, reverse=True)


def hybrid_search(conn: sqlite3.Connection, query: str, k: int = 5) -> tuple[list[str], list[str], list[str]]:
    """IDs of the top-k chunks for the query according to BM25, to vectors, and to their fusion."""
    by_keywords = keyword_search(conn, query, 2 * k)  # fusion works better with a few more candidates per ranking
    by_vectors = [id for _, id, _ in vector_search(conn, query, 2 * k)]
    return by_keywords[:k], by_vectors[:k], rrf(by_keywords, by_vectors)[:k]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--reindex", action="store_true")
    args = parser.parse_args()

    DB_FILE.parent.mkdir(exist_ok=True)
    reindex = args.reindex or not DB_FILE.exists()
    with connect(DB_FILE) as conn:
        if reindex:
            create_index(conn)
            print(f"# Indexed {len(chunks())} chunks into {DB_FILE}")
        rankings = hybrid_search(conn, args.query, args.k)
        print(f"{'#':<3}" + "".join(f"{name:<26}" for name in ("BM25 (keywords)", "Vectors (semantics)", "RRF (hybrid)")))
        for rank, ids in enumerate(zip(*rankings), start=1):
            print(f"{rank:<3}" + "".join(f"{id:<26}" for id in ids))
