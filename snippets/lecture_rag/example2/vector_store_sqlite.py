"""
A vector store from scratch, with Python's built-in sqlite3: chunks are stored along with their embeddings (as BLOBs),
and a query is answered by comparing its embedding with ALL the stored ones (brute force), keeping the top-k.
The index is persisted to output/rag-example2.db, and built only once (unless --reindex is given).

Run with: poetry run python -m snippets -l rag -e 2 "QUESTION" [--k 3] [--candidate mario-rossi] [--reindex]
e.g.: poetry run python -m snippets -l rag -e 2 "Is there an age limit to apply?"
Configure via env vars: EMBEDDINGS_BASE_URL, EMBEDDINGS_API_KEY, EMBEDDINGS_MODEL.
"""
import sqlite3
from array import array
from argparse import ArgumentParser
from pathlib import Path
from snippets.lecture_rag.corpus import chunks
from snippets.lecture_rag.embeddings import embed
from snippets.lecture_rag.example1.similarity import cosine

DB_FILE = Path("output/rag-example2.db")


def to_blob(vector: list[float]) -> bytes:  # e.g. 768 floats -> 768 * 4 bytes
    return array("f", vector).tobytes()


def from_blob(blob: bytes) -> list[float]:
    return array("f", blob).tolist()


def create_index(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute("""CREATE TABLE chunks (
        id TEXT PRIMARY KEY, source TEXT, candidate TEXT, section TEXT, text TEXT,
        embedding BLOB)""")  # a plain column: SQLite knows nothing about vectors
    all_chunks = chunks()
    vectors = embed([chunk.text for chunk in all_chunks])  # embedding is done once, at indexing time
    conn.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)", [
        (c.id, c.source, c.candidate, c.section, c.text, to_blob(v)) for c, v in zip(all_chunks, vectors)
    ])
    conn.commit()


def search(conn: sqlite3.Connection, query: str, k: int = 3, candidate: str | None = None) -> list[tuple[float, str, str]]:
    """The k chunks most similar to the query, as (similarity, id, text) triples, optionally about one candidate only."""
    [query_vector] = embed([query])  # the query is embedded with the same model as the chunks
    rows = conn.execute("SELECT id, text, embedding FROM chunks WHERE ? IS NULL OR candidate = ?", (candidate, candidate))
    scored = [(cosine(query_vector, from_blob(blob)), id, text) for id, text, blob in rows]  # O(N): fine for small N
    return sorted(scored, reverse=True)[:k]  # the most similar first


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--candidate", help="e.g. mario-rossi")
    parser.add_argument("--reindex", action="store_true")
    args = parser.parse_args()

    DB_FILE.parent.mkdir(exist_ok=True)
    reindex = args.reindex or not DB_FILE.exists()
    with sqlite3.connect(DB_FILE) as conn:
        if reindex:
            create_index(conn)
            print(f"# Indexed {conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]} chunks into {DB_FILE}")
        for similarity, id, text in search(conn, args.query, args.k, args.candidate):
            print(f"{similarity:.3f}  {id:<25} {' '.join(text.split())[:100]}...")
