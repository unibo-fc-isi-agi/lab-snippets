"""
The same vector store of Example 2, with the sqlite-vec extension: a `vec0` virtual table stores the embeddings,
and runs the k-nearest-neighbours (KNN) search in SQL, filtering by metadata too.
The index is persisted to output/rag-example2bis.db, and built only once (unless --reindex is given).

Run with: poetry run python -m snippets -l rag -e 2bis "QUESTION" [--k 3] [--candidate mario-rossi] [--reindex]
e.g.: poetry run python -m snippets -l rag -e 2bis "Is there an age limit to apply?"
Configure via env vars: EMBEDDINGS_BASE_URL, EMBEDDINGS_API_KEY, EMBEDDINGS_MODEL.
"""
import sqlite3
from argparse import ArgumentParser
from pathlib import Path
from snippets.lecture_rag.corpus import chunks
from snippets.lecture_rag.embeddings import embed
from snippets.lecture_rag.vec import connect, serialize

DB_FILE = Path("output/rag-example2bis.db")


def create_index(conn: sqlite3.Connection) -> None:
    all_chunks = chunks()
    vectors = embed([chunk.text for chunk in all_chunks])
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute(f"""CREATE VIRTUAL TABLE chunks USING vec0(
        id TEXT PRIMARY KEY,
        embedding float[{len(vectors[0])}] distance_metric=cosine,
        candidate TEXT,
        +source TEXT, +section TEXT, +text TEXT
    )""")  # candidate: metadata column (usable in KNN filters); +...: auxiliary columns (just stored)
    conn.executemany("INSERT INTO chunks(id, embedding, candidate, source, section, text) VALUES (?, ?, ?, ?, ?, ?)", [
        (c.id, serialize(v), c.candidate or "", c.source, c.section, c.text)  # metadata cannot be NULL
        for c, v in zip(all_chunks, vectors)
    ])
    conn.commit()


def search(conn: sqlite3.Connection, query: str, k: int = 3, candidate: str | None = None) -> list[tuple[float, str, str]]:
    """The k chunks most similar to the query, as (similarity, id, text) triples, optionally about one candidate only."""
    [query_vector] = embed([query])
    sql = "SELECT distance, id, text FROM chunks WHERE embedding MATCH ? AND k = ?"  # KNN: the k nearest, sorted
    params = [serialize(query_vector), k]
    if candidate:
        sql += " AND candidate = ?"  # filtered during the search, so we still get k results
        params.append(candidate)
    return [(1 - distance, id, text) for distance, id, text in conn.execute(sql, params)]  # cosine distance = 1 - similarity


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--candidate", help="e.g. mario-rossi")
    parser.add_argument("--reindex", action="store_true")
    args = parser.parse_args()

    DB_FILE.parent.mkdir(exist_ok=True)
    reindex = args.reindex or not DB_FILE.exists()
    with connect(DB_FILE) as conn:
        if reindex:
            create_index(conn)
            print(f"# Indexed {conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]} chunks into {DB_FILE}")
        for similarity, id, text in search(conn, args.query, args.k, args.candidate):
            print(f"{similarity:.3f}  {id:<25} {' '.join(text.split())[:100]}...")
