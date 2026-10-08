"""
The vector store: one row per slide (its embedding, plus text and metadata) in a SQLite file, via sqlite-vec,
and the digest of each indexed PDF, so that only the lectures whose PDF changed are re-indexed (embedding costs time, or money).

Run with: poetry run python -m snippets -l rag -x 1 [--tag TAG]   (then pick index.py: (re-)indexes the slides, if needed)
Configure via env vars: EMBEDDINGS_BASE_URL, EMBEDDINGS_API_KEY, EMBEDDINGS_MODEL, RAG_CACHE_DIR (default: .rag-cache/).
"""
import re
import sqlite3
from snippets.lecture_rag import embeddings, vec
from snippets.lecture_rag.exercise1.slides import CACHE_DIR, Pdf, Slide, download, slides

# vectors of different models are NOT comparable (and may differ in size): hence, one database per embedding model
DB_PATH = CACHE_DIR / ("slides-" + re.sub(r"\W", "_", embeddings.model) + ".db")  # e.g. .rag-cache/slides-nomic_embed_text.db


def open_db(path=DB_PATH) -> sqlite3.Connection:
    if path != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    db = vec.connect(path)
    db.execute("CREATE TABLE IF NOT EXISTS pdfs (lecture TEXT PRIMARY KEY, tag TEXT, digest TEXT)")  # what is indexed
    return db


def create_slides_table(db: sqlite3.Connection, dimensions: int) -> None:  # the size of vectors is only known after embedding
    db.execute(f"""CREATE VIRTUAL TABLE IF NOT EXISTS slides USING vec0(
        embedding float[{dimensions}] distance_metric=cosine,  -- the usual (dis)similarity for text embeddings
        lecture text,  -- metadata columns: can be used in the WHERE clause of KNN queries, e.g. to filter by lecture
        page integer,
        +title text,   -- auxiliary columns ('+'): just stored, and returned along with the results
        +text text,
        +tag text
    )""")


def needs_reindex(db: sqlite3.Connection, pdf: Pdf) -> bool:
    row = db.execute("SELECT digest FROM pdfs WHERE lecture = ?", (pdf.lecture,)).fetchone()
    return row is None or row[0] != pdf.digest  # new lecture, or changed PDF


def index(db: sqlite3.Connection, pdf: Pdf, chunks: list[Slide]) -> None:
    vectors = embeddings.embed([slide.text for slide in chunks])  # one (batch) request per lecture
    create_slides_table(db, len(vectors[0]))
    with db:  # one transaction: a lecture is either fully re-indexed, or not at all (e.g. upon errors, or Ctrl+C)
        db.execute("DELETE FROM slides WHERE lecture = ?", (pdf.lecture,))
        db.executemany("INSERT INTO slides(embedding, lecture, page, title, text, tag) VALUES (?, ?, ?, ?, ?, ?)",
                       [(vec.serialize(vector), s.lecture, s.page, s.title, s.text, s.tag) for s, vector in zip(chunks, vectors)])
        db.execute("INSERT OR REPLACE INTO pdfs VALUES (?, ?, ?)", (pdf.lecture, pdf.tag, pdf.digest))


def update(db: sqlite3.Connection, pdfs: list[Pdf]) -> None:
    """Incremental indexing: (re-)indexes the lectures whose PDF is new or changed, forgets the ones no longer released."""
    for pdf in pdfs:
        if needs_reindex(db, pdf):
            chunks = slides(pdf, download(pdf))
            index(db, pdf, chunks)
            print(f"# Indexed {len(chunks)} slides of {pdf.lecture} ({pdf.tag})")
    with db:
        lectures = [pdf.lecture for pdf in pdfs]
        for (gone,) in db.execute(f"SELECT lecture FROM pdfs WHERE lecture NOT IN ({','.join('?' * len(lectures))})", lectures).fetchall():
            db.execute("DELETE FROM slides WHERE lecture = ?", (gone,))
            db.execute("DELETE FROM pdfs WHERE lecture = ?", (gone,))


def search(db: sqlite3.Connection, query: str, k: int = 5, lecture: str | None = None) -> list[Slide]:
    """The k slides most similar to the query (optionally, within a lecture), most similar first."""
    sql, params = "SELECT lecture, page, title, text, tag FROM slides WHERE embedding MATCH ? AND k = ?", [vec.serialize(embeddings.embed([query])[0]), k]
    if lecture:  # filtering happens DURING the KNN search, so k slides are returned anyway (if the lecture has as many)
        sql, params = sql + " AND lecture = ?", params + [lecture]
    return [Slide(*row) for row in db.execute(sql + " ORDER BY distance", params)]


if __name__ == "__main__":
    import argparse
    import time
    from snippets.lecture_rag.exercise1.slides import release
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="latest", help="release of the slides, e.g. 2026.10.08 (default: latest)")
    start, db = time.time(), open_db()
    update(db, release(parser.parse_args().tag))
    count = db.execute("SELECT COUNT(*) FROM slides").fetchone()[0]
    print(f"# {count} slides in {DB_PATH}, up to date after {time.time() - start:.1f}s")
