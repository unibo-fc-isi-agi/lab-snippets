"""
The corpus of this lecture's examples, split into chunks: the candidates' recommendation letters (one chunk per paragraph)
plus the (fictional) regulations of the PhD programme (one chunk per article).

Try it with: poetry run python -m snippets.lecture_rag.corpus
"""
import re
from dataclasses import dataclass
import data


@dataclass
class Chunk:
    id: str                # unique and human-readable, e.g. "letter-mario-rossi#p7" or "regulations-phd#art8"
    text: str              # what gets embedded, retrieved, and shown to the LLM
    source: str            # file it comes from, e.g. "letter-mario-rossi"
    candidate: str | None  # which candidate it is about (None for the regulations), useful as a filter
    section: str           # where it is in the file, e.g. "paragraph 7" or "Article 8 - Evaluation criteria and their weights"


def letter_chunks(candidate: str) -> list[Chunk]:
    path = data.letter(candidate)
    paragraphs = path.read_text().split("\n\n")  # paragraphs are separated by blank lines
    context = f"[From the recommendation letter for {candidate.replace('-', ' ').title()}]\n"  # paragraphs may not name them
    return [Chunk(f"{path.stem}#p{i}", context + text.strip(), path.stem, candidate, f"paragraph {i}")
            for i, text in enumerate(paragraphs, start=1)
            if len(text.split()) >= 10]  # skip dates, greetings, subject, ...


def regulations_chunks() -> list[Chunk]:
    path = data.regulations()
    articles = re.split(r"^## ", path.read_text(), flags=re.MULTILINE)[1:]  # one per "## Article N - Title" heading
    return [Chunk(f"{path.stem}#art{i}", "## " + text.strip(), path.stem, None, text.splitlines()[0])
            for i, text in enumerate(articles, start=1)]


def chunks() -> list[Chunk]:
    """All the chunks of the corpus: letters first, then the regulations."""
    return [chunk for candidate in data.CANDIDATES for chunk in letter_chunks(candidate)] + regulations_chunks()


if __name__ == "__main__":
    for chunk in chunks():
        print(f"{chunk.id:<25} {len(chunk.text.split()):>4} words  {chunk.text[:60]!r}")
