"""
Objective facts about a thesis (PDF), for the pre-check: outline, unresolved references, captions never referenced,
repeated words, near-empty pages, bibliography. Prints a JSON report; with --split DIR, also writes the text of each
top-level section to DIR/NN-<title>.txt, so that it can be read section by section.

Usage: python scripts/pdf_facts.py THESIS.pdf [--split DIR]      (requires: pip install pypdf)
"""
import json
import re
import sys
from pathlib import Path

MAX_ITEMS = 20  # per kind of issue: enough to spot patterns, small enough for the context
UNRESOLVED = re.compile(r"\?\?|\[\?\]|\(\?\)")  # LaTeX's undefined \ref, \cite (numeric or author-year)
CAPTION = re.compile(r"^(Figure|Table|Listing)\s+(\d+(?:\.\d+)*)\s*:", re.MULTILINE)  # LaTeX default: "Figure 2.1: ..."
REFERENCE = re.compile(r"\b(fig|tab|lis)(?:ure|le|ting|\.)\s*(\d+(?:\.\d+)*)", re.IGNORECASE)  # e.g. "Figure 2.1", "fig. 2.1"
KINDS = {"fig": "Figure", "tab": "Table", "lis": "Listing"}
REPEATED = re.compile(r"\b([A-Za-z]{3,}) \1\b", re.IGNORECASE)  # same line only: line breaks are often in code
BIBLIOGRAPHY = re.compile(r"^\s*(Bibliography|References|Bibliografia)\s*$", re.MULTILINE | re.IGNORECASE)
ENTRY = re.compile(r"^\s*\[[^\]\n]{1,20}\]\s+\S", re.MULTILINE)  # e.g. "[12] ..." or "[YL99] ..."


def analyze(pages: list[str], outline: list[tuple[str, int]]) -> dict:
    """The facts, given the text of each page and the top-level outline as (title, first page index) pairs."""
    def where(pattern):  # (page number, matched text) pairs, page numbers starting from 1
        return [(i + 1, m.group(0)) for i, page in enumerate(pages) for m in pattern.finditer(page)]

    captions = {(kind, number): page for page, kind, number in
                ((i + 1, m[1], m[2]) for i, page in enumerate(pages) for m in CAPTION.finditer(page))}
    mentions = [(KINDS[m[1].lower()], m[2])
                for page in pages for m in REFERENCE.finditer(page)]
    unreferenced = [f"{kind} {number} (page {page})" for (kind, number), page in captions.items()
                    if mentions.count((kind, number)) <= 1]  # i.e. mentioned in its own caption only
    bibliography = next((i for i, page in enumerate(pages) if BIBLIOGRAPHY.search(page)), None)
    return {
        "pages": len(pages),
        "words": sum(len(page.split()) for page in pages),
        "outline": [{"title": title, "page": index + 1} for title, index in outline],
        "unresolved_references": where(UNRESOLVED)[:MAX_ITEMS],
        "captions": len(captions),
        "captions_never_referenced": unreferenced[:MAX_ITEMS],
        "repeated_words": [(page, text) for page, text in where(REPEATED)
                           if text.split()[0].lower() not in {"yes", "no", "that", "had"}][:MAX_ITEMS],
        "near_empty_pages": [i + 1 for i, page in enumerate(pages) if len(page.split()) < 30][:MAX_ITEMS],
        "bibliography": None if bibliography is None else {
            "page": bibliography + 1,
            "entries": sum(len(ENTRY.findall(page)) for page in pages[bibliography:]),
        },
    }


def extract(pdf: Path) -> tuple[list[str], list[tuple[str, int]]]:
    from pypdf import PdfReader  # imported here: the analysis above needs no third-party library
    reader = PdfReader(pdf)
    pages = [page.extract_text() or "" for page in reader.pages]
    outline = [(item.title, reader.get_destination_page_number(item))
               for item in reader.outline if not isinstance(item, list)]  # nested lists are sub-sections
    return pages, outline


def split(pages: list[str], outline: list[tuple[str, int]], folder: Path) -> list[str]:
    """Writes the text of each top-level section to a file; returns the files' paths."""
    folder.mkdir(parents=True, exist_ok=True)
    starts = [index for _, index in outline] + [len(pages)]
    files = []
    for n, ((title, start), end) in enumerate(zip(outline, starts[1:]), start=1):
        file = folder / f"{n:02d}-{re.sub(r'[^A-Za-z0-9]+', '-', title).strip('-').lower()[:40]}.txt"
        file.write_text("\n".join(f"[page {i + 1}]\n{pages[i]}" for i in range(start, max(start + 1, end))))
        files.append(str(file))
    return files


if __name__ == "__main__":
    if len(sys.argv) < 2 or not Path(sys.argv[1]).is_file():
        sys.exit("Usage: python scripts/pdf_facts.py THESIS.pdf [--split DIR]")
    pages, outline = extract(Path(sys.argv[1]))
    facts = analyze(pages, outline)
    if "--split" in sys.argv:
        facts["sections"] = split(pages, outline, Path(sys.argv[sys.argv.index("--split") + 1]))
    print(json.dumps(facts, indent=1, ensure_ascii=False))
