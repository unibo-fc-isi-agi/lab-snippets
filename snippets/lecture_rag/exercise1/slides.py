"""
The corpus: the PDFs of this course's slides, as attached to the releases of its GitHub repository (one <lecture>_slides.pdf per lecture).
PDFs are downloaded once (and again only if they change, as told by their SHA-256 digests), then split into one chunk per slide (= PDF page).

Run with: poetry run python -m snippets -l rag -x 1 [--tag TAG]   (then pick slides.py: prints the chunks of the slides, without indexing them)
Configure via env vars: RAG_CACHE_DIR (default: .rag-cache/).
"""
import hashlib
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from pypdf import PdfReader

REPO = "unibo-fc-isi-agi/slides-module2"
CACHE_DIR = Path(os.environ.get("RAG_CACHE_DIR", ".rag-cache"))


@dataclass(frozen=True)
class Pdf:
    lecture: str  # e.g. 'agents', from the asset's name 'agents_slides.pdf'
    tag: str      # the release, e.g. '2026.10.08'
    digest: str   # e.g. 'sha256:9405b0...': it changes iff the file changes


@dataclass(frozen=True)
class Slide:  # the chunk: one slide = one page of a PDF
    lecture: str
    page: int
    title: str
    text: str
    tag: str

    @property
    def url(self) -> str:  # a link to the very page, as a citation for humans
        return f"{pdf_url(self.lecture, self.tag)}#page={self.page}"


def pdf_url(lecture: str, tag: str) -> str:
    return f"https://github.com/{REPO}/releases/download/{tag}/{lecture}_slides.pdf"


# 1. which PDFs are in a release? (GitHub's REST API, no authentication needed for public repos, 60 requests/hour)
def release(tag: str = "latest") -> list[Pdf]:
    url = f"https://api.github.com/repos/{REPO}/releases/" + ("latest" if tag == "latest" else f"tags/{tag}")
    with urllib.request.urlopen(url) as response:
        info = json.load(response)
    return [Pdf(asset["name"].removesuffix("_slides.pdf"), info["tag_name"], asset["digest"])
            for asset in info["assets"] if asset["name"].endswith("_slides.pdf")]


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


# 2. download a PDF into the cache, unless it is already there (and intact)
def download(pdf: Pdf) -> Path:
    path = CACHE_DIR / "slides" / pdf.tag / f"{pdf.lecture}_slides.pdf"
    if path.exists() and sha256(path) == pdf.digest:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial, _ = urllib.request.urlretrieve(pdf_url(pdf.lecture, pdf.tag), path.with_suffix(".part"))
    if sha256(Path(partial)) != pdf.digest:  # truncated (or tampered with) download: never index it
        raise IOError(f"Digest mismatch for {pdf.lecture}: expected {pdf.digest}")
    return Path(partial).replace(path)  # atomic: the cache never holds half a file


# 3. from pages to chunks: drop the boilerplate repeated on every page, which would make all slides look alike to embeddings
FOOTER = re.compile(r'G\. Ciatto\s*[—–-]+\s*"Intelligent Agents[^"]*"')  # e.g. G. Ciatto — "Intelligent Agents — Module 2 @ LM-ISI"
PAGE_NUMBER = re.compile(r"\s*\d+\s*$")  # the page number, at the very end of the page's text
MIN_CHARS = 100  # shorter pages (e.g. 'Lecture is Over') carry no information


def clean(page_text: str) -> tuple[str, str]:
    """(title, text) of a page: the title is its first line, the text has whitespace collapsed (bullets come flattened anyway)."""
    lines = [line.strip() for line in PAGE_NUMBER.sub("", FOOTER.sub("", page_text)).splitlines() if line.strip()]
    return (lines[0] if lines else ""), " ".join(" ".join(lines).split())


def slides(pdf: Pdf, path: Path) -> list[Slide]:
    chunks = []
    for page, content in enumerate(PdfReader(path).pages, start=1):
        title, text = clean(content.extract_text())
        if len(text) >= MIN_CHARS:
            chunks.append(Slide(pdf.lecture, page, title, text, pdf.tag))
    return chunks


if __name__ == "__main__":  # shows what gets indexed, e.g. to tune the cleaning
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="latest", help="release of the slides, e.g. 2026.10.08 (default: latest)")
    for pdf in release(parser.parse_args().tag):
        for slide in slides(pdf, download(pdf)):
            print(f"[{slide.url}] {slide.title}\n    {slide.text[:200]}...")
