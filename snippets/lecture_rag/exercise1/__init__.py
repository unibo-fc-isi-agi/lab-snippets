"""
Exercise 1: Q/A about the Slides of this Course.

A RAG system answering questions about this course, citing the slides that support each answer.
The slides are released as PDFs (one <lecture>_slides.pdf per lecture) at https://github.com/unibo-fc-isi-agi/slides-module2/releases

TODO:
1. resolve a release (default: latest) via GitHub's API, and download its PDFs into a git-ignored cache,
   using the assets' digests (SHA-256) to skip unchanged files
   (tip: send a GITHUB_TOKEN, if set: unauthenticated calls get 60 requests/hour per IP, which shared networks exhaust quickly)
2. load the PDFs page by page (e.g. via pypdf): one page = one slide = one chunk;
   strip the boilerplate repeated on every page, skip near-empty pages, keep lecture, page, title, and a link to the page as metadata
3. store the chunks with their embeddings (snippets.lecture_rag.embeddings) in SQLite via sqlite-vec (snippets.lecture_rag.vec),
   re-indexing only the lectures whose PDF changed
4. answer questions: retrieve the top-k slides (optionally within one lecture), give them to the LLM as delimited DATA,
   and get a structured answer with citations (lecture, page, link), or "not covered by the slides"
5. evaluate: write a gold set of ~10 questions with the (lecture, slide title) answering them, pinned to a release,
   and compute recall@k and MRR; optionally, check that answers are grounded (e.g. DeepEval's FaithfulnessMetric)

Put your solution here.
"""
