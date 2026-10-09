# Pre-check checklist

Severity: __blocking__ (must be fixed before submission), __major__ (should be fixed), __minor__ (nice to fix).

## Objective checks (facts from `scripts/pdf_facts.py`)

| Check | Severity |
|-------|----------|
| Unresolved references or citations (`??`, `[?]`) | blocking |
| No bibliography, or a bibliography with no entries | blocking |
| Figures, tables, or listings never referenced in the text | major |
| Repeated words (e.g. "the the") | minor |
| Near-empty pages (often fine: part pages, blank versos; report only odd ones, e.g. in the middle of a chapter) | minor |
| Outline (PDF bookmarks) missing, or not matching the chapters | minor |

## Judgement checks (yours: quote the thesis)

| Check | Where to look | Severity |
|-------|---------------|----------|
| The abstract is self-contained: context, problem, contribution, results | abstract | major |
| The research questions (or goals) are stated explicitly | introduction | major |
| The contributions are listed, and match what the thesis does | introduction, conclusions | major |
| The conclusions answer the research questions, and discuss limitations and future work | conclusions | major |
| The introduction outlines the structure of the thesis, and it matches the actual chapters | introduction, outline | minor |
| Terminology and acronyms are consistent (e.g. defined once, then used) | abstract, introduction | minor |
| Typos and grammar issues, as noticed while reading (do not hunt for them) | anywhere | minor |
| The front matter states title, author, supervisor(s), degree, and academic year | first pages | major |
| Any statement on the use of generative AI tools, if the degree programme requires one (ask the student) | front or back matter | major |
