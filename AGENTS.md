# AGENTS.md

Guidance for AI agents (and humans) working on this repository.
For layout, setup, runner usage, and the suggested order of snippets, read `README.md` first: it is not repeated here.

## What this is

The runnable code (examples, exercises, running-example data) of **Module 2** of the course *Intelligent Agents*
(University of Bologna), whose slides live in <https://github.com/unibo-fc-isi-agi/slides-module2>
(published at <https://unibo-fc-isi-agi.github.io/slides-module2>).
The slides repo mounts this one as the git submodule `static/lab-snippets/`, and **excerpts its files by line range**.

## Coupling with the slides

- `snippets/lecture_<NAME>/{example,exercise}<ID>/`: `NAME` is the lecture's slides directory (`-` → `_`),
  `ID` is the index used in the slides (`1`, `1bis`, ...). Don't rename lectures, folders or files: slides link to them.
- Slides include excerpts via `{{% code path="static/lab-snippets/snippets/.../file.py" from="10" to="20" %}}`.
  **Any edit that shifts lines breaks those ranges**: after changing a snippet, grep the slides repo
  (`content/**/_index.md`) for the file's path and re-check every `from`/`to`.
- After pushing here, the slides repo must bump the submodule pointer (`git -C static/lab-snippets pull`, commit).
- Every example/exercise folder has an `__init__.py` whose docstring's first line is its title
  (`Example 1: ...`), shown by the runner's `--list`; each `lecture_<NAME>/__init__.py` names the lecture and links its slides.
- Files starting with `_` are not runnable by the runner (private helpers); lecture-wide utilities go in
  `lecture_<NAME>/*.py`, cross-lecture ones in `snippets/__init__.py`, data helpers in `data/__init__.py`.

## Branches: solutions vs. exercises

| Branch | Holds |
|--------|-------|
| `master` | examples + **solutions** of exercises (commented, as they are walked through in the slides) + their offline tests in `tests/` |
| `exercises` | `master` with each `exercise<ID>/` reduced to an `__init__.py` placeholder (docstring with a `TODO:` list, ending with "Put your solution here."), and no tests for them |

- Write and fix solutions on `master` only.
- Examples (and lecture-wide modules) must never import exercise solutions, of any lecture: on `exercises` they are missing.
  Reuse examples instead.
- Then `git checkout exercises && git merge master`, keeping the placeholders: drop solution files and
  their tests from the merge, and update a placeholder's TODO list if the exercise's text changed.
- A new exercise needs: the solution + a test in `tests/test_<NAME>.py` + a `CASES` entry in `smoke/test_snippets.py` on `master`; a placeholder on `exercises`;
  a row update in the README's "Suggested order" table.

## Checks (same as CI, `.github/workflows/check.yml`)

```bash
poetry run poe compile   # syntax
poetry run poe list      # every snippet discoverable by the runner
poetry run poe collect   # test snippets importable (set OPENAI_API_KEY=dummy)
poetry run poe test      # offline tests of the solutions
```

Plus `poetry run poe smoke` (`.github/workflows/smoke.yml`, daily and on demand, never on push): runs **every** snippet against
real LLMs (in CI: the snippets' defaults, i.e. Open Router's `:free` models and local Ollama ones, so keep defaults free)
and lists each one's outcome in the issue labelled `smoke-report` (rewritten at each full run). Every runnable snippet needs an entry in its `CASES` table
(`smoke/test_snippets.py`: args, stdin, or a reason to skip), or it fails.

- Tests must stay **offline**: no LLM / embedding calls; test only the deterministic parts (mock the model).
- Modules must be importable without network or API endpoints (CI collects them): no work at import time
  besides reading env vars; e.g. build indexes lazily.

## Code style

- Snippets are **teaching material**: short, readable top-to-bottom, commented where a slide will discuss them;
  prefer clarity over abstraction, and the same idiom as sibling snippets of the lecture.
- LLMs are reached via OpenAI-compatible APIs, configured by env vars (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`;
  `EMBEDDINGS_*` for `rag`); never hard-code keys or providers.
- Use the running example's data in `data/` (letters, passports, transcripts of 3 PhD candidates, `regulations-phd.md`)
  via the helpers in `data/__init__.py`, not hard-coded paths.
- Commands are always run from the repository root, through the runner (`poetry run python -m snippets -l <NAME> -e <ID>`).

## Don'ts

- Don't commit run artifacts (`output/`, `.rag-cache/`, `.llm-cache/`, `.deepeval/`, `mlruns/`, `mlflow.db`, `.venv/`).
- Don't put solutions on `exercises`, nor placeholders on `master`.
- Don't add dependencies lightly: students install them all (`poetry install`); update `poetry.lock` together with `pyproject.toml`.
