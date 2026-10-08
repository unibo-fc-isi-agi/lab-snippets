# Lab snippets, examples, and exercises

This repository contains the code snippets (examples and exercises)
for the course "Intelligent Agents" (module 2) at the University of Bologna,
part of the Master's degree in Computer Science and Engineering.

Slides are available at <https://unibo-fc-isi-agi.github.io/slides-module2>.
Each snippet corresponds to an example (or exercise) in the slides, with the same index.

## Branches: exercises vs. solutions

| Branch | Contents | Who |
|--------|----------|-----|
| [`exercises`](https://github.com/unibo-fc-isi-agi/lab-snippets/tree/exercises) | examples, plus a placeholder (with a TODO list) for each exercise | __students__: clone this one, and put your solutions in the placeholders |
| [`master`](https://github.com/unibo-fc-isi-agi/lab-snippets/tree/master) | examples, plus the (commented) __solutions__ of exercises, tested by CI | for checking your solution _after_ trying, and for the walkthroughs in the slides |

```bash
git clone -b exercises https://github.com/unibo-fc-isi-agi/lab-snippets.git
```

Most snippets work on the course's __running example__: an assistant for the admission committee of a PhD programme,
which has to assess the applications of some candidates (recommendation letter, passport, transcript of records).

## File structure

```
<root directory>
├── data/                                  # the running example's data, plus helpers to locate files (data/__init__.py)
├── tests/                                 # offline tests of the exercises' solutions (master only)
└── snippets/
    ├── __init__.py                        # utilities shared across lectures
    ├── __main__.py                        # the runner (see below)
    └── lecture_<NAME>/
        ├── <UTILITY>.py                   # utilities shared by the snippets of the lecture
        ├── example<ID>/
        │   └── <DESCRIPTION>.py
        └── exercise<ID>/                  # placeholder: put your solution here! (solution, on master)
```

where
- `NAME` is the name of the lecture, as in the URL of its slides (e.g. `llmaas`, `prompting`, `agents`)
- `ID` is the index of the example (or exercise) in the slides, e.g. `1`, `1bis`, `2`
- `DESCRIPTION` is a short description of the snippet

## Suggested order

Lectures are meant to be followed in the order below (the same as in the slides' table of contents),
and so are the snippets within each lecture (the order of the slides, which is _not_ always the order of IDs).
Exercises build upon the examples (and exercises) listed before them, as detailed in the last column.

| # | Lecture (`NAME`) | Snippets, in order | Notes |
|---|------------------|--------------------|-------|
| 1 | [Generative AI 101](https://unibo-fc-isi-agi.github.io/slides-module2/genai/) | _none_ | |
| 2 | [LLM-as-a-Service](https://unibo-fc-isi-agi.github.io/slides-module2/llmaas/) (`llmaas`) | example 1, example 2, example 1bis, exercise 1, exercise 2 | exercises 1 and 2 extend example 1 |
| — | [Free Access to LLMs](https://unibo-fc-isi-agi.github.io/slides-module2/free-access/) (`free_access`) | example 1 | appendix: read it _before_ running any snippet, to get an API key (or a local model) |
| 3 | [Prompt Engineering & Structured Outputs](https://unibo-fc-isi-agi.github.io/slides-module2/prompting/) (`prompting`) | example 1, example 1bis, exercise 1, example 2, example 3, exercise 2, example 4 | exercise 1 extends example 1bis |
| 4 | [Validating Generative Software](https://unibo-fc-isi-agi.github.io/slides-module2/validating/) (`validating`) | example 1, example 1bis, exercise 1, exercise 2 | exercise 1 extends example 1 (or 1bis); exercise 2 tests the solution of `prompting` exercise 2 |
| 5 | [Tools and Agents](https://unibo-fc-isi-agi.github.io/slides-module2/agents/) (`agents`) | example 1, example 1bis, example 2, example 3, exercise 1, exercise 2, exercise 3 | exercise 1 reuses `prompting` exercise 2 (pictures) and, optionally, `prompting` example 1 or 1bis (letter scoring); exercise 2 extends exercise 1; exercise 3 extends exercises 1 and 2 |
| 6–8 | _RAG, Agentic Skills, Workflows and Agent Orchestration_ | _coming soon_ | |
| 9 | [AI Governance 101](https://unibo-fc-isi-agi.github.io/slides-module2/governance/) (`governance`) | exercise 1, exercise 2 | exercise 1 runs `prompting` example 1 (or 1bis) with several models; exercise 2 builds on exercise 1 |

## Prepare the environment

To run the snippets, you need __Python__ (3.10 or later) installed on your machine.

You also need [Poetry](https://python-poetry.org), a Python dependency manager.
If that's not installed, you can install it by running the following command:

```bash
pip install -r requirements.txt
```

Once Poetry is installed, you can install the necessary dependencies by running the following command:

```bash
poetry install
```

This will create a virtual environment in the `.venv` directory, and install the necessary dependencies there.

> **Note**: after you create the virtual environment, VSCode may ask you to select the Python interpreter.
> You can select the one in the `.venv` directory.

### API keys and models

Most snippets call some LLM via an OpenAI-compatible API (by default, [OpenRouter](https://openrouter.ai)),
and they are configured via environment variables:
- `OPENAI_BASE_URL`: the API's URL (default: `https://openrouter.ai/api/v1/`)
- `OPENAI_API_KEY`: your API key (if missing, it is asked interactively)
- `OPENAI_MODEL`: the model to use (default: `openrouter/auto`)

See the slides for how to get free access to LLMs, or to run them locally (e.g. via [Ollama](https://ollama.com)).

## How to run a snippet

> Prefer the runner below to `python path/to/snippet.py`: same command for every snippet, no paths to remember.
> Always run commands from the root directory of the project.

To run a snippet, use the following command:

```bash
poetry run python -m snippets --lecture <NAME> --example <ID> [ARGS]
# or equivalently:
poetry run python -m snippets -l <NAME> -e <ID> [ARGS]
# e.g.:
poetry run python -m snippets -l prompting -e 1bis mario-rossi
```

where `ARGS` are passed to the snippet.
Use `--exercise` (or `-x`) instead of `--example` to run your solution to an exercise.

If more snippets match (e.g. when `-e` is omitted, or the example consists of several files), you are asked to pick one.
To list the snippets, without running them, use `--list`:

```bash
poetry run python -m snippets --list               # all snippets
poetry run python -m snippets -l agents --list     # all snippets of a lecture
```

If some of the snippet's arguments clash with the runner's ones (e.g. `-x`, for pytest), put them after `--`:

```bash
poetry run -- python -m snippets -l validating -e 1 -- -x -v
```

> **Note**: the `poetry run` prefix ensures that snippets run in the project's virtual environment.
> If you have activated the virtual environment (e.g. via `eval $(poetry env activate)`), you can omit it.
