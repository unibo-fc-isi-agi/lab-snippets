# Lab snippets, examples, and exercises

This repository contains the code snippets (examples and exercises)
for the course "Intelligent Agents" (module 2) at the University of Bologna,
part of the Master's degree in Computer Science and Engineering.

Slides are available at <https://unibo-fc-isi-agi.github.io/slides-module2>.
Each snippet corresponds to an example (or exercise) in the slides, with the same index.

Most snippets work on the course's __running example__: an assistant for the admission committee of a PhD programme,
which has to assess the applications of some candidates (recommendation letter, passport, transcript of records).

## File structure

```
<root directory>
├── data/                                  # the running example's data, plus helpers to locate files (data/__init__.py)
└── snippets/
    ├── __init__.py                        # utilities shared across lectures
    ├── __main__.py                        # the runner (see below)
    └── lecture_<NAME>/
        ├── <UTILITY>.py                   # utilities shared by the snippets of the lecture
        ├── example<ID>/
        │   └── <DESCRIPTION>.py
        └── exercise<ID>/                  # placeholder: put your solution here!
```

where
- `NAME` is the name of the lecture, as in the URL of its slides (e.g. `llmaas`, `prompting`, `agents`)
- `ID` is the index of the example (or exercise) in the slides, e.g. `1`, `1bis`, `2`
- `DESCRIPTION` is a short description of the snippet

| Lecture | Topic |
|---------|-------|
| `llmaas` | LLM-as-a-Service |
| `free_access` | Free Access to LLMs (Appendix) |
| `prompting` | Prompt Engineering & Structured Outputs |
| `validating` | Validating Generative Software |
| `agents` | Tools and Agents |
| `governance` | AI Governance 101 |

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

> Do NOT run snippets as standalone scripts, i.e. do NOT run them via `python path/to/snippet.py`.
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
