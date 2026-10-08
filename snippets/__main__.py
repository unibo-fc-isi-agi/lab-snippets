"""
Runs an example (or exercise) of a lecture, from the project's root directory, e.g.:

    poetry run python -m snippets --lecture prompting --example 1 data/letter-mario-rossi.txt
    poetry run python -m snippets -l prompting -e 1bis data/letter-mario-rossi.txt
    poetry run python -m snippets -l agents -x 1
    poetry run python -m snippets --list

Snippets live in `snippets/lecture_<NAME>/{example,exercise}<ID>/<module>.py`, where
- NAME is the lecture's name, as in the slides' URLs (`free-access` and `free_access` are both fine);
- ID is the example's (or exercise's) index, as in the slides (e.g. `1`, `1bis`, `2`).

Arguments not recognised here are passed to the snippet (put them after `--` in case of clashes).
If several snippets match, you are asked to pick one.
"""
import ast
import re
import runpy
import sys
from argparse import ArgumentParser
from pathlib import Path

SNIPPETS_ROOT = Path(__file__).parent
PROJECT_ROOT = SNIPPETS_ROOT.parent

# names of the example/exercise folders: kind, number, and optional suffix (e.g. "example1bis")
FOLDER_NAME = re.compile(r"(example|exercise)(\d+)(\w*)")


def create_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="poetry run python -m snippets", description="Runs an example (or exercise) of a lecture")
    parser.add_argument("--lecture", "-l", help="name of the lecture, e.g. 'prompting'")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--example", "-e", help="ID of the example, e.g. '1' or '1bis'")
    which.add_argument("--exercise", "-x", help="ID of the exercise, e.g. '2'")
    parser.add_argument("--list", action="store_true", help="list the matching snippets, without running them")
    return parser


def sort_key(folder: Path) -> tuple:  # e.g. lecture_agents/example2 < lecture_agents/example10
    kind, number, suffix = FOLDER_NAME.fullmatch(folder.name).groups()  # type: ignore[union-attr]
    return folder.parent.name, kind, int(number), suffix


def find_folders(lecture: str | None, kind: str | None, id: str | None) -> list[Path]:
    """Folders of the examples/exercises matching the given criteria (None means: any)."""
    folders = []
    for folder in SNIPPETS_ROOT.glob("lecture_*/*/"):
        match = FOLDER_NAME.fullmatch(folder.name)
        if not match:
            continue  # e.g. __pycache__
        if lecture and folder.parent.name != "lecture_" + lecture.replace("-", "_"):
            continue
        if kind and match[1] != kind:
            continue
        if id and match[2] + match[3] != id:
            continue
        folders.append(folder)
    return sorted(folders, key=sort_key)


def runnable_modules(folder: Path) -> list[Path]:
    """Python files of an example/exercise, except private ones (e.g. __init__.py, _helpers.py)."""
    return sorted(path for path in folder.glob("*.py") if not path.name.startswith("_"))


def title(folder: Path) -> str:
    """First line of the docstring in the folder's __init__.py, e.g. 'Example 1: Sync CLI Chat, ...'."""
    docstring = ast.get_docstring(ast.parse((folder / "__init__.py").read_text())) or ""
    return docstring.strip().splitlines()[0] if docstring.strip() else ""


def describe(path: Path) -> str:  # e.g. "llmaas/example1/repl_chat_openai.py: Example 1: Sync CLI Chat, ..."
    folder = path if path.is_dir() else path.parent
    lecture = folder.parent.name.removeprefix("lecture_")
    name = f"{lecture}/{folder.name}" + ("" if path.is_dir() else f"/{path.name}")
    return f"{name}: {title(folder)}"


def run(path: Path, args: list[str]) -> None:
    module = ".".join(path.relative_to(PROJECT_ROOT).with_suffix("").parts)  # e.g. snippets.lecture_llmaas.example1.repl_chat_openai
    print("# Running module", module, "with args:", *args)
    sys.argv = [module, *args]  # as if it were run with: python path/to/module.py ARGS (runpy sets argv[0] to the path)
    runpy.run_module(module, run_name="__main__", alter_sys=True)


def main() -> None:
    args, snippet_args = create_arg_parser().parse_known_args()
    if snippet_args[:1] == ["--"]:
        snippet_args = snippet_args[1:]
    kind = "example" if args.example else "exercise" if args.exercise else None
    folders = find_folders(args.lecture, kind, args.example or args.exercise)
    if not folders:
        sys.exit("# No examples/exercises found")

    modules = [module for folder in folders for module in runnable_modules(folder)]
    if args.list:
        for folder in folders:
            print(describe(folder))
            for module in runnable_modules(folder):
                print("    ", module.relative_to(PROJECT_ROOT))
    elif not modules:  # e.g. an exercise not solved yet
        for folder in folders:
            print(f"# Nothing to run in {folder.relative_to(PROJECT_ROOT)}: put your code there!")
    elif len(modules) == 1:
        run(modules[0], snippet_args)
    else:
        print("# Multiple snippets found, pick one:")
        for i, module in enumerate(modules, start=1):
            print(f"#    {i}) {describe(module)}")
        choice = input("# > ").strip()
        if not choice.isdigit() or not 1 <= int(choice) <= len(modules):
            sys.exit(f"# Invalid choice: {choice!r}")
        run(modules[int(choice) - 1], snippet_args)


if __name__ == "__main__":
    main()
