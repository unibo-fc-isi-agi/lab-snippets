"""
Checks the application of a candidate: which documents are there, and what the recommendation letter says.
Prints a JSON report. Standard library only, so that any harness can run it with any Python 3.

Usage: python scripts/check_application.py [CANDIDATE_ID] [--data DATA_DIR]
       (no candidate: lists the candidates; DATA_DIR defaults to the data/ folder of lab-snippets)
"""
import json
import re
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[5] / "data"  # i.e. lab-snippets/data/ (resolve() follows symlinks)


def candidates(data: Path) -> list[str]:
    return sorted(path.stem.removeprefix("letter-") for path in data.glob("letter-*.txt"))


def check(candidate: str, data: Path) -> dict:
    documents = {kind: data / f"{kind}-{candidate}.{ext}"
                 for kind, ext in [("letter", "txt"), ("passport", "png"), ("transcript", "png")]}
    report = {
        "candidate": candidate,
        "documents": {kind: str(path) for kind, path in documents.items() if path.exists()},
        "missing": [kind for kind, path in documents.items() if not path.exists()],
    }
    if documents["letter"].exists():
        text = documents["letter"].read_text()
        name = candidate.replace("-", " ")
        report["letter"] = {
            "words": len(text.split()),
            "dates": re.findall(r"\b\d{1,2} (?:January|February|March|April|May|June|July|August|"
                                r"September|October|November|December) \d{4}\b", text),
            "mentions_candidate": name.lower() in text.lower(),
            "text": text,
        }
    return report


if __name__ == "__main__":
    args = sys.argv[1:]
    data = Path(args.pop(args.index("--data") + 1)) if "--data" in args else DATA
    args = [arg for arg in args if arg != "--data"]
    if not args:
        print(json.dumps({"candidates": candidates(data)}, indent=2))
    elif args[0] not in candidates(data):  # never trust the arguments: e.g. "../../.ssh/id_rsa"
        sys.exit(f"Unknown candidate {args[0]!r}; known ones: {candidates(data)}")
    else:
        print(json.dumps(check(args[0], data), indent=2, ensure_ascii=False))
