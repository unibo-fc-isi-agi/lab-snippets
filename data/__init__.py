"""
Data of the running example: the applications of the candidates to a PhD programme.
Each candidate comes with a recommendation letter (text), and the pictures of their passport and transcript of records.

Use the helpers below to locate files, rather than hard-coding their paths, e.g.:

    import data
    data.CANDIDATES                         # ['jean-dupont', 'mario-rossi', 'mohammed-ali']
    data.letter("mario-rossi").read_text()  # the text of Mario Rossi's letter
    data.passport("jean-dupont")            # the path of Jean Dupont's passport picture
    data.find_letter(sys.argv[1])           # a letter, given either a candidate's ID or a path
    data.regulations().read_text()          # the (fictional) regulations of the PhD programme, in Markdown
    data.applications()["mario-rossi"]      # the e-mail address and the status of Mario Rossi's application
    data.inbox()                            # the e-mails received by the admission office (sender, subject, body)
"""
import json
from pathlib import Path

DIR = Path(__file__).parent  # i.e. <project root>/data/

# the candidates' IDs, as they appear in file names (so adding a letter-<ID>.txt file adds a candidate)
CANDIDATES = sorted(path.stem.removeprefix("letter-") for path in DIR.glob("letter-*.txt"))


def letter(candidate: str) -> Path:
    """Path of the candidate's recommendation letter, e.g. data/letter-mario-rossi.txt."""
    return DIR / f"letter-{candidate}.txt"


def passport(candidate: str) -> Path:
    """Path of the picture of the candidate's passport, e.g. data/passport-mario-rossi.png."""
    return DIR / f"passport-{candidate}.png"


def transcript(candidate: str) -> Path:
    """Path of the picture of the candidate's transcript of records, e.g. data/transcript-mario-rossi.png."""
    return DIR / f"transcript-{candidate}.png"


def regulations() -> Path:
    """Path of the (fictional) regulations of the PhD programme, i.e. data/regulations-phd.md, one article per section."""
    return DIR / "regulations-phd.md"


def letters() -> list[Path]:
    """Paths of all the candidates' letters."""
    return [letter(candidate) for candidate in CANDIDATES]


def find_letter(name: str) -> Path:
    """Path of a letter, given either a candidate's ID (e.g. 'mario-rossi') or a path (e.g. 'data/letter-mario-rossi.txt')."""
    return letter(name) if name in CANDIDATES else Path(name)


def applications() -> dict[str, dict]:
    """The (fictional) applications' records of the admission office: candidate's ID -> their e-mail address, and status."""
    return json.loads((DIR / "applications.json").read_text())


def inbox() -> list[dict]:
    """The e-mails received by the admission office (data/inbox/*.txt), as dicts with keys: id, sender, subject, body."""
    emails = []
    for path in sorted((DIR / "inbox").glob("*.txt")):
        headers, _, body = path.read_text().partition("\n\n")  # headers, a blank line, then the body
        fields = dict(line.split(": ", 1) for line in headers.splitlines())
        emails.append({"id": path.stem, "sender": fields["From"], "subject": fields["Subject"], "body": body.strip()})
    return emails
