"""Writes one row per snippet (outcome + reason) to the Markdown file $SMOKE_REPORT, if set, to see at a glance what broke."""
import os

ICONS = {"passed": "✅", "failed": "❌", "skipped": "⏭️"}


def pytest_runtest_logreport(report):
    path = os.environ.get("SMOKE_REPORT")
    if not path or not (report.when == "call" or report.outcome != "passed"):
        return  # one row per test: its call phase, or the setup phase that skipped/failed it
    snippet = report.nodeid.partition("[")[2].rstrip("]") or report.nodeid
    reason = report.longrepr[2].removeprefix("Skipped: ") if report.skipped else report.longreprtext.strip().splitlines()[0] if report.failed else ""
    with open(path, "a") as file:
        if file.tell() == 0:
            file.write("| Snippet | Outcome | Reason |\n|---|---|---|\n")
        file.write(f"| `{snippet}` | {ICONS[report.outcome]} | {str(reason).replace('|', '/')[:200]} |\n")
