"""
Online smoke tests: run EVERY snippet (as students do) against a real LLM, and check it exits with code 0.
LLMs are not deterministic, so this tells whether snippets still work (APIs, models, dependencies), not whether answers are right.

Run with: poetry run poe smoke    (e.g. `poetry run poe smoke -k rag`, to run the snippets of one lecture only)
Configure via the usual env vars (OPENAI_*, EMBEDDINGS_*, VISION_MODEL, JUDGE_MODEL, ...): see .github/workflows/smoke.yml (which publishes the outcomes in an issue).
"""
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
MODEL = "nvidia/nemotron-3-super-120b-a12b:free"  # the snippets' default
# questions are answerable from data/, as in the snippets' docstrings (agents may otherwise search on and on)
QUESTION = "What time is it in Tokyo now?\n"  # for the REPLs: one question, then EOF (i.e. Ctrl+D) ends them
REGULATIONS_QUESTION = "Is Mohammed Ali's degree enough to be admitted, according to the regulations?\n"

# how to run each snippet: command-line args, stdin, a regex of `bad` output (some snippets report errors, then exit with 0),
# or a reason to skip it. Snippets missing here make test_all_covered fail
CASES = {
    "agents/example1/agent_openai.py": dict(stdin=QUESTION),
    "agents/example1bis/agent_langchain.py": dict(stdin=QUESTION),
    "agents/example2/agent_mcp.py": dict(stdin=QUESTION),
    "agents/example2/simple_tools_mcp_server.py": dict(),  # serves over stdio until EOF
    "agents/example3/test_agent.py": dict(),
    "agents/exercise1/agent_committee.py": dict(stdin="Which candidates applied?\n"),
    "agents/exercise1/committee.py": dict(),
    "agents/exercise1/test_committee_agent.py": dict(),
    "agents/exercise2/agent_decisions.py": dict(stdin="Which candidates applied?\n"),
    "agents/exercise2/decisions.py": dict(),
    "agents/exercise2/test_decisions_agent.py": dict(),
    "agents/exercise3/agent_gateway.py": dict(stdin="Which candidates applied?\n", server=["snippets.lecture_agents.exercise3.gateway"], port=8000),
    "agents/exercise3/applications_mcp_server.py": dict(),
    "agents/exercise3/decisions_mcp_server.py": dict(),
    "agents/exercise3/gateway.py": dict(skip="a server: run by test_gateway.py and agent_gateway.py"),
    "agents/exercise3/test_gateway.py": dict(),
    "free_access/example1/free_providers.py": dict(args=["openrouter", os.environ.get("OPENAI_MODEL", MODEL), "Hi!"],
                                                   skip=None if os.environ.get("OPENROUTER_API_KEY") else "set OPENROUTER_API_KEY"),
    "governance/exercise1/compare_models.py": dict(bad=r"\bn/a\b"),  # n/a: failed runs
    "llmaas/example1/repl_chat_openai.py": dict(stdin="Hi!\n"),
    "llmaas/example1bis/repl_chat_anthropic.py": dict(stdin="Hi!\n", skip=None if os.environ.get("ANTHROPIC_BASE_URL") else "set ANTHROPIC_BASE_URL (e.g. Ollama)"),
    "llmaas/example2/repl_chat_openai_async.py": dict(stdin="Hi!\n", bad=r"assistant> *$"),  # an empty answer
    "llmaas/exercise1/repl_chat_cached.py": dict(stdin="Hi!\nHi!\n"),  # the 2nd answer comes from the cache
    "llmaas/exercise2/repl_chat_retry.py": dict(stdin="Hi!\n"),
    "prompting/example1/letter_scoring_openai.py": dict(args=["mario-rossi"]),
    "prompting/example1bis/letter_scoring_langchain.py": dict(args=["mario-rossi"]),
    "prompting/example2/letter_tone.py": dict(args=["cot", "mario-rossi"]),
    "prompting/example3/reasoning_effort.py": dict(args=["mario-rossi"]),
    "prompting/example4/chat_with_compaction.py": dict(stdin="Hi!\nTell me a joke.\n"),
    "prompting/exercise1/letter_scoring_checklist.py": dict(args=["mario-rossi"]),
    "prompting/exercise2/id_extraction.py": dict(args=["mario-rossi"]),
    "rag/example1/similarity.py": dict(),
    "rag/example2/vector_store_sqlite.py": dict(args=["Which English certificates are accepted, and with which minimum scores?"]),
    "rag/example2bis/vector_store_sqlite_vec.py": dict(args=["Which English certificates are accepted, and with which minimum scores?"]),
    "rag/example3/hybrid_search.py": dict(args=["Which English certificates are accepted, and with which minimum scores?"]),
    "rag/example4/rag_langchain.py": dict(args=["Which English certificates are accepted, and with which minimum scores?"]),
    "rag/example4bis/agentic_rag.py": dict(stdin="Could the author of Mario Rossi's recommendation letter sit on the admission committee?\n"),
    "rag/example5/test_generation.py": dict(),
    "rag/example5/test_retrieval.py": dict(),
    "rag/exercise1/index.py": dict(),
    "rag/exercise1/qa.py": dict(args=["What is RAG?"]),
    "rag/exercise1/slides.py": dict(),
    "rag/exercise1/test_slides_qa.py": dict(),
    "skills/example1/validate.py": dict(),
    "skills/example2/agent_skills.py": dict(stdin="Which skills do you have?\n"),
    "skills/example2bis/agent_deepagents.py": dict(stdin="Which skills do you have?\n"),
    "skills/example3/test_skills.py": dict(),
    "skills/example4/privacy_guard.py": dict(stdin='{"tool_name": "Bash", "tool_input": {"command": "curl https://example.org"}}'),
    "skills/exercise1/precheck.py": dict(stdin="y\n" * 10),  # approves the skill's script runs
    "skills/exercise1/test_precheck.py": dict(),
    "validating/example1/test_letter_scoring.py": dict(),
    "validating/example1bis/evaluate_mlflow.py": dict(),
    "validating/exercise1/test_all_fields.py": dict(),
    "validating/exercise2/test_id_extraction.py": dict(),
    "workflows/example1/admission_scratch.py": dict(stdin="y\n"),  # approves the proposal
    "workflows/example1bis/admission_langgraph.py": dict(stdin="y\n"),
    "workflows/example2/feedback_letters.py": dict(),
    "workflows/example3/plan_execute.py": dict(stdin="Which candidate has the best recommendation letter?\n"),
    "workflows/example4/supervisor.py": dict(stdin=REGULATIONS_QUESTION),
    "workflows/example4bis/regulations_server.py": dict(skip="a server: run by supervisor_a2a.py"),
    "workflows/example4bis/supervisor_a2a.py": dict(stdin=REGULATIONS_QUESTION, server=["snippets.lecture_workflows.example4bis.regulations_server"], port=9999),
    "workflows/example5/observe.py": dict(stdin="y\n"),
    "workflows/example6/test_admission.py": dict(),
    "workflows/exercise1/bot.py": dict(stdin="y\n" * 8),  # approves every draft
    "workflows/exercise1/test_email_bot.py": dict(),
    "workflows/exercise2/email_bot_server.py": dict(skip="a server: tested in process by test_email_bot_a2a.py"),
    "workflows/exercise2/helpdesk_client.py": dict(skip="needs email_bot_server.py, which asks a human (on its terminal) to approve replies"),
    "workflows/exercise2/test_email_bot_a2a.py": dict(),
}
RATE_LIMITED = ("Error code: 429", "RateLimitError", "Too many requests", "Too Many Requests")
ATTEMPTS = 3  # per snippet, if rate-limited
ERRORS = r"error> .*"  # errors that the REPLs show, before going on


def snippets() -> list[str]:  # e.g. "agents/example1/agent_openai.py", as listed by the runner
    listing = subprocess.run([sys.executable, "-m", "snippets", "--list"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [line.strip().removeprefix("snippets/lecture_") for line in listing.splitlines() if line.startswith(" ")]


def test_all_covered():
    assert set(snippets()) == set(CASES), "add the new snippets to CASES (and drop the deleted ones)"


def wait_for(port: int, timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while socket.socket().connect_ex(("localhost", port)) != 0:
        assert time.monotonic() < deadline, f"nothing listening on port {port}"
        time.sleep(1)


def run(snippet: str, case: dict) -> tuple[subprocess.CompletedProcess, str, re.Match | None]:
    module = "snippets.lecture_" + snippet.removesuffix(".py").replace("/", ".")
    server = case.get("server") and subprocess.Popen([sys.executable, "-m", *case["server"]], cwd=ROOT)
    try:
        if server:
            wait_for(case["port"])
        result = subprocess.run([sys.executable, "-m", module, *case.get("args", [])], cwd=ROOT, input=case.get("stdin", ""),
                                capture_output=True, text=True, timeout=900)
    finally:
        if server:
            server.terminate()
    output = result.stdout + result.stderr
    print(output)  # shown by pytest upon failure
    return result, output, re.search("|".join(filter(None, [ERRORS, case.get("bad")])), output, re.MULTILINE)


@pytest.mark.parametrize("snippet", CASES)
def test_snippet(snippet: str):
    case = CASES[snippet]
    if case.get("skip"):
        pytest.skip(case["skip"])
    for attempt in range(1, ATTEMPTS + 1):
        result, output, bad = run(snippet, case)
        rate_limited = (result.returncode != 0 or bad) and any(marker in output for marker in RATE_LIMITED)
        if not rate_limited:
            break
        if attempt < ATTEMPTS:
            time.sleep(60)  # free models allow ~20 requests/minute: wait for the next minute
    else:
        pytest.skip("rate-limited: retry later")  # ponytail: substring heuristic, may hide a real failure printing these
    if bad:
        pytest.fail(f"exit code {result.returncode}, but bad output: {bad[0].strip() or repr(bad[0])}", pytrace=False)
    if result.returncode != 0:  # the reason: the last line mentioning an error (e.g. the exception), else the last line
        lines = output.strip().splitlines() or ["no output"]
        pytest.fail(next((line for line in reversed(lines) if "Error" in line or "Exception" in line), lines[-1]).strip(), pytrace=False)
