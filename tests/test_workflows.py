import os

os.environ.setdefault("OPENAI_API_KEY", "dummy")
import asyncio
import httpx
import pytest
from a2a.client import ClientConfig, create_client
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
import data
from snippets.lecture_workflows import admission
from snippets.lecture_workflows.exercise1 import bot
from snippets.lecture_workflows.exercise2 import email_bot_server
from snippets.lecture_workflows.exercise2.helpdesk_client import delegate
from snippets.lecture_workflows.example4bis.regulations_server import app

EMAILS = {email["id"]: email for email in data.inbox()}
ROUTES = {"01-age-limit": "regulations", "03-status": "status", "06-restaurant": "out_of_scope", "07-admin-mode": "suspicious",
          "08-mohammed-status-from-other-address": "status"}
ROUTE_BY_SUBJECT = {EMAILS[id]["subject"]: route for id, route in ROUTES.items()}  # in A2A, e-mails' IDs are the tasks' ones


@pytest.fixture(autouse=True)
def fake_llm(tmp_path, monkeypatch):  # the bot's LLM steps (and RAG) are replaced by fakes; outputs go to a fresh directory
    monkeypatch.setattr(admission, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(bot, "classify_email", lambda email: ROUTE_BY_SUBJECT.get(email["subject"], "regulations"))
    monkeypatch.setattr(bot, "search_regulations", lambda query: "## Article 3 - Admission requirements\nThere is no age limit.")
    monkeypatch.setattr(bot, "write_reply", lambda email, facts, previous, issues: f"Reply to {email['sender']}, based on: {facts}")
    monkeypatch.setattr(bot, "judge_reply", lambda email, facts, reply: [])
    return tmp_path


def run(email: dict, decision: dict | None = None) -> dict:
    workflow, config = bot.build(InMemorySaver()), {"configurable": {"thread_id": email["id"]}}
    result = workflow.invoke({"email": email}, config)
    while "__interrupt__" in result and decision:
        result = workflow.invoke(Command(resume=decision), config)
    return result


def test_inbox():
    assert len(EMAILS) == 8 and all(email["sender"] and email["body"] for email in EMAILS.values())


def test_approved_replies_are_sent_once(fake_llm):
    final = run(EMAILS["01-age-limit"], {"decision": "approve"})
    assert final["sent"] and (fake_llm / "sent" / "01-age-limit.txt").exists()
    bot.send_reply(final)  # e.g. a resumed run
    assert len(list((fake_llm / "sent").iterdir())) == 1


def test_rejected_or_pending_replies_are_not_sent(fake_llm):
    assert "__interrupt__" in run(EMAILS["01-age-limit"])  # waiting for a human
    assert not run(EMAILS["03-status"], {"decision": "reject"}).get("sent")
    assert not (fake_llm / "sent").exists()


def test_edited_replies_are_sent_as_edited(fake_llm):
    run(EMAILS["01-age-limit"], {"decision": "edit", "text": "No age limit (Art. 3)."})
    assert "No age limit (Art. 3)." in (fake_llm / "sent" / "01-age-limit.txt").read_text()


def test_suspicious_emails_are_escalated_never_answered(fake_llm):
    final = run(EMAILS["07-admin-mode"], {"decision": "approve"})  # even with a careless human
    assert not final.get("sent") and not final.get("drafts") and (fake_llm / "escalations" / "07-admin-mode.txt").exists()


def test_status_only_for_verified_senders():
    assert "under evaluation" in run(EMAILS["03-status"])["facts"]
    assert "under evaluation" not in run(EMAILS["08-mohammed-status-from-other-address"])["facts"]  # address not on file


def test_judge_catches_other_candidates(monkeypatch):
    monkeypatch.setattr(bot, "write_reply", lambda *args: "Unlike Mario Rossi, your grade is low.")
    final = run(EMAILS["03-status"])  # the sender is Mohammed Ali
    assert len(final["drafts"]) == bot.MAX_DRAFTS and "mario-rossi" in final["critiques"][-1][0]


def a2a(approve, email: dict) -> tuple[str, str]:  # the A2A server, in process (no network), with a simulated office
    async def go():
        transport = httpx.ASGITransport(app=app(email_bot_server.card, email_bot_server.EmailBotExecutor(approve)))
        async with httpx.AsyncClient(transport=transport) as http:
            client = await create_client(email_bot_server.card, ClientConfig(httpx_client=http))
            return await delegate(client, email)
    return asyncio.run(go())


def test_a2a_completed_with_reply():
    state, reply = a2a(lambda request: {"decision": "approve"}, EMAILS["01-age-limit"])
    assert state == "TASK_STATE_COMPLETED" and "no age limit" in reply


def test_a2a_rejected_without_reply():
    assert a2a(lambda request: {"decision": "reject"}, EMAILS["01-age-limit"]) == ("TASK_STATE_REJECTED", "")
    assert a2a(lambda request: {"decision": "approve"}, EMAILS["07-admin-mode"]) == ("TASK_STATE_REJECTED", "")
