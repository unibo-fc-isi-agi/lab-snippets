"""
A minimal A2A client (e.g. the university's help desk): it sends an e-mail of the inbox to the e-mail agent, as a data Part,
and streams the task's updates until it ends; then prints the reply (an artifact), if any.

Run with: poetry run python -m snippets -l workflows -x 2 [EMAIL_ID ...]   (then pick helpdesk_client.py; start the server first)
e.g.: ... 01-age-limit 07-admin-mode   (default: every e-mail of data/inbox/)
Configure via env vars: EMAIL_BOT_URL (default: http://localhost:9998).
"""
import asyncio
import os
import sys
import httpx
from a2a.client import Client, ClientConfig, create_client
from a2a.helpers import get_artifact_text, get_message_text, new_data_message
from a2a.types import Role, SendMessageRequest, TaskState
import data

EMAIL_BOT_URL = os.environ.get("EMAIL_BOT_URL", "http://localhost:9998")


async def delegate(client: Client, email: dict) -> tuple[str, str]:
    """Sends an e-mail to the agent, printing the task's updates; returns the task's final state, and the reply (if any)."""
    request = SendMessageRequest(message=new_data_message({key: email[key] for key in ("sender", "subject", "body")}, role=Role.ROLE_USER))
    state, reply = "", ""
    async for event in client.send_message(request):
        if event.HasField("status_update"):
            status = event.status_update.status
            state = TaskState.Name(status.state)
            print(f"    [task {event.status_update.task_id[:8]}] {state}: {get_message_text(status.message)}")
        elif event.HasField("artifact_update"):
            reply += get_artifact_text(event.artifact_update.artifact)
        elif event.HasField("task"):
            state = TaskState.Name(event.task.status.state)
            reply = reply or "\n".join(get_artifact_text(artifact) for artifact in event.task.artifacts)
    return state, reply


async def main(ids: list[str]) -> None:
    async with httpx.AsyncClient(timeout=None) as http:  # no timeout: replies wait for a human's approval (hours?)
        client = await create_client(EMAIL_BOT_URL, ClientConfig(httpx_client=http))  # fetches the Agent Card
        for email in data.inbox():
            if not ids or email["id"] in ids:
                print(f"=== {email['id']}: {email['subject']}")
                state, reply = await delegate(client, email)
                print(f"--- {state}" + (f", reply:\n{reply}" if reply else ""))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
