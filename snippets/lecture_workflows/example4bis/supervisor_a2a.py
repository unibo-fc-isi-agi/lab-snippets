"""
The supervisor of Example 4, whose regulations agent is now a REMOTE peer, reached via A2A (cf. regulations_server.py):
only the body of the ask_regulations_agent tool changes. The documents agent is still local.

Run with: poetry run python -m snippets -l workflows -e 4bis   (then pick supervisor_a2a.py; start regulations_server.py first)
e.g., ask: "Is Mohammed Ali's degree enough to be admitted, according to the regulations?"
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (must support tool calling), VISION_MODEL,
REGULATIONS_AGENT_URL (default: http://localhost:9999).
"""
import asyncio
import os
import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import get_artifact_text, new_text_message
from a2a.types import Role, SendMessageRequest, TaskState
from langchain.agents import create_agent
from snippets.lecture_agents.example1bis.agent_langchain import llm
from snippets.lecture_workflows.example4.supervisor import ask_documents_agent, instructions

REGULATIONS_AGENT_URL = os.environ.get("REGULATIONS_AGENT_URL", "http://localhost:9999")


async def ask_regulations_agent(question: str) -> str:
    """Ask a colleague who knows the regulations of the PhD programme a self-contained question."""
    http = httpx.AsyncClient(timeout=300)  # remote agents call LLMs: the default timeout (5 seconds) is too short
    client = await create_client(REGULATIONS_AGENT_URL, ClientConfig(httpx_client=http))  # fetches the Agent Card
    answer = ""
    try:
        request = SendMessageRequest(message=new_text_message(question, role=Role.ROLE_USER))
        async for event in client.send_message(request):  # streamed: the task, its status updates, its artifacts
            if event.HasField("status_update"):
                print(f"    [regulations agent, task {event.status_update.task_id[:8]}] {TaskState.Name(event.status_update.status.state)}")
            elif event.HasField("artifact_update"):
                answer += get_artifact_text(event.artifact_update.artifact)
            elif event.HasField("task"):  # e.g. the final state of the task, with all its artifacts
                answer = answer or "\n".join(get_artifact_text(artifact) for artifact in event.task.artifacts)
    finally:
        await client.close()
        await http.aclose()
    return answer or "The regulations agent gave no answer."


supervisor = create_agent(llm, tools=[ask_documents_agent, ask_regulations_agent], system_prompt=instructions)


async def main() -> None:
    messages = []
    while True:
        try:
            messages.append(("user", input("You: ")))
        except (EOFError, KeyboardInterrupt):
            break
        messages = (await supervisor.ainvoke({"messages": messages}, {"recursion_limit": 20}))["messages"]  # async, as the A2A client
        print(f"AI: {messages[-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
