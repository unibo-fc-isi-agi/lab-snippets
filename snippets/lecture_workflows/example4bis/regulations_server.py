"""
The regulations agent of Example 4, run by ANOTHER organisation (e.g. the PhD school's office), and exposed via A2A:
an Agent Card describes it, an AgentExecutor turns A2A tasks into runs of the agent, and an HTTP server serves both.

Run with: poetry run python -m snippets -l workflows -e 4bis   (then pick regulations_server.py; stop it with Ctrl+C)
then fetch its Agent Card at http://localhost:9999/.well-known/agent-card.json, and run supervisor_a2a.py in another terminal.
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (must support tool calling),
EMBEDDINGS_BASE_URL, EMBEDDINGS_API_KEY, EMBEDDINGS_MODEL, REGULATIONS_AGENT_PORT (default: 9999).
"""
import os
import uvicorn
from a2a.helpers import get_message_text, new_task_from_user_message, new_text_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, TaskState
from starlette.applications import Starlette

PORT = int(os.environ.get("REGULATIONS_AGENT_PORT", "9999"))

card = AgentCard(  # what clients see of this agent: who it is, what it can do, where and how to reach it
    name="PhD Regulations Agent",
    description="Answers questions about the regulations of the PhD programme in Computer Science and Engineering, citing their articles.",
    version="1.0.0",
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    capabilities=AgentCapabilities(streaming=True),
    supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=f"http://localhost:{PORT}", protocol_version="1.0")],
    skills=[AgentSkill(
        id="regulations-qa",
        name="Questions on the PhD regulations",
        description="Admission requirements, deadlines, evaluation criteria, interviews, scholarships, appeals, ...",
        tags=["phd", "regulations", "admission"],
        examples=["Is there an age limit to apply?", "Which English certificates are accepted?"],
    )],
)


class RegulationsExecutor(AgentExecutor):  # turns A2A requests into runs of the agent, reporting progress as task events
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        from snippets.lecture_workflows.subagents import regulations_agent  # lazily: only when serving requests
        task = context.current_task or new_task_from_user_message(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)  # a new task: its lifecycle starts (submitted)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(TaskState.TASK_STATE_WORKING, message=new_text_message("Searching the regulations..."))
        question = get_message_text(context.message)
        result = await regulations_agent.ainvoke({"messages": [("user", question)]}, {"recursion_limit": 30})
        await updater.add_artifact(parts=[new_text_part(result["messages"][-1].content, media_type="text/plain")], name="answer")
        await updater.complete()  # the task ends: completed, with an artifact

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancellation is not supported.")


def app(card: AgentCard, executor: AgentExecutor) -> Starlette:  # the HTTP server: Agent Card + JSON-RPC endpoint
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card)
    return Starlette(routes=[*create_agent_card_routes(card), *create_jsonrpc_routes(handler, "/")])


if __name__ == "__main__":
    uvicorn.run(app(card, RegulationsExecutor()), host="localhost", port=PORT)
