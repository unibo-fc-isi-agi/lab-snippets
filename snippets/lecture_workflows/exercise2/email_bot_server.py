"""
The e-mail bot of Exercise 1, as an A2A agent: one A2A TASK per e-mail (sent as a data Part), one LangGraph THREAD per task.
The workflow's steps become task status updates; the approval is asked OUT OF BAND, to the admission office (on this server's
terminal), not to the client: the client belongs to another organisation, so it cannot approve the office's replies.
Outcomes: completed (with the reply as an artifact), or rejected (no reply: the office rejected it, or the e-mail was escalated).

Run with: poetry run python -m snippets -l workflows -x 2   (then pick email_bot_server.py; stop it with Ctrl+C)
then run helpdesk_client.py in another terminal, and approve the replies here.
Configure via env vars: as for Exercise 1, plus EMAIL_BOT_PORT (default: 9998).
"""
import asyncio
import os
import uvicorn
from a2a.helpers import get_data_parts, new_task_from_user_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, TaskState
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from snippets.lecture_workflows import admission
from snippets.lecture_workflows.example4bis.regulations_server import app
from snippets.lecture_workflows.exercise1 import bot

PORT = int(os.environ.get("EMAIL_BOT_PORT", "9998"))

card = AgentCard(  # reveals WHAT the agent does, not HOW (no prompts, models, tools, or data sources)
    name="PhD Admission E-mail Agent",
    description="Answers the e-mails of candidates to the PhD programme in Computer Science and Engineering. "
                "Replies are approved by the admission office, so tasks may take hours.",
    version="1.0.0",
    default_input_modes=["application/json"],
    default_output_modes=["text/plain"],
    capabilities=AgentCapabilities(streaming=True),
    supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=f"http://localhost:{PORT}", protocol_version="1.0")],
    skills=[AgentSkill(
        id="answer-candidate-email",
        name="Answer a candidate's e-mail",
        description='Input: a data part {"sender": ..., "subject": ..., "body": ...}. Output: the approved reply, as a text artifact.',
        tags=["phd", "admission", "e-mail"],
        examples=['{"sender": "mario.rossi@example.it", "subject": "Age limit?", "body": "Is there an age limit to apply?"}'],
    )],
)


class EmailBotExecutor(AgentExecutor):
    def __init__(self, approve=bot.ask_human):  # how the office is asked to approve (tests: a simulated human)
        self.approve = approve

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task_from_user_message(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        def say(text: str):  # a message of the agent, within this task
            return updater.new_agent_message([new_text_part(text)])

        emails = get_data_parts(context.message.parts)
        if not emails or not all(isinstance(emails[0].get(key), str) for key in ("sender", "subject", "body")):
            return await updater.reject(say('Expected a data part {"sender": ..., "subject": ..., "body": ...}.'))  # validate input
        email = {"id": task.id, **{key: emails[0][key] for key in ("sender", "subject", "body")}}

        config, input = {"configurable": {"thread_id": task.id}}, {"email": email}  # task <-> thread
        async with AsyncSqliteSaver.from_conn_string(str(admission.OUTPUT_DIR / "email-bot-a2a.db")) as checkpointer:
            workflow = bot.build(checkpointer)
            while True:
                async for event in workflow.astream(input, config, stream_mode="updates"):  # steps -> status updates
                    for node in event.keys() - {"__interrupt__"}:
                        await updater.update_status(TaskState.TASK_STATE_WORKING, message=say(f"Step done: {node}"))
                pending = (await workflow.aget_state(config)).interrupts
                if not pending:
                    break
                await updater.update_status(TaskState.TASK_STATE_WORKING, message=say("Waiting for the approval of the admission office"))
                input = Command(resume=await asyncio.to_thread(self.approve, pending[0].value))  # out of band, possibly for hours
            final = (await workflow.aget_state(config)).values

        if final.get("sent"):
            await updater.add_artifact([new_text_part(final["drafts"][-1], media_type="text/plain")], name="reply")
            await updater.complete()
        elif final["route"] == "suspicious":
            await updater.reject(say("This e-mail was forwarded to a human officer."))  # no details: the client may be the attacker
        else:
            await updater.reject(say("The admission office decided not to reply automatically."))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancellation is not supported.")


if __name__ == "__main__":
    admission.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app(card, EmailBotExecutor()), host="localhost", port=PORT)  # same HTTP server as Example 4 bis
