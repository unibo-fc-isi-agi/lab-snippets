"""
The agent of Example 2, via LangChain's Deep Agents (https://docs.langchain.com/oss/python/deepagents/skills), which supports skills natively.

Run with: poetry run python -m snippets -l skills -e 2bis [SKILLS_FOLDER ...]   (default: the skills of example 1)
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (must support tool calling).
"""
import json
import sys
import uuid
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from snippets.lecture_agents.example1bis.agent_langchain import llm  # a ChatOpenAI, configured via the OPENAI_* env vars
from snippets.lecture_skills.example1 import FOLDER

ROOT = FOLDER.parents[2]  # i.e. the lab-snippets folder: the agent can read (and run) anything inside it

agent = create_deep_agent(
    model=llm,
    system_prompt="You are the assistant of the admission committee of a PhD programme.",
    # files on disk + a shell (the `execute` tool): virtual_mode=False, so that paths are the same for files and shell
    backend=LocalShellBackend(root_dir=ROOT, virtual_mode=False, inherit_env=True),
    skills=[str(folder) for folder in (sys.argv[1:] if __name__ == "__main__" else []) or [FOLDER]],  # folders of skills
    interrupt_on={"execute": True},  # no sandbox: every shell command must be approved by a human
    checkpointer=InMemorySaver(),  # needed to resume after interrupts
)


def review_in_terminal(action: dict) -> dict:
    print(f"    [approval required] {action['name']}({json.dumps(action['args'], ensure_ascii=False)})")
    choice = input("    [a]pprove or [r]eject? ").strip().lower()
    return {"type": "approve"} if choice == "a" else {"type": "reject", "message": "The user rejected this command."}


def run(question: str, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}
    result = agent.invoke({"messages": [("user", question)]}, config)
    while "__interrupt__" in result:  # the agent stopped, waiting for approval of some shell commands
        request = result["__interrupt__"][0].value
        decisions = [review_in_terminal(action) for action in request["action_requests"]]
        result = agent.invoke(Command(resume={"decisions": decisions}), config)
    for message in result["messages"]:  # the trajectory: which files were read, which commands were run
        for call in getattr(message, "tool_calls", []):
            print(f"    [tool] {call['name']}({json.dumps(call['args'], ensure_ascii=False)[:120]})")
    return result["messages"][-1].content


if __name__ == "__main__":
    thread_id = str(uuid.uuid4())
    while True:
        try:
            question = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        print(f"AI: {run(question, thread_id)}")
