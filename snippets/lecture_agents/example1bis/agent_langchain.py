"""
The same agent of Example 1, with LangChain: create_agent implements the ReAct loop for us.

Run with: poetry run python -m snippets -l agents -e 1bis
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (must support tool calling).
"""
import os
from langchain.agents import create_agent
from langchain.agents.middleware import ToolErrorMiddleware
from langgraph.errors import GraphRecursionError
from langchain_openai import ChatOpenAI
from snippets.lecture_agents.simple_tools import instructions, tools

base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/")
api_key = os.environ.get("OPENAI_API_KEY") or input(f"Enter your API key for {base_url}: ")
model = os.environ.get("OPENAI_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

llm = ChatOpenAI(base_url=base_url, api_key=api_key, model=model)

# errors raised by tools (e.g. an unknown time zone) go back to the LLM as error messages, instead of crashing the run
tool_errors = ToolErrorMiddleware(lambda error, request: f"Error: {error}" if isinstance(error, ValueError) else None)

agent = create_agent(llm, tools=tools, system_prompt=instructions, middleware=[tool_errors])  # tools derived from functions


def print_tool_calls(messages) -> None:  # shows which tools the agent called, and how
    for message in messages:
        for tool_call in getattr(message, "tool_calls", []):
            print(f"    [tool] {tool_call['name']}({tool_call['args']})")


if __name__ == "__main__":
    messages = []
    while True:
        try:
            messages.append(("user", input("You: ")))
        except (EOFError, KeyboardInterrupt):
            break
        try:
            result = agent.invoke({"messages": messages}, {"recursion_limit": 20})  # 20 graph steps ≈ 10 model+tools rounds
        except GraphRecursionError:
            print("AI: step limit reached, sorry")
            continue
        print_tool_calls(result["messages"][len(messages):])  # only the new messages
        messages = result["messages"]  # the whole history, including tool calls and results
        print(f"AI: {messages[-1].content}")
