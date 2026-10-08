"""
The committee's agent of Exercise 2, whose tools ALL come from the MCP gateway (gateway.py), via a single streamable-HTTP connection.
Human approval stays in the host (the gateway cannot ask humans), for every tool coming from the decisions server.

Run with (after starting the gateway, in another terminal):
    poetry run python -m snippets -l agents -x 3   (then pick agent_gateway.py)
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (must support tool calling), GATEWAY_URL (default: http://localhost:8000/mcp).
No VISION_MODEL here: extraction from pictures happens behind the gateway, which holds the secrets it needs.
"""
import asyncio
import os
import uuid
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import InMemorySaver
from snippets.lecture_agents.example1bis.agent_langchain import llm, print_tool_calls
from snippets.lecture_agents.simple_tools import get_current_time
from snippets.lecture_agents.exercise2.agent_decisions import instructions, run

mcp_client = MultiServerMCPClient({
    "gateway": dict(transport="streamable_http", url=os.environ.get("GATEWAY_URL", "http://localhost:8000/mcp")),
})

instructions = instructions.replace("(letters, passports, transcripts)", "(letters, passports, transcripts), or Web pages (fetch)")


def create_gateway_agent(tools):
    """The agent of Exercise 2, with the gateway's tools (plus the local get_current_time)."""
    write_enabled = [tool.name for tool in tools if tool.name.startswith("decisions_")]  # names are prefixed by the gateway
    return create_agent(
        llm,
        tools=[*tools, get_current_time],
        system_prompt=instructions,
        middleware=[HumanInTheLoopMiddleware(interrupt_on={name: True for name in write_enabled})],
        checkpointer=InMemorySaver(),
    )


async def main():
    # ONE session for the whole conversation (rather than one per tool call): so the gateway can tell this host's calls apart
    async with mcp_client.session("gateway") as session:
        tools = await load_mcp_tools(session)
        print(f"Tools: {[tool.name for tool in tools]}")
        agent = create_gateway_agent(tools)
        thread_id = str(uuid.uuid4())
        while True:
            try:
                question = input("You: ")
            except (EOFError, KeyboardInterrupt):
                break
            messages = await run(agent, question, thread_id)  # same loop as in Exercise 2, approvals included
            print_tool_calls(messages)
            print(f"AI: {messages[-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
