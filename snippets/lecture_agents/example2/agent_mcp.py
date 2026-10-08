"""
The same agent of Example 1 (bis), whose tools come from an MCP server (simple_tools_mcp_server.py) rather than from local functions.

Run with: poetry run python -m snippets -l agents -e 2   (then pick agent_mcp.py)
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (must support tool calling).
"""
import asyncio
import sys
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from snippets.lecture_agents.simple_tools import instructions
from snippets.lecture_agents.example1bis.agent_langchain import llm, print_tool_calls

mcp_client = MultiServerMCPClient({
    "simple-tools": dict(
        transport="stdio",  # the server is a sub-process, talking via stdin/stdout
        command=sys.executable,  # i.e. the same Python interpreter running this script
        args=["-m", "snippets.lecture_agents.example2.simple_tools_mcp_server"],
    ),
    # more servers here, e.g. "fetch": dict(transport="stdio", command="uvx", args=["mcp-server-fetch"]),
})


async def main():
    tools = await mcp_client.get_tools()  # MCP: tools/list
    print(f"Tools: {[t.name for t in tools]}")
    agent = create_agent(llm, tools=tools, system_prompt=instructions)  # tool calls become MCP tools/call requests
    messages = []
    while True:
        try:
            messages.append(("user", input("You: ")))
        except (EOFError, KeyboardInterrupt):
            break
        result = await agent.ainvoke({"messages": messages}, {"recursion_limit": 20})  # async: MCP clients are async
        print_tool_calls(result["messages"][len(messages):])
        messages = result["messages"]
        print(f"AI: {messages[-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
