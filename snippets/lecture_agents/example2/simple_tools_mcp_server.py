"""
An MCP server exposing the simple tools of this lecture, so that any MCP host (e.g. the agent in agent_mcp.py) can use them.

It is launched by agent_mcp.py as a sub-process; to run it alone (e.g. to inspect it with the MCP Inspector):
    poetry run python -m snippets.lecture_agents.example2.simple_tools_mcp_server
"""
from mcp.server.fastmcp import FastMCP
from snippets.lecture_agents.simple_tools import tools

server = FastMCP("simple-tools", instructions="Tools to get the current time, the weather, and Web search results")
for function in tools:
    server.tool()(function)  # name, description, and input schema are derived from the function

if __name__ == "__main__":
    server.run()  # over stdio, by default; use server.run("streamable-http") to serve over HTTP
