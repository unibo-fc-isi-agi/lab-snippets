"""
MCP server exposing the WRITE-ENABLED tools of Exercise 2 (decisions.py): launched by the gateway (gateway.py), as a sub-process.
It needs no API key: secrets go only where they are needed.

To run it alone (e.g. to inspect it with the MCP Inspector):
    poetry run python -m snippets.lecture_agents.exercise3.decisions_mcp_server
"""
from mcp.server.fastmcp import FastMCP
from snippets.lecture_agents.exercise2 import decisions

server = FastMCP("decisions", instructions="Actions on behalf of the admission committee: they have side effects, and need human approval")
for function in decisions.tools:
    server.tool()(function)

if __name__ == "__main__":
    server.run()
