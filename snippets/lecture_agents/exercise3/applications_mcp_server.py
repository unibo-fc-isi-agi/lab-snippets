"""
MCP server exposing the READ-ONLY tools of Exercise 1 (committee.py): launched by the gateway (gateway.py), as a sub-process.
It is the only server needing an API key (to extract information from pictures), which it receives from the gateway.

To run it alone (e.g. to inspect it with the MCP Inspector):
    poetry run python -m snippets.lecture_agents.exercise3.applications_mcp_server
"""
from mcp.server.fastmcp import FastMCP
from snippets.lecture_agents.exercise1 import committee
from snippets.lecture_agents.exercise2 import decisions  # noqa: F401 (it adds the 4th candidate, with the malicious letter)

server = FastMCP("applications", instructions="Read-only access to the applications of the candidates to a PhD programme")
for function in committee.tools:
    server.tool()(function)  # as in Example 2

if __name__ == "__main__":
    server.run()
