"""
A minimal MCP gateway: ONE MCP server (over streamable HTTP) in front of several MCP servers (sub-processes, over stdio), which
- exposes their tools under prefixed names (e.g. applications_read_letter), avoiding name clashes across servers,
- exposes only the tools allowed for a given host (a profile, see PROFILES),
- logs every tool call, with arguments and results (as JSON lines, in COMMITTEE_OUTPUT/gateway-<profile>.jsonl),
- keeps secrets: only the gateway knows the API key, and passes it only to the server needing it,
- breaks the lethal trifecta: once a session has read untrusted Web content (fetch_*), it cannot send e-mails anymore.
Off-the-shelf gateways (e.g. Docker MCP Gateway, IBM's ContextForge) do the same, and much more: this one shows how they work.

Run with: poetry run python -m snippets -l agents -x 3 [--profile committee|ide] [--port 8000]   (then pick gateway.py)
then connect hosts to http://localhost:8000/mcp (e.g. agent_gateway.py, or the MCP Inspector).
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, VISION_MODEL (for the applications server), COMMITTEE_OUTPUT (default: output/).
The third-party fetch server (https://github.com/modelcontextprotocol/servers/tree/main/src/fetch) requires uv (https://docs.astral.sh/uv/).
"""
import argparse
import asyncio
import json
import os
import shutil
import sys
from contextlib import AsyncExitStack
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

SECRETS = {name: os.environ[name] for name in ["OPENAI_BASE_URL", "OPENAI_API_KEY", "VISION_MODEL"] if name in os.environ}
SETTINGS = {name: os.environ[name] for name in ["COMMITTEE_OUTPUT"] if name in os.environ}  # no secrets in here

# the servers behind the gateway: each one only gets the environment variables it needs
# (plus a few safe ones, e.g. PATH and HOME, which the MCP SDK always passes, while it never passes the gateway's whole environment)
SERVERS = {
    "applications": StdioServerParameters(command=sys.executable, args=["-m", "snippets.lecture_agents.exercise3.applications_mcp_server"],
                                          env=SETTINGS | SECRETS),
    "decisions": StdioServerParameters(command=sys.executable, args=["-m", "snippets.lecture_agents.exercise3.decisions_mcp_server"],
                                       env=SETTINGS),
    "fetch": StdioServerParameters(command="uvx", args=["mcp-server-fetch"], env=SETTINGS),  # third-party: reads Web pages
}

# which tools each host may see (patterns over prefixed names)
PROFILES = {
    "committee": ["applications_*", "decisions_*", "fetch_*"],  # the committee's own agent
    "ide": ["applications_list_candidates", "applications_read_letter", "fetch_*"],  # e.g. a developer's IDE: no costly or writing tools
}

UNTRUSTED = "fetch_*"  # tools bringing untrusted content into the session...
EXFILTRATING = "decisions_send_email"  # ... and tools communicating externally


class Gateway(FastMCP):
    """An MCP server whose tools are those of other MCP servers: listing and calling them is forwarded to the right server."""

    def __init__(self, routes: dict[str, tuple[ClientSession, Tool]], log_file: Path, **settings):
        super().__init__("committee-gateway", instructions="Tools of the admission committee, and Web access", **settings)
        self.routes = routes  # prefixed name -> (session with the server, tool as the server describes it)
        self.log_file = log_file
        self.tainted = set()  # IDs of the host sessions which read untrusted content

    async def list_tools(self) -> list[Tool]:  # overrides FastMCP's own tools/list handler
        return [tool.model_copy(update={"name": name}) for name, (_, tool) in self.routes.items()]

    async def call_tool(self, name: str, arguments: dict) -> CallToolResult:  # overrides FastMCP's own tools/call handler
        host_session = id(self.get_context().session)
        if name not in self.routes:
            result = error(f"Unknown tool: {name}")
        elif fnmatch(name, EXFILTRATING) and host_session in self.tainted:  # policy enforced HERE, whatever the host does
            result = error("Refused: this session has read untrusted Web content, so it cannot send e-mails anymore. Ask a human.")
        else:
            if fnmatch(name, UNTRUSTED):
                self.tainted.add(host_session)
            session, tool = self.routes[name]
            try:
                result = await session.call_tool(tool.name, arguments)  # forwarded with the original (non-prefixed) name
            except Exception as e:  # e.g. the server is down: the agent is told, and may recover
                result = error(f"Server unavailable: {e}")
        with self.log_file.open("a") as log:  # one JSON object per line, appended
            log.write(json.dumps(dict(time=datetime.now().isoformat(), session=host_session, tool=name, arguments=arguments,
                                      result=result.model_dump(mode="json", exclude_none=True))) + "\n")
        return result


def error(message: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=message)], isError=True)


async def main(profile: str, port: int):
    log_file = Path(os.environ.get("COMMITTEE_OUTPUT", "output")) / f"gateway-{profile}.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncExitStack() as stack:  # sessions with the servers stay open as long as the gateway runs
        routes = {}
        for server, parameters in SERVERS.items():
            if not shutil.which(parameters.command):
                print(f"# Skipping server {server}: {parameters.command} not found", file=sys.stderr)
                continue
            try:
                read, write = await stack.enter_async_context(stdio_client(parameters))  # launches the server as a sub-process
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                tools = (await session.list_tools()).tools
            except Exception as e:  # a server which does not start should not take the whole gateway down
                print(f"# Skipping server {server}: {e}", file=sys.stderr)
                continue
            for tool in tools:
                name = f"{server}_{tool.name}"  # prefixes avoid name clashes across servers
                if any(fnmatch(name, pattern) for pattern in PROFILES[profile]):
                    routes[name] = (session, tool)
        print(f"# Gateway for '{profile}' at http://localhost:{port}/mcp, exposing: {', '.join(routes)}", file=sys.stderr)
        await Gateway(routes, log_file, port=port).run_streamable_http_async()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A minimal MCP gateway for the admission committee")
    parser.add_argument("--profile", choices=PROFILES, default="committee", help="which host the gateway serves (default: committee)")
    parser.add_argument("--port", type=int, default=8000, help="port to listen on (default: 8000)")
    args = parser.parse_args()
    asyncio.run(main(args.profile, args.port))
