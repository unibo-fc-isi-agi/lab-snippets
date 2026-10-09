"""
A PreToolUse hook, in Claude Code's protocol (also understood by Codex): while the agent handles candidates' personal data,
it must not reach the network, nor write files outside output/. The harness runs it before each tool call.

Run with: echo '<event as JSON>' | poetry run python -m snippets -l skills -e 4
Register it as in settings.json (Claude Code: .claude/settings.json; Codex: .codex/hooks.json).
"""
import json
import re
import sys
from pathlib import Path

NETWORK_TOOLS = {"WebFetch", "WebSearch"}  # Claude Code's names of its Web tools
NETWORK_COMMANDS = re.compile(r"\b(curl|wget|nc|ncat|ssh|scp|rsync|ftp|telnet)\b|https?://")
WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}


def decide(event: dict) -> str | None:
    """The reason to deny the tool call described by the event, or None if there is no objection."""
    tool, args = event.get("tool_name", ""), event.get("tool_input", {})
    if tool in NETWORK_TOOLS or tool.startswith("mcp__"):  # MCP servers may reach the network too
        return f"{tool} is not allowed while handling candidates' personal data."
    if tool == "Bash" and NETWORK_COMMANDS.search(args.get("command", "")):
        return "Shell commands reaching the network are not allowed while handling candidates' personal data."
    if tool in WRITE_TOOLS:
        output = Path(event.get("cwd", ".")).resolve() / "output"
        target = Path(event.get("cwd", "."), args.get("file_path", "")).resolve()  # absolute paths stay absolute
        if not target.is_relative_to(output):
            return f"Files can only be written inside {output}, not {target}."
    return None


if __name__ == "__main__":
    event = json.load(sys.stdin)  # the harness sends the event as JSON on stdin
    if reason := decide(event):  # deny, and tell the LLM why (exit code 0 + JSON on stdout)
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }}))
    # no output: no objection (the usual permission rules apply)
