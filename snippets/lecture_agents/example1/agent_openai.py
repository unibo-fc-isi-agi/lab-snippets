"""
An agent from scratch: a ReAct loop over OpenAI's Chat Completions API, using the simple tools of this lecture.

Run with: poetry run python -m snippets -l agents -e 1
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (must support tool calling).
"""
import json
import os
from openai import OpenAI
from pydantic import TypeAdapter
from snippets.lecture_agents.simple_tools import instructions, tools

base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/")
api_key = os.environ.get("OPENAI_API_KEY") or input(f"Enter your API key for {base_url}: ")
model = os.environ.get("OPENAI_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

client = OpenAI(base_url=base_url, api_key=api_key)


def tool_definition(function) -> dict:  # name, docstring, and JSON Schema of the parameters
    return dict(type="function", function=dict(
        name=function.__name__,
        description=function.__doc__,
        parameters=TypeAdapter(function).json_schema(),
    ))


tool_definitions = [tool_definition(f) for f in tools]  # sent to the LLM at each request
tools_by_name = {f.__name__: f for f in tools}  # used to execute the calls requested by the LLM


def call_tool(tool_call) -> str:
    try:
        function = tools_by_name[tool_call.function.name]
        arguments = json.loads(tool_call.function.arguments or "{}")  # arguments come as a JSON string
        return json.dumps(function(**arguments))
    except Exception as e:  # errors are fed back to the LLM, which may recover
        return f"Error: {e}"


def react(messages: list, max_steps: int = 10) -> str:
    for _ in range(max_steps):  # bounded: the LLM may loop forever
        response = client.chat.completions.create(model=model, messages=messages, tools=tool_definitions)
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))  # the tool calls are part of the history too
        if not message.tool_calls:  # no more actions: this is the final answer
            return message.content
        for tool_call in message.tool_calls:  # act, then observe: results go back to the LLM as "tool" messages
            print(f"    [tool] {tool_call.function.name}({tool_call.function.arguments})")
            messages.append(dict(role="tool", tool_call_id=tool_call.id, content=call_tool(tool_call)))
    return f"Sorry, I could not answer in {max_steps} steps."


if __name__ == "__main__":
    messages = [dict(role="system", content=instructions)]
    while True:
        try:
            messages.append(dict(role="user", content=input("You: ")))
        except (EOFError, KeyboardInterrupt):
            break
        print(f"AI: {react(messages)}")
