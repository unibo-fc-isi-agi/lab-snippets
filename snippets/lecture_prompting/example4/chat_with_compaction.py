"""
A CLI chat which keeps its context within a token budget, by summarising ("compacting") the older messages.

Run with: poetry run python -m snippets -l prompting -e 4
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL, CONTEXT_BUDGET (in tokens), SYSTEM_PROMPT.
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately

base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/")
api_key = os.environ.get("OPENAI_API_KEY") or input(f"Enter your API key for {base_url}: ")
model = os.environ.get("OPENAI_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
budget = int(os.environ.get("CONTEXT_BUDGET", "1000"))  # tokens; tiny on purpose, to see compaction happen
keep_last = 4  # most recent messages, always kept verbatim

llm = ChatOpenAI(base_url=base_url, api_key=api_key, model=model)

instructions = os.environ.get("SYSTEM_PROMPT", "You are a helpful assistant. Be concise.")
summary = ""  # summary of the older part of the conversation (initially empty)
history: list[BaseMessage] = []


def prefix() -> list[BaseMessage]:
    # the fixed instructions first (a stable prefix, hence cacheable), then the summary (if any)
    return [SystemMessage(instructions)] + ([SystemMessage(f"Summary of the earlier conversation:\n{summary}")] if summary else [])


def context() -> list[BaseMessage]:
    return [*prefix(), *history]  # what is actually sent to the LLM


def compact() -> None:
    global summary, history
    older, history = history[:-keep_last], history[-keep_last:]
    # the LLM itself summarises the older messages (and the previous summary, if any)
    request = [*prefix(), *older, HumanMessage("Summarise the conversation so far in a few sentences. "
                                               "Keep all facts, names, numbers, and decisions.")]
    summary = llm.invoke(request).content


while True:
    try:
        history.append(HumanMessage(input("You: ")))
    except (EOFError, KeyboardInterrupt):
        break
    if count_tokens_approximately(context()) > budget and len(history) > keep_last:  # over budget: compact, then send
        compact()
        print(f"    [context compacted: now ~{count_tokens_approximately(context())} tokens]")
    response = llm.invoke(context())
    history.append(response)
    print(f"AI: {response.content}")
    usage = response.usage_metadata
    cached = usage.get("input_token_details", {}).get("cache_read", 0)  # prompt caching: a stable prefix is cheaper
    print(f"    [input: {usage['input_tokens']} tokens, of which {cached} cached | output: {usage['output_tokens']} tokens]")
