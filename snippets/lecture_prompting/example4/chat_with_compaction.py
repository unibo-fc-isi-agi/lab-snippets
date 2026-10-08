"""
A CLI chat which keeps its context within a token budget, by summarising ("compacting") the older messages.

Run with: poetry run python -m snippets -l prompting -e 4
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL, CONTEXT_BUDGET (in tokens).
"""
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately

base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/")
api_key = os.environ.get("OPENAI_API_KEY") or input(f"Enter your API key for {base_url}: ")
model = os.environ.get("OPENAI_MODEL", "openrouter/auto")
budget = int(os.environ.get("CONTEXT_BUDGET", "1000"))  # tokens; tiny on purpose, to see compaction happen
keep_last = 4  # most recent messages, always kept verbatim

llm = ChatOpenAI(base_url=base_url, api_key=api_key, model=model)

instructions = "You are a helpful assistant. Be concise."
summary = ""  # summary of the older part of the conversation (initially empty)
history: list[BaseMessage] = []


def context() -> list[BaseMessage]:
    # what is actually sent to the LLM: instructions (+ summary, if any), then the recent history
    system = instructions + (f"\n\nSummary of the earlier conversation:\n{summary}" if summary else "")
    return [SystemMessage(system), *history]


def compact() -> None:
    global summary, history
    older, history = history[:-keep_last], history[-keep_last:]
    # the LLM itself summarises the older messages (and the previous summary, which is in the system message)
    request = [*context()[:1], *older, HumanMessage("Summarise the conversation so far in a few sentences. "
                                                    "Keep all facts, names, numbers, and decisions.")]
    summary = llm.invoke(request).content


while True:
    try:
        history.append(HumanMessage(input("You: ")))
    except (EOFError, KeyboardInterrupt):
        break
    response = llm.invoke(context())
    history.append(response)
    print(f"AI: {response.content}")
    usage = response.usage_metadata
    cached = usage.get("input_token_details", {}).get("cache_read", 0)  # prompt caching: a stable prefix is cheaper
    print(f"    [input: {usage['input_tokens']} tokens, of which {cached} cached | output: {usage['output_tokens']} tokens]")
    if count_tokens_approximately(context()) > budget and len(history) > keep_last:  # over budget: compact
        compact()
        print(f"    [context compacted: now ~{count_tokens_approximately(context())} tokens]")
