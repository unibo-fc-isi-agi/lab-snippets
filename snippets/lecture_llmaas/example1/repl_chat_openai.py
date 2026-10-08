"""
A synchronous CLI chat (REPL) with an LLM, via OpenAI's Chat Completions API.

Run with: poetry run python -m snippets -l llmaas -e 1
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (any OpenAI-compatible provider works).
"""
import os
from openai import OpenAI

# connection parameters: env vars, with defaults (the API key is asked interactively, if missing)
base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/")
api_key = os.environ.get("OPENAI_API_KEY") or input(f"Enter your API key for {base_url}: ")
model = os.environ.get("OPENAI_MODEL", "openrouter/auto")

client = OpenAI(base_url=base_url, api_key=api_key)  # stateless: the conversation history is kept by us
messages = [dict(role="system", content="Just chat with the user, be friendly and helpful. Do not think. Just answer.")]

print(f"Using model: {model}")
print("Type '/exit' or '/quit' to stop. Type '/retry' to retry the last message.")

try:  # Ctrl+C (KeyboardInterrupt) or Ctrl+D (EOFError) exit gracefully
    while True:
        user_text = input("you> ").strip()
        if not user_text:
            continue
        if (cmd := user_text.lower().strip()) in {"/exit", "/quit"}:  # sub-commands first
            break
        if cmd not in {"/retry"}:  # on /retry, the history already ends with the last user message
            messages.append(dict(role="user", content=user_text))
        try:  # the WHOLE history is sent at each request
            response = client.chat.completions.create(model=model, messages=messages)
        except Exception as exc:  # e.g. network errors, rate limits, invalid API key
            print(f"error> {exc}")
            continue
        answer = response.choices[0].message.content or ""  # one choice expected, by default
        print(f"assistant> {answer}")
        messages.append(dict(role="assistant", content=answer))  # the answer becomes part of the history
except (EOFError, KeyboardInterrupt):
    pass
print("Goodbye!")
