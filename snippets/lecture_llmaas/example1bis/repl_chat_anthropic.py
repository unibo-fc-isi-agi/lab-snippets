"""
The same CLI chat of Example 1, via Anthropic's Messages API, served by a local Ollama instance.

Run with: poetry run python -m snippets -l llmaas -e 1bis
(with `ollama serve` active, and `gemma4:e2b` pulled via `ollama pull gemma4:e2b`)
Configure via env vars: ANTHROPIC_BASE_URL, ANTHROPIC_API_KEY, ANTHROPIC_MODEL.
"""
import os
from anthropic import Anthropic

# defaults point to Ollama, which exposes an Anthropic-compatible API too (the API key is ignored)
base_url = os.environ.get("ANTHROPIC_BASE_URL", "http://localhost:11434")
api_key = os.environ.get("ANTHROPIC_API_KEY", "ollama")
model = os.environ.get("ANTHROPIC_MODEL", "gemma4:e2b")

client = Anthropic(base_url=base_url, api_key=api_key)
system = "Just chat with the user, be friendly and helpful. Do not think. Just answer."  # NOT part of the messages
messages = []

print(f"Using model: {model}")
print("Type '/exit' or '/quit' to stop. Type '/retry' to retry the last message.")

try:
    while True:
        user_text = input("you> ").strip()
        if not user_text:
            continue
        if (cmd := user_text.lower().strip()) in {"/exit", "/quit"}:
            break
        if cmd not in {"/retry"}:
            messages.append(dict(role="user", content=user_text))
        try:  # max_tokens is mandatory here (raise it for reasoning models: thinking tokens count too)
            response = client.messages.create(model=model, system=system, messages=messages, max_tokens=2048)
        except Exception as exc:
            print(f"error> {exc}")
            continue
        # no "choices": the response content is a list of typed blocks (text, thinking, tool_use, ...)
        answer = "".join(block.text for block in response.content if block.type == "text")
        print(f"assistant> {answer}")
        messages.append(dict(role="assistant", content=answer))
except (EOFError, KeyboardInterrupt):
    pass
print("Goodbye!")
