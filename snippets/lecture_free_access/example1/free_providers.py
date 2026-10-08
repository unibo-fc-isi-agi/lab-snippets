"""
One client, many free providers: they all expose an OpenAI-compatible API, so only base URL, key, and model change.

Run with:
    poetry run python -m snippets -l free-access -e 1 groq                              # list the models available now
    poetry run python -m snippets -l free-access -e 1 groq openai/gpt-oss-20b "Hello!"  # one chat request
"""
import os
import sys
from openai import OpenAI

# provider -> (OpenAI-compatible base URL, env var holding the API key)
PROVIDERS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "github": ("https://models.github.ai/inference", "GITHUB_TOKEN"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),  # local, no key needed
    "vllm": ("http://localhost:8000/v1", None),  # local, no key needed
}

if len(sys.argv) < 2 or sys.argv[1] not in PROVIDERS:
    sys.exit(f"usage: {sys.argv[0]} {{{'|'.join(PROVIDERS)}}} [model] [prompt]")

provider = sys.argv[1]
base_url, key_var = PROVIDERS[provider]
api_key = os.environ.get(key_var) if key_var else "unused"  # the client requires a key, even if the server ignores it
if not api_key:
    api_key = input(f"Enter your API key for {provider} (or set {key_var}): ")
client = OpenAI(base_url=base_url, api_key=api_key)

if len(sys.argv) < 3:  # no model given: list the models currently available (model IDs go stale quickly)
    for m in client.models.list():
        print(m.id)
    sys.exit(0)

model = sys.argv[2]
prompt = sys.argv[3] if len(sys.argv) > 3 else "Say hello in one sentence."
response = client.chat.completions.create(model=model, messages=[dict(role="user", content=prompt)])
print(response.choices[0].message.content)
