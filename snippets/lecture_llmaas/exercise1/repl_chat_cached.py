"""
The Sync CLI Chat of Example 1, plus a file-system cache of Chat Completion requests:
a request identical to a past one (same model, parameters, and WHOLE history) is answered from the cache, without calling the model.

Run with: poetry run python -m snippets -l llmaas -x 1
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL, plus LLM_CACHE_DIR (default: .llm-cache/).
"""
import hashlib
import json
import os
from pathlib import Path
from openai import OpenAI

base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/")
api_key = os.environ.get("OPENAI_API_KEY") or input(f"Enter your API key for {base_url}: ")
model = os.environ.get("OPENAI_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

client = OpenAI(base_url=base_url, api_key=api_key)

# decision 1 (where?): a local, untracked folder (cf. .gitignore), configurable via env var
CACHE_DIR = Path(os.environ.get("LLM_CACHE_DIR", ".llm-cache"))


# decision 2 (when is it a hit?): same request = same provider, model, and messages (i.e. the whole history, not just the last message),
# plus any other parameter (e.g. temperature): all of them affect the answer, so all of them are part of the key
def cache_key(**request) -> str:
    canonical = json.dumps(request, sort_keys=True, ensure_ascii=False)  # same request => same string (keys are sorted)
    return hashlib.sha256(canonical.encode()).hexdigest()  # a fixed-length, file-name-friendly digest of that string


# decision 3 (how to store?): one JSON file per request, named after its key, so that lookup is just a file-existence check
def cached_completion(client: OpenAI, **request) -> str:
    """Like client.chat.completions.create(**request), but returns just the answer's text, from the cache if possible."""
    file = CACHE_DIR / f"{cache_key(base_url=str(client.base_url), **request)}.json"
    if file.exists():
        print("# Cache hit! Returning cached response.")
        return json.loads(file.read_text())["answer"]
    response = client.chat.completions.create(**request)  # cache miss: actually call the model
    answer = response.choices[0].message.content or ""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # the request is stored too, for debugging (and reproducibility): one can see what produced the answer
    file.write_text(json.dumps(dict(request=request, answer=answer), indent=2, ensure_ascii=False))
    return answer


# decision 4 (how to restructure?): the REPL is the same as in Example 1, except for the line calling the model
if __name__ == "__main__":
    messages = [dict(role="system", content="Just chat with the user, be friendly and helpful. Do not think. Just answer.")]

    print(f"Using model: {model}, caching in: {CACHE_DIR.absolute()}")
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
            try:
                answer = cached_completion(client, model=model, messages=messages)  # was: client.chat.completions.create(...)
            except Exception as exc:
                print(f"error> {exc}")
                continue
            print(f"assistant> {answer}")
            messages.append(dict(role="assistant", content=answer))
    except (EOFError, KeyboardInterrupt):
        pass
    print("Goodbye!")
