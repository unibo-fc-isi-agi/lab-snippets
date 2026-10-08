"""
Asks a reasoning model to rank the candidates by their letters, with low, medium, and high reasoning effort,
measuring latency and (reasoning) tokens for each effort level.

Run with: poetry run python -m snippets -l prompting -e 3 [LETTER ...]   (all letters, by default)
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (must be a reasoning model).
"""
import os
import time
from langchain_openai import ChatOpenAI

base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/")
api_key = os.environ.get("OPENAI_API_KEY") or input(f"Enter your API key for {base_url}: ")
model = os.environ.get("OPENAI_MODEL", "openai/gpt-oss-20b")  # must be a reasoning model

question = """Here are the presentation letters of three candidates to a PhD programme.
Rank the candidates from the most to the least recommended, in one line per candidate, with a one-sentence justification.

{letters}"""


def ask(prompt: str, effort: str) -> None:
    # OpenRouter's unified syntax; on OpenAI's own API use ChatOpenAI(..., reasoning_effort=effort) instead
    llm = ChatOpenAI(base_url=base_url, api_key=api_key, model=model, extra_body={"reasoning": {"effort": effort}})
    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start
    usage = response.usage_metadata  # token counts, as reported by the provider
    reasoning_tokens = usage.get("output_token_details", {}).get("reasoning", 0)  # hidden "thinking" tokens: paid, not shown
    print(f"## effort={effort}: {elapsed:.1f}s, {usage['output_tokens']} output tokens, of which {reasoning_tokens} for reasoning")
    print(response.content, end="\n\n")


if __name__ == "__main__":
    import sys
    import data

    paths = [data.find_letter(arg) for arg in sys.argv[1:]] or data.letters()
    letters = "\n\n".join(f"<letter>\n{path.read_text()}\n</letter>" for path in paths)
    print(f"Using model: {model}")
    for effort in ["low", "medium", "high"]:  # same question, increasing effort
        ask(question.format(letters=letters), effort)
