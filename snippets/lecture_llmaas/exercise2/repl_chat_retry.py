"""
The Sync CLI Chat of Example 1, plus automatic retries (with exponential backoff) of requests failing for transient reasons.

Run with: poetry run python -m snippets -l llmaas -x 2 [--retries N] [--initial-delay SECONDS] [--backoff FACTOR]
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL, plus LLM_RETRIES, LLM_INITIAL_DELAY, LLM_BACKOFF
(command-line arguments win over env vars, which win over defaults).
"""
import argparse
import os
import time
import openai
from openai import OpenAI

# decision 1 (how to detect a failure?): the client raises exceptions; only some of them are TRANSIENT, i.e. worth a retry
TRANSIENT_ERRORS = (
    openai.APIConnectionError,  # network errors (including timeouts)
    openai.RateLimitError,  # HTTP 429: too many requests
    openai.InternalServerError,  # HTTP 5xx: the provider is in trouble
)  # others (e.g. openai.AuthenticationError, openai.BadRequestError) would fail again: retrying them only wastes time


# decision 2 (how to retry?): a plain loop + try/except, in a helper function wrapping ANY call (no extra library needed)
def with_retries(call, retries: int = 3, initial_delay: float = 1.0, backoff: float = 2.0, sleep=time.sleep):
    """Calls call(), retrying it up to `retries` times upon transient errors, waiting
    initial_delay, initial_delay * backoff, initial_delay * backoff^2, ... seconds between attempts."""
    for attempt in range(retries + 1):  # 1 attempt + `retries` retries
        try:
            return call()
        except TRANSIENT_ERRORS as error:
            if attempt == retries:  # no more retries: give up, letting the caller handle the error
                raise
            delay = initial_delay * backoff ** attempt  # decision 3: exponential backoff, e.g. 1s, 2s, 4s, 8s, ...
            print(f"# Attempt {attempt + 1} failed ({type(error).__name__}), retrying in {delay:g}s...")
            sleep(delay)  # `sleep` is a parameter, so that tests can replace it, and not actually wait


# decision 4 (how to configure?): argparse for command-line arguments, whose defaults come from env vars, whose defaults are hard-coded
def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync CLI Chat with retries and exponential backoff")
    parser.add_argument("--retries", type=int, default=int(os.environ.get("LLM_RETRIES", 3)), help="max number of retries (default: 3)")
    parser.add_argument("--initial-delay", type=float, default=float(os.environ.get("LLM_INITIAL_DELAY", 1.0)), help="delay before the 1st retry, in seconds (default: 1)")
    parser.add_argument("--backoff", type=float, default=float(os.environ.get("LLM_BACKOFF", 2.0)), help="multiplier of the delay, at each retry (default: 2)")
    return parser.parse_args(args)


if __name__ == "__main__":
    config = parse_args()
    base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/")
    api_key = os.environ.get("OPENAI_API_KEY") or input(f"Enter your API key for {base_url}: ")
    model = os.environ.get("OPENAI_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

    # beware: the OpenAI client ALREADY retries twice, by default (with backoff), so we disable that, to be in control
    client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
    messages = [dict(role="system", content="Just chat with the user, be friendly and helpful. Do not think. Just answer.")]

    print(f"Using model: {model}, with up to {config.retries} retries (initial delay: {config.initial_delay}s, backoff: x{config.backoff})")
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
            try:  # decision 5 (how to restructure?): the call is wrapped in a lambda, the rest of the REPL is unchanged
                response = with_retries(lambda: client.chat.completions.create(model=model, messages=messages),
                                        config.retries, config.initial_delay, config.backoff)
            except Exception as exc:  # non-transient errors, or transient ones persisting after all retries
                print(f"error> {exc}")
                continue
            answer = response.choices[0].message.content or ""
            print(f"assistant> {answer}")
            messages.append(dict(role="assistant", content=answer))
    except (EOFError, KeyboardInterrupt):
        pass
    print("Goodbye!")
