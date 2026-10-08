from types import SimpleNamespace
import httpx
import openai
import pytest
from snippets.lecture_llmaas.exercise1 import repl_chat_cached
from snippets.lecture_llmaas.exercise2.repl_chat_retry import parse_args, with_retries


class FakeClient:  # mimics client.chat.completions.create, counting calls
    base_url = "https://example.com/v1/"

    def __init__(self):
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=f"answer #{self.calls}"))])


def test_cache_key_ignores_order_of_parameters():
    assert repl_chat_cached.cache_key(model="m", messages=[]) == repl_chat_cached.cache_key(messages=[], model="m")
    assert repl_chat_cached.cache_key(model="m", messages=[]) != repl_chat_cached.cache_key(model="n", messages=[])


def test_same_request_hits_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(repl_chat_cached, "CACHE_DIR", tmp_path)
    client, messages = FakeClient(), [dict(role="user", content="Hi")]
    first = repl_chat_cached.cached_completion(client, model="m", messages=messages)
    assert repl_chat_cached.cached_completion(client, model="m", messages=messages) == first
    assert client.calls == 1
    repl_chat_cached.cached_completion(client, model="m", messages=[dict(role="user", content="Hi!")])  # different history: miss
    assert client.calls == 2


def transient():
    return openai.APIConnectionError(request=httpx.Request("POST", "https://example.com"))


def failing(times: int, error=transient):
    attempts = []
    def call():
        attempts.append(1)
        if len(attempts) <= times:
            raise error()
        return "ok"
    return call, attempts


def test_retries_with_exponential_backoff():
    call, attempts = failing(3)
    delays = []
    assert with_retries(call, retries=3, initial_delay=1, backoff=2, sleep=delays.append) == "ok"
    assert len(attempts) == 4 and delays == [1, 2, 4]


def test_gives_up_after_max_retries():
    call, attempts = failing(5)
    with pytest.raises(openai.APIConnectionError):
        with_retries(call, retries=2, sleep=lambda _: None)
    assert len(attempts) == 3


def test_does_not_retry_non_transient_errors():
    call, attempts = failing(1, error=lambda: ValueError("bad request"))
    with pytest.raises(ValueError):
        with_retries(call, retries=3, sleep=lambda _: None)
    assert len(attempts) == 1


def test_configuration(monkeypatch):
    monkeypatch.setenv("LLM_RETRIES", "7")
    config = parse_args(["--backoff", "3"])
    assert (config.retries, config.initial_delay, config.backoff) == (7, 1.0, 3.0)  # env var, default, command line
