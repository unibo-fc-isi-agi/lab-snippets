from snippets.lecture_governance.exercise1.compare_models import client_for, markdown, summarise


def test_client_for():
    client, name = client_for("llama3.2@http://localhost:11434/v1")
    assert name == "llama3.2" and str(client.base_url) == "http://localhost:11434/v1/"
    assert client_for("openai/gpt-oss-120b:free")[1] == "openai/gpt-oss-120b:free"


def row(model, candidate, score, error=None):
    return dict(model=model, candidate=candidate, repetition=0, score=score, error=error, latency=1.0, prompt_tokens=10, completion_tokens=5, cost=0.001)


def test_summarise():
    rows = [row("m", "mario-rossi", 5), row("m", "mario-rossi", 4), row("m", "jean-dupont", 3), row("m", "mohammed-ali", None, error="boom")]
    [summary] = summarise(rows)
    assert summary["mario-rossi"] == "4.5 ± 0.5" and summary["mohammed-ali"] == "n/a"
    assert summary["failures"] == 1 and summary["tokens"] == 45 and summary["error"] == round(1 / 3, 2)
    assert summary["ranking"] == "mario-rossi > jean-dupont"
    assert markdown([summary]).count("\n") == 2
