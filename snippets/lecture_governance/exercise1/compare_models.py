"""
Compares models on the letter-scoring system (Example 1 of the prompting lecture): each model scores each letter several times,
and the runs are summarised per model: scores (mean, standard deviation), error w.r.t. the committee's own scores,
ranking of candidates, failures, latency, tokens, and cost.

Run with: poetry run python -m snippets -l governance -x 1 [MODEL ...]
where each MODEL is an Open Router model, or MODEL@BASE_URL for other providers, e.g. llama3.2@http://localhost:11434/v1 (Ollama)
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, REPETITIONS (default: 5), TEMPERATURE (default: the provider's),
COMMITTEE_OUTPUT (where the CSV file with one row per run is written, default: output/).
"""
import csv
import os
import statistics
import sys
import time
from pathlib import Path
from openai import OpenAI
import data
from snippets.lecture_prompting.example1.letter_scoring_openai import LetterInfo, general_instructions, api_key, base_url
from snippets.lecture_llmaas.exercise2.repl_chat_retry import with_retries

# step 1: the shortlist (free models are rate-limited: retries below)
DEFAULT_MODELS = ["google/gemma-4-26b-a4b-it:free", "openai/gpt-oss-120b:free", "llama3.2@http://localhost:11434/v1"]
REPETITIONS = int(os.environ.get("REPETITIONS", 5))
TEMPERATURE = os.environ.get("TEMPERATURE")
# the reference: the committee's OWN scores, by reading the letters (without it, one can tell whether models agree, not who is right)
REFERENCE = {"mario-rossi": 5, "jean-dupont": 3, "mohammed-ali": 2}
OUTPUT = Path(os.environ.get("COMMITTEE_OUTPUT", "output")) / "model-comparison.csv"


def client_for(model: str) -> tuple[OpenAI, str]:
    """e.g. 'llama3.2@http://localhost:11434/v1' -> a client for Ollama, and 'llama3.2'."""
    name, _, url = model.partition("@")
    return OpenAI(base_url=url or base_url, api_key=api_key if not url else "local", max_retries=0), name


def score_letter(client: OpenAI, model: str, letter_text: str):
    """As in Example 1, but returns the WHOLE response (i.e. also its usage), not just the parsed LetterInfo."""
    options = dict(temperature=float(TEMPERATURE)) if TEMPERATURE else {}
    return with_retries(lambda: client.chat.completions.parse(  # retries upon rate limits (cf. Exercise 2 of the LLM-as-a-Service lecture)
        model=model,
        messages=[
            dict(role="system", content=general_instructions),
            dict(role="user", content=f"Score the following letter:\n\n<letter>\n{letter_text}\n</letter>"),
        ],
        response_format=LetterInfo,
        **options,
    ), retries=5, initial_delay=5)


def run_once(model: str, candidate: str, repetition: int) -> dict:
    """Step 2-3: one run, as a row of the CSV file. Failures are rows too: they must be counted, not hidden."""
    client, name = client_for(model)
    row = dict(model=model, candidate=candidate, repetition=repetition, score=None, error=None, latency=None, prompt_tokens=None, completion_tokens=None, cost=None)
    start = time.perf_counter()
    try:
        response = score_letter(client, name, data.letter(candidate).read_text())
        row["latency"] = round(time.perf_counter() - start, 2)  # wall-clock time, retries included
        row["score"] = response.choices[0].message.parsed.score  # may fail, e.g. upon refusals or invalid JSON
        row["prompt_tokens"], row["completion_tokens"] = response.usage.prompt_tokens, response.usage.completion_tokens
        row["cost"] = getattr(response.usage, "cost", None)  # in USD, reported by Open Router (not part of OpenAI's API)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:200]
    return row


def summarise(rows: list[dict]) -> list[dict]:
    """Step 4: one line per model."""
    summary = []
    for model in dict.fromkeys(row["model"] for row in rows):  # models, in order of appearance
        ok = [row for row in rows if row["model"] == model and row["error"] is None]
        scores = {c: [row["score"] for row in ok if row["candidate"] == c] for c in REFERENCE}
        means = {c: statistics.mean(s) for c, s in scores.items() if s}
        summary.append(dict(
            model=model,
            **{c: f"{means[c]:.1f} ± {statistics.pstdev(s):.1f}" if s else "n/a" for c, s in scores.items()},  # consistency
            error=round(statistics.mean(abs(row["score"] - REFERENCE[row["candidate"]]) for row in ok), 2) if ok else None,  # w.r.t. the reference
            ranking=" > ".join(sorted(means, key=means.get, reverse=True)),  # agreement among models: compare these
            failures=sum(row["model"] == model for row in rows) - len(ok),
            latency=round(statistics.mean(row["latency"] for row in ok), 1) if ok else None,
            tokens=sum((row["prompt_tokens"] or 0) + (row["completion_tokens"] or 0) for row in ok),
            cost=round(sum(row["cost"] or 0 for row in ok), 4),
        ))
    return summary


def markdown(table: list[dict]) -> str:
    lines = ["| " + " | ".join(table[0]) + " |", "|" + "---|" * len(table[0])]
    return "\n".join(lines + ["| " + " | ".join(str(value) for value in row.values()) + " |" for row in table])


if __name__ == "__main__":
    models = sys.argv[1:] or DEFAULT_MODELS
    rows = []
    for model in models:
        for candidate in REFERENCE:
            for repetition in range(REPETITIONS):
                rows.append(row := run_once(model, candidate, repetition))
                outcome = row["error"] or f"score {row['score']} in {row['latency']}s"
                print(f"# {model} | {candidate} #{repetition + 1}: {outcome}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as f:  # raw data: one row per run, for further analysis
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nRuns written to {OUTPUT}, reference scores: {REFERENCE}\n")
    print(markdown(summarise(rows)))
