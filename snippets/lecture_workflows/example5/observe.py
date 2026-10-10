"""
The admission workflow of Example 1 bis, observed: while it runs, its steps are STREAMED (one line per completed step,
plus the custom progress events emitted by the steps); afterwards, its TRACES can be inspected with MLflow.

Run with: poetry run python -m snippets -l workflows -e 5 [--thread THREAD]
then browse the traces with: poetry run mlflow ui   (experiment "admission-workflow", tab "Traces")
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL, VISION_MODEL (must support images).
"""
from argparse import ArgumentParser
from datetime import datetime
import mlflow
from langgraph.types import Command
import data
from snippets.lecture_workflows.example1bis.admission_langgraph import build, checkpointer


def run(workflow, input, config) -> str | None:
    """Runs the workflow, printing its events as they come; returns the question of the pending interrupt, if any."""
    question = None
    for mode, event in workflow.stream(input, config, stream_mode=["updates", "custom"]):  # several modes: (mode, event) pairs
        if mode == "custom":  # emitted by the steps, via get_stream_writer()
            print(f"    [progress] {event['progress']}")
        elif "__interrupt__" in event:  # do NOT return here: let the stream end (or the run would look aborted)
            question = event["__interrupt__"][0].value
        else:  # updates: {node: the partial update it returned}
            for node, update in event.items():
                print(f"    [{datetime.now():%H:%M:%S}] {node} done, updated: {', '.join(update or {}) or 'nothing'}")
    return question


if __name__ == "__main__":
    mlflow.set_experiment("admission-workflow")
    mlflow.langchain.autolog()  # each run of the graph becomes a trace: nodes, LLM calls, and tool calls are its spans
    parser = ArgumentParser()
    parser.add_argument("--thread", default=datetime.now().strftime("%Y%m%d-%H%M%S"), help="ID of the run (to resume it)")
    thread = parser.parse_args().thread
    workflow, config = build(checkpointer()), {"configurable": {"thread_id": thread}}
    print(f"# Thread: {thread}")

    pending = workflow.get_state(config).interrupts
    question = pending[0].value if pending else run(workflow, {"run": thread, "candidates": data.CANDIDATES}, config)
    while question:
        try:
            answer = input(f"{question} [y/N] ")
        except (EOFError, KeyboardInterrupt):
            print(f"\n# Stopped: resume with --thread {thread}")
            break
        question = run(workflow, Command(resume=answer), config)
    print("# Traces: run `poetry run mlflow ui`, then open http://localhost:5000")
