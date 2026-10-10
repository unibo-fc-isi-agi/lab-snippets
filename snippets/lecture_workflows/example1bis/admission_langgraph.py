"""
The same admission workflow of Example 1, with LangGraph: same steps (admission.py), but the graph is declared
(nodes and edges), transcripts are read IN PARALLEL (one branch per candidate), checkpoints are saved into SQLite,
and the committee's approval is an interrupt().

Run with: poetry run python -m snippets -l workflows -e 1bis [--thread THREAD] [--mermaid]
(answer the approval question, or press Ctrl+D to stop there, and resume later with --thread THREAD)
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL, VISION_MODEL (must support images).
"""
import operator
import sqlite3
from argparse import ArgumentParser
from datetime import datetime
from typing import Annotated
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
import data
from snippets.lecture_workflows import admission


class State(admission.State):
    # a REDUCER: parallel branches return {"transcripts": {candidate: ...}}, and the engine merges them via dict union (|)
    transcripts: Annotated[dict[str, dict], operator.or_]


def read_transcripts(state: State) -> list[Send]:  # fan-out: one branch per candidate, run in parallel
    return [Send("read_transcript", {"candidate": candidate}) for candidate in state["candidates"]]


def read_transcript(branch: dict) -> State:  # the state of a branch is the argument of its Send
    transcript = admission.extract_transcript(branch["candidate"])
    get_stream_writer()({"progress": f"transcript of {branch['candidate']} read"})  # a custom event (cf. Example 5)
    return {"transcripts": {branch["candidate"]: transcript}}


def approve(state: State) -> State:  # user-input step: interrupt() stops the run here, and returns the answer upon resuming
    answer = interrupt(admission.approval_request(state))
    return {"approved": admission.is_yes(answer)}


def build(checkpointer=None):
    graph = StateGraph(State)
    graph.add_node("read_transcript", read_transcript)                 # LLM
    graph.add_node("check_eligibility", admission.check_eligibility)   # data
    graph.add_node("score_letters", admission.score_letters)           # LLM
    graph.add_node("rank", admission.rank)                             # data
    graph.add_node("approve", approve)                                 # user input
    graph.add_node("notify", admission.notify)                         # action
    graph.add_conditional_edges(START, read_transcripts, ["read_transcript"])
    graph.add_edge("read_transcript", "check_eligibility")  # waits for ALL the parallel branches
    graph.add_conditional_edges("check_eligibility", lambda state: "score_letters" if admission.eligible(state) else "approve",
                                ["score_letters", "approve"])  # the possible targets (to draw the graph)
    graph.add_edge("score_letters", "rank")
    graph.add_edge("rank", "approve")
    graph.add_conditional_edges("approve", lambda state: "notify" if state["approved"] else END, ["notify", END])
    graph.add_edge("notify", END)
    return graph.compile(checkpointer=checkpointer)


def checkpointer() -> SqliteSaver:  # a persistent checkpointer: runs survive the process
    admission.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(admission.OUTPUT_DIR / "checkpoints.db", check_same_thread=False))


def question(result: dict) -> str | None:  # the question of the pending interrupt, if any
    return result["__interrupt__"][0].value if "__interrupt__" in result else None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--thread", default=datetime.now().strftime("%Y%m%d-%H%M%S"), help="ID of the run (to resume it)")
    parser.add_argument("--mermaid", action="store_true", help="just print the graph, as a Mermaid diagram")
    args = parser.parse_args()

    workflow = build(checkpointer())
    if args.mermaid:
        print(workflow.get_graph().draw_mermaid())
        raise SystemExit
    config = {"configurable": {"thread_id": args.thread}}  # the thread selects the checkpoints
    print(f"# Thread: {args.thread}")

    pending = workflow.get_state(config).interrupts  # a thread stopped at an interrupt is resumed, a new one is started
    result = {"__interrupt__": pending} if pending else workflow.invoke({"run": args.thread, "candidates": data.CANDIDATES}, config)
    while question(result):
        try:
            answer = input(f"{question(result)} [y/N] ")
        except (EOFError, KeyboardInterrupt):
            print(f"\n# Stopped: resume with --thread {args.thread}")
            break
        result = workflow.invoke(Command(resume=answer), config)  # the answer becomes the return value of interrupt()
    else:
        print(f"# Done: notified {result.get('notified', [])}, e-mails in {admission.OUTPUT_DIR / 'outbox'}")
