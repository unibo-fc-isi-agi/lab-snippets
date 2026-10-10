"""
The admission workflow as a state machine, written from scratch: the steps of admission.py, a transition function
(i.e. the graph, as code), and an engine which saves a checkpoint (a JSON file) after each step.
The committee's approval is an INTERRUPT: the run stops, and it can be resumed later, even by another process.

Run with: poetry run python -m snippets -l workflows -e 1 [--thread THREAD]
(answer the approval question, or press Ctrl+D to stop there, and resume later with --thread THREAD)
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL, VISION_MODEL (must support images).
"""
import json
from argparse import ArgumentParser
from datetime import datetime
import data
from snippets.lecture_workflows import admission
from snippets.lecture_workflows.admission import State

END = "END"


class Interrupt(Exception):  # raised by user-input steps, to stop the run until a human answers
    def __init__(self, question: str):
        super().__init__(question)
        self.question = question


def approve(state: State) -> State:  # user-input step: the answer is put into the state when resuming the run
    if "answer" not in state:
        raise Interrupt(admission.approval_request(state))
    return {"approved": admission.is_yes(state["answer"])}


STEPS = {
    "read_transcripts": admission.read_transcripts,    # LLM
    "check_eligibility": admission.check_eligibility,  # data
    "score_letters": admission.score_letters,          # LLM
    "rank": admission.rank,                            # data
    "approve": approve,                                # user input
    "notify": admission.notify,                        # action
}


def next_step(step: str | None, state: State) -> str:  # the transitions, i.e. the edges of the graph (None: the start)
    match step:
        case None: return "read_transcripts"
        case "read_transcripts": return "check_eligibility"
        case "check_eligibility": return "score_letters" if admission.eligible(state) else "approve"  # a conditional edge
        case "score_letters": return "rank"
        case "rank": return "approve"
        case "approve": return "notify" if state["approved"] else END  # a conditional edge
        case "notify": return END


def save(thread: str, step: str, state: State) -> None:  # a checkpoint: the state, plus the step to run next
    file = admission.OUTPUT_DIR / "checkpoints" / f"{thread}.json"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps({"next": step, "state": state}, indent=2))


def load(thread: str) -> tuple[str, State] | None:
    file = admission.OUTPUT_DIR / "checkpoints" / f"{thread}.json"
    return tuple(json.loads(file.read_text()).values()) if file.exists() else None


def run(thread: str, state: State | None = None, answer: str | None = None) -> str | None:
    """Runs (or resumes) a thread, until its end (returns None) or an interrupt (returns its question)."""
    step, state = load(thread) or (next_step(None, state), state)  # resume from the checkpoint, if any
    if answer is not None:
        state = {**state, "answer": answer}
    while step != END:
        print(f"    [step] {step}")
        try:
            update = STEPS[step](state)
        except Interrupt as interrupt:
            save(thread, step, state)  # the run stops HERE: the same step is re-run when resuming
            return interrupt.question
        state = {**state, **update}  # merge the partial update (i.e. overwrite the changed fields)
        step = next_step(step, state)
        save(thread, step, state)
    return None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--thread", default=datetime.now().strftime("%Y%m%d-%H%M%S"), help="ID of the run (to resume it)")
    thread = parser.parse_args().thread
    print(f"# Thread: {thread}")

    question = run(thread, State(run=thread, candidates=data.CANDIDATES))
    while question:  # i.e. while interrupted
        try:
            answer = input(f"{question} [y/N] ")
        except (EOFError, KeyboardInterrupt):
            print(f"\n# Stopped: resume with --thread {thread}")
            break
        question = run(thread, answer=answer)
    else:
        print(f"# Done: final state in {admission.OUTPUT_DIR / 'checkpoints' / f'{thread}.json'}, e-mails in {admission.OUTPUT_DIR / 'outbox'}")
