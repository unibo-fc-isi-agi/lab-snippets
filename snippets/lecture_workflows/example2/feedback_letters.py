"""
Personalised feedback letters to the candidates, after the committee's decisions, combining three patterns:
- parallelisation (sectioning): one worker per candidate, run in parallel (Send), results gathered by a reducer;
- routing: the decision (admit / interview / reject) selects the drafter, i.e. the instructions for the letter;
- evaluator-optimizer: a judge checks each draft against a checklist; rejected drafts go back to the drafter, with
  the judge's issues, at most MAX_ROUNDS times.

Run with: poetry run python -m snippets -l workflows -e 2
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL.
"""
import operator
from typing import Annotated, Literal, TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from snippets.lecture_agents.example1bis.agent_langchain import llm

MAX_ROUNDS = 3

# the committee's decisions (e.g. the outcome of Example 1), with their motivations: the ONLY facts letters may mention
DECISIONS = {
    "mario-rossi": ("interview", "excellent academic record (105/110), and a detailed, very positive recommendation letter"),
    "jean-dupont": ("interview", "good academic record (14.7/20); the letter mentions the need for close supervision"),
    "mohammed-ali": ("reject", "not eligible: a Master's degree is required (Art. 3 of the regulations), the transcript is of a Bachelor's degree"),
}

INSTRUCTIONS = {  # routing: one set of instructions per decision
    "admit": "Congratulate the candidate, and explain how to confirm the enrolment within 10 days (Art. 6).",
    "interview": "Invite the candidate to the interview (6-10 July 2026, by videoconference), and tell them what will be discussed: their research proposal.",
    "reject": "Inform the candidate, kindly but clearly, of the outcome, explaining the reason, and how to appeal within 30 days (Art. 15).",
}


class Letter(TypedDict, total=False):  # the state of each worker
    candidate: str
    decision: Literal["admit", "interview", "reject"]
    motivation: str
    drafts: Annotated[list[str], operator.add]  # every draft, and every critique, is kept: the process is auditable
    critiques: Annotated[list[list[str]], operator.add]
    accepted: bool


class Verdict(BaseModel):  # a CHECKLIST, one boolean per criterion: small LLMs fill lists of issues with "none", "N/A", ...
    only_given_facts: bool = Field(description="The e-mail mentions no fact other than the decision and its motivation (no invented dates, grades, names, or promises).")
    no_other_candidates: bool = Field(description="The e-mail mentions no other candidate.")
    polite_and_short: bool = Field(description="The e-mail is polite and professional, and at most 150 words long.")
    follows_instructions: bool = Field(description="The e-mail does what the instructions given to the writer say.")


def drafter(decision: str):  # a factory: one drafter node per decision (the target of the routing)
    def draft(letter: Letter) -> Letter:
        prompt = (f"Write a short e-mail from the PhD admission committee to the candidate {letter['candidate']}.\n"
                  f"Decision: {decision}. Motivation: {letter['motivation']}.\n{INSTRUCTIONS[decision]}\n"
                  f"Use ONLY the facts above. Write the text of the e-mail only.")
        if letter.get("critiques"):  # not the first round: fix the issues found by the judge
            prompt += f"\nThis is your previous draft:\n{letter['drafts'][-1]}\nFix these issues: {'; '.join(letter['critiques'][-1])}"
        return {"drafts": [llm.invoke(prompt).content]}
    return draft


def judge(letter: Letter) -> Letter:  # the evaluator: an LLM-as-a-judge, filling in the checklist
    verdict = llm.with_structured_output(Verdict).invoke(
        f"Check this e-mail of a PhD admission committee, criterion by criterion.\n"
        f"Facts: decision {letter['decision']}, motivation: {letter['motivation']}.\n"
        f"Instructions given to the writer: {INSTRUCTIONS[letter['decision']]}\n\nE-mail:\n{letter['drafts'][-1]}")
    issues = [Verdict.model_fields[name].description for name, ok in verdict.model_dump().items() if not ok]  # by code
    return {"critiques": [issues], "accepted": not issues}


def after_judge(letter: Letter) -> str:  # the loop: back to the SAME drafter, until accepted, or MAX_ROUNDS drafts
    return END if letter["accepted"] or len(letter["drafts"]) >= MAX_ROUNDS else letter["decision"]


worker = StateGraph(Letter)  # the worker: route -> draft <-> judge
for decision in INSTRUCTIONS:
    worker.add_node(decision, drafter(decision))
    worker.add_edge(decision, "judge")
worker.add_node("judge", judge)
worker.add_conditional_edges(START, lambda letter: letter["decision"], list(INSTRUCTIONS))  # routing
worker.add_conditional_edges("judge", after_judge, [*INSTRUCTIONS, END])
worker = worker.compile()


class State(TypedDict, total=False):  # the state of the main graph
    decisions: dict[str, tuple[str, str]]
    letters: Annotated[dict[str, Letter], operator.or_]  # merged from the parallel workers


def write_letter(letter: Letter) -> State:  # runs a worker (a graph, inside a node of another graph)
    return {"letters": {letter["candidate"]: worker.invoke(letter)}}


letters = StateGraph(State)
letters.add_node("write_letter", write_letter)
letters.add_conditional_edges(START, lambda state: [  # fan-out: one worker per candidate
    Send("write_letter", {"candidate": candidate, "decision": decision, "motivation": motivation})
    for candidate, (decision, motivation) in state["decisions"].items()], ["write_letter"])
letters.add_edge("write_letter", END)
letters = letters.compile()


if __name__ == "__main__":
    result = letters.invoke({"decisions": DECISIONS})
    for candidate, letter in result["letters"].items():
        print(f"===== {candidate} ({letter['decision']}): {len(letter['drafts'])} round(s), accepted: {letter['accepted']}")
        for round, issues in enumerate(letter["critiques"], start=1):
            print(f"    [judge, round {round}] {'; '.join(issues) or 'no issues'}")
        print(letter["drafts"][-1])
