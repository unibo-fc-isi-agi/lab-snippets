"""
Plan-and-execute for the committee's ad-hoc requests: the LLM proposes a PLAN (a structured output), code VALIDATES it,
the committee APPROVES plans with write-enabled steps, then code EXECUTES it; results go back to the planner,
which either plans the next steps (re-planning) or declares the request fulfilled. The tools are those of committee.py.

Run with: poetry run python -m snippets -l workflows -e 3
e.g., ask: "Invite the candidate with the best letter to an interview next Monday at 10:00"
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL, VISION_MODEL (must support images),
WORKFLOWS_OUTPUT (where interviews and e-mails are written, default: output/workflows).
"""
import json
import operator
import uuid
from typing import Annotated, Literal, TypedDict
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from snippets.lecture_agents.example1bis.agent_langchain import llm
from snippets.lecture_agents.simple_tools import get_current_time
from snippets.lecture_workflows import committee

TOOLS = {function.__name__: function for function in [*committee.read_only_tools, *committee.write_tools, get_current_time]}
WRITE_TOOLS = {function.__name__ for function in committee.write_tools}  # irreversible, or norm-sensitive: the committee must approve
SCHEMAS = {name: StructuredTool.from_function(function).args_schema for name, function in TOOLS.items()}  # to validate arguments
MAX_PLANS = 4  # bounds re-planning, and attempts to fix invalid plans


class Step(BaseModel):
    tool: Literal[tuple(TOOLS)] = Field(description="Name of the tool to call.")
    args: str = Field(description='Arguments of the call, as a JSON object, e.g. {"candidate": "mario-rossi"}.')
    why: str = Field(description="Why this step is needed, in a few words.")


class Plan(BaseModel):
    steps: list[Step] = Field(description="The steps to execute now, in order. Steps cannot use the results of other steps of the same plan.")
    final: bool = Field(description="True only if, after these steps, the WHOLE request is fulfilled, including the actions it asks for; "
                                    "false if more steps will be needed (you will be asked to plan them, seeing the results).")


class State(TypedDict, total=False):
    request: str
    plan: dict                                    # the current plan (a Plan, as a dict)
    errors: list[str]                             # what the validator found wrong in it
    plans: int                                    # how many plans were made so far
    results: Annotated[list[dict], operator.add]  # every step executed so far, with its result
    rejected: bool                                # whether the committee rejected the plan
    report: str


def print_plan(plan: dict) -> None:
    for i, step in enumerate(plan["steps"], start=1):
        print(f"    {i}. {step['tool']}({step['args']})  # {step['why']}" + ("  <- WRITE" if step["tool"] in WRITE_TOOLS else ""))


def plan(state: State) -> State:  # LLM step: a plan, as a structured output
    prompt = f"""You plan how to fulfil the requests of a PhD admission committee, using ONLY these tools:
{json.dumps([convert_to_openai_tool(function)["function"] for function in TOOLS.values()])}
Request: {state['request']} (plan the actions it asks for, too: the committee approves them before execution)
Never use placeholders (e.g. "[best_candidate]"): plan only the steps whose arguments you know NOW; for the others, set final to false,
and you will plan them after seeing the results.
Steps executed so far, with their results (truncated): {json.dumps(state.get('results', []), default=str)}"""
    if state.get("errors"):  # the previous plan was invalid: tell the planner why
        prompt += f"\nYour previous plan was INVALID: {'; '.join(state['errors'])}. Fix it."
    result = llm.with_structured_output(Plan).invoke(prompt)
    print(f"    [plan {state.get('plans', 0) + 1}]" + (" (final)" if result.final else ""))
    print_plan(result.model_dump())
    return {"plan": result.model_dump(), "plans": state.get("plans", 0) + 1, "errors": []}


def validate(state: State) -> State:  # data step: the plan is checked by code, BEFORE anything is executed
    errors = []
    read_about = {result["args"].get("candidate") for result in state.get("results", []) if result["tool"] not in WRITE_TOOLS}
    for i, step in enumerate(state["plan"]["steps"], start=1):
        try:
            args = json.loads(step["args"])
            SCHEMAS[step["tool"]].model_validate(args)  # names and types of the arguments, as in the tool's signature
        except Exception as e:
            errors.append(f"step {i} ({step['tool']}): invalid arguments {step['args']}: {str(e).splitlines()[0]}")
            continue
        candidate = args.get("candidate")
        if candidate is not None and candidate not in committee.list_candidates():
            errors.append(f"step {i}: unknown candidate {candidate!r}, valid ones are {committee.list_candidates()}")
        elif step["tool"] in WRITE_TOOLS and candidate not in read_about:  # a precondition: no action on unread applications
            errors.append(f"step {i}: {step['tool']} about {candidate} without having read any of their documents first")
        else:
            read_about.add(candidate)
    print(f"    [validate] {'; '.join(errors) or 'ok'}")
    return {"errors": errors}


def after_validate(state: State) -> str:
    if state["errors"]:
        return "plan" if state["plans"] < MAX_PLANS else "report"  # try to fix the plan, a bounded number of times
    return "approve" if any(step["tool"] in WRITE_TOOLS for step in state["plan"]["steps"]) else "execute"


def approve(state: State) -> State:  # user-input step: only plans with write-enabled steps need it
    answer = interrupt(state["plan"])  # the committee sees the WHOLE plan, before any of its steps is executed
    return {"rejected": answer.strip().lower() not in ("y", "yes")}


def execute(state: State) -> State:  # code executes the plan: errors become results, so the planner can see them
    results = []
    for step in state["plan"]["steps"]:
        args = json.loads(step["args"])
        try:
            result = TOOLS[step["tool"]](**args)
        except Exception as e:
            result = f"ERROR: {e}"
        print(f"    [step] {step['tool']}({args}) -> {str(result)[:100]}")
        results.append({"tool": step["tool"], "args": args, "result": result if isinstance(result, dict) else str(result)[:300]})  # short: they go into prompts
    return {"results": results}


def after_execute(state: State) -> str:
    return "report" if state["plan"]["final"] or state["plans"] >= MAX_PLANS else "plan"  # re-planning, bounded


def report(state: State) -> State:  # LLM step: tells the committee what was done, based on the results only
    if state.get("rejected"):
        return {"report": "The plan was rejected by the committee: nothing was done."}
    return {"report": llm.invoke(
        f"Request of the committee: {state['request']}\nSteps executed, with their results: {json.dumps(state.get('results', []), default=str)}\n"
        f"Problems: {'; '.join(state.get('errors', [])) or 'none'}\n"
        "Report to the committee, in a few lines, what was done and what was found, based ONLY on the results above.").content}


graph = StateGraph(State)
for node in [plan, validate, approve, execute, report]:
    graph.add_node(node.__name__, node)
graph.add_edge(START, "plan")
graph.add_edge("plan", "validate")
graph.add_conditional_edges("validate", after_validate, ["plan", "approve", "execute", "report"])
graph.add_conditional_edges("approve", lambda state: "report" if state["rejected"] else "execute", ["report", "execute"])
graph.add_conditional_edges("execute", after_execute, ["plan", "report"])
graph.add_edge("report", END)
workflow = graph.compile(checkpointer=InMemorySaver())  # interrupts need a checkpointer


if __name__ == "__main__":
    while True:
        try:
            request = input("Committee: ")
        except (EOFError, KeyboardInterrupt):
            break
        config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 50}  # one thread per request
        result = workflow.invoke({"request": request}, config)
        while "__interrupt__" in result:
            try:
                answer = input("The plan above includes write-enabled steps: do you approve it? [y/N] ")
            except (EOFError, KeyboardInterrupt):
                answer = "no"
            result = workflow.invoke(Command(resume=answer), config)
        print(f"AI ({result['plans']} plan(s)): {result['report']}")
