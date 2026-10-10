"""
A supervisor agent, which delegates to two specialised sub-agents (subagents.py), seen as TOOLS:
the supervisor never sees the documents, nor the regulations, only the sub-agents' answers (context isolation).

Run with: poetry run python -m snippets -l workflows -e 4
e.g., ask: "Is Mohammed Ali's degree enough to be admitted, according to the regulations?"
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (must support tool calling), VISION_MODEL,
plus EMBEDDINGS_BASE_URL, EMBEDDINGS_API_KEY, EMBEDDINGS_MODEL.
"""
from langchain.agents import create_agent
from snippets.lecture_agents.example1bis.agent_langchain import llm
from snippets.lecture_workflows.subagents import ask, documents_agent, regulations_agent


def ask_documents_agent(question: str) -> str:
    """Ask a colleague who can read the candidates' applications (letters, transcripts, letter scores) a self-contained question."""
    answer = ask(documents_agent, question)
    print(f"    [documents agent] {question!r} -> {answer[:100]!r}...")
    return answer


def ask_regulations_agent(question: str) -> str:
    """Ask a colleague who knows the regulations of the PhD programme a self-contained question."""
    answer = ask(regulations_agent, question)
    print(f"    [regulations agent] {question!r} -> {answer[:100]!r}...")
    return answer


instructions = """
You assist the admission committee of a PhD programme. You cannot read documents yourself: delegate to your colleagues,
via your tools, with SELF-CONTAINED questions (they do not see this conversation), possibly to both of them.
Then combine their answers, saying which colleague said what. If they cannot answer, say so.
"""

supervisor = create_agent(llm, tools=[ask_documents_agent, ask_regulations_agent], system_prompt=instructions)


if __name__ == "__main__":
    messages = []
    while True:
        try:
            messages.append(("user", input("You: ")))
        except (EOFError, KeyboardInterrupt):
            break
        messages = supervisor.invoke({"messages": messages}, {"recursion_limit": 20})["messages"]
        print(f"AI: {messages[-1].content}")
