"""
Agentic RAG: retrieval is a tool, and the agent decides whether, when, and what to search (possibly several times),
before answering. The tool searches the sqlite-vec index of Example 2 bis, optionally restricted to one candidate.

Run with: poetry run python -m snippets -l rag -e 4bis
e.g., ask: "Could the author of Mario Rossi's recommendation letter sit on the admission committee?"
(two searches are needed: who wrote the letter, and what the regulations say about conflicts of interest)
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL (must support tool calling),
plus EMBEDDINGS_BASE_URL, EMBEDDINGS_API_KEY, EMBEDDINGS_MODEL.
"""
import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import data
from snippets.lecture_rag.example2bis.vector_store_sqlite_vec import DB_FILE, create_index, search
from snippets.lecture_rag.vec import connect

base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1/")
api_key = os.environ.get("OPENAI_API_KEY") or input(f"Enter your API key for {base_url}: ")
model = os.environ.get("OPENAI_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

llm = ChatOpenAI(base_url=base_url, api_key=api_key, model=model)

if not DB_FILE.exists():  # same index as Example 2 bis: build it, if missing
    DB_FILE.parent.mkdir(exist_ok=True)
    with connect(DB_FILE) as conn:
        create_index(conn)


def search_documents(query: str, candidate: str | None = None) -> str:
    """
    Searches the PhD programme's regulations and the candidates' recommendation letters,
    returning the 3 passages most relevant to the query, each with its ID.
    If `candidate` is given (e.g. 'mario-rossi'), ONLY that candidate's letter is searched, NOT the regulations:
    to combine information about a candidate with the regulations, search them separately.
    """
    with connect(DB_FILE) as conn:  # one connection per call: tools may run in other threads
        results = search(conn, query, k=3, candidate=candidate)
    return "\n".join(f'<document id="{id}">\n{text}\n</document>' for _, id, text in results)


instructions = f"""
You are assisting the admission committee of a PhD programme.
Answer questions about the programme's regulations and the candidates' applications,
based ONLY on what the search_documents tool returns: search as many times as needed, with focused queries.
Cite the IDs of the documents you used. If you cannot find the answer, say so.
Candidates are identified as: {', '.join(data.CANDIDATES)}.
"""

agent = create_agent(llm, tools=[search_documents], system_prompt=instructions)


def print_tool_calls(messages) -> None:  # shows which searches the agent made
    for message in messages:
        for tool_call in getattr(message, "tool_calls", []):
            print(f"    [tool] {tool_call['name']}({tool_call['args']})")


if __name__ == "__main__":
    print(f"Using model: {model}")
    messages = []
    while True:
        try:
            messages.append(("user", input("You: ")))
        except (EOFError, KeyboardInterrupt):
            break
        result = agent.invoke({"messages": messages}, {"recursion_limit": 20})
        print_tool_calls(result["messages"][len(messages):])  # only the new messages
        messages = result["messages"]
        print(f"AI: {messages[-1].content}")
