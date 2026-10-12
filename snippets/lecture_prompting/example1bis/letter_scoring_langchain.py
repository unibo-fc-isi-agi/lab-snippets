"""
The same letter-scoring system of Example 1, with LangChain: a prompt template, piped into an LLM with structured output.

Run with: poetry run python -m snippets -l prompting -e 1bis mario-rossi
(the letter can be given as a candidate's ID, or as a path, e.g. data/letter-mario-rossi.txt)
Configure via env vars: OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL.
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# same configuration, instructions, and output schema (Pydantic classes) as in Example 1
from snippets.lecture_prompting.example1.letter_scoring_openai import base_url, api_key, model, general_instructions, \
    ApplicantInfo, AuthorInfo, LetterInfo

llm = ChatOpenAI(base_url=base_url, api_key=api_key, model=model)


# a prompt template: messages with placeholders (e.g. {letter_text}), filled in at invocation time
prompt = ChatPromptTemplate.from_messages([
    ("system", general_instructions),
    ("user", "Score the following letter:\n\n<letter>\n{letter_text}\n</letter>"),
])


# a chain: the filled-in prompt goes to the LLM, whose answer is parsed into a LetterInfo
scoring_chain = prompt | llm.with_structured_output(LetterInfo)

def score_letter(letter_text: str) -> LetterInfo:
    return scoring_chain.invoke({"letter_text": letter_text})  # fills the {letter_text} placeholder


if __name__ == "__main__":
    import sys
    import data

    print(f"Using model: {model}")

    letter_path = sys.argv[1] if len(sys.argv) > 1 else input("Enter the letter to score (candidate's ID or path): ")
    letter_path = data.find_letter(letter_path)
    if not letter_path.is_file():
        print(f"File {letter_path} is not a file.")
        sys.exit(1)
    letter_text = letter_path.read_text()
    letter_info = score_letter(letter_text)
    print(letter_info.model_dump_json(indent=2))
